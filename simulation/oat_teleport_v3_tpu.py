"""
oat_teleport_v3_tpu.py — Tensor Network OAT teleportation.

Key insight: U_OAT = exp(-i chi_t J_z^A J_z^B) is diagonal in the
computational basis, so the MPS for |psi_OAT> can be built DIRECTLY
as an automaton — no MPO-MPS sweep, no Trotter error.

Bond dimension = n+1 (n = N/2). At N=100: bond dim = 51 vs 2^100 exact.

Layout: sites 0..n-1 = A chain, sites n..2n-1 = B chain.
Automaton tracks 2*M_A = {-n,-n+2,...,n} (n+1 values, index j=0..n).

Cross-validates against v2 for N<=16, extends to N=20..64+.

Primary metric: Gain = F_Schmidt - F_naive = (1-C)/6 (analytic prediction).
"""
import argparse, json, math, cmath, sys, os
import numpy as np
from pathlib import Path
import jax, jax.numpy as jnp

sys.path.insert(0, os.path.dirname(__file__))
from jila_oat_exact_tpu import (jz_table, plus_state, oat_evolve,
                                 boundary_rdm, concurrence)

print(f"[DEVICE] {jax.devices()}  backend={jax.default_backend()}", flush=True)
CL = 2.0 / 3.0

# ── Gates ──────────────────────────────────────────────────────────────────────
I2 = np.eye(2, dtype=complex)
X  = np.array([[0,1],[1,0]], dtype=complex)
Z  = np.array([[1,0],[0,-1]], dtype=complex)
H2 = np.array([[1,1],[1,-1]], dtype=complex) / math.sqrt(2)
P0 = np.array([[1,0],[0,0]], dtype=complex)
P1 = np.array([[0,0],[0,1]], dtype=complex)
def kron3(A,B,C): return np.kron(np.kron(A,B),C)
CNOT_01  = kron3(P0,I2,I2) + kron3(P1,X,I2)
CIRCUIT  = kron3(H2,I2,I2) @ CNOT_01
CIRCUIT_DAG = CIRCUIT.conj().T
PROJ  = {(m1,m2): kron3(P0 if m1==0 else P1, P0 if m2==0 else P1, I2)
         for m1 in range(2) for m2 in range(2)}
UCORR = {(0,0):I2, (0,1):X, (1,0):Z, (1,1):X@Z}

# ── MPS construction ───────────────────────────────────────────────────────────
def oat_mps(N, chi_t):
    """
    Direct MPS for |psi_OAT(chi_t)>.
    Automaton: j = # of |0> spins seen in A chain. M_A = j - n/2.
    A-site: s=0 -> j+1, s=1 -> j. B-site: diagonal phase exp(-i chi_t M_A m_B).

    Bond dims:
      A tensors: shape (min(j,1)+prev, 2, next)  — grows then truncates
      Simplified: use chi=n+1 internally, but correct boundary sizes.
      Left boundary:  chi_L = 1 (only j=0 possible before any spin)
      Right boundary: chi_R = 1 (only one end state after all B spins)
    """
    n   = N // 2
    chi = n + 1      # max bond dim
    h   = 1.0 / math.sqrt(2)
    tensors = []

    # A chain — site k has chi_L = min(k+1, chi), chi_R = min(k+2, chi)
    for k in range(n):
        cL = min(k + 1, chi)
        cR = min(k + 2, chi)
        A  = np.zeros((cL, 2, cR), dtype=complex)
        for j_in in range(cL):
            # s=0: j -> j+1
            if j_in + 1 < cR:
                A[j_in, 0, j_in + 1] = h
            # s=1: j -> j (stays)
            if j_in < cR:
                A[j_in, 1, j_in] = h
        tensors.append(A)

    # B chain — all chi=n+1 on left, shrinks on right boundary
    for k in range(n):
        cL = chi
        cR = chi if k < n - 1 else 1  # right boundary
        B  = np.zeros((cL, 2, cR), dtype=complex)
        for j in range(cL):
            M_A = j - n / 2.0
            ph0 = h * cmath.exp(-1j * chi_t * M_A *  0.5)
            ph1 = h * cmath.exp(-1j * chi_t * M_A * -0.5)
            if cR == 1:
                # right boundary: sum over all j (they all map to index 0)
                B[j, 0, 0] += ph0
                B[j, 1, 0] += ph1
            else:
                B[j, 0, j] = ph0
                B[j, 1, j] = ph1
        tensors.append(B)

    return tensors


def extract_boundary_rho(tensors, n):
    """
    Partial trace over all qubits except A_{n-1} (site n-1) and B_0 (site n).
    Returns rho_2 as a (4,4) complex numpy array.

    Algorithm:
      1. Contract left environment L[i,i'] from sites 0..n-2
      2. Contract right environment R[k,k'] from sites n+1..2n-1
      3. For each (a,b): amplitude[i,k] = sum_j (L @ T_A[:,a,:])[i,j] * T_B[j,b,k]
      4. rho_2[(a,b),(a',b')] = sum_{i,k,k'} amp[i,k] * R[k,k'] * amp'*[i,k']
    """
    N = len(tensors)

    # Left environment L[i,j] — starts as scalar 1
    L = np.ones((1, 1), dtype=complex)
    for k in range(n - 1):
        A = tensors[k]   # (chi_L, 2, chi_R)
        # L_new[l,m] = sum_{i,j,s} L[i,j] * A[i,s,l] * A*[j,s,m]
        L = np.einsum('ij,isl,jsm->lm', L, A, A.conj())

    # Right environment R[k,k']
    R = np.ones((1, 1), dtype=complex)
    for k in range(N - 1, n, -1):
        A = tensors[k]   # (chi_L, 2, chi_R)
        # R_new[i,j] = sum_{k,l,s} A[i,s,k] * R[k,l] * A*[j,s,l]
        R = np.einsum('isk,kl,jsl->ij', A, R, A.conj())

    T_A = tensors[n - 1]   # (chi_L_A, 2, chi_R_A): site A_{n-1}
    T_B = tensors[n    ]   # (chi_L_B, 2, chi_R_B): site B_0

    # vec[(a,b)][j, l] = sum_k T_A[j, a, k] * T_B[k, b, l]
    # shape: (chi_L_A, chi_R_B) for each (a,b)
    vec = {}
    for a in range(2):
        for b in range(2):
            vec[(a,b)] = T_A[:, a, :] @ T_B[:, b, :]  # (chi_L_A, chi_R_B)

    # Correct formula (keeps ket j and bra q separate via L[j,q]):
    # rho2[bra=(a',b'), ket=(a,b)] = sum_{j,q,l,m} L[j,q] * vec[a,b][j,l]
    #                                              * R[l,m] * vec[a',b'][q,m].conj()
    # = einsum('jq, jl, lm, qm ->', L, vec_ab, R, vec_apbp.conj())
    idx = {(0,0):0, (0,1):1, (1,0):2, (1,1):3}
    rho2 = np.zeros((4, 4), dtype=complex)
    for a in range(2):
        for b in range(2):
            vAB = vec[(a,b)]           # (chi_L_A, chi_R_B)
            for ap in range(2):
                for bp in range(2):
                    vABp = vec[(ap,bp)]
                    val = np.einsum('jq,jl,lm,qm->', L, vAB, R, vABp.conj())
                    rho2[idx[(ap,bp)], idx[(a,b)]] = val

    tr = np.trace(rho2).real
    return rho2 / tr if tr > 1e-14 else rho2


# ── Teleportation circuit (numpy) ──────────────────────────────────────────────
def teleport_fidelity(psi_A, rho2, schmidt_rotate=True):
    if schmidt_rotate:
        eigvals, eigvecs = np.linalg.eigh(rho2)
        psi = eigvecs[:, -1]
        M   = psi.reshape(2, 2)
        U, sigma, Vh = np.linalg.svd(M)
        U_loc = np.kron(U.conj().T, Vh.conj())
        rho2  = U_loc @ rho2 @ U_loc.conj().T

    rho_total = np.kron(np.outer(psi_A, psi_A.conj()), rho2)
    rho_bell  = CIRCUIT @ rho_total @ CIRCUIT_DAG
    rho_B_out = np.zeros((2,2), dtype=complex)
    for (m1,m2), Pi in PROJ.items():
        rho_proj = Pi @ rho_bell @ Pi
        prob = np.trace(rho_proj).real
        if prob < 1e-12: continue
        r = (rho_proj/prob).reshape(2,2,2,2,2,2)
        rho_q2 = np.einsum('ijkijl->kl', r)
        U = UCORR[(m1,m2)]
        rho_B_out += prob * (U @ rho_q2 @ U.conj().T)
    return float(np.clip((psi_A.conj() @ rho_B_out @ psi_A).real, 0, 1))


def haar_qubit(rng):
    u1,u2 = rng.uniform(0,1,2)
    th = math.acos(1-2*u1); ph = 2*math.pi*u2
    return np.array([math.cos(th/2), math.sin(th/2)*cmath.exp(1j*ph)], dtype=complex)

def avg_fidelity(rho2, K=5000, seed=42, schmidt_rotate=True):
    rng = np.random.default_rng(seed)
    Fv = [teleport_fidelity(haar_qubit(rng), rho2, schmidt_rotate) for _ in range(K)]
    return float(np.mean(Fv)), float(np.std(Fv))

import cmath  # needed by oat_mps

# ── Cross-validation: TN vs exact ─────────────────────────────────────────────
def cross_validate(N, chi_t):
    """Compare TN boundary RDM to exact (jila_oat_exact_tpu)."""
    import jax.numpy as jnp
    mA_np, mB_np = jz_table(N)
    mAj, mBj = jnp.array(mA_np), jnp.array(mB_np)
    psi0 = jnp.array(plus_state(N))
    psi_t = np.array(oat_evolve(psi0, mAj, mBj, float(chi_t)))
    rho2_exact = boundary_rdm(psi_t, N)
    C_exact = concurrence(rho2_exact)

    tensors = oat_mps(N, chi_t)
    n = N // 2
    rho2_tn = extract_boundary_rho(tensors, n)
    C_tn = concurrence(rho2_tn)

    diff = float(np.max(np.abs(rho2_exact - rho2_tn)))
    return C_exact, C_tn, diff, rho2_tn


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N",     nargs="+", type=int,
                    default=[2,4,6,8,10,12,14,16,20,24,32,40,48,64])
    ap.add_argument("--chi_t_steps", type=int, default=32)
    ap.add_argument("--K",     type=int,  default=2000)
    ap.add_argument("--crossval_N", nargs="+", type=int, default=[2,4,6,8,10,12,14,16])
    args = ap.parse_args()
    print(f"[CONFIG] N={args.N} chi_t_steps={args.chi_t_steps} K={args.K}", flush=True)

    chi_t_vals = np.linspace(0.05, math.pi * 0.95, args.chi_t_steps)

    # ── Phase 0: Cross-validation TN vs exact ─────────────────────────────────
    print("\n" + "="*65)
    print("  Phase 0: Cross-validation TN vs exact (N<=16)")
    print("="*65, flush=True)

    xv_pass = True
    for N in args.crossval_N:
        chi_t_test = math.pi / 3
        C_ex, C_tn, diff, _ = cross_validate(N, chi_t_test)
        ok = diff < 1e-6
        if not ok: xv_pass = False
        print(f"  [N={N:>2d}] C_exact={C_ex:.6f}  C_tn={C_tn:.6f}  "
              f"max|rho_diff|={diff:.2e}  {'PASS' if ok else 'FAIL'}", flush=True)

    print(f"\n  Cross-validation: {'ALL PASS' if xv_pass else 'SOME FAIL'}", flush=True)

    # ── Phase 1: Full sweep ───────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  Phase 1: TN teleportation sweep (N=2..64+)")
    print("="*65, flush=True)

    all_results = {}
    for N in args.N:
        n = N // 2
        chi_t_opt = None; F_peak = 0.0; rows = []

        for chi_t in chi_t_vals:
            tensors = oat_mps(N, float(chi_t))
            rho2    = extract_boundary_rho(tensors, n)
            C       = concurrence(rho2)
            Fp      = (2 + C) / 3
            if Fp > F_peak: F_peak = Fp; chi_t_opt = chi_t

        # At optimal chi_t — full fidelity sweep
        tensors = oat_mps(N, float(chi_t_opt))
        rho2    = extract_boundary_rho(tensors, n)
        Co      = concurrence(rho2)
        Fpo     = (2 + Co) / 3

        Fs, ss = avg_fidelity(rho2, K=args.K, schmidt_rotate=True)
        Fn, sn = avg_fidelity(rho2, K=args.K, schmidt_rotate=False)

        gain_meas = Fs - Fn
        gain_pred = (1 - Co) / 6
        gain_ok   = abs(gain_meas - gain_pred) < 0.005
        dF        = abs(Fs - Fpo)
        bond_dim  = n + 1

        print(f"  [N={N:>3d}] chi*={chi_t_opt:.3f}  C={Co:.4f}  "
              f"F_sch={Fs:.4f}(s={ss:.3f})  F_naive={Fn:.4f}  "
              f"gain={gain_meas:.4f}  pred={gain_pred:.4f}  "
              f"{'OK' if gain_ok else 'FAIL'}  bond_dim={bond_dim}  "
              f"|dF|={dF:.4f}", flush=True)

        all_results[f"N{N}"] = {
            "N": N, "chi_t_opt": float(chi_t_opt), "C": float(Co),
            "F_schmidt": float(Fs), "sigma_s": float(ss),
            "F_naive": float(Fn), "sigma_n": float(sn),
            "gain_meas": float(gain_meas), "gain_pred": float(gain_pred),
            "gain_ok": gain_ok, "dF": float(dF),
            "bond_dim": bond_dim, "F_pred": float(Fpo)
        }

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*65)
    h1_pass   = all(v["dF"]   < 0.005  for v in all_results.values())
    gain_pass = all(v["gain_ok"]        for v in all_results.values())
    print(f"  H1 F_avg=(2+C)/3:      {'CONFIRMED' if h1_pass else 'PARTIAL'}")
    print(f"  Gain=(1-C)/6 law:      {'CONFIRMED' if gain_pass else 'PARTIAL'}")
    print(f"  Cross-validation:      {'PASS' if xv_pass else 'FAIL'}")
    print("="*65+"\n", flush=True)

    out = {
        "experiment": "OAT Teleport v3 — Tensor Network MPS (direct automaton)",
        "config": {"N": args.N, "chi_t_steps": args.chi_t_steps, "K": args.K},
        "crossval_pass": xv_pass,
        "results": all_results,
        "summary": {
            "H1": "CONFIRMED" if h1_pass else "PARTIAL",
            "Gain_law": "CONFIRMED" if gain_pass else "PARTIAL",
            "CrossVal": "PASS" if xv_pass else "FAIL"
        }
    }
    with open("oat_teleport_v3_results.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("[DONE] oat_teleport_v3_results.json", flush=True)


if __name__ == "__main__":
    main()
