"""
jila_oat_exact_tpu.py  v3
=========================
Exact OAT quantum teleportation — JAX/TPU statevector simulation.

Physics: H = chi * Jz_A * Jz_B  (diagonal in computational basis)
         Evolution: psi[k] *= exp(-i chi t mA[k] mB[k])  [exact]

Fidelity metric (Bowen & Bose, PRL 87, 2001):
  F_opt = (2 + C) / 3   where C = concurrence of boundary qubit pair.
  F > 2/3  iff  C > 0  (any entanglement).

Concurrence of 2-qubit mixed state rho (Wootters 1998):
  C = max(0, sqrt(e1) - sqrt(e2) - sqrt(e3) - sqrt(e4))
  where e_i are eigenvalues of rho @ (sysy) @ rho.conj() @ (sysy)
  in decreasing order.
"""

import argparse, json, math
import numpy as np
from pathlib import Path
import jax, jax.numpy as jnp
from jax import jit

print(f"[DEVICE] {jax.devices()}", flush=True)
print(f"[DEVICE] backend={jax.default_backend()}", flush=True)


# ── Precompute Jz eigenvalues ─────────────────────────────────────────────────
def jz_table(N):
    NA = N // 2
    idx = np.arange(2**N, dtype=np.int64)
    mA = sum(0.5 - ((idx >> i) & 1).astype(np.float32) for i in range(NA))
    mB = sum(0.5 - ((idx >> (NA+i)) & 1).astype(np.float32) for i in range(N-NA))
    return mA.astype(np.float32), mB.astype(np.float32)


# ── Initial state |+>^N ───────────────────────────────────────────────────────
def plus_state(N):
    psi = np.ones(1, dtype=np.complex64)
    plus = np.array([1., 1.], dtype=np.complex64) / math.sqrt(2)
    for _ in range(N):
        psi = np.kron(psi, plus)
    return psi


# ── Exact OAT evolution ───────────────────────────────────────────────────────
@jit
def oat_evolve(psi, mA, mB, chi_t):
    return psi * jnp.exp((-1j * chi_t) * (mA * mB))


# ── 2-qubit reduced density matrix for boundary pair ─────────────────────────
def boundary_rdm(psi_np, N):
    """
    Trace over all qubits except the boundary pair:
      qubit (NA-1) = last qubit of chain A  [bit NA-1 of state index]
      qubit  NA    = first qubit of chain B  [bit NA of state index]
    Returns rho_2 of shape (4,4).
    """
    NA = N // 2
    low  = max(1, 2**(NA-1))   # bits 0..NA-2
    high = max(1, 2**(N-NA-1)) # bits NA+1..N-1
    # sv shape: (low, bit_A, bit_B, high)
    sv4 = psi_np.reshape(low, 2, 2, high)
    rho = np.zeros((4, 4), dtype=np.complex64)
    for i in range(2):
        for j in range(2):
            for ii in range(2):
                for jj in range(2):
                    rho[i*2+j, ii*2+jj] = np.sum(
                        sv4[:, i, j, :].conj() * sv4[:, ii, jj, :]
                    )
    rho /= np.trace(rho)
    return rho


# ── Concurrence (Wootters 1998) ───────────────────────────────────────────────
def concurrence(rho):
    sy = np.array([[0, -1j], [1j, 0]], dtype=np.complex64)
    sysy = np.kron(sy, sy)
    R = rho @ sysy @ rho.conj() @ sysy
    evals = np.sort(np.real(np.linalg.eigvals(R)))[::-1]
    evals = np.maximum(evals, 0)
    sqe = np.sqrt(evals)
    return float(max(0.0, sqe[0] - sqe[1] - sqe[2] - sqe[3]))


# ── Teleportation fidelity from concurrence ───────────────────────────────────
def fidelity_from_concurrence(C):
    return (2.0 + C) / 3.0


# ── Collective channel fidelity (full many-body) ─────────────────────────────
def collective_fidelity(psi_np, N):
    """
    Average fidelity of collective spin teleportation.
    Alice measures Jz_A, Bob recovers collective state of chain B.
    Target: |+>^NB.  Bob applies optimal Ry(theta) rotation.
    F = sum_m p(m) * max_theta |<+|^NB Ry(theta) |psi_B(m)>|^2
    """
    NA = N // 2
    NB = N - NA
    nA, nB = 2**NA, 2**NB
    mA_np, _ = jz_table(N)

    # CSS target on B
    css = np.ones(nB, dtype=np.complex64) / math.sqrt(nB)

    # Build collective Jy_B (sum of single-site Sy operators)
    Sy = np.array([[0, -0.5j], [0.5j, 0]], dtype=np.complex64)
    I2 = np.eye(2, dtype=np.complex64)
    Jy_B = np.zeros((nB, nB), dtype=np.complex64)
    for i in range(NB):
        op = np.eye(1, dtype=np.complex64)
        for j in range(NB):
            op = np.kron(op, Sy if j == i else I2)
        Jy_B += op
    evals, evecs = np.linalg.eigh(Jy_B)

    thetas = np.linspace(0, 2*math.pi, 64, endpoint=False)
    F_total = 0.0

    for m in np.unique(mA_np):
        mask = (mA_np == m)
        sv_proj = psi_np * mask
        prob = float(np.sum(np.abs(sv_proj)**2))
        if prob < 1e-14:
            continue
        sv_proj /= math.sqrt(prob)
        sv_b = sv_proj.reshape(nA, nB)
        rho_B = sv_b.T @ sv_b.conj()

        best = 0.0
        for theta in thetas:
            R = evecs @ np.diag(np.exp(-1j * theta * evals)) @ evecs.conj().T
            rho_rot = R @ rho_B @ R.conj().T
            f = float(np.real(css.conj() @ rho_rot @ css))
            if f > best:
                best = f
        F_total += prob * best

    return float(F_total)


# ── Scan over chi*t ───────────────────────────────────────────────────────────
def run_scan(N, chi_t_vals, use_collective=False):
    mA_np, mB_np = jz_table(N)
    mA = jnp.array(mA_np)
    mB = jnp.array(mB_np)
    psi0 = jnp.array(plus_state(N))

    Fvals, Cvals = [], []
    for chi_t in chi_t_vals:
        psi_t = np.array(oat_evolve(psi0, mA, mB, float(chi_t)))
        rho2 = boundary_rdm(psi_t, N)
        C = concurrence(rho2)
        F = fidelity_from_concurrence(C)
        if use_collective and N <= 10:
            F = collective_fidelity(psi_t, N)
        Fvals.append(F)
        Cvals.append(C)
    return np.array(Fvals), np.array(Cvals)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N",          nargs="+", type=int,   default=[2,4,6,8,10,12])
    ap.add_argument("--chi_max",    type=float, default=math.pi)
    ap.add_argument("--steps",      type=int,   default=64)
    ap.add_argument("--collective", action="store_true",
                    help="Also compute full collective-spin fidelity (slow, N<=10)")
    args = ap.parse_args()

    chi_t_vals = np.linspace(0.0, args.chi_max, args.steps)
    CL = 2.0/3.0

    print(f"\n{'='*65}")
    print("  JILA OAT Exact Teleportation  —  JAX/TPU Statevector Sim")
    print(f"  Devices : {jax.devices()}")
    print(f"  Fidelity: F = (2+C)/3  [Bowen & Bose, PRL 87 2001]")
    print(f"  Threshold: F > 2/3 = {CL:.4f}  (classical limit)")
    print(f"{'='*65}\n", flush=True)

    all_results = {}

    for N in args.N:
        if N % 2 != 0 or 2**N > 2**14:
            print(f"  [SKIP] N={N}")
            continue

        print(f"  N={N}  ({N//2} per chain) ...", flush=True)
        Fvals, Cvals = run_scan(N, chi_t_vals, use_collective=args.collective)

        F_peak  = float(Fvals.max())
        C_peak  = float(Cvals.max())
        t_opt   = float(chi_t_vals[Fvals.argmax()])
        adv     = bool(F_peak > CL)

        step = max(1, args.steps // 8)
        for i in range(0, args.steps, step):
            ct, F, C = chi_t_vals[i], Fvals[i], Cvals[i]
            tag = " *** QUANTUM ADVANTAGE ***" if F > CL else ""
            print(f"    chi*t={ct:.4f}  C={C:.5f}  F={F:.5f}{tag}", flush=True)

        print(f"  -> PEAK: chi*t={t_opt:.4f}  C={C_peak:.5f}  "
              f"F={F_peak:.5f}  F>2/3={adv}\n", flush=True)

        all_results[f"N{N}"] = dict(
            N_ions=N, NA=N//2, NB=N//2,
            F_peak=F_peak, C_peak=C_peak, chi_t_optimal=t_opt,
            quantum_advantage=adv, classical_limit=CL,
            F_curve=Fvals.tolist(), C_curve=Cvals.tolist(),
            chi_t_curve=chi_t_vals.tolist(),
        )

    print(f"\n{'='*65}")
    print(f"  SUMMARY  (classical limit F_cl = {CL:.4f})")
    print(f"  {'N':>4}  {'F_peak':>9}  {'C_peak':>8}  {'chi*t_opt':>10}  {'Advantage':>10}")
    print("  " + "-"*55)
    for v in all_results.values():
        tag = "YES ***" if v["quantum_advantage"] else "no"
        print(f"  {v['N_ions']:>4}  {v['F_peak']:>9.5f}  {v['C_peak']:>8.5f}  "
              f"{v['chi_t_optimal']:>10.4f}  {tag:>10}")
    print(f"{'='*65}\n")

    out = Path("jila_oat_exact_results.json")
    with open(out, "w") as f:
        json.dump(dict(
            experiment="JILA OAT Exact Teleportation - JAX/TPU",
            fidelity_formula="F = (2+C)/3, Bowen & Bose PRL 87 2001",
            concurrence_method="Wootters 1998, boundary qubit pair",
            devices=str(jax.devices()),
            classical_limit=CL,
            chi_t_range=[0.0, args.chi_max],
            results=all_results,
        ), f, indent=2)
    print(f"[DONE] {out}", flush=True)


if __name__ == "__main__":
    main()
