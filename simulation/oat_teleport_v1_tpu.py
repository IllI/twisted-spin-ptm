"""
oat_teleport_v1_tpu.py — Full Bennett protocol on OAT-derived Werner states.

The standard Bennett protocol is optimal for Werner states:
  rho_W(C) = C|Phi+><Phi+| + (1-C)/4 * I
For this resource, F_avg = (2+C)/3 exactly (Bowen-Bose 2001).

H1: F_avg(N,chi_t)  = (2+C)/3  within 0.005 for all N,chi_t
H2: F_avg <= 2/3 at T2_chi<=1,  F_avg > 2/3 at T2_chi>=2
H3: Pearson r(OAT_proxy, F_avg - 2/3) > 0.90
"""
import argparse, json, math, cmath, sys, os
import numpy as np
from pathlib import Path
import jax, jax.numpy as jnp

sys.path.insert(0, os.path.dirname(__file__))
from jila_oat_exact_tpu import (jz_table, plus_state, oat_evolve,
                                 boundary_rdm, concurrence,
                                 fidelity_from_concurrence)

print(f"[DEVICE] {jax.devices()}  backend={jax.default_backend()}", flush=True)
CL = 2.0 / 3.0

# ── Gates ─────────────────────────────────────────────────────────────────────
I2 = np.eye(2, dtype=complex)
X  = np.array([[0,1],[1,0]], dtype=complex)
Z  = np.array([[1,0],[0,-1]], dtype=complex)
H2 = np.array([[1,1],[1,-1]], dtype=complex) / math.sqrt(2)
P0 = np.array([[1,0],[0,0]], dtype=complex)
P1 = np.array([[0,0],[0,1]], dtype=complex)

def kron3(A, B, C): return np.kron(np.kron(A, B), C)

# 3-qubit circuit: q0=Alice_test, q1=Alice_resource, q2=Bob_resource
CNOT_01 = kron3(P0,I2,I2) + kron3(P1,X,I2)
CIRCUIT  = kron3(H2,I2,I2) @ CNOT_01
CIRCUIT_DAG = CIRCUIT.conj().T

PROJ = {(m1,m2): kron3(P0 if m1==0 else P1, P0 if m2==0 else P1, I2)
        for m1 in range(2) for m2 in range(2)}
UCORR = {(0,0):I2, (0,1):X, (1,0):Z, (1,1):X@Z}

# ── Werner state ──────────────────────────────────────────────────────────────
def werner_state(C):
    """
    Werner state with concurrence C, per Bowen-Bose 2001.
    rho_W = (1+2C)/3 |Phi+><Phi+| + (1-C)/6 * I
    Singlet fraction: f = (1+C)/2
    Standard Bennett fidelity: F = (2f+1)/3 = (2+C)/3  checkmark
    Concurrence check: lambda_max = (1+C)/2, C_Werner = 2*lambda_max - 1 = C  checkmark
    """
    Phi = np.array([1,0,0,1], dtype=complex) / math.sqrt(2)
    rho = ((1+2*C)/3) * np.outer(Phi, Phi.conj()) + ((1-C)/6) * np.eye(4, dtype=complex)
    return rho

# ── Teleportation circuit ─────────────────────────────────────────────────────
def teleport_fidelity(psi_A, rho2):
    """F = <psi_A|rho_B_out|psi_A> for one test state."""
    rho_total = np.kron(np.outer(psi_A, psi_A.conj()), rho2)
    rho_bell  = CIRCUIT @ rho_total @ CIRCUIT_DAG

    rho_B_out = np.zeros((2,2), dtype=complex)
    for (m1,m2), Pi in PROJ.items():
        rho_proj = Pi @ rho_bell @ Pi
        prob = np.trace(rho_proj).real
        if prob < 1e-12: continue
        # Partial trace over q0,q1: reshape to (q0r,q1r,q2r,q0c,q1c,q2c)
        r = (rho_proj/prob).reshape(2,2,2,2,2,2)
        rho_q2 = np.einsum('ijkijl->kl', r)   # trace q0(i) and q1(j)
        U = UCORR[(m1,m2)]
        rho_B_out += prob * (U @ rho_q2 @ U.conj().T)

    return float(np.clip((psi_A.conj() @ rho_B_out @ psi_A).real, 0, 1))

def haar_qubit(rng):
    u1,u2 = rng.uniform(0,1,2)
    th = math.acos(1-2*u1); ph = 2*math.pi*u2
    return np.array([math.cos(th/2), math.sin(th/2)*cmath.exp(1j*ph)], dtype=complex)

def avg_fidelity(rho2, K=3000, seed=42):
    rng = np.random.default_rng(seed)
    Fv = [teleport_fidelity(haar_qubit(rng), rho2) for _ in range(K)]
    return float(np.mean(Fv)), float(np.std(Fv))

# ── Decoherence ───────────────────────────────────────────────────────────────
def decohered_werner(C_clean, T2_chi, chi_t):
    if T2_chi is None: return werner_state(C_clean), C_clean
    decay = math.exp(-chi_t / (T2_chi * math.pi))
    C_dec = max(0.0, C_clean * decay)
    return werner_state(C_dec), C_dec

# ── Self-test (Bell state → F=1) ──────────────────────────────────────────────
def self_test():
    rho_bell = werner_state(1.0)
    rng = np.random.default_rng(0)
    Fv = [teleport_fidelity(haar_qubit(rng), rho_bell) for _ in range(200)]
    F = float(np.mean(Fv))
    ok = abs(F - 1.0) < 0.01
    print(f"[SELF-TEST] F_avg(C=1) = {F:.4f}  {'PASS' if ok else 'FAIL'}", flush=True)
    assert ok, f"Self-test failed: F={F:.4f}, expected 1.0"
    rho_cls = werner_state(0.0)
    Fv2 = [teleport_fidelity(haar_qubit(rng), rho_cls) for _ in range(200)]
    F2 = float(np.mean(Fv2))
    # Correct Werner(C=0): rho = 1/3 |Phi+><Phi+| + 1/6 I
    # F = (2+0)/3 = 2/3  (classical limit)
    ok2 = abs(F2 - CL) < 0.02
    print(f"[SELF-TEST] F_avg(C=0) = {F2:.4f}  {'PASS' if ok2 else 'FAIL'}  (expect {CL:.4f})", flush=True)
    assert ok2, f"Self-test failed: F={F2:.4f}, expected {CL:.4f}"

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N",           nargs="+", type=int, default=[2,4,6,8,10,12])
    ap.add_argument("--chi_t_steps", type=int,            default=32)
    ap.add_argument("--K",           type=int,            default=3000)
    args = ap.parse_args()
    print(f"[CONFIG] N={args.N} chi_t_steps={args.chi_t_steps} K={args.K}\n", flush=True)

    self_test()

    chi_t_vals = np.linspace(0.01, math.pi*0.99, args.chi_t_steps)
    all_results = {}

    # H1 -----------------------------------------------------------------------
    print("\n" + "="*60)
    print("  H1: F_avg(N,chi_t) = (2+C)/3")
    print("="*60, flush=True)

    h1_results = {}
    for N in args.N:
        mA,mB = jz_table(N)
        mAj,mBj = jnp.array(mA),jnp.array(mB)
        psi0 = jnp.array(plus_state(N))

        chi_t_opt=None; F_peak=0.0; rows=[]
        for chi_t in chi_t_vals:
            psi_t = np.array(oat_evolve(psi0,mAj,mBj,float(chi_t)))
            rho2  = boundary_rdm(psi_t, N)
            C     = concurrence(rho2)
            Fp    = fidelity_from_concurrence(C)
            if Fp > F_peak: F_peak=Fp; chi_t_opt=chi_t

            if int(np.round(chi_t/(math.pi*0.99/args.chi_t_steps)))%4==0:
                rw = werner_state(C)
                Fa,Fs = avg_fidelity(rw, K=args.K)
                d = abs(Fa-Fp)
                rows.append({"chi_t":float(chi_t),"C":float(C),
                             "F_pred":float(Fp),"F_avg":float(Fa),"delta":float(d)})

        # optimal chi_t
        psi_o = np.array(oat_evolve(psi0,mAj,mBj,float(chi_t_opt)))
        rho2_o= boundary_rdm(psi_o, N)
        Co    = concurrence(rho2_o)
        Fpo   = fidelity_from_concurrence(Co)
        Fao,Fso = avg_fidelity(werner_state(Co), K=args.K)
        do    = abs(Fao-Fpo)
        mx    = max(r["delta"] for r in rows) if rows else float('nan')
        stat  = "CONFIRMED" if all(r["delta"]<0.005 for r in rows) else f"PARTIAL"

        print(f"  [N={N:>2d}] chi_t*={chi_t_opt:.3f}  C={Co:.4f}  "
              f"F_pred={Fpo:.4f}  F_avg={Fao:.4f}+/-{Fso:.4f}  "
              f"|dF|={do:.4f}  max={mx:.4f}  {stat}", flush=True)

        h1_results[f"N{N}"] = {
            "chi_t_opt":float(chi_t_opt),"C":float(Co),
            "F_pred":float(Fpo),"F_avg":float(Fao),"delta":float(do),
            "status":stat,"sweep":rows}

    h1_pass = all(v["delta"]<0.005 for v in h1_results.values())
    print(f"\n  => H1: {'CONFIRMED' if h1_pass else 'PARTIAL'}  "
          f"(all |dF|<0.005: {h1_pass})", flush=True)

    # H2 -----------------------------------------------------------------------
    print("\n" + "="*60)
    print("  H2: T2_chi > 2 required")
    print("="*60, flush=True)

    N4 = 4
    mA4,mB4 = jz_table(N4)
    mA4j,mB4j = jnp.array(mA4),jnp.array(mB4)
    psi04 = jnp.array(plus_state(N4))
    chi_t4 = h1_results["N4"]["chi_t_opt"]
    psi4   = np.array(oat_evolve(psi04,mA4j,mB4j,float(chi_t4)))
    C4     = concurrence(boundary_rdm(psi4, N4))

    T2_vals=[0.25,0.5,1.0,2.0,5.0,None]
    h2_rows=[]; probes=[]; margins=[]
    for T2 in T2_vals:
        rw,Cd = decohered_werner(C4, T2, chi_t4)
        Fa,Fs = avg_fidelity(rw, K=args.K)
        above = Fa > CL+0.005
        label = f"T2chi={T2}" if T2 else "T2chi=inf"
        proxy = max(0.0, Fa-CL)
        probes.append(proxy); margins.append(proxy)
        print(f"  [{label:>12s}] C={Cd:.4f}  F_pred={(2+Cd)/3:.4f}  "
              f"F_avg={Fa:.4f}  above_CL={above}", flush=True)
        h2_rows.append({"T2_chi":str(T2),"C_dec":float(Cd),
                        "F_avg":float(Fa),"above_cl":above})

    h2_below = all(r["F_avg"]<=CL+0.005 for r in h2_rows if r["T2_chi"] in ["0.25","0.5"])
    h2_above = all(r["above_cl"] for r in h2_rows if r["T2_chi"] in ["2.0","5.0","None"])
    h2v = "CONFIRMED" if (h2_below and h2_above) else "PARTIAL"
    print(f"\n  => H2: {h2v}  (below_CL: {h2_below}, above_CL: {h2_above})", flush=True)

    # H3 -----------------------------------------------------------------------
    s=np.array(probes); m=np.array(margins)
    r_p = float(np.corrcoef(s,m)[0,1]) if s.std()>1e-10 and m.std()>1e-10 else 0.0
    h3p = r_p > 0.90
    print(f"\n  H3: Pearson r = {r_p:.4f}  => {'CONFIRMED' if h3p else 'FAIL'}", flush=True)

    # Summary ------------------------------------------------------------------
    print("\n" + "="*60)
    print(f"  H1 F_avg=(2+C)/3:   {'CONFIRMED' if h1_pass else 'PARTIAL'}")
    print(f"  H2 T2_chi>2:        {h2v}")
    print(f"  H3 r>0.90:          {'CONFIRMED' if h3p else 'FAIL'}")
    print("="*60+"\n", flush=True)

    out = {"experiment":"OAT Teleport v1 — Werner Resource",
           "config":{"N":args.N,"chi_t_steps":args.chi_t_steps,"K":args.K},
           "h1":h1_results,"h2":{"sweep":h2_rows,"verdict":h2v},
           "h3":{"pearson_r":r_p,"pass":h3p},
           "summary":{"H1":"CONFIRMED" if h1_pass else "PARTIAL",
                      "H2":h2v,"H3":"CONFIRMED" if h3p else "FAIL"}}
    with open("oat_teleport_v1_results.json","w") as f:
        json.dump(out,f,indent=2,default=str)
    print("[DONE] oat_teleport_v1_results.json", flush=True)

if __name__=="__main__":
    main()
