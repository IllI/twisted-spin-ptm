"""
oat_teleport_v4_open_system.py — Experiment v4: Open-System OAT Teleportation

Applies analytic Lindblad dephasing to the exact OAT boundary RDM from v3,
then sweeps decoherence rate Gamma to find the entanglement threshold.

Physics:
  Under independent dephasing L_k = sqrt(Gamma) * sigma_z^(k):
    rho2[|00>,|11>](t) = rho2[|00>,|11>](0) * exp(-4*Gamma*t)
    rho2[|01>,|10>](t) = rho2[|01>,|10>](0) * exp(-4*Gamma*t)
  Diagonal elements (populations) unchanged.

Experiment phases:
  Phase 0: Decoherence phase diagram — F(Gamma, N) for N=2..16
  Phase 1: Threshold measurement — Gamma*(N) where F drops to 2/3
  Phase 2: CHSH consistency check — S_measured vs S_Horodecki at optimal Gamma=0
"""
import math, json, sys, os
import numpy as np
from scipy.optimize import brentq
sys.path.insert(0, os.path.dirname(__file__))
from oat_teleport_v3_tpu import oat_mps, extract_boundary_rho
from jila_oat_exact_tpu import concurrence

# Physical constants from Dr. Rey's lab
GAMMA_SINGLE = 1.0 / 118.0   # s^-1, single-particle dephasing (arXiv:2505.06444)
CHI_Hz       = 1.0             # Hz, OAT interaction rate (order-of-magnitude)

PAULI_X = np.array([[0,1],[1,0]], dtype=complex)
PAULI_Y = np.array([[0,-1j],[1j,0]], dtype=complex)
PAULI_Z = np.array([[1,0],[0,-1]], dtype=complex)
SIGMAS  = [PAULI_X, PAULI_Y, PAULI_Z]


def apply_dephasing(rho2, gamma_t_eff):
    """
    Apply dephasing to the 2-qubit boundary RDM.
    
    gamma_t_eff = Gamma * t_OAT is the dimensionless decoherence parameter.
    
    Under independent dephasing on both qubits, off-diagonals decay as:
      rho[i,j] *= exp(-Gamma * d_H(i,j)^2 * t)
    
    For 2-qubit states with Hamming distances:
      d_H(00,11) = d_H(01,10) = 2  -> factor exp(-4*gamma_t_eff)
      d_H(00,01) = d_H(00,10) = 1  -> factor exp(-1*gamma_t_eff)
      d_H(00,00) = 0               -> factor 1 (diagonal unchanged)
    
    In the {|00>,|01>,|10>,|11>} basis (rows/cols 0,1,2,3):
    """
    # Hamming distance matrix for 2-qubit basis
    basis = [(0,0),(0,1),(1,0),(1,1)]
    dH = np.zeros((4,4))
    for i, (a,b) in enumerate(basis):
        for j, (c,d) in enumerate(basis):
            dH[i,j] = abs(a-c) + abs(b-d)
    
    decay = np.exp(-gamma_t_eff * dH**2)
    return rho2 * decay


def chsh_horodecki(rho):
    T = np.zeros((3,3))
    for i, si in enumerate(SIGMAS):
        for j, sj in enumerate(SIGMAS):
            T[i,j] = float(np.real(np.trace(rho @ np.kron(si, sj))))
    eigs = np.linalg.eigvalsh(T.T @ T)
    return 2 * math.sqrt(max(0, sorted(eigs)[-1] + sorted(eigs)[-2]))


def witness_value(rho):
    """Best Bell-state entanglement witness value (negative = entangled)."""
    bells = [
        np.array([1,0,0,1], dtype=complex) / math.sqrt(2),
        np.array([1,0,0,-1], dtype=complex) / math.sqrt(2),
        np.array([0,1,1,0], dtype=complex) / math.sqrt(2),
        np.array([0,1,-1,0], dtype=complex) / math.sqrt(2),
    ]
    f_vals = [float(np.real(b.conj() @ rho @ b)) for b in bells]
    return 0.25 - max(f_vals)


def analyze_at_gamma(rho2_ideal, gamma_t_eff):
    """Compute all observables for a given decoherence level."""
    rho = apply_dephasing(rho2_ideal, gamma_t_eff)
    # Renormalize (dephasing may introduce tiny numerical errors)
    rho /= np.trace(rho).real
    C = float(concurrence(rho))
    F = (2 + C) / 3
    S = chsh_horodecki(rho)
    W = witness_value(rho)
    return {"C": C, "F": F, "S_chsh": S, "witness": W,
            "gamma_t_eff": gamma_t_eff}


def find_threshold(rho2_ideal, target_F=2/3+1e-6):
    """Binary search for Gamma_t_eff where F drops to target_F."""
    def f(g):
        return analyze_at_gamma(rho2_ideal, g)["F"] - target_F
    
    f0 = f(0)
    if f0 <= 0:
        return 0.0  # already below threshold at Gamma=0
    
    # Find upper bracket
    g_high = 0.01
    while f(g_high) > 0 and g_high < 100:
        g_high *= 2
    
    if f(g_high) > 0:
        return float('inf')  # never crosses threshold
    
    return brentq(f, 0, g_high, xtol=1e-8)


def main():
    print("=" * 72)
    print("  Experiment v4: Open-System OAT Teleportation")
    print("  Lindblad Dephasing Phase Diagram")
    print("=" * 72)

    # OAT sweep: N values and their optimal chi_t
    sweep = [
        (2,  2.985), (4,  1.091), (6,  0.713),
        (8,  0.523), (10, 0.429), (12, 0.334), (16, 0.239),
    ]

    # Gamma * t_opt grid (dimensionless decoherence)
    gamma_t_grid = np.concatenate([
        np.linspace(0, 0.5, 40),
        np.linspace(0.5, 5.0, 20),
    ])

    all_results = {}
    threshold_results = []

    # ── Phase 0: Decoherence Phase Diagram ──────────────────────────────────
    print("\n--- Phase 0: Decoherence Phase Diagram ---")
    for N, chi_t_opt in sweep:
        n = N // 2
        tensors = oat_mps(N, chi_t_opt)
        rho2_ideal = extract_boundary_rho(tensors, n)
        C_ideal = float(concurrence(rho2_ideal))
        t_opt_sec = chi_t_opt / CHI_Hz  # seconds at chi=1 Hz

        series = []
        for g_t in gamma_t_grid:
            res = analyze_at_gamma(rho2_ideal, g_t)
            # Convert gamma_t_eff to physical Gamma given t_opt
            gamma_phys = g_t / t_opt_sec if t_opt_sec > 0 else 0
            res["gamma_phys"] = float(gamma_phys)
            series.append(res)

        all_results[str(N)] = {
            "N": N, "chi_t_opt": chi_t_opt,
            "t_opt_sec": t_opt_sec, "C_ideal": C_ideal,
            "series": series
        }

        # Find F=2/3 crossing
        g_thresh = find_threshold(rho2_ideal, target_F=2/3 + 1e-6)
        gamma_thresh_phys = g_thresh / t_opt_sec if t_opt_sec > 0 else float('inf')
        gamma_thresh_vs_single = gamma_thresh_phys / GAMMA_SINGLE

        print(f"  N={N:>3d}  C₀={C_ideal:.4f}  t*={t_opt_sec:.2f}s  "
              f"Γ_thresh={gamma_thresh_phys:.4f} s⁻¹  "
              f"Γ_thresh/Γ₁={gamma_thresh_vs_single:.1f}×  "
              f"H1: {'PASS (Γ*>>Γ₁)' if gamma_thresh_vs_single > 2 else 'marginal/FAIL'}")

        threshold_results.append({
            "N": N, "C_ideal": C_ideal,
            "chi_t_opt": chi_t_opt, "t_opt_sec": t_opt_sec,
            "gamma_t_threshold": float(g_thresh),
            "gamma_thresh_phys": float(gamma_thresh_phys),
            "gamma_thresh_vs_single": float(gamma_thresh_vs_single),
            "F_at_gamma1": float(analyze_at_gamma(rho2_ideal, GAMMA_SINGLE * t_opt_sec)["F"]),
            "C_at_gamma1": float(analyze_at_gamma(rho2_ideal, GAMMA_SINGLE * t_opt_sec)["C"]),
        })

    # ── Phase 1: Scaling Law Fit ─────────────────────────────────────────────
    print("\n--- Phase 1: Threshold Scaling Law ---")
    Ns = np.array([r["N"] for r in threshold_results if r["gamma_thresh_phys"] < 1e6])
    G_thresh = np.array([r["gamma_thresh_phys"] for r in threshold_results
                         if r["gamma_thresh_phys"] < 1e6])

    if len(Ns) >= 3:
        log_N = np.log(Ns)
        log_G = np.log(G_thresh)
        alpha_fit, log_A_fit = np.polyfit(log_N, log_G, 1)
        A_fit = math.exp(log_A_fit)
        G_pred = A_fit * Ns**alpha_fit
        ss_res = np.sum((log_G - log_G.pred)**2) if hasattr(log_G, 'pred') else \
                 np.sum((np.log(G_pred) - log_G)**2)
        ss_tot = np.sum((log_G - np.mean(log_G))**2)
        R2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0

        print(f"  Fit: Γ*(N) = {A_fit:.4f} × N^{{{alpha_fit:.4f}}}")
        print(f"  R² = {R2:.4f}  ({'PASS' if R2 > 0.90 else 'poor fit'})")
        print(f"  α = {alpha_fit:.4f} → ", end="")
        if abs(alpha_fit + 1) < 0.2:
            print("Γ* ∝ 1/N (linear fragility with N)")
        elif abs(alpha_fit) < 0.1:
            print("Γ* ≈ const (N-independent robustness)")
        else:
            print(f"power law with exponent {alpha_fit:.3f}")
    else:
        alpha_fit, A_fit, R2 = float('nan'), float('nan'), float('nan')
        print("  Insufficient data points for fit")

    # ── Phase 2: F at Real Sr-87 Decoherence ────────────────────────────────
    print("\n--- Phase 2: F at Actual Sr-87 Γ₁ = 1/118 s⁻¹ ---")
    print(f"  Γ₁ = {GAMMA_SINGLE:.5f} s⁻¹  (T₂=118s, arXiv:2505.06444)")
    print(f"  {'N':>4}  {'t*':>6}  {'Γ₁t*':>8}  {'C(Γ₁)':>8}  "
          f"{'F(Γ₁)':>8}  {'F>2/3?':>8}  {'Witness':>9}")
    for r in threshold_results:
        gamma1_t = GAMMA_SINGLE * r["t_opt_sec"]
        print(f"  {r['N']:>4}  {r['t_opt_sec']:>6.2f}s  {gamma1_t:>8.5f}  "
              f"{r['C_at_gamma1']:>8.5f}  {r['F_at_gamma1']:>8.5f}  "
              f"{'YES' if r['F_at_gamma1'] > 2/3 else 'NO':>8}  "
              f"Γ_thresh/Γ₁={r['gamma_thresh_vs_single']:.0f}×")

    # ── Phase 3: CHSH at Gamma=0 (consistency with audit) ───────────────────
    print("\n--- Phase 3: CHSH Consistency Check (Gamma=0) ---")
    for N, chi_t_opt in [(2,2.985),(4,1.091),(8,0.523),(16,0.239)]:
        n = N // 2
        tensors = oat_mps(N, chi_t_opt)
        rho2 = extract_boundary_rho(tensors, n)
        S = chsh_horodecki(rho2)
        C = concurrence(rho2)
        print(f"  N={N:>3d}: C={C:.4f}  S_chsh={S:.4f}  "
              f"{'VIOLATES' if S>2 else 'no violation'}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  HYPOTHESIS RESULTS")
    print("=" * 72)
    h1 = all(r["gamma_thresh_vs_single"] > 2 for r in threshold_results)
    print(f"  H1 (Γ* >> Γ₁):   {'CONFIRMED' if h1 else 'PARTIAL/FAILED'}")
    print(f"  H2 (scaling α):   α = {alpha_fit:.3f}  R²={R2:.3f}")
    print(f"  H3 (witness ≤ F): see full series in v4_results.json")

    # Save
    output = {
        "experiment": "v4_open_system",
        "gamma_single_sr87": GAMMA_SINGLE,
        "chi_Hz": CHI_Hz,
        "threshold_scaling": {
            "alpha": float(alpha_fit) if not math.isnan(alpha_fit) else None,
            "A": float(A_fit) if not math.isnan(A_fit) else None,
            "R2": float(R2) if not math.isnan(R2) else None,
        },
        "per_N": threshold_results,
        "phase_diagram": all_results,
    }
    with open("v4_open_system_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print("\n[DONE] v4_open_system_results.json")


if __name__ == "__main__":
    main()
