"""
ibm_emulator.py — Predictive IBM Falcon/Eagle Noise Emulator
=============================================================
Applies realistic IBM hardware noise to theoretical OAT PTMs and asks:
  "Do hardware-corrupted channels still land in the inferred TRANSPORT stratum?"

This converts D-LinOSS from exploratory tool to predictive instrument:
  "The IBM run is not exploratory. D-LinOSS predicts the hardware PTMs
   should fall into the transport-enriched stratum identified blindly
   from simulated operator geometry."

Noise model (calibrated to ibm_marrakesh, Run 1-3 data):
  - T1/T2 decay on 2-qubit gate layer
  - Readout asymmetry P(1|0)=0.002, P(0|1)=0.014
  - Depolarizing gate error p_gate=0.002 per 2q gate
  - Shot noise: multinomial sampling at N_shots

Usage:
  python ibm_emulator.py                         # predict next IBM run
  python ibm_emulator.py --validate teleport_results.json   # compare vs real data
  python ibm_emulator.py --chi-scan              # full chi_t scan
"""
import argparse, json, os
import numpy as np
from scipy.optimize import minimize

from ptm_features import (PTMFeatures, PTMFeatureVector, rho2_exact,
                           compute_T_matrix, f_max_analytic)
from dlinoss_geometry import spectral_entropy
from dlinoss_program_f import infer_strata, _MASK

# ── IBM hardware parameters (ibm_marrakesh calibration) ──────────────────────
IBM_PARAMS = {
    "T1_us":    150.0,    # T1 relaxation time (microseconds)
    "T2_us":     80.0,    # T2 dephasing time
    "t_gate_us":  0.5,    # 2-qubit gate duration
    "n_2q_gates": 6,      # approximate 2q gates in teleportation circuit
    "p_readout_10": 0.002,  # P(measure 1 | state 0) — from Run 1 calibration
    "p_readout_01": 0.014,  # P(measure 0 | state 1)
    "p_depol_2q":   0.002,  # depolarizing per 2q gate
    "N_shots":    1024,   # default shots per measurement setting
}


# ── Noise channels ────────────────────────────────────────────────────────────
def t1t2_decay(rho: np.ndarray, T1: float, T2: float, t: float) -> np.ndarray:
    """
    Amplitude damping + dephasing on each qubit independently.
    Applied as approximate diagonal channel (realistic for gate-based circuits).
    T1, T2 in same units as t.
    """
    p1 = 1 - np.exp(-t / T1)    # amplitude damping probability
    pd = 1 - np.exp(-t / T2)    # dephasing probability
    result = rho.copy().astype(complex)
    # Apply to each qubit independently (product channel)
    for q in range(2):  # qubit 0 and 1
        K0 = np.array([[1, 0], [0, np.sqrt(1 - p1)]])
        K1 = np.array([[0, np.sqrt(p1)], [0, 0]])
        Kd0 = np.array([[1, 0], [0, np.sqrt(1 - pd)]])
        Kd1 = np.array([[0, 0], [0, np.sqrt(pd)]])
        kraus = [np.kron(np.eye(2), K) if q == 0 else np.kron(K, np.eye(2))
                 for K in [K0, K1, Kd0, Kd1]]
        tmp = np.zeros((4, 4), dtype=complex)
        for K in kraus:
            tmp += K @ result @ K.conj().T
        result = tmp
    return result / np.trace(result)


def depolarize_2q(rho: np.ndarray, p: float) -> np.ndarray:
    """2-qubit depolarizing channel: (1-p)*rho + p*I/4."""
    return (1 - p) * rho + p * np.eye(4, dtype=complex) / 4


def readout_bias(probs: np.ndarray,
                 p10: float = 0.002, p01: float = 0.014) -> np.ndarray:
    """
    Apply readout error matrix to 4-outcome probability vector.
    M_ij = P(report i | true j)
    """
    # Per-qubit readout matrix
    M1 = np.array([[1 - p10, p01],
                   [p10,     1 - p01]])
    M = np.kron(M1, M1)  # 2-qubit readout
    return M @ probs


def shot_noise(probs: np.ndarray, N_shots: int, seed: int = 0) -> np.ndarray:
    """Multinomial sampling of probabilities."""
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(N_shots, np.abs(probs) / np.abs(probs).sum())
    return counts / N_shots


# ── PTM estimation under noise ────────────────────────────────────────────────
def noisy_rho(chi_t: float, N: int, params: dict) -> np.ndarray:
    """Apply IBM noise model to theoretical rho2."""
    rho = rho2_exact(chi_t, N, 0.0)
    # T1/T2 decay over gate time
    t = params["t_gate_us"] * params["n_2q_gates"]
    rho = t1t2_decay(rho, params["T1_us"], params["T2_us"], t)
    # Depolarizing per 2q gate (accumulated)
    for _ in range(params["n_2q_gates"]):
        rho = depolarize_2q(rho, params["p_depol_2q"])
    return rho


def estimate_T_from_shots(rho: np.ndarray, params: dict,
                           seed: int = 0) -> tuple:
    """
    Simulate Pauli tomography under shot noise and readout errors.
    Returns (T_estimated, T_stderr) — 3x3 matrices.
    """
    sx = np.array([[0,1],[1,0]], dtype=complex)
    sy = np.array([[0,-1j],[1j,0]], dtype=complex)
    sz = np.array([[1,0],[0,-1]], dtype=complex)
    paulis = [sx, sy, sz]
    T_est = np.zeros((3, 3))
    T_err = np.zeros((3, 3))
    rng_seed = seed

    for i, si in enumerate(paulis):
        for j, sj in enumerate(paulis):
            # Theoretical expectation value
            obs = np.kron(si, sj)
            eig_vals, eig_vecs = np.linalg.eigh(obs)

            # Rotate rho into eigenbasis
            rho_rot = eig_vecs.conj().T @ rho @ eig_vecs
            probs_true = np.real(np.diag(rho_rot)).clip(0)
            probs_true /= probs_true.sum()

            # Apply readout bias
            probs_biased = readout_bias(probs_true,
                                        params["p_readout_10"],
                                        params["p_readout_01"])
            probs_biased = probs_biased.clip(0)
            probs_biased /= probs_biased.sum()

            # Shot noise
            probs_shot = shot_noise(probs_biased, params["N_shots"], rng_seed)
            rng_seed += 1

            # Reconstruct expectation value
            T_ij = float(np.sum(eig_vals * probs_shot))
            T_est[i, j] = T_ij

            # Standard error from binomial (approximate)
            T_err[i, j] = 1.0 / np.sqrt(params["N_shots"])

    return T_est, T_err


# ── Stratum prediction for IBM hardware ───────────────────────────────────────
def predict_stratum(chi_t: float, N: int = 4,
                    params: dict = IBM_PARAMS,
                    n_trials: int = 20) -> dict:
    """
    Run n_trials noisy PTM estimates at (chi_t, N).
    Returns stratum distribution + E prediction with uncertainty.
    """
    rho_noisy = noisy_rho(chi_t, N, params)
    T_theory = compute_T_matrix(rho2_exact(chi_t, N, 0.0))
    T_noisy  = compute_T_matrix(rho_noisy)
    fm_theory = f_max_analytic(T_theory)
    fm_noisy  = f_max_analytic(T_noisy)

    strata_counts = {}
    F_estimates = []
    for trial in range(n_trials):
        T_est, T_err = estimate_T_from_shots(rho_noisy, params, seed=trial)
        svs = np.sort(np.linalg.svd(T_est, compute_uv=False))[::-1]
        fm_est = f_max_analytic(T_est)
        F_estimates.append((2 * fm_est + 1) / 3)

        # Build blind feature row for stratum inference
        row = _build_blind_row(T_est, svs, chi_t)
        st, _ = infer_strata(row[_MASK].reshape(1, -1))
        s = st[0]
        strata_counts[s] = strata_counts.get(s, 0) + 1

    F_arr = np.array(F_estimates)
    dominant = max(strata_counts, key=strata_counts.get)
    return {
        "chi_t": chi_t,
        "N": N,
        "F_theory": float((2 * fm_theory + 1) / 3),
        "F_noisy_mean": float((2 * fm_noisy + 1) / 3),
        "F_estimated_mean": float(F_arr.mean()),
        "F_estimated_std": float(F_arr.std()),
        "dominant_stratum": dominant,
        "stratum_counts": strata_counts,
        "n_trials": n_trials,
    }


def _build_blind_row(T, svs, chi_t):
    """Minimal blind feature row for stratum inference."""
    sp = float(max(0, min(1, 1 - np.linalg.norm(T, "fro") / 2.0)))
    rk = int((svs > 1e-3).sum())
    # 34-dim row (matches PTMFeatureVector layout)
    row = np.zeros(34)
    row[3:12] = T.flatten()
    row[12:15] = svs
    row[30] = rk
    row[33] = sp
    return row  # full 34-dim; caller slices with _MASK


# ── Validation against real IBM data ─────────────────────────────────────────
def validate_vs_real(results_path: str, params: dict = IBM_PARAMS):
    """Compare emulator predictions against real teleport_results.json."""
    with open(results_path) as f:
        data = json.load(f)

    print("\n=== IBM Emulator Validation vs Real Data ===")
    print(f"{'chi_t':>8} {'F_real':>8} {'F_emul':>8} {'sigma':>6} {'stratum':>12}")
    print("-" * 50)

    for entry in data.get("results", data if isinstance(data, list) else []):
        chi_t = float(entry.get("chi_t", entry.get("chi", 0)))
        F_real = float(entry.get("F_avg", entry.get("fidelity", 0)))
        pred = predict_stratum(chi_t, N=4, params=params, n_trials=10)
        delta = abs(F_real - pred["F_estimated_mean"])
        print(f"{chi_t:>8.4f} {F_real:>8.4f} {pred['F_estimated_mean']:>8.4f} "
              f"{pred['F_estimated_std']:>6.4f} {pred['dominant_stratum']:>12}"
              f"  delta={delta:.4f}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--validate", default=None,
                   help="Path to teleport_results.json for validation")
    p.add_argument("--chi-scan", action="store_true",
                   help="Scan chi_t from 0 to pi and predict strata")
    p.add_argument("--chi-t", type=float, default=1.456,
                   help="Single chi_t to predict (default: optimal)")
    p.add_argument("--N", type=int, default=4)
    p.add_argument("--shots", type=int, default=1024)
    p.add_argument("--out", default="ibm_emulator_results.json")
    args = p.parse_args()

    params = {**IBM_PARAMS, "N_shots": args.shots}

    if args.validate:
        validate_vs_real(args.validate, params)
        return

    if args.chi_scan:
        print("=== IBM Emulator: chi_t scan ===")
        print(f"{'chi_t':>8} {'F_theory':>9} {'F_noisy':>8} "
              f"{'F_emul':>8} {'stratum':>12}")
        print("-" * 55)
        results = []
        for chi_t in np.linspace(0.1, np.pi, 20):
            pred = predict_stratum(chi_t, args.N, params, n_trials=15)
            results.append(pred)
            print(f"{chi_t:>8.4f} {pred['F_theory']:>9.4f} "
                  f"{pred['F_noisy_mean']:>8.4f} "
                  f"{pred['F_estimated_mean']:>8.4f} "
                  f"{pred['dominant_stratum']:>12}")
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved -> {args.out}")
        return

    # Single point prediction
    print(f"=== IBM Emulator: chi_t={args.chi_t:.4f}  N={args.N} ===")
    pred = predict_stratum(args.chi_t, args.N, params, n_trials=50)
    print(f"  F_theory    = {pred['F_theory']:.4f}")
    print(f"  F_noisy     = {pred['F_noisy_mean']:.4f}")
    print(f"  F_estimated = {pred['F_estimated_mean']:.4f} "
          f"+/- {pred['F_estimated_std']:.4f}")
    print(f"  Dominant stratum: {pred['dominant_stratum']}")
    print(f"  Stratum distribution: {pred['stratum_counts']}")
    print(f"\n  Prediction: IBM hardware at chi_t={args.chi_t:.4f} should")
    print(f"  land in [{pred['dominant_stratum']}] stratum with "
          f"F_avg = {pred['F_estimated_mean']:.3f} +/- {pred['F_estimated_std']:.3f}")

    # Key diagnostic points
    print("\n=== Key diagnostic predictions ===")
    key_points = [
        ("chi_t*  (optimal)",   1.456),
        ("pi/2   (mid-range)",  np.pi/2),
        ("chi_t=pi (singular)", np.pi - 0.01),
    ]
    for label, ct in key_points:
        pred = predict_stratum(ct, args.N, params, n_trials=20)
        print(f"  {label:<25} F={pred['F_estimated_mean']:.4f}"
              f"+/-{pred['F_estimated_std']:.4f}  "
              f"stratum={pred['dominant_stratum']}")

    with open(args.out, "w") as f:
        json.dump(pred, f, indent=2)


if __name__ == "__main__":
    main()
