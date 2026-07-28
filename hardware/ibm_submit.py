import os
"""
ibm_submit.py — IBM PTM 3-point job submission
Calibration job first, PTM job second.
Records job IDs to ibm_job_ids.txt for pre-registration.
"""
import numpy as np
import datetime, json
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

TOKEN = os.environ["QISKIT_IBM_TOKEN"]  # set in your shell; never hardcode

# ── Connect ───────────────────────────────────────────────────────────────
print("Connecting to IBM Quantum...")
try:
    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=TOKEN)
except Exception:
    service = QiskitRuntimeService(token=TOKEN)

# ── Pick best backend ─────────────────────────────────────────────────────
backends = service.backends(operational=True, simulator=False, min_num_qubits=4)
best, best_q = None, 9999
for b in backends:
    try:
        q = b.status().pending_jobs
        print(f"  {b.name:30s}  pending={q}")
        if q < best_q:
            best, best_q = b, q
    except Exception as e:
        print(f"  {b.name}: {e}")
print(f"\nSelected backend: {best.name}  (pending={best_q})")

# ── Circuit builder ───────────────────────────────────────────────────────
def oat_ptm_circuit(chi_t, n=4):
    """H = chi_t * JzL * JzR = chi_t/4 * (ZZ_02+ZZ_03+ZZ_12+ZZ_13)"""
    qc = QuantumCircuit(n, 2)
    qc.h(range(n))
    theta = chi_t / 4
    for i, j in [(0,2),(0,3),(1,2),(1,3)]:
        qc.cx(i,j); qc.rz(2*theta, j); qc.cx(i,j)
    qc.h(1); qc.h(2)
    qc.measure([1,2],[0,1])
    return qc

def cal_circuit(state, qubit, n=4):
    """Readout calibration: prepare |0> or |1> on a single qubit, measure."""
    qc = QuantumCircuit(n, 1)
    if state == 1:
        qc.x(qubit)
    qc.measure(qubit, 0)
    return qc

CHI_TS = [0.01*np.pi, 0.355*np.pi, np.pi]
LABELS = ['chi_t~0', 'chi_t_star', 'chi_t_pi']
SHOTS  = 500

# ── Transpile circuits ────────────────────────────────────────────────────
print("\nTranspiling circuits...")
ptm_circuits = []
for chi_t in CHI_TS:
    qc = oat_ptm_circuit(chi_t)
    ptm_circuits.append(transpile(qc, backend=best, optimization_level=2))

# Calibration: |0> and |1> on each of qubits 1 and 2
cal_circuits = []
for q in [1, 2]:
    for s in [0, 1]:
        cal_circuits.append(transpile(cal_circuit(s, q), backend=best, optimization_level=1))

print(f"  PTM circuit depth (chi_t*): {ptm_circuits[1].depth()}")
print(f"  Calibration circuits: {len(cal_circuits)}")

# ── Write pre-registration file ───────────────────────────────────────────
timestamp = datetime.datetime.utcnow().isoformat() + "Z"
prereg = {
    "experiment":      "IBM PTM 3-point, N=4 OAT boundary channel",
    "hamiltonian":     "H = chi_t * JzL * JzR (cross-half ZZ only)",
    "observable":      "T_xx = <X1 X2>/2",
    "backend":         best.name,
    "submission_time": timestamp,
    "shots":           SHOTS,
    "n_trotter":       1,
    "predictions": {
        "T_xx_product": {"value": 0.500, "sigma": 0.028, "label": "chi_t~0, calibration anchor"},
        "T_xx_quantum": {"value": 0.360, "sigma": 0.028, "label": "chi_t*=0.355pi, quantum phase"},
        "T_xx_null":    {"value": 0.000, "sigma": 0.028, "label": "chi_t=pi, exact singular null"},
        "invariant_ratio": {"value": 0.720, "sigma": 0.060, "label": "T_xx(*)/T_xx(0), convention-independent"},
        "passage_criterion": "T_xx(chi_t*) > T_xx(pi) + 2*sigma_combined = 0.281"
    },
    "convention_note": "T_xx_circuit = cos^{N-2}(chi_t/2)/2 (PTM normalized). Theorem value = cos^{N-2}(chi_t/2) (unnormalized). Factor-of-2 from T_ij=Tr[si*E(sj/2)].",
    "failure_modes": {
        "a": "All T_xx degraded uniformly -> systematic noise; apply calibration; NOT failure",
        "b": "T_xx(pi) < 2-sigma nonzero -> shot noise; NOT failure",
        "c": "T_xx(chi_t*) <= T_xx(0) -> circuit angle error; check theta=chi_t/4",
        "d": "T_xx(pi) > 2-sigma calibrated -> genuine theorem failure"
    },
    "job_ids": {"calibration": None, "ptm": None}
}
with open("ibm_prereg.json", "w") as f:
    json.dump(prereg, f, indent=2)
print(f"\nPre-registration filed: ibm_prereg.json  ({timestamp})")

# ── Job 1: Calibration ────────────────────────────────────────────────────
print("\n--- JOB 1: READOUT CALIBRATION ---")
sampler = Sampler(mode=best)
cal_job = sampler.run(cal_circuits, shots=SHOTS)
cal_job_id = cal_job.job_id()
print(f"Calibration job ID: {cal_job_id}")
prereg["job_ids"]["calibration"] = cal_job_id
with open("ibm_prereg.json","w") as f: json.dump(prereg, f, indent=2)
with open("ibm_job_ids.txt","w") as f:
    f.write(f"Calibration job: {cal_job_id}\n")
    f.write(f"Backend: {best.name}\n")
    f.write(f"Submitted: {timestamp}\n")

# ── Wait for calibration ──────────────────────────────────────────────────
print("Waiting for calibration job to complete...")
cal_result = cal_job.result()
print(f"Calibration complete.")

# Parse calibration matrix (2 qubits, 2 states each)
def p1_from_pub(pub_result):
    counts = pub_result.data.c.get_counts()
    total = sum(counts.values())
    return counts.get('1', 0) / total

cal_results_raw = cal_result
# For each qubit: P(measure 1 | prepare 0), P(measure 1 | prepare 1)
ro_err = {}
for idx, (qubit, state) in enumerate([(1,0),(1,1),(2,0),(2,1)]):
    pub = cal_result[idx]
    counts = pub.data.c.get_counts()
    total = sum(counts.values())
    p1 = counts.get('1', 0)/total
    ro_err[f"q{qubit}_prep{state}"] = p1
    print(f"  q{qubit} prep|{state}>: P(1)={p1:.4f}")

# ── Job 2: PTM 3-point ────────────────────────────────────────────────────
print("\n--- JOB 2: PTM 3-POINT ---")
ptm_job = sampler.run(ptm_circuits, shots=SHOTS)
ptm_job_id = ptm_job.job_id()
print(f"PTM job ID: {ptm_job_id}")
prereg["job_ids"]["ptm"] = ptm_job_id
with open("ibm_prereg.json","w") as f: json.dump(prereg, f, indent=2)
with open("ibm_job_ids.txt","a") as f:
    f.write(f"PTM job:         {ptm_job_id}\n")

# ── Wait and extract T_xx ─────────────────────────────────────────────────
print("Waiting for PTM job to complete...")
ptm_result = ptm_job.result()
print("PTM complete.")

def txx_from_pub(pub_result):
    counts = pub_result.data.c.get_counts()
    total = sum(counts.values())
    s = 0.0
    for b, c in counts.items():
        b0, b1 = int(b[-1]), int(b[-2])
        s += (1-2*b0)*(1-2*b1)*c/total
    return s / 2

print("\n=== RESULTS ===")
print(f"{'Point':>14}  {'T_xx_raw':>10}  {'Prediction':>12}")
print("-"*40)
txx_raw = {}
for i, label in enumerate(LABELS):
    t = txx_from_pub(ptm_result[i])
    txx_raw[label] = t
    pred = [0.500, 0.360, 0.000][i]
    print(f"  {label:>12}  {t:10.5f}  {pred:12.3f}")

# Self-calibration
anchor_analytic = 0.500
anchor_noisy = txx_raw['chi_t~0']
cal_factor = anchor_analytic / anchor_noisy if abs(anchor_noisy) > 0.01 else 1.0
print(f"\nCalibration factor: {cal_factor:.4f}")
print(f"\n{'Point':>14}  {'T_xx_cal':>10}  {'Prediction':>12}  {'sigma':>8}  {'pass?':>6}")
print("-"*54)
txx_cal = {}
for label, pred, sig in zip(LABELS, [0.500,0.360,0.000], [0.028,0.028,0.028]):
    tc = txx_raw[label]*cal_factor
    txx_cal[label] = tc
    ok = abs(tc-pred) <= 2*sig
    print(f"  {label:>12}  {tc:10.5f}  {pred:12.3f}  {sig:8.3f}  {'OK' if ok else 'FAIL':>6}")

sep = abs(txx_cal['chi_t_star'] - txx_cal['chi_t_pi'])
sig_combined = 0.028 * np.sqrt(2)
n_sigma = sep / sig_combined
print(f"\nSeparation: {sep:.4f}  ({n_sigma:.1f} sigma)")
print(f"Passage (>0.281): {'PASSED' if txx_cal['chi_t_star'] > 0.281 else 'FAILED'}")

# ── Save results ──────────────────────────────────────────────────────────
results = {
    "backend": best.name,
    "submission_time": timestamp,
    "completion_time": datetime.datetime.utcnow().isoformat()+"Z",
    "calibration_job_id": cal_job_id,
    "ptm_job_id": ptm_job_id,
    "readout_errors": ro_err,
    "calibration_factor": cal_factor,
    "T_xx_raw": txx_raw,
    "T_xx_calibrated": txx_cal,
    "separation_sigma": n_sigma,
    "passed": bool(txx_cal['chi_t_star'] > 0.281)
}
with open("ibm_results.json","w") as f: json.dump(results, f, indent=2)
print(f"\nResults saved: ibm_results.json")
print(f"Job IDs saved: ibm_job_ids.txt")
