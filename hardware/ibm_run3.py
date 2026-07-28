import os
"""
ibm_run3.py — Topology-Controlled Replication
Same 9-point protocol, disjoint physical qubit layout on ibm_marrakesh.
Tests: layout robustness, backend independence, observable invariance.
"""
import numpy as np, json, datetime
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

TOKEN     = os.environ["QISKIT_IBM_TOKEN"]  # set in your shell; never hardcode
N         = 4
CHAIN_A   = [0, 1, 2, 3]   # Run 2 layout

try:
    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=TOKEN)
except Exception:
    service = QiskitRuntimeService(token=TOKEN)
backend = service.backend("ibm_marrakesh")
cmap    = backend.coupling_map
print(f"Backend: {backend.name}  ({backend.num_qubits} qubits)")

# ── Find disjoint 4-qubit linear chain ───────────────────────────────────
exclude = set(CHAIN_A)
chain_b = None
for q0 in range(backend.num_qubits):
    if q0 in exclude: continue
    for q1 in range(backend.num_qubits):
        if q1 in exclude or q1==q0: continue
        if not cmap.graph.has_edge(q0,q1): continue
        for q2 in range(backend.num_qubits):
            if q2 in exclude or q2 in (q0,q1): continue
            if not cmap.graph.has_edge(q1,q2): continue
            for q3 in range(backend.num_qubits):
                if q3 in exclude or q3 in (q0,q1,q2): continue
                if not cmap.graph.has_edge(q2,q3): continue
                chain_b = [q0,q1,q2,q3]
                break
            if chain_b: break
        if chain_b: break
    if chain_b: break

if chain_b is None:
    # Relax: allow overlap with chain_a but not same qubits
    for q0 in range(10, backend.num_qubits):
        for q1 in range(backend.num_qubits):
            if q1==q0: continue
            if not cmap.graph.has_edge(q0,q1): continue
            for q2 in range(backend.num_qubits):
                if q2 in (q0,q1): continue
                if not cmap.graph.has_edge(q1,q2): continue
                for q3 in range(backend.num_qubits):
                    if q3 in (q0,q1,q2): continue
                    if not cmap.graph.has_edge(q2,q3): continue
                    if set([q0,q1,q2,q3]).isdisjoint(set(CHAIN_A)):
                        chain_b = [q0,q1,q2,q3]; break
                if chain_b: break
            if chain_b: break
        if chain_b: break

print(f"Layout A (Run 2): {CHAIN_A}")
print(f"Layout B (Run 3): {chain_b}")
print(f"Disjoint: {set(chain_b).isdisjoint(set(CHAIN_A))}")

# ── Circuit ───────────────────────────────────────────────────────────────
def oat_circuit(chi_t, n=4):
    qc = QuantumCircuit(n, 2)
    qc.h(range(n))
    theta = chi_t / 4
    for i,j in [(0,2),(0,3),(1,2),(1,3)]:
        qc.cx(i,j); qc.rz(2*theta, j); qc.cx(i,j)
    qc.h(1); qc.h(2)
    qc.measure([1,2],[0,1])
    return qc

def cal_circuit(qubit_idx, state, n=4):
    qc = QuantumCircuit(n, 1)
    if state==1: qc.x(qubit_idx)
    qc.measure(qubit_idx, 0)
    return qc

CHI_TS = np.array([0,1,2,3,4,5,6,7,8]) * np.pi/8
tk = dict(backend=backend, optimization_level=0, initial_layout=chain_b)
circuits   = [transpile(oat_circuit(c), **tk) for c in CHI_TS]
cal_circs  = [transpile(cal_circuit(q,s), **tk)
              for q in [1,2] for s in [0,1]]

print(f"\nDepth (Layout B, chi_t*): {circuits[4].depth()}")
print(f"Depth (Layout A, chi_t*): 76  (Run 2)")

# Verify Rz angle at chi_t=pi/4
angles0   = [float(i.operation.params[0]) for i in circuits[0].data if i.operation.name=='rz']
angles_pi4= [float(i.operation.params[0]) for i in circuits[2].data if i.operation.name=='rz']
diffs = [(a-b) for a,b in zip(angles_pi4, angles0) if abs(a-b)>0.001]
print(f"Rz deltas at chi_t=pi/4 (Layout B): {[f'{d/np.pi:.4f}pi' for d in diffs]}")
print(f"  Expected: 0.1250*pi per ZZ pair (x4)")

# ── Pre-registration ──────────────────────────────────────────────────────
timestamp = datetime.datetime.utcnow().isoformat()+"Z"
prereg3 = {
    "run": 3,
    "rationale": "Topology-controlled replication: disjoint layout, same protocol",
    "layout_A": CHAIN_A, "layout_B": chain_b,
    "backend": backend.name,
    "optimization_level": 0,
    "chi_ts_pi_units": (CHI_TS/np.pi).tolist(),
    "shots_sweep": 500, "shots_null": 4000,
    "submission_time": timestamp,
    "predictions": {
        "functional_form": "T_xx = A_B * cos^2(chi_t/2)/2  with different A_B from A_A=0.909",
        "null":            "T_xx(pi) compatible with 0",
        "R2":              "R^2 > 0.95",
        "layout_independence": "Same phase structure, same null, different A"
    }
}
with open("ibm_run3_prereg.json","w") as f: json.dump(prereg3, f, indent=2)
print(f"\nPre-registration filed: ibm_run3_prereg.json ({timestamp})")

# ── Calibration job ───────────────────────────────────────────────────────
print("\n--- JOB 1: CALIBRATION (Layout B) ---")
sampler = Sampler(mode=backend)
cal_job = sampler.run([(c,[],500) for c in cal_circs])
cal_id  = cal_job.job_id()
print(f"Cal job ID: {cal_id}")
prereg3["cal_job_id"] = cal_id
with open("ibm_run3_prereg.json","w") as f: json.dump(prereg3, f, indent=2)

print("Waiting for calibration...")
cal_res = cal_job.result()

calib = {}
for idx,(lq,s) in enumerate([(1,0),(1,1),(2,0),(2,1)]):
    counts = cal_res[idx].data.c.get_counts()
    total  = sum(counts.values())
    p1 = counts.get('1',0)/total
    calib[f"q{lq}_prep{s}"] = p1
    print(f"  q{lq} prep|{s}>: P(1)={p1:.4f}")

M_q1 = np.array([[1-calib["q1_prep0"], 1-calib["q1_prep1"]],
                  [  calib["q1_prep0"],   calib["q1_prep1"]]])
M_q2 = np.array([[1-calib["q2_prep0"], 1-calib["q2_prep1"]],
                  [  calib["q2_prep0"],   calib["q2_prep1"]]])
M_inv = np.linalg.inv(np.kron(M_q1, M_q2))

# ── Sweep job ─────────────────────────────────────────────────────────────
print("\n--- JOB 2: SWEEP (Layout B, 8 pts) ---")
job_sweep = sampler.run([(c,[],500) for c in circuits[:8]])
jid_sweep = job_sweep.job_id()
print(f"Sweep job ID: {jid_sweep}")

print("\n--- JOB 3: NULL (Layout B, 4000 shots) ---")
job_null = sampler.run([(circuits[8],[],4000)])
jid_null = job_null.job_id()
print(f"Null job ID: {jid_null}")

prereg3.update({"sweep_job_id": jid_sweep, "null_job_id": jid_null})
with open("ibm_run3_prereg.json","w") as f: json.dump(prereg3, f, indent=2)
with open("ibm_run3_jobids.json","w") as f:
    json.dump({"cal":cal_id,"sweep":jid_sweep,"null":jid_null,"chain_b":chain_b},f,indent=2)

print("\nWaiting for sweep + null...")
res_sweep = job_sweep.result()
res_null  = job_null.result()
print("Both complete.")

# ── Extract ───────────────────────────────────────────────────────────────
def txx_raw(counts, shots):
    s=0.0
    for b,c in counts.items():
        b=b.replace(' ',''); b0,b1=int(b[-1]),int(b[-2])
        s+=(1-2*b0)*(1-2*b1)*c/shots
    return s/2

def txx_mit(counts, shots):
    order=['00','01','10','11']
    p=np.array([counts.get(b,0)/shots for b in order])
    pm=M_inv@p; pm=np.clip(pm,0,None); pm/=pm.sum()
    return ((pm[0]+pm[3])-(pm[1]+pm[2]))/2

print(f"\n{'='*70}")
print("LAYOUT B — ANGLE SWEEP (readout-matrix mitigated)")
print(f"{'='*70}")
print(f"  {'chi_t/pi':>9}  {'T_raw':>8}  {'T_mit':>8}  {'analytic':>10}")
print("  "+"-"*42)

results_b = []
for i, chi_t in enumerate(CHI_TS[:8]):
    counts = res_sweep[i].data.c.get_counts()
    tr = txx_raw(counts, 500)
    tm = txx_mit(counts, 500)
    an = np.cos(chi_t/2)**2/2
    results_b.append({"chi_t_pi":chi_t/np.pi,"T_raw":tr,"T_mit":tm,"analytic":an,"counts":counts})
    print(f"  {chi_t/np.pi:9.4f}  {tr:8.4f}  {tm:8.4f}  {an:10.4f}")

counts_pi = res_null[0].data.c.get_counts()
tr_pi = txx_raw(counts_pi,4000); tm_pi = txx_mit(counts_pi,4000)
sig_pi = 1/(2*np.sqrt(4000))
results_b.append({"chi_t_pi":1.0,"T_raw":tr_pi,"T_mit":tm_pi,"analytic":0.0,"counts":counts_pi})
print(f"  {'1.0000':>9}  {tr_pi:8.4f}  {tm_pi:8.4f}  {'0.0000':>10}  <- NULL ({tm_pi/sig_pi:.1f}sig)")

# ── Fit Layout B ──────────────────────────────────────────────────────────
from scipy.optimize import curve_fit
T_b  = np.array([r["T_mit"] for r in results_b])
A_b  = np.array([np.cos(r["chi_t_pi"]*np.pi/2)**2/2 for r in results_b])
mask = A_b > 0.001
A_fit_b, = curve_fit(lambda x,a: a*x, A_b[mask], T_b[mask])[0]
resid_b  = T_b[mask] - A_fit_b*A_b[mask]
r2_b     = 1 - np.var(resid_b)/np.var(T_b[mask])

print(f"\nLayout B fit:  A={A_fit_b:.4f}  R^2={r2_b:.4f}")
print(f"Layout A fit:  A=0.9089  R^2=0.9856  (Run 2)")

print(f"\n{'='*70}")
print("COMPARISON TABLE  (Layout A vs Layout B)")
print(f"{'='*70}")
print(f"  {'Metric':30s}  {'Layout A':>12}  {'Layout B':>12}")
print("  "+"-"*56)
print(f"  {'Physical chain':30s}  {str(CHAIN_A):>12}  {str(chain_b):>12}")
print(f"  {'Attenuation A':30s}  {'0.909':>12}  {A_fit_b:12.4f}")
print(f"  {'R^2':30s}  {'0.9856':>12}  {r2_b:12.4f}")
print(f"  {'T_xx(pi) mitigated':30s}  {'0.0034':>12}  {tm_pi:12.5f}")
print(f"  {'Null (sigma from 0)':30s}  {'0.4':>12}  {tm_pi/sig_pi:12.2f}")
print(f"  {'Signed ordering holds':30s}  {'Yes':>12}  {'Yes' if results_b[0]['T_mit']>results_b[4]['T_mit']>0>tm_pi else 'Check':>12}")

# ── Save ──────────────────────────────────────────────────────────────────
with open("ibm_run3_results.json","w") as f:
    json.dump({"run":3,"layout_B":chain_b,"layout_A":CHAIN_A,
               "cal_job":cal_id,"sweep_job":jid_sweep,"null_job":jid_null,
               "A_fit":A_fit_b,"R2":r2_b,
               "null_T_mit":tm_pi,"null_sigma":sig_pi,
               "results":results_b,
               "M_q1":M_q1.tolist(),"M_q2":M_q2.tolist()},
              f,indent=2,default=str)
print(f"\nResults saved: ibm_run3_results.json")
