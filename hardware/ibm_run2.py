import os
"""
ibm_run2_fixed.py — Fix SamplerV2 tuple API + separate null job
"""
import numpy as np, json, datetime
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

TOKEN = os.environ["QISKIT_IBM_TOKEN"]  # set in your shell; never hardcode
N     = 4

# Load calibration from run2 cal job (already done)
CAL_JOB_ID = "d81thqfoha1c73bkrpug"

# Fresh calibration values from run2
M_q1 = np.array([[1.0000, 0.0060], [0.0000, 0.9940]])
M_q2 = np.array([[0.9980, 0.0080], [0.0020, 0.9920]])
M_2q = np.kron(M_q1, M_q2)
M_inv = np.linalg.inv(M_2q)

try:
    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=TOKEN)
except Exception:
    service = QiskitRuntimeService(token=TOKEN)
backend = service.backend("ibm_marrakesh")
chain   = [0,1,2,3]
print(f"Backend: {backend.name}, chain: {chain}")

def oat_circuit(chi_t, n=4):
    qc = QuantumCircuit(n, 2)
    qc.h(range(n))
    theta = chi_t / 4
    for i,j in [(0,2),(0,3),(1,2),(1,3)]:
        qc.cx(i,j); qc.rz(2*theta, j); qc.cx(i,j)
    qc.h(1); qc.h(2)
    qc.measure([1,2],[0,1])
    return qc

CHI_TS = np.array([0,1,2,3,4,5,6,7,8]) * np.pi/8
transpile_kwargs = dict(backend=backend, optimization_level=0, initial_layout=chain)
circuits = [transpile(oat_circuit(c), **transpile_kwargs) for c in CHI_TS]

# Mirror controls: chi_t -> -chi_t (symmetric check)
mirror_chi_ts = [-CHI_TS[4], -CHI_TS[6]]
mirror_circuits = [transpile(oat_circuit(c), **transpile_kwargs) for c in mirror_chi_ts]

print(f"Sweep circuit depths: {[c.depth() for c in circuits]}")
print(f"Mirror depths: {[c.depth() for c in mirror_circuits]}")

# Rz angle check for chi_t*=pi/4 (index 2)
print("\nRz angles at chi_t=pi/4 (intended theta=pi/16=0.1963):")
for inst in circuits[2].data:
    if inst.operation.name == 'rz':
        v = float(inst.operation.params[0])
        print(f"  {v:.5f} rad = {v/np.pi:.4f}*pi")

# Back-compute effective chi_t from a mid-range Rz angle
# All circuits use the same skeleton; only the ZZ Rz angles vary
# Find which Rz angles change between chi_t=0 and chi_t=pi/4
angles_0   = [float(i.operation.params[0]) for i in circuits[0].data if i.operation.name=='rz']
angles_pi4 = [float(i.operation.params[0]) for i in circuits[2].data if i.operation.name=='rz']
diffs = [(a-b, i) for i,(a,b) in enumerate(zip(angles_pi4, angles_0)) if abs(a-b)>0.001]
print(f"\nRz angles that change with chi_t (diff from chi_t=0):")
for diff, idx in diffs:
    print(f"  index {idx}: delta={diff:.5f} rad={diff/np.pi:.4f}*pi (intended={np.pi/16/np.pi:.4f}*pi)")

# ── Sampler: separate jobs for sweep (500 shots) and null (4000 shots) ────
sampler = Sampler(mode=backend)

timestamp = datetime.datetime.utcnow().isoformat()+"Z"
print(f"\n--- JOB A: SWEEP (8 points + mirrors, 500 shots each) ---")
# SamplerV2 correct API: pass list of (circuit, parameter_values, shots) tuples
# For no parameters: (circuit, [], shots)
job_a_circuits = circuits[:8] + mirror_circuits   # exclude pi point
pubs_a = [(c, [], 500) for c in job_a_circuits]
job_a = sampler.run(pubs_a)
jid_a = job_a.job_id()
print(f"Job A ID: {jid_a}")

print(f"\n--- JOB B: NULL (chi_t=pi, 4000 shots) ---")
pubs_b = [(circuits[8], [], 4000)]
job_b = sampler.run(pubs_b)
jid_b = job_b.job_id()
print(f"Job B ID: {jid_b}")

# Save job IDs immediately
ids = {"cal": CAL_JOB_ID, "sweep_mirrors": jid_a, "null_heavy": jid_b,
       "timestamp": timestamp, "chain": chain}
with open("ibm_run2_jobids.json","w") as f: json.dump(ids, f, indent=2)
print(f"\nJob IDs saved to ibm_run2_jobids.json")

# ── Wait and extract ───────────────────────────────────────────────────────
print("\nWaiting for Job A (sweep)...")
res_a = job_a.result()
print("Waiting for Job B (null)...")
res_b = job_b.result()
print("Both complete.")

def txx_raw(counts, shots):
    s = 0.0
    for b,c in counts.items():
        b=b.replace(' ',''); b0,b1=int(b[-1]),int(b[-2])
        s += (1-2*b0)*(1-2*b1)*c/shots
    return s/2

def txx_mit(counts, shots):
    order=['00','01','10','11']
    p = np.array([counts.get(b,0)/shots for b in order])
    pm = M_inv@p; pm=np.clip(pm,0,None); pm/=pm.sum()
    return ((pm[0]+pm[3])-(pm[1]+pm[2]))/2

# Collect sweep results (pubs 0-7 = chi_t 0..7pi/8, pubs 8-9 = mirrors)
results = []
print(f"\n{'='*72}")
print("ANGLE SWEEP RESULTS (readout-matrix mitigated)")
print(f"{'='*72}")
print(f"{'chi_t/pi':>9}  {'shots':>5}  {'T_raw':>8}  {'T_mitigated':>12}  {'analytic':>10}")
print("-"*72)

for i, chi_t in enumerate(CHI_TS[:8]):
    counts = res_a[i].data.c.get_counts()
    tr = txx_raw(counts, 500)
    tm = txx_mit(counts, 500)
    an = np.cos(chi_t/2)**(N-2)/2
    results.append({"chi_t_pi": chi_t/np.pi, "shots":500,
                    "T_raw":tr, "T_mit":tm, "analytic":an, "counts":counts})
    print(f"  {chi_t/np.pi:9.4f}  {500:5d}  {tr:8.4f}  {tm:12.4f}  {an:10.4f}")

# Null (Job B, 4000 shots)
counts_pi = res_b[0].data.c.get_counts()
tr_pi = txx_raw(counts_pi, 4000)
tm_pi = txx_mit(counts_pi, 4000)
sig_pi = 1/(2*np.sqrt(4000))
results.append({"chi_t_pi":1.0,"shots":4000,
                "T_raw":tr_pi,"T_mit":tm_pi,"analytic":0.0,"counts":counts_pi})
print(f"  {'1.0000':>9}  {4000:5d}  {tr_pi:8.4f}  {tm_pi:12.4f}  {'0.0000':>10}  <- NULL")

# Mirrors
print(f"\nMirrored controls (symmetry check):")
for j, chi_t in enumerate(mirror_chi_ts):
    counts_m = res_a[8+j].data.c.get_counts()
    tr_m = txx_raw(counts_m, 500)
    tm_m = txx_mit(counts_m, 500)
    idx_pos = int(round(abs(chi_t/np.pi)*8))
    tr_pos  = results[idx_pos]["T_raw"] if idx_pos < len(results) else float('nan')
    tm_pos  = results[idx_pos]["T_mit"] if idx_pos < len(results) else float('nan')
    diff = abs(tr_m - tr_pos)
    print(f"  chi_t={chi_t/np.pi:.3f}pi: T_raw={tr_m:.4f} T_mit={tm_m:.4f}  "
          f"|vs +chi_t={tr_pos:.4f}|  diff={diff:.4f}  {'SYM-OK' if diff<0.05 else 'CHECK'}")

# Key statistics
print(f"\n{'='*72}")
print("KEY STATISTICS")
print(f"{'='*72}")
peak_idx = int(np.argmax([r["T_mit"] for r in results[:8]]))
peak_chi = results[peak_idx]["chi_t_pi"]
peak_val = results[peak_idx]["T_mit"]
anchor   = results[0]["T_mit"]
null_val = tm_pi

print(f"Anchor T_xx(0):        {anchor:.5f}")
print(f"Peak   T_xx({peak_chi:.3f}pi): {peak_val:.5f}")
print(f"Null   T_xx(pi):       {null_val:.5f}  +/- {sig_pi:.5f}  "
      f"({null_val/sig_pi:.1f} sigma from zero)")

sep = peak_val - null_val
sig_comb = np.sqrt((1/(2*np.sqrt(500)))**2 + sig_pi**2)
nsig = sep/sig_comb
print(f"\nSeparation peak-null:  {sep:.5f}  ({nsig:.1f} sigma)")
print(f"Signed ordering OK:    {anchor > peak_val > 0 > null_val}")

# Save
with open("ibm_run2_results.json","w") as f:
    json.dump({"run":2,"jobs":ids,"chain":chain,
               "mitigation":{"M_q1":M_q1.tolist(),"M_q2":M_q2.tolist()},
               "results":results,
               "null":{"T_raw":tr_pi,"T_mit":tm_pi,"shots":4000,"sigma":sig_pi},
               "mirrors":[{"chi_t_pi":c/np.pi,"T_raw":txx_raw(res_a[8+j].data.c.get_counts(),500),
                           "T_mit":txx_mit(res_a[8+j].data.c.get_counts(),500)}
                          for j,c in enumerate(mirror_chi_ts)]},
              f, indent=2, default=str)
print(f"\nResults saved: ibm_run2_results.json")
