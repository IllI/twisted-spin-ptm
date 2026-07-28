import os
"""
Diagnose the effective chi_t on hardware by extracting transpiled Rz angles
from the PTM circuit for chi_t* = 0.355*pi.

The transpiler modified depth 12 -> 43, changing effective chi_t.
We compute what chi_t the hardware actually saw from the Rz angles in the
transpiled circuit, then predict the T_xx we should have gotten.
"""
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService

TOKEN = os.environ["QISKIT_IBM_TOKEN"]  # set in your shell; never hardcode
CHI_T_STAR = 0.355 * np.pi
N = 4

def oat_ptm_circuit(chi_t, n=4):
    """Cross-half ZZ: H = chi_t * JzL * JzR, theta=chi_t/4 per pair"""
    qc = QuantumCircuit(n, 2)
    qc.h(range(n))
    theta = chi_t / 4
    for i, j in [(0,2),(0,3),(1,2),(1,3)]:
        qc.cx(i,j); qc.rz(2*theta, j); qc.cx(i,j)
    qc.h(1); qc.h(2)
    qc.measure([1,2],[0,1])
    return qc

try:
    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=TOKEN)
except Exception:
    service = QiskitRuntimeService(token=TOKEN)

backends = service.backends(name="ibm_marrakesh")
backend = backends[0]

# Transpile at opt=2 (same as job submission)
qc_ideal = oat_ptm_circuit(CHI_T_STAR)
qc_trans  = transpile(qc_ideal, backend=backend, optimization_level=2)

print(f"Original depth:    {qc_ideal.depth()}")
print(f"Transpiled depth:  {qc_trans.depth()}")
print(f"Original CX count: {qc_ideal.count_ops().get('cx',0)}")
print(f"Transpiled ops:    {dict(qc_trans.count_ops())}")

# Extract all Rz angles from the transpiled circuit
rz_angles = []
ecr_count = 0
for inst in qc_trans.data:
    name = inst.operation.name
    if name == 'rz':
        rz_angles.append(float(inst.operation.params[0]))
    elif name in ('ecr','cx','cz'):
        ecr_count += 1

print(f"\nRz angles found: {len(rz_angles)}")
print(f"2Q gate count:   {ecr_count}")

# The intended Rz angles from ZZ pairs: 2*theta = chi_t/2
intended_rz = CHI_T_STAR / 2
print(f"\nIntended Rz angle (2*theta=chi_t/2): {intended_rz:.6f} rad = {intended_rz/np.pi:.4f}*pi")

# Filter for angles that are close to the intended ones (the ZZ-pair Rzs)
# In the transpiled circuit these may be rotated/merged
relevant = [a for a in rz_angles if abs(abs(a) - intended_rz) < 0.5]
print(f"Rz angles near intended value: {[f'{a:.4f}' for a in relevant]}")

# Estimate effective chi_t from the median ZZ-like Rz angle
if relevant:
    median_rz = np.median(np.abs(relevant))
    # 2*theta = effective_rz; theta = effective_rz/2; chi_t_eff = 4*theta = 2*effective_rz
    chi_t_eff = 2 * median_rz
    T_xx_eff = np.cos(chi_t_eff/2)**(N-2) / 2
    print(f"\nEffective chi_t from median Rz: {chi_t_eff:.4f} rad = {chi_t_eff/np.pi:.4f}*pi")
    print(f"Predicted T_xx at effective chi_t: {T_xx_eff:.4f}")
    print(f"Hardware result (calibrated):       0.4335")
    print(f"Match: {'YES' if abs(T_xx_eff - 0.4335) < 0.05 else 'NO -- deeper inspection needed'}")

# Also show all Rz angles for manual inspection
print(f"\nAll Rz angles (rad):")
for i, a in enumerate(rz_angles):
    print(f"  [{i:2d}] {a:.6f}  ({a/np.pi:.4f}*pi)")

# What chi_t would give T_xx = 0.434?
T_measured = 0.434
from scipy.optimize import brentq
chi_t_back = brentq(lambda x: np.cos(x/2)**(N-2)/2 - T_measured, 0.001, np.pi-0.001)
print(f"\nBack-computed chi_t giving T_xx=0.434: {chi_t_back:.4f} rad = {chi_t_back/np.pi:.4f}*pi")
print(f"Intended chi_t*: {CHI_T_STAR:.4f} rad = {CHI_T_STAR/np.pi:.4f}*pi")
print(f"Ratio (intended/effective): {CHI_T_STAR/chi_t_back:.4f}")
