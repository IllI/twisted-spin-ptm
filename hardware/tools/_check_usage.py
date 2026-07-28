import os
"""Check IBM job details and QPU time used — no QPU time consumed by this script."""
import json
from qiskit_ibm_runtime import QiskitRuntimeService

TOKEN = os.environ["QISKIT_IBM_TOKEN"]  # set in your shell; never hardcode

try:
    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=TOKEN)
except Exception:
    service = QiskitRuntimeService(token=TOKEN)

job_ids = ["d81qrbegbeec73akuheg", "d81qrcvtjchs73bn8rqg"]
labels  = ["Calibration", "PTM 3-point"]

total_s = 0.0
print("IBM Job Usage Report")
print("="*50)
for jid, label in zip(job_ids, labels):
    try:
        job = service.job(jid)
        metrics = job.metrics()
        usage = metrics.get("usage", {})
        seconds = usage.get("seconds", None)
        status = job.status()
        print(f"\n{label}  [{jid}]")
        print(f"  Status:      {status}")
        print(f"  QPU seconds: {seconds}")
        if seconds:
            total_s += float(seconds)
    except Exception as e:
        print(f"  Error: {e}")

print(f"\n{'='*50}")
print(f"Total QPU time used this session: {total_s:.1f} s  ({total_s/60:.2f} min)")
print(f"Monthly budget (open plan):       10 min = 600 s")
print(f"Estimated remaining:              {max(0, 600-total_s):.0f} s  ({max(0, 600-total_s)/60:.1f} min)")
print(f"\nNo active sessions or instances. Connection closes when this script exits.")
