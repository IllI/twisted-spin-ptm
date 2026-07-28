import os
"""Archive raw IBM job results before 90-day expiry."""
import json, datetime
from qiskit_ibm_runtime import QiskitRuntimeService

TOKEN = os.environ["QISKIT_IBM_TOKEN"]  # set in your shell; never hardcode

try:
    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=TOKEN)
except Exception:
    service = QiskitRuntimeService(token=TOKEN)

JOBS = {
    "d81qrbegbeec73akuheg": "calibration",
    "d81qrcvtjchs73bn8rqg": "ptm_3point"
}

for jid, label in JOBS.items():
    print(f"Fetching {label} [{jid}]...")
    job = service.job(jid)
    result = job.result()

    # Archive metadata
    meta = {
        "job_id":        jid,
        "label":         label,
        "backend":       job.backend().name,
        "status":        str(job.status()),
        "metrics":       job.metrics(),
        "creation_date": str(job.creation_date),
        "archived_at":   datetime.datetime.utcnow().isoformat() + "Z",
    }
    with open(f"ibm_archive_{label}_meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    # Archive raw counts from every pub
    counts_all = []
    for i, pub in enumerate(result):
        try:
            c = pub.data.c.get_counts()
            counts_all.append({"pub_index": i, "counts": c})
        except Exception as e:
            counts_all.append({"pub_index": i, "error": str(e)})

    with open(f"ibm_archive_{label}_counts.json", "w") as f:
        json.dump(counts_all, f, indent=2)

    print(f"  Saved: ibm_archive_{label}_meta.json")
    print(f"  Saved: ibm_archive_{label}_counts.json")
    for entry in counts_all:
        print(f"    pub {entry['pub_index']}: {entry.get('counts', entry.get('error'))}")

print("\nArchive complete. Commit these files immediately.")
