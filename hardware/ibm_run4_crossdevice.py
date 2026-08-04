#!/usr/bin/env python3
"""ibm_run4_crossdevice.py -- Cross-device replication of the OAT PTM sweep.

Paper 1 (twisted-spin-ptm) established the PTM identity
    T_xx(chi_t) = A * cos^(N-2)(chi_t/2) / 2
on three runs, all on ibm_marrakesh. This is the missing device-independence
control: the same pre-registered 9-point sweep + null, on a *different*
156-qubit Heron r2 chip, with a design that fixes the bug found during the
Page-Wootters work (initial_layout was computed but never passed to
transpile -- confirmed on 36 jobs to silently default to the trivial layout).

Also generalizes to N qubits (default 4, matching prior runs; --n-qubits 6
adds a genuinely new data point: N=4 tested one point on the predicted curve
family T_xx = cos^(N-2)(chi_t/2)/2, N=6 tests the N-dependence itself).

General cross-half OAT circuit for N qubits (N even):
    left  = 0 .. N/2-1
    right = N/2 .. N-1
    H_OAT = (chi_t / (n_L*n_R)) * sum_{i in left, j in right} Z_i Z_j
    boundary pair = (N/2 - 1, N/2)          -- innermost cross-half pair
    T_xx = <X_boundary_L X_boundary_R> / 2 = cos^(N-2)(chi_t/2) / 2   (Theorem 3)

Circuit: H on all N qubits; for each cross pair (i,j): CNOT(i,j), RZ(2*theta)_j,
CNOT(i,j), theta = chi_t / (n_L*n_R); H on the two boundary qubits; measure.
N=4 reduces exactly to the paper's circuit (4 pairs, theta = chi_t/4).

Usage:
    # Smoke test against Aer first, zero hardware shots:
    python ibm_run4_crossdevice.py --dry --backend ibm_fez

    # Real hardware (token via env var, never hardcoded):
    export QISKIT_IBM_TOKEN=...
    python ibm_run4_crossdevice.py --backend ibm_fez
    python ibm_run4_crossdevice.py --backend ibm_fez --n-qubits 6
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import time
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, transpile

CHI_TS = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8]) * np.pi / 8
SHOTS_SWEEP = 500
SHOTS_NULL = 4000
SHOTS_CAL = 500

# --- Resilience: checkpoint job IDs to disk so a dropped connection (laptop
# sleep, wifi blip, DNS hiccup) doesn't strand an already-submitted job with
# no way to resume it short of re-running the whole multi-stage pipeline from
# scratch. Learned the hard way: an overnight run died on a DNS resolution
# failure with a job still sitting, live, in IBM's queue -- costing nothing,
# but with no way for the script to find it again. ---

CHECKPOINT_DIR = Path(".run4_checkpoints")

TRANSIENT_MARKERS = (
    "NameResolutionError", "getaddrinfo", "ConnectionError", "MaxRetryError",
    "Connection aborted", "RemoteDisconnected", "ConnectionResetError",
    "Timeout", "TimeoutError",
)


def checkpoint_path(backend_name: str, n: int) -> Path:
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    return CHECKPOINT_DIR / f"{backend_name}_n{n}.json"


def load_checkpoint(backend_name: str, n: int) -> dict:
    path = checkpoint_path(backend_name, n)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_checkpoint(backend_name: str, n: int, stage: str, job_id: str) -> None:
    path = checkpoint_path(backend_name, n)
    data = load_checkpoint(backend_name, n)
    data[stage] = job_id
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def clear_checkpoint(backend_name: str, n: int) -> None:
    path = checkpoint_path(backend_name, n)
    if path.exists():
        path.unlink()


def wait_with_retry(job, max_wait_hours: float = 8.0):
    """Poll job.result(), surviving transient network errors by retrying
    with backoff instead of crashing. Does NOT retry genuine terminal states
    (job cancelled/failed on IBM's side) -- those are real information and
    should surface immediately, not be silently retried forever."""
    deadline = time.time() + max_wait_hours * 3600
    backoff = 10.0
    while True:
        try:
            return job.result()
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            transient = any(marker in message for marker in TRANSIENT_MARKERS)
            if not transient or time.time() > deadline:
                raise
            print(
                f"    [network hiccup: {type(exc).__name__}] retrying in {backoff:.0f}s "
                f"(job is still safely queued on IBM's side)...",
                flush=True,
            )
            time.sleep(backoff)
            backoff = min(backoff * 1.5, 300.0)


def cross_pairs(n: int) -> list[tuple[int, int]]:
    left, right = range(n // 2), range(n // 2, n)
    return [(i, j) for i in left for j in right]


def boundary_qubits(n: int) -> tuple[int, int]:
    return n // 2 - 1, n // 2


def oat_circuit(chi_t: float, n: int) -> QuantumCircuit:
    pairs = cross_pairs(n)
    theta = chi_t / len(pairs)
    bl, br = boundary_qubits(n)
    qc = QuantumCircuit(n, 2)
    qc.h(range(n))
    for i, j in pairs:
        qc.cx(i, j)
        qc.rz(2 * theta, j)
        qc.cx(i, j)
    qc.h(bl)
    qc.h(br)
    qc.measure([bl, br], [0, 1])
    return qc


def cal_circuit(qubit_idx: int, state: int, n: int) -> QuantumCircuit:
    qc = QuantumCircuit(n, 1)
    if state == 1:
        qc.x(qubit_idx)
    qc.measure(qubit_idx, 0)
    return qc


def find_chain(coupling_map, num_qubits: int, length: int) -> list[int]:
    adj: dict[int, set[int]] = {q: set() for q in range(num_qubits)}
    for a, b in coupling_map.get_edges():
        adj[a].add(b)
        adj[b].add(a)

    def extend(path: list[int]) -> list[int] | None:
        if len(path) == length:
            return path
        for nxt in sorted(adj[path[-1]]):
            if nxt in path:
                continue
            found = extend(path + [nxt])
            if found:
                return found
        return None

    for start in range(num_qubits):
        found = extend([start])
        if found:
            return found
    raise RuntimeError(f"no contiguous chain of length {length} found")


class DryBackend:
    """Aer stand-in with a heavy-hex-ish coupling map and calibrated noise,
    for a zero-shot smoke test of the full pipeline before spending QPU."""

    def __init__(self) -> None:
        from qiskit_aer import AerSimulator
        from qiskit_aer.noise import NoiseModel, ReadoutError, thermal_relaxation_error
        from qiskit.transpiler import CouplingMap

        t1, t2 = 150e3, 80e3
        nm = NoiseModel()
        e1 = thermal_relaxation_error(t1, t2, 50.0)
        e2 = thermal_relaxation_error(t1, t2, 300.0).expand(thermal_relaxation_error(t1, t2, 300.0))
        nm.add_all_qubit_quantum_error(e1, ["rz", "sx", "x", "h"])
        nm.add_all_qubit_quantum_error(e2, ["cx", "cz", "ecr", "swap"])
        nm.add_all_qubit_readout_error(ReadoutError([[0.98, 0.02], [0.02, 0.98]]))
        self.num_qubits = 16
        self.name = "aer_dry"
        self.coupling_map = CouplingMap.from_line(self.num_qubits)
        self._sim = AerSimulator(noise_model=nm)

    def run(self, circuits: list[QuantumCircuit], shots: int, layout: list[int], stage: str | None = None):
        tqc = [
            transpile(c, self._sim, optimization_level=0, initial_layout=layout)
            for c in circuits
        ]
        result = self._sim.run(tqc, shots=shots).result()
        return [result.get_counts(i) for i in range(len(circuits))], tqc, None


class HardwareBackend:
    def __init__(self, backend_name: str, n: int) -> None:
        from qiskit_ibm_runtime import QiskitRuntimeService

        token = os.environ.get("QISKIT_IBM_TOKEN")
        if not token:
            raise SystemExit(
                "QISKIT_IBM_TOKEN is not set. Set it in this shell before running:\n"
                '  PowerShell:  $env:QISKIT_IBM_TOKEN = "<token>"'
            )
        try:
            self._service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
        except Exception:
            self._service = QiskitRuntimeService(token=token)
        self._backend = self._service.backend(backend_name)
        self.num_qubits = self._backend.num_qubits
        self.name = self._backend.name
        self.coupling_map = self._backend.coupling_map
        self._checkpoint_key = (self.name, n)

    def run(self, circuits: list[QuantumCircuit], shots: int, layout: list[int], stage: str | None = None):
        from qiskit_ibm_runtime import SamplerV2 as Sampler
        from qiskit_ibm_runtime.exceptions import RuntimeInvalidStateError

        tqc = [
            transpile(c, backend=self._backend, optimization_level=0, initial_layout=layout)
            for c in circuits
        ]

        def submit_fresh():
            sampler = Sampler(mode=self._backend)
            new_job = sampler.run([(c,) for c in tqc], shots=shots)
            print(f"    job {new_job.job_id()} submitted; waiting...", flush=True)
            if stage is not None:
                save_checkpoint(*self._checkpoint_key, stage, new_job.job_id())
            return new_job

        existing_job_id = load_checkpoint(*self._checkpoint_key).get(stage) if stage else None
        if existing_job_id:
            print(f"    resuming previously submitted job {existing_job_id} (stage '{stage}')...", flush=True)
            job = self._service.job(existing_job_id)
        else:
            job = submit_fresh()

        try:
            result = wait_with_retry(job)
        except RuntimeInvalidStateError:
            if not existing_job_id:
                raise
            print(f"    checkpointed job {existing_job_id} is dead (cancelled/failed) -- resubmitting fresh...", flush=True)
            job = submit_fresh()
            result = wait_with_retry(job)

        out = []
        for i in range(len(circuits)):
            creg = tqc[i].cregs[0].name
            out.append(getattr(result[i].data, creg).get_counts())
        return out, tqc, job.job_id()


def verify_layout(transpiled: QuantumCircuit, intended: list[int]) -> dict:
    """Post-hoc check: which physical qubits did the transpiled circuit
    actually touch, versus what we intended. Written from day one this time,
    not bolted on after a bug was found.

    The check is "touched is a subset of intended", not equality: a
    calibration circuit only operates one qubit at a time and legitimately
    leaves the rest of the layout idle (no gates emitted for idle qubits),
    so requiring the full layout to appear would flag a false positive.
    What actually matters -- did the transpiler honor initial_layout instead
    of silently reverting to the trivial layout -- is fully captured by the
    subset check."""
    touched = set()
    for instr in transpiled.data:
        for q in instr.qubits:
            touched.add(transpiled.find_bit(q).index)
    return {
        "intended_layout": intended,
        "actual_touched_qubits": sorted(touched),
        "matches_intended": touched.issubset(set(intended)),
        "depth": transpiled.depth(),
        "two_qubit_ops": int(sum(v for k, v in transpiled.count_ops().items() if k in ("cx", "cz", "ecr", "swap"))),
    }


def txx_raw(counts: dict[str, int], shots: int) -> float:
    s = 0.0
    for b, c in counts.items():
        b = b.replace(" ", "")
        b0, b1 = int(b[-1]), int(b[-2])
        s += (1 - 2 * b0) * (1 - 2 * b1) * c / shots
    return s / 2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry", action="store_true", help="run against Aer, zero hardware shots")
    parser.add_argument("--backend", required=True, help="e.g. ibm_fez, ibm_kingston")
    parser.add_argument("--n-qubits", type=int, default=4, help="4 (paper-matching) or 6 (new scaling point)")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--fresh", action="store_true",
        help="ignore any checkpointed job IDs from a previous interrupted run and resubmit from scratch",
    )
    args = parser.parse_args()
    if args.n_qubits % 2 or args.n_qubits < 4:
        raise SystemExit("--n-qubits must be an even integer >= 4")

    n = args.n_qubits
    pairs = cross_pairs(n)
    bl, br = boundary_qubits(n)
    out_dir = args.out_dir or Path(f"results_run4_{args.backend}_n{n}")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.fresh and not args.dry:
        clear_checkpoint(args.backend, n)

    backend = DryBackend() if args.dry else HardwareBackend(args.backend, n)
    layout = find_chain(backend.coupling_map, backend.num_qubits, n)
    print(f"ibm_run4_crossdevice  backend={backend.name}  N={n}  dry={args.dry}", flush=True)
    print(f"  cross pairs: {pairs}  (theta = chi_t / {len(pairs)})", flush=True)
    print(f"  boundary qubits (logical): {bl}, {br}", flush=True)
    print(f"  physical layout: {layout}", flush=True)
    if not args.dry:
        print(
            f"  checkpoint file: {checkpoint_path(backend.name, n)} "
            f"(delete or pass --fresh to force a clean resubmit)",
            flush=True,
        )

    timestamp = datetime.datetime.utcnow().isoformat() + "Z"
    prereg = {
        "run": "4-crossdevice",
        "rationale": "Device-independence replication of the OAT PTM sweep on a second Heron r2 chip"
        + (f"; N={n} scaling point beyond the paper's N=4" if n != 4 else ""),
        "backend": backend.name,
        "n_qubits": n,
        "cross_pairs": pairs,
        "boundary_qubits_logical": [bl, br],
        "physical_layout": layout,
        "optimization_level": 0,
        "chi_ts_pi_units": (CHI_TS / np.pi).tolist(),
        "shots_sweep": SHOTS_SWEEP,
        "shots_null": SHOTS_NULL,
        "submission_time": timestamp,
        "predictions": {
            "functional_form": f"T_xx = A * cos^{n-2}(chi_t/2) / 2",
            "null": "T_xx(pi) compatible with 0",
            "R2": "R^2 > 0.90" if n == 4 else "R^2 > 0.80 (deeper circuit, weaker prior)",
            "device_independence": "Same functional form and null as ibm_marrakesh runs 1-3, on a different chip",
        },
    }
    prereg_path = out_dir / "run4_prereg.json"
    prereg_path.write_text(json.dumps(prereg, indent=2), encoding="utf-8")
    print(f"  pre-registration filed: {prereg_path}", flush=True)

    # -- Calibration --
    cal_specs = [(q, s) for q in (bl, br) for s in (0, 1)]
    cal_circuits = [cal_circuit(q, s, n) for q, s in cal_specs]
    print("\n--- JOB: CALIBRATION ---", flush=True)
    cal_counts, cal_tqc, cal_jid = backend.run(cal_circuits, SHOTS_CAL, layout, stage="calibration")
    calib = {}
    for (q, s), counts in zip(cal_specs, cal_counts):
        total = sum(counts.values())
        p1 = counts.get("1", 0) / total if total else float("nan")
        calib[f"q{q}_prep{s}"] = p1
        print(f"  q{q} prep|{s}>: P(1)={p1:.4f}", flush=True)
    m_bl = np.array([[1 - calib[f"q{bl}_prep0"], 1 - calib[f"q{bl}_prep1"]],
                      [calib[f"q{bl}_prep0"], calib[f"q{bl}_prep1"]]])
    m_br = np.array([[1 - calib[f"q{br}_prep0"], 1 - calib[f"q{br}_prep1"]],
                      [calib[f"q{br}_prep0"], calib[f"q{br}_prep1"]]])
    m_inv = np.linalg.inv(np.kron(m_bl, m_br))

    layout_check_cal = verify_layout(cal_tqc[0], layout)

    # -- Sweep + null --
    circuits = [oat_circuit(c, n) for c in CHI_TS]
    print(f"\n--- JOB: SWEEP ({len(circuits) - 1} pts) ---", flush=True)
    sweep_counts, sweep_tqc, sweep_jid = backend.run(circuits[:-1], SHOTS_SWEEP, layout, stage="sweep")
    print("\n--- JOB: NULL (chi_t=pi) ---", flush=True)
    null_counts_list, null_tqc, null_jid = backend.run([circuits[-1]], SHOTS_NULL, layout, stage="null")
    null_counts = null_counts_list[0]

    layout_check_sweep = verify_layout(sweep_tqc[len(sweep_tqc) // 2], layout)
    layout_check_null = verify_layout(null_tqc[0], layout)

    def txx_mit(counts: dict[str, int], shots: int) -> float:
        order = ["00", "01", "10", "11"]
        p = np.array([counts.get(b, 0) / shots for b in order])
        pm = m_inv @ p
        pm = np.clip(pm, 0, None)
        pm /= pm.sum()
        return ((pm[0] + pm[3]) - (pm[1] + pm[2])) / 2

    print(f"\n{'=' * 66}\nSWEEP RESULTS (N={n}, backend={backend.name})\n{'=' * 66}", flush=True)
    print(f"  {'chi_t/pi':>9}  {'T_raw':>8}  {'T_mit':>8}  {'analytic':>10}", flush=True)
    results = []
    for i, chi_t in enumerate(CHI_TS[:-1]):
        counts = sweep_counts[i]
        tr = txx_raw(counts, SHOTS_SWEEP)
        tm = txx_mit(counts, SHOTS_SWEEP)
        an = np.cos(chi_t / 2) ** (n - 2) / 2
        results.append({"chi_t_pi": float(chi_t / np.pi), "T_raw": tr, "T_mit": tm, "analytic": an, "counts": counts})
        print(f"  {chi_t / np.pi:9.4f}  {tr:8.4f}  {tm:8.4f}  {an:10.4f}", flush=True)

    tr_pi = txx_raw(null_counts, SHOTS_NULL)
    tm_pi = txx_mit(null_counts, SHOTS_NULL)
    sig_pi = 1 / (2 * np.sqrt(SHOTS_NULL))
    results.append({"chi_t_pi": 1.0, "T_raw": tr_pi, "T_mit": tm_pi, "analytic": 0.0, "counts": null_counts})
    print(f"  {'1.0000':>9}  {tr_pi:8.4f}  {tm_pi:8.4f}  {'0.0000':>10}  <- NULL ({tm_pi / sig_pi:.1f}sigma)", flush=True)

    from scipy.optimize import curve_fit
    t_meas = np.array([r["T_mit"] for r in results])
    a_pred = np.array([np.cos(r["chi_t_pi"] * np.pi / 2) ** (n - 2) / 2 for r in results])
    mask = a_pred > 0.001
    if mask.sum() >= 2:
        a_fit, = curve_fit(lambda x, a: a * x, a_pred[mask], t_meas[mask])[0]
        resid = t_meas[mask] - a_fit * a_pred[mask]
        r2 = 1 - np.var(resid) / max(np.var(t_meas[mask]), 1e-12)
    else:
        a_fit, r2 = float("nan"), float("nan")

    print(f"\nFit: A={a_fit:.4f}  R^2={r2:.4f}", flush=True)
    print(f"Null: T_xx(pi)_mit = {tm_pi:.4f} ({tm_pi / sig_pi:.2f} sigma from 0)", flush=True)

    all_layout_ok = all(c["matches_intended"] for c in (layout_check_cal, layout_check_sweep, layout_check_null))
    print(f"\nLayout verification: cal={layout_check_cal['matches_intended']} "
          f"sweep={layout_check_sweep['matches_intended']} null={layout_check_null['matches_intended']} "
          f"-> all_ok={all_layout_ok}", flush=True)

    payload = {
        "run": "4-crossdevice",
        "backend": backend.name,
        "dry_run": bool(args.dry),
        "n_qubits": n,
        "physical_layout": layout,
        "job_ids": {"calibration": cal_jid, "sweep": sweep_jid, "null": null_jid},
        "A_fit": float(a_fit),
        "R2": float(r2),
        "null_T_mit": float(tm_pi),
        "null_sigma": float(sig_pi),
        "null_sigma_from_zero": float(tm_pi / sig_pi),
        "results": results,
        "M_bl": m_bl.tolist(),
        "M_br": m_br.tolist(),
        "layout_verification": {
            "calibration": layout_check_cal,
            "sweep_midpoint": layout_check_sweep,
            "null": layout_check_null,
            "all_ok": all_layout_ok,
        },
    }
    results_path = out_dir / "run4_results.json"
    results_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    if not args.dry:
        clear_checkpoint(args.backend, n)
    print(f"\n[DONE] results: {results_path}", flush=True)


if __name__ == "__main__":
    main()
