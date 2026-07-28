# Hardware Recovery of the Analytic PTM Structure of One-Axis-Twisting Boundary Channels

Companion repository for Paper 1: theory, exact simulation, pre-registration, and
three runs on IBM Quantum hardware (`ibm_marrakesh`) measuring the predicted
Pauli-transfer-matrix element `T_xx = A·cos^(N-2)(χt/2)` of the OAT boundary-pair
channel, including its exact null at `χt = π`.

**Claim boundary.** This work is a hardware measurement of an analytically predicted
two-body observable. It is **not** a demonstration of quantum teleportation, a
teleportation-advantage claim, or a hardware-confirmed theorem. The full CAN/CANNOT
claim list is locked in `docs/EXPERIMENT_ARCHIVE_OAT_IBM.md` §135 references.

---

## The chain of evidence

| Stage | Artifact | Result |
|---|---|---|
| 1. Analytic theory | `theory/ptm_analytic.py` | `T_zz = 0`, `T_yy = 0`, `T_xx = cos^(N-2)(χt/2)` proved from the closed-form boundary density matrix |
| 2. Exact simulation | `simulation/jila_oat_exact_tpu.py` + v1–v4 | Closed-form ρ₂ validated vs brute force; MPS automaton scales to N≥64; Lindblad dephasing threshold mapped |
| 3. Null controls | `simulation/chsh_bell_test.py` | No CHSH violation for N≥4 (S=1.71) despite F>2/3 — advantage without Bell nonlocality |
| 4. Noise forecast | `simulation/p2_hardware_noise.py` | Predicted hardware attenuation envelope before submission |
| 5. Pre-registration | `results/hardware/ibm_prereg.json` (+ run2/run3) | Numerical predictions with σ and explicit pass criterion, written before job submission |
| 6. Hardware Run 1 | `results/hardware/ibm_results.json` | 3-point PTM: signed ordering at **12.5σ** |
| 7. Hardware Run 2 | `ibm_run2_fit.json`, `ibm_run2_bootstrap.json` | 9-point sweep, layout A `[0,1,2,3]`: **R²=0.986**, A=0.909, null at **+0.43σ** (4000 shots) |
| 8. Hardware Run 3 | `ibm_run3_results.json` | Disjoint layout B `[4,5,6,7]`: **R²=0.976**, A=0.903, null at **−0.12σ** — layout-independent |

Raw shot counts, calibration matrices, and IBM job IDs for every run are archived
under `results/hardware/ibm_archive_*` — the entire analysis is reproducible from
counts upward.

## Run-by-run story (what each iteration fixed)

- **v1 → v2:** Werner-state proxy replaced by the actual OAT boundary state via
  Schmidt rotation.
- **v2 → v3:** exact contraction replaced by an MPS automaton exploiting the
  diagonality of `U_OAT` — bond dimension n+1, no Trotter error, N=64+ reachable.
- **v3 → v4:** analytic Lindblad dephasing added; decoherence threshold `Γ*(N)`
  measured. Dephasing rate anchored to the published Sr-87 single-particle value
  (arXiv:2505.06444).
- **Strategy pivot (pre-hardware):** scope cut to the single most analytically
  anchored observable (3-point PTM). `V_Q`, Hessian stiffness, and full teleportation
  fidelity were explicitly excluded as unmeasurable within a 10-min/month budget.
- **Dry run:** `hardware/exp_ibm_trotterized_oat.py` on `qiskit_aer` with calibrated
  noise — root-caused an Rz-angle transpilation defect; hardware runs use
  `optimization_level=0` as a result.
- **Run 1 → Run 2:** 3 points → 9 points + mirror circuits + 4000-shot null;
  bootstrap CIs added.
- **Run 2 → Run 3:** full replication on a disjoint qubit chain, answering the
  transpiler/routing-artifact objection. Depth decreased (76→64) while attenuation
  stayed within CI — noise is qubit-dominated, not depth-dominated.

## Repository layout

```
theory/            analytic PTM theorems (closed-form ρ₂ → T_xx prediction)
simulation/        exact OAT core, teleport v1–v4, channel tomography, CHSH, noise forecast
hardware/          submission scripts (token via QISKIT_IBM_TOKEN env var) + dry-run harness
hardware/tools/    archival, usage, and transpiler-diagnostic utilities
results/simulation own-machine validation results (JSON)
results/hardware/  prereg, job IDs, fits, bootstrap, raw counts + metadata per run
figures/           fidelity, concurrence, and scaling figures
docs/              paper draft + full experiment archive/audit
```

## Reproducing

```bash
pip install -r requirements.txt
python simulation/jila_oat_exact_tpu.py        # exact core, CPU is fine
python theory/ptm_analytic.py                  # analytic identities
# Hardware resubmission (requires an IBM Quantum account):
export QISKIT_IBM_TOKEN=...                    # never hardcode
python hardware/ibm_run3.py
```

Note on filenames: several simulation scripts carry `_tpu` in their names; the
archived result JSONs record the actual device used per run (some runs executed on
CPU). Device provenance should always be taken from the `devices` field of the
result JSON, not the filename.

## Provenance

- Backend: `ibm_marrakesh` (156-qubit Heron), IBM Open Plan.
- All job IDs in `results/hardware/*jobids*` and `ibm_job_ids.txt`; server-side
  timestamps in the result/meta JSONs are authoritative for chronology.
- Compute for simulation stages supported by Google's TPU Research Cloud (TRC).
- The views expressed are those of the author and do not reflect the official
  policy or position of IBM or the IBM Quantum team.
