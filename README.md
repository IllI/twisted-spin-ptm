# Hardware Recovery of the Analytic PTM Structure of One-Axis-Twisting Boundary Channels

Companion repository for Paper 1: theory, exact simulation, pre-registration, and
three runs on IBM Quantum hardware (`ibm_marrakesh`) measuring the predicted
Pauli-transfer-matrix element `T_xx = A·cos^(N-2)(χt/2)` of the OAT boundary-pair
channel, including its exact null at `χt = π`.

**Claim boundary.** This work is a hardware measurement of an analytically predicted
two-body observable. It is **not** a demonstration of quantum teleportation, a
teleportation-advantage claim, or a hardware-confirmed theorem. The full CAN/CANNOT
claim list is locked in `docs/EXPERIMENT_ARCHIVE_OAT_IBM.md` §135 references.

**How we got here.** `docs/DISCOVERY_NARRATIVE.md` is a readable account of the
research path — falsified formulas, a wrong circuit caught before it cost real
hardware time, a retracted "scrambling" claim, and the internal null (`χt = π`)
that anchored everything else. It's a condensed, public-facing companion to the
3,000+ line internal research log; start there if you want the story before the
tables.

**The simulation phase that made this possible was funded by Google's TPU
Research Cloud** (`v6e`, one month from 2026-04-21, extended a second month on
the strength of the results). That is not an acknowledgement formality: the
tensor-network scale-up to `N ≥ 64`, the dephasing-threshold sweeps, the
bootstrap/Haar/seed-independence rigor passes, and the entire
recovery-geometry search that told us *which* observable was worth a
10-minute-per-month hardware budget all ran on TRC. **The hardware phase
measured one number. The accelerator phase decided which number.**

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

## What the failures bought

Nearly every numeric claim in this repository was wrong at least once before it
was right, and **the wrong version is usually what said where to look next.**
Recorded compactly here; the full account is in `docs/DISCOVERY_NARRATIVE.md`.

| failure | what it bought |
|---|---|
| Headline fidelity formula **falsified** after being the paper's opening line | forced the move to the closed-form boundary density matrix — the analytic spine everything now rests on |
| Concurrence scaling exponent corrected **three times** | the discipline of re-deriving rather than re-fitting |
| The "12.5σ" Run-1 result was measuring **a different angle than intended** | caught in review → made Run 2 rigorous enough to publish (9 points, mirror circuits, bootstrap CIs) |
| A "scrambling" claim, **retracted** | scope narrowed to what the analytics actually support |
| Critical-exponent detour: `β ≈ 0.87` → **revised to `β = 1.00`** | the fit window had mixed two physical regimes; walking it back is why the remaining claims hold |
| Wrong circuit found in the **dry run** (Rz-angle transpilation defect) | `optimization_level=0` on all hardware runs — caught before it cost hardware time |
| **N=6 dry run fails decisively**: R² = −2.0, null at 21σ | a real scaling bound established at **zero shot cost** — the N-scaling test needs a shallower construction, not more shots |
| Cross-device Run 4: three attempts **abandoned** in queue | descoped by decision, script kept; the claim never rested on it |
| Entanglement **not identifiable** from PTM anisotropy alone | resolved by `V_Q` over Haar-random settings — the same wall the sister program hit, and the origin of a shared standard (below) |

**The internal null did more work than any positive result.** At `χt = π` the
theory says six independent observables must vanish exactly. They do — across
dozens of runs, every seed, every optimizer restart, every SU(2)
parameterization. That is what licensed trusting a single hardware run: if a
null is that reliable in simulation, hardware reproducing the *shape* around it
is meaningful even under device noise.

## The shared method

This repository and its sister,
[relational-time-ibm-quantum](https://github.com/IllI/relational-time-ibm-quantum),
were run as two independent programs and hit **the same epistemic wall from
opposite directions**:

> **A single measurement configuration does not certify quantum structure;
> measurement diversity does.**

Here it appeared as *entanglement is not identifiable from PTM anisotropy
alone*, resolved by `V_Q` over Haar-random settings. There it appeared as
*no single local product-basis distribution can certify clock–system
entanglement*, resolved by a multi-setting fidelity witness. Neither program
anticipated the other's version. That two independent lines converged on it is
the strongest methodological claim either makes, and the sister repository
documents the resulting protocol — pre-registration, statevector assertions
before backend contact, self-built adversaries, per-run provenance — in a form
meant to be reused. It is deliberately cheap: both programs together ran on a
free-tier quantum account and a TPU grant.

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

## Prepared but deliberately not run

`hardware/ibm_run4_crossdevice.py` is a complete, dry-run-verified cross-device
replication of the Run 3 protocol on a second Heron r2 chip, generalized to
arbitrary boundary-pair N. It fixes a layout bug found during a sister-branch
hardware run (`initial_layout` is now actually passed to the transpiler and
verified post-hoc from the job's own compiled circuit, not assumed), and it
checkpoints job IDs so a dropped connection can't strand a submitted job.

**It was not executed, by decision rather than by omission.** Three attempts
were queued on `ibm_fez` and `ibm_kingston` and abandoned after multi-hour
waits (free-tier Open Plan queueing; no quota was consumed — pending jobs are
not billed). On review, the run was descoped: it would have added one
sentence — "also recovered on a second device" — to a claim that does not
rest on it. The paper's claim is that a specific analytic observable was
recovered with R²=0.986 and a null at 0.4σ, already established by three runs
with pre-registration, archived raw counts, and a disjoint-layout replication.
Cross-device consistency for this hardware family is separately evidenced in
the sister repository, where the Page–Wootters protocol ran on both
`ibm_marrakesh` and `ibm_fez`.

The script is kept because it is correct and cheap to run if a future session
has a quiet queue — not because a gap in the evidence depends on it.

**The N=6 dry run is a result in its own right.** Extending the protocol to
six qubits (nine cross-half ZZ pairs, 18 CX gates) fails decisively under the
calibrated noise model: R² = −2.0, with the χt=π null landing at 21σ instead
of ~0. The circuit is too deep for this gate decomposition to survive
decoherence. That bound was established at zero shot cost, and it says the
N-scaling test needs a fundamentally shallower construction — not more shots.

## Provenance

- Backend: `ibm_marrakesh` (156-qubit Heron), IBM Open Plan.
- All job IDs in `results/hardware/*jobids*` and `ibm_job_ids.txt`; server-side
  timestamps in the result/meta JSONs are authoritative for chronology.
- **Compute for the simulation phase (exact boundary-state core, MPS automaton,
  dephasing sweeps, the full PTM/recovery-geometry search documented in
  `docs/DISCOVERY_NARRATIVE.md`) was supported by Google's TPU Research Cloud
  (TRC)** — `v6e` instances, an initial one-month grant from 2026-04-21
  extended a further month on the strength of the results. Without it the
  tensor-network scale-up to `N ≥ 64` and the statistical-rigor passes (Gate 2
  bootstrap CIs, Haar convergence, seed-independence) would not have been
  feasible, and the `N = 6` scaling bound — a genuine negative result — could
  not have been established at zero hardware cost. The IBM hardware stage used
  no TRC resources. The same grant window funded the sister program's synthetic
  phase; see its README for the accelerator-as-hypothesis-generator argument in
  full.
- The views expressed are those of the author and do not reflect the official
  policy or position of IBM, the IBM Quantum team, or Google.
