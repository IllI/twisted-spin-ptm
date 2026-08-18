# Hardware Recovery of the Analytic PTM Structure of One-Axis-Twisting Boundary Channels

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21878783.svg)](https://doi.org/10.5281/zenodo.21878783)

*The DOI above is the concept DOI — it always resolves to the latest archived
version. Cite this one, not a version-pinned link, so the citation stays
current if the repository is ever re-released.*

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

**Research supported with Cloud TPUs from Google's TPU Research Cloud (TRC).**
The TRC allocation supported the simulation and hypothesis-development phase;
the IBM Quantum hardware experiments used separate IBM Open Plan access. What
that compute bought is not incidental — the tensor-network scale-up to
`N ≥ 64`, the dephasing-threshold sweeps, the bootstrap/Haar/seed-independence
passes, and the recovery-geometry search that identified *which* observable was
worth a 10-minute-per-month hardware budget all ran on it. **The hardware phase
measured one number; the simulation phase decided which number.** Allocation
details are in *Computational resources* below.

---

## The chain of evidence

| Stage | Artifact | Result |
|---|---|---|
| 1. Analytic theory | `theory/ptm_analytic.py` | `T_zz = 0`, `T_yy = 0`, `T_xx = cos^(N-2)(χt/2)` proved from the closed-form boundary density matrix |
| 2. Exact simulation | `simulation/jila_oat_exact_tpu.py` + v1–v4 | Closed-form ρ₂ validated vs brute force; MPS automaton scales to N≥64; Lindblad dephasing threshold mapped |
| 3. Null controls | `simulation/chsh_bell_test.py` | No CHSH violation for N≥4 (S=1.71) despite the simulated teleportation fidelity `F` exceeding the classical 2/3 benchmark — i.e. the fidelity benchmark is passed without Bell nonlocality |
| 4. Noise forecast | `simulation/p2_hardware_noise.py` | Predicted hardware attenuation envelope before submission |
| 5. Pre-registration | `results/hardware/ibm_prereg.json` (+ run2/run3) | Numerical predictions with σ and explicit pass criterion, written before job submission |
| 6. Hardware Run 1 *(pilot)* | `results/hardware/ibm_results.json` | 3-point PTM, signed ordering at 12.5σ — **diagnostic, not confirmatory**: review showed it measured a different angle than intended (see *What the failures bought*). Superseded by Runs 2–3 |
| 7. Hardware Run 2 | `ibm_run2_fit.json`, `ibm_run2_bootstrap.json` | 9-point sweep, layout A `[0,1,2,3]`: **R²=0.986**, A=0.909, null at **+0.43σ** (4000 shots) |
| 8. Hardware Run 3 | `ibm_run3_results.json` | **Replication on a disjoint layout** B `[4,5,6,7]`: **R²=0.976**, A=0.903, null at **−0.12σ** |

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
  stayed within CI — consistent with qubit-dependent noise dominating over a
  depth difference of this size. (One disjoint layout is a replication, not a
  demonstration of layout independence in general.)

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

## Companion result: one budget, and what this repository contributes to it

This repository and its sibling
[relational-time-ibm-quantum](https://github.com/IllI/relational-time-ibm-quantum)
ran as independent programmes and measure the **same law**: coherence surviving
as a product of half-angle state overlaps, one factor for every party that has
acquired distinguishing information. `cos(θ/2)` is the overlap between two
qubit states separated by Bloch angle `θ`.

- **Here:** `T_xx = cos(χt/2)^(N−2)` — the exponent counts the `N−2` bulk
  spins, each learning partial *which-state* information about the boundary
  pair. Measured against **coupling angle**.
- **There:** `ρ_C[t,t'] = cos((t−t')·π/d)` — the same half-angle overlap,
  measured against **time**.

Exact, not analogical: at `d = 8` adjacent clock records overlap at `0.923880`,
and `cos(π/4 / 2) = 0.923880`.

**The joint result is stronger than a shared functional form.** Every
correlation magnitude either programme measured turns out to be drawn on **one
conserved unit**, and that unit is spent twice over: between a subsystem's local
coherence and its entanglement with a partner
(Jakob–Bergou, Phys. Rev. A **68**, 022107, 2003), and between its entanglement
with one partner and with another (Coffman–Kundu–Wootters, Phys. Rev. A **61**,
052306, 2000). Both totals are exactly 1, and depolarizing noise never pushes
either above it.

The consequence, which neither programme states alone: a Page–Wootters clock is
a good clock in proportion to the entanglement it spends on its own system, so
**no correlation magnitude can serve as a shared temporal reference between two
good clocks.** The sibling measured both trade-offs as continuous hardware
curves — `r = −0.9573` within a pair, `r = −0.9833` across two clocks, with both
endpoints of the latter reaching exactly zero.

**This repository's contribution to that argument is the separable state.** At
`χt = π` the boundary state is `I/4` — maximally mixed marginals, concurrence
exactly zero, **separable**. The sibling's `d = 2` null has maximally mixed
marginals too, on a Bell pair carrying **1.0 ebits**. A marginal observable
reads zero on both; `T_xx`, a joint correlator, separates them at `0` versus
`+1`. That contrast is what makes marginal blindness concrete rather than a
single-programme curiosity, and neither programme could make it alone.

It also supplies the **first** instance of the certification wall both
programmes hit — entanglement is not identifiable from PTM anisotropy alone —
which is §"The shared method" below.

Full statement, with the identities attributed and the scope stated as limits:
[`docs/COMPANION_RESULT.md`](docs/COMPANION_RESULT.md). The synthesis
repository is
[relational-entanglement-network](https://github.com/IllI/relational-entanglement-network),
where the argument is executable end to end and the open question — whether the
Aharonov–Anandan geometric phase, which is *not* a magnitude, escapes the
budget — is specified as a run.

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

## Taking the stack to real astronomy

Late in the TRC allocation, the same tensor-network and state-space machinery
was pointed at real observational data — a deliberate deviation from the grant
proposal, to test whether the method transfers out of simulation into a working
physics domain. Targets: TRAPPIST-1e visit-variable stellar contamination
across four JWST/NIRSpec PRISM transits, a DREAMS per-visit Gaussian-process
reproduction, paired TRAPPIST-1b/e transits, and the GLIMPSE-17775 spectrum.
Full record in `docs/TRC_ASTROPHYSICS_MODELING_PROGRAM_CLOSEOUT_2026-06-21.md`.

**What transferred.** The infrastructure did: JAX CPU/JIT smoke paths and
eight-device TPU `pmap` execution, strict null suites with fail-fast verdict
files, reproducible product ledgers with checksums, and synthetic positive
controls gating every real-data claim. On the WASP-39b positive control the
pipeline recovered an injected CO₂-window residual at **0.9897**, passing
phase, wavelength, shifted-template and false-positive controls. Three clean
public b/e pairs were assembled with verified labels and offsets, and the
released Program 1331 spectra reproduced the *direction* of the DREAMS result
(log-likelihood −1752.99 → −1694.46 → −1670.58 across no-GP, per-visit GP, and
shared-spectrum-plus-per-visit-GP).

**What did not.** D-LinOSS did not promote as the detector. On the same
WASP-39b control it reached 0.9828 against a plain linear SSM at 0.9997, and
damping ablation was negligible. MPS spectral compression materially improved
missing-window recovery (0.4359 → 0.7462), but under the tested gates
**MPS + linear SSM was preferred to MPS + D-LinOSS**.

Two honest qualifications belong with that verdict. The first experiments ran
at the **wrong reduction stage** — on detector-level integration flux rather
than extracted transmission spectra — so the failed 25–500 ppm injections are
negative results for a different objective, not limits on DREAMS sensitivity.
And the closeout identified four concrete implementation defects in the model
as built: wavelength bins treated as independent scalar streams, soft spectral
windows standing in for hard temporal pole constraints, full-spectrum
reconstruction rewarding the static baseline, and hidden state overwritten
rather than accumulated.

**This is a model-selection result, not evidence that state-space modelling is
inappropriate for astronomy.** The methodology — residual-first targets,
predict-before-assimilate updates, synthetic positive controls before real-data
claims, and explicit prohibition of atmosphere or molecule claims when gates
failed — held up. The implementation was rushed and is where the losses came
from. Nothing here claims a TRAPPIST-1e atmosphere, methane, or a detection of
any kind, and the closeout says so explicitly.

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
- **Quantum hardware:** IBM Quantum Open Plan access, `ibm_marrakesh`
  (156-qubit Heron r2).

## Computational resources

Research supported with Cloud TPUs from Google's TPU Research Cloud (TRC).

- **Allocation:** `v6e` instances; an initial one-month allocation from
  2026-04-21, extended by one further month.
- **Workloads:** exact boundary-state core, MPS automaton (`N ≥ 64`), Lindblad
  dephasing sweeps and the `Γ*(N)` threshold, Gate-2 bootstrap CIs, Haar
  convergence and seed-independence passes, and the PTM/recovery-geometry
  search documented in `docs/DISCOVERY_NARRATIVE.md`. The `N = 6` scaling bound
  — a negative result — was also established here, at zero hardware cost.
- **Separation of stages:** the IBM hardware stage used no TRC resources; the
  TRC stage used no quantum hardware.
- The same allocation window supported the sister program's synthetic phase;
  see its README for the accelerator-as-hypothesis-generator argument in full.

## References

1. Kitagawa, M. & Ueda, M. *Squeezed spin states.* Phys. Rev. A **47**, 5138 (1993).
   — one-axis twisting.
2. Ma, J., Wang, X., Sun, C. P. & Nori, F. *Quantum spin squeezing.* Phys. Rep.
   **509**, 89 (2011).
3. Chuang, I. L. & Nielsen, M. A. *Prescription for experimental determination of
   the dynamics of a quantum black box.* J. Mod. Opt. **44**, 2455 (1997).
   — process/transfer-matrix tomography.
4. Nielsen, M. A. & Chuang, I. L. *Quantum Computation and Quantum Information*
   (Cambridge University Press, 2010). — PTM conventions, teleportation, the
   classical 2/3 fidelity benchmark.
5. Schollwöck, U. *The density-matrix renormalization group in the age of matrix
   product states.* Ann. Phys. **326**, 96 (2011). — MPS methods.
6. Lindblad, G. *On the generators of quantum dynamical semigroups.* Commun.
   Math. Phys. **48**, 119 (1976).
7. Clauser, J. F., Horne, M. A., Shimony, A. & Holt, R. A. *Proposed experiment
   to test local hidden-variable theories.* Phys. Rev. Lett. **23**, 880 (1969).
   — CHSH.
8. Javanainen, J. & Yoo, S. M. and subsequent JILA Sr-87 clock literature; the
   single-particle dephasing rate used here is anchored to
   [arXiv:2505.06444](https://arxiv.org/abs/2505.06444).
9. Qiskit contributors. *Qiskit: An Open-source Framework for Quantum Computing*
   (2023). — transpilation, `optimization_level`, SamplerV2 primitives.
10. IBM Quantum. *Heron r2 processor documentation* (2026).
11. Companion program: *Measuring the Quantum Signature of Relational Time on
    Superconducting Hardware*,
    <https://github.com/IllI/relational-time-ibm-quantum>.

*(Bibliography is indicative for the repository record; the manuscript version
will carry full DOIs and page ranges.)*
- The views expressed are those of the author and do not reflect the official
  policy or position of IBM, the IBM Quantum team, or Google.
