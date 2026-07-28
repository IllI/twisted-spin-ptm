# OAT / Teleportation-Adjacent Program — Experiment Archive and Audit

**Scope:** the OAT boundary-state entanglement program and its IBM Quantum hardware
validation, covering roughly the first month of the TRC TPU grant.
**Audit basis:** source code, result JSONs, and `FINDINGS.md` (138 numbered findings),
read directly. Claims in prose documents were not taken at face value where code or
results could be checked instead.
**Status of this document:** provenance record and narrative audit. It is not a paper
draft and makes no claims of its own.

---

## 0. Provenance flags (read first)

Three things must be settled before any of this is published.

**Credential exposure.** A live IBM Quantum API token is hardcoded at the top of seven
tracked files: `ibm_submit.py:11`, `ibm_run2.py:8`, `ibm_run3.py:10`,
`_archive_results.py:5`, `_check_usage.py:5`, `_diag_transpiler.py:13`,
`_update_paper_run2.py:6`. All seven are committed to git history. Revoke and reissue
the token; move to `os.environ["QISKIT_IBM_TOKEN"]`. Because the paper repo should be
built fresh rather than forked, the contaminated history does not need to travel.

**Chronology is not defensible as recorded.** Nearly all 138 findings carry session
headers dated `2026-05-11` or `2026-05-12`, including three IBM hardware submissions
with real queue times. The session dates were evidently not updated as work progressed.
The authoritative chronology is the IBM job IDs and their server-side timestamps, plus
TPU node metadata. Do not cite `FINDINGS.md` dates in a methods section.

**Device labels are unreliable in filenames.** `jila_oat_exact_tpu.py` carries `tpu` in
its name, but its own result JSON records `"devices": "[CpuDevice(id=0)]"`. Several
"TPU" scripts ran on CPU. Any compute-provenance statement must come from the recorded
device field, not the filename.

---

## 1. Experiment ledger

### 1.1 Foundation — exact OAT boundary state

**`src/simulation/jila_oat_exact_tpu.py`** (243 lines)

Implements genuine one-axis-twisting physics: state evolution
`psi[k] *= exp(-i·chi_t·mA[k]·mB[k])`, i.e. `H = χ·Jz_A·Jz_B`, collective-spin and
diagonal in the computational basis. Provides Wootters concurrence on the full 4×4
reduced density matrix and the Bowen–Bose relation `F_opt = (2+C)/3`. This module is
imported by every downstream experiment; it is the shared physics core and it is
correctly implemented.

**Key early results** (`FINDINGS.md` §1–4):
- The relevant pair is a **cross-half** pair (one atom from each half), not the first
  and last atom. Same-half pairs have C=0 identically. The earlier "first and last
  atom" reading was wrong and was corrected here.
- Closed-form `ρ₂` derived analytically and validated against brute force. O(1)
  computation, valid for all N≥2 — this is what made the whole N-sweep tractable.
- `ρ₂` is **not** an X-state. All 16 elements are generically nonzero. The prior
  X-state proof was invalid.

### 1.2 The v-series — teleportation protocol, four generations

| Version | File | What changed | Why it was superseded |
|---|---|---|---|
| v1 | `archive/oat_teleport_v1_tpu.py` | Full Bennett protocol on OAT-derived **Werner proxy** states | Werner proxy is not the real OAT state |
| v2 | `archive/oat_teleport_v2_tpu.py` | **Schmidt rotation** — teleports the actual OAT boundary state; MPS-style contraction | Exact contraction limits N |
| v3 | `src/simulation/oat_teleport_v3_tpu.py` | **True MPS automaton.** Exploits diagonality of U_OAT to build the MPS directly — no MPO sweep, no Trotter error. Bond dim n+1 (51 at N=100 vs 2¹⁰⁰ exact) | Closed system only |
| v4 | `src/simulation/oat_teleport_v4_open_system.py` | **Analytic Lindblad dephasing** on the exact RDM; Γ-sweep to entanglement threshold | Became the decoherence workhorse; not superseded |

v3 is the genuine technical achievement of this sequence. Recognizing that
`U_OAT = exp(-i·χt·Jz^A·Jz^B)` is diagonal, and therefore that the MPS can be
constructed as an automaton rather than by sweeping, is what made N=64+ reachable at
all. That is a reusable result independent of the physics conclusions.

v4 hardcodes `GAMMA_SINGLE = 1.0/118.0` s⁻¹, commented **"Physical constants from
Dr. Rey's lab (arXiv:2505.06444)"**. This is the concrete anchor to JILA — a real
published single-particle dephasing rate parameterizing the simulated open system.
It is an anchor, not a comparison: no JILA dataset was ingested.

### 1.3 The v5 series — the D-LinOSS framework-discrimination saga

This is the most instructive sequence in the archive, and the one that carries a
genuinely transferable lesson.

**v5** (`archive/oat_teleport_v5_framework_discriminator.py`, 436 lines) attempts to
discriminate five dynamical frameworks — Lindblad, MBL, SYK, Penrose OR, Heisenberg
null — by Prony/ESPRIT modal decomposition of the **concurrence** decay series
`C(Γt)`. Also computes a "bi-twistor null residual" and a winding number.

**v5b** (264 lines) applies three fixes: truncate the signal at the noise floor before
fitting; redefine the null residual via `det(M)`; use the Bloch polar angle rather than
Berry phase for winding. The fits remained unstable.

**v5c** (254 lines) abandons Prony entirely — the docstring states it is "numerically
ill-conditioned" on a single-mode exponential — and switches to direct model comparison:
fit each framework's predicted `C(x)` shape and rank by residual. Winding is corrected
again, to Bloch arc length, with the note "(not winding number — correct for dephasing)".

**v5d** (`src/analysis/oat_teleport_v5d_witness_observer.py`, 326 lines) identifies the
actual root cause. Concurrence is a **nonlinear** functional of ρ (Wootters involves
eigenvalues of a matrix product), so it does not decompose into a clean sum of
exponentials — no modal method can be well-posed on it. The entanglement witness
`Tr[W·ρ(t)]` is **linear** in ρ, and under Lindblad dephasing decays exactly as
`Σ_k c_k·exp(-Γ·d_H(k)²·t)`, summed over coherence pathways with Hamming distance
`d_H(k)`. The docstring records the fix plainly: "No Wootters nonlinearity artifact ✓".

**The lesson, stated generally:** a linear modal observer requires a linear observable.
Three iterations were spent fighting numerical artifacts that were actually a category
error in the choice of target. This generalizes well beyond OAT and is worth stating
explicitly in any methods paper.

### 1.4 Supporting analyses

- **`src/analysis/chsh_bell_test.py`** — genuine null control. Only N=2 violates the
  classical bound (S=2.824, near Tsirelson 2.828). N=4 gives S=1.71: **no CHSH
  violation** even where teleportation fidelity exceeds 2/3. Later reinforced by direct
  Horodecki computation (§17): `S_max < 2` for all N≥4, with no X-state assumption.
  Fidelity advantage without Bell nonlocality is a real and self-consistent finding.
- **`channel_tomography.py`** — PTM reconstruction. Produced the centerpiece structure
  `T_xx ≫ T_yy, T_zz` (§31: 0.962, 0.007, 0.000 at N=4).
- **`ptm_analytic.py`** — proved `T_zz = 0`, `T_yy = 0`, `T_xx = cos^(N-2)(χt/2)`
  analytically from the closed-form ρ₂ (§34). This is what made hardware testing viable.
- **`dicke_ptm.py`** — Dicke-model generalization (§60–62), detailed in §3.2 below.
- **`collective_dephasing.py`**, `vq_phase_diagram.py`, `vq_n_scaling.py` — recovery
  basin volume `V_Q` phase diagram and N-scaling.

### 1.5 IBM Quantum — dry run and three hardware runs

**Dry run.** `src/experiments/exp_ibm_trotterized_oat.py` (314 lines) runs on
`qiskit_aer.AerSimulator` with a calibrated noise model, explicitly "no API key
required". Its result JSON records `"backend": "qiskit_aer_fake"`. **This script never
touched hardware** — it is a pre-registration and circuit-validation harness. It
measures the v5d witness `W = I/4 - |Φ+⟩⟨Φ+|`, so the witness insight fed directly into
the hardware planning.

**Strategy pivot (§108) — the decision that made the hardware runs work.** With ~10
minutes/month of free quantum time, the scope was cut hard: measure the PTM at
`χt ∈ {0, χt*, π}` and observe the rank 1→4→1 transition. Explicitly excluded were
`V_Q` (needs Haar sampling over many shots), `κ_Q` (needs a Hessian, far too noisy), and
full teleportation fidelity (Bell measurement plus feedforward overhead). Choosing the
single most analytically anchored observable with a hard predicted null is why a
10-minute budget produced a publishable result.

**Circuit.** `oat_ptm_circuit` builds `H = χt·JzL·JzR = χt/4·(ZZ₀₂+ZZ₀₃+ZZ₁₂+ZZ₁₃)` as
four CX–RZ–CX blocks on exactly the cross-half pairs, then rotates qubits 1,2 into the
X basis and measures. Minimal, correct, and directly matched to the analytic prediction.

**Pre-registration.** `ibm_submit.py` writes a prereg JSON *before* submission
containing numerical predictions with uncertainties and an explicit pass criterion:
`T_xx(product)=0.500±0.028`, `T_xx(quantum)=0.360±0.028`, `T_xx(null)=0.000±0.028`,
`invariant_ratio=0.720±0.060`, and `passage_criterion: T_xx(χt*) > T_xx(π) + 2σ = 0.281`.
This is the single strongest methodological feature of the entire program.

| Run | Design | Result |
|---|---|---|
| 1 | 3-point PTM, `ibm_marrakesh` | Signed ordering confirmed, **12.5σ** separation |
| 2 | 9-point sweep, layout A `[0,1,2,3]`, depth 76 | **R²=0.986**, A=0.909, null at **+0.4σ** |
| 3 | Same on **disjoint** layout B `[4,5,6,7]`, depth 64 | **R²=0.976**, A=0.903, null at **−0.12σ** |

Run 3 programmatically searches the coupling map for a 4-qubit linear chain disjoint
from layout A, then verifies the Rz angles match intent. Both runs 2 and 3 use
`optimization_level=0` specifically to stop the transpiler from altering Rz angles —
the fix for the root cause found in §123. Run 3 answers the "transpiler/routing
artifact" reviewer objection before it can be raised.

Bootstrap (§136): A = 0.8584 ± 0.0345, 95% CI [0.814, 0.930]; offset 0.0197 ± 0.0132,
compatible with zero; residual mean 0.005, max residual ~1.4σ.

---

## 2. The corrections ledger

The program repeatedly falsified its own claims. This is its chief credibility asset
and should be presented, not hidden.

| # | Claim | Fate |
|---|---|---|
| §3 | ρ₂ is an X-state; X-state proof of F_opt | **Wrong** — proof applied to the wrong reduced state |
| §10 | SYK4 shows multi-mode scrambling (K_eff=6) | **Retracted** — K_max=6 was a truncation ceiling; BIC gives K=1 |
| §17 | `f_max=(1+C)/2` and `F_opt=(2+C)/3` | **Falsified** for all N≥4 (up to 3.9% violation), by direct optimization validated against exact Werner ground truth |
| §18 | `C_peak ~ N^(-1)` | **Revised** to `N^(-0.636)`; `χt* ~ N^(-1/2)`, not `N^(-1)` |
| §22 | Phase-winding geometry hypothesis | **Not confirmed** |
| §30 | `F_avg = 0.823` | **2× normalization error**; caught by auditing against Bell / Werner / maximally-mixed ground truths |
| §59 | Dual-TPU entanglement persistence | **Rejected as stated** — "analogous to emailing a density matrix to two computers"; reframed as a hardware-agnostic reproducibility check |
| §77 | D-LinOSS isolates χt=π in feature space | **Partial** — KMeans does not isolate it; transition is continuous, not first-order |
| §94–95 | κ_Q as precision observable; ν exponent status | **Downgraded** to phase indicator; epistemic status corrected |

§17 deserves particular attention: `F_opt = (2+C)/3` was effectively the headline
theorem, and it was killed by the project's own numerics — with the optimizer first
validated to machine precision against exact Werner ground truth so the falsification
could not be blamed on the tool. That is how it should be done.

---

## 3. The two narratives you remembered — grounded status

### 3.1 D-LinOSS trained on quantum parameters

**What genuinely works:**
- **Exact calibration anchor.** `γ_dom = 4Γ` recovered to four decimal places for every
  OAT record, N=2–64, all Γ (§6, §14). The damped-oscillator control recovers γ and ω
  exactly (K=2, as required for a damped sinusoid). Where the underlying dynamics are
  genuinely low-mode, the recovery is exact.
- **Unsupervised class separation** (§76). Curve-shape fingerprints separate
  OAT-QUANTUM (γ=0.049, K=4), CLASSICAL (γ=0.223, K=4), and OAT-SINGULAR (γ=2.146,
  K=3, zero-terminating) without labels.
- **The linearity lesson** (§1.3 above) — the most transferable methodological result.

**Documented limits, from the project's own adversarial suite:**
- **Cannot distinguish SYK4 from OAT.** Both collapse to K_BIC=1, I_inf=0 (§15). The
  SYK scrambling claim was retracted on this basis.
- **False-positives on classical random telegraph noise** (K_BIC=13, §11, §14). The
  log's own conclusion: the framework "reads spectral morphology, not quantum structure
  per se."
- χt=π is not isolated in 6D feature space — it is the extreme of a continuous
  trajectory, not an outlier (§77).

**Honest framing:** this is a dynamical-class fingerprinting tool with a characterized
false-positive envelope. It is not a quantum-structure detector. That is still a useful
thing to offer an experimental group — and the characterized failure modes are what make
it offerable.

### 3.2 Dicke damping, precession, and the Rey connection

This thread contains the program's most concrete falsifiable prediction for an
experimental group, and it survives.

**The Dicke result (§60–62), `dicke_ptm.py`:**
- Dicke boundary pairs have `C>0` and `V_Q>0` — the recovery-basin framework
  generalizes beyond pure OAT.
- `V_Q` peaks at `g/g_c ≈ 0.8`, **just below** the critical point (V_Q=0.075), not at it.
- **The dispersive approximation `χ_eff = g²/ω_m` is a lower bound, not an equality.**
  The fitted ratio is ~10 at g=0.05, ~3 at g=0.20, and converges to ~1 only near `g_c`.
- **The prediction:** Dicke boundary states are *more* entangled than the naive OAT
  approximation suggests. At `g=g_c`, `V_Q=0.060` versus OAT's `V_Q=0.038` at `χt*`.
  Dicke near criticality is the stronger resource.
- `T_xx < 0` for all g>0 — expected, since `H_eff = -χ_eff·Jx²` squeezes in z; requires
  a basis rotation, not a correction.

**The precession thread (§0, §32, §53).** The phase-alignment prediction is that two
`Rz` gates applied before Bell measurement recover the full teleportation advantage,
`ΔF = (1-C)/6` — a pure-software gain requiring **zero new hardware**, and one that
*increases* with N as entanglement weakens. The gate mapping is exact for Sr-87:

- `Rz` = clock laser phase shift — **already implemented at JILA**
- `Rx` = Rabi pulse — already implemented

§32 establishes that Rz alone is insufficient (F_avg=0.661 < 2/3); full SU(2) is
required (F_avg=0.719). The gate set is standard, not exotic. Recommended platform is
an N=4–8 tweezer array, explicitly **not** a large lattice clock, where `V_Q ≈ 0`.

**Open item that must be closed before this is pitched:** §17 falsified
`F_opt = (2+C)/3`, which supersedes the ΔF table in §0. The phase-alignment gain
numbers need recomputation from the `F_opt_direct` table. The *direction* of the result
survives — §19 still lists "phase alignment θ_A→π as N grows" as CONFIRMED — but the
magnitudes as currently written are stale. This is a recomputation, not a rescue.

---

## 4. Narratives detected in the record

Beyond the two you named, three patterns are visible across the corpus.

**4.1 The discipline arc is the real story.** The program falsified its own headline
theorem, caught its own 2× normalization error by auditing against known ground truths,
retracted a scrambling claim when it turned out to be a truncation artifact, rejected
its own most exciting conjecture on correct physical grounds, downgraded observables
when statistics did not support them, pre-registered numerical predictions with pass
criteria before spending hardware time, and locked an explicit CANNOT-claim list (§135)
before writing. For an unaffiliated researcher, this audit trail *is* the credential —
it is the clearest available evidence of distinguishing real work from the crank
traffic that gatekeeping exists to filter.

**4.2 Scope reduction is what produced every success.** The pattern repeats at every
scale. v5→v5d: from five-framework discrimination to one linear observable. IBM: from
teleportation demonstration to three PTM points. §110: from "we discovered teleportation
enhancement" to "we characterized a recoverability phase structure." Every time the
claim narrowed, the result got more solid. The hardware result exists *because* of §108's
refusal to measure V_Q and κ_Q.

**4.3 A recurring pull toward premature unification — mostly well-quarantined.** v5
included Penrose objective reduction as a discriminable framework alongside
Lindblad/MBL/SYK, with bi-twistor residuals and winding numbers. §55 extends D-LinOSS
to fMRI, auditory ERP, and CaMKII lattice dynamics. §59 proposed non-local entanglement
between TPUs. These were tagged `[HYPOTHESIS]` / `[SPECULATIVE]` and kept out of the
paper scope, and §59 was self-rejected with a correct physical argument — the quarantine
worked. But the pattern is worth naming, because it is the single thing most likely to
damage reception if any of it leaks into the manuscript. The published claim should stop
where §135 says it stops.

---

## 5. What is publishable, and what carries forward

**Paper 1 — IBM hardware PTM validation.** Theory (closed-form ρ₂ → analytic
`T_xx = cos^(N-2)(χt/2)`) + TPU/MPS simulation + pre-registration + three hardware runs
across two disjoint layouts. Self-contained, verifiable, modest, and already
framing-locked by §135. Target: PRA / PRResearch, per §52's own assessment. The
defensible statement is that a theoretically derived two-body observable was recovered
on real superconducting hardware with the predicted phase structure, layout-independent
functional form (R²>0.975), and an internal null confirmed to within 0.4σ.

**Paper 2 — methods.** The linear-observable requirement for modal observers (v5→v5d),
the D-LinOSS fingerprint taxonomy with its characterized false-positive envelope, and
the pre-registration / promotion-gate discipline. The negative results are the content,
not an embarrassment.

**Carried forward as an experimental proposal, not a paper.** The Dicke prediction
(§3.2) — that boundary entanglement exceeds the dispersive OAT approximation, peaking
just below `g_c` — is a specific, falsifiable, measurable claim addressed to a group
already doing this physics. Pair it with the phase-alignment protocol once the ΔF
numbers are recomputed against `F_opt_direct`.

**Open items before submission:**
1. Revoke the IBM token; rebuild the paper repo with fresh history.
2. Recompute the §0 phase-alignment table from `F_opt_direct` (§17 supersedes it).
3. Reconstruct the true chronology from IBM job IDs and TPU node metadata.
4. Correct device provenance where filenames say TPU and results say CPU.
5. Derive `C_peak` scaling analytically from closed-form ρ₂ (§20A, still open).
6. Γ_mb (many-body dephasing contribution) remains uncharacterized — §135 already flags
   that "survives dephasing" holds only under the untested assumption `Γ_mb ≪ Γ*`.
