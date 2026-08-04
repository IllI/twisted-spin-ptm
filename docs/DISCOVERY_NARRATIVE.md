# How We Got Here: A Story of Wrong Turns That Worked

*A narrative companion to `FINDINGS.md`, the 3,273-line research log this work was
distilled from. That log is the historical appendix — the receipts. This is the
story: what we tried, what broke, and why the breaks were the useful part.*

Research funded in part by Google's **TPU Research Cloud (TRC)**, whose free
compute access made the tensor-network simulation phase of this work possible —
a one-month grant in the OAT/PTM line, extended a further month once the results
looked worth finishing.

---

## The shape of this story

Most papers present a result and let you infer the effort behind it. This one
is different on purpose: nearly every numeric claim in this repository was
wrong at least once before it was right, and the wrong version is usually what
told us where to look next. A concurrence scaling exponent got corrected three
times. A headline fidelity formula got falsified after being the paper's
opening line. A "12.5-sigma" hardware result turned out to be measuring a
different angle than intended — and *that* discovery is what made the second
run rigorous enough to publish.

If you only read one lesson from this: **the internal null did more work than
any single positive result.** Nearly every section below ends up leaning on
the same move — find a point where the theory says the signal must vanish
exactly, then show the same apparatus, same code, same analysis pipeline
gives zero there and something real everywhere else. That move is what turned
a plausible-sounding numerics story into something we were willing to put on
real quantum hardware.

---

## Act I — The state we thought we understood, and didn't

The starting point was simple: entangle the two boundary atoms of an N-atom
chain evolving under one-axis twisting (OAT), and ask whether that boundary
pair carries a usable teleportation resource. The first move was to reach for
a shortcut. Boundary-pair density matrices in spin chains are often
**X-states** — a special sparse structure with only six nonzero entries — and
X-states have a clean, textbook formula for optimal teleportation fidelity:
`F_opt = (2+C)/3`, where `C` is the concurrence.

That shortcut was wrong. The conservation argument that predicts X-state
structure applies to the *full* bipartite state (left half vs. right half of
the chain) — not to the two-atom reduced state left after tracing out
everything in between. All sixteen matrix elements of the boundary pair are
generically nonzero. The clean formula didn't apply, and neither did anything
built on top of it.

This forced a closed-form re-derivation of the exact boundary density matrix
directly from the OAT wavefunction — no X-state assumption, no shortcuts. It
held up. Every later result in this program traces back to that formula.

## Act II — Falsifying our own headline, twice

With the exact density matrix in hand, direct numerical optimization (differential
evolution, 120 restarts, Nelder-Mead and Powell polish, validated against exact
ground-truth Werner states to machine precision) delivered the first hard blow:
`F_opt = (2+C)/3` doesn't hold for N ≥ 4. Not approximately — up to 3.9% off, in
the wrong direction, for every N we tested. A formula that had been sitting at the
top of an early paper draft as *the* result was retracted in the same session it
was checked properly.

The replacement was slower and less quotable, but real: a table of directly
optimized fidelities (`F_opt_direct`), the Horodecki bound proved without any
X-state assumption (ruling out Bell nonlocality for N ≥ 4 rigorously, not by
inference), and — months later — the discovery that the falsified formula
does hold asymptotically, because the boundary state's purity climbs to 1 as
N grows. The "wrong" formula wasn't nonsense; it was the N → ∞ limit of the
right one, and figuring out *why* it converges was its own small result.

The concurrence scaling was corrected almost as many times. First guess:
`C_peak ~ N^{-1}`. First large-scale fit (N = 4–768, dense grid): `N^{-0.65}`.
Restricted to large N only: drifting toward `N^{-0.74}`, not yet converged
in the range tested. None of these numbers made it into a final claim without
the caveat "not yet analytically derived" attached — which, on reflection, is
the correct epistemic state for a fit that kept moving every time we looked
harder.

A companion hypothesis about the optimal recovery rotation — that it should
be a simple phase gate related to the phase of one specific coherence in the
density matrix — was tested directly and failed outright: the optimizer's
answer had no simple relationship to that phase. The physically appealing
guess wasn't just imprecise, it was pointing in the wrong conceptual
direction. Full two-qubit local optimization, not a single Berry-phase-style
correction, turned out to be structurally necessary. That's a negative result
worth keeping precisely because it closed off a whole family of "surely it's
just this one angle" hypotheses.

## Act III — What quantum advantage actually costs

Once the numerics were trustworthy, the operational question became: can a
lab actually recover the advantage with real gates? The first honest answer
was no. Under Rz-only local corrections — the cheapest, "free" virtual-phase
gates available on a real clock platform — the channel's average fidelity
sits *at or below* the classical limit of 2/3 for every N and every coupling
angle we tried. This wasn't a numerical near-miss; it became a proved
corollary of the exact Pauli-transfer-matrix theorems derived later in the
program: `F_avg(Rz-only) ≤ 2/3` identically.

Recovering the advantage requires a second gate — an Rx Rabi pulse, still
completely standard hardware, but no longer free. With both gates, fidelity
climbs to 0.719 at N = 4: a genuine, if modest, 7.8% excess over the
classical bound. That number, not the earlier headline formula, became the
paper's actual quantitative claim, and it survived every subsequent
adversarial check we threw at it — dephasing sweeps, entanglement-destruction
controls, coordinate-invariance tests — while the impressive-sounding earlier
formula did not.

## Act IV — Finding the anchor: a place where the signal must be exactly zero

The project's turning point wasn't a bigger number. It was a smaller one —
zero, exactly, at a single predictable point.

Three proved theorems fell out of the closed-form density matrix: the
correlation-tensor elements `T_zz` and `T_yy` vanish identically for any N
and any coupling time, and `T_xx = cos^{N-2}(χt/2)` in closed form. At
`χt = π`, that last expression hits zero for any N ≥ 4, and — proved as a
corollary — the boundary state becomes exactly the maximally mixed state.
Every local unitary recovery strategy gives exactly `F_avg = 1/2` there,
independent of N, independent of what optimizer or parameterization you use.

That single point became the spine of everything that followed, because it's
an *internal* control: same hardware, same protocol, same reconstruction
pipeline, only the coupling angle changes — and the theory says the answer
must come out to zero. Six independent observables (T_xx, the recovery basin
volume V_Q, a Hessian curvature we called κ_Q, the optimized fidelity, the
landscape variance, the smallest Hessian eigenvalue) were checked at that
point across dozens of runs. All six hit zero, always, regardless of which
random seed or optimizer restart or SU(2) parameterization was used to probe
them. That robustness is what let us later trust a single IBM hardware run
enough to build a whole publication strategy around it: if the null shows up
that reliably in simulation, a hardware run that reproduces the *shape* of
the theory around that null is meaningful evidence, even under real device
noise.

## Act V — A geometric detour, and the discipline of walking it back

For a stretch of the project, the most exciting-sounding results were
critical-exponent claims: the curvature near the `χt = π` singularity
appeared to follow a clean power law with a non-mean-field exponent (first
estimate `β ≈ 0.87`, later revised after we realized the fit window had
mixed two different physical regimes together — `β = 1.00`, cusp-like, not
the smooth quadratic approach a naive mean-field picture would predict). A
second exponent, `ν ≈ 0.73`, appeared to describe how the recoverable basin
opens up near a separate phase boundary — tantalizingly close to known 3D
critical universality classes, though not matching any of them.

Then came the discipline pass, and it mattered. A dedicated statistical-rigor
gate — bootstrap confidence intervals, seed-independence checks, Haar-sampling
convergence tests — revealed that the *peak* values of the curvature
observable were wildly optimizer-dependent (44–47% coefficient of variation:
the landscape has multiple comparable local optima, and different restarts
found different ones). The precise peak number from an earlier session
(`κ_Q = 1.057`) had to be retracted as unreliable. The critical *exponents*
survived properly: `β = 1.00 ± 0.19` and `ν = 0.63 ± 0.15`, both with 95%
confidence intervals that exclude the naive mean-field prediction. But the
lesson generalized past this one measurement: **only report a number after
you've tried to break it, and say plainly which parts survive that and which
don't.** The singularity itself — a hard, optimizer-independent, exact zero —
turned out to be far more robust than any of the impressive-looking numbers
describing the terrain around it. That asymmetry between "the null is solid"
and "the peak is noisy" became a standing rule for the rest of the project:
lead with the zero, hedge the peak.

## Act VI — Borrowing a classifier from adjacent physics, and watching it fail informatively

Parallel to the OAT work, a spectral-decomposition classifier (later
formalized as D-LinOSS in the sister branch of this program) was pointed at
several other many-body systems to see whether it could tell them apart from
their dynamics alone, without being told the underlying Hamiltonian.

The first version of this test produced an exciting-sounding claim: SYK4 (a
canonical fast-scrambling model) showed multi-mode spectral structure that
Dicke and OAT didn't, suggesting the classifier had found a genuine scrambling
signature. It hadn't. The apparent multi-mode structure was a **truncation
artifact** — the mode-count ceiling in the fitting routine was capped at 6,
and raising it revealed the true count climbing past 10. Under a proper
information-criterion penalty (BIC), SYK4 collapsed to a single dominant mode
— the *same* classification bucket as plain OAT decay. The scrambling claim
was retracted the session it was found to be an artifact, and it stayed
retracted.

A second, quieter failure was more useful precisely because it was boring:
classical random telegraph noise — pure stochastic switching, no quantum
structure whatsoever — was classified by the same pipeline as multi-mode,
indistinguishable from a genuinely structured system like disordered XXZ
spin chains. That's a **false positive**, discovered by deliberately including
a classical adversarial baseline in the test suite rather than only feeding
the classifier systems we expected to succeed on. It proved something
important about the tool's limits: this classifier reads spectral shape, not
quantum structure per se, and the two are only sometimes the same thing.
Distinguishing genuine localization (XXZ) from classical noise (RTN) turned
out to require a second, independent observable (a long-time imbalance
measure) — the spectral fingerprint alone wasn't enough, and knowing that
early saved the D-LinOSS branch from overclaiming for the rest of its run.

Both of these — the SYK retraction and the RTN false positive — earned their
place here for the same reason: neither one weakened the eventual result.
They sharpened exactly what the tool could and couldn't be trusted to say,
which is what let later claims about it (in the sister Page-Wootters and
entropic-clock work) be stated with real confidence instead of hope.

## Act VII — A conjecture proposed, tested against itself, and declined

Not every idea needs a full experiment to be evaluated. One proposal — deploy
a pre-entangled tensor network to two separate TPU pods and treat matching
outputs as evidence of non-local quantum correlation between the machines —
was worked through on paper and rejected before any compute was spent on it.
The tensor network is classical data; copying a file to two computers
produces identical deterministic outputs, not a quantum channel between them.
It's the same category error as emailing a density matrix to two colleagues
and calling the result entanglement.

The idea wasn't wasted, though — it was reframed into something legitimate
and genuinely useful: split a parameter sweep across two TPU pods, verify
that both independently reconstruct the same recovery geometry, and use that
agreement as a scientific-integrity check (reproducibility, not physics) plus
a practical way to run the grid twice as fast. Same instinct, redirected from
"this proves something exotic" to "this makes the boring part of the pipeline
faster and more trustworthy" — arguably the right final home for most ideas
that start out sounding more dramatic than they are.

*(One more idea from this same period is worth naming honestly: a speculative
connection between this recovery-basin geometry and neural time-series
classification — fMRI categorical learning, synaptic memory dynamics — was
logged as a labeled, three-tiered hypothesis with an explicit list of what
would and would not constitute evidence for it. It was never pursued past
that log entry. It doesn't appear in this paper. It's mentioned here only
because writing down *why* an idea is being shelved, precisely enough that
someone could pick it back up later, is itself part of doing this kind of
work honestly.)*

## Act VIII — The reframe that unified everything: rank, not just angle

Roughly two-thirds of the way through the project, the geometric machinery
(the recovery basin volume V_Q, the curvature observable, the exact PTM
theorems) got reorganized around a single cleaner idea: track the *rank* of
the full 4×4 Pauli transfer matrix as the coupling angle sweeps from 0 to
2π. The pattern that fell out was almost too tidy — rank 1 at zero coupling
(product state, informationally trivial), rank 4 across the entangled middle
region (informationally expressive — this is where teleportation-capable
recovery basins live), and rank 1 again exactly at `χt = π`, but for the
opposite physical reason: not under-entangled but completely scrambled,
outputting the maximally mixed state regardless of input.

That "rank-1 → rank-4 → rank-1" shape did two things for the project.
Scientifically, it unified four previously separate observables (the PTM
theorem, the basin volume, the curvature, the optimized fidelity) into one
coherent story about *informational capacity*, not just fidelity. Practically,
it told us exactly what to spend ten minutes of scarce IBM quantum hardware
time measuring: not the basin volume (needs thousands of Haar-random samples,
too many shots), not the curvature (too noisy to survive real device noise),
but the plain PTM element `T_xx` at three points — the two rank-1 endpoints
and one point in the rank-4 middle. Cheap, robust, and directly falsifiable
against a specific pre-registered number.

## Act IX — Finding the actual circuit before it cost real shots

Before any hardware was touched, the circuit itself was gotten wrong three
separate times in dry-run testing: first a full Bell-measurement teleportation
circuit that wasn't what the theorem actually predicted; then two different
guesses at how the full OAT Hamiltonian should map onto gates, each of which
gave a null value at `χt = π` that didn't match the proved theorem (0.243 and
0.5 instead of 0). The root cause, once found, was almost embarrassingly
simple: the relevant Hamiltonian for the *boundary-pair* observable isn't the
full OAT interaction `J_z²` — it's the cross-half term `J_z^A J_z^B` alone,
which decomposes into exactly four ZZ interactions between the two halves of
a four-qubit chain. Once that was fixed, exact matrix exponentiation matched
the analytic prediction to 1.9%, and the null at `χt = π` fell out cleanly.

This is worth dwelling on for a sentence: three wrong circuits were caught
and fixed entirely in simulation, at zero cost, because the discipline was to
verify every circuit against the exact theorem *before* it was allowed near
real hardware. That habit — dry-run first, always — is the single practice
from this program most worth carrying into any future hardware-facing work,
in this repository or otherwise.

With the right circuit confirmed, a full noise-injection dry run (realistic
IBM device parameters: T1 = 150 µs, T2 = 80 µs, 2% readout error) predicted
92.6% signal survival at the quantum operating point and a preserved null —
and, critically, a pre-registered failure-mode taxonomy: what a uniform
degradation would mean (systematic, fixable by calibration), what a nonzero
null would mean (shot noise, not failure), what a null point *above* two
sigma after calibration would mean (a genuine problem worth investigating).
Writing that taxonomy down before submission is what turned the actual
hardware run from an open-ended fishing expedition into a pre-committed test.

## Act X — Three hardware runs, in order, with the mistakes on the record

**Run 1** (`ibm_marrakesh`, three points, 2026-05-12): a 12.5-sigma separation
between the quantum operating point and the null — the headline number,
and a real one. But it shipped with two loose threads, reported rather than
buried: the measured value at the "quantum" point was higher than predicted,
and the null point missed zero by 2.1 sigma. Both were investigated rather
than waved away, and both turned out to be understood, fixable artifacts —
the transpiler (running at `optimization_level=2`) had quietly re-synthesized
the intended coupling angle into a different one entirely (0.237π instead of
0.355π); once that *effective* angle was used in the formula instead of the
intended one, the match was near-exact. The null offset traced to a small,
measurable readout asymmetry between the two possible bit outcomes.

**Run 2** fixed both root causes directly rather than patching around them:
`optimization_level=0` to stop the transpiler from rewriting gate angles,
a full 9-point angle sweep instead of three points (so the *shape* of the
curve, not just isolated values, could be checked), 4000 shots concentrated
at the null point to tighten its uncertainty eightfold, and proper
readout-matrix mitigation reported alongside the raw counts rather than a
blanket rescaling. Every fix targeted a specific, named failure from Run 1.
The result: the curve fit the predicted `cos²` functional form with
R² = 0.986 across all nine points, and the null resolved to 0.4 sigma from
zero — a fivefold tightening from Run 1's 2.1 sigma, using the same physical
qubits and the same underlying circuit design.

**Run 3** asked one more question: is any of this a property of the
particular four physical qubits used, rather than of the physics being
measured? The identical protocol, run on a fully disjoint four-qubit chain
on the same chip, gave R² = 0.976, a null at −0.12 sigma, and — genuinely
informative on its own — a *shallower* circuit (depth 64 vs. 76) that
nonetheless showed *slightly worse* attenuation (0.903 vs. 0.909). That
combination ruled out the simplest alternative explanation (deeper circuit,
more noise, worse signal) and pointed instead to qubit-specific decoherence
dominating over routing overhead in this noise regime — a small, honest,
falsifiable physical claim about the hardware itself, earned as a byproduct
of a control experiment rather than sought directly.

## Act XI — Locking the language, on purpose

The last deliberate act in this line of the program wasn't a new measurement.
It was a short, explicit table of what the results could and could not be
described as, written down and then held to. "Hardware-confirmed theorem,"
"quantum teleportation demonstrated," "rank-collapse proved experimentally" —
all of these were considered, and all of them were ruled *out*, replaced with
narrower, defensible language: a correlation observable predicted by tensor-
network theory, recovered on real superconducting hardware, with a functional
form fit of R² = 0.986 and an internal null confirmed to well under one sigma.
That is a smaller claim than the exciting version. It is also one that
doesn't need to be walked back later, which — after a project defined by
walking things back — was worth treating as an achievement in its own right.

---

## What actually survived, in one table

| Claim | Status |
|---|---|
| `T_xx = cos^{N-2}(χt/2)`, `T_yy = T_zz = 0` | Proved analytically; matches exact simulation to 8 decimal places |
| `F_avg(Rz-only) ≤ 2/3` for all N, all χt | Proved (corollary of the PTM theorems) |
| Full-SU(2) fidelity 0.719 at N = 4 | Observed, survives dephasing sweep and entanglement-destruction controls |
| `κ_Q`, `V_Q` = 0 exactly at χt = π | Proved + observed, optimizer- and parameterization-independent |
| Critical exponents β = 1.00 ± 0.19, ν = 0.63 ± 0.15 | Observed, 95% CI excludes mean-field, bootstrap-verified |
| `F_opt = (2+C)/3` | **Falsified** for N ≥ 4; holds only asymptotically as N → ∞ |
| SYK4 multi-mode scrambling signature | **Falsified** — BIC-corrected truncation artifact |
| Simple phase-rotation hypothesis for optimal recovery | **Falsified** by direct test |
| PTM rank transition 1 → 4 → 1 | Proved analytically; confirmed under realistic noise simulation |
| Hardware Run 1 (12.5σ, 3 points) | Confirmed, with two named artifacts later explained and fixed |
| Hardware Run 2 (R² = 0.986, 9 points) | Confirmed; both Run 1 artifacts resolved |
| Hardware Run 3 (R² = 0.976, disjoint layout) | Confirmed; layout-independence established |
| "Theorem 3 hardware-confirmed" | Retracted in favor of narrower, defensible language |

Every falsified row above earns its place in this table for the same reason
the hardware artifacts do: the correction is what made the surviving rows
trustworthy. None of this would be worth writing up if the wrong turns had
been quietly dropped instead of logged, checked, and — where they were
genuinely wrong — said so in public.
