# One law, and a witness that runs backwards

*A result that neither companion programme can state alone.*

Two hardware programmes were run as independent lines. One measures how
entanglement survives across a spin chain
([twisted-spin-ptm](https://github.com/IllI/twisted-spin-ptm)); the other
measures whether relational time leaves a quantum signature distinguishable
from its classical mimic
([relational-time-ibm-quantum](https://github.com/IllI/relational-time-ibm-quantum)).

Placed side by side, they measure the same law — and together they show that
the coherence observable at the centre of the relational-time literature is not
merely uninformative about entanglement. **It is anti-correlated with it, and
necessarily so.**

Everything below is reproducible in under a second:

```bash
python theory/verify_companion_result.py
```

---

## 1. One law, two variables

`cos(θ/2)` is the overlap between two qubit states separated by Bloch angle
`θ`. Both programmes measure coherence surviving as a **product of those
overlaps — one factor for every party that has acquired distinguishing
information.**

| | measured quantity | form | what the exponent counts |
|---|---|---|---|
| **twisted-spin-ptm** | `T_xx = cos(χt/2)^(N−2)` | overlap^(N−2) | the `N−2` bulk spins, each learning partial *which-state* information about the boundary pair |
| **relational-time** | `ρ_C[t,t'] = cos((t−t')·π/d)` | overlap¹ | *which-time* distinguishability between two clock records |
| **relational-time** (IBM-1) | witness `∝ cos(μ/2)^p` | overlap^p | environment modes coupled to the clock at strength `μ` |

The identification is **exact, not analogical**. At `d = 8` adjacent clock
records overlap at `0.923880`; `cos(π/4 / 2) = 0.923880`. The clock-record
overlap *is* the half-angle state overlap, and the OAT boundary element is that
same overlap raised to the number of informed spins.

> Entanglement decay and temporal distinguishability are the same bookkeeping.
> Information leaving a subsystem is simultaneously entanglement lost and
> moments becoming distinguishable. One programme measures it against coupling
> angle; the other against time.

## 2. Why the witness runs backwards

The relational-time programme's headline observable reads the **clock
marginal** — trace out the system, read the clock in its Fourier basis, measure
the deviation from uniform. That choice has a consequence that follows from one
line of standard theory:

> For a bipartite **pure** state, the marginal entropy *is* the entanglement
> entropy.

So entanglement with the system is precisely what mixes the clock marginal, and
a mixed marginal is precisely what flattens the Fourier distribution the witness
measures. **Coherence of the marginal and entanglement across the cut are in
direct tension, by construction.**

Evaluated exactly, across clock dimensions:

| `d` | state | clock-marginal witness | entanglement |
|---|---|---|---|
| 2 | history | **0.000000** | **1.0000 ebits** |
| 2 | product | **0.500000** | 0.0000 ebits |
| 4 | history | 0.176777 | **1.0000 ebits** |
| 4 | product | **0.750000** | 0.0000 ebits |
| 8 | history | 0.496689 | **1.0000 ebits** |
| 8 | product | **0.875000** | 0.0000 ebits |

The witness is *highest* exactly where entanglement is *zero*.

**This predicts a hardware result quantitatively.** The relational-time
programme's IBM-2 run prepared a zero-entanglement product state as an
adversarial control and measured it scoring **4.2× higher** on the witness than
the real history state at `d = 4`. The exact ratio above is
`0.750 / 0.176777 = 4.24`. The adversary did not exploit a loophole; it
saturated a quantity that entanglement necessarily suppresses.

## 3. Two exact nulls, and what they add

Each programme has an exact structural zero, each predicted from theory,
pre-registered, and confirmed on hardware — the internal controls that
validated each apparatus.

```
twisted-spin-ptm, χt = π      relational-time, d = 2
T_xx        = 0.000000        witness TVD  = 0.000000
ρ₂          = I/4 exactly     state        = Bell pair
concurrence = 0.000000        entanglement = 1.0000 ebits
→ SEPARABLE                   → MAXIMALLY ENTANGLED
```

Both null states share the property that drives §2: **maximally mixed
marginals.** A maximally entangled pure state has them; so does a globally
maximally mixed state. Evaluating both observables on both states separates
what each can see:

| state | `T_xx` (joint correlator) | clock-marginal witness |
|---|---|---|
| `I/4` — separable | `0.000000` | `0.000000` |
| Bell — maximally entangled | **`+1.000000`** | `0.000000` |

The two observables are **not the same functional**: `T_xx` is a joint
correlator and does separate these states, while the marginal witness cannot.
What the programmes share is the *law* and an exact null. twisted-spin-ptm's
null supplies the separable state with maximally mixed marginals, which is what
makes the marginal blindness concrete rather than a single-programme curiosity.

Two cautions, so this is not over-read. `T_xx` separating these two particular
states does **not** make it an entanglement witness — twisted-spin-ptm's own
finding is that entanglement is not identifiable from PTM anisotropy alone. And
the §2 anti-correlation is stated for the pure-state family actually tested; it
is not a claim that every coherence measure anti-correlates with every
entanglement measure everywhere.

## 4. Independent convergence

Svozil, *Certified Private Relational Time from Entanglement*
([arXiv:2512.09100](https://arxiv.org/abs/2512.09100)), builds a relational
clock from a singlet shared between two separated observers and reaches the
same wall from a fourth direction: **the individual marginal tick rates remain
`1/2` and carry nothing.** The temporal structure exists only in the *joint
coincidence record*, `R(θ) = ½sin²(θ/2)`, and he certifies it
device-independently through a CHSH violation.

That is the same mechanism as §2 — a maximally entangled state has
uninformative marginals — arrived at independently, and it supplies a concrete
protocol for the device-dependence limitation the relational-time programme
records but does not close.

## 5. The certification threshold

Both programmes hit the wall independently and resolved it the same way:

| | how it appeared | how it was resolved |
|---|---|---|
| **twisted-spin-ptm** | entanglement is not identifiable from PTM anisotropy alone | `V_Q` over Haar-random measurement settings |
| **relational-time** | no single local product-basis distribution certifies clock–system entanglement *(proved, then verified adversarially)* | multi-setting fidelity witness exceeding the exact separable bound `λ_max = ½` |

The relational-time programme got there the hard way, building two adversarial
states against its own results — the product state of §2 scoring 4.2× higher,
and a separable state scoring 1.7× higher on the joint witness built to repair
it — before a two-line theorem generalized the failure.

**The combined statement:**

> The survival of entanglement and the distinguishability of moments are the
> same quantity, measured in different variables: a product of state overlaps,
> one factor per informed party. Read on the marginal, that quantity does not
> merely fail to certify entanglement — it is suppressed by it, so the
> best-scoring states are the least entangled ones. Certification requires
> agreement across incompatible measurements, a conclusion reached
> independently by two hardware programmes here and by a third line elsewhere.

What follows is a **certification threshold** higher than the observables
usually offered as evidence for relational time can reach. Conditional
evolution is reproduced by a classical clock control. Local clock coherence is
maximized by a state with zero entanglement. Single-basis joint correlation is
beaten by a separable state. What survives is a multi-setting witness — and, in
the relational-time programme's final run, a single 3-qubit preparation
certified **entangled, stationary as a quantum ray, and internally evolving
simultaneously**.

## 6. What this does not claim

No gauge field, photon, or vacuum appears in either programme: OAT is a spin
model and the Page–Wootters clock is engineered qubits, so nothing here is a
statement about quantum electrodynamics. Neither programme claims that time in
nature is emergent, tests the Wheeler–DeWitt constraint, or realizes a physical
Page–Wootters universe. Both states are prepared by externally timed gates, and
both clock/system splits are imposed rather than derived.

The nearest legitimate route to field states is the **Dicke model** — genuine
light–matter coupling, reducing to OAT in the adiabatic limit
(`χ_eff = g²/ω_m`). It was characterized once during the TPU campaign,
returning `K_eff = 1, γ_dom ≈ 0`, cleanly separated from OAT by damping alone.
That is a fingerprint, not a test of the law against field occupation.
Extending it is a further paper, not a claim in these two.
