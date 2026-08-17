# One law, two exact nulls

*A result that neither companion program can state alone.*

These two repositories were run as independent lines. One measures how
entanglement survives across a spin chain
([twisted-spin-ptm](https://github.com/IllI/twisted-spin-ptm)); the other
measures whether relational time leaves a quantum signature
([relational-time-ibm-quantum](https://github.com/IllI/relational-time-ibm-quantum)).
Placed side by side they turn out to measure **the same function**, and their
two internal nulls form a matched pair that settles what that function can
certify.

---

## 1. The same law, measured in two variables

`cos(θ/2)` is the overlap between two qubit states separated by Bloch angle
`θ`. Both programs measure coherence surviving as a **product of those
overlaps — one factor for every party that has acquired distinguishing
information.**

| | measured quantity | form | what the exponent counts |
|---|---|---|---|
| **Paper 1** | `T_xx = cos(χt/2)^(N−2)` | overlap^(N−2) | the `N−2` bulk spins, each learning partial *which-state* information about the boundary pair |
| **Paper 2** | `ρ_C[t,t'] = cos((t−t')·π/d)` | overlap¹ | *which-time* distinguishability between two clock records |
| **Paper 2 (IBM-1)** | witness `∝ cos(μ/2)^p` | overlap^p | environment modes coupled to the clock at strength `μ` |

The identification is exact, not analogical. At `d = 8`, adjacent clock records
have overlap `0.923880`; `cos(π/4 / 2) = 0.923880`. The clock-record overlap
*is* the half-angle state overlap, and the OAT boundary element is that same
overlap raised to the number of informed spins.

**One programme measures it against coupling angle. The other measures it
against time.** Entanglement decay and temporal distinguishability are the same
bookkeeping: information leaving a subsystem is simultaneously entanglement lost
and moments becoming distinguishable.

That is the two halves of the originating research question — *time emission*
and *quantum information theory* — turning out to be one quantity.

## 2. The matched null

Each programme has an exact structural zero. Each was **predicted from theory,
pre-registered, and confirmed on hardware** — these are the internal controls
that validated each apparatus, the results each line trusted most.

```
Paper 1, at χt = π          Paper 2, at d = 2
T_xx        = 0.000000      witness TVD = 0.000000
ρ₂          = I/4 exactly   state       = Bell pair
concurrence = 0.000000      entanglement = 1.0000 ebits
→ SEPARABLE                 → MAXIMALLY ENTANGLED
```

**The same observable reads exactly zero on a separable state and on a
maximally entangled one.**

This is not an argument that the coherence observable is entanglement-blind. It
is a measurement of it, at both extremes, using each programme's most trusted
control. No single reading of this quantity — however clean, however
replicated, however well it fits theory — carries information about whether the
underlying state is entangled.

Neither repository can demonstrate this alone. Paper 1's null shows the
observable vanishing on a separable state; Paper 2's shows it vanishing on a
maximally entangled one. **The demonstration requires both.**

## 3. The convergent resolution

Both programmes then hit the same wall from opposite directions, and neither
anticipated the other's version:

| | how it appeared | how it was resolved |
|---|---|---|
| **Paper 1** | entanglement is not identifiable from PTM anisotropy alone | `V_Q` over Haar-random measurement settings |
| **Paper 2** | no single local product-basis distribution certifies clock–system entanglement *(proved, then verified adversarially)* | multi-setting fidelity witness exceeding the exact separable bound `λ_max = ½` |

Paper 2 reached its version the hard way. It built two adversarial states
against its own results — a zero-entanglement product state that scored **4.2×
higher** on the coherence witness than the real history state, and a separable
state that scored **1.7× higher** on the joint witness built to repair it —
before a two-line theorem generalized the failure.

## 4. The combined statement

> **The survival of entanglement and the distinguishability of moments are the
> same quantity, measured in different variables: a product of state overlaps,
> one factor per informed party. That quantity is measurable to high precision,
> scales as predicted, and has exact structural nulls in both programmes — and
> it certifies nothing. The same reading of zero occurs on a separable state
> and on a maximally entangled one. Certification of quantum structure requires
> agreement across incompatible measurements, a conclusion two independent
> programmes reached from opposite directions.**

What follows from this is a **certification threshold**, and it is higher than
the observables usually offered as evidence for relational time can reach.
Conditional evolution is reproduced by a classical clock control. Local clock
coherence is beaten by a state with zero entanglement. Single-basis joint
correlation is beaten by a separable state. What survives is a multi-setting
witness — and, in Paper 2's final run, a single 3-qubit preparation certified
**entangled, stationary as a quantum ray, and internally evolving
simultaneously**.

## 5. What this does not claim

No gauge field, photon, or vacuum appears in either programme: OAT is a spin
model and the Page–Wootters clock is engineered qubits, so nothing here is a
statement about quantum electrodynamics. Neither programme claims that time in
nature is emergent, tests the Wheeler–DeWitt constraint, or realizes a physical
Page–Wootters universe. Both states are prepared by externally timed gates, and
both clock/system splits are imposed rather than derived.

The nearest legitimate route to field states is the **Dicke model**, which is
genuine light–matter coupling and reduces to OAT in the adiabatic limit
(`χ_eff = g²/ω_m`). It was characterized during the TPU campaign — returning
`K_eff = 1, γ_dom ≈ 0`, cleanly separated from OAT by damping alone — and
extending the cosine law into an actual field theory is a further paper, not a
claim in these two.
