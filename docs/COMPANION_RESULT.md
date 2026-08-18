# The one-budget result

*Stated formally, with the theorems attributed and the measured content
separated from them. The narrative version, with the full tables and the two
programmes' context, is the [synthesis repository's README](https://github.com/IllI/relational-entanglement-network); this document is the
claim itself and its scope.*

Everything below is reproducible in under a second, from the synthesis
repository ([`theory/verify_companion_result.py`](https://github.com/IllI/relational-entanglement-network/blob/main/theory/verify_companion_result.py)):

```bash
python theory/verify_companion_result.py
```

---

## The statement

> Every correlation magnitude measured by either companion programme is drawn
> on a single conserved unit, and that unit is spent twice over: once between a
> subsystem's local coherence and its entanglement with a partner, once between
> its entanglement with one partner and with another. A Page–Wootters clock is
> a good clock in proportion to the entanglement it spends on its own system.
> Therefore **no correlation magnitude can serve as a shared temporal reference
> between two good clocks** — not because the correlation is weak, but because
> it is already spent.

## The two identities, and whose they are

**Within a pair** — Jakob & Bergou, *Quantitative complementarity relations in
bipartite systems*, Phys. Rev. A **68**, 022107 (2003):

```
V² + P² + C² = 1
```

`V` single-party visibility, `P` predictability, `C` concurrence. The
relational-time programme's clock-marginal witness is `W = V/2`, and `P = 0`
for the family it uses because the clock populations are flat by construction.
The relation the programme derived independently, `(2W)² + C² = 1`, is that
theorem's `P = 0` slice. **It is not a new result and is not presented as one.**

**Across pairs** — Coffman, Kundu & Wootters, *Distributed entanglement*,
Phys. Rev. A **61**, 052306 (2000):

```
C²(A:B) + C²(A:C) + τ_ABC = 4·det ρ_A
```

For the family measured in IBM-12, `ρ_A` is maximally mixed and `τ_ABC = 0`, so
the two concurrences sum to exactly 1.

**The obstruction for Page–Wootters clocks specifically** is stated by Kuypers
& Rijavec, *Measuring time in a timeless universe*, Phys. Rev. D **112**,
063544 (2025), who resolve it by adding an interaction so the timer can read
the clock.

## What is actually new

1. **The two identities are one budget.** Run the verifier: the λ-family's
   visibility and the μ-family's `C(A:Sₐ)` print the identical curve
   (`1.000000, 0.923880, 0.707107, 0.382683, 0.000000`) from two different
   three-qubit constructions answering two different physical questions.
2. **Mixedness never creates headroom.** Under depolarizing noise at
   `p ∈ {0, 0.05, 0.2, 0.5, 1}`, neither total ever exceeds 1. The exhaustion
   is therefore not a pure-state artifact — it is *stronger* on hardware.
3. **Both trade-offs measured as continuous hardware curves**, which appears
   not to have been done:

   | | run | result |
   |---|---|---|
   | local coherence vs entanglement | IBM-11, 53 circuits, 9/9 gates | `r = −0.9573` |
   | own-system vs other-clock | IBM-12, 135 circuits, 6/6 gates | `r = −0.9833` |

   In IBM-12 both endpoints reach **exactly zero**: a clock maximally entangled
   with its own system has no measurable correlation with the second clock.
   That is the no-go, measured rather than argued.
4. **The exhaustion consequence**, i.e. that (1) covers the whole class of
   observables either programme measured, so the failure to find a shared-clock
   magnitude is structural rather than a limitation of the apparatus.

## Scope, stated as limits

- **The argument covers functions of pairwise correlation magnitudes.** That is
  every observable either programme measured. It does **not** cover genuine
  multipartite invariants — `τ_ABC` was zero by construction in the family used
  and is untouched by the argument.
- **It does not cover quantities that are not magnitudes.** See below.
- **It is substrate-blind.** The identities hold for *any* pure state, so
  nothing here is evidence for a common generative structure underlying
  different physical paradigms. Stated because that reading is tempting and
  wrong.
- **The identities are not the measurement.** Saturation is forced by
  construction. It is asserted in preflight and excluded from the gates in both
  runs, because measuring a quantity fixed by construction is a mistake this
  programme has published twice (IBM-6, IBM-8).
- **Concurrence from tomography is a biased estimator.** IBM-12's raw deficit
  bows at intermediate entanglement even in pure simulation. Only excess over a
  noise-matched Aer reference run through the same estimator is interpretable;
  a gate against the ideal `1.0` would have manufactured a result.
- **Nothing here is about time in nature.** Both states are prepared by
  externally timed gates and both clock/system splits are imposed rather than
  derived. Endpoints measured at exactly zero are statements about correlation
  between engineered registers, not about simultaneity.

## The open question

The geometric phase is **not a magnitude**. The Aharonov–Anandan phase is a
property of the path traced in projective Hilbert space and is
reparameterisation-invariant by construction — it depends on the loop, not on
the rate of traversal. That is precisely the invariance a shared time between
two clocks running at different rates would require, and it is the one property
no quantity in the budget has: concurrence and visibility are functions of the
instantaneous state, blind to the path that produced it.

Whether the AA phase is *also* on the budget is not known here. Sjöqvist's
two-particle geometric phase for entangled spins depends on the degree of
entanglement, so the phase is certainly not independent of it; what is unclear
is whether that dependence is a **trade** — a conserved total that spending on
one pair removes from another — or a functional dependence with no monogamy
structure. Only the first would close the door.

The run designed to answer it in either direction is
[`NEXT_RUN_PROPOSAL.md`](https://github.com/IllI/relational-entanglement-network/blob/main/docs/NEXT_RUN_PROPOSAL.md). It tests the invariance
directly rather than sweeping phase against entanglement, because the naive
sweep is confounded: `ρ_A` is maximally mixed at every `μ` in IBM-12's family,
and the marginals of `Sₐ` and `B` have Bloch lengths `sin²μ` and `cos²μ`, so a
phase swept against `μ` would track marginal purity through a closed form
already in the literature.

## Provenance

Both runs on `ibm_marrakesh`, 2026-08-17, calibration snapshot
`2026-08-17T13:17:21-05:00`. Job IDs, layouts, backend properties and raw shot
counts are archived in the relational-time repository under
`results/hardware/ibm11/` and `results/hardware/ibm12/`; every analysis
reproduces from counts without IBM access.
