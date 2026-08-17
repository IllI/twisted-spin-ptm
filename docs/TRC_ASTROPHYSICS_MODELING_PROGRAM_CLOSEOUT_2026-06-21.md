# TRC Astrophysics Modeling Program Closeout

**Date:** 2026-06-21  
**Status:** Grant-cycle closeout and next-cycle handoff  
**Canonical scope:** TRAPPIST-1e stellar-contamination modeling, DREAMS reproduction, paired TRAPPIST-1b/e correction, D-LinOSS architecture selection, and GLIMPSE-17775 spectral decomposition

## Executive result

The program produced a validated JAX/TPU experiment stack, repaired several data and model-contract failures, reproduced the qualitative need for the DREAMS per-visit Gaussian-process contamination model, recovered three clean public TRAPPIST-1b/e pairs, and established useful synthetic architecture and observing-design results.

It did **not** produce evidence for a TRAPPIST-1e atmosphere, methane, an LRD time anomaly, a gravitational-wave timing signature, or a distinct real-data D-LinOSS temporal operator.

The final model-selection result is:

- D-LinOSS-v0 is retired from JWST detector duty.
- Residual-first linear SSM and classical residual baselines are the validated detector path.
- MPS spectral compression improves missing-window recovery, but MPS + linear SSM has not yet earned a real-astronomy claim.
- DREAMS-style e-only per-visit GP modeling is the active TRAPPIST interpretation path.
- Current public b/e pairs are cadence-limited for wavelength-dependent transfer.
- GLIMPSE-17775 supports static radiative-history compatibility work only; its current control library is insufficient for a scientific promotion.
- Temporal-origin D-LinOSS modeling is a next-cycle simulator project, not a completed grant result.

## What scientific attributes were modeled

### TRAPPIST-1e Program 1331

The target was visit-variable stellar contamination in four JWST/NIRSpec PRISM transits. The detector-level tensors contained:

- integration-time spectra across wavelength;
- flux uncertainties and data-quality masks;
- absolute integration time and transit phase;
- visit identity and visit ordering;
- white-light, continuum, and visit-mean residualizations;
- injected spot/facula-like chromatic structure for sensitivity tests.

The intended astrophysical separation was:

`visit-variable stellar contamination + stable planetary transmission residual + noise`

The contamination attributes of interest were spot/facula color contrast, stellar activity-state drift, transit-chord dependence, flare structure, and visit-to-visit chromatic variability. No run used a trustworthy atmosphere or methane truth label.

### Paired TRAPPIST-1b/e Programs 9256 and 6456

The paired branch tested whether an airless or weak-atmosphere `b` transit could act as a nearby stellar-state proxy for `e`. The paired tensors contained:

- matched `b` and `e` spectra on a common wavelength grid;
- verified planet labels and phase labels;
- pair identity and measured b/e time offset;
- common masks, normalization, and flare handling.

Three clean pairs were recovered, with offsets of `4.26`, `4.42`, and `6.01` hours.

### WASP-39b positive control

WASP-39b ERS Program 1366 G395H data supplied a known high-SNR residual-injection benchmark. The tested target was a phase-localized, wavelength-localized CO2-window residual. This was an architecture and null-test validation, not a new molecular detection.

### GLIMPSE-17775

The public DDT Program 9223 G395M spectrum was converted into a static spectral ABI with:

- 768 inverse-variance-rebinned bins;
- rest-frame coverage `0.597-1.221 um` at externally fixed `z = 3.50102`;
- flux, uncertainty, validity mask, and line-agnostic continuum residual;
- 37 published line windows grouped into hydrogen recombination, helium, oxygen fluorescence, iron fluorescence, and forbidden/host families;
- velocity coordinates and line-core/wing masks.

This observable can test spectral covariance, line-family organization, scattering-wing structure, and radiative-reprocessing compatibility. It has no measured chronological axis. The Balmer break is outside the G395M rest-frame range and cannot be claimed from this ABI.

## The decisive pipeline gap

The first TRAPPIST experiments were run at the wrong reduction stage.

The DREAMS analysis operates on four extracted transmission spectra:

`transit_depth[visit, wavelength] +/- uncertainty`

Those spectra are obtained only after:

1. white-light transit and systematics fitting;
2. wavelength-channel light-curve fitting;
3. transit-depth extraction for each wavelength and visit.

The experimental models were initially trained directly on residualized `x1dints` integration flux shaped approximately `[visit, integration, wavelength]`. Raw integration flux retains static stellar continuum, instrumental structure, and adjacent-integration autocorrelation that are largely modeled or divided out during transmission-spectrum extraction.

Therefore the failed 25-500 ppm Program 1331 injections are **not** limits on DREAMS transmission-depth sensitivity. They are negative results for the tested detector-level objective.

## Experiment ladder and verdicts

| Stage | Observable or task | Result | Meaning |
|---|---|---|---|
| Program 1331 tensor/timebase | Native `INT_TIMES`, wavelength, phase, masks | Passed after repair | Data ABI and timing can be reproduced; invalid earlier time-aligned shard remains quarantined |
| D-LinOSS-v0 | Residualized integration flux | Retired | Scalar spectral input, soft wavelength priors, full-spectrum targets, and weak state persistence made the implementation degenerate |
| Residual-first oracle | Injected residual in corrected target space | Passed | Preprocessing can preserve a known differential residual |
| WASP-39b linear SSM | Strict CO2-window injected positive control | `PROMOTE LINEAR_SSM_POSITIVE_CONTROL` | Recovery `0.9897`; phase, wavelength, shifted-template, and false-positive controls passed |
| Program 1331 linear SSM | Spot/facula injection in detector-level residual flux | `DO NOT PROMOTE` | Observable-only recovery remained `0.0039-0.0136`; no sensitivity floor through 500 ppm in this representation |
| Equal-supervision synthetic rematch | Known latent and residual targets | `DO NOT PROMOTE` | Corrected D-LinOSS recovered synthetic state but beat linear SSM by only about `1.25-1.36%`, below the 5% gate; mask and leave-window controls failed |
| Fixed D-LinOSS-v2 real-data audit | WASP-39b strict residual injection; Program 1331 paired-copy and observable injections | `FAILS_REALDATA_POSITIVE_CONTROL` | WASP-39b recovery was `0.9828`, but linear SSM (`0.9997`) and smooth residual (`0.9887`) remained stronger and damping ablation was negligible. Program 1331 paired-copy recovery was `0.9999` but failed time-order and damping gates; observable-only recovery collapsed to `0.0118` |
| MPS spectral capacity run 0 | Masked 96-bin synthetic spectra | `DO NOT PROMOTE` | MPS raised D-LinOSS leave-window recovery from `0.4359` to `0.7462`, but mask dependence failed |
| MPS spectral capacity run 1 | Explicit mask channel, bond dimension 16 | `PROMOTE MPS_LINEARSSM_ONLY` | MPS + linear SSM leave-window `0.7882`; MPS + D-LinOSS `0.7525`; D-LinOSS failed margin and mask gates |
| DREAMS e-only GP reproduction | Released four visit-level spectra | `PASS_GP_REPRO_DIRECTIONAL` | Shared-spectrum + per-visit GP likelihood improved directionally over no-GP and per-visit-only models |
| b/e pair ABI | Programs 9256/6456 | Passed with 3 pairs | Paired data exist and labels/offsets can be made trustworthy |
| Naive b-proxy subtraction | `e - b` residual | `PROMOTE_GP_ONLY` | Scatter worsened; b is not a directly subtractable simultaneous stellar state |
| b-proxy + residual GP | Learned transfer with 3 pairs | `PROMOTE_GP_ONLY` | In-sample scatter improved, but leave-one-pair-out prediction and required nulls failed |
| Pair-count/power audit | 165,888 synthetic trials | Conditional requirement | State stability and time offset matter more than pair count alone |
| Pair cadence closeout | Actual `4.26-6.01 h` offsets | `PROMOTE_CLOSER_PAIRING_REQUIRED` | Only coarse scalar coupling was robust; wavelength-dependent transfer did not generalize |
| GLIMPSE-17775 data ABI | Static G395M spectrum | `PASS_DATA_ABI` | Static spectral decomposition is technically possible |
| GLIMPSE model/null smoke | Masked reconstruction and six null paths | `PASS_NULL_SMOKE` | Plumbing works; linear SSM hidden-window correlation `0.708` exceeded MPS + linear SSM `0.322` in the smoke |
| GLIMPSE control library | SWIRE AGN/starburst/host templates | `STOP_CONTROLS` | Continuum controls exist, but resolution-matched line-profile and independent dense-cocoon controls are missing |

## What worked and should be preserved

### Data and provenance

- Reproducible MAST/Zenodo product ledgers and checksums.
- Native integration-time handling and explicit quarantine of invalid time alignment.
- TTV/ephemeris-aware planet labeling for paired TRAPPIST observations.
- Fixed-shape NPZ ABIs with explicit masks, uncertainties, coordinates, and claim boundaries.
- Public GLIMPSE-17775 spectrum identification under Program 9223 rather than imaging Program 3293.

### Compute infrastructure

- JAX CPU/JIT smoke paths and eight-device TPU `pmap` execution.
- Strict null suites and fail-fast verdict files.
- VM-local JAX compilation-cache reuse during follow-up runs.
- TRC-compliant spot TPU queueing in allowed US and Europe zones.
- Compact result retrieval and explicit resource cleanup.

### Method discipline

- Residual-first target definitions.
- Predict-before-assimilate state updates.
- Matched-filter, smooth, GP, PCA, frozen/static, shuffle, and permutation controls.
- Synthetic positive controls before real-data claims.
- Explicit prohibition of atmosphere, molecule, time-anomaly, and black-hole-star claims when gates failed.

## Why D-LinOSS did not promote

The original implementation had four concrete defects:

1. wavelength bins were treated as independent scalar streams instead of one full spectral feature vector;
2. the supposed physical dampers were soft wavelength windows, not hard temporal pole constraints;
3. full-spectrum reconstruction rewarded the dominant static baseline;
4. the hidden state was overwritten rather than allowed to accumulate a trajectory.

Those defects were repaired. The repaired model demonstrated high synthetic in-distribution capacity, including robustness to imperfect physical frequency priors. It still did not earn practical promotion because:

- its advantage over an equal-supervision linear SSM stayed below 5%;
- mask-shuffle dependence stayed below the 20% gate;
- leave-window-out generalization failed or was unstable;
- MPS compression improved missing-window recovery, but the D-LinOSS arm still did not outperform MPS + linear SSM;
- on real GLIMPSE smoke data, linear SSM reconstructed held-out structure more strongly than MPS + linear SSM.

The final locked-form real-data audit removed the remaining uncertainty about whether the repaired model had touched JWST tensors. It did: the model recovered the exposed WASP-39b and paired-copy injections, but did not beat the promoted baselines, did not depend materially on learned damping, and recovered essentially none of the Program 1331 observable-only injection. No hyperparameter sweep or post-hoc tuning was used. This closes the fixed D-LinOSS-v2 detector audit without an atmosphere, molecule, stellar-state, or detector-promotion claim.

This is a model-selection result, not evidence that state-space modeling is scientifically inappropriate.

## DREAMS and paired-data conclusions

The released Program 1331 spectra reproduce the direction of the DREAMS result. Directional log likelihood improved from:

- no GP: `-1752.99`
- per-visit GP: `-1694.46`
- shared spectrum + per-visit GP: `-1670.58`

This supports the need for visit-specific contamination flexibility. It does not reproduce exact Bayesian evidence or establish an atmosphere.

The paired b/e concept remains physically plausible, but the existing public cadence does not support a flexible wavelength-dependent transfer. The power audits found:

- high-stability regimes can already work with three pairs;
- low-stability regimes remain poor even with twenty pairs;
- 2-hour synthetic offsets performed better than 4-8 hour offsets;
- more pairs alone do not solve state decorrelation;
- future observations should prioritize shorter offsets and independent activity-state measurements before simply increasing pair count.

The smooth-transfer follow-up explicitly fitted `alpha(lambda)` with a six-component low-rank basis and ridge regularization. Its all-pair descriptive fit had median `0.0918`, range `0.0635-0.0974`, and wavelength correlation `-0.841`. This apparent redward decline is **not** evidence for a spot/facula transfer law: one leave-one-pair-out fold drove the median coefficient to `0.94` and produced catastrophic predictive likelihood, while the aggregate hybrid likelihood remained worse than the e-only GP. The full coefficient curve is preserved in `results/dreams_repro_followup/gp_be_hybrid_alpha_lambda.csv` as a diagnostic artifact only.

A new factorization `alpha(lambda, delta_t) = a(delta_t) f(lambda)` was not fitted after closeout. With only three offsets and two training pairs in each held-out fold, separate offset dependence and wavelength shape are not stably identifiable without an externally fixed prior. Adding that model after seeing the failed smooth-transfer result would be post-hoc tuning; the preregistered power/cadence audit is the appropriate record of offset dependence.

## GLIMPSE-17775 boundary

The authorized question is:

> Can static spectral structure be organized into channels compatible with direct emission, host continuum, recombination, fluorescence, and dense-cocoon scattering/reprocessing?

The unauthorized question is:

> Does GLIMPSE-17775 exhibit a measured clock anomaly, gravitational-wave timing signature, or literal time distortion?

The current GLIMPSE branch cannot proceed to a scientific TPU sweep because the control library lacks independent, resolution-matched ordinary AGN, Fe II-rich AGN, and electron-scattering/dense-cocoon line-profile controls across the full rest-frame interval. SWIRE templates are adequate only for broad continuum alternatives.

## Claims supported by this grant cycle

1. A controlled JAX/TPU pipeline can reproduce synthetic and high-SNR residual positive controls while enforcing strict nulls.
2. The public Program 1331 products reproduce the qualitative need for per-visit GP contamination modeling after transmission-spectrum extraction.
3. Three clean public b/e pairs can be assembled, but current offsets do not support a robust learned wavelength-dependent stellar-transfer function.
4. MPS compression materially improves synthetic missing-window recovery; under the tested gates, MPS + linear SSM is preferred to MPS + D-LinOSS.
5. GLIMPSE-17775 has a valid static spectral ABI suitable for controlled radiative-history compatibility work once line-profile controls are available.

## Claims not supported

- TRAPPIST-1e atmosphere, methane, CO2, or planetary residual detection.
- A real LRD operational-time path or local clock.
- Gravitational-wave or spacetime timing anomaly in GLIMPSE-17775.
- Proof of a black-hole star, binary black hole, or quasi-star origin.
- A stable wavelength-dependent b-to-e stellar-contamination correction from the three current pairs.
- D-LinOSS superiority over the promoted linear SSM detector stack.

## Next-cycle opening proposal

The clean next research question is temporal-origin compatibility, not time distortion:

> Can a state-space model distinguish temporal signatures predicted by competing LRD-origin simulations, and can clock-bearing LRD observations be scored against those manifolds without confusing compatibility with detection?

The sequence must be:

1. **Oscillatory architecture gate.** Construct a simple, preregistered oscillatory task under equal supervision. D-LinOSS must beat linear SSM under time-order, reverse-time, frequency holdout, and noise controls before returning to astronomy.
2. **Generator development.** Independently validate dense-cocoon escape/reprocessing, binary-MBH inspiral/disk, quasi-star pulsation, ordinary AGN, and starburst temporal generators against their source literature.
3. **Origin-family benchmark.** Test time-order shuffle, reverse-time shuffle, parameter holdout, generator-source holdout, origin-family holdout, and ordinary-control false positives.
4. **Real-data gate.** Apply compatibility scoring only to LRD datasets with a genuine clock axis: repeated photometry or spectroscopy, measured lensing delays, reverberation-like lags, or velocity-resolved changes.

The quasi-star pulsation scenario is the most natural opening benchmark because it provides explicit long-range oscillatory modes. Its period families must come from an independently implemented physical generator, not hand-authored sine waves tuned to favor D-LinOSS.

## Canonical artifacts

Use this document as the grant-level index. Detailed branch authority remains in:

- [`README_TRAPPIST1E_DLINOSS.md`](../aq_trappist1e_dlinoss_stellar_state_residual_0/README_TRAPPIST1E_DLINOSS.md)
- [`trappist1e_dlinoss_failure_assessment.md`](../aq_trappist1e_dlinoss_stellar_state_residual_0/trappist1e_dlinoss_failure_assessment.md)
- [`TRAPPIST1E_LINEARSSM_1331_CLOSEOUT_0.md`](../aq_trappist1e_dlinoss_stellar_state_residual_0/TRAPPIST1E_LINEARSSM_1331_CLOSEOUT_0.md)
- [`aq_dlinoss_mera_spectral_capacity_test_0/README.md`](../aq_dlinoss_mera_spectral_capacity_test_0/README.md)
- [`aq_lrd_glimpse17775_mps_linearssm_radiative_history_0/README.md`](../aq_lrd_glimpse17775_mps_linearssm_radiative_history_0/README.md)

Machine-readable verdicts remain authoritative for exact metrics. The many run-spec Markdown files are historical implementation scaffolding, not current branch verdicts.

## Final disposition

The grant cycle closes with a validated infrastructure stack, a defensible negative result for D-LinOSS as the current JWST detector, a directional DREAMS GP reproduction, a cadence requirement for paired b/e observations, and a technically ready but scientifically control-blocked GLIMPSE static spectral branch.

No TPU resources remain active. The next cycle should begin with the equal-supervision oscillatory architecture gate and literature-validated temporal-origin generators.
