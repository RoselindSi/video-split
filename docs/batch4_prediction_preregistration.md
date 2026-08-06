# Pre-registration v1 -- SUPERSEDED, kept as the record

> **Superseded by `batch4_prediction_preregistration_v2.md`.** The central
> argument below is wrong: it used `precision / prevalence` as evidence that
> discrimination was unchanged, and that ratio is itself prevalence-dependent.
> Reconstructing TPR and FPR shows a conditional shift as well
> (0.667/0.108 versus 0.478/0.149), and a prevalence-only prediction gives
> 0.826 against an observed 0.711. This file is left unedited because a
> pre-registration that gets rewritten after the fact is not one.

# What batch4 will show, and what each outcome means

Written 2026-08-06, **before batch4 is labelled**. batch4 exists and its
media is rendered; its `temporal_truth` column is empty. Nothing below was
chosen after seeing any batch4 outcome.

## The frozen artefact under test

`configs/c3_selective_policy_v2_relabel_v1_combined.json`, refit on the
corrected labels over 412 development events (174 from the original 145 pool +
238 from the relabelled batch3). Selected point:

```
B_global_local_agreement: P1 vs local agreement,
keep_above 0.75, min_reliability 0.5, reject_below 0.3
```

Development numbers, full taxonomy: AUTO_KEEP 82/412 at precision 0.915
[0.834, 0.958]; REVIEW 0.612; clean-binary auto-keep precision 0.974; sharp
false-reject 0.034.

## The prediction, and why it is not about the model

An absolute score threshold is sensitive to the prevalence of the population
it is applied to. Measured on the two development sub-populations at the same
P1 threshold of 0.75:

| population | base rate | precision @0.75 | recall | lift over base rate |
|---|---|---|---|---|
| original 145 pool | 0.745 | 0.947 | 0.667 | 1.27x |
| batch3, relabelled | 0.435 | 0.711 | 0.478 | 1.63x |

The model is not worse on batch3 -- its lift over the base rate is *higher*
there. Precision is lower because positives are scarcer. So the combined
development figure of 0.974 is an average dominated by the higher-prevalence
half, and it is not a property that transports to a population with a
different mix.

**Therefore, conditional on batch4's positive rate `p` (measurable only after
labelling):**

- **p >= 0.65** — the constraints should hold. Expect full-taxonomy auto-keep
  precision near 0.90-0.95. A breach here WOULD be evidence about the
  representation or about transportability across recordings.
- **p <= 0.50** — expect full-taxonomy auto-keep precision in the 0.70-0.85
  band and the 0.95 clean-binary constraint to break. **This outcome is
  predicted by prevalence alone and is NOT evidence that the representation
  failed.** The correct response is to change how the decision rule is
  defined, not to change the model and not to retune the threshold.
- **0.50 < p < 0.65** — indeterminate; report it, draw no conclusion about
  cause.

Also predicted, independent of `p`: the nested-selection diagnostic already
breaches on development (1/5 folds below the precision constraint, 2/5 above
the false-reject constraint), so a batch4 breach of the false-reject
constraint has prior evidence behind it and must not trigger retuning either.

## What is NOT predicted

Nothing here says the policy is good. It says which of two failure
explanations a given batch4 result supports. A pass at low `p` would be
genuinely surprising and would falsify the prevalence account above.

## Recording composition matters too, and is not controlled

batch4's recordings were sampled to exclude the 48 development and 35 batch3
recordings, so it is a clean recording-level held-out set. But the batch3
relabel showed that 23.3% of `gt_boundary` candidates carry no visible
boundary at all (`annotation_convention`), and that fraction is a property of
how the source data was segmented, not of the model. If batch4's candidate mix
differs -- more or fewer GT-derived candidates -- its base rate moves for that
reason alone. Report the candidate-source breakdown alongside `p`.

## The rule this exists to enforce

batch4 is spent once. 233 unused recordings remain and there is no batch6
after batch5. If a breach is read as "the model failed", the next round will
spend effort on representation and the prevalence sensitivity will still be
there for batch5 to rediscover. This file is here so that the reading is fixed
before the data can influence it.
