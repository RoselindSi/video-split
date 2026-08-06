# Where the system actually stands, 2026-08-06

Three things that had been treated as established turned out not to be. They
look like three setbacks and they are one: the evaluation apparatus was
concealing the system's real state in three different places, and each fix
moved the headline numbers the same way.

## What the numbers were, and what they are

| | as reported before | corrected |
|---|---|---|
| REVIEW rate (full taxonomy) | 0.612 | **0.811** |
| automatic coverage | 0.388 | **0.189** |
| AUTO_KEEP precision, full taxonomy | 0.915 | **0.829** |
| AUTO_KEEP precision, clean binary | 0.974 | 0.967 |
| sharp false-reject | 0.034 | **0.000** |
| feasible operating point exists | yes, with slack | only after adding a keep-only option |

The one number that improved is sharp false-reject, and it improved because
the new operating point does not auto-reject at all.

## 1. Over a third of the batch3 labels were wrong, and they were mine

240 batch3 subtypes were converted from a sheet whose `temporal_truth` column
Claude filled by reading 2.5 fps contact sheets. Transcript evidence
identifies 90 of them; a blind human re-annotation of those 90 changed **58**.

The error was not mostly about the sign. Only 17 of the 90 flipped inside the
clean binary; **25 left the clean set entirely**, most becoming
`annotation_convention` -- the GT segmentation cuts there and nothing visible
happens. The population itself was shaped by the bad labels, not just its
targets. batch3's base rate moved 0.310 -> 0.435, which is the exact quantity
the batch3 held-out failure was diagnosed on.

Correcting them raised all-clean AUROC 0.778 -> 0.804 against a random-flip
null of -0.026 (p = 0.000), and 72% of that gain was on events whose own
labels never changed, i.e. the model learning from cleaner supervision.

**Unresolved:** 150 batch3 labels have no provenance trace either way. A random
30 would estimate their error rate for a fraction of the cost of relabelling
them all.

## 2. The nuisance events cannot be detected automatically

19% of all labelled events carry no decidable visual boundary. Removing them
upstream was quoted for two rounds as worth ~10.7 points of review. That
arithmetic assumed they could be **identified**, and they cannot.

A gate over global, local and reliability features reaches AUROC 0.793 for
`annotation_convention` + `camera_or_viewpoint_shift`, and at a
0.95-precision threshold chosen out-of-fold it removes **2 of 55** at
precision 0.500 while discarding one real boundary. The observability target
is worse and fold-unstable: removals `[7, 0, 0, 0, 0]`.

So the decomposition of REVIEW into nuisance / recoverable / genuinely hard
remains diagnostically true and operationally unavailable. Those events keep
going to a human, and `0.612 - 0.107` was never a reachable number.

## 3. Every previous operating point was selected on scores that were not comparable

`c3_local_eval` fits on either the original pool or batch3, never both, so the
two development pools' scores came from **two separate fits**. Their score
levels differed accordingly, and a single global threshold therefore sorted
partly BY POOL -- toward the pool with prevalence 0.745 rather than 0.435.
Part of what looked like a high-precision auto-keep region was the threshold
picking a population, not picking easy events.

Fitting both pools once removes the separation, and with it the feasible
region: on consistently-scaled scores **no candidate in the pre-registered
space satisfied the constraints**, and full-taxonomy auto-keep precision read
0.829 rather than 0.915. Restoring the non-clean taxonomy rows (299 -> 370
events) changed nothing, so the cause is the scoring path and not the
population.

The infeasibility had a narrow cause worth stating: the closest candidate met
auto-keep precision at 0.967 and failed only on sharp false-reject, 0.120
against a 0.05 limit. Every one of the 144 candidates was **forced** to
auto-reject -- `reject_below` ranged over [0.2 .. 0.5] with no "do not reject"
option anywhere in the space. Adding one (v4) changes no constraint and is
strictly more conservative, and it makes the search feasible at 19% automatic
coverage.

## The operating point that survives

```
A_p1_threshold_with_reliability_abstention
  score P1 (global) alone, keep_above 0.95, min_reliability 0.7,
  reject_below -1.0  (never auto-rejects)
```

- 60 of 299 clean events auto-kept, precision 0.967 [0.886, 0.991], 2 false keeps
- **70 of 370 auto-kept under the full taxonomy, precision 0.829, 12 wrong keeps**
- sharp recall after automation 1.000; same-action false-accept 0.016
- REVIEW 0.811; slack 1 more false keep, 8 more false rejects

The clean-binary figure reads 0.967 and the full-taxonomy one 0.829 because 12
auto-kept events are not clean boundary decisions at all: 5
`gradual_phase_transition`, 2 `annotation_convention`, 1
`camera_or_viewpoint_shift`, 2 `ambiguous`. Two thirds of the `ambiguous`
class is being auto-accepted. **0.829 is the number that describes what a user
would experience.**

Nested selection re-runs the whole choice inside training folds: median
held-out auto-keep precision 0.864, with **3 of 5 folds below the 0.95
constraint**. A batch4 breach is therefore expected and has evidence behind
it, not a prediction.

## What this does not overturn

The relabelling comparison was measured on separately-fitted scores on both
sides, so the *improvement* stands (delta +0.028 on all clean events, null
-0.026, p = 0.000). What does not survive is the absolute claim that the
policy had become feasible with slack; that rested on the scale artefact.

The REVIEW decomposition stands as a description: of 252 REVIEW events under
the old partition, ~23% had no decidable boundary, ~42% were real boundaries
the model missed, ~35% genuinely hard. Only the *actionability* of the first
group was refuted.

## What is not yet blocked

batch4 is untouched and unlabelled. It now has, for the first time, an
artefact worth testing: a frozen scorer covering both pools with cache hashes
and commit recorded, and a policy selected on scores from that scorer's own
path. Neither existed a week ago -- nothing in the repository persisted a
fitted model at all, so any batch4 result would have come from a model fitted
with batch4's own labels.

The pre-registration (`batch4_prediction_preregistration_v2.md`) predicts by
decomposition rather than attribution: observed prevalence, observed TPR/FPR,
prevalence-only expected precision, and the residual gap. It needs updating
for the v4 operating point.

## The honest one-line summary

After removing three sources of measurement error, the system safely automates
**19% of candidates at 0.829 precision** and sends 81% to a human, and its
selection procedure already breaches its own precision constraint on 3 of 5
development folds. Everything previously reported above that was measured
through one of the three errors.
