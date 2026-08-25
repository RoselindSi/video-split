# Morphology v1 — frozen 2026-08-24

A real learned NO_TRANSITION veto, measured against a perfect one on the same
candidates. It buys something and it does not reach the pre-registered bar.

## Setup

```
train   287 of 288 morphology labels over 125 recordings
        POINT 144/144 | NO_TRANSITION 101/101 | INTERVAL 25/25 | UNOBSERVABLE 17/18
        the one drop is recording_000007, which has no cached features
eval    36 detector out-of-fold recordings, 3707 frozen candidates
        TRAIN INTERSECT EVAL RECORDINGS = 0, asserted at startup
oracle  1316 false_mid_segment from the OOF error audit, 1316/1316 matched
```

Everything fitted — PCA, scale, class weights — saw training frames only. Only
the morphology head sent gradient; relation, offset, width and the visibility
heads were never trained and are not emitted.

## Result, at a 10% review budget

| release | score only | learned (best) | oracle | learned / oracle gain |
|---:|---:|---:|---:|---:|
| 1% | 79.0% | 72.7% | 63.4% | 40.4% |
| 3% | 66.5% | 58.4% | 36.3% | 26.8% |
| 5% | 56.8% | 44.6% | 24.8% | 38.1% |

`true lost` is the share of the 2124 recoverable boundaries destroyed to hold
review at 10% under that release cap. The learned head captures roughly a
third of what a perfect NO_TRANSITION veto would.

**The pre-registered GO bar was half the oracle gain, read on the 3% column.
26.8% does not meet it.** It also sits above the 25% STOP line, in a band the
criterion did not name — the same defect the driver A pre-registration had, and
it is recorded rather than resolved by reinterpretation.

## Why raising the threshold cannot help

```
p_nt    vetoed   precision vs oracle   true boundaries in the veto pool
0.5      1253            52.4%                 31.0%
0.7      1100            53.5%                 30.2%
0.9       879            53.9%                 29.8%
```

Against a pool base rate of 57.3% true, the veto pool is enriched by 27 points
— the head discriminates. But **its confidence carries no further
information**: discarding 374 candidates by moving the cut from 0.5 to 0.9
leaves the composition unchanged. A candidate at 0.95 is no likelier to be a
real NO_TRANSITION than one at 0.55.

So the failure is not calibration and not the operating point. Sweeping 0.91,
0.92, 0.93 has nothing to find.

## What this does and does not license

The evidence architecture is not the problem. 1316 of the 1583 wrong
candidates — 83% — are false_mid_segment, exactly the family a NO_TRANSITION
veto addresses, and the oracle converts that into 30.2 points of recovered
recall at a 3% release cap. The target is well chosen and the headroom is real.

What is insufficient is the learned representation. 120k parameters against 287
supervised events with a training loss near zero is consistent with a
generalisation gap, and the flat confidence profile is what that looks like at
inference.

**Frozen. Do not tune p_nt, the policy, or the release rules.** The threshold
sweep above is diagnostic and no operating point was selected from it.

## Next, and the two experiments must not be mixed

```
A   does a stronger visual representation fix the morphology head?
    same 287 labels, same 36 recordings, same 3707 candidates, same window,
    same grid, same model, same optimizer, same class weighting, same policy.
    ONLY the frozen visual features change.
    -> one row added to the table above

B   does it help the production detector?
    same temporal head, same grouped 5-fold, new logits.
    -> F1@1.0, candidate ranking, GT-boundary percentile, and above all
       signal_present_not_top, which is 86.0% of the current misses
```

**F1@1.0, not F1@0.5.** The tolerance has been 1.0s since 2026-08-19 and every
measurement in this document already uses it — the candidate matching, the
oracle audit join, the review-budget table. Writing F1@0.5 for experiment B
was a slip, and it is the kind that does not announce itself: the number would
have looked like a detector metric and silently answered a question about a
different tolerance than everything it sits next to.

The historical baseline it invites comparison with — global f1@0.5 0.299 — was
measured at the old tolerance and **cannot be compared to an F1@1.0**. If a
bridge to it is wanted, report both and label which is which; do not quietly
adopt 0.5 to make the comparison work.


A asks about the auditor's representation, B about the detector's. Running
them together produces a number that answers neither.

---

## Appendix — how the Qwen3.5 extraction settings were validated

The caches carry no manifest, so the settings the originals were extracted
with had to be recovered rather than read. Two of them came out of the data:

```
frames        100% retention at a uniform 0.50s spacing, so nothing was
              filtered -- which also proves the originals did NOT use the
              defaults, since --th_blur 100 discards two thirds
fps           2.0, from the spacing
```

`--max_pixels` leaves no trace: the patch grid is pooled away, so a different
resolution produces a same-shaped feature. It could not be recovered and it
decides what the ViT actually saw.

**The check that closed it: extract one recording with the OLD backbone under
the NEW settings and compare to its cached features.** If the settings match,
the two differ only by numerical noise.

```
recording_000005, (1303, 5760) both

relative error   0.002941
cosine           0.99999568
per frame        median 0.999999, min 0.999925, none below 0.99

per block        global 0.0017 | left 0.0020 | right 0.0022
                 center 0.0027 | spatial_max 0.0032
```

Every block agrees to within 0.3%, and `spatial_max` is the largest — the
signature of numerical noise amplified by a max, where a tiny difference flips
which patch wins. That is also where the max absolute difference of 384 comes
from, against a feature whose largest magnitude is 15616.

So the settings are verified rather than assumed, and the Qwen3.5 features
differ from the current ones only by the ViT weights.

**Generalisable: when a cache has no manifest, re-extract one item with the
known-good model under the candidate settings and compare. It converts an
assumption into a measurement for the cost of a single recording.**

---

# Experiment A — result, 2026-08-24. The backbone swap did not move morphology.

Same 3707 candidates, same detector scores, same oracle, same 287 labels, same
window, grid, encoder, optimizer and class weighting. Only the frozen visual
features changed, and only the global stream — the local hand-crop cache is
still the 8B one, so this is a partial swap and the reading below was fixed
before the numbers arrived.

| release | score only | old morph | Qwen3.5 morph | oracle | old / oracle gain | Qwen3.5 / oracle gain |
|---:|---:|---:|---:|---:|---:|---:|
| 1% | 79.0% | 72.7% | 74.4% | 63.4% | 40.4% | 29.5% |
| 3% | 66.5% | 58.4% | 58.2% | 36.3% | 26.8% | 27.5% |
| 5% | 56.8% | 44.6% | 44.6% | 24.8% | 38.1% | 38.1% |

Identical at 5%, 0.2 points apart at 3%, 1.7 points worse at 1%. Not a gain
too small to clear a bar — flat.

The confidence distribution DID change:

```
             median P(NO_TRANSITION)   >=0.5   >=0.9
old                          0.036      1253     879
Qwen3.5                      0.013      1022     694
```

The new features make the head markedly less willing to call NO_TRANSITION, and
the usable discrimination between a true boundary and a false_mid_segment is
unchanged. So the problem is not calibration either.

## What this closes and what it does not

It does not close experiment B. `F1@1.0 = 0.558` printed in that run comes
from `oof_logits.pt` — the OLD detector — and morphology predictions do not
enter it. Whether a Qwen3.5 backbone improves the DETECTOR's own ranking needs
its features put through the same temporal head and the same grouped folds, and
that has not been run.

It also is not comparable to the 0.381 reported for the val split. Both are the
old detector; the pools differ, and the OOF recordings carry 8.31 annotated
boundaries per 100 frames against the val split's 5.15. Precision and recall
rise with density mechanically.

## The bottleneck statement, narrowed

Before: "the learned representation is insufficient." That was too coarse. A
frozen visual-backbone intervention has now been run and morphology did not
move, so the evidence no longer points at frame-level visual features.

What remains between the frame features and the decision:

```
Qwen frame features
    -> +/-6s, 25-frame grid
    -> 120k TemporalEncoder
    -> 4-class cross-entropy on morphology
```

And what the system actually fails at is not classification accuracy. It is
that `E(true reset) < E(internal motion)` for the candidates that matter — 86%
of misses are `signal_present_not_top`. A cross-entropy on the class never
optimises that ordering, and a veto applied after ranking cannot repair it,
because a veto can only remove and the failure is a promotion failure.

**So the next intervention is the objective and where the ontology enters, not
the backbone.** That is level C: morphology inside the score, trained by a
within-recording ranking loss on exactly the pairs that fail.

**Frozen. Do not tune the Qwen3.5 morphology head** — no threshold sweep, no
hidden size, no dropout. Continuing would be architecture fishing on the
evaluation set that every arm above shares.
