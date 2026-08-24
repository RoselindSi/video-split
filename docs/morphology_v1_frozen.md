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
    -> F1@0.5, candidate ranking, GT-boundary percentile, and above all
       signal_present_not_top, which is 86.0% of the current misses
```

A asks about the auditor's representation, B about the detector's. Running
them together produces a number that answers neither.
