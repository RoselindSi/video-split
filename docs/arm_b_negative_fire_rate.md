# Arm B — detector behaviour on the boundary-enriched task-timing pool

EXPLORATORY. 39 of 69 events, 18 recordings, held-out peaks (the val
`--save_logits` dump from a head trained on `--train`). Nothing here is
confirmatory and the peak-blind 45-event alignment gold stays separate.

    positive            21   human asserts a task boundary
    no_action_change    10   nothing changed here
    phase_change_only    8   something changed and the frozen ontology says it
                             is not a task boundary

19 recordings absent — `recording_000230` and up, i.e. test_batch2 / part2,
which have never had logits.

## Fire rate against each arm's own null

The three arms do NOT share a chance baseline: negatives sit in recordings
with denser peaks, so read excess and ratio, never raw differences.

| window |  positive excess / ratio | no_action | phase_change_only |
|--------|--------------------------|-----------|-------------------|
| 0.5s   | **+0.55 / 8.86**         | +0.19 / 2.73 | **+0.34 / 3.12** |
| 1.0s   | +0.59 / 5.92             | +0.21 / 2.11 | +0.24 / 1.92 |
| 2.0s   | +0.54 / 3.45             | +0.19 / 1.61 | +0.22 / 1.55 |
| 3.0s   | +0.51 / 2.70             | +0.11 / 1.28 | +0.13 / 1.27 |
| 5.0s   | +0.37 / 1.84             | +0.08 / 1.15 | +0.28 / 1.47 |

At ±2.0s: positive 0.762 vs null 0.218, p = 0.0005. Neither negative arm
clears its own null (p 0.147 and 0.189, on n = 10 and 8).

**Both ratios peak at the TIGHTEST tolerance.** The positive arm reaches 8.86
at 0.5s: 62% of task boundaries have a peak within half a second against a 7%
chance rate. That is a sharply localised signal, and my first reading of this
result — that the detector localises to 2s but not 0.5s — was wrong, taken
from the raw rate rising with the window without checking it against the nulls.

## The two pools disagree, at the same tolerance

    this pool, 0.5s, held-out peaks          hit rate 0.62
    45-event pool, 0.5s, OOF peaks           hit rate 0.16 - 0.20
                                             median distance 2.5 - 3.0s

Same tolerance, same detector family, held-out peaks in both. Three to four
times apart. This is not a number to average; it is a fact about the pools,
and there are two plausible causes and they push the same way:

1. **This pool is boundary-enriched.** Events are here because the boundary
   audit picked them, which inflates the positive arm directly.
2. **The tightened task-level standard REMOVED the hard positives.** The
   re-check moved five amplitude and fold-to-wipe changes out of `point` and
   into `motion_phase_only`. What survives as a positive is a boundary with a
   real reset or idle gap — visually the easiest kind. The 45-event pool never
   had that filter applied.

So 0.62 is the hit rate on the easiest positive set this project can
construct, twice over. It is not evidence that the detector localises well in
general.

## What it does support

On task-level boundaries carrying a real reset or idle gap, the detector puts
a peak within 0.5s about 62% of the time against a 7% chance rate, on
recordings it never trained on. That is a statement about WHICH boundaries are
detectable, and it is the first timing measurement in this project to clear
its null.

## Hypothesis, n = 8

`phase_change_only` has its highest ratio at 0.5s too — 3.12, excess +0.34,
higher than `no_action_change` at every width below 2s. If that survives a
larger sample it says the detector fires at motion-phase changes the ontology
deliberately does not cut on, which is a disagreement about the ontology
rather than a failure to see. On eight events it is a hypothesis and the arm
does not clear its own null.

## What would settle it

Peaks on the test_batch2 / part2 recordings would take coverage from 39 to 69,
but the split membership of those recordings has to be checked first — if they
are in `--train`, the peaks are in-sample and the phase arm becomes
uninterpretable, since the stored annotations DO cut at some of the places now
called phase-only.
