# Three-class student: what it takes to ship

Written **before any three-class student is trained or scored**, and committed
as its own file so the date in git is the date it was fixed. The two-class
student that ships today was accepted under criteria written the same way, and
the one time this project chose a configuration on the test set it had to say
so and hand the decision to a later batch.

## What changes

The classifier head goes from `nn.Linear(128, 2)` to `nn.Linear(128, 3)`:

    0  other      somebody else's hand        cover it
    1  owner      the wearer's hand           deliver it
    2  not a hand a part, a bench, a bin      deliver nothing, cover nothing

Nothing else moves. Same detector, same tracker, same `OwnHold`, same views
(`qwen.views`: resize to 1280x704, draw the box at width 4, crop 1.6x the box
from the **undrawn** image, 448x246 and 224). Geometric prior stays at 0 and
the two-owner cap stays off, as `distil_ablate` settled for the wide student.

## The data

| class | source | n | labelled by |
|---|---|---:|---|
| 0 / 1 | the existing distillation pool | 22,622 hands | Qwen Q1 + GPT-6, audited blind at 300 boxes (owner 99.8%, other 97.9%) |
| 2 | `neg_harvest` over 500 recordings | 3,372 | the hand-ness probe, audited blind at 140 boxes (98/100) |

Class 2 is written as **clean native frames plus boxes**, never as baked
views, because the harvest drew its green box before scaling and cut its crop
from the drawn copy — neither of which is what the student's transform does.

### The held-out set

494 of the 994 indexed recordings have been touched by nothing: batch 1 used
the first 100 and batch 2 the next 400 under the same seed. The held-out set
is drawn from those 494 and is read **once**.

## The criteria

Against `S_wide_g6` (the shipped two-class student) on the same detections and
the same frames, through the same post-processing.

**The two that must not regress.** These are the metrics the two-class student
was accepted on and they keep their meaning, with class 2 folded into "not the
wearer's" for the comparison:

1. **M2** — foreign-hand frames called the wearer's ≤ `S_wide_g6`'s
2. **M1** — own-hand flips per 100 pairs ≤ `S_wide_g6`'s
3. **G** — own-hand frames called the wearer's ≥ `S_wide_g6`'s − guard

**The new one, and it is reported by box size or it is not reported.**

4. **N** — on held-out boxes a person judged, the class-2 verdict's precision
   ≥ 0.90, reported in four width bands: `<150`, `150–200`, `200–340`, `340+`.

   The band below 150 px is the one that matters and the one the training data
   cannot speak to: **every class-2 example is ≥150 px**, because the probe
   that labelled them is right 98 times in 100 above that and 3 times in 55 at
   25 px. A student trained only on large non-hands knows nothing about small
   ones, and small is exactly where other people's hands are — a median 87 px.
   An aggregate N would average that blind spot away, so an aggregate N is not
   an acceptable report.

   **If the `<150` band cannot be measured** (too few judged boxes), the
   student ships with class 2 **disabled below 150 px** — the box is delivered
   and counted as before — and that restriction is written into the code
   rather than assumed.

**The control, because the third class comes from a different pass.**

5. **S** — class-2 rate on NEW-pool boxes the probe called hands ≤ 0.12.

   Classes 0 and 1 come from the existing pool and class 2 from a new pass
   over new recordings. If the two renders differ in any way a network can
   see, the model can learn "this came from the new pass" instead of "this is
   not a hand". JPEG quality is matched to `distil_prep` at 92 and the decoder
   is the same OpenCV, which is not proof. The probe's own rate on those boxes
   is 6.4–7.7%; a student reading provenance would push this far above it.
   4,668 control boxes are written alongside the negatives for exactly this.

**And one of them has to be strictly better.**

6. At least one of M1, M2, N is strictly better than the incumbent under a
   95% bootstrap over recordings — a three-class student that merely ties is
   a new failure surface for nothing.

## What is deliberately not required

**Class 2 recall.** Precision is required and recall is not, because the two
errors are not alike: calling a hand a non-hand removes it from the
deliverable, while missing a non-hand leaves today's behaviour in place. A
student that finds half the bench clutter at 0.95 precision is worth shipping;
one that finds all of it at 0.7 is not.

**Better ownership.** Class 2 is not supposed to improve owner-vs-other, and
if it appears to, the first thing to check is whether the split moved because
the hard cases left the population rather than because the model improved.

## Weighting

**No inverse-frequency weighting.** Class 2 is 13% of the training rows and
about 7% of reality; reweighting to balance is how the enrichment run pushed a
prior to 11x and took precision from 0.688 to 0.250. If the class needs help,
it gets more data, not a larger coefficient.

## The probe stays offline

Nothing here removes the 27B from the loop for the measurements it is already
doing. The student replaces it **in the shipped pipeline**, where 0.6 s a box
is not available; the probe remains what audits the student.


---

# Result: the first attempt failed S, and the reason was the carrier

Trained 2026-09-23. Dev looked fine -- ownership F1 0.958 to 0.963 across
three seeds, class-2 recall near 100% at ~97% precision. On the held-out pool
it called 97.9% of 6,703 boxes not-a-hand, in every width band.

**S, the provenance control, is what it failed.** The 4,668 control positives
written beside the negatives -- boxes the probe called hands, never trained
on -- come back 242 of 256 as class 2 through the training path itself. The
bar was 0.12 and the value is 0.945.

The carrier, not the labels:

    teacher rows, classes 0/1     1600x900    rendered panorama, render(0.6m)
    all five evaluation roots     1600x900    the same
    the negatives, class 2        1920x1520   raw fisheye camera half
    the held-out pool             1920x1520   the same

So class 2 was separable by geometry alone and the model took it. The labels
are not at fault: 178 of 180 blind judgements say the negatives are negatives.

**And a second thing, larger and older.** Every number that accepted the
shipped two-class student -- flips 0.04, foreign frames 27, own frames 99.47%
-- was measured on 1600x900 renders, while `demo_video --camera cam3` hands it
a 1920x1520 raw half. Aspect 1.26 against 1.78, squeezed to the same 1280x704.
The student has been running outside its training distribution on every cam3
run this project has made.

That does not invalidate the cam3 measurements themselves: coverage, the 17
confirmed false OWNER claims and the ownership vote were measured on cam3
output against human gold, not extrapolated from panorama. What it invalidates
is the evidence behind CHOOSING this student for cam3.

**Two ways out, and they are not equivalent.** Harvest the negatives from
rendered frames, matching the teacher pool -- limited by calibration, which
exists for 15 of 40 databags. Or move the whole pool to raw frames, which
means re-deriving the teacher labels and re-running every acceptance test, and
which is the direction deployment has already gone without anyone deciding it.
