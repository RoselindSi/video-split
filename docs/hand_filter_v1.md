# Hand filter, V1 frozen

One page. Four conclusions, the configuration they justify, and what is still
owed.

## A. The admission bottleneck was resolution, not policy

At `imgsz=512` a colleague's hand across the bench is about fifteen pixels
wide. The detector finds it on every frame and scores it 0.53–0.58; a new
track needs 0.60; `demo_video` drops any detection without a track id. The
hand is on screen, found, and treated as absent — for up to sixty-seven
frames.

This was diagnosed by running the detector at floor 0.05 and chaining
detections into runs, then having a person label the long ones in the
[0.50, 0.60) band. Of 34 runs audited across three clips, 26 were real hands
and **all 26 were somebody else's**. Per-frame score does not separate them
from junk (real median 0.550, junk 0.535); run length does (`>=12` frames,
precision 0.944).

Two failure classes were separated and only one turned out to matter here:
frames with a low-scoring candidate (a policy problem), versus frames with no
candidate at any score (a recall problem). Over 18 randomly sampled clips the
median clip had **0%** of the second kind; one clip had 48% and was the sole
outlier.

## B. 1024 is the intervention, validated twice

Same admission policy, same tracker, same ownership model — only the detector
input changed.

**Rescue precision.** 75 tracks over six recordings, each judged once, with
each track carrying per frame whether 512 admitted it too. 38 tracks were
rescued (512 never admitted them, or admitted under half their frames):

```
real hands              37/38 = 0.974  [0.87, 1.00]
  of those, foreign     27/31 = 0.871  [0.71, 0.95]
not a hand               1/38 = 0.031
per recording           1.000 on five of six; 0.667 on R0828_172337 (n=3)
control (512 admitted)  37/37 real, zero false
```

**End to end**, on the same physical hands and the same frames, with the 512
arm given the better arm's ownership verdicts so the comparison is
conservative:

```
                              512                1024
foreign hand-frames covered   0.316              0.892   [0.870, 0.911]
wearer hand-frames covered    0.018              0.020
non-hand frames covered       0/2                0/2
foreign tracks reached at all 28/55 = 0.509      55/55 = 1.000
median delay to first cover   3 frames           0 frames
```

Recall of foreign hands nearly triples. The price is one additional frame of
false cover. Inference time does not move: 768 and 1024 both run 4–5s per 200
frames against 512's 4.4s.

Three alternative admission policies were ablated against the same audited
runs and lost. Offline retroactive promotion from 0.50 reached 24/26 real
hands but admitted **8/8** of the labelled non-hands; DeepSORT-style tentative
tracks reached 18/26 with a confirmation delay by construction; 1024 reached
22/26 with **0/8** false. The policies act on weak evidence; the resolution
change makes the evidence strong.

## C. The tracker is not the problem

Handoffs — a track ending and another starting within three frames in the same
place, which is one hand losing its name — were 0/6, 1/8 and 1/27 on three
clips. Zero of 34 audited tracks were judged `mixed`. Same-frame duplicate
detections ran about 0.1 per frame, an upper bound rather than a count.

So association gating, `MAX_LOST`, motion prediction and duplicate removal are
**not** current failure sources, and the ablations planned for them were not
run. Twenty-seven short tracks in one clip are twenty-seven separate
appearances, not fragmentation.

## D. Ownership: context is the representation, and it never hurts

The shipped hand-only classifier recalls 0.644 of foreign hands on the frozen
independent set. Its 31 misses have no visible pattern and 20 of them are
confident errors at P(owner) ≥ 0.70 — a feature ceiling, which is unsurprising
because a colleague's hand and the wearer's hand look the same.

A two-branch model (hand crop + a 2.5× context window + frame-normalised
geometry), trained on the 1014 hands that carry all three inputs against the
incumbent's 2533:

```
frozen test, 29 unseen recordings, 87 foreign hands
  hand-only    precision 0.889   recall 0.644
  both_geom    precision 0.89-0.96   recall 0.816-0.851 over three seeds
```

On 34 tracks judged individually, the four-way census is:

```
                ctx correct   ctx wrong
hand correct        22            0        <- context hurts nobody
hand wrong           9            3
```

The three `both_wrong` tracks are all in one recording, the one where the
wearer's hands and a colleague's share the frame at close range. Input
ablation on those three shows the hand branch driving the verdict toward
`owner` and the context branch failing to override it. Replacing the geometry
vector with a neutral one barely moves any output on this set, so geometry is
not the decision source here even though it was worth 0.056 f1 on dev.

Track-level pooling was measured and is worth +0.03 f1 against +0.39 for the
representation change, though it does take precision to 1.000. It is not part
of V1: it cannot help the `both_wrong` tracks, which are steadily wrong rather
than flickering, and pooling would make them more stable.

## The frozen configuration

```
detector          WiLoR YOLO, imgsz 1024, floor 0.25
admission         new track >= 0.60, continue >= 0.25
tracker           Hungarian with gating, MAX_LOST 5, motion prediction on
ownership         hand + context + geometry (own_ctx, --clf_ctx)
two-hand cap      on, with the state writeback (see debt)
mask              GrabCut on a window round the box
panorama          baseline three-view (the depth renderer is opt-in)
faces             YOLOv8-face at 0.35, no size cap
privacy           other -> blur
```

`imgsz=512` stays as the baseline to compare against.

## Owed

**The cap writes a structural decision into a hand's belief.** How many hands
are in a frame is not evidence about any one of them, and the writeback means
a hand demoted once needs a fresh crossing of `hi` to recover rather than
simply waiting for the third hand to leave. Removing it is the principled
change and it measured worse — track flips went 3.5 → 4.9 and 3.4 → 4.1 per
100 hand-frames — because the cap's per-frame verdict is itself unstable and
the writeback was latching that instability rather than creating it. The fix
is a hysteresis on the cap's own structural state, kept out of the belief.
Unwritten and untested, so what ships is the behaviour with numbers behind it.

**Six recordings decide the resolution change**, and one of them supplied the
only false rescue. The 0.60 asymmetry is untouched: an existing track
continues at 0.25 while a new one needs 0.60, and a hand that stays under the
bar at 1024 is stranded exactly as before. Resolution removed the failure from
the population measured, not from the design.

**Three tracks are steadily wrong under both ownership models**, all in one
recording, all where the wearer's and a colleague's hands are close together.
That is the next representation question and it is not a smoothing problem.

**`testpkg_2` has now been read twice** — once for the incumbent, once for the
context model. A third architecture scored on it makes it a selection set.
