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

## E. An independent test sampled the way the system now runs

Every earlier test package was collected with the detector at 512, so none of
them contains a hand that only a 1024 input proposes -- and the rescue audit
showed those are half the tracks and 87% of them foreign. Measuring V1 on a
512-sampled package measures a different system.

`testpkg_3` is 228 labelled hands over 43 recordings, sampled at 1024, with
**zero** overlap with training: 115 owner, 102 other, 11 skipped, 217
decisive.

```
                        n   other   precision  recall   f1     recall 95% CI
shipped hand-only      217    102     0.831    0.578   0.682  [0.481, 0.670]
V1 hand+context+geom   217    102     0.914    0.833   0.872  [0.749, 0.893]
+ target mask          217    102     0.925    0.843   0.882  [0.760, 0.901]
```

Both precision and recall improve, and the recall intervals do not overlap.
This reproduces on fresh, deployment-matched data what `testpkg_2` said, and
it is the number V1 is frozen on.

The target-mask model is **not** adopted. Its +0.010 f1 here is about one
positive example at this sample size, and development does not support it
(below). `testpkg_3` has now been read and must not be used for selection.

## F. What was tried against the near-hand failure, and did not work

The residue after V1 is close-range: the wearer's hand and a colleague's hand
in the same part of the image, the colleague's called the wearer's with high
confidence. Four explanations were testable and were tested.

**The owner mask eating the other mask.** Excluded by measurement: over 200
frames of the hardest clip, `veto_px / oth_px` has a median of 0.0000, a 95th
percentile of 0.0017 and no frame where the cover was fully cancelled.

**The two-hand cap.** It fired zero times on that clip, and the owner count
was below two on 157 of 200 frames -- so the second slot was simply free and
the cap could not detect a contradiction. The cap does not cause this error;
it fails to catch it. That distinction matters: it can only turn `owner` into
`other`, never the reverse.

**The context window not saying which hand is being asked about.** Two
statistics supported this before it was tested. The two `both_wrong` tracks
have a median of two hands inside their context window against one for every
other group; and among same-frame pairs with opposite ownership whose windows
actually overlap, the context encoder's embeddings have a median cosine of
0.600 with 21% above 0.9, against 0.274 for opposite-ownership pairs overall.
Both are small samples (2 tracks, 14 pairs).

**The window being clipped at the frame edge and stretched to a square.**
20.5% of samples lose more than a tenth of their intended window and 13.0%
come out past an aspect ratio of 1.3 -- and the two `both_wrong` tracks are
clipped 31.7% at ratio 1.46 against 0.0% and 1.00 for every other group.

The last two were separated by a 2x2, three seeds each, everything else held:

```
arm                             seeds              median   vs A
A  stretch + no target      0.868 0.860 0.835      0.860      —
B  letterbox + no target    0.766 0.729 0.777      0.766   -0.094
C  stretch + target mask    0.876 0.813 0.851      0.835   -0.025
D  letterbox + target mask  0.791 0.691 0.796      0.791   -0.069
   target mask as its own branch                   0.819   -0.041
```

**Letterboxing is harmful**, and not marginally: A's seed range [0.835, 0.868]
and B's [0.729, 0.777] do not overlap. The arithmetic that motivated it was
correct and the conclusion drawn from it was wrong. The most likely reading is
that the distortion carries real signal rather than a shortcut -- a window is
clipped precisely when the hand is against a frame edge, and the wearer's own
hands enter from the bottom edge.

**The target indicator does not help**, in any of three forms, and it fails
where it was aimed:

```
A's category            n    under C
both right             52    52 retained
context rescues        18    17 retained
context hurts           3     2 repaired
both wrong              2     0 repaired
```

Zero of the two tracks it was designed for. Read together with the geometry
attribution -- replacing the geometry vector with a neutral one barely moves
any output on these tracks -- the crop, the window, the marker and the
geometry have all now been eliminated as the missing ingredient.

## G. Track-level body-relation veto — tested, not adopted

A residual failure mode of V1 is that a close-range hand belonging to another
person can occasionally be classified as `owner`, particularly when the
wearer's own hand is absent or visually confounded.

A lightweight post-hoc relation veto was tested WITHOUT modifying the frozen
V1 ownership model. For tracks V1 predicts as `owner`, pose estimation
measures whether the target hand is persistently associated with a visible
person's wrist-elbow-shoulder chain. The rule was frozen before independent
validation existed:

```
relation_veto = OTHER
  if a complete arm chain is present in >= 60% of pose-evaluable frames
```

The denominator is POSE-EVALUABLE frames: a frame the pose pass could not read
is not evidence that the hand is unattached, and folding those in would
depress every support ratio by however often the reader failed. The veto is
asymmetric on purpose -- it may overturn `OWNER -> OTHER` and does nothing
else.

### Development result

```
false-owner tracks rescued              6/17 = 0.353
true-owner tracks incorrectly vetoed    0/19 = 0.000
veto precision                          6/6  = 1.000
```

Frame-level pose evidence was not usable at all: it rescued 53 frames while
damaging 109. Sustained arm-chain evidence was substantially more selective,
which is the finding that made the rule worth freezing. The sample was small
-- the Wilson 95% upper bound on the observed 0/19 owner false-veto rate was
about 0.17.

### Independent validation

Eight further recordings were sampled and every admitted track judged once.
Three `not_hand` tracks were EXCLUDED rather than folded into `other`.

That exclusion matters and is easy to lose. `track_pool.load_pkg` maps every
label that is not `owner` onto the same class, so `not_hand`, `mixed` and
`skip` arrive indistinguishable from a colleague's hand. It was harmless while
the corpus held one `not_hand` in a hundred tracks and wrong the moment it
held three: a veto that fails to rescue a mislabelled bench object is not a
veto failure, and counting it as one mixes PROPOSAL quality into a measurement
of OWNERSHIP quality.

```
validation set    19 owner tracks
                  14 other tracks
                   3 not_hand, excluded
```

V1 itself classified 12 of the 14 `other` tracks correctly -- track-level
recall 0.857, consistent with the 0.833 measured on `testpkg_3` -- leaving
only two false-owner tracks for the veto to rescue.

```
                        development        independent validation
false-owner rescue      6/17 = 0.353       0/2  = 0.000
owner false veto        0/19 = 0.000       1/19 = 0.053
veto precision          6/6  = 1.000       0/1  = 0.000

combined                rescue    6/19 = 0.316  [0.15, 0.54]
                        false veto 1/38 = 0.026  [0.00, 0.13]
                        precision  6/7  = 0.857  [0.49, 0.97]
```

The predefined deployment criterion was veto precision > 0.95. It was not met
under any of the three readings.

### Interpretation

The negative validation does NOT show that the arm/body relation is
uninformative. The development experiment showed that a persistent
wrist-elbow-shoulder relationship distinguishes some other-person hands from
the wearer's, and 6 of 6 vetoes there were correct.

The failure is operational rather than representational:

1. residual V1 false-owner tracks are rare;
2. so the veto has few opportunities to help;
3. a low false-veto rate on true owner tracks can still outweigh those few;
4. and the independent set did not reproduce development's zero owner damage.

On the validation set the veto fired exactly once, and that once was on a
hand that really was the wearer's. Net effect: minus one.

**The relation signal is real. The current evidence does not support using it
as an automatic production veto.** Establishing precision above 0.95 with a
useful lower bound needs a much larger independent sample of the rare
false-owner regime -- at roughly half a false-owner track per randomly drawn
recording, of order eighty recordings.

### Decision

The relation veto is NOT part of V1. Close-range `other -> owner` errors are a
documented residual failure mode. Future work here should collect more
independent examples of the rare false-owner regime BEFORE introducing a more
complex ownership model.

## The frozen configuration

```
detector          WiLoR YOLO, imgsz 1024, floor 0.25
admission         new track >= 0.60, continue >= 0.25
tracker           Hungarian with gating, MAX_LOST 5, motion prediction on
ownership         hand + context + geometry (own_ctx, --clf_ctx)
two-hand cap      on, with the state writeback (see debt)
relation veto     off (section G)
target mask       off (section F)
letterbox         off (section F)
mask              GrabCut on a window round the box
panorama          baseline three-view (the depth renderer is opt-in)
faces             YOLOv8-face at 0.35, no size cap
privacy           other -> blur
```

`imgsz=512` stays as the baseline to compare against, and `target_mask` and
`letterbox` are both off.

The remaining close-range ownership errors are treated as a known residual
failure mode rather than patched with additional unvalidated heuristics.

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
