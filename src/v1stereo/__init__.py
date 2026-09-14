"""V1, retrained on the rectified cam3|cam4 pair instead of the panorama.

THE HYPOTHESIS. The V1-initialised zone arms lost to ImageNet initialisation,
and the suspected reason is that V1's filters were fitted to six-camera
panorama crops, not to the stereo pair those arms read. If so, the same V1
trained on stereo pairs should (1) be a better V1 on its own task and (2) be a
better starting point for the zone arms.

THE ONLY THING THAT CHANGES IS THE IMAGE. Both retrained V1s use V1's own
architecture (`own_ctx.build("both_geom")`), V1's own labelled rows (exactly
what `own_ctx.load` returns for `otherpkg_1` + `trainpkg_T1`, the packages
`o1.sh` trained `own_ctx_best.pt` on), V1's split, optimiser, epochs,
augmentation and dev-F1 selection, three seeds each:

    V1p13   panorama hand crop + panorama 2.5x context   (V1's input)
    V1s13   stereo hand pair   + stereo 2.5x context pair (the new input)

A retrained panorama V1 is the control, not the shipped checkpoint: the
shipped one saw a slightly different row set and fourteen geometry features,
so a gap against it would mix the input with both. Rows a stereo pair cannot
be cut for (off the rectified canvas) are removed from BOTH arms.

THIRTEEN GEOMETRY FEATURES, NOT FOURTEEN. `hand_span` needs the 21 hand
keypoints, which the detection dumps never stored, so it cannot be computed
for any scored detection. It is dropped from both arms alike; the other
thirteen are rebuilt exactly from dump columns (`dir` is the unit vector of
`arm_angle`, the same wrist-minus-mean vector `own_label` normalises).

WRITTEN BEFORE RUNNING -- WHAT COUNTS. The priority is that the wearer's own
hand is recognised STABLY, so own-hand stability is primary and foreign-hand
recall is a guardrail:

    better  own-hand false-blur frame rate lower
            AND own-hand label switches per 100 consecutive own frames lower
            AND foreign-hand tracks missed no more than the comparison + 1

    Q1  V1s13 vs V1p13                          (does stereo make V1 better)
    Q2  zone arm A/B with V1s13 trunk vs ImageNet trunk  (is it a better init)

Per-frame predictions are scored raw -- no smoothing for anyone except the
`V1 deployed` reference row -- so a stability gap is the classifier's.

TWO TEST SETS, AND ONE OF THEM HAS BEEN LOOKED AT. `testpkg_2` is V1's frozen
frame-level set: 85 recordings disjoint from both training packages, e2e_main2
and fresh29, never used by any zone arm; it answers Q1 per hand. fresh29 gives
tracks, which stability needs, but its results for the zone arms were already
read, so a result on it here is evidence, not a clean verdict.

Pipeline: recut.py -> panocrops.py -> train.py -> (selfother.train for the
V1s13-initialised zone arms) -> evaluate.py.
"""
