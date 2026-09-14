"""Self/other hand ownership learned from drawn zones, kept apart from V1.

THE QUESTION. Can an appearance model trained only on zone-derived labels
decide whose hand it is as well as the deployed classifier (V1) -- and does
starting from V1's learned filters help it?

A TWO BY TWO, SO EVERY GAP HAS ONE CAUSE.

                                     trunk from ImageNet   trunk from V1
    A  cam3|cam4, 2.5x context window     A/imagenet          A/v1 (`ctx`)
    B  cam3|cam4, hand box padded 0.6     B/imagenet          B/v1 (`hand`)

Every input is the rectified cam3|cam4 stereo pair -- the canvas
`stabstitch_pair.stereo_rectify_maps` builds -- cut at the same coordinates
in both eyes and laid side by side. Labels, sampling, head, epochs and
aggregation are identical in all four; only the scale and where the trunk's
weights start differ. V1's trunks were fitted to panorama crops, so the V1
arms test its learned filters on this new input, not its old input.

WHAT IS SHARED WITH V1 AND WHY. The detections and track ids, and the render
they were cut from. The detector and tracker run before any ownership
decision in `own_dump` and nothing flows back into them, and the human gold is
keyed to those track ids, so sharing them is what lets the comparison isolate
the ownership decision.

WHAT IS NOT. No V1 module is imported anywhere in this package. No V1 output
column is read outside `evaluate.py`, which needs them for the baseline rows.
No V1 post-processing (EMA, owner cap, geometry prior). No V1 training labels.
V1 weights enter only through `model.build(init="v1")`, by checkpoint path,
into a trunk and nothing else. `tests/test_selfother_isolation.py` enforces
the import and column rules.

THE SPLIT RUNS AGAINST INTUITION ON PURPOSE. e2e_main2 is where V1's choices
(owner cap, geometry weight, track majority) were made against the 274-track
gold, so V1 scores optimistically there. fresh29 was selected as V1's frozen
held-out set. So the new arms train and select on e2e_main2, exactly where V1
selected, and every arm is scored once on fresh29.

Pipeline: labels.py -> crops.py -> train.py -> evaluate.py.
"""
