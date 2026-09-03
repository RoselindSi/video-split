# StabStitch++ stereo-pair validation

This experiment is module-internal stereo (`cam1 + cam2`). It is not the
existing three-view experiment that uses `cam1 + cam3 + cam5`.

The upstream source is pinned to commit
`646dad01f2f1159aeca10bcef2fbb1416d658df7`. Keep it outside this repository;
the GPU host already has source and the six official checkpoints under
`/shared/models/StabStitch2`.

The GPU host uses OpenCV 5, which no longer accepts CUDA scalar tensors as a
`VideoWriter` frame size. Apply
`third_party/patches/stabstitch2-pair-validation.patch` to a working copy of
the upstream source. It converts the output dimensions to Python ints and
emits the two warped source layers plus validity masks before fusion. Those
layers are debug artifacts, not a production output. Do not modify the shared
`/shared/models` checkout.

## Export

```bash
python3 -m src.rig.stabstitch_pair export \
  --calibration /path/to/calibration.yaml \
  --video /path/to/cam12.mp4 \
  --pair cam12 \
  --start 3000 --n 120 \
  --out /workspace/stabstitch_pair/input
```

The result has the layout required by the unmodified official program:

```text
input/
  cam12_f003000_n000120_s01/
    video1/000000.jpg ...
    video2/000000.jpg ...
    rectified_pair.mp4
    manifest.json
```

`manifest.json` pins the source/calibration hashes, frame range,
rectification matrices, valid-pixel fraction, and a sparse epipolar residual.

## Inference

Run on the GPU host. The wrapper intentionally selects the slower, cleaner
official settings: `NORMAL` warping and `LINEAR` fusion.

```bash
/workspace/venv_rig/bin/python -m src.rig.stabstitch_pair run \
  --stabstitch_root /shared/models/StabStitch2 \
  --input /workspace/stabstitch_pair/input \
  --output /workspace/stabstitch_pair/output \
  --python /workspace/venv_rig/bin/python \
  --gpu 0
```

Do not start with all six views. Inspect the 120-frame `cam12` output first,
especially the near arm/hand and the white table edge. A useful result must
remove double edges without bending the table edge or making the hand change
shape from frame to frame.

## Same-frame alignment audit

```bash
python3 -m src.rig.stabstitch_pair_eval \
  --case /workspace/stabstitch_pair/input/cam12_f003000_n000120_s01 \
  --warped /workspace/stabstitch_pair/output/cam12_f003000_n000120_s01_warped.mp4 \
  --masks /workspace/stabstitch_pair/output/cam12_f003000_n000120_s01_masks.mp4 \
  --out_json /workspace/stabstitch_pair/pair_alignment.json
```

The meaningful decision number is `p90_ratio = after / before`. Below one
means that the learned warp aligns hard edges better than calibrated stereo
rectification alone. The tail and listed worst frames matter more than the
median: persistent high residuals there indicate that one smooth 2D mesh
cannot represent the scene's depth discontinuities. The `near` statistic is
computed separately on the lower half containing the wearer's arm and hands.
`ghost_risk_gt3_fraction` is the fraction of edge samples whose counterpart
is still more than three pixels away. With linear fusion, those samples are
expected to become visible double contours rather than one aligned edge.

## Result: 2026-09-03

Input was `cam12.mp4`, source frames 3000 through 3119, rectified to two
960x720 images. Both remaps had 100% valid samples. Sparse epipolar residual
was 0.64 px median and 2.44 px p90, so bad fisheye normalisation was not the
limiting factor.

The official `full_model_tra` checkpoints produced a 1090x798, 120-frame
video. End-to-end inference including debug-video writes ran at 7.5 fps on an
RTX PRO 6000; the normal output without debug layers ran at 12.6 fps.

```text
hard-edge Chamfer                 before warp    after warp
median, median over frames           4.78 px        2.32 px
p90, median over frames             28.82 px       16.43 px
near-half p90, median               >=32.00 px      17.39 px
near-half p90, worst                >=32.00 px      20.13 px
edges displaced >3 px, median          59.2%          43.9%
near edges displaced >3 px, median     65.5%          45.5%
after/before full-frame p90 ratio                    0.573
```

The warp removes about 43% of the full-frame p90 edge displacement, so the
model is doing useful alignment. It does not pass the seamless criterion:
the remaining tail is visible at the arm/hand contour, white bench edge, bin
edge, and other foreground/background boundaries. This pair is the rig's
easiest case (59.6 mm baseline and almost parallel optical axes), and pairing
the two eyes adds almost no field of view. Do not build the six-view pipeline
as three first-stage StabStitch++ pairs on the strength of this result.

The fused video also fails independently of the Chamfer improvement. Its
linear overlap fusion retains both warped colors at occlusion boundaries, so
the residual displacement becomes strong arm, sleeve, bench, and bin
ghosting. Exposure compensation or a wider feather cannot remove this: they
change the appearance of the two contours but do not choose which surface is
visible. Subsequent experiments must render from one texture owner per pixel
using depth and a z-buffer, and treat any multi-source color blend as a narrow
photometric transition only after geometric visibility has been resolved.
