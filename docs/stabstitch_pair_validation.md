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

## Replacement experiment: Fast-FoundationStereo

StabStitch++ is retired from the RGB path for this rig. The replacement pair
experiment uses NVIDIA's official
[Fast-FoundationStereo](https://github.com/NVlabs/Fast-FoundationStereo)
implementation and the fixed-resolution 576x960, eight-iteration ONNX model.
The source archive was pinned at
`a290ba04c1b3ad1ec41a33974a157b2917b624d4`; the model SHA-256 is
`5b6c41cf5677055216b9726a6e7dcf3cc3316fd257054bcb6110e8e56b45a45a`.
The upstream code and weights are research-only/non-commercial; this is a
technical validation, not a production licensing decision.

Run dense disparity on the already rectified pair:

```bash
PYTHONPATH=/workspace/stabstitch_pair/ffs_runtime:/workspace/tr1/vs \
/workspace/venv_rig/bin/python -m src.rig.fast_foundation_stereo \
  --case /workspace/stabstitch_pair/input/cam12_f003000_n000120_s01 \
  --model /workspace/stabstitch_pair/ffs/model/model.onnx \
  --out /workspace/stabstitch_pair/ffs/output_full_v1 \
  --cuda_lib_dir /workspace/tr1/venv_cpu/lib/python3.12/site-packages/nvidia/cu13/lib \
  --cudnn_lib_dir /workspace/tr1/venv_cpu/lib/python3.12/site-packages/nvidia/cudnn/lib
```

GPU mode is strict: if CUDA fails to load, the runner raises instead of
silently falling back to the 15-second-per-frame CPU path. On the 120 frames,
after warm-up, inference ran at 96.4 ms median and 126.1 ms p90 per stereo
pair. Valid metric depth was 99.56% median and 98.04% in the worst frame.

Temporal stability is scored after transporting the previous disparity into
the current left image with backward optical flow and rejecting pixels whose
forward/backward flow is inconsistent:

```bash
python -m src.rig.fast_foundation_stereo_eval \
  --case /workspace/stabstitch_pair/input/cam12_f003000_n000120_s01 \
  --disparity_dir /workspace/stabstitch_pair/ffs/output_full_v1/disparity \
  --out_json /workspace/stabstitch_pair/ffs/output_full_v1/temporal_metrics.json
```

```text
flow-consistent support, median                    99.38%
aligned disparity change, median                   0.114 px
aligned disparity change p90, median over pairs    0.705 px
pixels changing more than 3 px, median              1.92%
pixels changing more than 3 px, p90                  3.39%
pixels changing more than 3 px, worst                5.70%
worst current frames                 60,64,30,45,62,48,46,90
```

The worst frames coincide with real hand/finger motion and changing
occlusions; sampled depth boundaries remain attached to the arm and hand.
This passes the geometry-layer gate, but it does not by itself prove a
seamless panorama.

The render contract is therefore now explicit:

1. Each module's right eye is correspondence evidence only.
2. Only the three module-left images may supply RGB texture.
3. Their measured coloured points are forward-projected into the virtual
   camera.
4. A z-buffer resolves different surfaces; candidates within 2 cm of the
   nearest surface use the most on-axis camera with a deterministic tie-break.
5. Every output pixel copies one physical camera colour. RGB averaging is not
   permitted.

`depth.module_depth(..., matcher=...)` and `wide_depth.wide_rgbd(...)` carry
this path. `DepthAwarePanorama(texture_mode="module_left")` also prevents
cam2/cam4/cam6 from re-entering the legacy inverse-warp compositor. A complete
six-view video test still requires synchronized `cam12.mp4`, `cam34.mp4`, and
`cam56.mp4`; the current validation host only has the exported cam12 pair.
