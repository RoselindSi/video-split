# StabStitch++ stereo-pair validation

This experiment is module-internal stereo (`cam1 + cam2`). It is not the
existing three-view experiment that uses `cam1 + cam3 + cam5`.

The upstream source is pinned to commit
`646dad01f2f1159aeca10bcef2fbb1416d658df7`. Keep it outside this repository;
the GPU host already has source and the six official checkpoints under
`/shared/models/StabStitch2`.

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
