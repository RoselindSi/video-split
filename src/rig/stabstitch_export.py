"""Feed the three modules to StabStitch++ as three rectified image sequences.

`test_online_tra_threeview.py` wants three directories of jpgs and says what
it expects of them: "video1 should overlap with video2, and video2 should
overlap with video3". That is this rig exactly -- module B sits between A and
C on the fan and shares its field with both -- so video2 is the MIDDLE module,
not the first one.

RECTIFIED, NOT RAW FISHEYE, AND THE REASON IS NOT TIDINESS. The network
resizes to 360x480 and predicts a warp mesh; nothing in its training set is
distorted the way a 130-degree fisheye is, and a mesh regressor asked to
absorb barrel distortion on top of parallax is being asked the wrong
question. Undistorting first is also the part this project can already do
correctly and has verified -- the KB4 model, the stereo rectification, the
0.0000 ms hardware sync -- so what reaches StabStitch++ is only the thing it
is actually good at and our constant-depth renderer was bad at: aligning two
already-straight images that see the same scene from 94 mm apart.

WHAT IT COSTS. A 130-degree fisheye flattened to a perspective image stretches
its corners past usefulness, so `fov_scale` trades field for sanity: about
100-110 degrees survives per module. Three of those still span most of the
176-degree union, and they overlap MORE cleanly than the raw views do -- the
overlap is where the stitch happens, and it is the part of a fisheye that
rectifies best.

FULL RESOLUTION IS WORTH EXPORTING. The script resizes a copy to 360x480 for
the network and keeps `img1_hr` at whatever it was given, applying the
predicted warp to that. Exporting small would throw away detail for nothing.
"""
from __future__ import annotations

import os

import numpy as np

# 4:3, because the network resizes to 480x360 and a different aspect would be
# squeezed on the way in and stretched on the way out.
OUT_SIZE = (960, 720)

# How much of the fisheye to keep. 1.0 is the calibrated field and stretches
# the corners badly; 0.6 keeps roughly the middle 100-110 degrees, which is
# where the overlap between neighbouring modules lives.
FOV_SCALE = 0.6

# Which module goes to which directory. video2 is the middle one BECAUSE the
# script requires video1-video2 and video2-video3 to overlap, and only the
# middle module overlaps both.
SLOT = {0: "video1", 1: "video2", 2: "video3"}


def rectify_map(rig, camera, size=OUT_SIZE, fov_scale=FOV_SCALE, balance=0.0):
    """Fisheye -> perspective sampling map for one camera."""
    import cv2
    cam = rig.cameras[camera]
    K, D = cam.K, cam.D
    newK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, (cam.width, cam.height), np.eye(3), balance=balance,
        new_size=size, fov_scale=fov_scale)
    return cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), newK, size, cv2.CV_32FC1)


def export(rig, videos, out_dir, start, n, stride=1, size=OUT_SIZE,
           fov_scale=FOV_SCALE, quality=95):
    """-> {slot: n_written}. Zero-padded names, because the script sorts them."""
    import cv2
    from src.rig.render_wide import split_halves

    caps, maps, dirs = {}, {}, {}
    for i, m in enumerate(rig.modules):
        key = f"cam{m.left.name[-1]}{m.right.name[-1]}"
        if key not in videos:
            continue
        c = cv2.VideoCapture(videos[key])
        if not c.isOpened():
            raise SystemExit(f"cannot open {videos[key]}")
        c.set(cv2.CAP_PROP_POS_FRAMES, int(start))
        caps[i] = c
        maps[i] = rectify_map(rig, m.left.name, size, fov_scale)
        dirs[i] = os.path.join(out_dir, SLOT[i])
        os.makedirs(dirs[i], exist_ok=True)
    if len(caps) != 3:
        raise SystemExit(f"need all three modules, have {sorted(caps)}")

    written = {SLOT[i]: 0 for i in caps}
    for k in range(n):
        frames = {}
        for i, c in caps.items():
            for _ in range(stride - 1 if k else 0):
                if not c.grab():
                    frames = {}
                    break
            ok, img = c.read()
            if not ok:
                frames = {}
                break
            frames[i] = split_halves(img)[0]          # left eye
        if len(frames) != 3:
            break
        for i, img in frames.items():
            mx, my = maps[i]
            und = cv2.remap(img, mx, my, cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT)
            cv2.imwrite(os.path.join(dirs[i], f"{k:06d}.jpg"), und,
                        [cv2.IMWRITE_JPEG_QUALITY, quality])
            written[SLOT[i]] += 1
    for c in caps.values():
        c.release()
    return written


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--video", action="append", required=True,
                    metavar="FILEKEY=PATH")
    ap.add_argument("--out", required=True, help="the SAN is /workspace")
    ap.add_argument("--start", type=int, default=3000)
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--width", type=int, default=OUT_SIZE[0])
    ap.add_argument("--height", type=int, default=OUT_SIZE[1])
    ap.add_argument("--fov_scale", type=float, default=FOV_SCALE,
                    help="1.0 keeps the whole calibrated field and stretches "
                         "the corners past use; 0.6 keeps the middle 100-110 "
                         "degrees, which is where the overlap lives.")
    a = ap.parse_args()

    from src.rig.calibration import RigCalibration
    from src.rig.class2_census import _check_space
    _check_space(a.out)
    rig = RigCalibration(a.calibration)
    videos = dict(s.split("=", 1) for s in a.video)
    size = (a.width, a.height)

    print(f"{a.n} frames from {a.start}, stride {a.stride}, "
          f"{size[0]}x{size[1]}, fov_scale {a.fov_scale}")
    for i, m in enumerate(rig.modules):
        print(f"  {SLOT.get(i,'?'):7s} <- {m.name} ({m.left.name})"
              + ("   <- middle: it must overlap BOTH neighbours" if i == 1
                 else ""))
    w = export(rig, videos, a.out, a.start, a.n, a.stride, size, a.fov_scale)
    print(f"\n  {w}")
    tot = sum(os.path.getsize(os.path.join(dp, f))
              for dp, _, fs in os.walk(a.out) for f in fs)
    print(f"  {tot/1e6:.0f} MB -> {a.out}")
    print(f"\n  next:\n    cd /shared/models/StabStitch2/Full_model_inference/"
          f"Codes\n    python test_online_tra_threeview.py \\\\\n"
          f"      --video1_path {a.out}/video1/ \\\\\n"
          f"      --video2_path {a.out}/video2/ \\\\\n"
          f"      --video3_path {a.out}/video3/")
    print("\n  video2 is the MIDDLE module. The script requires video1 to "
          "overlap video2\n  and video2 to overlap video3, and only the "
          "middle module overlaps both.")


if __name__ == "__main__":
    main()
