"""The constant-depth wide render -- the baseline the depth pass must beat.

This warps each module's left eye into the virtual wide camera assuming every
pixel sits at one distance, and composites by AUTHORITY rather than by
blending. It is deliberately the simple version:

    a single depth is right at exactly one distance and wrong everywhere else,
    so near objects in the overlap will not line up between modules.

That misalignment is the thing the depth pass exists to remove, and without
this baseline there is no way to say how much it removed. Run this first, look
at the seams, then compare.

COMPOSITING IS A CHOICE, NOT AN AVERAGE. Each output pixel takes its colour
from ONE camera:

    inside the central ROI      the middle module, always
    elsewhere                   whichever module's axis is angularly closest

Blending two views of the same near hand produces two half-transparent hands.
The central ROI exists because that is where the hand-object interaction lives
and where a compositing artefact costs the most, so it is served by real pixels
from one physical camera with nothing done to them.

WHICH EYE. One eye per module, the left, chosen for consistency rather than
quality -- the modules are 60 mm pairs, so the choice shifts the virtual centre
by less than the module spacing. The depth-aware renderer will not make this
choice at all; it renders from the reconstruction.

Usage:
    python -m src.rig.render_wide \
        --calibration .../calibration.yaml \
        --video cam12=.../cam12.mp4 --video cam34=.../cam34.mp4 \
        --video cam56=.../cam56.mp4 \
        --frame 6000 --depth_m 0.6 --out wide.png
"""
from __future__ import annotations

import argparse
import os

import numpy as np

# Half-width of the region the middle module owns outright, in degrees of
# azimuth. Wide enough to contain the work surface and both hands at a normal
# working distance; narrow enough that the outer modules still supply the
# field of view they were added for.
CENTRAL_ROI_DEG = 22.0


def read_frame(path, index):
    """One frame by index. Hardware sync means the same index across the three
    files is the same instant -- measured at 0.0000 ms, so no timestamp
    matching is needed."""
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"no frame {index} in {path}")
    return frame


def split_halves(frame):
    """(left, right) = (lower-numbered camera, higher-numbered camera).

    Verified rather than assumed: rectifying cam12 both ways and running SGBM
    gives 52.2% valid pixels and 98.5% positive disparity with the left half as
    cam1, against 24.5% and 77.2% the other way, and the implied depths land at
    0.19-4.06 m, which is a work surface."""
    w = frame.shape[1] // 2
    return frame[:, :w], frame[:, w:]


def render(rig, vcam, sources, depth_m, central_roi_deg=CENTRAL_ROI_DEG):
    """sources: {camera_name: image}. -> (rgb, owner) with owner as an index
    into the module list, -1 where nothing reaches."""
    import cv2
    from src.rig.geometry import source_maps

    H, W = vcam.height, vcam.width
    xx = np.arange(W)
    az = np.degrees((xx / W - 0.5) * vcam.hfov)          # per-column azimuth
    mods = rig.modules
    mid_i = len(mods) // 2

    # angular distance from each module's own axis, per column
    mod_az = np.array([vcam.angles_of(m.left.axis + m.right.axis)[0]
                       for m in mods])
    dist = np.abs(az[None, :] - mod_az[:, None])          # [M, W]

    rgb = np.zeros((H, W, 3), np.uint8)
    owner = np.full((H, W), -1, np.int8)
    warped, valid = {}, {}
    for i, m in enumerate(mods):
        name = m.left.name
        if name not in sources:
            continue
        mx, my, ok = source_maps(rig, name, vcam, depth_m)
        warped[i] = cv2.remap(sources[name], mx, my, cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(0, 0, 0))
        valid[i] = ok

    central = np.abs(az) <= central_roi_deg               # [W]
    # priority order per column: the middle module first inside the ROI, then
    # by angular proximity. A pixel is written once, by the first module that
    # actually reaches it.
    for x in range(W):
        order = ([mid_i] if central[x] else []) + \
            [i for i in np.argsort(dist[:, x]) if not (central[x] and i == mid_i)]
        col_done = np.zeros(H, bool)
        for i in order:
            if i not in warped:
                continue
            take = valid[i][:, x] & ~col_done
            if not take.any():
                continue
            rgb[take, x] = warped[i][take, x]
            owner[take, x] = i
            col_done |= take
            if col_done.all():
                break
    return rgb, owner


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--video", action="append", required=True,
                    metavar="FILEKEY=PATH",
                    help="cam12=..., cam34=..., cam56=... APPEND one each.")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--depth_m", type=float, default=0.6,
                    help="the single distance this baseline assumes. 0.43 m "
                         "was the measured median on this rig; the render is "
                         "correct there and progressively wrong elsewhere.")
    ap.add_argument("--hfov_deg", type=float, default=150.0)
    ap.add_argument("--size", default="1600x900")
    ap.add_argument("--central_roi_deg", type=float, default=CENTRAL_ROI_DEG)
    ap.add_argument("--out", required=True)
    ap.add_argument("--owner_map", help="also write which module owns each "
                                        "pixel, as a colour map")
    a = ap.parse_args()

    import cv2
    from src.rig.calibration import RigCalibration
    from src.rig.geometry import VirtualWideCamera

    rig = RigCalibration(a.calibration)
    w, h = (int(x) for x in a.size.lower().split("x"))
    vcam = VirtualWideCamera.from_rig(rig, size=(w, h), hfov_deg=a.hfov_deg)

    files = {}
    for spec in a.video:
        if "=" not in spec:
            raise SystemExit(f"--video wants FILEKEY=PATH, got {spec!r}")
        k, p = spec.split("=", 1)
        if not os.path.isfile(p):
            raise SystemExit(f"{p} does not exist")
        files[k] = p

    # each packed file holds one module, low-numbered camera on the left half
    sources = {}
    for m in rig.modules:
        key = f"{m.left.name}{m.right.name[-1]}".replace("cam", "cam")
        key = f"cam{m.left.name[-1]}{m.right.name[-1]}"
        if key not in files:
            print(f"  no video for {m.name} ({key}); it will be missing from "
                  f"the render")
            continue
        left, right = split_halves(read_frame(files[key], a.frame))
        sources[m.left.name] = left
        sources[m.right.name] = right
    if not sources:
        raise SystemExit("no module had a video")

    rgb, owner = render(rig, vcam, sources, a.depth_m, a.central_roi_deg)
    cv2.imwrite(a.out, rgb)
    cov = (owner >= 0).mean()
    print(f"wrote {a.out}  {w}x{h}  hfov {a.hfov_deg:.0f} deg  "
          f"assumed depth {a.depth_m} m")
    print(f"  filled {cov:.1%} of the frame")
    for i, m in enumerate(rig.modules):
        print(f"    {m.name} ({m.left.name}) owns {(owner == i).mean():6.1%}")
    print(f"  central ROI is +/-{a.central_roi_deg:.0f} deg of azimuth, owned "
          f"outright by {rig.modules[len(rig.modules)//2].name}")
    print(f"\n  This is the CONSTANT-DEPTH baseline. Misalignment in the "
          f"overlaps is\n  expected and is what the depth pass has to remove; "
          f"look at the joins\n  between modules, especially on anything close "
          f"to the camera.")
    if a.owner_map:
        colours = np.array([[220, 90, 60], [70, 200, 120], [80, 130, 240]],
                           np.uint8)
        vis = np.zeros_like(rgb)
        for i in range(len(rig.modules)):
            vis[owner == i] = colours[i % 3]
        cv2.imwrite(a.owner_map, cv2.addWeighted(rgb, 0.6, vis, 0.4, 0))
        print(f"  wrote {a.owner_map}")


if __name__ == "__main__":
    main()
