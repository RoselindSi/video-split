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

    the middle module          everywhere it sees within MID_AUTHORITY_DEG
                               of its own optical axis
    elsewhere                  the valid camera with the smallest off-axis
                               angle

Blending two views of the same near hand produces two half-transparent hands.
The middle module holds an unbroken region across the whole working area, so
the hands are served by real pixels from one physical camera with nothing done
to them, and no boundary passes through them.

THE FIRST VERSION GOT THIS WRONG in a way worth keeping written down. It gave
each output column to whichever module was nearest in AZIMUTH, which split the
frame into equal thirds and put a seam at +30 degrees -- through the operator's
right hand, fingers on one side and forearm on the other, not lining up. But
cam3 alone spans 144 degrees and sees that direction at only 20 degrees off
axis, perfectly well. The seam was not forced by the optics; it was created by
the rule.

EXPOSURE IS MATCHED BEFORE SELECTION. The three modules disagree about
brightness and white balance -- the left strip of the first render is visibly
darker and bluer -- and that difference is about half of why a seam reads as a
seam. It is separable from geometry and costs one gain per module, estimated in
the overlap against the middle.

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

# The middle module keeps every pixel it can see at less than this angle off
# its own optical axis. It is deliberately large: cam3 alone spans about 144
# degrees, so it covers the whole default output, and the outer modules are
# only needed where its samples fall into the extreme fisheye periphery.
#
# The first version handed territory to whichever module was closest in
# AZIMUTH, which gave each module a third of the frame and put a seam at +30
# degrees -- straight through the operator's right hand. Seam placement is not
# a cosmetic choice when the seam lands on the thing being understood.
MID_AUTHORITY_DEG = 72.0

# OFF. The feather was a mistake worth recording: it blurs along the owner
# boundary, and that boundary is not a short straight seam but a long curved
# arc following the middle module's fisheye edge, so it painted a soft band
# down both sides of the frame. Once exposure is matched there is very little
# left for it to hide, and what it does instead is destroy real detail along
# an arc hundreds of pixels long.
FEATHER_PX = 0


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


def off_axis_deg(rig, camera, vcam):
    """Angle between each output ray and `camera`'s optical axis, in degrees.

    This is the right cost for choosing a source: it says how far into the
    fisheye periphery the sample sits, where resolution falls off and the
    calibration is least constrained. Azimuth distance to a module's centre --
    what the first version used -- ignores that a camera spanning 144 degrees
    is perfectly comfortable 20 degrees off its own axis."""
    cam = rig.cameras[camera]
    d = vcam.directions().reshape(-1, 3)
    cosang = np.clip(d @ cam.axis, -1, 1)
    return np.degrees(np.arccos(cosang)).reshape(vcam.height, vcam.width)


def match_gain(src, ref, mask):
    """Per-channel gain taking `src` onto `ref` over their shared pixels.

    The three modules do not agree on exposure -- the left strip of the first
    render is visibly darker and bluer than the middle -- and that difference
    is half of why a seam is visible. It is also entirely separable from
    geometry, so it is fixed here rather than waited on."""
    if mask.sum() < 500:
        return np.ones(3, np.float32)
    a = src[mask].reshape(-1, 3).astype(np.float32)
    b = ref[mask].reshape(-1, 3).astype(np.float32)
    keep = (a.mean(1) > 8) & (a.mean(1) < 247) & (b.mean(1) > 8) \
        & (b.mean(1) < 247)
    if keep.sum() < 500:
        return np.ones(3, np.float32)
    g = b[keep].mean(0) / np.maximum(a[keep].mean(0), 1e-3)
    return np.clip(g, 0.5, 2.0).astype(np.float32)


def render(rig, vcam, sources, depth_m, mid_authority_deg=MID_AUTHORITY_DEG,
           feather_px=FEATHER_PX, colour_match=True):
    """sources: {camera_name: image}. -> (rgb, owner).

    Selection, not blending. The middle module owns every pixel it sees within
    `mid_authority_deg` of its own axis; elsewhere the least off-axis valid
    module wins. Seams therefore sit at the outer edges of the field, where
    the content is far and the parallax small, instead of across the hands."""
    import cv2
    from src.rig.geometry import source_maps

    H, W = vcam.height, vcam.width
    mods = rig.modules
    mid_i = len(mods) // 2

    warped, valid, cost = {}, {}, {}
    for i, m in enumerate(mods):
        name = m.left.name
        if name not in sources:
            continue
        mx, my, ok = source_maps(rig, name, vcam, depth_m)
        warped[i] = cv2.remap(sources[name], mx, my, cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(0, 0, 0))
        valid[i] = ok
        cost[i] = off_axis_deg(rig, name, vcam)

    if colour_match and mid_i in warped:
        for i in list(warped):
            if i == mid_i:
                continue
            g = match_gain(warped[i], warped[mid_i],
                           valid[i] & valid[mid_i])
            warped[i] = np.clip(warped[i].astype(np.float32) * g,
                                0, 255).astype(np.uint8)

    # priority: the middle module first where it is comfortably on-axis, then
    # everything else by how far off-axis the sample is.
    big = np.full((H, W), 1e6, np.float32)
    score = {}
    for i in warped:
        s = np.where(valid[i], cost[i], big)
        if i == mid_i:
            s = np.where(valid[i] & (cost[i] <= mid_authority_deg),
                         -1000.0 + cost[i], s)
        score[i] = s
    keys = sorted(warped)
    stack = np.stack([score[i] for i in keys], 0)
    pick = stack.argmin(0)
    reach = stack.min(0) < 1e5
    owner = np.where(reach, np.array(keys, np.int8)[pick], -1).astype(np.int8)

    rgb = np.zeros((H, W, 3), np.uint8)
    for j, i in enumerate(keys):
        sel = reach & (pick == j)
        rgb[sel] = warped[i][sel]

    # SEAM DISAGREEMENT, the objective version of "look at the join".
    # Where two modules both reach a pixel, how different are the two views of
    # it? At a well-aligned seam they agree; at a misaligned one the same edge
    # lands in two places and the difference spikes. This is the number the
    # depth pass has to reduce, and eyeballing four renders cannot rank them --
    # 88% of the frame is owned outright by the middle module and is
    # pixel-identical across every depth assumption, so the only thing that
    # differs is two narrow bands.
    seam_stats = {}
    for j, i in enumerate(keys):
        if i == mid_i or mid_i not in warped:
            continue
        both = valid[i] & valid[mid_i] & (owner == i)
        if both.sum() < 200:
            continue
        d = np.abs(warped[i][both].astype(np.float32)
                   - warped[mid_i][both].astype(np.float32)).mean()
        seam_stats[mods[i].name] = (float(d), int(both.sum()))

    if feather_px > 0 and len(keys) > 1:
        # Feather ONLY across boundaries, and only where both sides exist.
        # Near the hands this never fires, because the middle module owns an
        # unbroken region there and there is no boundary to soften.
        edges = np.zeros((H, W), np.uint8)
        edges[:, 1:][owner[:, 1:] != owner[:, :-1]] = 255
        edges[1:, :][owner[1:, :] != owner[:-1, :]] = 255
        band = cv2.dilate(edges, np.ones((3, feather_px), np.uint8)) > 0
        blur = cv2.GaussianBlur(rgb, (0, 0), feather_px / 3.0)
        rgb = np.where(band[..., None] & (owner[..., None] >= 0), blur, rgb)
    return rgb, owner, seam_stats


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
    ap.add_argument("--mid_authority_deg", type=float,
                    default=MID_AUTHORITY_DEG,
                    help="the middle module keeps everything within this "
                         "angle of its own axis. Raise it to push the seams "
                         "further out; the first version's azimuth rule put "
                         "one through the operator's hand.")
    ap.add_argument("--feather_px", type=int, default=FEATHER_PX)
    ap.add_argument("--no_colour_match", action="store_true")
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

    rgb, owner, seam_stats = render(rig, vcam, sources, a.depth_m,
                                    mid_authority_deg=a.mid_authority_deg,
                                    feather_px=a.feather_px,
                                    colour_match=not a.no_colour_match)
    cv2.imwrite(a.out, rgb)
    cov = (owner >= 0).mean()
    print(f"wrote {a.out}  {w}x{h}  hfov {a.hfov_deg:.0f} deg  "
          f"assumed depth {a.depth_m} m")
    print(f"  filled {cov:.1%} of the frame")
    for i, m in enumerate(rig.modules):
        print(f"    {m.name} ({m.left.name}) owns {(owner == i).mean():6.1%}")
    mid = rig.modules[len(rig.modules) // 2]
    print(f"  {mid.name} keeps everything within {a.mid_authority_deg:.0f} "
          f"deg of its own axis")
    # How much would the middle module cover on its own? If the outer modules
    # only add a few percent, they are buying that at the price of two seams,
    # and the seams are the thing that damages a hand crossing them.
    from src.rig.geometry import source_maps
    _, _, mid_ok = source_maps(rig, mid.left.name, vcam, a.depth_m)
    print(f"  {mid.name} alone would cover {mid_ok.mean():6.1%}; the outer "
          f"modules add {(owner >= 0).mean() - mid_ok.mean():+.1%}")
    seam_cols = np.where(np.any(owner[:, 1:] != owner[:, :-1], 0))[0]
    if len(seam_cols):
        az = np.degrees((seam_cols / w - 0.5) * vcam.hfov)
        print(f"  seams at azimuth {', '.join(f'{x:+.0f}' for x in az[:8])} deg"
              + ("" if len(az) <= 8 else f"  (+{len(az)-8} more)"))
    if seam_stats:
        tot = sum(n for _, n in seam_stats.values())
        wavg = sum(d * n for d, n in seam_stats.values()) / max(tot, 1)
        print(f"  seam disagreement (mean |diff| where an outer module owns a "
              f"pixel the middle can also see):")
        for name, (d, n) in sorted(seam_stats.items()):
            print(f"    {name}  {d:5.2f} over {n} px")
        print(f"    weighted mean {wavg:5.2f}   <- LOWER IS BETTER ALIGNED")
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
