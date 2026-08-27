"""Photometric matching and a fixed, gated soft transition. Geometry unchanged.

The seam audit put numbers on what is and is not wrong with the geometric
renderer, and the answer was narrow:

    TEMPORAL   ownership map bit-identical on 40/40 frames      pass
    excess     +0.024 .. +0.038  no structural fracture         pass
    stereo     97.7% of the field has depth                     pass
    ratio      1.23 .. 1.30  the join is visible as a line      FAIL

A visible line with no fracture under it is a PHOTOMETRIC failure, not a
geometric one, so nothing here touches the geometry. `render_wide.render`
stays exactly as it is and remains the baseline this is measured against.

TWO THINGS ARE FIXED, AND "FIXED" IS THE OPERATIVE WORD.

1. The gain-only, PER-FRAME colour match becomes a gain AND bias fitted ONCE
   over sample frames and then held for the whole recording. The per-frame
   version was a real defect hiding behind a passing metric: the audit checked
   that the ownership map never changes and it never does, but the colour
   correction applied to two thirds of the frame was being re-estimated on
   every frame, so the mapping drifted even where the geometry was frozen. A
   downstream video model sees that drift; the audit did not look for it.

2. The hard argmin over off-axis cost becomes a SOFTMAX over the same cost.
   The weights therefore depend only on rig geometry and the virtual camera --
   constant per pixel for the whole recording, like the argmin was -- so the
   transition is smooth without becoming content-adaptive. A dynamic seam
   would score better on a single frame and would show the downstream model a
   moving boundary that no real camera produces.

AND THE BLEND IS GATED, BECAUSE AVERAGING TWO VIEWS OF A NEAR HAND MAKES TWO
HANDS. Where the two sources disagree by more than `gate`, the pixel reverts
to the hard geometric winner. Disagreement is measured on the images rather
than on depth: it is a proxy, but it fires on exactly the cases that matter --
parallax on near objects, and disocclusion where one camera sees a surface the
other does not.
"""
from __future__ import annotations

import numpy as np

# Softmax temperature over the off-axis cost, in degrees. Larger is a wider,
# gentler transition. 6 puts most of the crossfade inside about 12 degrees.
BLEND_TEMP_DEG = 6.0

# Above this mean absolute channel difference (0-255) the two sources are not
# looking at the same surface, and blending them would ghost.
GATE_ABS_DIFF = 40.0

# Frames sampled to fit the photometric mapping. It is a 2-parameter fit per
# channel over hundreds of thousands of overlap pixels; more frames buy
# robustness to a passing shadow, not precision.
FIT_FRAMES = 12


def fit_photometric(pairs, min_px=5000):
    """[(src_img, ref_img, overlap_mask)] -> (gain[3], bias[3]).

    Least squares of ref = gain * src + bias per channel, pooled over every
    supplied frame. Bias matters and gain alone did not have it: two cameras
    on this rig differ by a black-level offset as well as a sensitivity, and a
    pure gain forced through the origin leaves a step at the join that no
    amount of gain can remove."""
    xs = [[], [], []]
    ys = [[], [], []]
    for src, ref, m in pairs:
        if m is None or m.sum() < min_px:
            continue
        for c in range(3):
            xs[c].append(src[..., c][m].astype(np.float64))
            ys[c].append(ref[..., c][m].astype(np.float64))
    g = np.ones(3)
    b = np.zeros(3)
    for c in range(3):
        if not xs[c]:
            continue
        x = np.concatenate(xs[c])
        y = np.concatenate(ys[c])
        if x.size < min_px or x.std() < 1e-6:
            continue
        A = np.stack([x, np.ones_like(x)], 1)
        sol, *_ = np.linalg.lstsq(A, y, rcond=None)
        g[c], b[c] = float(sol[0]), float(sol[1])
    return g, b


def apply_photometric(img, gain, bias):
    return np.clip(img.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)


def warp_all(rig, vcam, sources, depth_m, depth_by_module=None, map_cache=None):
    """-> (warped, valid, cost, mid_index) keyed by module index."""
    import cv2
    from src.rig.geometry import source_maps
    from src.rig.render_wide import off_axis_deg
    depth_by_module = depth_by_module or {}
    map_cache = {} if map_cache is None else map_cache
    warped, valid, cost = {}, {}, {}
    for i, m in enumerate(rig.modules):
        name = m.left.name
        if name not in sources:
            continue
        z = float(depth_by_module.get(m.name, depth_m))
        key = (name, round(z, 6), vcam.width, vcam.height,
               round(float(vcam.hfov), 9), round(float(vcam.vfov), 9))
        if key not in map_cache:
            map_cache[key] = source_maps(rig, name, vcam, z)
        mx, my, ok = map_cache[key]
        warped[i] = cv2.remap(sources[name], mx, my, cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(0, 0, 0))
        valid[i] = ok
        cost[i] = off_axis_deg(rig, name, vcam)
    return warped, valid, cost, len(rig.modules) // 2


def blend_weights(valid, cost, mid_i, mid_authority_deg=72.0,
                  temp=BLEND_TEMP_DEG):
    """Fixed per-pixel weights from geometry alone. -> ({i: w}, hard_owner)

    A softmax over the SAME cost the argmin used, so the region each module
    dominates is unchanged and only the boundary softens. Nothing here reads
    the image, so the weights are identical on every frame of the recording --
    which is the property a dynamic seam gives up and a downstream video model
    would see it give up."""
    keys = sorted(valid)
    big = 1e6
    adj = {}
    for i in keys:
        s = np.where(valid[i], cost[i].astype(np.float32), big)
        if i == mid_i:
            s = np.where(valid[i] & (cost[i] <= mid_authority_deg),
                         -1000.0 + cost[i], s)
        adj[i] = s
    stack = np.stack([adj[i] for i in keys], 0)
    reach = stack.min(0) < 1e5
    hard = np.where(reach, np.array(keys, np.int8)[stack.argmin(0)],
                    -1).astype(np.int8)
    z = np.exp(-(stack - stack.min(0, keepdims=True)) / max(temp, 1e-6))
    z = np.where(stack < 1e5, z, 0.0)
    tot = z.sum(0)
    w = {i: np.where(tot > 0, z[j] / np.maximum(tot, 1e-9), 0.0)
         for j, i in enumerate(keys)}
    return w, hard, reach


def compose(warped, valid, w, hard, reach, gate=GATE_ABS_DIFF):
    """Weighted sum where the sources agree, hard winner where they do not.

    -> (rgb, hard_owner, gated_mask). `gated_mask` is where the blend was
    refused, and it is reported rather than hidden: it is the map of places
    the two views genuinely see different surfaces."""
    keys = sorted(warped)
    H, W = hard.shape
    acc = np.zeros((H, W, 3), np.float32)
    for i in keys:
        acc += warped[i].astype(np.float32) * w[i][..., None]

    # Where do the two best-weighted sources disagree? Compare each pixel's
    # blended value with its hard winner: a large gap means the blend has
    # averaged two different surfaces, which is the ghost this exists to stop.
    hardimg = np.zeros((H, W, 3), np.uint8)
    for i in keys:
        sel = reach & (hard == i)
        hardimg[sel] = warped[i][sel]
    diff = np.abs(acc - hardimg.astype(np.float32)).mean(2)
    gated = diff > gate

    out = np.where(gated[..., None], hardimg, np.clip(acc, 0, 255)
                   ).astype(np.uint8)
    out[~reach] = 0
    return out, hard, gated & reach


def fit_from_video(rig, vcam, videos, frames, depth_m=0.6, map_cache=None):
    """-> {module_index: (gain, bias)} fitted once over `frames`."""
    from src.rig.render_wide import read_frame, split_halves
    map_cache = {} if map_cache is None else map_cache
    acc = {}
    mid_i = len(rig.modules) // 2
    for f in frames:
        sources = {}
        for m in rig.modules:
            key = f"cam{m.left.name[-1]}{m.right.name[-1]}"
            if key not in videos:
                continue
            l, r = split_halves(read_frame(videos[key], f))
            sources[m.left.name], sources[m.right.name] = l, r
        if not sources:
            continue
        warped, valid, cost, mid_i = warp_all(rig, vcam, sources, depth_m,
                                              map_cache=map_cache)
        if mid_i not in warped:
            continue
        for i in warped:
            if i == mid_i:
                continue
            acc.setdefault(i, []).append(
                (warped[i], warped[mid_i], valid[i] & valid[mid_i]))
    out = {}
    for i, pairs in acc.items():
        out[i] = fit_photometric(pairs)
    out[mid_i] = (np.ones(3), np.zeros(3))
    return out


class ClipReader:
    """The three videos, opened once, seeked once, then read straight through.

    `render_wide.read_frame` opens the file, seeks, reads one frame and closes
    it again. That is right for scattered instants and catastrophic for a
    contiguous clip: 300 frames across three videos is 900 opens and 900 seeks
    on network storage, and a seek in H.264 decodes forward from the previous
    keyframe anyway. Sequential reading pays for that decode once."""

    def __init__(self, rig, videos, start):
        import cv2
        self.rig = rig
        self.caps = {}
        for m in rig.modules:
            key = f"cam{m.left.name[-1]}{m.right.name[-1]}"
            if key not in videos:
                continue
            c = cv2.VideoCapture(videos[key])
            if not c.isOpened():
                raise SystemExit(f"cannot open {videos[key]}")
            c.set(cv2.CAP_PROP_POS_FRAMES, int(start))
            self.caps[m] = c

    def next(self, skip=0):
        """-> {camera_name: image} or None at end of file."""
        from src.rig.render_wide import split_halves
        out = {}
        for m, c in self.caps.items():
            for _ in range(skip):
                if not c.grab():
                    return None
            ok, img = c.read()
            if not ok:
                return None
            l, r = split_halves(img)
            out[m.left.name], out[m.right.name] = l, r
        return out

    def close(self):
        for c in self.caps.values():
            c.release()


def main():
    import argparse
    import json
    import time
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--video", action="append", required=True,
                    metavar="FILEKEY=PATH")
    ap.add_argument("--start", type=int, default=3000)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--hfov", type=float, default=150.0)
    ap.add_argument("--vfov", type=float, default=90.0)
    ap.add_argument("--depth_m", type=float, default=0.6)
    ap.add_argument("--mid_authority", type=float, default=72.0)
    ap.add_argument("--temp", type=float, default=BLEND_TEMP_DEG)
    ap.add_argument("--gate", type=float, default=GATE_ABS_DIFF)
    ap.add_argument("--mode", default="fixed",
                    choices=("baseline", "fixed", "sidebyside"),
                    help="baseline is render_wide untouched; fixed is this "
                         "module; sidebyside stacks them for looking at")
    ap.add_argument("--out", required=True, help="mp4 path")
    a = ap.parse_args()

    import cv2
    from src.rig.calibration import RigCalibration
    from src.rig.geometry import VirtualWideCamera
    from src.rig.render_wide import read_frame, split_halves, render

    rig = RigCalibration(a.calibration)
    vcam = VirtualWideCamera.from_rig(rig, hfov_deg=a.hfov, vfov_deg=a.vfov)
    videos = dict(s.split("=", 1) for s in a.video)
    frames = list(range(a.start, a.start + a.n * a.stride, a.stride))
    mc = {}

    photo = None
    if a.mode in ("fixed", "sidebyside"):
        fit_at = frames[::max(1, len(frames) // FIT_FRAMES)][:FIT_FRAMES]
        print(f"fitting photometric mapping on {len(fit_at)} frames...")
        photo = fit_from_video(rig, vcam, videos, fit_at, a.depth_m, mc)
        for i, (g, b) in sorted(photo.items()):
            print(f"  {rig.modules[i].name}  gain {np.round(g,3).tolist()}  "
                  f"bias {np.round(b,1).tolist()}")
        print("  fitted ONCE and held for the whole clip -- the baseline "
              "re-estimates\n  gain on every frame, so its colours drift even "
              "where its geometry does not.")

    w = hard = reach = None
    writer = None
    t0, n_gated = time.time(), []
    reader = ClipReader(rig, videos, a.start)
    for k in range(len(frames)):
        sources = reader.next(skip=(a.stride - 1) if k else 0)
        if not sources:
            print(f"  end of file after {k} frames")
            break
        panels = []
        if a.mode in ("baseline", "sidebyside"):
            try:
                base, _, _, _ = render(rig, vcam, sources, a.depth_m,
                                       map_cache=mc)
            except TypeError:
                base, _, _, _ = render(rig, vcam, sources, a.depth_m)
            panels.append(("baseline", base))
        if a.mode in ("fixed", "sidebyside"):
            warped, valid, cost, mid_i = warp_all(rig, vcam, sources,
                                                  a.depth_m, map_cache=mc)
            for i in warped:
                g, b = photo.get(i, (np.ones(3), np.zeros(3)))
                warped[i] = apply_photometric(warped[i], g, b)
            if w is None:
                w, hard, reach = blend_weights(valid, cost, mid_i,
                                               a.mid_authority, a.temp)
            rgb, _, gated = compose(warped, valid, w, hard, reach, a.gate)
            n_gated.append(float(gated.mean()))
            panels.append(("fixed", rgb))

        if a.mode == "sidebyside":
            for name, img in panels:
                cv2.putText(img, name, (24, 48), cv2.FONT_HERSHEY_SIMPLEX,
                            1.3, (255, 255, 255), 3)
            frame_img = np.vstack([p[1] for p in panels])
        else:
            frame_img = panels[0][1]
        if writer is None:
            h, ww = frame_img.shape[:2]
            writer = cv2.VideoWriter(a.out,
                                     cv2.VideoWriter_fourcc(*"mp4v"), 30.0,
                                     (ww, h))
        writer.write(frame_img)
        if (k + 1) % 50 == 0:
            el = time.time() - t0
            print(f"  [{k+1}/{len(frames)}] {el:.0f}s, "
                  f"{el/(k+1)*(len(frames)-k-1):.0f}s left")
    reader.close()
    if writer is not None:
        writer.release()
    print(f"\n  wrote {a.out}  ({len(frames)} frames)")
    if n_gated:
        print(f"  blend refused on {np.mean(n_gated):.2%} of pixels "
              f"(median {np.median(n_gated):.2%})")
        print("  Those are places the two views see different surfaces. "
              "Blending them\n  would put two hands where there is one.")


if __name__ == "__main__":
    main()
