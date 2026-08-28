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

# The range map is smoothed before it is used as a warp field. A median of
# this width removes SGBM speckle; the bilateral range is in METRES, so 0.15
# keeps a hand distinct from a bench 40 cm behind it while erasing the
# few-centimetre noise that scrubs the texture.
MEDIAN_PX = 5
BILAT_SIGMA_M = 0.15


# Fitted gains are refused outside this range. A camera on the same rig under
# the same light differs from its neighbour by tens of percent, not by half.
GAIN_BOUNDS = (0.6, 1.7)


def fit_photometric(pairs, min_px=5000):
    """[(src_img, ref_img, overlap_mask)] -> (gain[3], bias[3]).

    MOMENT MATCHING, NOT PER-PIXEL REGRESSION, AND THE DIFFERENCE IS NOT
    COSMETIC. Least squares of ref = g*src + b assumes the two pixels are the
    same surface. In the overlap they often are not: the measured mean
    absolute difference there is 69 of 255, because the join sits in the
    fisheye periphery where two modules 94 mm apart genuinely see different
    things. Regressing y on a noisy x shrinks the slope toward zero and lets
    the intercept absorb the mean -- regression dilution -- and the first run
    produced exactly that signature: gain 0.51 with bias +53. Applying it
    would have darkened both outer modules to fix a step that is not there.

    Matching the first two moments needs no correspondence at all. It asks
    only that the two views see the same DISTRIBUTION of the same scene, which
    they do, and it recovers a true gain and bias exactly when one exists."""
    ms, mr, ss, sr, n = (np.zeros(3), np.zeros(3), np.zeros(3),
                         np.zeros(3), 0)
    xs = [[], [], []]
    ys = [[], [], []]
    for src, ref, m in pairs:
        if m is None or m.sum() < min_px:
            continue
        n += 1
        for c in range(3):
            xs[c].append(src[..., c][m].astype(np.float64))
            ys[c].append(ref[..., c][m].astype(np.float64))
    if not n:
        return np.ones(3), np.zeros(3)
    g, b = np.ones(3), np.zeros(3)
    for c in range(3):
        x = np.concatenate(xs[c])
        y = np.concatenate(ys[c])
        sx, sy = x.std(), y.std()
        if sx < 1e-6:
            continue
        gc = sy / sx
        if not (GAIN_BOUNDS[0] <= gc <= GAIN_BOUNDS[1]):
            # Outside physical range the overlap is not comparable content;
            # refusing beats applying a correction fitted to a mismatch.
            continue
        g[c] = gc
        b[c] = float(y.mean() - gc * x.mean())
    return g, b


def apply_photometric(img, gain, bias):
    return np.clip(img.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)


def source_maps_perpixel(rig, camera, vcam, range_m):
    """Sampling map using the MEASURED range at each output pixel.

    `source_maps` places every pixel on one plane at an assumed depth. That is
    exact for the reference camera and only the reference camera: cam1 sits at
    the virtual optical centre, so its mapping is a pure rotation and no depth
    enters it. Every other module has a baseline to cam1, and its error is
    that baseline times the difference between assumed and true inverse depth.
    On this rig, at an assumed 0.6 m against a bench edge at 1.5 m:

        module_A  cam1     0.0 mm      0.0 px
        module_B  cam3    94.4 mm     57.7 px      <- owns 88% of the frame
        module_C  cam5   174.5 mm    106.6 px

    Which is why tuning module_A's assumed depth changed nothing and why the
    join jumped by tens of pixels: the reference was exact and everything
    beside it was displaced. Feeding the measured range removes the
    assumption instead of choosing a better value for it."""
    import cv2
    cam = rig.cameras[camera]
    d = vcam.directions().reshape(-1, 3)
    r = np.asarray(range_m, np.float64).reshape(-1, 1)
    p_ref = vcam.eye + d * r
    p_cam = (cam.R.T @ (p_ref - cam.t).T).T
    ok = (p_cam[:, 2] > 1e-6) & np.isfinite(r[:, 0])
    uv = np.full((len(p_cam), 2), -1.0)
    if ok.any():
        pts, _ = cv2.fisheye.projectPoints(
            p_cam[ok].reshape(-1, 1, 3).astype(np.float64),
            np.zeros(3), np.zeros(3), cam.K, cam.D)
        uv[ok] = pts.reshape(-1, 2)
    inside = ok & (uv[:, 0] >= 0) & (uv[:, 0] < cam.width) \
        & (uv[:, 1] >= 0) & (uv[:, 1] < cam.height)
    H, W = vcam.height, vcam.width
    return (uv[:, 0].reshape(H, W).astype(np.float32),
            uv[:, 1].reshape(H, W).astype(np.float32),
            inside.reshape(H, W))


def densify_range(range_m, valid, fallback=1.2, guide=None,
                  median_px=MEDIAN_PX, bilat_sigma=BILAT_SIGMA_M):
    """Fill and SMOOTH the range, FOR RENDERING ONLY.

    `wide_depth` refuses to fill or smooth its output and that stays true: an
    interpolated depth written to a file is indistinguishable from a measured
    one. Deciding where to SAMPLE is a different act -- a wrong guess there
    costs a misplaced pixel, not a false measurement.

    RAW SGBM IS NOT USABLE AS A WARP FIELD, and the first per-pixel render
    showed exactly why. The sampling coordinate moves with the depth, so
    everything the matcher gets wrong turns into geometry:

        few-cm noise per pixel   ->  the sample point jitters a few px
                                     -> the mottled, scrubbed texture
        speckle outliers         ->  isolated pixels displaced far
                                     -> salt-and-pepper tearing
        inpainted holes          ->  a guessed depth over a whole region
                                     -> that region ghosts

    A median kills the speckle, and a joint bilateral guided by the image
    smooths the noise while holding the depth edges where the picture has
    edges -- so a hand keeps its own depth instead of being averaged into the
    bench behind it."""
    import cv2
    r = np.where(valid, np.nan_to_num(range_m, nan=fallback), np.nan)
    m = (~np.isfinite(r)).astype(np.uint8)
    base = np.where(np.isfinite(r), r, fallback).astype(np.float32)
    if m.any():
        filled = cv2.inpaint((base / 8.0 * 255).clip(0, 255).astype(np.uint8),
                             m, 7, cv2.INPAINT_TELEA).astype(np.float32)
        base = np.where(m > 0, filled / 255.0 * 8.0, base)
    if median_px >= 3:
        base = cv2.medianBlur(base, median_px | 1)
    if guide is None:
        base = cv2.bilateralFilter(base, 9, bilat_sigma, 9)
    else:
        try:
            base = cv2.ximgproc.jointBilateralFilter(
                guide, base, 9, bilat_sigma, 9)
        except Exception:
            # ximgproc is a contrib module and may be absent; the plain
            # bilateral smooths without the image's edges but still removes
            # the jitter that scrubs the texture.
            base = cv2.bilateralFilter(base, 9, bilat_sigma, 9)
    return np.clip(base, 0.15, 8.0)


def warp_all(rig, vcam, sources, depth_m, depth_by_module=None, map_cache=None,
             range_m=None):
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
        if range_m is not None:
            mx, my, ok = source_maps_perpixel(rig, name, vcam, range_m)
        else:
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


# Frames used to fit the residual flow. It is a per-pixel median over the
# overlap, so more frames buy robustness to a hand passing through, not
# precision.
FLOW_FRAMES = 16

# The residual is supposed to be SMALL -- a few pixels of calibration error
# after per-pixel depth has removed the parallax. Anything larger is the flow
# estimator latching onto a moving hand or a textureless patch, and is thrown
# away rather than applied.
MAX_RESIDUAL_PX = 12.0


def residual_flow(src, ref, overlap):
    """Dense flow from `src` to `ref`, inside the overlap only. -> [H,W,2]

    Estimated AFTER the calibrated, per-pixel-depth projection has done its
    work, so it has a few pixels to find rather than 31 degrees of geometry.
    That is the whole reason this is tractable where a general stitcher was
    not: a learned 2D warp had to represent a displacement field with a depth
    discontinuity at every arm boundary, and this only has to represent what
    is left after the depth was used properly."""
    import cv2
    g1 = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    f = cv2.calcOpticalFlowFarneback(g1, g2, None, 0.5, 4, 31, 3, 5, 1.2, 0)
    f[~overlap] = 0.0
    return f


def fit_residual_flow(per_frame, max_px=MAX_RESIDUAL_PX):
    """[(flow, overlap)] -> one frozen flow field, or None.

    PER-PIXEL MEDIAN, AND FROZEN. A hand crossing the overlap produces a large
    honest flow that has nothing to do with the calibration residual this is
    correcting; a median over frames drops it. Freezing is what keeps the
    ownership map's bit-identical property from being given away -- a flow
    re-estimated every frame is a boundary that moves, which is the thing a
    downstream video model must never be shown."""
    if not per_frame:
        return None
    flows = np.stack([f for f, _ in per_frame], 0)
    seen = np.stack([o for _, o in per_frame], 0)
    n = seen.sum(0)
    med = np.zeros(flows.shape[1:], np.float32)
    ok = n >= max(3, len(per_frame) // 3)
    if ok.any():
        masked = np.where(seen[..., None], flows, np.nan)
        import warnings
        with warnings.catch_warnings():
            # Pixels seen in no frame are an all-NaN slice by construction;
            # `ok` already excludes them below.
            warnings.simplefilter("ignore", RuntimeWarning)
            m = np.nanmedian(masked, axis=0)
        med[ok] = np.nan_to_num(m[ok])
    mag = np.linalg.norm(med, axis=-1)
    # Refuse the pixels where the estimator found something too big to be a
    # residual: those are moving hands and blank bench, not calibration.
    med[mag > max_px] = 0.0
    return med


def apply_flow(img, flow):
    """Warp `img` to match the reference, given flow(img -> reference).

    THE SIGN IS SUBTRACTED, AND IT IS NOT A CONVENTION QUIBBLE. Farneback
    returns f with img(p) ~ ref(p + f(p)) -- it says where a pixel of img went
    in ref. To BUILD a version of img that sits where ref does, each output
    pixel must be sampled from img at p - f(p). Adding it instead moves
    everything twice as far in the wrong direction, which showed up as the
    correction making a synthetic 4-pixel shift slightly worse rather than
    removing it."""
    import cv2
    H, W = img.shape[:2]
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32),
                         np.arange(H, dtype=np.float32))
    return cv2.remap(img, gx - flow[..., 0], gy - flow[..., 1],
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def fit_from_video(rig, vcam, videos, frames, depth_m=0.6, map_cache=None,
                   depth_by_module=None):
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
        warped, valid, cost, mid_i = warp_all(
            rig, vcam, sources, depth_m, depth_by_module=depth_by_module,
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
            self.caps[m.left.name, m.right.name] = c

    def next(self, skip=0):
        """-> {camera_name: image} or None at end of file."""
        from src.rig.render_wide import split_halves
        out = {}
        for (ln, rn), c in self.caps.items():
            for _ in range(skip):
                if not c.grab():
                    return None
            ok, img = c.read()
            if not ok:
                return None
            l, r = split_halves(img)
            out[ln], out[rn] = l, r
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
    ap.add_argument("--depth_module", action="append", default=[],
                    metavar="NAME=METRES",
                    help="per-module assumed depth, e.g. module_A=8. ONE "
                         "CONSTANT CANNOT SERVE ALL THREE: the middle module "
                         "sees a bench at 0.6 m while the outer two see floor "
                         "and the far aisle metres away, and reprojecting "
                         "those as if they were at 0.6 m displaces them by "
                         "most of a wedge -- which is what the periphery of "
                         "the baseline render actually shows.")
    ap.add_argument("--mid_authority", type=float, default=72.0)
    ap.add_argument("--temp", type=float, default=BLEND_TEMP_DEG)
    ap.add_argument("--gate", type=float, default=GATE_ABS_DIFF)
    ap.add_argument("--per_pixel_depth", action="store_true",
                    help="reproject with the MEASURED range at every pixel "
                         "instead of one assumed plane. Removes the "
                         "assumption that displaces module_B by 58 px and "
                         "module_C by 107 px; costs a stereo solve per frame.")
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
    dbm = {}
    for spec in a.depth_module:
        k, v = spec.split("=", 1)
        dbm[k] = float(v)
    if dbm:
        print("per-module depth: " + ", ".join(f"{k}={v:g}m"
                                               for k, v in sorted(dbm.items())))

    photo = None
    if a.mode in ("fixed", "sidebyside"):
        fit_at = frames[::max(1, len(frames) // FIT_FRAMES)][:FIT_FRAMES]
        print(f"fitting photometric mapping on {len(fit_at)} frames...")
        photo = fit_from_video(rig, vcam, videos, fit_at, a.depth_m, mc,
                               depth_by_module=dbm)
        for i, (g, b) in sorted(photo.items()):
            print(f"  {rig.modules[i].name}  gain {np.round(g,3).tolist()}  "
                  f"bias {np.round(b,1).tolist()}")
        print("  fitted ONCE and held for the whole clip -- the baseline "
              "re-estimates\n  gain on every frame, so its colours drift even "
              "where its geometry does not.")

    w = hard = reach = None
    rectc = {}
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
                                       map_cache=mc,
                                       depth_by_module=dbm or None)
            except TypeError:
                base, _, _, _ = render(rig, vcam, sources, a.depth_m)
            panels.append(("baseline", base))
        if a.mode in ("fixed", "sidebyside"):
            rng_m = None
            if a.per_pixel_depth:
                from src.rig.wide_depth import wide_depth
                wd = wide_depth(rig, vcam, sources, rect_cache=rectc)
                rng_m = densify_range(wd.range_m, wd.valid,
                                      guide=None)
            warped, valid, cost, mid_i = warp_all(
                rig, vcam, sources, a.depth_m, depth_by_module=dbm,
                map_cache=mc, range_m=rng_m)
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
