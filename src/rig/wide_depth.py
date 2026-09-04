"""WIDE_DEPTH -- per-pixel metric range in the virtual wide camera.

The other half of the deliverable, and the depth channels the segmentation
head is trained on. Until now `render_wide` assumed ONE constant depth per
module, which is enough to place pixels but carries no per-pixel geometry at
all; this renders the real thing.

IT IS RANGE, NOT Z, AND THE NAME MATTERS. The virtual camera is
equirectangular over 150 degrees. There is no single optical axis for a field
that wide -- a pinhole z would diverge toward the edges and a consumer that
assumed z would silently read the periphery 40% too near. Every number here is
the distance from the virtual optical centre to the point, along the ray. A
`z` view is recoverable (multiply by cos of the ray angle) and deliberately
not provided by default.

HOW IT IS BUILT. Each module solves its own 60 mm stereo, its valid pixels
become 3D points in the rig reference frame, and those points are projected
into the virtual camera with a z-buffer -- nearest wins. This is a render from
a reconstruction, the same stance `geometry` takes for RGB, and it is why
overlapping modules do not average a near hand into two hands.

SPLAT FOOTPRINT, NOT HOLE FILLING. A source pixel has area. At this rig the
module samples about 11.7 px per degree and the output about 10.7, so a
one-pixel splat lands roughly 1:1 and leaves a scatter of quantisation holes
that are artefacts of the projection rather than of the sensor. Each point is
therefore written across its 2x2 footprint. That is not interpolation: no
value is invented, an existing measurement is given the extent it physically
has. Everything beyond it stays INVALID, because `depth` refuses to fill in
and a wide depth map that quietly filled the bench would be worse than one
that admits it did not measure it.

ALIGNMENT WITH THE RGB. The RGB comes from the constant-depth render, which is
dense, and the depth from this, which is sparse. They therefore disagree by
the constant-depth displacement -- measured in `render_wide.seam_shift` at
under 2.6 px anywhere in the field, on a 1600x900 output. For a segmentation
head that is below the mask precision; for anything wanting exact
correspondence it is not, and the fix is to render RGB from the same points,
at the cost of the same holes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Each source point is written across its 2x2 pixel footprint. See the module
# docstring: this is extent, not interpolation.
SPLAT = 2

# A point nearer than this to the virtual centre is inside the wearer, and one
# past this is beyond anything the 60 mm baseline resolves. Both are matcher
# failures rather than measurements. Matches depth.MIN/MAX_DEPTH_M.
MIN_RANGE_M = 0.12
MAX_RANGE_M = 8.0


@dataclass
class WideDepth:
    """Range in metres in the virtual camera, with its validity and provenance."""
    range_m: np.ndarray        # [H,W] float32, NaN where nothing was measured
    valid: np.ndarray          # [H,W] bool
    module: np.ndarray         # [H,W] int8, index of the module that won, -1
    n_points: int              # points projected in
    module_names: tuple

    def coverage(self):
        return float(self.valid.mean())

    def channels(self):
        """(range, valid) as the two extra input planes, NaN replaced by 0.

        The head must never see NaN, and it must never be told 0 m is a
        measurement -- which is exactly what the validity plane is for. They
        are returned together so a caller cannot take one without the other."""
        r = np.where(self.valid, self.range_m, 0.0).astype(np.float32)
        return r, self.valid.astype(np.float32)


@dataclass
class WideRGBD:
    """One-owner RGB and range rendered from measured 3D points."""
    rgb: np.ndarray              # [H,W,3] uint8; never colour-averaged
    range_m: np.ndarray          # [H,W] float32, NaN where unmeasured
    valid: np.ndarray            # [H,W] bool
    module: np.ndarray           # [H,W] int8, winning module or -1
    n_points: int
    module_names: tuple

    def coverage(self):
        return float(self.valid.mean())


def project(vcam, p_ref):
    """Reference-frame points -> (x, y, range) in the virtual camera.

    Inverse of `VirtualWideCamera.directions`. Returns float pixel coordinates
    and the range along the ray; `inside` marks what actually lands on the
    sensor."""
    w = (np.asarray(p_ref, np.float64) - vcam.eye) @ vcam.R    # ref -> virtual
    r = np.linalg.norm(w, axis=1)
    ok = r > 1e-9
    r = np.where(ok, r, np.nan)
    el = np.arcsin(np.clip(-w[:, 1] / r, -1.0, 1.0))
    az = np.arctan2(w[:, 0], w[:, 2])
    x = (az / vcam.hfov + 0.5) * vcam.width
    y = (0.5 - el / vcam.vfov) * vcam.height
    inside = (ok & np.isfinite(x) & np.isfinite(y)
              & (x >= 0) & (x < vcam.width)
              & (y >= 0) & (y < vcam.height)
              & (r >= MIN_RANGE_M) & (r <= MAX_RANGE_M))
    return x, y, r, inside


def composite(vcam, per_module, splat=SPLAT):
    """per_module: [(name, points [N,3] in the reference frame)] -> WideDepth.

    Nearest wins. The z-buffer is done by sorting far-to-near and letting the
    nearer write last, which is exact and costs one sort rather than a scatter
    reduction over seven million points."""
    H, W = vcam.height, vcam.width
    rng = np.full((H, W), np.inf, np.float32)
    mod = np.full((H, W), -1, np.int8)
    names, xs, ys, rs, ms = [], [], [], [], []
    for i, (name, pts) in enumerate(per_module):
        names.append(name)
        if len(pts) == 0:
            continue
        x, y, r, ok = project(vcam, pts)
        xs.append(x[ok]); ys.append(y[ok]); rs.append(r[ok])
        ms.append(np.full(int(ok.sum()), i, np.int8))
    if not xs:
        return WideDepth(np.full((H, W), np.nan, np.float32),
                         np.zeros((H, W), bool), mod, 0, tuple(names))
    x = np.concatenate(xs); y = np.concatenate(ys)
    r = np.concatenate(rs).astype(np.float32); m = np.concatenate(ms)

    order = np.argsort(-r, kind="stable")        # far first, near overwrites
    x, y, r, m = x[order], y[order], r[order], m[order]
    xi = np.floor(x).astype(np.int32)
    yi = np.floor(y).astype(np.int32)
    for dy in range(splat):
        for dx in range(splat):
            px, py = xi + dx, yi + dy
            k = (px >= 0) & (px < W) & (py >= 0) & (py < H)
            # Later writes win, and the array is ordered far-to-near, so this
            # IS the z-buffer. A same-pixel tie keeps the nearer by the sort.
            rng[py[k], px[k]] = r[k]
            mod[py[k], px[k]] = m[k]
    valid = np.isfinite(rng)
    return WideDepth(np.where(valid, rng, np.nan).astype(np.float32),
                     valid, mod, int(len(r)), tuple(names))


def composite_rgbd(vcam, per_module, splat=SPLAT, depth_tie_m=0.02):
    """Forward-render coloured point clouds with one texture owner per pixel.

    ``per_module`` contains ``(name, points, colours, quality)`` tuples, where
    lower quality values mean a more on-axis sample.  Depth differences over
    ``depth_tie_m`` are true visibility decisions and the nearer point wins.
    Candidates on the same surface use quality with a deterministic module
    tiebreak.  RGB values are selected, never averaged.
    """
    H, W = vcam.height, vcam.width
    names, flats, ranges, modules, colours, qualities = [], [], [], [], [], []
    n_points = 0
    for module_index, item in enumerate(per_module):
        if len(item) == 3:
            name, points, colour = item
            quality = np.zeros(len(points), np.float32)
        else:
            name, points, colour, quality = item
        names.append(name)
        points = np.asarray(points)
        colour = np.asarray(colour, np.uint8)
        quality = np.asarray(quality, np.float32)
        if len(points) != len(colour) or len(points) != len(quality):
            raise ValueError(f"{name}: point, colour and quality counts differ")
        if not len(points):
            continue
        x, y, radius, inside = project(vcam, points)
        x, y, radius = x[inside], y[inside], radius[inside]
        colour, quality = colour[inside], quality[inside]
        n_points += len(radius)
        xi, yi = np.floor(x).astype(np.int32), np.floor(y).astype(np.int32)
        for dy in range(splat):
            for dx in range(splat):
                px, py = xi + dx, yi + dy
                keep = (px >= 0) & (px < W) & (py >= 0) & (py < H)
                flats.append(py[keep] * W + px[keep])
                ranges.append(radius[keep].astype(np.float32))
                modules.append(np.full(int(keep.sum()), module_index,
                                       np.int8))
                colours.append(colour[keep])
                qualities.append(quality[keep] + module_index * 1e-6)

    rgb = np.zeros((H * W, 3), np.uint8)
    output_range = np.full(H * W, np.inf, np.float32)
    output_module = np.full(H * W, -1, np.int8)
    if not flats:
        return WideRGBD(rgb.reshape(H, W, 3),
                        np.full((H, W), np.nan, np.float32),
                        np.zeros((H, W), bool), output_module.reshape(H, W),
                        0, tuple(names))

    flat = np.concatenate(flats)
    radius = np.concatenate(ranges)
    module = np.concatenate(modules)
    colour = np.concatenate(colours)
    quality = np.concatenate(qualities)
    nearest = np.full(H * W, np.inf, np.float32)
    np.minimum.at(nearest, flat, radius)
    same_surface = radius <= nearest[flat] + float(depth_tie_m)
    best_quality = np.full(H * W, np.inf, np.float32)
    np.minimum.at(best_quality, flat[same_surface], quality[same_surface])
    eligible = same_surface & (quality <= best_quality[flat] + 1e-7)

    indices = np.flatnonzero(eligible)
    order = np.lexsort((radius[indices], flat[indices]))
    indices = indices[order]
    ordered_pixels = flat[indices]
    first = np.r_[True, ordered_pixels[1:] != ordered_pixels[:-1]]
    winners = indices[first]
    pixels = flat[winners]
    rgb[pixels] = colour[winners]
    output_range[pixels] = radius[winners]
    output_module[pixels] = module[winners]
    valid = np.isfinite(output_range)
    return WideRGBD(
        rgb.reshape(H, W, 3),
        np.where(valid, output_range, np.nan).reshape(H, W).astype(np.float32),
        valid.reshape(H, W), output_module.reshape(H, W), n_points,
        tuple(names))


def wide_depth(rig, vcam, sources, rect_cache=None, stride=1, splat=SPLAT,
               matcher=None):
    """Full path: module images -> stereo -> points -> composite.

    `sources` is {camera_name: image}, the same dict `render_wide.render`
    takes, so one frame feeds both. A module missing either eye is skipped
    rather than guessed at."""
    from src.rig.depth import module_depth, rectify_maps, to_reference_points
    rect_cache = {} if rect_cache is None else rect_cache
    per_module = []
    for m in rig.modules:
        if m.left.name not in sources or m.right.name not in sources:
            continue
        if m.name not in rect_cache:
            rect_cache[m.name] = rectify_maps(rig, m)
        md = module_depth(rig, m, sources[m.left.name], sources[m.right.name],
                          rect_cache[m.name], matcher=matcher)
        pts, _ = to_reference_points(rig, m, md, stride=stride)
        per_module.append((m.name, pts))
    return composite(vcam, per_module, splat=splat)


def wide_rgbd(rig, vcam, sources, rect_cache=None, stride=1, splat=SPLAT,
              matcher=None, depth_tie_m=0.02):
    """Stereo pairs -> left-eye coloured points -> one-owner RGBD panorama."""
    from src.rig.depth import module_depth, rectify_maps, to_reference_points

    rect_cache = {} if rect_cache is None else rect_cache
    per_module = []
    for module in rig.modules:
        if (module.left.name not in sources or
                module.right.name not in sources):
            continue
        if module.name not in rect_cache:
            rect_cache[module.name] = rectify_maps(rig, module)
        measured = module_depth(
            rig, module, sources[module.left.name], sources[module.right.name],
            rect_cache[module.name], matcher=matcher)
        points, colour = to_reference_points(
            rig, module, measured, stride=stride)
        camera = rig.cameras[module.left.name]
        rays = points - camera.t
        lengths = np.linalg.norm(rays, axis=1)
        cosine = np.divide(
            rays @ camera.axis, lengths,
            out=np.zeros_like(lengths), where=lengths > 1e-9)
        quality = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
        per_module.append((module.name, points, colour,
                           quality.astype(np.float32)))
    return composite_rgbd(vcam, per_module, splat=splat,
                          depth_tie_m=depth_tie_m)


def _self_test():
    """Each case is a way this gets silently wrong."""
    from src.rig.geometry import VirtualWideCamera
    ok = []

    def chk(cond, msg):
        ok.append(bool(cond))
        print(f"  {'ok ' if cond else 'FAIL'} {msg}")

    v = VirtualWideCamera(R=np.eye(3), eye=np.zeros(3), width=200, height=100,
                          hfov=np.radians(100), vfov=np.radians(50))

    # A point straight ahead lands in the centre.
    x, y, r, i = project(v, np.array([[0, 0, 1.0]]))
    chk(i[0] and abs(x[0] - 100) < .01 and abs(y[0] - 50) < .01
        and abs(r[0] - 1) < 1e-6, "a point on the axis lands at the centre")

    # RANGE, NOT Z. 45 degrees off axis at range 1 must read 1.0, not 1.41.
    p = np.array([[np.sin(np.radians(45)), 0, np.cos(np.radians(45))]])
    x, y, r, i = project(v, p)
    chk(abs(r[0] - 1.0) < 1e-6, "off-axis range is range, not pinhole z")
    chk(abs(x[0] - (45 / 100 + 0.5) * 200) < .01,
        "...and its azimuth maps linearly across the equirectangular field")

    # Behind the camera is dropped, not wrapped round to the front.
    _, _, _, i = project(v, np.array([[0, 0, -1.0]]))
    chk(not i[0], "a point behind the camera is dropped, not wrapped")

    # Z-buffer: the near point must win regardless of the order given.
    near = np.array([[0, 0, 0.5]]); far = np.array([[0, 0, 3.0]])
    for tag, pm in (("near first", [("a", near), ("b", far)]),
                    ("far first", [("a", far), ("b", near)])):
        w = composite(v, pm, splat=1)
        chk(abs(w.range_m[50, 100] - 0.5) < 1e-5,
            f"nearest wins with the {tag}")
    chk(composite(v, [("a", far), ("b", near)], splat=1).module[50, 100] == 1,
        "...and the winning module is reported, not the last one drawn")

    # Nothing is invented outside the projected extent.
    w = composite(v, [("a", near)], splat=1)
    chk(w.valid.sum() == 1 and w.n_points == 1,
        "one point marks one pixel valid and the rest stay unmeasured")
    chk(np.isnan(w.range_m[0, 0]), "an unmeasured pixel is NaN, not 0 m")

    # The splat footprint is extent, and bounded by it.
    chk(composite(v, [("a", near)], splat=2).valid.sum() == 4,
        "a 2x2 footprint marks exactly 4 pixels, not a blur")

    # The two channels the head sees must not disagree about what is measured.
    rc, vc = composite(v, [("a", near)], splat=1).channels()
    chk(rc[0, 0] == 0 and vc[0, 0] == 0 and vc[50, 100] == 1,
        "an unmeasured pixel is 0 with validity 0, never 0 m with validity 1")

    # A matcher failure is not a measurement.
    _, _, _, i = project(v, np.array([[0, 0, 40.0], [0, 0, 0.02]]))
    chk(not i.any(), "40 m and 2 cm are both refused as matcher failures")

    # Empty input must not crash the production render.
    w = composite(v, [("a", np.zeros((0, 3)))], splat=1)
    chk(w.valid.sum() == 0 and w.coverage() == 0.0,
        "a module with no valid stereo yields an empty map, not an exception")

    print(f"\n  {sum(ok)}/{len(ok)} cases pass.")
    print("  Range not z, nearest wins, and nothing outside the footprint is\n"
          "  invented. A filled-in bench would be indistinguishable from a\n"
          "  measured one once it is written to a file.")
    return all(ok)


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self_test", action="store_true")
    ap.add_argument("--calibration")
    ap.add_argument("--video", action="append", default=[],
                    metavar="FILEKEY=PATH")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.self_test:
        raise SystemExit(0 if _self_test() else 1)
    if not a.calibration or not a.video:
        ap.error("--calibration and --video are required without --self_test")

    import cv2
    from src.rig.calibration import RigCalibration
    from src.rig.geometry import VirtualWideCamera
    from src.rig.render_wide import read_frame, split_halves

    rig = RigCalibration(a.calibration)
    vcam = VirtualWideCamera.from_rig(rig)
    files = dict(s.split("=", 1) for s in a.video)
    sources = {}
    for m in rig.modules:
        key = f"cam{m.left.name[-1]}{m.right.name[-1]}"
        if key in files:
            l, r = split_halves(read_frame(files[key], a.frame))
            sources[m.left.name], sources[m.right.name] = l, r
    wd = wide_depth(rig, vcam, sources)

    v = wd.range_m[wd.valid]
    print(f"WIDE_DEPTH  {vcam.width}x{vcam.height}  frame {a.frame}")
    print(f"  modules in    {', '.join(wd.module_names) or 'none'}")
    print(f"  points        {wd.n_points:,}")
    print(f"  coverage      {wd.coverage():.1%}")
    if v.size:
        print(f"  range m       p05 {np.percentile(v,5):.2f}   "
              f"median {np.median(v):.2f}   p95 {np.percentile(v,95):.2f}")
        for i, n in enumerate(wd.module_names):
            s = wd.module == i
            print(f"    {n} owns {s.mean():6.1%} of the frame")
    if a.out:
        vis = np.zeros(wd.range_m.shape + (3,), np.uint8)
        lo, hi = 0.2, 2.5
        # nan_to_num BEFORE the cast: NaN -> uint8 is undefined, and the
        # blacking-out below happens after the colour map has already run.
        t = np.nan_to_num(np.clip((wd.range_m - lo) / (hi - lo), 0, 1))
        vis = cv2.applyColorMap((t * 255).astype(np.uint8),
                                cv2.COLORMAP_TURBO)
        vis[~wd.valid] = 0
        cv2.imwrite(a.out, vis)
        print(f"\n  wrote {a.out}  (TURBO {lo}-{hi} m, black = unmeasured)")
    print("\n  Every number is RANGE from the virtual centre along the ray, "
          "not a\n  pinhole z. Over a 150 degree field the two differ by 40% "
          "at the edge.")


if __name__ == "__main__":
    main()
