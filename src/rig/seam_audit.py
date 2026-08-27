"""Does the join show? Measured against the three acceptance criteria.

The question "is the stitch seamless" is not answerable as stated -- six
optical centres and a hand 30 cm from the lens guarantee that some pixel
somewhere is wrong. What IS answerable is whether the join is visible to
whatever consumes the frame, and that decomposes into three things a person
asked for and all three can be measured:

    1  a hand crossing the overlap must not double, break or stretch
    2  straight structures -- the bench edge, a bin, a cable -- must not
       fracture where the source changes
    3  the boundary must not flicker or shift between frames

THE THIRD IS FREE HERE, AND IT IS WORTH KNOWING WHY. This renderer selects
per pixel by GEOMETRY -- off-axis angle and validity -- not by content. Those
depend on the rig, the virtual camera and the assumed depth, none of which
vary within a clip, so the ownership map is bit-identical on every frame and
the seam cannot flicker. That is a property a content-adaptive seam finder or
a learned warp gives up, and it is checked here rather than assumed.

THE SECOND IS THE REAL TEST, AND THE MEASUREMENT IS: THE SEAM MUST NOT ITSELF
BE AN EDGE. A fracture in the bench edge at the join creates a gradient
exactly along the seam that is not present in the scene. So the gradient
magnitude on the seam is compared with a control band the same shape and size,
offset to one side. A ratio near 1 means the join is invisible to a gradient
operator; a ratio well above 1 means the seam is drawing a line.

THE FIRST IS MEASURED WHERE IT MATTERS, WHICH MAY BE NOWHERE. The middle
module owns 88% of this frame outright and the wearer's hands sit in the
middle of it, so a hand may never approach a join at all. That is reported
first, because an artifact in a region the hands never enter is a different
and much smaller problem than one they cross constantly.

WHAT THIS DOES NOT MEASURE: whether a learned stitcher would do better. It
measures whether this one has a problem worth replacing. Those are different
questions and only the second is cheap.
"""
from __future__ import annotations

import numpy as np

# Pixels within this distance of an ownership boundary count as "on the seam".
SEAM_BAND_PX = 6

# The control band is offset this far from the seam -- far enough to be off it,
# close enough to sample the same kind of content.
CONTROL_OFFSET_PX = 40


def seam_mask(owner, band=SEAM_BAND_PX):
    """Pixels within `band` of a change in ownership. -> bool [H,W]

    Only boundaries between two VALID modules count. The outer edge of the
    rendered field is a boundary against nothing, and treating it as a seam
    would measure the frame's border instead of the join."""
    import cv2
    o = np.asarray(owner)
    edge = np.zeros(o.shape, np.uint8)
    a, b = o[:, 1:], o[:, :-1]
    edge[:, 1:][(a != b) & (a >= 0) & (b >= 0)] = 255
    a, b = o[1:, :], o[:-1, :]
    edge[1:, :][(a != b) & (a >= 0) & (b >= 0)] = 255
    k = 2 * band + 1
    return cv2.dilate(edge, np.ones((k, k), np.uint8)) > 0


def control_mask(seam, owner, offset=CONTROL_OFFSET_PX):
    """A band the same size as the seam band, shifted off it, inside valid
    pixels. Comparing the seam with the whole frame would compare a band that
    follows the periphery against an average dominated by the bench."""
    import cv2
    k = 2 * offset + 1
    wide = cv2.dilate(seam.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
    return wide & ~seam & (np.asarray(owner) >= 0)


def gradient(rgb):
    import cv2
    g = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def seam_edge_ratio(rgb, owner, band=SEAM_BAND_PX, offset=CONTROL_OFFSET_PX):
    """-> dict. The seam should not itself be an edge. TWO numbers, because
    the two ways it can fail need different statistics.

    `ratio` is median-on-seam over median-beside-it, and it catches a seam
    that is wrong ALONG ITS WHOLE LENGTH -- ghosting, a colour step, a blur
    band. Medians, so one specular highlight cannot move it.

    `excess` is the share of seam pixels whose gradient exceeds the 95th
    percentile of the surrounding content, minus the 0.05 that definition
    produces by construction. It catches a LOCAL fracture -- a bench edge
    broken at the join -- which is what a person actually sees and which the
    median is specifically robust against: a break spanning 10% of the seam
    moves no median at all. That failure showed up as a test that would not go
    red for an obviously fractured frame, which is why both are here."""
    s = seam_mask(owner, band)
    c = control_mask(s, owner, offset)
    if s.sum() < 200 or c.sum() < 200:
        return {"n_seam": int(s.sum()), "n_control": int(c.sum()),
                "ratio": float("nan"), "excess": float("nan"),
                "seam": float("nan"), "control": float("nan")}
    g = gradient(rgb)
    gs, gc = float(np.median(g[s])), float(np.median(g[c]))
    hi = float(np.percentile(g[c], 95))
    return {"n_seam": int(s.sum()), "n_control": int(c.sum()),
            "seam": gs, "control": gc,
            "excess": float((g[s] > hi).mean() - 0.05),
            "ratio": gs / gc if gc > 1e-6 else float("nan")}


def skin_at_seam(rgb, owner, band=SEAM_BAND_PX):
    """-> (skin pixels on the seam, skin pixels total, fraction).

    Whether the hands ever meet a join at all. If they never do, criterion 1
    is about a region the downstream model does not care about."""
    from src.rig.near_other_miner import skin_mask
    sk = skin_mask(rgb) > 0
    s = seam_mask(owner, band)
    n = int(sk.sum())
    return int((sk & s).sum()), n, (float((sk & s).sum()) / n if n else 0.0)


def audit_clip(rig, vcam, videos, frames, depth_m=0.6, band=SEAM_BAND_PX):
    """-> (per-frame rows, owner map of the first frame)."""
    from src.rig.render_wide import read_frame, split_halves, render
    rows, owner0, mc = [], None, {}
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
        try:
            rgb, owner, stats, _ = render(rig, vcam, sources, depth_m,
                                          map_cache=mc)
        except TypeError:                      # renderer without the cache
            rgb, owner, stats, _ = render(rig, vcam, sources, depth_m)
        if owner0 is None:
            owner0 = owner.copy()
        er = seam_edge_ratio(rgb, owner, band)
        sn, st, sf = skin_at_seam(rgb, owner, band)
        rows.append({
            "frame": f,
            "owner_identical_to_first": bool(np.array_equal(owner, owner0)),
            "seam_grad": er["seam"], "control_grad": er["control"],
            "seam_edge_ratio": er["ratio"], "seam_excess": er["excess"],
            "n_seam_px": er["n_seam"],
            "skin_px": st, "skin_on_seam_px": sn, "skin_on_seam_frac": sf,
            "overlap_disagreement": {k: round(v[0], 2)
                                     for k, v in stats.items()},
        })
    return rows, owner0


def report(rows):
    if not rows:
        print("  nothing rendered")
        return
    def col(k):
        v = [r[k] for r in rows if np.isfinite(r.get(k, np.nan))]
        return np.array(v) if v else np.array([np.nan])

    stable = all(r["owner_identical_to_first"] for r in rows)
    print(f"\n  3. TEMPORAL  ownership map identical on all "
          f"{len(rows)} frames: {stable}")
    if stable:
        print("     The seam is fixed by geometry, not by content, so it "
              "cannot flicker.\n     A content-adaptive seam finder or a "
              "learned warp gives this up.")
    else:
        n = sum(1 for r in rows if not r["owner_identical_to_first"])
        print(f"     !! {n} frames differ -- the seam MOVES, and that is a "
              f"real defect.")

    r = col("seam_edge_ratio")
    print(f"\n  2. STRUCTURE  gradient on the seam / gradient beside it")
    print(f"     median {np.nanmedian(r):.3f}   p90 {np.nanpercentile(r,90):.3f}"
          f"   max {np.nanmax(r):.3f}")
    print(f"     1.0 = the join is invisible along its whole length.")
    e = col("seam_excess")
    print(f"     excess (local fractures)  median {np.nanmedian(e):+.3f}   "
          f"p90 {np.nanpercentile(e,90):+.3f}   max {np.nanmax(e):+.3f}")
    print(f"     0.000 = nothing on the seam is sharper than the scene "
          f"around it.\n     The median catches ghosting along the whole "
          f"join; excess catches a\n     bench edge broken at one point, "
          f"which no median can see.")

    f = col("skin_on_seam_frac")
    n_touch = sum(1 for x in [r_["skin_on_seam_frac"] for r_ in rows] if x > 0.01)
    print(f"\n  1. HANDS      frames where >1% of skin sits on a seam: "
          f"{n_touch}/{len(rows)}")
    print(f"     median share of skin on a seam {np.nanmedian(f):.2%}   "
          f"max {np.nanmax(f):.2%}")
    if n_touch == 0:
        print("     The hands never meet a join. An artifact there is not on "
              "the path\n     the downstream model cares about.")

    dis = {}
    for row in rows:
        for k, v in row["overlap_disagreement"].items():
            dis.setdefault(k, []).append(v)
    if dis:
        print(f"\n  overlap disagreement (mean |A-B| where two modules both "
              f"reach, 0-255)")
        for k, v in sorted(dis.items()):
            print(f"     {k}: median {np.median(v):.1f}")


def _self_test():
    ok = []

    def chk(c, m):
        ok.append(bool(c))
        print(f"  {'ok ' if c else 'FAIL'} {m}")

    H = W = 200
    owner = np.full((H, W), -1, np.int8)
    owner[:, :100] = 0
    owner[:, 100:180] = 1                 # a real join at x=100
    s = seam_mask(owner, band=3)
    chk(s[:, 97:104].all() and not s[:, :90].any(),
        "the seam band follows the ownership change")
    chk(not s[:, 178:184].any(),
        "the edge of the rendered field is NOT a seam -- it borders nothing")
    c = control_mask(s, owner, offset=20)
    chk(c.sum() > 0 and not (c & s).any(),
        "the control band is beside the seam and never on it")

    # A frame with no fracture: the ratio must sit near 1. The bench carries
    # texture, because a perfectly flat control band has a median gradient of
    # exactly zero and would make the ratio undefined -- a test frame must not
    # be cleaner than anything the measurement will meet.
    rng = np.random.default_rng(0)
    rgb = (np.array((60, 90, 45), np.int16)
           + rng.integers(-12, 13, (H, W, 3))).clip(0, 255).astype(np.uint8)
    rgb[80:90, :] = (200, 200, 200)       # a straight bench edge, unbroken
    r = seam_edge_ratio(rgb, owner, band=3, offset=20)
    chk(0.6 < r["ratio"] < 1.6,
        f"an unbroken structure gives ratio {r['ratio']:.2f}, near 1")

    # The same structure fractured at the join: the ratio must rise.
    broken = rgb.copy()
    broken[80:90, 100:180] = rgb[0:10, 100:180]        # bench edge removed
    broken[105:115, 100:180] = (200, 200, 200)         # and displaced
    r2 = seam_edge_ratio(broken, owner, band=3, offset=20)
    chk(abs(r2["ratio"] - r["ratio"]) < 0.2,
        f"...and the MEDIAN does not move ({r['ratio']:.2f} -> "
        f"{r2['ratio']:.2f}) -- a\n       local break is exactly what a "
        f"median is robust against")
    chk(r2["excess"] > r["excess"] + 0.03,
        f"but excess does: {r['excess']:+.3f} -> {r2['excess']:+.3f}")
    chk(abs(r["excess"]) < 0.03,
        f"an unbroken seam sits near zero excess ({r['excess']:+.3f})")

    chk(np.isnan(seam_edge_ratio(rgb, np.zeros((H, W), np.int8))["ratio"]),
        "a frame with no seam reports nan, not a number about nothing")
    flat = np.full((H, W, 3), (60, 90, 45), np.uint8)
    chk(np.isnan(seam_edge_ratio(flat, owner, band=3, offset=20)["ratio"]),
        "a textureless control band reports nan rather than dividing by zero")

    print(f"\n  {sum(ok)}/{len(ok)} cases pass.")
    return all(ok)


def main():
    import argparse
    import json
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self_test", action="store_true")
    ap.add_argument("--calibration")
    ap.add_argument("--video", action="append", default=[],
                    metavar="FILEKEY=PATH")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--hfov", type=float, default=150.0)
    ap.add_argument("--vfov", type=float, default=90.0)
    ap.add_argument("--depth_m", type=float, default=0.6)
    ap.add_argument("--out_json")
    a = ap.parse_args()
    if a.self_test:
        raise SystemExit(0 if _self_test() else 1)
    if not a.calibration or not a.video:
        ap.error("--calibration and --video are required")

    from src.rig.calibration import RigCalibration
    from src.rig.geometry import VirtualWideCamera, coverage
    rig = RigCalibration(a.calibration)
    vcam = VirtualWideCamera.from_rig(rig, hfov_deg=a.hfov, vfov_deg=a.vfov)
    videos = dict(s.split("=", 1) for s in a.video)
    frames = list(range(a.start, a.start + a.n * a.stride, a.stride))

    cov = coverage(rig, vcam, a.depth_m)
    print(f"seam audit  hfov {a.hfov:g}deg vfov {a.vfov:g}deg  "
          f"{len(frames)} frames from {a.start}")
    print(f"  black invalid region {1-cov['visible'].mean():.2%}   "
          f"stereo depth available {cov['stereo'].mean():.1%}")

    rows, owner0 = audit_clip(rig, vcam, videos, frames, a.depth_m)
    if owner0 is not None:
        for i, m in enumerate(rig.modules):
            share = (owner0 == i).mean()
            if share > 0:
                print(f"  {m.name} owns {share:6.1%}")
    report(rows)
    if a.out_json:
        json.dump(rows, open(a.out_json, "w"), indent=1, default=float)
        print(f"\n  wrote {a.out_json}")
    print("\n  This says whether THIS renderer has a problem worth replacing. "
          "It does not\n  say whether a learned stitcher would do better -- "
          "that costs a comparison,\n  and the comparison is only worth "
          "running if something here fails.")


if __name__ == "__main__":
    main()
