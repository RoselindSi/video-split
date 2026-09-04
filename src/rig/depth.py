"""Metric depth from each module's own 60 mm stereo pair.

Depth is computed WITHIN a module, never between them. The three modules sit
31 degrees apart with a 94 mm baseline and see very different occlusions; a
module's own two eyes are 60 mm apart with their axes 0.1 to 0.5 degrees from
parallel, which is the configuration the hardware was built for and the one a
stereo matcher can actually solve.

WHY THIS IS NOT HERE TO FIX THE SEAMS. The first plan was depth-aware
reprojection with a z-buffer, on the theory that the joins between modules
carried parallax ghosting. Measured, they do not: the systematic shift at both
joins is under 0.2 and 2.6 px at every assumed depth, and the per-patch shift
field has an incoherence near 1.0 -- the value expected of pure noise, against
near 0 for a real displacement field. The joins sit in the fisheye periphery
where neither phase correlation nor a stereo matcher nor a viewer finds much,
and the constant-depth composite is already adequate for RGB.

So depth exists for its own sake. It is half the deliverable -- WIDE_RGB plus
WIDE_DEPTH -- and it is the strongest signal available for deciding whose
hands are in frame: the wearer's own sit at 0.2 to 0.6 m and a colleague
across the bench is past 1.5 m, which `hand_ownership` uses when supplied.

SGBM FIRST, ON PURPOSE. It is not the best matcher available; Fast-
FoundationStereo is. But SGBM needs no weights, no network and no GPU, so the
geometry can be verified end to end before a learned model is introduced --
and when the learned model lands, this is the baseline that says what it
bought. The rectification, the Q matrix, the disparity-to-metres conversion
and the reprojection into the rig frame are all shared; only `disparity()`
changes.

WHAT COMES OUT AND WHAT DOES NOT. Depth is valid where the matcher found a
correspondence, which excludes the textureless bench top over much of the
frame. It is returned with its validity mask rather than filled in, because a
plausible interpolated depth is indistinguishable from a measured one once it
is written to a file, and the downstream hand rule would rather abstain than
be told 0.5 m by an interpolator.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# The rectified pair is searched over this range. 192 disparities at the
# rectified focal length of about 620 px and a 60 mm baseline reaches from
# 0.19 m outward; anything nearer than that is the wearer's own sleeve.
NUM_DISPARITIES = 192
MIN_DISPARITY = 0
BLOCK_SIZE = 7

# Depths outside this are rejected outright. The rig sees a work surface: the
# measured distribution on this data is a median 0.43 m with the 5th and 95th
# percentiles at 0.19 and 4.06 m, so a returned 40 m is a matcher failure
# rather than a distant wall.
MIN_DEPTH_M = 0.12
MAX_DEPTH_M = 8.0


@dataclass
class ModuleDepth:
    """One module's depth, in ITS OWN rectified frame."""
    module: str
    depth_m: np.ndarray        # [H,W] float32, NaN where invalid
    disparity: np.ndarray      # [H,W] float32
    valid: np.ndarray          # [H,W] bool
    Q: np.ndarray              # 4x4, disparity-to-3D for this rectification
    R_rect: np.ndarray         # 3x3, rectified frame -> left camera frame
    left_rect: np.ndarray      # the rectified left image, for colour lookup
    size: tuple

    def coverage(self):
        return float(self.valid.mean())


def rectify_maps(rig, module, size=None, balance=0.0, fov_scale=1.0):
    """Fisheye rectification for one module. Cached by the caller if needed."""
    import cv2
    K1, D1, K2, D2, sensor_size, R, T = rig.stereo_pair(module)
    output_size = tuple(size or sensor_size)
    R1, R2, P1, P2, Q = cv2.fisheye.stereoRectify(
        K1, D1, K2, D2, sensor_size, R, T, cv2.CALIB_ZERO_DISPARITY,
        newImageSize=output_size, balance=balance, fov_scale=fov_scale)
    m1 = cv2.fisheye.initUndistortRectifyMap(K1, D1, R1, P1, output_size,
                                             cv2.CV_32FC1)
    m2 = cv2.fisheye.initUndistortRectifyMap(K2, D2, R2, P2, output_size,
                                             cv2.CV_32FC1)
    return {"maps": (m1, m2), "Q": Q, "P1": P1, "R1": R1,
            "size": output_size,
            "baseline_m": float(np.linalg.norm(T))}


def disparity(left, right, num_disp=NUM_DISPARITIES, block=BLOCK_SIZE):
    """SGBM. Replaceable -- everything around it is matcher-agnostic."""
    import cv2
    sg = cv2.StereoSGBM_create(
        minDisparity=MIN_DISPARITY, numDisparities=num_disp, blockSize=block,
        P1=8 * block * block, P2=32 * block * block,
        disp12MaxDiff=1, uniquenessRatio=10,
        speckleWindowSize=100, speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    gl = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY) if left.ndim == 3 else left
    gr = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY) if right.ndim == 3 else right
    return sg.compute(gl, gr).astype(np.float32) / 16.0


def module_depth(rig, module, left, right, rect=None, matcher=None, **kw):
    """Rectify, match, and convert to metres. -> ModuleDepth."""
    import cv2
    rect = rect or rectify_maps(rig, module, **kw)
    (mx1, my1), (mx2, my2) = rect["maps"]
    lr = cv2.remap(left, mx1, my1, cv2.INTER_LINEAR)
    rr = cv2.remap(right, mx2, my2, cv2.INTER_LINEAR)
    d = (matcher or disparity)(lr, rr)

    fx = float(rect["P1"][0, 0])
    b = rect["baseline_m"]
    with np.errstate(divide="ignore", invalid="ignore"):
        z = fx * b / d
    ok = (d > MIN_DISPARITY + 0.5) & np.isfinite(z) \
        & (z >= MIN_DEPTH_M) & (z <= MAX_DEPTH_M)
    z = np.where(ok, z, np.nan).astype(np.float32)
    return ModuleDepth(module=module.name, depth_m=z, disparity=d, valid=ok,
                       Q=rect["Q"], R_rect=rect["R1"], left_rect=lr,
                       size=rect["size"])


def to_reference_points(rig, module, md, stride=1):
    """Valid pixels as 3D points in the REFERENCE frame, with their colours.

    -> (points [N,3] metres, colours [N,3] uint8). This is what any later
    virtual-view renderer consumes, and it is also how a depth map from one
    module is expressed in the same coordinates as another's."""
    import cv2
    h, w = md.depth_m.shape
    ys, xs = np.mgrid[0:h:stride, 0:w:stride]
    z = md.depth_m[::stride, ::stride]
    keep = np.isfinite(z)
    if not keep.any():
        return np.zeros((0, 3)), np.zeros((0, 3), np.uint8)
    xs, ys, z = xs[keep], ys[keep], z[keep]
    # Q maps (x, y, disparity) to a 3D point in the RECTIFIED frame. Using it
    # keeps the rectified intrinsics in one place rather than restating fx,
    # cx and the baseline here, where they could drift out of step with the
    # rectification that produced the disparity.
    disp = md.disparity[::stride, ::stride][keep]
    pts = cv2.perspectiveTransform(
        np.stack([xs, ys, disp], -1).reshape(-1, 1, 3).astype(np.float32),
        md.Q.astype(np.float32)).reshape(-1, 3)
    left_cam = (md.R_rect.T @ pts.T).T          # rectified -> left camera
    cam = rig.cameras[module.left.name]
    ref = (cam.R @ left_cam.T).T + cam.t        # left camera -> reference
    col = md.left_rect[::stride, ::stride][keep]
    return ref.astype(np.float32), col


def sample_depth(md, boxes):
    """Median valid depth inside each box, or None. For hand ownership.

    The median rather than the mean, and None rather than a guess: a box that
    is mostly textureless bench returns nothing, and `hand_ownership` treats
    that as no evidence instead of as a distance."""
    out = []
    h, w = md.depth_m.shape
    for x0, y0, x1, y1 in boxes:
        x0 = max(0, int(x0)); y0 = max(0, int(y0))
        x1 = min(w, int(x1)); y1 = min(h, int(y1))
        if x1 <= x0 or y1 <= y0:
            out.append(None)
            continue
        patch = md.depth_m[y0:y1, x0:x1]
        v = patch[np.isfinite(patch)]
        out.append(float(np.median(v)) if v.size >= 32 else None)
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--video", action="append", required=True,
                    metavar="FILEKEY=PATH")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--out_prefix")
    a = ap.parse_args()

    import cv2
    from src.rig.calibration import RigCalibration
    from src.rig.render_wide import read_frame, split_halves

    rig = RigCalibration(a.calibration)
    files = dict(s.split("=", 1) for s in a.video)

    print(f"{'module':<10}{'baseline':>10}{'coverage':>10}"
          f"{'p05':>8}{'median':>8}{'p95':>8}   metres")
    for m in rig.modules:
        key = f"cam{m.left.name[-1]}{m.right.name[-1]}"
        if key not in files:
            print(f"  {m.name}: no video ({key})")
            continue
        left, right = split_halves(read_frame(files[key], a.frame))
        rect = rectify_maps(rig, m)
        md = module_depth(rig, m, left, right, rect)
        v = md.depth_m[np.isfinite(md.depth_m)]
        if v.size:
            print(f"{m.name:<10}{rect['baseline_m']*1000:9.1f}mm"
                  f"{md.coverage():10.1%}"
                  f"{np.percentile(v,5):8.2f}{np.median(v):8.2f}"
                  f"{np.percentile(v,95):8.2f}")
        else:
            print(f"{m.name:<10}{rect['baseline_m']*1000:9.1f}mm"
                  f"{0.0:10.1%}   no valid depth")
        if a.out_prefix:
            vis = np.clip((md.disparity / NUM_DISPARITIES) * 255,
                          0, 255).astype(np.uint8)
            vis = cv2.applyColorMap(vis, cv2.COLORMAP_TURBO)
            vis[~md.valid] = 0
            cv2.imwrite(f"{a.out_prefix}_{m.name}_depth.png", vis)
            cv2.imwrite(f"{a.out_prefix}_{m.name}_left.png", md.left_rect)
    if a.out_prefix:
        print(f"\nwrote {a.out_prefix}_module_*_depth.png "
              f"(TURBO, black = no correspondence)")
    print("\n  Depth is returned with its validity, never filled in. A "
          "textureless bench\n  top has no correspondence, and an interpolated "
          "0.5 m written to a file is\n  indistinguishable from a measured "
          "one.")


if __name__ == "__main__":
    main()
