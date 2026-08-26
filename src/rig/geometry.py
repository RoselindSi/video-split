"""The virtual wide camera the six modules render into.

The rig's three stereo modules sit on a planar fan, and the fan's own rotation
axis -- not the reference camera's vertical -- is what makes a wide view look
upright. cam1 is mounted rolled: in its raw frame the workbench edge runs at
about 60 degrees and the wearer's arm enters from the lower left. Rendered
into a camera whose up is the fan axis, the bench is level and the arm enters
from the bottom, which is what an egocentric view is supposed to look like.

    module A   cam1 cam2    azimuth  -30.1 deg   elevation -8.0 deg
    module B   cam3 cam4    azimuth    0.0 deg   elevation  0.0 deg   <- forward
    module C   cam5 cam6    azimuth  +30.4 deg   elevation -8.2 deg

The +/-8 degree elevation on the outer modules is a real bow in the fan, not a
fitting error: the six optical axes are coplanar to a third singular value of
0.003 against 0.996, but the plane is not exactly the one through the module
centres.

THE ONE THING GEOMETRY CANNOT DECIDE is which way along the fan axis is up --
both signs are equally valid rotations. It is fixed here by observation and
recorded as a constant: with `FAN_UP_SIGN` as written, the wearer's own arm
enters the rendered frame from the BOTTOM. That is also the signal the
hand-ownership rule depends on, so the two modules must agree on it, and
flipping this constant silently inverts both.

WHY A VIRTUAL CAMERA RATHER THAN A PANORAMA STITCH. The output is rendered
FROM a 3D reconstruction, not warped from images: each module supplies metric
depth from its own 60 mm pair, points go into the rig frame, and the renderer
resolves overlaps with a z-buffer. A blend would average two views of the same
near hand into two hands. Rendering is also what keeps a second viewpoint
cheap -- `eye` shifts the virtual optical centre, so a stereo pair later is a
parameter rather than a second pipeline.

Production emits ONE wide RGB plus its depth. The second viewpoint stays
available and switched off.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Determined by looking at a rendered frame, not derivable from the extrinsics.
# With this sign the wearer's arm enters from the bottom of the frame.
FAN_UP_SIGN = -1.0

# The fan spans about 60 deg of azimuth between the outer module axes, and each
# module sees roughly 130 deg, so the union reaches about 176 deg. The default
# output is a little narrower than the union to avoid the extreme fisheye edge,
# where the calibration is least constrained and the pixels are most stretched.
DEFAULT_HFOV_DEG = 150.0
DEFAULT_VFOV_DEG = 90.0
DEFAULT_SIZE = (1600, 900)

PLANARITY_MAX = 0.02       # third singular value of the six optical axes


def fan_basis(rig):
    """(right, up, forward) of the fan, expressed in the reference frame.

    forward is the middle module's axis; up is the fan's rotation axis, which
    is the direction the six optical axes vary LEAST along. A rig whose axes
    are not coplanar has no such direction and is refused rather than fitted."""
    from src.rig.calibration import CalibrationError
    names = sorted(rig.cameras)
    axes = np.stack([rig.cameras[n].axis for n in names])
    _, s, vt = np.linalg.svd(axes - axes.mean(0))
    if s[2] / s[0] > PLANARITY_MAX:
        raise CalibrationError(
            f"the six optical axes are not coplanar: singular values "
            f"{s.round(4)}.\n  This module assumes a planar fan; a rig with "
            f"another layout needs its own\n  virtual-camera definition rather "
            f"than a best-fit plane through it.")
    up = vt[2] / np.linalg.norm(vt[2])
    if up[1] > 0:                       # y is down; start from the y<0 branch
        up = -up
    up = FAN_UP_SIGN * up
    mods = rig.modules
    mid = mods[len(mods) // 2]
    fwd = mid.left.axis + mid.right.axis
    fwd = fwd / np.linalg.norm(fwd)
    right = np.cross(up, fwd)
    right /= np.linalg.norm(right)
    up = np.cross(fwd, right)
    up /= np.linalg.norm(up)
    return right, up, fwd


@dataclass(frozen=True)
class VirtualWideCamera:
    """Equirectangular, centred on the fan.

    `eye` is the optical centre in the reference frame. It is zero for the
    production view; a second viewpoint is this same class with `eye` shifted
    along `right`, which is the whole cost of stereo output later."""
    R: np.ndarray                  # 3x3, virtual -> reference
    eye: np.ndarray                # 3,   metres, in the reference frame
    width: int
    height: int
    hfov: float                    # radians
    vfov: float

    @classmethod
    def from_rig(cls, rig, size=DEFAULT_SIZE, hfov_deg=DEFAULT_HFOV_DEG,
                 vfov_deg=DEFAULT_VFOV_DEG, eye=None):
        right, up, fwd = fan_basis(rig)
        return cls(R=np.stack([right, up, fwd], 1),
                   eye=np.zeros(3) if eye is None else np.asarray(eye, float),
                   width=size[0], height=size[1],
                   hfov=np.radians(hfov_deg), vfov=np.radians(vfov_deg))

    def directions(self):
        """[H,W,3] unit rays in the REFERENCE frame, one per output pixel."""
        xx, yy = np.meshgrid(np.arange(self.width), np.arange(self.height))
        az = (xx / self.width - 0.5) * self.hfov
        el = -(yy / self.height - 0.5) * self.vfov
        d = np.stack([np.sin(az) * np.cos(el), -np.sin(el),
                      np.cos(az) * np.cos(el)], -1)
        return d @ self.R.T

    def angles_of(self, v):
        """(azimuth, elevation) in degrees for a reference-frame direction."""
        w = self.R.T @ np.asarray(v, float)
        return (float(np.degrees(np.arctan2(w[0], w[2]))),
                float(np.degrees(np.arcsin(-w[1] / np.linalg.norm(w)))))


def source_maps(rig, camera, vcam, depth_m=None):
    """(map_x, map_y, valid) sampling `camera` for every output pixel.

    `depth_m` matters and defaults to infinity on purpose. A ray only becomes a
    POINT once it has a depth, and the parallax between the virtual centre and
    a physical camera 150 mm away is exactly what the depth pass exists to
    resolve. Passing a constant here is the cheap approximation -- correct only
    at that distance -- and the real renderer replaces it with the per-pixel
    depth. It is kept because a constant-depth render is the honest baseline to
    compare the depth-aware one against."""
    import cv2
    cam = rig.cameras[camera]
    d = vcam.directions().reshape(-1, 3)
    if depth_m is None:
        p_ref = d                                # direction only
    else:
        p_ref = vcam.eye + d * float(depth_m)
    p_cam = (cam.R.T @ (p_ref - (0 if depth_m is None else cam.t)).T).T
    ok = p_cam[:, 2] > 1e-6
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


def coverage(rig, vcam, depth_m=1.0):
    """Per-pixel counts: how many cameras see it, and does any MODULE see it.

    A module seeing a direction with BOTH eyes is what makes stereo depth
    available there. Measured on this rig: 176 deg visible, 170 deg with
    stereo, a 6 deg gap only at the extreme edges -- so the observability
    concern about single-view regions does not apply, because the stereo is
    inside each module rather than between them."""
    vis = {n: source_maps(rig, n, vcam, depth_m)[2] for n in rig.cameras}
    n_cams = np.sum([vis[n] for n in vis], 0)
    stereo = np.zeros_like(n_cams, bool)
    per_module = {}
    for m in rig.modules:
        s = vis[m.left.name] & vis[m.right.name]
        per_module[m.name] = s
        stereo |= s
    return {"visible": n_cams > 0, "n_cameras": n_cams,
            "stereo": stereo, "per_module": per_module,
            "per_camera": vis}


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.
                                 RawDescriptionHelpFormatter)
    ap.add_argument("calibration")
    ap.add_argument("--depth_m", type=float, default=1.0)
    a = ap.parse_args()
    from src.rig.calibration import RigCalibration
    rig = RigCalibration(a.calibration)
    vcam = VirtualWideCamera.from_rig(rig)
    right, up, fwd = fan_basis(rig)
    print(f"virtual wide camera  {vcam.width}x{vcam.height}  "
          f"hfov {np.degrees(vcam.hfov):.0f} deg  "
          f"vfov {np.degrees(vcam.vfov):.0f} deg")
    print(f"  right {right.round(3)}\n  up    {up.round(3)}  "
          f"(FAN_UP_SIGN={FAN_UP_SIGN:+.0f}: wearer's arm enters from the "
          f"bottom)\n  fwd   {fwd.round(3)}")
    print("\n  module placement in the output:")
    for m in rig.modules:
        az, el = vcam.angles_of(m.left.axis + m.right.axis)
        print(f"    {m.name}  {m.left.name}|{m.right.name}  "
              f"azimuth {az:+6.1f} deg  elevation {el:+5.1f} deg")
    cov = coverage(rig, vcam, a.depth_m)
    tot = cov["visible"].size
    print(f"\n  at {a.depth_m} m, over the {np.degrees(vcam.hfov):.0f} deg "
          f"output:")
    print(f"    visible by some camera   {cov['visible'].mean():6.1%}")
    print(f"    stereo from some module  {cov['stereo'].mean():6.1%}")
    gap = cov["visible"] & ~cov["stereo"]
    print(f"    visible without depth    {gap.mean():6.1%}")
    for name, s in cov["per_module"].items():
        print(f"      {name} stereo {s.mean():6.1%}")


if __name__ == "__main__":
    main()
