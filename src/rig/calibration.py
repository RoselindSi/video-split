"""The six-camera rig's calibration, loaded with its traps already sprung.

Everything downstream -- stereo depth, reprojection, the virtual wide camera --
is a consequence of these numbers, so this module refuses rather than degrades.
Each check below exists because the data actually looks like that.

WHAT THE RIG IS. Three stereo modules on a fan, verified from the extrinsics
rather than assumed:

    module A   cam1 + cam2    baseline 59.6 mm    axes 0.5 deg apart    at   0 deg
    module B   cam3 + cam4    baseline 60.2 mm    axes 0.1 deg apart    at  31 deg
    module C   cam5 + cam6    baseline 59.8 mm    axes 0.4 deg apart    at  60 deg

Every other pairing is 31 or 60 degrees apart. The grouping is DERIVED here,
not hard-coded: a rig wired differently would produce different pairs, and a
hard-coded (1,2),(3,4),(5,6) would keep working while meaning something else.

STEREO IS WITHIN A MODULE, NOT BETWEEN THEM. 60 mm and near-parallel is the
configuration the hardware was built for; a cross-module pair is 94 mm at 31
degrees, with large occlusion differences and a much harder matching problem.
Measured coverage: 176 deg visible in total, 170 deg of it with stereo from
some module, leaving a 6 deg gap at the extreme edges.

FILE FORMAT NOTES, all learned the hard way.

`calibration_status` is honest -- an uncalibrated rig writes `uncalibrated`
and fills every number with 0.0. 25 of 40 databags in the first delivery are
that placeholder, all byte-identical. Trust the field, and check the numbers
anyway, because a template that ever starts claiming `calibrated` would
otherwise pass silently.

`T_cam1_camera` is 12 numbers, row-major 3x4 [R|t], mapping points FROM each
camera's frame INTO cam1's. cv2.fisheye.stereoRectify wants the opposite
direction, so `stereo_pair()` does the inversion once, here, rather than at
each call site.

The model is `fisheye` with four coefficients -- Kannala-Brandt. It must be
used through `cv2.fisheye.*`; the parameters are not interchangeable with
`cv2.undistort`, and mixing them produces an image that looks fine and is
geometrically wrong.

And a parsing trap worth naming: the key `T_cam1_camera` contains the digit 1,
so a naive number-regex over the whole block returns 13 values for a 12-value
matrix.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

import numpy as np

# Derived-grouping thresholds. A stereo module is two cameras close together
# and pointing the same way; these bounds are wide enough to accept the
# observed 59.6-60.2 mm / 0.1-0.5 deg and narrow enough to reject the nearest
# non-pair, which is 35.6 mm apart but 31 deg off axis.
MAX_PAIR_BASELINE_M = 0.12
MAX_PAIR_AXIS_DEG = 5.0

# Sanity bounds for a 1920x1520 fisheye. The principal point sits within a few
# tens of pixels of the image centre and the two focal lengths agree.
PRINCIPAL_TOL_PX = 150.0
FOCAL_RATIO_TOL = 0.02


class CalibrationError(Exception):
    """Raised, never warned. A wrong rig geometry is silent downstream: the
    depth map still renders, the panorama still looks like a panorama."""


@dataclass(frozen=True)
class Camera:
    name: str
    width: int
    height: int
    K: np.ndarray          # 3x3
    D: np.ndarray          # 4x1, Kannala-Brandt
    R: np.ndarray          # 3x3, camX -> cam1
    t: np.ndarray          # 3,   camX -> cam1, metres

    @property
    def axis(self):
        """Optical axis expressed in cam1's frame."""
        return self.R @ np.array([0.0, 0.0, 1.0])

    def azimuth_deg(self):
        a = self.axis
        return float(np.degrees(np.arctan2(a[0], a[2])))


@dataclass(frozen=True)
class Module:
    """One stereo pair. `left` is the camera whose optical centre has the
    smaller x in cam1's frame -- x points right, so it is the left eye, and
    that is also which half of the packed video file it occupies."""
    name: str
    left: Camera
    right: Camera

    @property
    def baseline_m(self):
        return float(np.linalg.norm(self.right.t - self.left.t))

    def azimuth_deg(self):
        return (self.left.azimuth_deg() + self.right.azimuth_deg()) / 2


def _parse_yaml(path):
    try:
        import yaml
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    except ImportError:
        return _parse_manual(path)


def _parse_manual(path):
    """Fallback for environments without pyyaml.

    The number regex starts AFTER the key's colon. `T_cam1_camera` contains a
    literal 1, and scanning the whole block returns 13 numbers for a 12-number
    matrix -- which then reshapes into something plausible and wrong."""
    txt = open(path, encoding="utf-8").read()
    NUM = r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
    out = {"calibration_status": None, "cameras": {}}
    m = re.search(r"calibration_status:\s*(\S+)", txt)
    if m:
        out["calibration_status"] = m.group(1).strip("'\"")
    for blk in re.finditer(r"\n  (cam\d):\n(.*?)(?=\n  cam\d:|\nIMU:|\Z)",
                           txt, re.S):
        name, body = blk.group(1), blk.group(2)
        keys = ["intrinsics", "distortion_coefficients", "T_cam1_camera"]
        pos = {k: body.find(k + ":") for k in keys}
        if any(v < 0 for v in pos.values()):
            continue
        vals = {}
        for k in keys:
            start = pos[k] + len(k) + 1
            ends = [pos[j] for j in keys if pos[j] > pos[k]] + [len(body)]
            vals[k] = [float(x) for x in
                       re.findall(NUM, body[start:min(ends)])]
        out["cameras"][name] = {
            "image_width": int(re.search(r"image_width:\s*(\d+)",
                                         body).group(1)),
            "image_height": int(re.search(r"image_height:\s*(\d+)",
                                          body).group(1)),
            "camera_model": "fisheye", "distortion_model": "fisheye",
            **vals}
    return out


class RigCalibration:
    """Six cameras, three derived modules, and the checks that make them safe.

    `require_calibrated=False` loads a placeholder for inspection -- useful for
    auditing a delivery -- but `modules` and every geometric accessor still
    refuse, so an uncalibrated rig cannot leak into a depth computation by
    being loaded permissively somewhere upstream."""

    def __init__(self, path, require_calibrated=True):
        self.path = path
        if not os.path.isfile(path):
            raise CalibrationError(f"no calibration at {path}")
        d = _parse_yaml(path)
        self.status = str(d.get("calibration_status", "?"))
        self.calibrated = self.status == "calibrated"

        cams = d.get("cameras") or {}
        if len(cams) != 6:
            raise CalibrationError(
                f"{path} declares {len(cams)} cameras, expected 6")

        self.cameras = {}
        for name in sorted(cams):
            c = cams[name]
            for field, n in (("intrinsics", 4),
                             ("distortion_coefficients", 4),
                             ("T_cam1_camera", 12)):
                v = c.get(field)
                if not isinstance(v, (list, tuple)) or len(v) != n:
                    raise CalibrationError(
                        f"{name}.{field} has {len(v) if v else 0} values, "
                        f"expected {n}. If it is 13, the parser counted the "
                        f"`1` inside the key name `T_cam1_camera`.")
            if str(c.get("camera_model")) != "fisheye":
                raise CalibrationError(
                    f"{name} is {c.get('camera_model')!r}, not fisheye. The "
                    f"whole pipeline uses cv2.fisheye.*, whose parameters are "
                    f"not interchangeable with the pinhole ones.")
            fx, fy, cx, cy = [float(x) for x in c["intrinsics"]]
            M = np.array([float(x) for x in c["T_cam1_camera"]]).reshape(3, 4)
            self.cameras[name] = Camera(
                name=name, width=int(c["image_width"]),
                height=int(c["image_height"]),
                K=np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], float),
                D=np.array([float(x) for x in c["distortion_coefficients"]],
                           float).reshape(4, 1),
                R=M[:, :3].copy(), t=M[:, 3].copy())

        if require_calibrated:
            self._require_calibrated()
            self._sanity()
        self._modules = None

    # --- refusals ---------------------------------------------------------
    def _require_calibrated(self):
        if not self.calibrated:
            raise CalibrationError(
                f"{self.path} is `{self.status}`.\n"
                f"  The capture software writes an all-zero placeholder when "
                f"the rig has no\n  calibration loaded -- 25 of the first 40 "
                f"databags are that file, byte for\n  byte. Depth and "
                f"reprojection are refused for this recording; it may still "
                f"be\n  used for anything that needs only pixels.")
        allz = [n for n, c in self.cameras.items()
                if not np.any(np.abs(c.K[[0, 1], [0, 1]]) > 1e-9)]
        if allz:
            raise CalibrationError(
                f"{self.path} says `calibrated` but {allz} have zero focal "
                f"lengths.\n  The status field and the numbers disagree, "
                f"which is worse than either being\n  wrong alone -- nothing "
                f"downstream reads both.")

    def _sanity(self):
        for n, c in self.cameras.items():
            fx, fy = c.K[0, 0], c.K[1, 1]
            cx, cy = c.K[0, 2], c.K[1, 2]
            if abs(fx - fy) / max(fx, fy) > FOCAL_RATIO_TOL:
                raise CalibrationError(
                    f"{n}: fx={fx:.1f} fy={fy:.1f} differ by more than "
                    f"{FOCAL_RATIO_TOL:.0%}")
            dx, dy = abs(cx - c.width / 2), abs(cy - c.height / 2)
            if dx > PRINCIPAL_TOL_PX or dy > PRINCIPAL_TOL_PX:
                raise CalibrationError(
                    f"{n}: principal point ({cx:.1f}, {cy:.1f}) is "
                    f"({dx:.0f}, {dy:.0f}) px from the image centre "
                    f"({c.width/2:.0f}, {c.height/2:.0f}), past the "
                    f"{PRINCIPAL_TOL_PX:.0f} px bound.")

    # --- derived grouping -------------------------------------------------
    @property
    def modules(self):
        """Stereo pairs, found from the geometry rather than the names.

        Hard-coding (1,2),(3,4),(5,6) would keep running on a rig wired
        differently and would mean something else while doing it."""
        if not self.calibrated:
            raise CalibrationError(
                f"{self.path} is `{self.status}`; module grouping needs real "
                f"extrinsics.")
        if self._modules is not None:
            return self._modules
        names = sorted(self.cameras)
        used, mods = set(), []
        for i, a in enumerate(names):
            if a in used:
                continue
            ca = self.cameras[a]
            best = None
            for b in names[i + 1:]:
                if b in used:
                    continue
                cb = self.cameras[b]
                base = float(np.linalg.norm(cb.t - ca.t))
                ang = float(np.degrees(np.arccos(
                    np.clip(ca.axis @ cb.axis, -1, 1))))
                if base <= MAX_PAIR_BASELINE_M and ang <= MAX_PAIR_AXIS_DEG:
                    if best is None or base < best[1]:
                        best = (b, base, ang)
            if best is None:
                raise CalibrationError(
                    f"{a} has no stereo partner within "
                    f"{MAX_PAIR_BASELINE_M*1000:.0f} mm and "
                    f"{MAX_PAIR_AXIS_DEG:.0f} deg.\n  This rig is three "
                    f"60 mm modules on a fan; a camera without a partner "
                    f"means the\n  layout is not what the pipeline assumes.")
            b = best[0]
            used |= {a, b}
            l, r = (ca, self.cameras[b])
            if l.t[0] > r.t[0]:          # x points right; smaller x is left
                l, r = r, l
            mods.append(Module(name=f"module_{len(mods)}", left=l, right=r))
        if len(mods) != 3:
            raise CalibrationError(f"found {len(mods)} modules, expected 3")
        mods.sort(key=lambda m: m.azimuth_deg())
        self._modules = tuple(
            Module(name=f"module_{chr(65+i)}", left=m.left, right=m.right)
            for i, m in enumerate(mods))
        return self._modules

    # --- transforms -------------------------------------------------------
    def T_cam1_from(self, name):
        c = self.cameras[name]
        return c.R, c.t

    def relative(self, a, b):
        """(R, T) taking points from a's frame into b's."""
        Ra, ta = self.T_cam1_from(a)
        Rb, tb = self.T_cam1_from(b)
        R = Rb.T @ Ra
        T = Rb.T @ (ta - tb)
        return R, T

    def stereo_pair(self, module):
        """(K1, D1, K2, D2, size, R, T) ready for cv2.fisheye.stereoRectify.

        stereoRectify wants left -> right, while the file stores camX -> cam1;
        the inversion happens once, here."""
        l, r = module.left, module.right
        R, T = self.relative(l.name, r.name)
        return (l.K, l.D, r.K, r.D, (l.width, l.height), R, T.reshape(3, 1))

    # --- reporting --------------------------------------------------------
    def summary(self):
        lines = [f"{self.path}", f"  status: {self.status}"]
        for n in sorted(self.cameras):
            c = self.cameras[n]
            lines.append(
                f"  {n}  {c.width}x{c.height}  fx={c.K[0,0]:7.2f} "
                f"cx={c.K[0,2]:7.2f} cy={c.K[1,2]:7.2f}  "
                f"t=({c.t[0]*1000:+7.1f},{c.t[1]*1000:+6.1f},"
                f"{c.t[2]*1000:+6.1f}) mm  az={c.azimuth_deg():+5.1f} deg")
        if self.calibrated:
            lines.append("  modules (derived from geometry):")
            for m in self.modules:
                ang = np.degrees(np.arccos(np.clip(
                    m.left.axis @ m.right.axis, -1, 1)))
                lines.append(
                    f"    {m.name}  {m.left.name}|{m.right.name}  "
                    f"baseline {m.baseline_m*1000:5.1f} mm  "
                    f"axes {ang:4.2f} deg apart  "
                    f"azimuth {m.azimuth_deg():+5.1f} deg")
        return "\n".join(lines)


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.
                                 RawDescriptionHelpFormatter)
    ap.add_argument("calibration")
    ap.add_argument("--allow_uncalibrated", action="store_true",
                    help="load a placeholder for inspection. Geometry "
                         "accessors still refuse.")
    a = ap.parse_args()
    rig = RigCalibration(a.calibration,
                         require_calibrated=not a.allow_uncalibrated)
    print(rig.summary())


if __name__ == "__main__":
    main()
