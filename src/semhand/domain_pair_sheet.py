"""The hands the two projections disagree about, put to a person side by side.

WHY THIS AND NOT A RETRAIN. 44 of 771 paired hands get a different ownership
verdict in the two domains, and only 5 of those sit close enough to 0.5 to be
jitter -- 8 have both sides confident in opposite directions, one running
0.999 in raw against 0.232 in panorama. That is not a calibration offset and
no threshold moves it.

But disagreement is not error. Which domain is RIGHT is unknown, and it is the
only thing that decides whether a raw-domain rebuild would buy anything: if
the raw answer is the correct one on these cases, a model trained on panorama
is already doing better where the domains part company, and there is nothing
to fix. The question needs a person, and it needs one on 44 items.

BOTH PICTURES, SIDE BY SIDE, BECAUSE THE PAIR IS THE POINT. The reader is not
being asked which rendering looks nicer; they are being asked whose hand it
is, once, with both views of the same instant available, because a hand that
is ambiguous in one projection is often obvious in the other -- and that is
exactly the information the model has in one domain and not the other.

NEITHER MODEL'S ANSWER IS ON THE PICTURE. The probabilities and which side
flipped are in the key file. A reader shown "raw says self, panorama says
other" is being asked to arbitrate rather than to look.

WHAT THE ANSWER FEEDS. Correct-in-raw against correct-in-panorama, split by
OWNER and OTHER. If raw wins, the domain gap is real and harmless. If panorama
wins on the wearer's own hands and those propagate to a mosaic, the rebuild
has a measured reason. 28 of the 44 were traced into the delivered video and
4 ended up mosaicked, so the propagation is already bounded.
"""
from __future__ import annotations

import argparse
import collections
import csv
import os

DEPTH_M = 0.6


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", default="/workspace/domain_pair.csv")
    ap.add_argument("--jobs", default="/workspace/cam3_jobs.txt")
    ap.add_argument("--cell", type=int, default=760)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import cv2
    import numpy as np
    from src.rig.calibration import RigCalibration
    from src.rig.geometry import VirtualWideCamera, source_maps
    from src.rig.render_wide import render
    from src.rig.seam_fix import RawCameraReader

    rows = [r for r in csv.DictReader(open(a.pairs))]
    for r in rows:
        for k in ("p_own_raw", "p_own_pano", "cx_raw", "cy_raw"):
            r[k] = float(r[k])
        r["frame"] = int(float(r["frame"]))
        r["w_raw"] = int(float(r["w_raw"]))
    fl = [r for r in rows
          if (r["p_own_raw"] >= 0.5) != (r["p_own_pano"] >= 0.5)]
    print("翻转 %d / %d" % (len(fl), len(rows)))

    jobs = {}
    for line in open(a.jobs):
        f = line.strip().split("|")
        if len(f) >= 4 and not line.startswith("#"):
            jobs[f[0]] = (f[1], int(f[2]), int(f[3]))

    for sub in ("context", "crops"):
        os.makedirs(os.path.join(a.out, sub), exist_ok=True)
    by_rec = collections.defaultdict(list)
    for r in fl:
        by_rec[r["rec"]].append(r)

    out, key = [], []
    for rec in sorted(by_rec):
        bag = jobs[rec][0]
        rig = RigCalibration(os.path.join(bag, "calibration.yaml"))
        vcam = VirtualWideCamera.from_rig(rig)
        mx, my, valid = source_maps(rig, "cam3", vcam, depth_m=DEPTH_M)
        vids = {k: os.path.join(bag, f"{k}.mp4")
                for k in ("cam12", "cam34", "cam56")}
        want = sorted({r["frame"] for r in by_rec[rec]})
        readers = {c: RawCameraReader(vids, c, want[0])
                   for c in ("cam1", "cam2", "cam3", "cam4", "cam5", "cam6")}
        cur = want[0] - 1
        cache = {}
        frames = {}
        for f in want:
            src = {}
            while cur < f:
                for c, rd in readers.items():
                    src[c] = rd.next()
                cur += 1
            if any(v is None for v in src.values()):
                break
            frames[f] = (src["cam3"], render(rig, vcam, src, DEPTH_M,
                                             map_cache=cache)[0])
        for rd in readers.values():
            rd.close()

        for j, r in enumerate(sorted(by_rec[rec], key=lambda x: x["frame"])):
            got = frames.get(r["frame"])
            if got is None:
                continue
            raw, pano = got
            H, W = raw.shape[:2]
            cx, cy = r["cx_raw"] * W, r["cy_raw"] * H
            w = r["w_raw"]
            b = [int(cx - w / 2), int(cy - w / 2), int(cx + w / 2), int(cy + w / 2)]
            m_in = (valid & (mx >= b[0]) & (mx < b[2])
                    & (my >= b[1]) & (my < b[3]))
            if m_in.sum() < 50:
                continue
            ys, xs = np.nonzero(m_in)
            pb = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
            pano_bgr = pano[:, :, ::-1] if pano.shape[2] == 3 else pano

            def panel(img, box, tag):
                v = img.copy()
                cv2.rectangle(v, (box[0], box[1]), (box[2], box[3]),
                              (0, 230, 0), 5)
                cv2.putText(v, tag, (24, 56), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                            (0, 255, 255), 3, cv2.LINE_AA)
                sc = a.cell / float(v.shape[1])
                return cv2.resize(v, (a.cell, int(v.shape[0] * sc)))

            pr, pp = panel(raw, b, "RAW cam3"), panel(pano_bgr, pb, "PANORAMA")
            h = max(pr.shape[0], pp.shape[0])
            sheet = np.zeros((h, a.cell * 2, 3), np.uint8)
            sheet[:pr.shape[0], :a.cell] = pr
            sheet[:pp.shape[0], a.cell:] = pp
            stem = "%s_f%06d_h%d" % (rec, r["frame"], j)
            cv2.imwrite(os.path.join(a.out, "context", stem + ".jpg"), sheet,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 84])
            pad = max(120, w // 2)
            y0, y1 = max(0, b[1] - pad), min(H, b[3] + pad)
            x0, x1 = max(0, b[0] - pad), min(W, b[2] + pad)
            crop = raw[y0:y1, x0:x1]
            if crop.size:
                cv2.imwrite(os.path.join(a.out, "crops", stem + ".jpg"),
                            cv2.resize(crop, (192, 192)),
                            [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            out.append({"stem": stem, "frame": r["frame"], "hand": j,
                        "conf": float(r["det_conf"]), "w_frac": 0.0,
                        "h_frac": 0.0, "cx_frac": 0.0, "cy_frac": 0.0,
                        "model": "domain_pair", "label": "", "label_mode": ""})
            key.append({"stem": stem, "rec": rec, "frame": r["frame"],
                        "w_raw": w, "p_own_raw": round(r["p_own_raw"], 4),
                        "p_own_pano": round(r["p_own_pano"], 4),
                        "raw_says": "self" if r["p_own_raw"] >= 0.5 else "other",
                        "pano_says": "self" if r["p_own_pano"] >= 0.5 else "other"})
        print("  %-16s %d 对" % (rec, len(by_rec[rec])), flush=True)

    with open(os.path.join(a.out, "hands.csv"), "w", newline="") as fh:
        w_ = csv.DictWriter(fh, fieldnames=list(out[0]))
        w_.writeheader()
        w_.writerows(out)
    with open(a.out.rstrip("/") + "_key.csv", "w", newline="") as fh:
        w_ = csv.DictWriter(fh, fieldnames=list(key[0]))
        w_.writeheader()
        w_.writerows(key)
    with open(os.path.join(a.out, "MAPPING.txt"), "w") as fh:
        fh.write("问题：绿框里的手是佩戴者自己的，还是别人的？\n"
                 "左边是原始鱼眼帧，右边是同一时刻的全景渲染，框是同一只物理手。\n"
                 "佩戴者 -> owner   别人 -> other   看不清 -> unsure\n"
                 "两个模型的答案在 %s_key.csv，不在这张表里。\n"
                 % a.out.rstrip("/"))
    print("-> %s (%d)  两边答案另存 %s_key.csv"
          % (a.out, len(out), a.out.rstrip("/")))


if __name__ == "__main__":
    main()
