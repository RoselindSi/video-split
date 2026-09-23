"""The same physical hand, through both projections, into the same frozen student.

WHAT THIS DECIDES AND WHAT IT DOES NOT. The shipped student was trained and
accepted entirely on 1600x900 panorama renders, and `--camera cam3` hands it a
1920x1520 raw fisheye half. That the two domains differ is now certain. How
much the model's answer moves between them is not, and no amount of rereading
the code will say -- it is a measurement, and a cheap one, because it needs no
training at all.

If the answer barely moves, the cam3 numbers keep their meaning and the domain
gap is a tidiness problem. If it moves a lot, every ownership decision on cam3
was taken by a model reading pixels it was never shown, and the raw-domain
rebuild stops being optional.

THE PAIRING IS GEOMETRIC, NOT APPROXIMATE, AND THAT IS THE WHOLE DIFFICULTY.
Comparing a detection found in the panorama against a detection found in the
raw half would measure the detector's disagreement, not the classifier's. So
ONE box is found, in the raw half -- which is what deployment uses -- and it is
carried into the panorama through the rig's own sampling map: `source_maps`
gives, for every panorama pixel, the raw coordinate it samples, so the panorama
region that reads from inside the raw box is exactly the same physical content.
Its bounding box is the pair. Nothing is matched by appearance or position.

AT THE RENDER'S OWN DEPTH. The map depends on depth through parallax, and the
training frames were rendered at 0.6 m, so the pairing uses 0.6 m. A different
constant would pair a hand with a slightly different piece of bench.

PAIRS THAT CANNOT BE FORMED ARE DROPPED AND COUNTED. A raw box outside the
panorama's coverage of cam3, or one whose mapped region collapses, is not a
disagreement -- it is a hand the panorama pipeline would never have seen, and
scoring it in one domain only would compare two populations.
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
    ap.add_argument("--jobs", default="/workspace/cam3_jobs.txt")
    ap.add_argument("--ckpt", default="/workspace/distil/student/S_wide_g6_seed*.pt")
    ap.add_argument("--weights",
                    default="/shared/models/HaWoR/weights/external/detector.pt")
    ap.add_argument("--per_rec", type=int, default=40)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import glob
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from ultralytics import YOLO
    from src.rig import own_ctx
    from src.rig.calibration import RigCalibration
    from src.rig.geometry import VirtualWideCamera, source_maps
    from src.rig.hand_detect import detect
    from src.rig.render_wide import render
    from src.rig.seam_fix import RawCameraReader
    from src.semhand.crossv1 import CROP_PX, FRAME_PX, norm_rgb
    from src.semhand.qwen import views

    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = []
    for p in sorted(glob.glob(a.ckpt)):
        ck = torch.load(p, map_location=device, weights_only=False)
        m = own_ctx.build(ck["arm"], n_out=int(ck.get("n_out", 2))).to(device)
        m.load_state_dict(ck["state"])
        m.eval()
        models.append(m)
    if not models:
        raise SystemExit("no checkpoints matched " + a.ckpt)
    n_out = int(ck.get("n_out", 2))
    print("%d 个 seed，n_out=%d —— 冻结，不训练" % (len(models), n_out))
    yolo = YOLO(a.weights)

    def score(img_bgr, box, tmp):
        """One (frame, box) through the student's own transform. -> probs"""
        cv2.imwrite(tmp, img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
        frame, crop = views({"image": tmp, "box": [int(v) for v in box]})
        h = norm_rgb(np.asarray(crop.resize((CROP_PX, CROP_PX), Image.BICUBIC)))
        c = norm_rgb(np.asarray(frame.resize(FRAME_PX, Image.BICUBIC)))
        with torch.no_grad():
            pr = np.mean([torch.softmax(
                m(h[None].to(device), c[None].to(device),
                  torch.zeros(1, 14).to(device)), 1).cpu().numpy()
                for m in models], 0)[0]
        return pr

    jobs = {}
    for line in open(a.jobs):
        f = line.strip().split("|")
        if len(f) >= 4 and not line.startswith("#"):
            jobs[f[0]] = (f[1], int(f[2]), int(f[3]))

    rows = []
    dropped = collections.Counter()
    tmp_r, tmp_p = "/tmp/_pair_raw.jpg", "/tmp/_pair_pano.jpg"
    for rec in sorted(jobs):
        bag, start, n = jobs[rec]
        cal = os.path.join(bag, "calibration.yaml")
        if not os.path.exists(cal):
            dropped["无标定"] += 1
            continue
        rig = RigCalibration(cal)
        vcam = VirtualWideCamera.from_rig(rig)
        mx, my, valid = source_maps(rig, "cam3", vcam, depth_m=DEPTH_M)
        vids = {k: os.path.join(bag, f"{k}.mp4")
                for k in ("cam12", "cam34", "cam56")}
        want = list(range(start, start + n, a.stride))[:a.per_rec]
        readers = {c: RawCameraReader(vids, c, want[0])
                   for c in ("cam1", "cam2", "cam3", "cam4", "cam5", "cam6")}
        cur = want[0] - 1
        cache = {}
        got = 0
        for f in want:
            src = {}
            while cur < f:
                for c, rd in readers.items():
                    src[c] = rd.next()
                cur += 1
            if any(v is None for v in src.values()):
                break
            raw = src["cam3"]
            pano = render(rig, vcam, src, DEPTH_M, map_cache=cache)[0]
            dets = detect(yolo, cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
                          if raw.shape[2] == 3 else raw, min_conf=a.conf)
            for k, d in enumerate(dets):
                b = [int(v) for v in d["box"]]
                # THE PAIR: panorama pixels that sample from inside this box.
                m_in = (valid & (mx >= b[0]) & (mx < b[2])
                        & (my >= b[1]) & (my < b[3]))
                if m_in.sum() < 50:
                    dropped["全景里看不到"] += 1
                    continue
                ys, xs = np.nonzero(m_in)
                pb = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
                if pb[2] - pb[0] < 8 or pb[3] - pb[1] < 8:
                    dropped["映射后退化"] += 1
                    continue
                pr_raw = score(raw, b, tmp_r)
                pr_pan = score(pano[:, :, ::-1] if pano.shape[2] == 3 else pano,
                               pb, tmp_p)
                rows.append({
                    "rec": rec, "frame": f, "k": k,
                    "w_raw": b[2] - b[0], "w_pano": pb[2] - pb[0],
                    "cx_raw": round((b[0] + b[2]) / 2 / raw.shape[1], 4),
                    "cy_raw": round((b[1] + b[3]) / 2 / raw.shape[0], 4),
                    "det_conf": round(float(d.get("conf", 0)), 3),
                    "p_own_raw": round(float(pr_raw[1]), 4),
                    "p_own_pano": round(float(pr_pan[1]), 4),
                    "pred_raw": int(np.argmax(pr_raw)),
                    "pred_pano": int(np.argmax(pr_pan))})
                got += 1
        for rd in readers.values():
            rd.close()
        print("  %-16s %d 对" % (rec, got), flush=True)

    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("\n配不成对而丢弃：%s" % dict(dropped))
    report(rows)
    print("-> %s" % a.out)


def report(rows):
    import numpy as np
    d = np.array([r["p_own_pano"] - r["p_own_raw"] for r in rows])
    flip = sum(1 for r in rows if (r["p_own_raw"] >= 0.5) != (r["p_own_pano"] >= 0.5))
    print("\n%d 对配上的手" % len(rows))
    print("  Δp_owner（全景 − 原始）中位 %+.3f  |Δ| 中位 %.3f  p90 %.3f"
          % (np.median(d), np.median(np.abs(d)), np.percentile(np.abs(d), 90)))
    print("  两个域给出不同归属判定的：%d / %d = %.1f%%"
          % (flip, len(rows), 100.0 * flip / len(rows)))
    print("\n%-14s %6s %10s %10s" % ("分层", "n", "|Δ| 中位", "判定翻转"))
    strata = [
        ("小手 <150px", lambda r: r["w_raw"] < 150),
        ("大手 >=150px", lambda r: r["w_raw"] >= 150),
        ("中央 |x-.5|<.2", lambda r: abs(r["cx_raw"] - 0.5) < 0.2),
        ("边缘 |x-.5|>=.3", lambda r: abs(r["cx_raw"] - 0.5) >= 0.3),
        ("原始判自己", lambda r: r["pred_raw"] == 1),
        ("原始判别人", lambda r: r["pred_raw"] == 0),
    ]
    for name, f in strata:
        sub = [r for r in rows if f(r)]
        if not sub:
            continue
        dd = np.array([abs(r["p_own_pano"] - r["p_own_raw"]) for r in sub])
        fl = sum(1 for r in sub
                 if (r["p_own_raw"] >= 0.5) != (r["p_own_pano"] >= 0.5))
        print("%-14s %6d %10.3f %8.1f%%"
              % (name, len(sub), np.median(dd), 100.0 * fl / len(sub)))
    print("\n这测的是同一只物理手在两种投影下的预测位移，不是准确率。"
          "位移大，说明 cam3 上的每一个归属判定都由一个在没见过的像素上工作的模型做出；"
          "位移小，说明域间隙是整洁性问题，cam3 的人工 gold 结果照旧成立。")


if __name__ == "__main__":
    main()
