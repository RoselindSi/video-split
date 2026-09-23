"""Run the student on the held-out pool, at every width including the one it was never taught.

WHY THIS IS SEPARATE FROM THE TEST ROOTS. Those carry ownership gold and
answer M1, M2 and G. This pool carries no labels at all: it is 6,703 boxes
from 57 recordings nothing has touched, and its job is to show where the
student says `not a hand` -- by width, so the band under 150 px is visible
rather than averaged in.

THAT BAND IS THE POINT. Every class-2 training example is 150 px or wider,
because the probe that labelled them collapses below that. A person judged
225 boxes of this pool and put non-hands at 12.2% under 150 px against 5.9%
above -- so half the non-hands in the stream are in the band the student
learned nothing about. What it does there is not predictable from the dev
split, which is entirely above the floor.

NO GOLD IS ASSUMED HERE. This writes what the model said; the precision that
the criteria call `N` needs a person, on a sample drawn FROM these calls,
because precision is a property of the flagged set. What this file produces
is the population that sample comes from.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import os

BANDS = [(0, 100), (100, 150), (150, 200), (200, 340), (340, 10 ** 9)]


def band_of(w):
    for lo, hi in BANDS:
        if lo <= w < hi:
            return "%d-%d" % (lo, hi) if hi < 10 ** 9 else "%d+" % lo
    return "?"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--views", default="/workspace/heldout_views")
    ap.add_argument("--frames", required=True,
                    help="clean native frames from `semhand.clean_frames`")
    ap.add_argument("--ckpt", required=True, help="glob over the seeds")
    ap.add_argument("--thr", type=float, default=0.5,
                    help="class 2 fires when it is the argmax; this only "
                         "reports an additional stricter count")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import numpy as np
    import torch
    from PIL import Image
    from src.rig import own_ctx
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
    print("%d 个 seed，n_out=%d" % (len(models), int(ck.get("n_out", 2))))

    rows = list(csv.DictReader(open(os.path.join(a.views, "index.csv"))))
    print("留出池 %d 个框 / %d 条录像" % (len(rows), len({r["rec"] for r in rows})))

    out = []
    B = 64
    with torch.no_grad():
        for i in range(0, len(rows), B):
            chunk = rows[i:i + B]
            hs, cs = [], []
            keep = []
            for r in chunk:
                # THE CLEAN NATIVE FRAME, through the same `qwen.views` the
                # training used. The first version passed the harvest's
                # `_full.jpg` -- 1280 wide with the box already drawn -- while
                # the index carries native 1920x1520 coordinates, so `views`
                # computed a scale of 1.0, drew the box off the edge and cut
                # the crop from the wrong place. Every box came back looking
                # like nothing, and the run reported 100% not-a-hand in every
                # band from a model whose dev precision was 97%. The comment
                # here said "from the clean image" while the line below did
                # not; now it does.
                img = os.path.join(a.frames, r["rec"], r["cam"],
                                   "%06d.jpg" % int(r["frame"]))
                if not os.path.exists(img):
                    continue
                frame, crop = views({"image": img,
                                     "box": [int(r[c]) for c in
                                             ("x0", "y0", "x1", "y1")]})
                hs.append(norm_rgb(np.asarray(
                    crop.resize((CROP_PX, CROP_PX), Image.BICUBIC))))
                cs.append(norm_rgb(np.asarray(frame.resize(FRAME_PX,
                                                           Image.BICUBIC))))
                keep.append(r)
            if not keep:
                continue
            h = torch.stack(hs).to(device)
            c = torch.stack(cs).to(device)
            g = torch.zeros(len(keep), 14).to(device)
            pr = np.mean([torch.softmax(m(h, c, g), 1).cpu().numpy()
                          for m in models], 0)
            for r, p in zip(keep, pr):
                out.append({"stem": r["stem"], "rec": r["rec"], "cam": r["cam"],
                            "frame": r["frame"], "w_px": r["w_px"],
                            "band": band_of(int(r["w_px"])),
                            "p_other": round(float(p[0]), 4),
                            "p_owner": round(float(p[1]), 4),
                            "p_nothand": round(float(p[2]), 4),
                            "pred": int(np.argmax(p))})
            if i % 640 == 0:
                print("  %d/%d" % (i, len(rows)), flush=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)

    print("\n%-10s %7s %9s %9s %10s" % ("宽(px)", "框数", "判非手", "占比",
                                        "人工普查的非手率"))
    prev = {"0-100": 11.1, "100-150": 13.3, "150-200": 6.7,
            "200-340": 4.4, "340+": 6.7}
    for lo, hi in BANDS:
        b = "%d-%d" % (lo, hi) if hi < 10 ** 9 else "%d+" % lo
        sub = [r for r in out if r["band"] == b]
        if not sub:
            continue
        k = sum(1 for r in sub if r["pred"] == 2)
        print("%-10s %7d %9d %8.1f%% %13.1f%%"
              % (b, len(sub), k, 100.0 * k / len(sub), prev.get(b, float("nan"))))
    k = sum(1 for r in out if r["pred"] == 2)
    print("%-10s %7d %9d %8.1f%% %13.1f%%"
          % ("合计", len(out), k, 100.0 * k / len(out), 7.7))
    print("\n-> %s" % a.out)
    print("右边那一列是人工在同一个池子上普查出来的非手率。"
          "学生判非手的比例明显低于它，就是漏；明显高于它，就是误判。"
          "两者都要靠从学生的判定里抽样人工确认，这里只给出总量。")


if __name__ == "__main__":
    main()
