"""Score hand-ness with the distilled student, in the format the pipeline eats.

DROP-IN FOR `handness_arm`. Same output file -- one `{stem, p, frame, box}` per
line, keyed by box geometry so `blur_check --handness_boxes` can filter before
the tracker spends an id -- but the score comes from two resnet18 pairs on one
GPU instead of a vision-language model over HTTP. That stage was 47.9% of
end-to-end, so this is the single largest time item in the pipeline.

THE OPERATING POINT IS 0.10, AND IT WAS CHOSEN ON OWNER TRACKS ALONE. On the
20 frozen recordings, at 0.10 the student drops 50 of 100 human-confirmed
non-hand tracks and 0 of 137 OWNER tracks. Pooling OWNER with other people's
hands made this look like it cost 10.8% of real hands and nearly got the
student rejected; the loss is almost entirely on other people's hands (6.2%
here), which the OWNER-exemption objective does not score. Zero out of 137
puts the 95% upper bound near 2.2%.

WHAT THIS GIVES UP AGAINST THE TEACHER, stated so it is not discovered later:
at matched 85% recall the teacher's non-hand false positive rate is 1.0% and
the student's is 10.0% (AUC 0.988 vs 0.945). The student is kept because at
0.10 the errors it makes are not OWNER errors, not because it matches the
teacher. If the objective ever widens back to other people's hands, this
choice has to be re-measured, not inherited.

Small boxes are still the weak band -- 60% false positive below 60px on the
held-out split, 47% at 100-150px -- so a threshold raised above 0.30 to catch
more non-hands will start costing OWNER tracks in exactly that band.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import time

DEFAULT_MODELS = "/storage/hbin_model/handness_bin_seed*.pt"


def read_jobs(path):
    jobs = {}
    for line in open(path, encoding="utf-8"):
        parts = line.rstrip("\n").split("|")
        if len(parts) >= 4:
            jobs[parts[0]] = (parts[1], int(parts[2]), int(parts[3]))
    return jobs


def boxes_of(path):
    """每个拿到 tid 的框，按框本身索引 —— 过滤发生在归属之前。"""
    out = []
    for row in csv.DictReader(open(path, encoding="utf-8")):
        if row.get("tid") in (None, ""):
            continue
        try:
            box = [int(float(row[c])) for c in ("x0", "y0", "x1", "y1")]
        except (KeyError, TypeError, ValueError):
            continue
        out.append((int(row["frame"]), box))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default=DEFAULT_MODELS)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--rec", action="append")
    a = ap.parse_args()

    import glob
    import numpy as np
    import torch
    from src.rig import own_ctx
    from src.semhand.student import views
    from src.semhand.crossv1 import norm_rgb
    from src.rig.seam_fix import RawCameraReader

    paths = sorted(glob.glob(a.models))
    if not paths:
        raise SystemExit("没有模型：%s" % a.models)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = []
    for path in paths:
        ck = torch.load(path, map_location=device, weights_only=False)
        model = own_ctx.build(ck.get("arm", "both"),
                              n_out=int(ck.get("n_out", 2))).to(device)
        model.load_state_dict(ck["state"])
        model.eval()
        models.append(model)
    print("集成 %d 个：%s" % (len(models), [os.path.basename(p) for p in paths]),
          flush=True)

    jobs = read_jobs(a.jobs)
    want = set(a.rec) if a.rec else None
    start = time.time()
    written = low = 0
    with open(a.out, "w", encoding="utf-8") as fh:
        for rec, (bag, _s, _n) in sorted(jobs.items()):
            if want and rec not in want:
                continue
            csv_path = os.path.join(a.arm, "%s.csv" % rec)
            if not os.path.exists(csv_path):
                continue
            picks = boxes_of(csv_path)
            if not picks:
                continue
            per_frame = collections.defaultdict(list)
            for frame, box in picks:
                per_frame[frame].append(box)
            order = sorted(per_frame)
            videos = {k: os.path.join(bag, "%s.mp4" % k)
                      for k in ("cam12", "cam34", "cam56")}
            reader = RawCameraReader(videos, "cam3", order[0])
            current, image, made = order[0] - 1, None, 0
            for frame in order:
                while current < frame:
                    image = reader.next()
                    current += 1
                    if image is None:
                        break
                if image is None:
                    break
                boxes = per_frame[frame]
                hs, cs = [], []
                for box in boxes:
                    h, c = views(image, box)
                    hs.append(norm_rgb(h))
                    cs.append(norm_rgb(c))
                scores = []
                with torch.no_grad():
                    for i in range(0, len(hs), a.batch):
                        h = torch.stack(hs[i:i + a.batch]).to(device)
                        c = torch.stack(cs[i:i + a.batch]).to(device)
                        g = torch.zeros(len(h), 14, device=device)
                        p = np.mean([torch.softmax(m(h, c, g), 1)[:, 1]
                                     .cpu().numpy() for m in models], 0)
                        scores.extend(float(x) for x in p)
                for box, p in zip(boxes, scores):
                    stem = "%s_f%06d_x%d_y%d" % (rec, frame, box[0], box[1])
                    fh.write(json.dumps({
                        "stem": stem, "p": p, "frame": frame,
                        "box": "%d,%d,%d,%d" % tuple(box)}) + "\n")
                    written += 1
                    low += int(p < 0.10)
                    made += 1
            reader.close()
            print("  %-22s %5d 框" % (rec, made), flush=True)
    elapsed = time.time() - start
    print("%.1fs（%.0f 框/分钟），写出 %d 条 -> %s"
          % (elapsed, 60 * written / max(1e-9, elapsed), written, a.out))
    print("  P(hand) < 0.10 的 %d 个（%.1f%%）—— 跟踪前会被摘掉的"
          % (low, 100 * low / max(1, written)))


if __name__ == "__main__":
    main()
