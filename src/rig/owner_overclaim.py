"""How many frames really claim more hands than the wearer has.

THE FIRST VERSION OF THIS WAS NOT A FLOOR AND THE CLAIM IS WITHDRAWN. It
counted frames with three own-hand boxes at IoU below 0.10 and called that
proof, on the argument that a person has two hands. Low IoU is not evidence
of two hands: a box on the fingers and a box on the wrist of the SAME hand
overlap hardly at all, and the detector splits hands exactly that way -- it
is why `new_hand_grace` exists. So three disjoint boxes are three candidates,
not three hands, and 9.3%/12.7% were candidate rates.

FOUR STAGES, AND THE MODEL IS ONLY ALLOWED THE SECOND ONE.

    >=3 own boxes            arithmetic on the delivered run
    - confirmed non-hands    hand-ness, which is the one thing it can answer
    - duplicates merged      geometry, swept, never a single fitted constant
    = still >=3 real hands?  a person, on every frame that survives

Hand-ness cannot say whether two boxes are two hands and certainly cannot say
whose they are, so it is used only to throw away boxes it is confident are
not hands at all -- the half of its behaviour that was measured cleanly, 34 of
35 non-hands caught, with its errors running the safe way as refusals of real
hands. Its AUC of 0.950 comes from 88 boxes in a quieter population than this
one, so it is a workload filter here and not a source of truth: nothing it
accepts is counted without a person looking.

THE MERGE IS SWEPT BECAUSE IT IS A GUESS. Single-linkage on centre distance
in units of box size, from 0 (merge nothing) upward. Reporting one value
would hide that the answer moves with it; the sweep makes the sensitivity
part of the result, and the final count is taken at a setting that merges
generously, so what survives is a floor rather than an estimate.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    i = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def disjoint(items, thr):
    items = sorted(items, key=lambda it: -(it["box"][2] - it["box"][0])
                   * (it["box"][3] - it["box"][1]))
    keep = []
    for it in items:
        if all(iou(it["box"], k["box"]) < thr for k in keep):
            keep.append(it)
    return keep


def merge(items, t):
    """Single-linkage on centre distance in units of the smaller box. -> groups

    `t` is a multiple of the smaller box's longer side, so the rule scales
    with how close the hand is to the camera -- a hand filling the frame and
    one across the room are split by the same t."""
    n = len(items)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def cen(b):
        return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)

    for i in range(n):
        for j in range(i + 1, n):
            a, b = items[i]["box"], items[j]["box"]
            ca, cb = cen(a), cen(b)
            d = ((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2) ** 0.5
            sa = max(a[2] - a[0], a[3] - a[1])
            sb = max(b[2] - b[0], b[3] - b[1])
            if d <= t * min(sa, sb):
                parent[find(i)] = find(j)
    g = collections.defaultdict(list)
    for i in range(n):
        g[find(i)].append(items[i])
    return list(g.values())


def load_run(arm, jobs):
    out = collections.defaultdict(lambda: collections.defaultdict(list))
    for rec in jobs:
        p = os.path.join(arm, rec + ".csv")
        if not os.path.exists(p):
            continue
        for r in csv.DictReader(open(p, encoding="utf-8")):
            if r["own"] != "1":
                continue
            out[rec][int(r["frame"])].append(
                {"box": [float(r[c]) for c in ("x0", "y0", "x1", "y1")],
                 "conf": float(r.get("conf") or 0), "tid": r.get("tid") or ""})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", default="/workspace/cam3_c1")
    ap.add_argument("--jobs", default="/workspace/cam3_jobs.txt")
    ap.add_argument("--scores", default="/workspace/three_c1.jsonl")
    ap.add_argument("--iou", type=float, default=0.10)
    ap.add_argument("--p_nonhand", type=float, default=0.10,
                    help="drop a box only when the model is this confident it "
                         "is NOT a hand; the accept side is left to a person")
    ap.add_argument("--merge_t", type=float, default=1.0,
                    help="the setting the audit package is built at")
    ap.add_argument("--out_pkg", default="")
    ap.add_argument("--cell_px", type=int, default=1100)
    a = ap.parse_args()

    jobs = {}
    for line in open(a.jobs):
        f = line.strip().split("|")
        if len(f) >= 4 and not line.startswith("#"):
            jobs[f[0]] = (f[1], int(f[2]), int(f[3]))

    run = load_run(a.arm, jobs)
    P = {}
    if os.path.exists(a.scores):
        for line in open(a.scores):
            d = json.loads(line)
            P[d["stem"]] = float(d.get("p_true", d.get("p", 0.0)))
    print("hand-ness 打分 %d 个" % len(P))

    # stage 1 -- candidates
    cand = []
    for rec in sorted(jobs):
        s, n = jobs[rec][1], jobs[rec][2]
        for f in range(s, s + n):
            its = run[rec].get(f)
            if not its:
                continue
            keep = disjoint(its, a.iou)
            if len(keep) >= 3:
                for k, it in enumerate(keep):
                    it["stem"] = "%s_f%06d_b%d" % (rec, f, k)
                cand.append({"rec": rec, "frame": f, "items": keep})
    print("\n阶段 1  >=3 个互不重叠的 own 框：%d 帧（%d 个框）"
          % (len(cand), sum(len(c["items"]) for c in cand)))

    # stage 2 -- drop what the model is confident is not a hand
    miss = 0
    for c in cand:
        kept = []
        for it in c["items"]:
            p = P.get(it["stem"])
            if p is None:
                miss += 1
                kept.append(it)          # unscored is not evidence of anything
            elif p > a.p_nonhand:
                it["p_hand"] = p
                kept.append(it)
        c["hands"] = kept
    if miss:
        print("  （%d 个框没有分数，按保留处理）" % miss)
    n2 = [c for c in cand if len(c["hands"]) >= 3]
    print("阶段 2  去掉 p_hand<=%.2f 的框后仍 >=3 个：%d 帧"
          % (a.p_nonhand, len(n2)))
    dropped = sum(len(c["items"]) - len(c["hands"]) for c in cand)
    print("        被判定为非手的框 %d / %d（%.0f%%）"
          % (dropped, sum(len(c["items"]) for c in cand),
             100.0 * dropped / max(1, sum(len(c["items"]) for c in cand))))

    # stage 3 -- merge duplicates of one physical hand, swept
    print("\n阶段 3+4  合并同一只手的重复框后仍 >=3 组的帧：")
    print("   %-10s %10s %10s %14s" % ("merge t", "帧数", "占候选", "最长连续"))
    for t in (0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
        surv = []
        for c in n2:
            if len(merge(c["hands"], t)) >= 3:
                surv.append((c["rec"], c["frame"]))
        best = 0
        by_rec = collections.defaultdict(list)
        for rec, f in surv:
            by_rec[rec].append(f)
        for rec, fs in by_rec.items():
            fs.sort()
            run_len = 1
            for x, y in zip(fs, fs[1:]):
                run_len = run_len + 1 if y == x + 1 else 1
                best = max(best, run_len)
            best = max(best, 1 if fs else 0)
        print("   %-10.2f %10d %9.0f%% %13d" % (
            t, len(surv), 100.0 * len(surv) / max(1, len(cand)), best))

    if not a.out_pkg:
        return

    surv = [c for c in n2 if len(merge(c["hands"], a.merge_t)) >= 3]
    print("\n在 merge t=%.2f 下有 %d 帧要人工看" % (a.merge_t, len(surv)))
    if not surv:
        return

    import cv2
    from src.rig.seam_fix import RawCameraReader
    for sub in ("context", "crops"):
        os.makedirs(os.path.join(a.out_pkg, sub), exist_ok=True)
    by_rec = collections.defaultdict(list)
    for c in surv:
        by_rec[c["rec"]].append(c)
    rows = []
    for rec in sorted(by_rec):
        bag = jobs[rec][0]
        vids = {k: os.path.join(bag, f"{k}.mp4") for k in ("cam12", "cam34", "cam56")}
        want = sorted({c["frame"] for c in by_rec[rec]})
        rd = RawCameraReader(vids, "cam3", want[0])
        cur, cache = want[0] - 1, {}
        for f in want:
            img = None
            while cur < f:
                img = rd.next()
                cur += 1
                if img is None:
                    break
            if img is None:
                break
            cache[f] = img.copy()
        rd.close()
        for j, c in enumerate(sorted(by_rec[rec], key=lambda x: x["frame"])):
            img = cache.get(c["frame"])
            if img is None:
                continue
            H, W = img.shape[:2]
            vis = img.copy()
            groups = merge(c["hands"], a.merge_t)
            # ONE COLOUR PER GROUP, and the group letter, not the count. The
            # reader is being asked how many different hands are boxed; a
            # number written on the picture is the answer being tested.
            cols = [(0, 230, 0), (0, 200, 255), (255, 160, 0), (230, 0, 230),
                    (255, 255, 0), (0, 255, 160)]
            for gi, grp in enumerate(groups):
                col = cols[gi % len(cols)]
                for it in grp:
                    b = [int(v) for v in it["box"]]
                    cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), col, 5)
                    cv2.putText(vis, chr(65 + gi), (b[0] + 6, b[1] - 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, col, 4, cv2.LINE_AA)
            sc = a.cell_px / float(W)
            ctx = cv2.resize(vis, (a.cell_px, int(H * sc)))
            stem = "%s_f%06d_h%d" % (rec, c["frame"], 900 + j)
            cv2.imwrite(os.path.join(a.out_pkg, "context", stem + ".jpg"), ctx,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 84])
            b0 = [int(v) for v in groups[0][0]["box"]]
            pad = max(160, b0[2] - b0[0])
            y0, y1 = max(0, b0[1] - pad), min(H, b0[3] + pad)
            x0, x1 = max(0, b0[0] - pad), min(W, b0[2] + pad)
            crop = vis[y0:y1, x0:x1]
            if crop.size:
                cv2.imwrite(os.path.join(a.out_pkg, "crops", stem + ".jpg"),
                            cv2.resize(crop, (192, 192)),
                            [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            rows.append({"stem": stem, "frame": c["frame"], "hand": 900 + j,
                         "conf": 0.0, "w_frac": 0.0, "h_frac": 0.0,
                         "cx_frac": 0.0, "cy_frac": 0.0,
                         "model": "groups=%d" % len(groups),
                         "n_box": len(c["hands"]), "n_group": len(groups),
                         "p_min": round(min(it.get("p_hand", 1.0)
                                            for it in c["hands"]), 3),
                         "label": "", "label_mode": ""})
        print("  %-16s %d 帧" % (rec, len(by_rec[rec])), flush=True)
    with open(os.path.join(a.out_pkg, "hands.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(a.out_pkg, "MAPPING.txt"), "w") as fh:
        fh.write("问题：这张图里被框住的，是几只不同的手？\n"
                 "1 只手   -> nothand\n2 只手   -> owner\n"
                 "3 只或更多 -> other\n看不清   -> unsure\n"
                 "（回填的 CSV 只有这些值，没有中文；对照这张表读）\n")
    print("-> %s (%d 帧)" % (a.out_pkg, len(rows)))


if __name__ == "__main__":
    main()
