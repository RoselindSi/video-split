"""The OWNER table, with the denominator a person confirmed.

A frame the wearer's hand was never in is not a frame the pipeline lost it
on. 156 of the 175 frames where both systems delivered nothing are the hand
out of shot, so every rate computed before this table was wrong in the same
direction -- against us -- and by a different amount for each recording,
because those frames are not spread evenly.
"""
import collections
import csv
import glob
import os
import sys

ARMS = [("base", "/workspace/cam3_base"), ("Rolan", "/workspace/cam3_rolan")]
for extra in ("cam3_c1", "cam3_c2", "cam3_gate"):
    if glob.glob("/workspace/%s/*.csv" % extra):
        ARMS.append(({"cam3_c1": "C1 cont.10", "cam3_c2": "C2 new.40",
                      "cam3_gate": "gate.05"}[extra], "/workspace/" + extra))

N = {}
for line in open("/workspace/cam3_jobs.txt"):
    f = line.strip().split("|")
    if len(f) >= 4 and not line.startswith("#"):
        N[f[0]] = (int(f[2]), int(f[3]))

# out of shot, judged by eye, keyed by (recording, frame)
gone = set()
vis = set()
for r in csv.DictReader(open("/workspace/visiblepkg/hands.csv")):
    if r["label"] not in ("owner", "other", "nothand"):
        continue
    rec, rest = r["stem"].rsplit("_f", 1)
    f = int(rest.split("_h")[0])
    (gone if r["label"] == "other" else vis).add((rec, f))
print("人工判定：手不在画面 %d 帧，手可见 %d 帧（只有小臂 0）" % (len(gone), len(vis)))


def owner_frames(path):
    out = collections.defaultdict(list)
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if r["own"] == "1":
            out[int(r["frame"])].append(
                (float(r["covered"]),
                 (float(r["x1"]) - float(r["x0"])) * (float(r["y1"]) - float(r["y0"]))))
    return out


rows = []
for rec in sorted(N):
    start, n = N[rec]
    frames = [f for f in range(start, start + n) if (rec, f) not in gone]
    for name, d in ARMS:
        p = os.path.join(d, rec + ".csv")
        if not os.path.exists(p):
            continue
        own = owner_frames(p)
        have = [f for f in frames if f in own]
        runs, cur = [], 0
        for f in frames:
            if f in own:
                if cur:
                    runs.append(cur)
                    cur = 0
            else:
                cur += 1
        if cur:
            runs.append(cur)
        px = sum(c * ar for f in have for c, ar in own[f])
        tot = sum(ar for f in have for _, ar in own[f])
        rows.append((rec, name, len(frames), len(have), max(runs) if runs else 0,
                     100.0 * px / max(1.0, tot)))

print()
print("  %-16s %-11s %8s %9s %9s %10s" % (
    "录像", "配置", "可见帧", "覆盖率", "最长断档", "false-blur"))
agg = collections.defaultdict(lambda: [0, 0, 0, 0.0, 0.0])
for rec, name, vn, hv, wg, fb in rows:
    print("  %-16s %-11s %8d %8.1f%% %9d %9.3f%%" % (
        rec, name, vn, 100.0 * hv / max(1, vn), wg, fb))
    a = agg[name]
    a[0] += vn
    a[1] += hv
    a[2] = max(a[2], wg)
    a[3] += fb * hv
    a[4] += hv
print()
print("  %-11s %8s %9s %9s %10s" % ("配置", "可见帧", "覆盖率", "最长断档", "false-blur"))
for name, _ in ARMS:
    a = agg.get(name)
    if not a or not a[0]:
        continue
    print("  %-11s %8d %8.1f%% %9d %9.3f%%" % (
        name, a[0], 100.0 * a[1] / a[0], a[2], a[3] / max(1.0, a[4])))
