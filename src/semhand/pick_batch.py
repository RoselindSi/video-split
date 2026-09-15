"""Pick zone-annotation batch 4: recordings no model or label set has touched.

WHAT THIS BATCH IS FOR. Every ownership number so far was read on fresh29 or
e2e_main2, and both have been scored many times; e2e_main2 also shares days
and devices with V1's training packages (28 of 29 recordings on a training
day). A student distilled from Qwen needs a test it has never been near, on
days and devices V1 never saw. Zone labels on these clips give that test.

EXCLUDED, BY DATABAG NAME:
    V1's training packages (otherpkg_1, trainpkg_T1) and testpkg_2
    fresh29, e2e_main2, and every earlier zone batch
    every bag already in /shared/ownership_labels/v1 (the Qwen window labels
    another team is producing) -- so that set stays usable for training
    without leaking into this test. It is still growing; this list is
    written out, and any later training pool must exclude it.

BLIND. No detector, classifier, VLM or teacher label chooses a recording or a
window. Strata come from metadata only, relative to V1's training packages:
    new_day_new_device    day and device both absent from V1's packages
    new_day_seen_device   a day V1 never saw, on a device it did
    seen_day_new_device   a training day, on a device V1 never saw
PER_STRATUM recordings each, one per device at most within a stratum, drawn
with a fixed seed. A drawn bag is kept only if its calibration is real (not
the zero template), cam34.mp4 opens, and it is at least MIN_S long; the
window is WINDOW frames at a start drawn from the bag's own name, so a rerun
picks the same frames.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import hashlib
import json
import os
import random
import subprocess

WINDOW = 400
MIN_S = 30.0
PER_STRATUM = 10
SEED = 20260915


def bag_of(path):
    return os.path.basename(path.rstrip("/"))


def device_of(path):
    return path.rstrip("/").split("/")[-2]


def day_of(bag):
    return bag.split("_")[1][:4]          # databag-26_0824_160752 -> 0824


def listed_bags(paths):
    out = set()
    for p in paths:
        if not os.path.exists(p):
            continue
        if p.endswith(".csv"):
            for r in csv.DictReader(open(p, encoding="utf-8-sig")):
                if r.get("databag"):
                    out.add(bag_of(r["databag"]))
        else:
            for line in open(p):
                line = line.strip()
                if line and not line.startswith("#"):
                    out.add(bag_of(line.split(":")[0]))
    return out


def frames_of(video):
    """-> (frames, fps) from the container header; no full read over shared storage."""
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=nb_frames,r_frame_rate:format=duration", "-of", "json", video],
                       capture_output=True, text=True, timeout=120)
    j = json.loads(r.stdout)
    s = j["streams"][0]
    num, den = (int(x) for x in s["r_frame_rate"].split("/"))
    fps = num / den
    n = int(s.get("nb_frames") or 0) or int(float(j["format"]["duration"]) * fps)
    return n, fps


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", default="/workspace/calibrated.txt")
    ap.add_argument("--out", default="/workspace/zonestereo4/batch4.txt")
    a = ap.parse_args()
    pool = {}
    for line in open(a.pool):
        line = line.strip()
        if line:
            pool.setdefault(bag_of(line), line.rstrip("/"))
    train = listed_bags(["/workspace/otherpkg_1/sources.csv", "/workspace/crops/trainpkg_T1/sources.csv"])
    train_paths = [pool[b] for b in train if b in pool]
    train_days = {day_of(b) for b in train}
    train_devs = {device_of(p) for p in train_paths}
    for pkg in ("/workspace/otherpkg_1/sources.csv", "/workspace/crops/trainpkg_T1/sources.csv"):
        train_devs |= {device_of(r["databag"]) for r in csv.DictReader(open(pkg, encoding="utf-8-sig"))}
    used = (train | listed_bags(["/workspace/testpkg_2/sources.csv", "/workspace/fresh29.txt",
                                 "/workspace/e2e_main2.txt"]))
    for d in ("/workspace/zonestereo", "/workspace/zonestereo2", "/workspace/zonestereo3"):
        used |= {os.path.basename(h)[5:-5].replace("R", "databag-26_", 1)
                 for h in glob.glob(os.path.join(d, "zone_*.html"))}
    qwen_bags = {os.path.basename(p)[:-6] for p in glob.glob("/shared/ownership_labels/v1/labels/*/*.jsonl")}
    excluded = used | qwen_bags
    print(f"候选池 {len(pool)}；排除：训练/测试/区域批次 {len(used)}，Qwen 窗口标签 {len(qwen_bags)}；"
          f"V1 训练日 {sorted(train_days)}，训练设备 {len(train_devs)} 台")

    strata = collections.defaultdict(list)
    for bag, path in sorted(pool.items()):
        if bag in excluded:
            continue
        nd, ndev = day_of(bag) not in train_days, device_of(path) not in train_devs
        s = ("new_day_new_device" if nd and ndev else "new_day_seen_device" if nd
             else "seen_day_new_device" if ndev else None)
        if s:
            strata[s].append(path)
    rng = random.Random(SEED)
    picked, log = [], []
    for s in ("new_day_new_device", "new_day_seen_device", "seen_day_new_device"):
        cands = strata[s][:]
        rng.shuffle(cands)
        devs = set()
        n_ok = 0
        print(f"  {s}: 候选 {len(cands)} 段，{len({device_of(p) for p in cands})} 台设备", flush=True)
        for path in cands:
            if n_ok >= PER_STRATUM:
                break
            dev = device_of(path)
            if dev in devs:
                continue
            cal, vid = os.path.join(path, "calibration.yaml"), os.path.join(path, "cam34.mp4")
            why = None
            if not os.path.exists(cal) or not os.path.exists(vid):
                why = "missing_file"
            elif hashlib.md5(open(cal, "rb").read()).hexdigest().startswith("53c4c6c1"):
                why = "zero_template_calibration"
            else:
                try:
                    n, fps = frames_of(vid)
                    if n / fps < MIN_S:
                        why = f"too_short_{n / fps:.0f}s"
                except Exception as e:
                    why = f"unreadable:{type(e).__name__}"
            log.append({"stratum": s, "databag": path, "rejected": why or ""})
            if why:
                continue
            start = random.Random(bag_of(path)).randrange(0, n - WINDOW)
            picked.append((s, path, start, start + WINDOW))
            devs.add(dev)
            n_ok += 1
        print(f"    选中 {n_ok}", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        f.write(f"# batch 4, blind, seed {SEED}, {WINDOW} frames per clip; see src/semhand/pick_batch.py\n")
        for s, path, st, en in picked:
            f.write(f"{path}:{st}:{en}\n")
    with open(a.out.replace(".txt", "_strata.txt"), "w") as f:
        for s, path, st, en in picked:
            f.write(f"{bag_of(path)}\t{st}\t{en}\t{s}\n")
    with open(a.out.replace(".txt", "_draws.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=("stratum", "databag", "rejected"))
        w.writeheader()
        w.writerows(log)
    with open(a.out.replace(".txt", "_excluded_bags.txt"), "w") as f:
        f.write("\n".join(sorted(excluded)) + "\n")
    print(f"-> {a.out}  {len(picked)} 段  {dict(collections.Counter(s for s, *_ in picked))}")


if __name__ == "__main__":
    main()
