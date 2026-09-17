"""Pick zone batches 7-9: clips enriched for other people's hands, without any ownership model.

WHY ENRICH. Batches 4-6 were drawn blind and hold 49 foreign-hand tracks over
89 recordings, because most blind clips show one person working alone. That is
too few to separate a student from V1 on the privacy metric (the interval of
the difference still crossed zero).

HOW, WITHOUT BIASING THE TEST. No classifier, no VLM, no teacher label and no
window label from the other team chooses anything. The only signal is the hand
detector both V1 and every student share: a frame with three or more hand
detections must contain a hand that is not the wearer's (or a duplicate box).
Because every compared system reads the same detector boxes, enriching on the
detector's count favours none of them.

What it does not see, stated now: a colleague's hand in a frame where the
wearer's own hands are out of view, and a colleague whose hand the detector
misses. Those cases are under-represented here and remain covered only by the
blind batches 4-6.

    --mode scan   (sharded) for each candidate recording, render SCAN_FRAMES
                  evenly spaced frames and count detections per frame
    --mode pick   keep recordings with a frame of >= MIN_HANDS detections;
                  centre a WINDOW-frame clip on the frame with the most; deal
                  them into three batches of PER_BATCH, round-robin over
                  (device, day) so no batch is one workstation

EXCLUDED: V1's packages, testpkg_2, fresh29, e2e_main2, every zone batch so
far, the distillation pool, and every bag in the other team's Qwen window
labels (so that set stays usable for training).
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import random

from src.semhand.pick_batch import bag_of, day_of, device_of, listed_bags

SCAN_RECORDINGS = 480
SCAN_FRAMES = 12
MIN_HANDS = 3
WINDOW = 400
PER_BATCH = 30
SEED = 20260917


def candidates(pool_list):
    used = listed_bags(["/workspace/otherpkg_1/sources.csv", "/workspace/crops/trainpkg_T1/sources.csv",
                        "/workspace/testpkg_2/sources.csv", "/workspace/fresh29.txt",
                        "/workspace/e2e_main2.txt"])
    used |= {os.path.basename(h)[5:-5].replace("R", "databag-26_", 1)
             for h in glob.glob("/workspace/zonestereo*/zone_*.html")}
    used |= listed_bags(sorted(glob.glob("/workspace/zonestereo*/batch*.txt")))
    if os.path.exists("/workspace/distil/pool.json"):
        used |= {bag_of(r["databag"]) for r in json.load(open("/workspace/distil/pool.json"))}
    used |= {os.path.basename(p)[:-6] for p in glob.glob("/shared/ownership_labels/v1/labels/*/*.jsonl")}
    pool = {}
    for line in open(pool_list):
        line = line.strip()
        if line and bag_of(line) not in used:
            pool.setdefault(bag_of(line), line.rstrip("/"))
    by = collections.defaultdict(list)
    for bag, path in sorted(pool.items()):
        by[(device_of(path), day_of(bag))].append(path)
    keys = sorted(by)
    random.Random(SEED).shuffle(keys)
    order, r = [], 0
    while any(len(by[k]) > r for k in keys):
        order += [by[k][r] for k in keys if len(by[k]) > r]
        r += 1
    return order, len(used)


def scan(a):
    import cv2
    from ultralytics import YOLO
    from src.rig.calibration import RigCalibration
    from src.rig.geometry import VirtualWideCamera
    from src.rig.hand_detect import detect
    from src.rig.render_wide import render
    from src.rig.seam_fix import ClipReader
    from src.semhand.frames import cap_decoder_threads
    cap_decoder_threads()
    cv2.setNumThreads(2)
    order, _ = candidates(a.pool_list)
    mine = order[:SCAN_RECORDINGS][a.shard::a.nshards]
    yolo = YOLO(a.weights)
    out = os.path.join(a.out, f"scan_{a.shard}.jsonl")
    done = set()
    if os.path.exists(out):
        done = {json.loads(l)["databag"] for l in open(out) if l.strip()}
    with open(out, "a") as fh:
        for path in mine:
            if path in done:
                continue
            rec = {"databag": path, "counts": {}, "status": "ok"}
            try:
                cap = cv2.VideoCapture(os.path.join(path, "cam34.mp4"))
                n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                cap.release()
                if n < WINDOW + 30:
                    rec["status"] = "too_short"
                else:
                    rig = RigCalibration(os.path.join(path, "calibration.yaml"))
                    vcam = VirtualWideCamera.from_rig(rig)
                    vids = {k: os.path.join(path, f"{k}.mp4") for k in ("cam12", "cam34", "cam56")}
                    cache = {}
                    rec["n_frames"] = n
                    for i in range(SCAN_FRAMES):
                        f = int((i + 0.5) * (n - 1) / SCAN_FRAMES)
                        rd = ClipReader(rig, vids, f)
                        src = rd.next()
                        rd.close()
                        if not src:
                            continue
                        rgb = render(rig, vcam, src, 0.6, map_cache=cache)[0]
                        rec["counts"][f] = len(detect(yolo, rgb, min_conf=0.25))
            except (Exception, SystemExit) as e:
                rec["status"] = f"failed:{type(e).__name__}"
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(f"  {bag_of(path)} {rec['status']} max {max(rec['counts'].values(), default=0)}", flush=True)


def pick(a):
    order, n_used = candidates(a.pool_list)
    scanned = {}
    for p in glob.glob(os.path.join(a.out, "scan_*.jsonl")):
        for line in open(p):
            if line.strip():
                r = json.loads(line)
                scanned[r["databag"]] = r
    rich = []
    for path in order[:SCAN_RECORDINGS]:
        r = scanned.get(path)
        if not r or r["status"] != "ok" or not r["counts"]:
            continue
        f, k = max(((int(f), c) for f, c in r["counts"].items()), key=lambda t: (t[1], -t[0]))
        if k >= MIN_HANDS:
            start = min(max(0, f - WINDOW // 2), r["n_frames"] - WINDOW - 1)
            rich.append((path, start, k, sum(c >= MIN_HANDS for c in r["counts"].values())))
    print(f"扫描 {len(scanned)} 段（排除 {n_used}）；有 >= {MIN_HANDS} 只手的帧的段 {len(rich)}")
    batches = {7: [], 8: [], 9: []}
    for i, item in enumerate(rich[:3 * PER_BATCH]):
        batches[7 + i % 3].append(item)
    for n, items in batches.items():
        d = f"/workspace/zonestereo{n}"
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, f"batch{n}.txt"), "w") as fh:
            fh.write(f"# detector-enriched (>= {MIN_HANDS} hands in a scanned frame), seed {SEED}; "
                     f"see src/semhand/pick_rich.py\n")
            for path, s, k, m in items:
                fh.write(f"{path}:{s}:{s + WINDOW}\n")
        with open(os.path.join(d, f"batch{n}_strata.txt"), "w") as fh:
            for path, s, k, m in items:
                fh.write(f"{bag_of(path)}\t{s}\t{s + WINDOW}\trich_max{k}\n")
        print(f"  batch {n}: {len(items)} 段，设备 {len({device_of(p) for p, *_ in items})} 台，"
              f"最多手数分布 {dict(collections.Counter(k for _, _, k, _ in items))}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("scan", "pick"), required=True)
    ap.add_argument("--out", default="/workspace/richscan")
    ap.add_argument("--pool_list", default="/workspace/calibrated.txt")
    ap.add_argument("--weights", default="/shared/models/HaWoR/weights/external/detector.pt")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    (scan if a.mode == "scan" else pick)(a)


if __name__ == "__main__":
    main()
