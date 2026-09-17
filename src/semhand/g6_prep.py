"""GPT-6's finished labels as a second teacher, without calling GPT-6.

THE TWO TEACHERS NEVER SAW THE SAME HANDS. GPT-6's shared labels cover
monocular cam3/cam4 frames from the other team's has_other windows; our
distillation pool is 264 different recordings rendered as panoramas. So the
agreement filter runs the other way round: our teacher Q1 judges the hands
GPT-6 already labelled, and the hands both call the same way become extra
training rows. Q1 was checked on exactly this kind of image in the blind audit
(276 of 297 right, 98 of 98 on GPT-6-only boxes).

WHAT IS TAKEN. A frozen snapshot of `labels_completed.jsonl` (the run is still
going; the snapshot's sha256 is recorded). Images GPT-6 marked adequate;
hands labelled owner or other with a usable crop. Excluded by recording:
e2e_main2 and fresh29 (the ablation's dev sets), every zone batch 4-9 (the
tests), the distillation pool; excluded by image: the 300 audited boxes'
frames. At most PER_REC images per recording, drawn by seed, so a few
long recordings cannot dominate.

WRITTEN for `semhand.qwen` (Q1) and for `distil_train`:
    <out>/fresh/<rec>/index.csv    rec, frame (image index), tid (hand index),
                                   image, box, plus gpt6 label and camera
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
import re
import shutil

G = "/shared/ownership_labels/boxes_gpt6_qwen50958_v1"
PER_REC = 6
SEED = 20260917
COLS = ("rec", "frame", "tid", "databag", "image", "x0", "y0", "x1", "y1", "W", "H",
        "camera", "gpt6", "recording", "status")


def bags(paths):
    out = set()
    for p in paths:
        if os.path.exists(p):
            for line in open(p):
                line = line.strip()
                if line and not line.startswith("#"):
                    out.add(os.path.basename(line.split(":")[0].rstrip("/")))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="/workspace/g6teach")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    snap = os.path.join(a.out, "gpt6_labels_snapshot.jsonl")
    if not os.path.exists(snap):
        shutil.copyfile(os.path.join(G, "labels_completed.jsonl"), snap)
    sha = hashlib.sha256(open(snap, "rb").read()).hexdigest()

    # Only the clip lists themselves (batch<N>.txt): the folders also hold
    # batch<N>_excluded_bags.txt, which is a whole exclusion list, not a batch.
    lists = [p for p in glob.glob("/workspace/zonestereo*/batch*.txt")
             if re.fullmatch(r"batch\d+\.txt", os.path.basename(p))]
    excl = bags(["/workspace/e2e_main2.txt", "/workspace/fresh29.txt"] + lists)
    excl |= {os.path.basename(r["databag"]) for r in json.load(open("/workspace/distil/pool.json"))}
    audit_imgs = set()
    key = "/workspace/audit_gpt6_qwen/audit_key.json"
    if os.path.exists(key):
        audit_imgs = {(it["window_id"], it["camera"], it["frame_idx"]) for it in json.load(open(key))["items"]}

    by_rec = collections.defaultdict(list)
    dropped = collections.Counter()
    for line in open(snap):
        r = json.loads(line)
        L = r.get("labels") or {}
        if r["recording"] in excl:
            dropped["excluded_recording"] += 1
            continue
        if (r["window_id"], r["camera"], int(r["cache_frame_idx"])) in audit_imgs:
            dropped["audited_image"] += 1
            continue
        if L.get("image_readability") != "adequate":
            dropped["not_adequate"] += 1
            continue
        hands = [h for h in L.get("hands") or [] if h["label"] in ("owner", "other")
                 and h.get("crop_status") == "usable"]
        if not hands:
            dropped["no_usable_hand"] += 1
            continue
        by_rec[r["recording"]].append((r, hands))
    rng = random.Random(SEED)
    rows = []
    for rec in sorted(by_rec):
        imgs = by_rec[rec][:]
        rng.shuffle(imgs)
        tag = rec.replace("databag-26_", "G")
        for k, (r, hands) in enumerate(imgs[:PER_REC]):
            for t, h in enumerate(hands):
                x0, y0, x1, y1 = h["box_xyxy"]
                rows.append({"rec": tag, "frame": k, "tid": t, "databag": rec,
                             "image": os.path.join(G, r["image"]), "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                             "W": r["width"], "H": r["height"], "camera": r["camera"],
                             "gpt6": h["label"], "recording": rec, "status": "ok"})
    for rec_tag in {r["rec"] for r in rows}:
        d = os.path.join(a.out, "fresh", rec_tag)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "index.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=COLS)
            w.writeheader()
            w.writerows([r for r in rows if r["rec"] == rec_tag])
    lab = collections.Counter(r["gpt6"] for r in rows)
    json.dump({"snapshot_sha256": sha, "per_rec": PER_REC, "seed": SEED, "dropped_images": dropped,
               "recordings": len({r['rec'] for r in rows}), "hands": len(rows), "gpt6_labels": lab},
              open(os.path.join(a.out, "prep.json"), "w"), indent=1)
    print(f"快照 sha256 {sha[:12]}；丢弃 {dict(dropped)}")
    print(f"-> {a.out}: {len({r['rec'] for r in rows})} 段录像、"
          f"{len({(r['rec'], r['frame']) for r in rows})} 张图、{len(rows)} 只手  GPT-6 标签 {dict(lab)}")


if __name__ == "__main__":
    main()
