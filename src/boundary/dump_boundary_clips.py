"""B6 (visual pass) -- pull actual frames around a BALANCED, diverse sample of
missed boundaries / false peaks / true positives, so a human can confirm or
refute what the text-level audit (fp_fn_text_audit.py) suggested: false peaks
fire mid-segment (spurious motion) and misses are broadly distributed, not
concentrated in "same-name repeated action" alone.

Reuses the exact same categorization as fp_fn_text_audit.py (missed same-name
/ missed diff-name / false-peak) plus adds TRUE POSITIVES for contrast (what
does a correctly-caught boundary look like, for calibration), each capped per
category AND per video so ~50-75 clips span the dataset instead of one video.

For each selected (recording_id, time) makes a 9-frame contact sheet from
t-2s to t+2s (0.5s spacing) with the category + before/after GT segment names
printed, for fast human eyeballing (state change? hand in/out? camera shake?
occlusion? pause-then-continue? genuinely ambiguous?).

Usage (server, after train_head_multi.py --save_logits with segments saved):
    python -m src.boundary.dump_boundary_clips \
        --logits /tmp/b2_logits.pt --data /workspace/tr1/data_recseg/recseg_val.json \
        --out_dir /tmp/boundary_clips --thr 0.45 --min_gap 1.0 --tol 0.5 \
        --n_per_category 15
"""
import argparse, json, os, random
from collections import Counter

import numpy as np
import torch
from decord import VideoReader
from PIL import Image, ImageDraw


def peaks_threshold(prob, times, thr, min_gap):
    cand = [i for i in range(len(prob))
            if prob[i] >= thr
            and (i == 0 or prob[i] >= prob[i - 1])
            and (i == len(prob) - 1 or prob[i] >= prob[i + 1])]
    cand.sort(key=lambda i: -prob[i]); kept = []
    for i in cand:
        if all(abs(times[i] - times[j]) >= min_gap for j in kept):
            kept.append(i)
    return sorted(times[i] for i in kept)


def make_sheet(vr, vfps, center_t, category, before, after, thumb_w=160):
    n = len(vr)
    offsets = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    pts = [max(0, min(n - 1, int((center_t + o) * vfps))) for o in offsets]
    imgs = []
    for i in pts:
        im = Image.fromarray(vr[i].asnumpy())
        w, h = im.size
        im = im.resize((thumb_w, int(h * thumb_w / w)))
        imgs.append(im)
    H = max(im.height for im in imgs)
    sheet = Image.new("RGB", (thumb_w * 9, H + 70), "white")
    d = ImageDraw.Draw(sheet)
    for i, (im, o) in enumerate(zip(imgs, offsets)):
        sheet.paste(im, (i * thumb_w, 50))
        tag = "BOUNDARY" if o == 0 else f"{o:+.1f}s"
        d.text((i * thumb_w + 4, 4), tag, fill="red" if o == 0 else "black")
    d.text((4, H + 55), f"[{category}] before='{before}'", fill="darkgreen")
    d.text((4, H + 65), f"after='{after}'", fill="darkblue")
    return sheet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logits", required=True)
    ap.add_argument("--data", required=True, help="recseg json, for video paths")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--thr", type=float, default=0.45)
    ap.add_argument("--min_gap", type=float, default=1.0)
    ap.add_argument("--tol", type=float, default=0.5)
    ap.add_argument("--n_per_category", type=int, default=15)
    ap.add_argument("--per_video_cap", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    from src.eval.run_manifest import print_manifest_if_exists
    print_manifest_if_exists(a.logits)
    data = torch.load(a.logits, weights_only=False)
    video_path = {r["recording_id"]: r["video"] for r in json.load(open(a.data))}
    rng = random.Random(a.seed)

    # collect ALL candidates per category first, then sample -- avoids the
    # earlier bug where "first N encountered" was dominated by one video
    cands = {"missed_same": [], "missed_diff": [], "false_peak": [], "true_positive": []}
    for v in data:
        segs = sorted(v["segments"], key=lambda s: s[1])
        starts = {round(s[1], 2): s[0] for s in segs}
        ends = {round(s[2], 2): s[0] for s in segs}
        gts, preds = v["gt"], peaks_threshold(v["prob"], v["times"], a.thr, a.min_gap)
        used = set()
        for p in preds:
            best, bj = a.tol + 1, -1
            for j, g in enumerate(gts):
                if j not in used and abs(p - g) < best:
                    best, bj = abs(p - g), j
            if bj >= 0 and best <= a.tol:
                used.add(bj)
        for j, g in enumerate(gts):
            before = ends.get(round(g, 2), "<gap>")
            after = starts.get(round(g, 2), "<gap>")
            if j in used:
                cands["true_positive"].append((v["recording_id"], g, before, after))
            else:
                kind = "missed_same" if before == after else "missed_diff"
                cands[kind].append((v["recording_id"], g, before, after))
        for p in preds:
            d = min((abs(p - g) for g in gts), default=999)
            if d > a.tol:
                containing = next((s for s in segs if s[1] <= p <= s[2]), None)
                name = containing[0] if containing else "<gap>"
                cands["false_peak"].append((v["recording_id"], p, name, name))

    selected = []
    for kind, items in cands.items():
        rng.shuffle(items)
        cap = Counter(); picked = []
        for rid, t, before, after in items:
            if len(picked) >= a.n_per_category:
                break
            if cap[rid] >= a.per_video_cap:
                continue
            cap[rid] += 1
            picked.append((kind, rid, t, before, after))
        selected += picked
        print(f"{kind}: pool={len(items)} selected={len(picked)}")

    print(f"\ntotal clips to render: {len(selected)}")
    vr_cache = {}
    for kind, rid, t, before, after in selected:
        vp = video_path.get(rid)
        if vp is None:
            print(f"!! {rid} not in {a.data}, skipping"); continue
        if rid not in vr_cache:
            vr_cache[rid] = VideoReader(vp, num_threads=1)
        vr = vr_cache[rid]
        vfps = vr.get_avg_fps()
        sheet = make_sheet(vr, vfps, t, kind, before, after)
        out_path = os.path.join(a.out_dir, f"{kind}_{rid}_t{t:.1f}.png")
        sheet.save(out_path)
        print(f"{kind} {rid} @ {t:.1f}s -> {out_path}")


if __name__ == "__main__":
    main()
