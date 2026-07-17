"""P0 -- sanity check: are high-repetition-prediction recordings actually being
fed DIFFERENT local video content per segment, or is there a processor/cache
bug silently reusing the same frames?

For each target recording, pick a handful of GT segments and save a contact
sheet (5 frames: before-start, start, middle, end, after-end) + the GT name +
the model's prediction (looked up from an existing eval_naming_persegment.py
output jsonl) as one PNG, so a human can eyeball whether segments visually
differ and whether the prediction is at least locally plausible.

Usage (server):
    python dump_contact_sheets.py \
        --data /workspace/tr1/data_recseg/recseg_val.json \
        --persegment_jsonl /tmp/naming_persegment.jsonl \
        --recording_ids recording_000220 recording_000022 recording_000196 recording_000213 \
        --n_segments 8 --out_dir /tmp/contact_sheets
"""
import argparse, json, os
import numpy as np
from decord import VideoReader
from PIL import Image, ImageDraw

try:
    from src.seg_rewards import _as_segs
except ImportError:
    from src.rewards.seg_rewards import _as_segs


def load_predictions(jsonl_path):
    preds = {}
    for line in open(jsonl_path):
        r = json.loads(line)
        preds[(r["recording_id"], r["segment_idx"])] = r
    return preds


def make_sheet(vr, vfps, start, end, gt_name, pred_name, thumb_w=220):
    lo = int(start * vfps); hi = int(end * vfps); n = len(vr)
    pts = [max(0, lo - 5), lo, (lo + hi) // 2, hi, min(n - 1, hi + 5)]
    labels = ["before-start", "start", "middle", "end", "after-end"]
    imgs = []
    for i in pts:
        f = vr[i].asnumpy()
        im = Image.fromarray(f)
        w, h = im.size
        im = im.resize((thumb_w, int(h * thumb_w / w)))
        imgs.append(im)
    H = max(im.height for im in imgs)
    sheet = Image.new("RGB", (thumb_w * 5, H + 70), "white")
    d = ImageDraw.Draw(sheet)
    for i, (im, lab, pt) in enumerate(zip(imgs, labels, pts)):
        sheet.paste(im, (i * thumb_w, 50))
        d.text((i * thumb_w + 4, 4), f"{lab}\nf={pt}", fill="black")
    d.text((4, H + 55), f"GT: {gt_name}", fill="darkgreen")
    d.text((4, H + 65), f"PRED: {pred_name}", fill="darkred")
    return sheet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--persegment_jsonl", required=True)
    ap.add_argument("--recording_ids", nargs="+", required=True)
    ap.add_argument("--n_segments", type=int, default=8)
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    preds = load_predictions(a.persegment_jsonl)
    rows = {r["recording_id"]: r for r in json.load(open(a.data))}

    for rid in a.recording_ids:
        r = rows.get(rid)
        if r is None:
            print(f"!! {rid} not found in {a.data}"); continue
        gts = _as_segs(r["solution"])
        vr = VideoReader(r["video"], num_threads=1)
        vfps = vr.get_avg_fps()
        # use the segment indices that were actually predicted for this recording
        pred_idxs = sorted(i for (rr, i) in preds if rr == rid)[:a.n_segments]
        if not pred_idxs:
            print(f"!! no predictions found for {rid} in {a.persegment_jsonl}"); continue
        for si in pred_idxs:
            name, s, e = gts[si]
            rec = preds.get((rid, si), {})
            pred_name = rec.get("pred_name", "?")
            sheet = make_sheet(vr, vfps, s, e, name, pred_name)
            out_path = os.path.join(a.out_dir, f"{rid}_seg{si:03d}.png")
            sheet.save(out_path)
            print(f"{rid} seg{si} [{s:.1f}-{e:.1f}s] GT='{name}' PRED='{pred_name}' -> {out_path}")
        del vr


if __name__ == "__main__":
    main()
