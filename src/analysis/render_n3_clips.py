"""N3 companion -- render a contact sheet (before-start/start/middle/end/
after-end, same 5-point layout as dump_contact_sheets.py) for every row in
the CSV produced by sample_n3_candidates.py, so the annotator has something
to actually look at (not just the GT text) while filling in
{verbs,object,state_before,state_after,atomicity}.

Usage (server):
    python -m src.analysis.render_n3_clips \
        --csv /tmp/n3_candidates.csv --out_dir /tmp/n3_clips
"""
import argparse, csv, os

from decord import VideoReader
from PIL import Image, ImageDraw


def make_sheet(vr, vfps, start, end, gt_name, tags, thumb_w=220):
    lo = int(start * vfps); hi = int(end * vfps); n = len(vr)
    pts = [max(0, lo - 5), lo, (lo + hi) // 2, hi, min(n - 1, hi + 5)]
    labels = ["before-start", "start", "middle", "end", "after-end"]
    imgs = []
    for i in pts:
        im = Image.fromarray(vr[i].asnumpy())
        w, h = im.size
        im = im.resize((thumb_w, int(h * thumb_w / w)))
        imgs.append(im)
    H = max(im.height for im in imgs)
    sheet = Image.new("RGB", (thumb_w * 5, H + 70), "white")
    d = ImageDraw.Draw(sheet)
    for i, (im, lab, pt) in enumerate(zip(imgs, labels, pts)):
        sheet.paste(im, (i * thumb_w, 50))
        d.text((i * thumb_w + 4, 4), f"{lab}\nf={pt}", fill="black")
    d.text((4, H + 55), f"GT (reference only): {gt_name}", fill="darkgreen")
    d.text((4, H + 65), f"tags: {tags}", fill="darkred")
    return sheet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    rows = list(csv.DictReader(open(a.csv)))
    vr_cache = {}
    for row in rows:
        vp = row["video"]
        if vp not in vr_cache:
            vr_cache[vp] = VideoReader(vp, num_threads=1)
        vr = vr_cache[vp]
        vfps = vr.get_avg_fps()
        sheet = make_sheet(vr, vfps, float(row["start"]), float(row["end"]),
                           row["gt_name"], row["category_tags"])
        out_path = os.path.join(a.out_dir, f"n3_{row['id']}_{row['recording_id']}"
                                            f"_seg{row['segment_idx']}.png")
        sheet.save(out_path)
        print(f"id={row['id']} {row['recording_id']} seg{row['segment_idx']} "
              f"[{row['start']}-{row['end']}s] tags={row['category_tags']} -> {out_path}")
    print(f"\nrendered {len(rows)} contact sheets -> {a.out_dir}")


if __name__ == "__main__":
    main()
