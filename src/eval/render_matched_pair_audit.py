"""N7f follow-up -- human-audit template for matched-pair FAILURES
(compound_wins=False: the scorer did NOT rate the compound clip higher than
its object+primary_verb-matched atomic twin). This is where "gate/scorer
missed a real secondary action" cases live, now that N7e/N7f established the
global gate carries no independent signal -- worth a manual look at WHICH
failure pattern dominates before investing in N8-style fixes further.

For each failure pair, renders ONE contact sheet: top row = 5 frames of the
ATOMIC clip, bottom row = 5 frames of the COMPOUND clip (same 5-point
before-start/start/middle/end/after-end layout as render_n3_clips.py /
dump_contact_sheets.py), labeled with both GT names and both scores.

CSV columns to fill in while looking at the image (suggested categories from
the N7f/N8 discussion, free text / comma-separated, don't force exactly one):
  same_action_family (secondary is visually/semantically close to primary,
    no clear state transition -- e.g. "scrub" vs "rinse" on the same object),
  short_tail_diluted (secondary only occupies a small fraction of the clip,
    e.g. the "inspect"/"repack" tail of "unbox+inspect+repack"),
  ontology_mismatch (the GT compound label describes something a human
    annotator considers a distinct action but isn't obviously a SEPARATE
    action visually -- e.g. "wipe X with cleaning cloth" vs "wipe X"),
  annotation_ambiguous (hard to tell even for a human which GT name is more
    correct), other, notes.

Usage (server):
    python -m src.eval.render_matched_pair_audit \
        --scored /tmp/n7f_matched_scored.jsonl \
        --out_dir /tmp/matched_pair_audit --out_csv /tmp/matched_pair_audit/audit.csv
"""
import argparse, csv, json, os

from decord import VideoReader
from PIL import Image, ImageDraw


def five_frames(vr, vfps, start, end, thumb_w=200):
    lo = int(start * vfps); hi = int(end * vfps); n = len(vr)
    pts = [max(0, lo - 5), lo, (lo + hi) // 2, hi, min(n - 1, hi + 5)]
    imgs = []
    for i in pts:
        im = Image.fromarray(vr[i].asnumpy())
        w, h = im.size
        im = im.resize((thumb_w, int(h * thumb_w / w)))
        imgs.append(im)
    return imgs


def make_pair_sheet(vr_atomic, vfps_a, atomic, vr_compound, vfps_c, compound,
                     atomic_score, compound_score, thumb_w=200):
    top = five_frames(vr_atomic, vfps_a, atomic["start"], atomic["end"], thumb_w)
    bot = five_frames(vr_compound, vfps_c, compound["start"], compound["end"], thumb_w)
    H = max(im.height for im in top + bot)
    sheet = Image.new("RGB", (thumb_w * 5, 2 * H + 90), "white")
    d = ImageDraw.Draw(sheet)
    d.text((4, 4), f"ATOMIC (score={atomic_score:.2f}): {atomic['gt_name']}", fill="darkgreen")
    for i, im in enumerate(top):
        sheet.paste(im, (i * thumb_w, 30))
    d.text((4, H + 34), f"COMPOUND (score={compound_score:.2f}): {compound['gt_name']}", fill="darkred")
    for i, im in enumerate(bot):
        sheet.paste(im, (i * thumb_w, H + 60))
    return sheet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", required=True, help="output of eval_naming_n7f_matched_gate.py")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    a = ap.parse_args()

    from src.eval.run_manifest import print_manifest_if_exists
    print_manifest_if_exists(a.scored)
    recs = [json.loads(l) for l in open(a.scored)]
    failures = [r for r in recs if not r["compound_wins"]]
    print(f"total pairs: {len(recs)}  failures (compound_wins=False): {len(failures)}")

    os.makedirs(a.out_dir, exist_ok=True)
    vr_cache = {}
    rows = []
    for r in failures:
        for side in ("atomic", "compound"):
            vp = r[side]["video"]
            if vp not in vr_cache:
                vr_cache[vp] = VideoReader(vp, num_threads=1)
        vr_a, vr_c = vr_cache[r["atomic"]["video"]], vr_cache[r["compound"]["video"]]
        sheet = make_pair_sheet(vr_a, vr_a.get_avg_fps(), r["atomic"],
                                vr_c, vr_c.get_avg_fps(), r["compound"],
                                r["atomic_score"], r["compound_score"])
        out_path = os.path.join(a.out_dir, f"{r['pair_id'].replace('|', '_').replace(' ', '-')}.png")
        sheet.save(out_path)
        rows.append({"pair_id": r["pair_id"], "object": r["object"], "primary_verb": r["primary_verb"],
                    "atomic_gt_name": r["atomic"]["gt_name"], "compound_gt_name": r["compound"]["gt_name"],
                    "atomic_score": round(r["atomic_score"], 2), "compound_score": round(r["compound_score"], 2),
                    "image": out_path})
        print(f"{r['pair_id']} -> {out_path}")

    os.makedirs(os.path.dirname(a.out_csv) or ".", exist_ok=True)
    with open(a.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair_id", "object", "primary_verb", "atomic_gt_name", "compound_gt_name",
                    "atomic_score", "compound_score", "image",
                    # -- fill these in while looking at the image --
                    "same_action_family", "short_tail_diluted", "ontology_mismatch",
                    "annotation_ambiguous", "other", "notes"])
        for row in rows:
            w.writerow([row["pair_id"], row["object"], row["primary_verb"], row["atomic_gt_name"],
                       row["compound_gt_name"], row["atomic_score"], row["compound_score"],
                       row["image"], "", "", "", "", "", ""])

    print(f"\nwrote {len(rows)} audit rows -> {a.out_csv}")
    print("suggested categories (free text, comma-separate multiple, don't force one): "
          "same_action_family / short_tail_diluted / ontology_mismatch / "
          "annotation_ambiguous / other")


if __name__ == "__main__":
    main()
