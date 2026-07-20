"""N9 final step (4) -- light qualitative audit of compound items where the
contrastive scorer's grouped-CV decode still got the secondary set wrong.
This is the human-judgment step the numeric metrics can't replace: is the
remaining ~20% compound-only exact accuracy a visual-grounding limit, an
ontology/annotation granularity mismatch (e.g. "wipe with cleaning cloth"
vs "wipe"), or a genuinely short/subtle secondary action?

Selects up to --n failing compound items (grouped-CV predicted secondary set
!= true secondary set, using the SAME fixed threshold eval_naming_n9c_bootstrap.py
computed), renders a 9-frame contact sheet per item (frames come from the
already-saved frame_indices, no re-sampling), and writes a CSV with the
scores/options as context plus empty columns for the human.

Usage (server, no GPU needed):
    python -m src.eval.render_compound_failure_audit \
        --n7_jsonl /workspace/tr1/results/naming/n7_scored.jsonl \
        --n9_jsonl /workspace/tr1/results/naming/n9_full_contrastive.jsonl \
        --out_dir /workspace/tr1/results/naming/compound_failure_audit \
        --n 20
"""
import argparse, csv, json, os

from decord import VideoReader
from PIL import Image, ImageDraw


def best_f1_threshold(pairs):
    if not pairs:
        return 0.0
    best_tau, best_f1 = 0.0, -1
    for tau in sorted({s for s, _ in pairs}):
        tp = sum(1 for s, l in pairs if s > tau and l == 1)
        fp = sum(1 for s, l in pairs if s > tau and l == 0)
        fn = sum(1 for s, l in pairs if s <= tau and l == 1)
        p = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1)
        f1 = 2 * p * rc / max(p + rc, 1e-9)
        if f1 > best_f1:
            best_f1, best_tau = f1, tau
    return best_tau


def make_sheet(vr, vfps, start, end, frame_indices, gt_name, options, gt_letters,
              primary_letter, pred_letters, scores, thumb_w=160):
    n = len(vr)
    imgs = []
    for i in frame_indices:
        im = Image.fromarray(vr[min(i, n - 1)].asnumpy())
        w, h = im.size
        im = im.resize((thumb_w, int(h * thumb_w / w)))
        imgs.append(im)
    H = max(im.height for im in imgs)
    ncols = len(imgs)
    sheet = Image.new("RGB", (thumb_w * ncols, H + 90), "white")
    d = ImageDraw.Draw(sheet)
    for i, im in enumerate(imgs):
        sheet.paste(im, (i * thumb_w, 30))
    d.text((4, 4), f"GT: {gt_name}", fill="darkgreen")
    opts_str = ", ".join(f"{l}={v}(score={scores.get(l, float('nan')):.1f})"
                         for l, v in zip("ABCDEF"[:len(options)], options))
    d.text((4, H + 34), f"options: {opts_str}", fill="black")
    d.text((4, H + 50), f"true secondary={sorted(set(gt_letters)-{primary_letter})}  "
                        f"predicted secondary={sorted(pred_letters)}", fill="darkred")
    return sheet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n7_jsonl", required=True)
    ap.add_argument("--n9_jsonl", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n", type=int, default=20)
    a = ap.parse_args()

    n7 = {(r["recording_id"], r["segment_idx"]): r for r in (json.loads(l) for l in open(a.n7_jsonl))}
    n9 = {(r["recording_id"], r["segment_idx"]): r for r in (json.loads(l) for l in open(a.n9_jsonl))}
    keys = sorted(set(n7) & set(n9))

    con_pairs = []
    for k in keys:
        r = n9[k]
        secondary_gt = set(r["gt_letters"]) - {r["primary_letter"]}
        for l, s in r["contrastive_scores"].items():
            con_pairs.append((s, int(l in secondary_gt)))
    tau = best_f1_threshold(con_pairs)
    print(f"fixed threshold (same rule as n9c_bootstrap): tau={tau:.2f}")

    failures = []
    for k in keys:
        r7, r9 = n7[k], n9[k]
        secondary_gt = set(r9["gt_letters"]) - {r9["primary_letter"]}
        if not secondary_gt:
            continue
        pred = {l for l, s in r9["contrastive_scores"].items() if s > tau}
        if pred != secondary_gt:
            failures.append((k, r7, r9, pred))
    print(f"compound items: {sum(1 for k in keys if set(n9[k]['gt_letters'])-{n9[k]['primary_letter']})}"
          f"  failures (pred != true secondary set): {len(failures)}")

    os.makedirs(a.out_dir, exist_ok=True)
    selected = failures[:a.n]
    vr_cache = {}
    rows = []
    for k, r7, r9, pred in selected:
        vp = r7["video"]
        if vp not in vr_cache:
            vr_cache[vp] = VideoReader(vp, num_threads=1)
        vr = vr_cache[vp]
        sheet = make_sheet(vr, vr.get_avg_fps(), r7["start"], r7["end"], r7["frame_indices"],
                           r7["gt_name"], r7["options"], r9["gt_letters"], r9["primary_letter"],
                           pred, r9["contrastive_scores"])
        out_path = os.path.join(a.out_dir, f"{k[0]}_seg{k[1]}.png")
        sheet.save(out_path)
        rows.append({"recording_id": k[0], "segment_idx": k[1], "gt_name": r7["gt_name"],
                    "true_secondary": sorted(set(r9["gt_letters"]) - {r9["primary_letter"]}),
                    "predicted_secondary": sorted(pred), "image": out_path})
        print(f"{k[0]} seg{k[1]}: GT='{r7['gt_name']}' true_secondary="
              f"{sorted(set(r9['gt_letters'])-{r9['primary_letter']})} predicted={sorted(pred)} -> {out_path}")

    csv_path = os.path.join(a.out_dir, "audit.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["recording_id", "segment_idx", "gt_name", "true_secondary", "predicted_secondary",
                    "image", "failure_type", "secondary_duration_short", "ontology_mismatch",
                    "same_action_family", "notes"])
        for row in rows:
            w.writerow([row["recording_id"], row["segment_idx"], row["gt_name"],
                       ";".join(row["true_secondary"]), ";".join(row["predicted_secondary"]),
                       row["image"], "", "", "", "", ""])
    print(f"\nwrote {len(rows)} audit rows -> {csv_path}")
    print("fill in while looking at the image: failure_type (free text), "
          "secondary_duration_short (yes/no -- was the true secondary action "
          "only visible for a small fraction of the clip), ontology_mismatch "
          "(yes/no -- is the GT compound label describing something that "
          "isn't obviously a SEPARATE visual action), same_action_family "
          "(yes/no -- is the missed/extra secondary visually close to the "
          "primary, e.g. scrub vs rinse), notes")


if __name__ == "__main__":
    main()
