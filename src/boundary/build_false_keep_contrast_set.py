"""Build the manual-review contrast sheet for the HAL held-out failure.

Input: hal_failure_diagnostic.py's hal_false_keep_contributions.csv (already
contains each false keep grouped with its matched true keeps -- same
recording first, then same source_category, then nearest HAL score). This
script joins in the batch2 media paths + segment-label context and emits ONE
review CSV with blank diagnostic columns for a human to fill while watching
each clip.

The six diagnostic fields (fill yes/no/unclear) are chosen so their column
sums directly select the next fix -- this is a mechanism census, not another
re-audit of gold:

  returns_to_previous_state : after the candidate, does the scene/hand state
      revert to what it was before? mostly-yes on false keeps => the missing
      feature is return-to-baseline, build that first.
  new_object_contact        : hand makes contact with an object it wasn't
      touching before.
  object_released           : an object is put down / let go.
  interaction_target_changed: hand switches to a different object/part.
      (these three mostly-no on false keeps while context_change is high =>
      persistent VISUAL change without a new action state -- representation
      limit of the frozen embedding, motivates the contrastive adapter.)
  nearby_boundary_contamination : another real transition within ~3s of the
      candidate, inside the feature windows. mostly-yes => the windows are
      polluted -- fix window design, not the representation.
  camera_or_viewpoint_change : head/camera motion or viewpoint shift is the
      dominant visual change. mostly-yes => need a camera-motion nuisance
      feature + recording-relative normalization.

Usage (server):
    python -m src.boundary.build_false_keep_contrast_set \
        --contributions /workspace/tr1/results/hal/failure_diagnostic/hal_false_keep_contributions.csv \
        --media_csv /workspace/tr1/results/hal/batch2_media/audit_sample.csv \
        --out /workspace/tr1/results/hal/failure_diagnostic/false_keep_review_sheet.csv
Then print the watch-list:
    python -c "import csv; rows=list(csv.DictReader(open('/workspace/tr1/results/hal/failure_diagnostic/false_keep_review_sheet.csv')));\
[print(r['group_id'], r['role'], r['event_id'], r['hal_score'], r['clip_path']) for r in rows]"
"""
from __future__ import annotations

import argparse
import csv
import os

DIAG_FIELDS = [
    "returns_to_previous_state",
    "new_object_contact",
    "object_released",
    "interaction_target_changed",
    "nearby_boundary_contamination",
    "camera_or_viewpoint_change",
    "failure_mechanism_notes",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--contributions", required=True,
                    help="hal_false_keep_contributions.csv from hal_failure_diagnostic.py")
    ap.add_argument("--media_csv", required=True,
                    help="batch2_media/audit_sample.csv (for clip/contact-sheet/plot paths + labels)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    with open(a.media_csv, newline="", encoding="utf-8", errors="replace") as f:
        media = {r["event_id"].strip(): r for r in csv.DictReader(f) if r.get("event_id")}

    rows_out = []
    with open(a.contributions, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            eid = r["event_id"]
            m = media.get(eid, {})
            if not m:
                print(f"  !! {eid}: not found in {os.path.basename(a.media_csv)} "
                      f"(media may not have been rendered for it)")
            rows_out.append({
                "group_id": r["group_id"], "role": r["role"], "event_id": eid,
                "recording_id": r["recording_id"],
                "source_category": r.get("source_category", ""),
                "gold_temporal_truth": r["gold_temporal_truth"],
                "hal_score": r["score"],
                "top_contribution": _top_contrib(r),
                "gt_time": m.get("gt_time", ""), "pred_time": m.get("pred_time", ""),
                "prev_segment_label": m.get("prev_segment_label", ""),
                "next_segment_label": m.get("next_segment_label", ""),
                "containing_segment_label": m.get("containing_segment_label", ""),
                "clip_path": m.get("clip_path", ""),
                "contact_sheet_path": m.get("contact_sheet_path", ""),
                "score_plot_path": m.get("score_plot_path", ""),
                **{k: "" for k in DIAG_FIELDS},
            })

    rows_out.sort(key=lambda r: (int(r["group_id"]), 0 if r["role"] == "false_keep" else 1))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    n_fk = sum(1 for r in rows_out if r["role"] == "false_keep")
    print(f"wrote {a.out}: {len(rows_out)} rows ({n_fk} false keeps + matched true keeps), "
          f"{len(DIAG_FIELDS)} blank diagnostic columns to fill")
    print("\nWATCH LIST (group by group; false_keep first, then its matched true keeps):")
    for r in rows_out:
        print(f"  [{r['group_id']}] {r['role']:<17} {r['event_id']}  "
              f"score={r['hal_score']}  gold={r['gold_temporal_truth']}  drv={r['top_contribution']}")


def _top_contrib(row):
    """Name the single largest |contribution| feature for quick orientation."""
    best, best_v = "", 0.0
    for k, v in row.items():
        if k.startswith("contrib_") and v not in ("", None):
            try:
                fv = float(v)
            except ValueError:
                continue
            if abs(fv) > abs(best_v):
                best, best_v = k[len("contrib_"):], fv
    return f"{best}{best_v:+.2f}" if best else ""


if __name__ == "__main__":
    main()
