"""Generate the pair-relabelling worksheet: every audited decisive event,
ordered so the most diagnostic cases are reviewed first.

This is step 4-5 of the revised plan, and it is now the highest-value action
rather than more model work. Evidence: the contrastive adapter got train
AUROC 0.758 / test 0.499 -- train being only mediocre means it could not fit
even its own training pairs, i.e. the supervision contradicts itself, which
manual video review then explained directly (see pair_taxonomy.py).

Priority order (highest first), each recorded in `priority_reason` so the
reviewer knows why a row is near the top:

  1. contradicts_raw_geometry_positive  -- gold `positive` whose left/right
     windows are among the CLOSEST in the raw embedding. Either the split is
     an annotation convention / gradual phase, or the features genuinely
     cannot see it; both change how it may be used.
  2. contradicts_raw_geometry_negative  -- gold `motion_hard_negative` whose
     windows are among the FARTHEST apart. Prime suspects for camera shift,
     offscreen, or an unlabelled sub-action.
  3. systematic_recording_error        -- recordings where the frozen v1
     scorer was confidently wrong more than once (e.g. recording_000406 gave
     three high-confidence keeps, all wrong): a per-recording mechanism, not
     scattered noise.
  4. window_contamination_suspect      -- another audited event within 3s, so
     the +-3s feature windows overlap heavily. The manual watch list already
     contains a pair 1.0s apart (~83% window overlap) with opposite gold
     labels -- unresolvable by construction, not by better features.
  5. remainder                         -- everything else, still needs a
     subtype before the clean subset can be called complete.

Pre-filled: the seven events already reviewed on video (pair_taxonomy.
REVIEWED_SUBTYPES) come with their subtype and the reviewer's reason, so
that work is not repeated.

The reviewer fills `temporal_pair_subtype` (closed vocabulary, see
pair_taxonomy.SUBTYPES). `pair_supervision` may be left blank -- it is then
derived from the subtype by the default mapping.

Usage (server):
    python -m src.boundary.build_pair_relabel_sheet \
        --feat_cache /workspace/tr1/data_recseg/feat_train_full_noblur_multi.pt \
        --feat_cache /workspace/tr1/data_recseg/feat_val_full_noblur_multi.pt \
        --feat_cache /workspace/tr1/data_recseg_part2/feat_part2_full_noblur_multi.pt \
        --media_csv /workspace/tr1/results/boundary/error_audit/media/audit_sample.csv \
        --media_csv /workspace/tr1/results/hal/batch2_media/audit_sample.csv \
        --out /workspace/tr1/results/hal/pair_relabel_sheet.csv
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import torch.nn.functional as F

from src.auditor import gold_schema as S
from src.boundary.hal_features import load_feature_caches
from src.boundary.state_adapter import build_events
from src.boundary.adapter_diagnostics import lr_distance
from src.boundary import pair_taxonomy as T

CONTRADICT_K = 20
CONTAMINATION_S = 3.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feat_cache", action="append", required=True)
    ap.add_argument("--gold", default="data/gold/audit_188_gold_v2.jsonl")
    ap.add_argument("--context", default="data/gold/audit_188_context.jsonl")
    ap.add_argument("--media_csv", action="append", default=[],
                    help="audit_sample.csv files (repeat for batch1 + batch2)")
    ap.add_argument("--heldout_json",
                    help="heldout_validation.json -- marks recordings with repeated "
                         "confident errors as priority 3")
    ap.add_argument("--top_k", type=int, default=CONTRADICT_K)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    gold = S.load_gold(a.gold)
    ctx = S.load_context(a.context)
    by_rid = load_feature_caches(a.feat_cache)
    events = build_events(gold, ctx, by_rid)
    print(f"decisive events: {len(events)}")

    media = {}
    for path in a.media_csv:
        if not os.path.exists(path):
            print(f"  !! media csv not found, skipping: {path}")
            continue
        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            for r in csv.DictReader(f):
                eid = (r.get("event_id") or "").strip()
                if eid:
                    media.setdefault(eid, r)
    print(f"media rows available: {len(media)}")

    bad_recordings = set()
    if a.heldout_json and os.path.exists(a.heldout_json):
        import json
        hv = json.load(open(a.heldout_json, encoding="utf-8"))
        cnt = {}
        for fk in hv.get("false_keeps", []):
            cnt[fk["recording_id"]] = cnt.get(fk["recording_id"], 0) + 1
        bad_recordings = {r for r, c in cnt.items() if c >= 2}
        print(f"recordings with >=2 confident false keeps: {sorted(bad_recordings)}")

    dist = []
    for e in events:
        emb = F.normalize(e["rec"]["feats"].float(), dim=-1)
        d = lr_distance(emb, e["rec"]["times"], e["t"])
        dist.append(np.nan if d is None else d)
    dist = np.array(dist, dtype=float)

    by_rec_times = {}
    for e in events:
        by_rec_times.setdefault(e["recording_id"], []).append(e["t"])

    pos_idx = [i for i, e in enumerate(events) if e["y"] == 1 and not np.isnan(dist[i])]
    neg_idx = [i for i, e in enumerate(events) if e["y"] == 0 and not np.isnan(dist[i])]
    p1 = set(sorted(pos_idx, key=lambda i: dist[i])[:a.top_k])
    p2 = set(sorted(neg_idx, key=lambda i: -dist[i])[:a.top_k])

    rows = []
    for i, e in enumerate(events):
        others = [abs(e["t"] - o) for o in by_rec_times[e["recording_id"]] if abs(e["t"] - o) > 1e-6]
        nearest_other = min(others) if others else None
        if i in p1:
            prio, reason = 1, "contradicts_raw_geometry_positive"
        elif i in p2:
            prio, reason = 2, "contradicts_raw_geometry_negative"
        elif e["recording_id"] in bad_recordings:
            prio, reason = 3, "systematic_recording_error"
        elif nearest_other is not None and nearest_other < CONTAMINATION_S:
            prio, reason = 4, "window_contamination_suspect"
        else:
            prio, reason = 5, "remainder"

        pre = T.REVIEWED_SUBTYPES.get(e["event_id"])
        m = media.get(e["event_id"], {})
        rows.append({
            "priority": prio, "priority_reason": reason,
            "event_id": e["event_id"], "recording_id": e["recording_id"],
            "t": round(e["t"], 2),
            "current_binary_role": "positive" if e["y"] == 1 else "motion_hard_negative",
            "raw_left_right_distance": None if np.isnan(dist[i]) else round(float(dist[i]), 6),
            "nearest_other_audited_event_s": None if nearest_other is None else round(nearest_other, 2),
            "temporal_pair_subtype": pre[0] if pre else "",
            "pair_supervision": "",
            "notes": pre[1] if pre else "",
            "prev_segment_label": m.get("prev_segment_label", ""),
            "next_segment_label": m.get("next_segment_label", ""),
            "containing_segment_label": m.get("containing_segment_label", ""),
            "clip_path": m.get("clip_path", ""),
            "contact_sheet_path": m.get("contact_sheet_path", ""),
            "score_plot_path": m.get("score_plot_path", ""),
        })

    rows.sort(key=lambda r: (r["priority"], -(r["raw_left_right_distance"] or 0)
                             if r["priority"] == 2 else (r["raw_left_right_distance"] or 0)))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    from collections import Counter
    print(f"\nwrote {a.out} ({len(rows)} rows)")
    for p, c in sorted(Counter(r["priority"] for r in rows).items()):
        reason = next(r["priority_reason"] for r in rows if r["priority"] == p)
        print(f"  priority {p} ({reason}): {c}")
    n_pre = sum(1 for r in rows if r["temporal_pair_subtype"])
    n_missing_media = sum(1 for r in rows if not r["clip_path"])
    print(f"  pre-filled from prior video review: {n_pre}")
    if n_missing_media:
        print(f"  !! {n_missing_media} rows have no clip path -- media was never rendered "
              f"for them (the original 72-event batch only rendered its own sample). "
              f"Those rows can still be labelled from the segment labels, but video "
              f"review needs render_audit_media.py run for their events first.")
    print(f"\nfill `temporal_pair_subtype` with one of: {T.SUBTYPES}")
    print(f"leave `pair_supervision` blank to accept the default mapping: "
          f"{T.SUBTYPE_TO_SUPERVISION}")


if __name__ == "__main__":
    main()
