"""Turn the contrast-completion sheet into what the media renderer wants.

`render_batch3_media` takes a manifest (event_id, recording_id, t) and a blind
CSV keyed by event_id that supplies segment-label context and nothing else.
This writes both, plus a third file that keeps the sampling story away from
the person doing the labelling.

WHY THE EVENT IDS ARE OPAQUE. batch4's were shaped
`recording_000004_batch3_gt_boundary_t117.0`, and the renderer names each clip
after the event id -- so the candidate type was printed on the media filename
the reviewer opened. A review that is blind in the CSV and not blind in the
file listing is blind in the part nobody looks at. These are `b5_0001`.

WHAT THE REVIEWER SEES AND DOES NOT. The blind CSV carries the timestamp, the
surrounding segment labels, and empty verdict columns. `needed_side` and
`sampled_because` go to a separate provenance file: they say why the instant
was chosen, which is legitimate to keep and anchoring to show. Half the rows
sampled next to a stored boundary are expected to come back as not a boundary,
so telling the reviewer "this was sampled as a positive" would be telling them
the answer we are hoping for.

SEGMENT LABELS COME FROM THE STORED ANNOTATION, and that is fine here in a way
it is not elsewhere: they are context for a person watching a clip, not a
verdict. The same stored annotation is 44% wrong about where boundaries are,
which is exactly what the reviewer is being asked about -- so the labels
describe what is happening around the instant and never assert that the
instant is or is not a boundary.

Usage:
    python -m src.auditor.boundary.completion_sheet_to_media \
        --sheet data/gold/batch5_contrast_completion_sheet.csv \
        --feat_cache ... --feat_cache ... \
        --out_dir results/hal/batch5
"""
from __future__ import annotations

import argparse
import csv
import json
import os


def segment_context(segs, t):
    """(prev, containing, next) labels around t. Junctions have no containing.

    A candidate sitting exactly on a segment edge is inside no segment, which
    is why `containing_segment_label` is empty for precisely the rows that
    matter most -- reading that column alone once produced 29 events where
    there were 63, and concluding from it that the labels did not exist."""
    prev = cont = nxt = ""
    best_p = best_n = None
    for name, s0, s1 in segs:
        s0, s1 = float(s0), float(s1)
        if s0 < t < s1:
            cont = name
        if s1 <= t + 1e-6 and (best_p is None or s1 > best_p):
            best_p, prev = s1, name
        if s0 >= t - 1e-6 and (best_n is None or s0 < best_n):
            best_n, nxt = s0, name
    return prev, cont, nxt


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sheet", required=True)
    ap.add_argument("--feat_cache", action="append", required=True,
                    help="APPEND one flag per file -- a non-append flag keeps "
                         "only the last, which has silently shrunk a run "
                         "before.")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--prefix", default="b5")
    a = ap.parse_args()

    from src.auditor.boundary.ontology_constitution import Constitution
    Constitution().check_dataset_use("batch4_joint_audit", "mine_pairs")

    rows = [r for r in csv.DictReader(
        l for l in open(a.sheet, encoding="utf-8-sig") if not l.startswith("#"))]
    print(f"{len(rows)} rows from {a.sheet}")

    from src.boundary.hal_features import load_feature_caches
    caches = load_feature_caches(a.feat_cache)
    print(f"caches cover {len(caches)} recordings")

    os.makedirs(a.out_dir, exist_ok=True)
    man = os.path.join(a.out_dir, f"{a.prefix}_manifest.jsonl")
    blind = os.path.join(a.out_dir, f"{a.prefix}_blind_review.csv")
    prov = os.path.join(a.out_dir, f"{a.prefix}_provenance.jsonl")

    FILL = ("temporal_event_type", "within_1s_tolerance",
            "interaction_relation", "true_boundary_start_s", "audit_note")
    cols = (["event_id", "recording_id", "t", "prev_segment_label",
             "containing_segment_label", "next_segment_label"] + list(FILL)
            + ["clip_path", "contact_sheet_path"])

    nm = nb = 0
    missing = 0
    with open(man, "w", encoding="utf-8") as fm, \
            open(blind, "w", encoding="utf-8-sig", newline="") as fb, \
            open(prov, "w", encoding="utf-8") as fp:
        w = csv.DictWriter(fb, fieldnames=cols)
        w.writeheader()
        for i, r in enumerate(rows, start=1):
            eid = f"{a.prefix}_{i:04d}"
            rid = r["recording_id"]
            t = float(r["candidate_time_s"])
            rec = caches.get(rid)
            if rec is None:
                missing += 1
                continue
            prev, cont, nxt = segment_context(rec.get("segments") or [], t)
            fm.write(json.dumps({"event_id": eid, "recording_id": rid,
                                 "t": t}) + "\n")
            nm += 1
            w.writerow({"event_id": eid, "recording_id": rid, "t": f"{t:.1f}",
                        "prev_segment_label": prev,
                        "containing_segment_label": cont,
                        "next_segment_label": nxt,
                        **{c: "" for c in FILL},
                        "clip_path": "", "contact_sheet_path": ""})
            nb += 1
            fp.write(json.dumps({
                "event_id": eid, "recording_id": rid, "t": t,
                "candidate_type": r.get("candidate_type", ""),
                "needed_side": r.get("needed_side", ""),
                "sampled_because": r.get("sampled_because", ""),
            }, ensure_ascii=False) + "\n")

    print(f"\nwrote {nm} -> {man}")
    print(f"wrote {nb} -> {blind}")
    print(f"wrote {nb} -> {prov}   (NOT for the reviewer)")
    if missing:
        print(f"  {missing} rows dropped: recording absent from the caches")
    print(f"\n  The reviewer gets the blind CSV and the media. needed_side and")
    print(f"  sampled_because stay in the provenance file, because about half "
          f"the rows\n  sampled beside a stored boundary should come back as "
          f"NOT a boundary --\n  saying which side we were hoping for would "
          f"be telling them the answer.")
    print(f"\n  next: python -m src.boundary.render_batch3_media \\")
    print(f"          --manifest {man} \\")
    print(f"          --blind_csv {blind} \\")
    print(f"          --data <recseg json> [--data ...] \\")
    print(f"          --out_dir {a.out_dir}")


if __name__ == "__main__":
    main()
