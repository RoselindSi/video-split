"""Turn saved val-split logits into the two files `auditor_v1 --run` consumes.

`--run` has only ever been exercised on a synthetic fixture, because the
predicted recseg lives on a machine that is packed. b2_logits.pt turns out to
carry everything the end-to-end path needs: per-frame probability, the
annotated segments with their labels, and the boundaries.

WHAT THIS IS AND IS NOT. The segments here are the ANNOTATION's, not a
decoder's output, so this exercises the auditor end to end on real scores and
real labels -- it does not evaluate a decoder. The boundary each segment is
audited on is its own start, matched to the nearest detector peak within
tolerance; a segment whose start no peak proposed gets no score, and
`route_boundary` sends it to REVIEW saying exactly that. Those are real cases,
not padding: the detector recovers about a third of the annotated boundaries.

NO SEMANTIC SCORES ARE PRODUCED. Scoring these 30 recordings needs their video,
which is on the same packed machine. Every segment therefore arrives with no
semantic score and routes to REVIEW for that reason, which is the honest state
of the semantic arm anyway -- driver A did not clear its capability gate, so no
--semantic_thr could take effect even if the scores existed.

Usage:
    python -m src.auditor.boundary.logits_to_recseg \
        --logits ~/Downloads/tr1_audits/results/boundary/b2_logits.pt \
        --out_recseg /tmp/recseg_val30.json \
        --out_scores /tmp/boundary_scores_val30.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from src.auditor.boundary.detector_calibration import BASE_THR, MIN_GAP_S, peaks

TOL_S = 1.0


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logits", required=True)
    ap.add_argument("--out_recseg", required=True)
    ap.add_argument("--out_scores", required=True)
    ap.add_argument("--tol_s", type=float, default=TOL_S)
    ap.add_argument("--base_thr", type=float, default=BASE_THR)
    a = ap.parse_args()

    import torch
    recs = torch.load(a.logits, map_location="cpu", weights_only=False)

    segs, scores, matched, unmatched = [], [], 0, 0
    for r in recs:
        rid = r["recording_id"]
        idx, p, t = peaks(r["prob"], r["times"], a.base_thr, MIN_GAP_S)
        pk = sorted((float(t[i]), float(p[i])) for i in idx)
        for j, (label, s0, s1) in enumerate(sorted(r["segments"],
                                                   key=lambda x: x[1])):
            sid = f"{rid}#seg{j:03d}"
            near = [(abs(pt - s0), pt, ps) for pt, ps in pk
                    if abs(pt - s0) <= a.tol_s]
            segs.append({"segment_id": sid, "boundary_id": sid,
                         "recording_id": rid, "start": float(s0),
                         "end": float(s1), "boundary_time": float(s0),
                         "label": label})
            if near:
                _, pt, ps = min(near)
                scores.append({"event_id": sid, "score": ps,
                               "peak_time": pt})
                matched += 1
            else:
                unmatched += 1

    json.dump(segs, open(a.out_recseg, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    json.dump(scores, open(a.out_scores, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"{len(recs)} recordings -> {len(segs)} segments")
    print(f"  {matched} carry a detector peak within {a.tol_s}s of their start")
    print(f"  {unmatched} do not, and route to REVIEW saying so "
          f"({unmatched / len(segs):.1%})")
    print(f"\nwrote {a.out_recseg}\nwrote {a.out_scores}")


if __name__ == "__main__":
    main()
