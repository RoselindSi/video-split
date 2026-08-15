"""The real task: does the 8B reranker detect annotation errors people made?

Everything measured so far is a SYNTHETIC counterfactual -- one word swapped, a
clause dropped, two clauses exchanged. The 8B reranker scores 0.933 across
those. The task this project exists for is different: judge a label an
annotator actually wrote, where the error is subtler than a substituted word
and sometimes is not an error at all but a disagreement.

The 89 audited events are the only real ruler. 46 were judged
`claim_support=yes` and 17 `no`. This arm scores each event's segments against
its stored label and asks whether the score separates the two.

IT REUSES THE COSINE ARM'S JOIN EXACTLY -- the same event map, the same
`shown_in_sheet` restriction, the same per-event aggregation over the segments
the annotator was actually shown. Only the scorer changes, so a difference from
the cosine arm's numbers is the scorer.

THE CONFOUND THAT MADE THE COSINE VERSION UNREADABLE HAS NOT GONE AWAY. One
recording of 32 carries both classes and 99% of YES/NO pairs straddle
recordings, so a video-text scorer can reach a high AUROC by recognising the
kitchen. `print_within_between` runs before the score, every time, and if
fewer than ten percent of pairs are within-recording the number cannot separate
semantics from scene no matter how large it is.

WHAT A LOW NUMBER WOULD MEAN. 0.933 on controlled pairs and something near
chance here would not be a failure of the model. It would say that real
annotation errors are not the same object as a swapped word -- which is a
finding about how auditable this annotation is, and this project has reached
that conclusion from two other directions already.

Usage:
    python -m src.auditor.semantic.reranker_claim_support \
        --model /workspace/tr1/ckpts/Qwen3-VL-Reranker-8B \
        --gold data/gold/semantic_ontology_gold_48.json \
        --gold data/gold/semantic_enrichment_gold_41.csv \
        --join /workspace/tr1/results/auditor/naming_run_join.json \
        --data /workspace/tr1/results/auditor/naming_run.json \
        --event_map /workspace/tr1/results/auditor/naming_targets_48_event_map.json \
        --event_map /workspace/tr1/results/auditor/naming_targets_enrichment_event_map.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from src.auditor.semantic.claim_support_diagnostic import (
    auroc, grouped_boot, load_gold, min_detectable, norm_key,
    print_within_between)
from src.auditor.semantic.cosine_baseline import sample_times, write_frames
from src.auditor.semantic.reranker_baseline import score_batch


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model")
    ap.add_argument("--gold", action="append", required=True)
    ap.add_argument("--join", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--event_map", action="append", required=True)
    ap.add_argument("--n_frames", type=int, default=32)
    ap.add_argument("--total_pixels", type=int, default=3584 * 28 * 28)
    ap.add_argument("--frame_dir", default="/tmp/claimsupport_frames")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()

    rows = load_gold(a.gold)
    lab = {(r.get("audit_key") or r.get("event_id")): r["claim_support"]
           for r in rows}
    join = json.load(open(a.join, encoding="utf-8"))
    emap = {}
    for p in a.event_map:
        for k, v in json.load(open(p, encoding="utf-8")).items():
            emap[norm_key(k)] = v
    video_of = {r["recording_id"]: r["video"]
                for r in json.load(open(a.data, encoding="utf-8"))}
    print(f"{len(rows)} audited events; {len(join)} joined segments; "
          f"{len(emap)} mapped events; {len(video_of)} videos")

    need = {}
    for key, cs in lab.items():
        if cs not in ("yes", "no"):
            continue
        m = emap.get(norm_key(key))
        if not m:
            continue
        for s in m["segments"]:
            if not s.get("shown_in_sheet", True):
                continue
            j = join.get(s["segment_uid"])
            if j and video_of.get(j["recording_id"]):
                need[s["segment_uid"]] = j
    print(f"  {len(need)} distinct segments to score "
          f"(shown_in_sheet, YES/NO events only)")
    if not need:
        raise SystemExit("no segments joined; the event map or the join file "
                         "does not match this gold.")

    model = proc = None
    if not a.dry_run:
        if not a.model or not os.path.exists(a.model):
            raise SystemExit(f"--model {a.model} does not exist")
        from sentence_transformers import CrossEncoder
        from transformers import AutoProcessor
        model = CrossEncoder(a.model, trust_remote_code=True)
        proc = AutoProcessor.from_pretrained(a.model)
        os.makedirs(a.frame_dir, exist_ok=True)

    rng = np.random.default_rng(a.seed)
    score = {}
    for i, (uid, j) in enumerate(sorted(need.items())):
        text = j.get("stored_label") or j.get("label")
        if not text:
            print(f"  !! {uid} has no stored label; skipped")
            continue
        if a.dry_run:
            score[uid] = float(rng.normal())
            continue
        frames = write_frames(video_of[j["recording_id"]],
                              sample_times(j["start"], j["end"], a.n_frames),
                              a.frame_dir, uid.replace("/", "_"))
        dur = max(float(j["end"]) - float(j["start"]), 1e-3)
        nf = len(frames)
        meta = {"fps": (nf - 1) / dur if nf > 1 else 1.0,
                "total_num_frames": nf, "duration": dur,
                "frames_indices": np.arange(nf)}
        score[uid] = float(score_batch(model, proc, "sentence_transformers",
                                       frames, [text], a.total_pixels, None,
                                       meta)[0])
        for q in frames:
            os.remove(q)
        if (i + 1) % 25 == 0:
            print(f"    {i + 1}/{len(need)} scored", flush=True)

    # THE SAME AGGREGATION THE COSINE ARM USED, over the segments the annotator
    # was shown. Changing it here would make the two arms incomparable while
    # both still printed an AUROC.
    ev, y, grp = [], [], []
    for key, cs in lab.items():
        if cs not in ("yes", "no"):
            continue
        m = emap.get(norm_key(key))
        if not m:
            continue
        ss = [score[s["segment_uid"]] for s in m["segments"]
              if s.get("shown_in_sheet", True) and s["segment_uid"] in score]
        if not ss:
            continue
        ev.append({"audit_key": key, "min": min(ss),
                   "mean": sum(ss) / len(ss), "n_segments": len(ss)})
        y.append(1 if cs == "yes" else 0)
        grp.append(m["recording_id"])
    print(f"\nCOVERAGE: {len(ev)} events scored, {sum(y)} YES vs "
          f"{len(y) - sum(y)} NO over {len(set(grp))} recordings")
    if sum(y) < 2 or len(y) - sum(y) < 2:
        raise SystemExit("not enough of one class")

    print_within_between(y, grp, "reranker AUROC")
    bar = min_detectable(sum(y), len(y) - sum(y), a.n_boot, a.seed)
    print(f"\n  A RANDOM scorer reaches AUROC {bar:.3f} at the 97.5th "
          f"percentile with {sum(y)} vs {len(y) - sum(y)}.")

    print(f"\nRERANKER ON REAL claim_support"
          + ("   (DRY RUN, random)" if a.dry_run else ""))
    print(f"  {'aggregation':<12}{'AUROC':>8}{'grouped 95%':>22}{'vs bar':>9}")
    res = {}
    for f in ("min", "mean"):
        s = [e[f] for e in ev]
        au = auroc(s, y)
        lo, hi = grouped_boot(s, y, grp, a.n_boot, a.seed)
        mark = "   > chance band" if au - bar > 0.02 and lo > 0.5 else ""
        print(f"  {f:<12}{au:>8.3f}   [{lo:.3f}, {hi:.3f}]{au - bar:>+9.3f}"
              f"{mark}")
        res[f] = {"auroc": au, "lo": lo, "hi": hi, "bar": bar}

    print(f"\n  0.933 on synthetic counterfactuals and something near this "
          f"bar here is not a\n  model failure -- it says a real annotation "
          f"error is not the same object as a\n  swapped word. Read the "
          f"within/between line above before reading any of these\n  numbers "
          f"as semantics rather than scene.")

    if a.out:
        json.dump({"n_events": len(ev), "n_yes": int(sum(y)),
                   "bar": bar, "results": res,
                   "events": [dict(e, y=yy, recording_id=g)
                              for e, yy, g in zip(ev, y, grp)]},
                  open(a.out, "w", encoding="utf-8"), indent=2)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
