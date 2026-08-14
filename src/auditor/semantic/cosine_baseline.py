"""Baseline 1: video-text cosine on the exact audited segment windows.

ZERO TRAINING. A dual encoder puts the segment and its stored label in one
space and the score is their cosine. It answers one question the naming arm
could not: is there ANY global video-text alignment signal about whether a
stored label is supported, before anything is fitted.

WHAT IT REPLACES AND WHY. The failed arm generated a name from the video and
compared it to the stored label with verb and object overlap -- so the stored
label never entered the model, and every comparison ran through the naming
model's own 22% verb accuracy. Here the label is an input. That is a different
mechanism, not a better version of the same one.

THE SAME EVALUATION AS EVERY OTHER ARM, so the number goes in the same table:
the same YES/NO events, per-event aggregation over the segments the annotator
was SHOWN, recording-grouped bootstrap, and the random-scorer bar recomputed
at this n.

AND THE SAME WARNING, which matters more here than anywhere. A video-text
encoder sees the scene, and the scene identifies the recording. On this gold
one recording of 32 carries both classes and 99% of the YES/NO pairs straddle
recordings, so a high cosine AUROC would be as unreadable as the video prior's
0.827 was. The pair structure prints before the score, every time.

THE EMBEDDING CALL IS NOT VERIFIED. I have not run this model and its API
comes from its model card, not from anything I have executed -- and guessing
signatures from memory has cost this project four API errors in one file
already. So the call sits behind one function, and `--dry_run` exercises frame
extraction, aggregation, joining and scoring with random unit vectors so the
plumbing is known good before the model exists. A dry run that reports AUROC
near 0.5 is the plumbing working, not a result.

Usage:
    python -m src.auditor.semantic.cosine_baseline --dry_run \
        --join .../naming_run_join.json --data .../naming_run.json \
        --event_map .../naming_targets_48_event_map.json \
        --gold data/gold/semantic_ontology_gold_48.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re

import numpy as np

from src.auditor.semantic.claim_support_diagnostic import (
    auroc, grouped_boot, load_gold, min_detectable, norm_key,
    print_within_between)


def sample_times(start, end, n):
    """Uniform frame times inside the segment, endpoints included."""
    if n <= 1 or end <= start:
        return [(start + end) / 2.0]
    return [start + i * (end - start) / (n - 1) for i in range(n)]


def embed(model, proc, frames, texts, device):
    """The one unverified call. Returns (video_vecs, text_vecs), L2-normed.

    Qwen3-VL-Embedding is a dual tower over a shared space, so a video and a
    text go through separate forward passes and the score is a cosine. The
    exact argument names come from the model card; if they are wrong this is
    the only place to fix, which is why nothing else in the file touches the
    model."""
    import torch
    with torch.no_grad():
        v = model.get_video_embeddings(frames) if hasattr(
            model, "get_video_embeddings") else model.encode_video(frames)
        t = model.get_text_embeddings(texts) if hasattr(
            model, "get_text_embeddings") else model.encode_text(texts)
    v = torch.nn.functional.normalize(v, dim=-1).float().cpu().numpy()
    t = torch.nn.functional.normalize(t, dim=-1).float().cpu().numpy()
    return v, t


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", action="append", required=True)
    ap.add_argument("--join", required=True)
    ap.add_argument("--data", required=True,
                    help="naming_run.json, for the video path per recording")
    ap.add_argument("--event_map", action="append", required=True)
    ap.add_argument("--model", help="local path to Qwen3-VL-Embedding-2B")
    ap.add_argument("--n_frames", type=int, default=8)
    ap.add_argument("--dry_run", action="store_true",
                    help="random unit vectors instead of the model. Verifies "
                         "the plumbing; an AUROC near 0.5 from a dry run is "
                         "the plumbing working, not a result")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
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

    # which segments are actually needed: shown_in_sheet only, YES/NO only
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
    print(f"  {len(need)} distinct segments to encode "
          f"(shown_in_sheet, YES/NO events only)")

    model = proc = None
    if not a.dry_run:
        if not a.model or not os.path.exists(a.model):
            raise SystemExit(
                f"--model must point at a local Qwen3-VL-Embedding checkpoint."
                f"\n  Download with:\n    hf download "
                f"Qwen/Qwen3-VL-Embedding-2B --local-dir "
                f"/workspace/tr1/ckpts/Qwen3-VL-Embedding-2B\n"
                f"  (`huggingface-cli` was renamed to `hf` and no longer "
                f"works.)\n  Or use --dry_run to verify the plumbing "
                f"first.")
        import torch
        from transformers import AutoModel, AutoProcessor
        proc = AutoProcessor.from_pretrained(a.model, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            a.model, dtype=torch.bfloat16, device_map="cuda",
            trust_remote_code=True).eval()

    rng = random.Random(a.seed)
    score = {}
    for i, (uid, j) in enumerate(sorted(need.items())):
        if a.dry_run:
            v = np.array([rng.gauss(0, 1) for _ in range(64)])
            t = np.array([rng.gauss(0, 1) for _ in range(64)])
            score[uid] = float(v @ t / (np.linalg.norm(v)
                                        * np.linalg.norm(t)))
            continue
        from decord import VideoReader
        vr = VideoReader(video_of[j["recording_id"]])
        fps = vr.get_avg_fps()
        idx = [max(0, min(len(vr) - 1, int(t * fps)))
               for t in sample_times(j["start"], j["end"], a.n_frames)]
        frames = vr.get_batch(idx).asnumpy()
        vv, tt = embed(model, proc, [frames], [j["stored_label"]], "cuda")
        score[uid] = float(vv[0] @ tt[0])
        if (i + 1) % 25 == 0:
            print(f"    {i + 1}/{len(need)} encoded", flush=True)

    # per event: the same aggregation the naming features used
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
        ev.append({"audit_key": key, "cos_min": min(ss),
                   "cos_mean": sum(ss) / len(ss), "n_segments": len(ss)})
        y.append(1 if cs == "yes" else 0)
        grp.append(m["recording_id"])
    print(f"\nCOVERAGE: {len(ev)} events scored, "
          f"{sum(y)} YES vs {len(y) - sum(y)} NO over {len(set(grp))} "
          f"recordings")
    if sum(y) < 2 or len(y) - sum(y) < 2:
        raise SystemExit("not enough of one class")

    print_within_between(y, grp, "cosine AUROC")
    bar = min_detectable(sum(y), len(y) - sum(y), a.n_boot, a.seed)
    print(f"\n  A RANDOM scorer reaches AUROC {bar:.3f} at the 97.5th "
          f"percentile with {sum(y)} vs {len(y) - sum(y)}.")

    print(f"\nCOSINE BASELINE"
          + ("  (DRY RUN -- random vectors, no model)" if a.dry_run else ""))
    print(f"  {'feature':<12}{'AUROC':>8}{'grouped 95%':>22}")
    res = {}
    for f in ("cos_min", "cos_mean"):
        s = [e[f] for e in ev]
        au = auroc(s, y)
        lo, hi = grouped_boot(s, y, grp, a.n_boot, a.seed)
        print(f"  {f:<12}{au:>8.3f}   [{lo:.3f}, {hi:.3f}]"
              + ("   > chance band" if au > bar else ""))
        res[f] = {"auroc": au, "grouped_95": [lo, hi]}
    print(f"\n  for the same table: naming verb_min 0.558  verb_mean 0.566  "
          f"obj_min 0.566\n  obj_mean 0.563  |  video prior 0.827 "
          f"(uninterpretable, 99% between-recording)")
    if a.dry_run:
        print(f"\n  DRY RUN. These AUROCs come from random vectors and are "
              f"expected near 0.5.\n  What this run verifies is the join, the "
              f"segment selection, the aggregation and\n  the scoring -- so "
              f"that the first real run tests the model and nothing else.")

    if a.out:
        json.dump({"dry_run": a.dry_run, "n_events": len(ev),
                   "n_yes": sum(y), "n_no": len(y) - sum(y),
                   "bar": bar, "results": res, "events": ev},
                  open(a.out, "w", encoding="utf-8"), indent=2)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
