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
from collections import defaultdict

import numpy as np

from src.auditor.semantic.claim_support_diagnostic import (
    auroc, grouped_boot, load_gold, min_detectable, norm_key,
    print_within_between)


def sample_times(start, end, n):
    """Uniform frame times inside the segment, endpoints included."""
    if n <= 1 or end <= start:
        return [(start + end) / 2.0]
    return [start + i * (end - start) / (n - 1) for i in range(n)]


def embed(model, video_inputs, texts, prompt=None, meta=None):
    """The one place that touches the model. Returns (video, text) vectors.

    Read off the checkpoint rather than guessed a third time. It is a
    SENTENCE-TRANSFORMERS model: modules.json lists Transformer / Pooling /
    Normalize, 1_Pooling says lasttoken over 2048 dims, and
    sentence_bert_config.json carries a modality_config with text, image,
    video and message entries. So the call is `model.encode(list)` where each
    item is a string or a dict like {"text": ...} / {"image": ...} /
    {"video": ...}. Normalize is already the last module, so
    normalize_embeddings would be redundant -- it is left off rather than
    applied twice.

    THE FRAMES ARE ALREADY SAMPLED. `do_sample_frames=False` goes through
    `processing_kwargs={"video": {...}}`; without it the video processor tries
    to sample frames from a list of images and raises. Sampling the window
    here rather than letting the processor sample the file is the whole reason
    the segment can be encoded at all.

    THE PROMPT IS A CHOICE THAT MOVES THE NUMBER. Every input is wrapped in
    "Represent the user's input." by default, and the model card shows
    retrieval usage passing a different one to the query side. Symmetric
    similarity is what this baseline measures, so both sides get the same
    prompt, and --prompt makes that visible instead of silent."""
    kw = {"show_progress_bar": False}
    if prompt:
        kw["prompt"] = prompt
    # do_sample_frames=False because the frames are ALREADY the sample. The
    # video processor otherwise tries to sample from a list of images and
    # refuses. The route is `processing_kwargs`, read off
    # Transformer.preprocess's signature and its
    # `effective_processing_kwargs.get(modality_key)` override loop -- not
    # guessed, after three guesses in this file already.
    vid = {"do_sample_frames": False}
    if meta:
        # Qwen3VL writes frame TIMESTAMPS into the prompt and derives them
        # from fps. With pre-sampled frames it cannot infer one and defaults
        # to 24, which makes every 8-frame clip look like a third of a second
        # -- across segments whose real durations run from 3s to 88s, and the
        # distortion scales with duration. This is metadata the model asks for
        # and would have in normal use, not a knob. `video_metadata` is part
        # of VideosKwargs, so it takes the same processing_kwargs route as
        # do_sample_frames, in the three-field shape this repo already uses in
        # eval_naming_decoupled.
        vid["video_metadata"] = meta
    vkw = dict(kw, processing_kwargs={"video": vid})
    return (np.asarray(model.encode(video_inputs, **vkw)),
            np.asarray(model.encode(texts, **kw)))


def write_frames(video, times, out_dir, uid):
    """Segment frames as JPEGs, because the video input is a list of frames.

    qwen-vl-utils accepts a video as a list of frame paths, which is what lets
    an arbitrary [start, end] window be encoded at all -- handing it the whole
    file would encode the recording, not the segment, and the segment is the
    unit the audit judged."""
    from decord import VideoReader
    from PIL import Image
    vr = VideoReader(video)
    fps = vr.get_avg_fps()
    idx = [max(0, min(len(vr) - 1, int(t * fps))) for t in times]
    arr = vr.get_batch(idx).asnumpy()
    paths = []
    for i, fr in enumerate(arr):
        p = os.path.join(out_dir, f"{uid}_{i:02d}.jpg")
        Image.fromarray(fr).save(p, quality=90)
        paths.append(p)
    return paths


def score_benchmark(a):
    """Score the paired benchmark. Each SEGMENT is encoded once.

    A segment appears in several pairs -- the original plus one counterfactual
    per kind -- and its video embedding does not change between them. Encoding
    it per pair would cost five times the GPU for identical vectors, and worse,
    would make the same video contribute five slightly different embeddings if
    anything in the pipeline were nondeterministic, which would show up as a
    margin that is really noise."""
    import os as _os
    bench = [json.loads(l) for l in open(a.benchmark_in, encoding="utf-8")
             if l.strip()]
    segs, texts = {}, defaultdict(set)
    for p in bench:
        segs[p["segment_uid"]] = p
        texts[p["segment_uid"]].add(p["original"])
        texts[p["segment_uid"]].add(p["counterfactual"])
    n_pairs = sum(len(v) for v in texts.values())
    print(f"{len(bench)} pairs over {len(segs)} segments; "
          f"{n_pairs} distinct (segment, text) entries to score")

    if a.dry_run:
        rng = random.Random(a.seed)
        rows = [{"segment_uid": u, "text": t, "score": rng.random()}
                for u, ts in texts.items() for t in ts]
    else:
        if not a.model or not _os.path.exists(a.model):
            raise SystemExit("--model is required unless --dry_run")
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(a.model, device="cuda",
                                    trust_remote_code=True)
        _os.makedirs(a.frame_dir, exist_ok=True)
        video_of = {r["recording_id"]: r["video"]
                    for r in json.load(open(a.data, encoding="utf-8"))}
        rows = []
        for i, (uid, p) in enumerate(sorted(segs.items())):
            vid = video_of.get(p["recording_id"])
            if not vid:
                print(f"  !! no video for {p['recording_id']}; "
                      f"{uid} skipped")
                continue
            paths = write_frames(vid, sample_times(p["start"], p["end"],
                                                   a.n_frames),
                                 a.frame_dir, uid.replace("/", "_"))
            dur = max(float(p["end"]) - float(p["start"]), 1e-3)
            n = len(paths)
            meta = [{"fps": (n - 1) / dur if n > 1 else 1.0,
                     "total_num_frames": n, "duration": dur,
                     "frames_indices": np.arange(n)}]
            tl = sorted(texts[uid])
            vv, tt = embed(model, [{"video": paths}], tl, a.prompt,
                           None if a.no_metadata else meta)
            v = vv[0] / (np.linalg.norm(vv[0]) + 1e-12)
            for t, e in zip(tl, tt):
                rows.append({"segment_uid": uid, "text": t,
                             "score": float(v @ (e / (np.linalg.norm(e)
                                                      + 1e-12)))})
            for q in paths:
                _os.remove(q)
            if (i + 1) % 25 == 0:
                print(f"    {i + 1}/{len(segs)} segments encoded", flush=True)

    out = a.scores_out or (a.benchmark_in.replace(".jsonl", "")
                           + "_cosine_scores.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nwrote {len(rows)} scores -> {out}"
          + ("   (DRY RUN, random)" if a.dry_run else ""))
    print(f"  then:\n    python -m src.auditor.semantic.paired_benchmark "
          f"\\\n      --evaluate {out} --benchmark {a.benchmark_in} "
          f"--gold {a.gold[0]}")


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
    ap.add_argument("--prompt", default=None,
                    help="applied to BOTH sides. None uses the model default, "
                         "\"Represent the user's input.\" A different prompt "
                         "gives a different number, so it is a flag rather "
                         "than a constant buried in the call")
    ap.add_argument("--frame_dir", default="/tmp/cosine_frames")
    ap.add_argument("--no_metadata", action="store_true",
                    help="reproduce the first run, which let the processor "
                         "assume fps=24 for every segment. Kept so the two "
                         "can be compared rather than one silently replacing "
                         "the other")
    ap.add_argument("--dry_run", action="store_true",
                    help="random unit vectors instead of the model. Verifies "
                         "the plumbing; an AUROC near 0.5 from a dry run is "
                         "the plumbing working, not a result")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--benchmark_in",
                    help="paired_semantic_benchmark.jsonl. Scores every "
                         "(segment, text) it names and writes them for "
                         "paired_benchmark --evaluate, instead of running the "
                         "per-event arm")
    ap.add_argument("--scores_out")
    ap.add_argument("--out")
    a = ap.parse_args()

    if a.benchmark_in:
        score_benchmark(a)
        return

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
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(a.model, device="cuda",
                                    trust_remote_code=True)
        mods = os.path.join(a.model, "modules.json")
        if os.path.exists(mods):
            print(f"  sentence-transformers modules: "
                  f"{[m.get('type', '').split('.')[-1] for m in json.load(open(mods))]}")
        print(f"  max_seq_length {getattr(model, 'max_seq_length', '?')}   "
              f"dim {model.get_sentence_embedding_dimension()}")

    tmp = a.frame_dir
    if not a.dry_run:
        os.makedirs(tmp, exist_ok=True)
    rng = random.Random(a.seed)
    score = {}
    for i, (uid, j) in enumerate(sorted(need.items())):
        if a.dry_run:
            v = np.array([rng.gauss(0, 1) for _ in range(64)])
            t = np.array([rng.gauss(0, 1) for _ in range(64)])
            score[uid] = float(v @ t / (np.linalg.norm(v)
                                        * np.linalg.norm(t)))
            continue
        paths = write_frames(video_of[j["recording_id"]],
                             sample_times(j["start"], j["end"], a.n_frames),
                             tmp, uid.replace("/", "_"))
        dur = max(float(j["end"]) - float(j["start"]), 1e-3)
        n = len(paths)
        # frames_indices is REQUIRED here and nothing fills it. With
        # do_sample_frames=True the processor sets it from its own sampling;
        # with a pre-sampled list nobody does, and _calculate_timestamps then
        # calls .tolist() on None. It also calls .tolist(), so this must be an
        # ARRAY -- a Python list fails the same way one line later.
        #
        # Indices 0..n-1 with fps=(n-1)/duration put the frames at 0..duration,
        # i.e. timestamps relative to the SEGMENT. Absolute times inside the
        # recording were the other option and would tell the model this clip
        # begins at 466.8s, which is not what a clip-trained model expects.
        meta = [{"fps": (n - 1) / dur if n > 1 else 1.0,
                 "total_num_frames": n, "duration": dur,
                 "frames_indices": np.arange(n)}]
        vv, tt = embed(model, [{"video": paths}], [j["stored_label"]],
                       a.prompt, None if a.no_metadata else meta)
        score[uid] = float(vv[0] @ tt[0] / (np.linalg.norm(vv[0])
                                            * np.linalg.norm(tt[0])))
        for q in paths:
            os.remove(q)
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
        # "> chance band" at a 0.001 margin reads as a result and is not
        # one. The margin is printed, and the marker needs 0.02 of daylight
        # before it appears at all.
        print(f"  {f:<12}{au:>8.3f}   [{lo:.3f}, {hi:.3f}]"
              f"   bar{au - bar:+.3f}"
              + ("   > chance band" if au > bar + 0.02 else
                 "   ON the bar" if au > bar else ""))
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
