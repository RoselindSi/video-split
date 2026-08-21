"""batch4's within-recording YES/NO pairs: the first readable natural semantic test.

The frozen 89-event gold could not answer whether a scorer judges
`claim_support` on real annotation, because 6 of its 765 YES/NO pairs sat inside
one recording and the rest compared one kitchen against another. batch4 changed
that: 31 of 61 recordings carry both classes, giving 361 within-recording
comparisons.

THE STATISTIC IS THE WITHIN-RECORDING PAIR, not an AUROC over everything. Inside
one recording the scene, the person, the camera and the session are shared, so a
scorer that recognises the kitchen gains nothing. The global number is printed
beside it -- labelled as the confounded one -- so the difference between the two
is visible rather than asserted.

ONE JOIN AMBIGUITY THAT IS REPORTED, NOT GUESSED. batch4's candidates are
DETECTOR predictions; `recseg` holds the ANNOTATION's segments. A candidate that
the human judged `no_boundary` may therefore fall inside a single annotated
segment, in which case the left and right segments the human was shown are not
two segments in recseg and cannot be recovered from it. Those candidates are
counted and excluded rather than resolved by picking whichever segment is
nearest, which would silently score a different window than the human saw.

`partial` and `uncertain` are carried through and excluded from the primary
comparison. The frozen protocol makes YES vs NO the endpoint and declines to
force `partial` into a binary; that decision is not revisited here.

Usage:
    python -m src.auditor.semantic.batch4_within_recording --emit \
        --audit data/gold/batch4_joint_audit.csv \
        --recseg '/workspace/tr1/data_recseg*/recseg_*.json' \
        --out /workspace/tr1/results/auditor/batch4_observations.jsonl

    python -m src.auditor.semantic.batch4_within_recording --score \
        --observations ... --model /workspace/tr1/ckpts/Qwen3-VL-Reranker-8B \
        --out /workspace/tr1/results/auditor/batch4_scores.jsonl

    python -m src.auditor.semantic.batch4_within_recording --evaluate \
        --observations ... --scores ...
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict

import numpy as np

from src.auditor.semantic.compose_supervision import resolve
from src.auditor.semantic.render_ontology_clips import get_segments, get_video

TOL = 1.0   # project tolerance from 2026-08-19; see memory tolerance-is-1s


def rid_full(x):
    return f"recording_{int(x):06d}"


def emit(a):
    rows = [r for r in csv.DictReader(open(a.audit, newline="",
                                           encoding="utf-8-sig"))
            if (r.get("candidate_key") or "").strip()]
    print(f"{len(rows)} candidates over "
          f"{len({r['recording_id'] for r in rows})} recordings")

    segs, vids = {}, {}
    for p in resolve(a.recseg):
        blob = json.load(open(p, encoding="utf-8"))
        if isinstance(blob, dict):
            blob = blob.get("recordings") or blob.get("data") or []
        for r in blob:
            k = r.get("recording_id")
            if not k or k in segs:
                continue
            segs[k] = sorted(([str(x[0]), float(x[1]), float(x[2])]
                              for x in get_segments(r)[0]), key=lambda x: x[1])
            vids[k] = get_video(r)
    print(f"  {len(segs)} recordings loaded from recseg")

    obs, skip = [], Counter()
    for r in rows:
        rid = rid_full(r["recording_id"])
        S = segs.get(rid)
        if not S:
            skip["recording not in recseg"] += 1
            continue
        t = float(r["candidate_time_s"])
        # A JUNCTION, NOT A NEAREST NEIGHBOUR. The left segment must END and the
        # right segment must START within tolerance of the candidate. A
        # candidate sitting inside one annotated segment has no such pair, and
        # picking the nearest segment on each side would hand the scorer a
        # window the human never saw.
        left = [s for s in S if abs(s[2] - t) <= a.tol_s]
        right = [s for s in S if abs(s[1] - t) <= a.tol_s]
        if not left or not right:
            skip["candidate is not at an annotated junction"] += 1
            continue
        L = min(left, key=lambda s: abs(s[2] - t))
        R = min(right, key=lambda s: abs(s[1] - t))
        if L is R or (L[1] == R[1] and L[2] == R[2]):
            skip["left and right resolve to one segment"] += 1
            continue
        for side, seg, col in (("L", L, "left_segment_naming_support"),
                               ("R", R, "right_segment_naming_support")):
            v = (r.get(col) or "").strip().lower()
            if not v:
                skip[f"{col} blank"] += 1
                continue
            obs.append({
                "obs_id": f"{r['candidate_key']}#{side}",
                "candidate_key": r["candidate_key"],
                "recording_id": rid, "video": vids.get(rid), "side": side,
                "start": seg[1], "end": seg[2], "label": seg[0],
                "support": v,
                "interaction_relation": r.get("interaction_relation"),
                "temporal_event_type": r.get("temporal_event_type")})

    print(f"\n  {len(obs)} (segment, label, verdict) observations")
    for k, v in skip.most_common():
        print(f"  skipped, {k}: {v}")
    for k, v in Counter(o["support"] for o in obs).most_common():
        print(f"    support={k:<12}{v:>4}")

    report_pairs(obs)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            for o in obs:
                f.write(json.dumps(o, ensure_ascii=False) + "\n")
        print(f"\nwrote {a.out}")
    return obs


def report_pairs(obs):
    per = defaultdict(lambda: Counter())
    for o in obs:
        per[o["recording_id"]][o["support"]] += 1
    both = [r for r, c in per.items() if c["yes"] and c["no"]]
    within = sum(per[r]["yes"] * per[r]["no"] for r in both)
    tot = (sum(c["yes"] for c in per.values())
           * sum(c["no"] for c in per.values()))
    print(f"\n  PAIR STRUCTURE")
    print(f"    {len(per)} recordings, {len(both)} carrying BOTH classes")
    print(f"    within-recording YES x NO pairs {within}")
    print(f"    all YES x NO pairs {tot}  -> within is "
          f"{within / tot:.1%}" if tot else "")
    print(f"    the frozen 89-event gold had 1 recording and 6 within pairs; "
          f"that arm\n    could not separate semantics from scene at all.")
    return both, within


def score(a):
    obs = [json.loads(l) for l in open(a.observations, encoding="utf-8")
           if l.strip()]
    print(f"{len(obs)} observations to score")

    # SHARDING IS BY OBSERVATION, which is sound only because each forward is
    # independent -- one clip against one label, no shared state. Sorting by
    # obs_id first makes the split deterministic, so a shard can be re-run and
    # produce the same subset.
    if a.shard:
        i, n = (int(x) for x in a.shard.split("/"))
        obs = sorted(obs, key=lambda o: o["obs_id"])[i::n]
        print(f"  shard {i}/{n}: {len(obs)} of them")

    # RESUME, because this runs for hours on a machine that may not be there
    # tomorrow. Scores already written are read back and skipped rather than
    # recomputed; the file is appended, not truncated.
    done = set()
    if a.resume and os.path.exists(a.out):
        done = {json.loads(l)["obs_id"] for l in open(a.out, encoding="utf-8")
                if l.strip()}
        obs = [o for o in obs if o["obs_id"] not in done]
        print(f"  resuming: {len(done)} already scored, {len(obs)} to go")
    if not obs:
        print("  nothing left to score")
        return
    if not a.model or not os.path.exists(a.model):
        raise SystemExit(f"--model {a.model} does not exist")
    import torch  # noqa: F401
    from sentence_transformers import CrossEncoder
    from transformers import AutoProcessor

    from src.auditor.semantic.cosine_baseline import sample_times, write_frames
    from src.auditor.semantic.reranker_baseline import score_batch

    model = CrossEncoder(a.model, trust_remote_code=True)
    proc = AutoProcessor.from_pretrained(a.model)
    os.makedirs(a.frame_dir, exist_ok=True)

    # Opened in append mode and flushed per score: a shard killed at 400/472
    # keeps its 400. Truncating here is what makes --resume a lie.
    fout = open(a.out, "a" if (a.resume and done) else "w", encoding="utf-8")
    out = []
    for i, o in enumerate(obs):
        if not o.get("video"):
            print(f"  !! {o['obs_id']} has no video; skipped")
            continue
        frames = write_frames(o["video"],
                              sample_times(o["start"], o["end"], a.n_frames),
                              a.frame_dir, o["obs_id"].replace("/", "_"))
        dur = max(float(o["end"]) - float(o["start"]), 1e-3)
        nf = len(frames)
        meta = {"fps": (nf - 1) / dur if nf > 1 else 1.0,
                "total_num_frames": nf, "duration": dur,
                "frames_indices": np.arange(nf)}
        s = score_batch(model, proc, "sentence_transformers", frames,
                        [o["label"]], a.total_pixels, None, meta)[0]
        rec = {"obs_id": o["obs_id"], "score": float(s)}
        out.append(rec)
        fout.write(json.dumps(rec) + "\n")
        fout.flush()
        for q in frames:
            os.remove(q)
        if (i + 1) % 25 == 0:
            print(f"    {i + 1}/{len(obs)} scored", flush=True)

    fout.close()
    print(f"\nwrote {len(out)} scores -> {a.out}"
          + (f"  (+{len(done)} kept from a previous run)" if done else ""))
    if a.shard:
        print(f"  this is shard {a.shard}; --evaluate needs every shard's "
              f"file concatenated")


def evaluate(a):
    obs = [json.loads(l) for l in open(a.observations, encoding="utf-8")
           if l.strip()]
    sc = {json.loads(l)["obs_id"]: json.loads(l)["score"]
          for l in open(a.scores, encoding="utf-8") if l.strip()}
    obs = [o for o in obs if o["obs_id"] in sc]
    print(f"{len(obs)} observations with a score")
    both, within = report_pairs(obs)

    by = defaultdict(lambda: {"yes": [], "no": []})
    for o in obs:
        if o["support"] in ("yes", "no"):
            by[o["recording_id"]][o["support"]].append(sc[o["obs_id"]])

    # THE PRIMARY STATISTIC. A pair is won when the YES observation scores
    # above the NO one; a tie is half, the same convention paired_null uses.
    pairs = []
    for r, d in by.items():
        for y in d["yes"]:
            for n in d["no"]:
                pairs.append((r, float(y > n) + 0.5 * float(y == n)))
    if not pairs:
        raise SystemExit("no within-recording YES/NO pair survived scoring")
    acc = float(np.mean([p[1] for p in pairs]))

    recs = sorted({r for r, _ in pairs})
    idx = {r: [w for rr, w in pairs if rr == r] for r in recs}
    rng = np.random.default_rng(a.seed)
    boot = []
    for _ in range(a.n_boot):
        take = rng.choice(len(recs), len(recs), replace=True)
        v = [w for i in take for w in idx[recs[i]]]
        if v:
            boot.append(float(np.mean(v)))
    lo, hi = np.percentile(boot, [2.5, 97.5])

    print(f"\n  WITHIN-RECORDING PAIRED ACCURACY   {acc:.3f}  "
          f"[{lo:.3f}, {hi:.3f}]")
    print(f"    {len(pairs)} pairs over {len(recs)} recordings; the interval "
          f"resamples recordings.\n    0.5 is chance. Scene, person, camera "
          f"and session are shared inside a pair,\n    so recognising the "
          f"recording earns nothing here.")

    # The confounded number, printed for contrast and labelled as such.
    ys = [sc[o["obs_id"]] for o in obs if o["support"] == "yes"]
    ns = [sc[o["obs_id"]] for o in obs if o["support"] == "no"]
    glob = float(np.mean([[float(y > n) + 0.5 * float(y == n) for n in ns]
                          for y in ys])) if ys and ns else float("nan")
    print(f"\n  global (all pairs, CONFOUNDED)     {glob:.3f}")
    print(f"    {len(ys)} x {len(ns)} pairs, {within / (len(ys) * len(ns)):.1%}"
          f" of them within a recording. This is the\n    number the old arm "
          f"reported and could not read; it is here for contrast only.")

    if a.out:
        json.dump({"within_accuracy": acc, "lo": float(lo), "hi": float(hi),
                   "n_pairs": len(pairs), "n_recordings": len(recs),
                   "global_confounded": glob},
                  open(a.out, "w", encoding="utf-8"), indent=2)
        print(f"\nwrote {a.out}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emit", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--audit", default="data/gold/batch4_joint_audit.csv")
    ap.add_argument("--recseg", action="append", default=[])
    ap.add_argument("--observations")
    ap.add_argument("--scores")
    ap.add_argument("--model")
    ap.add_argument("--n_frames", type=int, default=32)
    ap.add_argument("--shard", help="i/n -- score only observation i, i+n, ... "
                                    "so one process can own one GPU. Each "
                                    "shard writes its own --out.")
    ap.add_argument("--resume", action="store_true",
                    help="skip obs_ids already in --out and append. Without "
                         "it --out is truncated and hours are lost to a "
                         "restart.")
    ap.add_argument("--total_pixels", type=int, default=3584 * 28 * 28)
    ap.add_argument("--frame_dir", default="/tmp/batch4_frames")
    ap.add_argument("--tol_s", type=float, default=TOL,
                    help="how close a candidate must be to a segment edge to "
                         "count as sitting at that junction")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    a = ap.parse_args()

    if a.emit:
        if not a.recseg:
            raise SystemExit("--emit needs --recseg")
        emit(a)
    elif a.score:
        score(a)
    elif a.evaluate:
        evaluate(a)
    else:
        raise SystemExit("pick one of --emit / --score / --evaluate")


if __name__ == "__main__":
    main()
