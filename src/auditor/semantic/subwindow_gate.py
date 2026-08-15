"""G0: is the verb evidence localised? Run this before building anything.

The frame sweep killed resolution as an explanation -- 8, 16 and 32 frames over
the whole segment gave verb accuracy 0.690 / 0.707 / 0.701 and separation 0.42 /
0.44 / 0.46. What is left of the hypothesis is SELECTION: a verb happens in a
short stretch, a global pool averages it against the rest of the window, and
sampling the same window more finely does not undo an average. An object
survives pooling because its appearance is in most frames, which is why
`wrong_object` is the strongest axis and `wrong_verb` the weakest in both
architectures.

That is checkable with the existing reranker and no training. Score each pair on
each of K sub-windows and let the sub-window with the largest |margin| decide.
If evidence is localised the best sub-window beats the whole window; if it is
not, no selector built later can find anything, and the model in
`atomic_verifier_design.md` should not be built.

FRAMES ARE HELD EQUAL, which is the whole reason this is a fair test. K
sub-windows of n frames each see K*n frames in total, so comparing them against
an 8-frame whole window would compare localisation against frame count -- the
variable the sweep already settled. The whole-window arm here is scored at K*n
frames, in the same run and the same batch composition, so the only thing that
differs is whether those frames are pooled together or apart.

`wrong_object` IS THE CONTROL, NOT A BONUS. Localisation that helps both axes
equally is not a verb finding; it is a general effect of scoring shorter
windows, and it would not justify a claim-conditioned selector. The gate passes
on the DIFFERENCE between the two axes.

SELECTION DOES NOT BIAS THE SIGN. Taking the largest |margin| over K windows
picks a magnitude, not a direction, so pure noise still lands at 0.5. What it
can amplify is a preference that is consistent across windows -- a text prior --
and that is what the wrong-video null is for; it runs under the identical rule.

Usage:
    python -m src.auditor.semantic.subwindow_gate \
        --model /workspace/tr1/ckpts/Qwen3-VL-Reranker-2B \
        --benchmark data/gold/paired_semantic_benchmark.jsonl \
        --data /workspace/tr1/results/auditor/naming_run.json \
        --out /workspace/tr1/results/auditor/subwindow_gate.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np

from src.auditor.semantic.cosine_baseline import sample_times, write_frames
from src.auditor.semantic.paired_null import boot_excess, wins
from src.auditor.semantic.reranker_baseline import score_batch
from src.auditor.semantic.text_prior_null import permute_across_recordings


def windows(start, end, k):
    """(-1, whole) plus k contiguous sub-windows. -1 carries k*n frames."""
    edges = np.linspace(float(start), float(end), k + 1)
    return [(-1, float(start), float(end))] + \
           [(i, float(edges[i]), float(edges[i + 1])) for i in range(k)]


def evaluate(rows, bench, k, n_boot, seed, ref_kind):
    """true / null / excess / |margin| for three decision rules."""
    sc = defaultdict(dict)
    for r in rows:
        sc[(r["pairing"], r["segment_uid"])][(r["window"], r["text"])] = \
            r["score"]
    kinds = np.array([p["kind"] for p in bench], dtype=object)
    rec = np.array([p["recording_id"] for p in bench], dtype=object)
    pairings = sorted({r["pairing"] for r in rows})
    rng = np.random.default_rng(seed)

    def margin(p, pairing, rule):
        d = sc.get((pairing, p["segment_uid"]))
        if not d:
            return np.nan
        got = []
        for w in ([-1] if rule == "whole" else range(k)):
            a = d.get((w, p["original"]))
            b = d.get((w, p["counterfactual"]))
            if a is None or b is None:
                return np.nan
            got.append(a - b)
        if rule == "whole":
            return got[0]
        if rule == "mean":
            return float(np.mean(got))
        return got[int(np.argmax(np.abs(got)))]   # max_abs

    print(f"\n  {'rule':<10}{'kind':<16}{'n':>5}{'true':>8}{'null':>8}"
          f"{'excess':>9}{'excess 95%':>19}{'|margin|':>10}{'sep':>7}")
    out = {}
    for rule in ("whole", "max_abs", "mean"):
        t_m = np.array([margin(p, 0, rule) for p in bench], float)
        n_m = [np.array([margin(p, j, rule) for p in bench], float)
               for j in pairings if j != 0]
        # THE REFERENCE IS FIXED, NOT THE FIRST ROW. sep only means anything
        # against the same denominator the frozen table used, and taking
        # whichever kind happened to sort first made wrong_verb its own
        # reference, which is 1.00 by definition.
        rsel = (kinds == ref_kind) & ~np.isnan(t_m)
        ref = float(np.nanmean(np.abs(t_m[rsel]))) if rsel.any() else None
        for k_ in sorted(set(kinds.tolist()),
                         key=lambda x: -int((kinds == x).sum())):
            ok = (kinds == k_) & ~np.isnan(t_m)
            if not ok.any():
                continue
            t = float(np.mean(wins(t_m[ok])))
            n = float(np.nanmean(wins(np.concatenate([m[ok] for m in n_m]))))
            am = float(np.nanmean(np.abs(t_m[ok])))
            d = boot_excess(rec, ok, t_m, n_m, n_boot, rng)
            lo, hi = np.percentile(d, [2.5, 97.5])
            print(f"  {rule:<10}{k_:<16}{int(ok.sum()):>5}{t:>8.3f}{n:>8.3f}"
                  f"{t - n:>+9.3f}{f'[{lo:+.3f}, {hi:+.3f}]':>19}"
                  f"{am:>10.4f}{am / ref if ref else float('nan'):>7.2f}")
            out[(rule, k_)] = {"n": int(ok.sum()), "true": t, "null": n,
                               "excess": t - n, "lo": float(lo),
                               "hi": float(hi), "abs_margin": am}
        print()
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model")
    ap.add_argument("--benchmark",
                    default="data/gold/paired_semantic_benchmark.jsonl")
    ap.add_argument("--data")
    ap.add_argument("--kinds", default="wrong_verb,wrong_object")
    ap.add_argument("--n_windows", type=int, default=3)
    ap.add_argument("--n_frames", type=int, default=8,
                    help="frames PER SUB-WINDOW. The whole-window arm gets "
                         "n_windows times this many, so the two see the same "
                         "frames and differ only in whether they are pooled "
                         "together")
    ap.add_argument("--n_pairings", type=int, default=2)
    ap.add_argument("--total_pixels", type=int, default=3584 * 28 * 28)
    ap.add_argument("--frame_dir", default="/tmp/subwindow_frames")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--reference_kind", default="wrong_object")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--evaluate", help="skip scoring, read this score file")
    ap.add_argument("--out")
    a = ap.parse_args()

    bench = [json.loads(l) for l in open(a.benchmark, encoding="utf-8")
             if l.strip()]
    want = {x.strip() for x in a.kinds.split(",") if x.strip()}
    missing = want - {r["kind"] for r in bench}
    if missing:
        raise SystemExit(f"--kinds names {sorted(missing)}, absent from this "
                         f"benchmark")
    bench = [r for r in bench if r["kind"] in want]

    if a.evaluate:
        rows = [json.loads(l) for l in open(a.evaluate, encoding="utf-8")
                if l.strip()]
        evaluate(rows, bench, a.n_windows, a.n_boot, a.seed,
                 a.reference_kind)
        return

    segs, texts = {}, defaultdict(set)
    for p in bench:
        segs[p["segment_uid"]] = p
        texts[p["segment_uid"]].add(p["original"])
        texts[p["segment_uid"]].add(p["counterfactual"])
    uids = sorted(segs)
    n_entry = sum(len(texts[u]) for u in uids)
    print(f"{len(bench)} pairs over {len(uids)} segments; {n_entry} entries")
    print(f"  {a.n_windows} sub-windows of {a.n_frames} frames, plus a whole "
          f"window of {a.n_windows * a.n_frames}")
    print(f"  {n_entry * (a.n_windows + 1) * (a.n_pairings + 1)} scorings")

    rec = np.array([segs[u]["recording_id"] for u in uids], dtype=object)
    rng = np.random.default_rng(a.seed)
    perms = [np.arange(len(uids))]
    for _ in range(a.n_pairings):
        perms.append(permute_across_recordings(rec, rng))

    model = proc = None
    if not a.dry_run:
        import torch
        from sentence_transformers import CrossEncoder
        from transformers import AutoProcessor
        model = CrossEncoder(a.model, trust_remote_code=True)
        proc = AutoProcessor.from_pretrained(a.model)
        os.makedirs(a.frame_dir, exist_ok=True)
    video_of = ({r["recording_id"]: r["video"]
                 for r in json.load(open(a.data, encoding="utf-8"))}
                if a.data else {})
    drng = np.random.default_rng(a.seed + 1)

    rows, done = [], 0
    for j, perm in enumerate(perms):
        for i, u in enumerate(uids):
            vu = uids[perm[i]]
            vp = segs[vu]
            tl = sorted(texts[u])
            for w, ws, we in windows(vp["start"], vp["end"], a.n_windows):
                nf = a.n_frames * (a.n_windows if w == -1 else 1)
                if a.dry_run:
                    sc = drng.normal(size=len(tl)).tolist()
                else:
                    path = vp.get("video") or video_of.get(vp["recording_id"])
                    if not path:
                        print(f"  !! no video for {vu}")
                        continue
                    frames = write_frames(
                        path, sample_times(ws, we, nf), a.frame_dir,
                        f"{j}_{w}_{vu}".replace("/", "_"))
                    dur = max(we - ws, 1e-3)
                    meta = {"fps": (nf - 1) / dur if nf > 1 else 1.0,
                            "total_num_frames": nf, "duration": dur,
                            "frames_indices": np.arange(nf)}
                    sc = []
                    for b in range(0, len(tl), a.batch):
                        sc += list(score_batch(
                            model, proc, "sentence_transformers", frames,
                            tl[b:b + a.batch], a.total_pixels, None, meta))
                    for q in frames:
                        os.remove(q)
                for t, s in zip(tl, sc):
                    rows.append({"pairing": j, "segment_uid": u,
                                 "video_uid": vu, "window": w,
                                 "start": ws, "end": we, "text": t,
                                 "score": float(s)})
            done += 1
            if done % 50 == 0:
                print(f"    {done}/{len(uids) * len(perms)} "
                      f"(segment, pairing) scored", flush=True)

    out = a.out or "subwindow_gate.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nwrote {len(rows)} scores -> {out}"
          + ("   (DRY RUN, random)" if a.dry_run else ""))

    evaluate(rows, bench, a.n_windows, a.n_boot, a.seed,
             a.reference_kind)
    print(f"  THE GATE. Compare `max_abs` against `whole` on wrong_verb, and "
          f"then compare that\n  difference against the same difference on "
          f"wrong_object. A gain on both axes is\n  shorter windows helping "
          f"in general, not verb evidence being localised, and does\n  not "
          f"justify a claim-conditioned selector.")
    print(f"  Whole-window wrong_verb at 32 frames was true 0.701, sep 0.46. "
          f"`max_abs` has to\n  beat the `whole` row measured HERE, not that "
          f"one -- same run, same batching.")
    print(f"  |margin| IS NOT COMPARABLE ACROSS RULES: taking the largest of "
          f"{a.n_windows} inflates it\n  mechanically, and averaging shrinks "
          f"it, whatever the scores are. Only `sep` -- a\n  ratio inside one "
          f"rule -- and `true` carry across the three.")


if __name__ == "__main__":
    main()
