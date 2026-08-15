"""One wrong-video null, evaluated identically for any scorer.

The cosine arm could permute videos a thousand times for free: both sides were
vectors, so a permutation was a matmul. A cross-encoder has no separable sides
-- the video and the text are consumed together -- so every permutation costs a
forward pass and a thousand of them is not affordable.

That difference is a trap. Comparing a cosine null estimated from 1000
permutations against a reranker null estimated from 4 would compare two
statistics, and the reranker would look noisier for a reason that has nothing
to do with the reranker. So BOTH arms are evaluated here, on the same
statistic:

    true    accuracy on the real pairing
    null    accuracy pooled over every wrong-video pairing, treating each
            (pair, pairing) as one observation
    excess  true - null, with an interval from a bootstrap that resamples
            RECORDINGS, because the 306 pairs come from 26 of them and pairs
            inside a recording share a scene, a labelling session and an
            annotator

Pooling over pairings rather than averaging per-pairing accuracies is what
makes 4 pairings usable: the estimate's precision then comes from the number
of PAIRS, which is the same for both arms, and the pairings only have to be
enough to average over which wrong video each pair received.

NO p-VALUE IS PRINTED FOR AN ARM WITH FEW PAIRINGS. A permutation p-value
cannot resolve below 1/(pairings+1), and quoting 0.001 for one arm and 0.2 for
the other would read as a difference in evidence rather than in how many
forward passes were affordable. The excess interval is the comparison.

Input is one jsonl with a `pairing` field, 0 for the true assignment:
    {"pairing": 0, "segment_uid": ..., "video_uid": ..., "text": ..., "score": ...}

Usage:
    python -m src.auditor.semantic.paired_null \
        --scores /workspace/tr1/results/auditor/reranker_paired_scores.jsonl \
        --benchmark data/gold/paired_semantic_benchmark.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

import numpy as np


def load_extended(path):
    """{(pairing, segment_uid, text): score}, and the video each pairing used."""
    sc, vid = {}, {}
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        p = int(r.get("pairing", 0))
        sc[(p, r["segment_uid"], r["text"])] = float(r["score"])
        if "video_uid" in r:
            vid[(p, r["segment_uid"])] = r["video_uid"]
    return sc, vid


def margins(bench, sc, pairing):
    """Signed margin per pair under one pairing; None where either side is absent."""
    out = []
    for p in bench:
        u = p["segment_uid"]
        a = sc.get((pairing, u, p["original"]))
        b = sc.get((pairing, u, p["counterfactual"]))
        out.append(None if a is None or b is None else a - b)
    return out


def wins(m):
    """A tie is half a win, not a loss.

    `m > 0` charges every exact tie to the counterfactual. That is invisible
    with cosine, where two float dot products essentially never collide, and it
    is not invisible with a reranker whose head emits coarsely quantised logits
    -- the smoke run returned 31 distinct values in 80 scores, all multiples of
    0.125. Scoring ties as losses would then deflate the cross-encoder against
    a baseline that cannot tie, and the architecture comparison would be partly
    a comparison of output precision. Half credit is the usual convention for
    paired comparisons and is what AUROC already does with ties."""
    return np.where(np.isnan(m), np.nan, (m > 0) + 0.5 * (m == 0))


def boot_excess(rec, kind_sel, t_m, n_m, n_boot, rng):
    """Resample RECORDINGS, not pairs.

    306 pairs come from 26 recordings and pairs inside one share a scene, a
    labelling session and an annotator, so an interval that resamples pairs
    treats 306 correlated observations as 306 independent ones and comes out
    too narrow. The same clustering the rest of this project uses."""
    recs = sorted(set(rec))
    idx = {r: np.flatnonzero((rec == r) & kind_sel) for r in recs}
    out = []
    for _ in range(n_boot):
        take = rng.choice(len(recs), len(recs), replace=True)
        sel = np.concatenate([idx[recs[i]] for i in take]) if len(recs) else []
        if len(sel) == 0:
            continue
        t = np.nanmean(wins(t_m[sel]))
        n = np.nanmean(wins(np.concatenate([n_m[j][sel]
                                            for j in range(len(n_m))])))
        out.append(t - n)
    return np.array(out)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--benchmark",
                    default="data/gold/paired_semantic_benchmark.jsonl")
    ap.add_argument("--label", default=None,
                    help="what produced these scores, printed on the table so "
                         "two arms cannot be mixed up in a notebook")
    ap.add_argument("--reference_kind", default="wrong_object",
                    help="the kind separation is expressed relative to. A "
                         "mean |margin| in raw units cannot be compared across "
                         "architectures -- a cosine and a logit are different "
                         "scales -- but its ratio to a kind the same model "
                         "handles well can be")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    a = ap.parse_args()

    bench = [json.loads(l) for l in open(a.benchmark, encoding="utf-8")
             if l.strip()]
    sc, vid = load_extended(a.scores)
    pairings = sorted({k[0] for k in sc})
    if 0 not in pairings:
        raise SystemExit("no pairing 0 in the score file; that is the real "
                         "assignment and without it there is nothing to "
                         "compare the null against.")
    nulls = [p for p in pairings if p != 0]
    print(f"{a.label or a.scores}")
    print(f"  {len(bench)} pairs; {len(nulls)} wrong-video pairings")
    if not nulls:
        raise SystemExit("no wrong-video pairings; the true accuracy alone "
                         "cannot be read -- that is the whole point of this "
                         "file.")

    # A PAIRING THAT LEFT A SEGMENT ON ITS OWN RECORDING IS NOT A NULL. The
    # scorer file carries which video each pairing used, so the constraint is
    # verified here rather than trusted from the module that built it.
    if vid:
        bad = Counter()
        rec_of_uid = {p["segment_uid"]: p["recording_id"] for p in bench}
        for (pp, u), v in vid.items():
            if pp == 0:
                if v != u:
                    bad["pairing 0 is not the true assignment"] += 1
            elif v not in rec_of_uid:
                bad["null used a video outside the benchmark"] += 1
            elif rec_of_uid[v] == rec_of_uid.get(u):
                bad["null kept its own recording"] += 1
        if bad:
            raise SystemExit(f"the pairings do not do what they claim: "
                             f"{dict(bad)}")

    t_m = np.array([np.nan if m is None else m
                    for m in margins(bench, sc, 0)], float)
    n_m = [np.array([np.nan if m is None else m
                     for m in margins(bench, sc, p)], float) for p in nulls]
    gone = int(np.isnan(t_m).sum())
    if gone:
        print(f"  !! {gone} pairs have no true score and are dropped from "
              f"every row")

    kinds = np.array([p["kind"] for p in bench], dtype=object)
    rec = np.array([p["recording_id"] for p in bench], dtype=object)
    order = sorted(set(kinds.tolist()), key=lambda k: -int((kinds == k).sum()))
    rng = np.random.default_rng(a.seed)

    # THE REFERENCE SCALE, computed before the loop so every row divides by
    # the same number. A tie rate cannot serve this purpose: ties are exact
    # equalities, so they are an artifact of how coarsely a model quantises
    # its output -- the reranker's drop_claim tie rate halved, 0.21 to 0.10,
    # on pairs that did not change, purely because adding two kinds altered
    # the batch composition and therefore the padding. A model with float
    # outputs would score ~0 ties without any additional competence.
    rsel = (kinds == a.reference_kind) & ~np.isnan(t_m)
    ref = float(np.nanmean(np.abs(t_m[rsel]))) if rsel.any() else 0.0
    if not ref:
        print(f"  !! reference kind {a.reference_kind!r} is absent; "
              f"separation cannot be computed")

    print(f"\n  {'kind':<17}{'n':>5}{'true':>8}{'null':>8}{'excess':>9}"
          f"{'excess 95%':>19}{'margin':>9}{'|margin|':>10}{'sep':>7}"
          f"{'ties':>7}")
    res = {}
    for k in order + ["ALL"]:
        sel = np.ones(len(bench), bool) if k == "ALL" else (kinds == k)
        ok = sel & ~np.isnan(t_m)
        t = float(np.mean(wins(t_m[ok])))
        pooled = np.concatenate([m[ok] for m in n_m])
        n = float(np.nanmean(wins(pooled)))
        tie = float(np.mean(t_m[ok] == 0))
        d = boot_excess(rec, ok, t_m, n_m, a.n_boot, rng)
        lo, hi = np.percentile(d, [2.5, 97.5])
        mm = float(np.nanmean(t_m[ok]))
        am = float(np.nanmean(np.abs(t_m[ok])))
        print(f"  {k:<17}{int(ok.sum()):>5}{t:>8.3f}{n:>8.3f}{t - n:>+9.3f}"
              f"{f'[{lo:+.3f}, {hi:+.3f}]':>19}{mm:>9.4f}{am:>10.4f}"
              f"{am / ref if ref else float('nan'):>7.2f}{tie:>7.2f}")
        res[k] = {"n": int(ok.sum()), "true": t, "null": n,
                  "excess": t - n, "lo": float(lo), "hi": float(hi),
                  "mean_margin": mm, "mean_abs_margin": am,
                  "separation": (am / ref if ref else None),
                  "tie_rate": tie}

    print(f"\n  null pools {len(nulls)} pairings, so its precision comes from "
          f"the number of PAIRS,\n  which is what makes a handful of forward-"
          f"pass pairings comparable to a thousand\n  free ones. The interval "
          f"resamples recordings.")
    print(f"  |margin| = how far apart the scorer put the two texts, sign ignored.\n"
          f"  sep = that, over the same model's {a.reference_kind}. Scale-free, "
          f"so it compares across\n  architectures where a raw margin and a "
          f"tie rate both cannot.\n"
          f"  ties = share of pairs the scorer could not separate at all, each counted "
          f"as half a win.\n  A high tie rate is its own finding: it is the "
          f"scorer declining to choose, not\n  choosing wrongly.\n"
          f"  An excess interval containing 0 means that kind's accuracy is "
          f"reachable without\n  the right video, whatever the accuracy "
          f"itself looks like.")

    if a.out:
        json.dump({"label": a.label, "scores": a.scores,
                   "n_pairings": len(nulls), "kinds": res},
                  open(a.out, "w", encoding="utf-8"), indent=2)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
