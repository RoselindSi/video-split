"""The null the paired benchmark still needs: does the original win WITHOUT the video?

The paired benchmark cancels recording identity -- both texts in a pair are
scored against the same segment, so the scene cannot separate them. That closed
the confound that made the video prior's 0.827 unreadable. It does NOT close a
second one.

A pair is decided by `v . (t_original - t_counterfactual)`. Nothing in that
expression requires `v` to be THIS segment's video. If original labels are
systematically better-formed English -- fluent, in-vocabulary, the shape a
caption usually takes -- their embeddings can sit closer to the video manifold
generally, and the original wins against ANY video. That is a text prior, and
it would print as a benchmark result.

SO SWAP THE VIDEO. Each pair is rescored against a segment from a DIFFERENT
recording, both texts against the same wrong video so the comparison stays
paired. Under this null the original is not a true description of what is on
screen, so a scorer that reads the video has no reason to prefer it and should
land at 0.5. Whatever accuracy survives is the part that never needed the
video.

WHY THIS IS THE RIGHT NULL AND `shuffle the labels` IS NOT. Permuting labels
across segments breaks the pairing and changes what is being asked. Permuting
the VIDEO holds the text pair, the kind, and the perturbation fixed, and moves
exactly one thing: whether the video is the right one.

READ IT AS EXCESS OVER ITS OWN NULL, per kind. The kinds have different nulls
because they perturb text differently -- a dropped clause changes length, a
one-word swap does not -- so a single global null would over-credit some kinds
and under-credit others.

    `reorder` is the one this was built for. It scored 0.857 with a margin of
    0.0196, a fifth of `wrong_object`'s margin at comparable accuracy: tiny
    differences that are nonetheless almost always signed the same way, which
    is what a systematic text-side preference looks like and is not what
    reading the video looks like.

THE VECTORS COME FROM THE SAME RUN THAT PRODUCED THE REPORTED TABLE. With
--scores the true accuracies are recomputed from the embeddings and checked
against the scored pairs, so a null computed off a different encoding pass
cannot be compared to a table it does not belong to.

Usage:
    python -m src.auditor.semantic.text_prior_null \
        --embeddings /workspace/tr1/results/auditor/paired_cosine_vecs.npz \
        --benchmark data/gold/paired_semantic_benchmark.jsonl \
        --scores /workspace/tr1/results/auditor/paired_cosine_scores_v2.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

import numpy as np


def permute_across_recordings(rec, rng, tries=400):
    """A permutation of the segments in which nobody keeps their own recording.

    Not merely their own SEGMENT: two segments from the same kitchen share the
    scene, and handing a pair the neighbouring segment of its own recording
    would leave most of the video content in place. The constraint is on the
    recording, which is the unit the confound lives at."""
    n = len(rec)
    for _ in range(tries):
        p = rng.permutation(n)
        for i in range(n):
            if rec[p[i]] != rec[i]:
                continue
            cand = [j for j in range(n)
                    if rec[p[j]] != rec[i] and rec[p[i]] != rec[j] and j != i]
            if cand:
                j = int(rng.choice(cand))
                p[i], p[j] = p[j], p[i]
        if all(rec[p[i]] != rec[i] for i in range(n)):
            return p
    raise SystemExit(
        "could not build a permutation that moves every segment to a "
        "different recording. With few recordings this can be impossible, "
        "and silently relaxing it would make the null weaker than its name.")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--benchmark",
                    default="data/gold/paired_semantic_benchmark.jsonl")
    ap.add_argument("--scores",
                    help="the scored pairs the reported table came from. The "
                         "true accuracies are recomputed from the vectors and "
                         "checked against it")
    ap.add_argument("--n_perm", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    a = ap.parse_args()

    z = np.load(a.embeddings, allow_pickle=True)
    V = z["V"]
    seg_uids = [str(x) for x in z["seg_uids"]]
    T = z["T"]
    tkey = {(str(s), str(t)): i
            for i, (s, t) in enumerate(zip(z["text_seg"], z["text_str"]))}
    row_of = {u: i for i, u in enumerate(seg_uids)}
    print(f"{len(seg_uids)} video vectors, {len(tkey)} text vectors, "
          f"dim {V.shape[1]}")

    bench = [json.loads(l) for l in open(a.benchmark, encoding="utf-8")
             if l.strip()]
    rec_of = {}
    pairs = []
    missing = Counter()
    for p in bench:
        u = p["segment_uid"]
        rec_of[u] = p["recording_id"]
        ko, kc = (u, p["original"]), (u, p["counterfactual"])
        if u not in row_of or ko not in tkey or kc not in tkey:
            missing[p["kind"]] += 1
            continue
        pairs.append((row_of[u], tkey[ko], tkey[kc], p["kind"]))
    if missing:
        # A pair whose text was never encoded is not a pair that scored 0.5;
        # it is a pair that is absent, and averaging over the survivors while
        # printing the benchmark's n would misstate both.
        print(f"  !! {sum(missing.values())} pairs have no vector: "
              f"{dict(missing)}")
    print(f"{len(pairs)} of {len(bench)} pairs scorable from these vectors")

    rec = np.array([rec_of[u] for u in seg_uids], dtype=object)
    n_rec = len(set(rec))
    print(f"  {n_rec} recordings over {len(seg_uids)} segments")

    si = np.array([p[0] for p in pairs])
    oi = np.array([p[1] for p in pairs])
    ci = np.array([p[2] for p in pairs])
    kinds = np.array([p[3] for p in pairs], dtype=object)
    diff = T[oi] - T[ci]                       # the whole pair, as one vector

    true_m = np.einsum("ij,ij->i", V[si], diff)
    if a.scores:
        sc = {}
        for l in open(a.scores, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                sc[(r["segment_uid"], r["text"])] = float(r["score"])
        chk, n_chk = 0.0, 0
        for (s, o, c, _k), m in zip(pairs, true_m):
            ko = (seg_uids[s], str(z["text_str"][o]))
            kc = (seg_uids[s], str(z["text_str"][c]))
            if ko in sc and kc in sc:
                chk = max(chk, abs((sc[ko] - sc[kc]) - m))
                n_chk += 1
        print(f"  checked {n_chk} pair margins against {a.scores}: "
              f"max disagreement {chk:.2e}")
        if chk > 1e-5:
            raise SystemExit(
                "the vectors do not reproduce the scored pairs, so a null "
                "built from them does not belong to that table.")

    rng = np.random.default_rng(a.seed)
    order = sorted(set(kinds.tolist()),
                   key=lambda k: -int((kinds == k).sum()))
    null = defaultdict(list)
    for _ in range(a.n_perm):
        p = permute_across_recordings(rec, rng)
        m = np.einsum("ij,ij->i", V[p[si]], diff)
        w = m > 0
        for k in order:
            null[k].append(float(w[kinds == k].mean()))
        null["ALL"].append(float(w.mean()))

    print(f"\n{a.n_perm} permutations; every segment scored against a video "
          f"from a different recording.\n")
    print(f"  {'kind':<17}{'n':>5}{'true':>8}{'null':>8}"
          f"{'null 95%':>17}{'excess':>9}{'p':>8}")
    for k in order + ["ALL"]:
        sel = np.ones(len(pairs), bool) if k == "ALL" else (kinds == k)
        t = float((true_m[sel] > 0).mean())
        d = np.array(null[k])
        lo, hi = np.percentile(d, [2.5, 97.5])
        pv = (1.0 + (d >= t).sum()) / (len(d) + 1.0)
        print(f"  {k:<17}{int(sel.sum()):>5}{t:>8.3f}{d.mean():>8.3f}"
              f"{f'[{lo:.3f}, {hi:.3f}]':>17}{t - d.mean():>+9.3f}{pv:>8.3f}")

    # WHAT THE PRIOR IS MADE OF, measured rather than explained. Averaged over
    # videos the null margin is `diff . mean(V)`: the text prior is a
    # projection onto ONE direction, the mean video vector, so it has a size
    # and a sign per pair and both can be regressed on something. Length is
    # the first candidate because it is the one thing the kinds differ in --
    # dropping a clause shortens the text, swapping a word does not -- and a
    # longer sentence has more content words to overlap with any scene at all.
    vbar = V.mean(0)
    vbar = vbar / (np.linalg.norm(vbar) + 1e-12)
    proj = diff @ vbar
    ntok = {}
    for p in bench:
        for t in (p["original"], p["counterfactual"]):
            ntok[t] = len(str(t).split())
    dlen = np.array([ntok[str(z["text_str"][o])] - ntok[str(z["text_str"][c])]
                     for _s, o, c, _k in pairs], float)

    def corr(x, y):
        if x.std() < 1e-12 or y.std() < 1e-12:
            return float("nan")
        return float(np.corrcoef(x, y)[0, 1])

    print(f"\n  the prior is one direction -- `diff . mean(video)`. Its size "
          f"and sign per kind,\n  against the one thing the kinds differ in:\n")
    print(f"  {'kind':<17}{'mean proj':>11}{'mean len':>10}{'corr':>8}")
    for k in order + ["ALL"]:
        sel = np.ones(len(pairs), bool) if k == "ALL" else (kinds == k)
        print(f"  {k:<17}{proj[sel].mean():>+11.4f}{dlen[sel].mean():>+10.2f}"
              f"{corr(proj[sel], dlen[sel]):>8.3f}")
    print(f"\n  mean len = words in the original minus words in the "
          f"counterfactual.\n  THE `ALL` CORRELATION IS COMPOSITION, NOT A "
          f"RELATION. Four kinds swap a word or\n  reorder clauses and have "
          f"exactly zero length variance, so pooling them with the one\n  "
          f"kind that has any measures which kind a pair belongs to. Read the "
          f"per-kind rows.")

    # THE PAIR-LEVEL CORRELATION IS UNDERPOWERED AND THE POOLED ONE IS
    # CONFOUNDED, so neither settles length. Test it where the power is: each
    # of the 409 TEXTS has a projection onto the same direction and a word
    # count, and the labels vary in length on their own. n goes from 29 to
    # 409 and the length variance is the corpus's rather than one
    # perturbation's.
    tp = T @ vbar
    tl = np.array([ntok[str(s)] for s in z["text_str"]], float)
    r = corr(tl, tp)
    slope = float(np.polyfit(tl, tp, 1)[0]) if tl.std() > 1e-12 else float("nan")
    print(f"\n  length, tested at n={len(tp)} texts instead of n=29 pairs:")
    print(f"    words {tl.min():.0f}-{tl.max():.0f}, mean {tl.mean():.1f}")
    print(f"    corr(words, projection) = {r:+.3f}   slope = "
          f"{slope:+.5f} per word")
    dc = [k for k in order if k == "drop_claim"]
    if dc and not np.isnan(slope):
        sel = kinds == "drop_claim"
        pred = slope * dlen[sel].mean()
        obs = proj[sel].mean()
        # A PREDICTION, not a restatement. If length is the mechanism, the
        # slope fitted on all 409 texts times drop_claim's 2.9-word gap has to
        # land on drop_claim's observed projection. A correlation can be
        # positive and still predict a tenth of the effect.
        print(f"\n    length predicts drop_claim's prior as {pred:+.4f}; "
              f"observed {obs:+.4f}"
              f"  ({100 * pred / obs if obs else float('nan'):.0f}% of it)")
        print(f"    the rest, if any, is something other than word count.")

    # WHAT THE BIAS IS WORTH, on the real scores rather than on the projection
    # onto the mean direction. The projection is one component of a specific
    # video's cosine and the rest of that cosine could in principle cancel it,
    # so the deployment quantity is fitted on `V[seg] . T` itself -- the number
    # an auditor would actually threshold.
    #
    # AND IT IS PUT IN THE SAME UNITS AS THE ERRORS. A slope in cosine per word
    # is unreadable on its own; a slope divided by a kind's mean margin says
    # how many words of extra length cancel that kind of error, which is a
    # quantity a threshold has to survive.
    seg_row = np.array([row_of[str(s)] for s in z["text_seg"]])
    s_true = np.einsum("ij,ij->i", V[seg_row], T)
    r2 = corr(tl, s_true)
    sl2 = float(np.polyfit(tl, s_true, 1)[0]) if tl.std() > 1e-12 else float("nan")
    print(f"\n  the same slope on the REAL scores, which is what a threshold "
          f"sees:")
    print(f"    corr(words, score) = {r2:+.3f}   slope = {sl2:+.5f} cosine "
          f"per word\n")
    print(f"  {'kind':<17}{'mean margin':>13}{'= words of length':>19}")
    for k in order:
        sel = kinds == k
        m = float(true_m[sel].mean())
        print(f"  {k:<17}{m:>13.4f}{(m / sl2 if sl2 else float('nan')):>19.1f}")
    print(f"\n  A label that many words longer scores as well as a correct "
          f"label of that kind.\n  Where the figure is near 1, one extra word "
          f"buys back the whole error, and a\n  scorer thresholded on cosine "
          f"ranks by length before it ranks by truth.")

    print(f"\n  null   = the same pair, the same perturbation, a video from "
          f"another recording.\n"
          f"  excess = how much of the accuracy needed the right video.\n"
          f"  A kind whose true accuracy sits inside its own null interval is "
          f"measuring the text,\n  not the video, and its benchmark number "
          f"cannot be read as a semantic result.")

    if a.out:
        json.dump({"n_perm": a.n_perm, "n_pairs": len(pairs),
                   "n_recordings": n_rec,
                   "kinds": {k: {
                       "n": int((np.ones(len(pairs), bool) if k == "ALL"
                                 else kinds == k).sum()),
                       "true": float((true_m[(np.ones(len(pairs), bool)
                                              if k == "ALL"
                                              else kinds == k)] > 0).mean()),
                       "null_mean": float(np.mean(null[k])),
                       "null_lo": float(np.percentile(null[k], 2.5)),
                       "null_hi": float(np.percentile(null[k], 97.5)),
                   } for k in order + ["ALL"]}},
                  open(a.out, "w", encoding="utf-8"), indent=2)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
