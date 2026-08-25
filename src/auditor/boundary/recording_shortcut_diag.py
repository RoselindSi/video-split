"""Why the ontology term ranks recordings but not candidates inside them.

C0 came back split: pooled AUROC 0.676 -> 0.723 while same-recording pair
accuracy went 0.540 -> 0.510, with the morphology term alone at 0.4992 --
chance. This module does not propose a fix. It measures where the morphology
score's variation actually lives, because the two live explanations imply
opposite next steps and the fused table cannot tell them apart.

    IF the term is ~0.5 in EVERY recording, there is no candidate-local
    signal to recover, and no per-recording normalisation can create one.

    IF it is 0.8 in some recordings and 0.2 in others, the head did learn
    something local and it flips sign across recordings, which is an
    unstable-generalisation problem rather than an absent-signal one.

A NOTE THAT KILLS AN ATTRACTIVE STEP BEFORE IT COSTS ANYTHING. Subtracting a
recording's median, z-scoring within a recording, and replacing scores by
within-recording rank percentile are all MONOTONE WITHIN A RECORDING. They
therefore leave every within-recording comparison exactly as it was, and
`transform_invariance` prints the three side by side to show it rather than
asserting it. Recording-relative scoring is a fix for POOLED metrics and for
what happens when the term is ADDED to another score. It cannot raise 0.4992,
because 0.4992 already holds recording fixed.

WHY THE PAIR COUNTS DO NOT MEAN WHAT THEY LOOK LIKE. 81416 pairs come from 36
recordings, and pairs sharing a recording share whatever that recording does.
Treating them as independent gives an interval about ten times too narrow, so
every accuracy here carries a CLUSTER BOOTSTRAP over recordings -- resample
the 36, not the 81416.

Usage:
    python -m src.auditor.boundary.recording_shortcut_diag \
        --candidates results/auditor/oof_candidates.jsonl \
        --morphology results/auditor/morphology_external_oof.jsonl \
        --oracle_audit results/boundary/oof/audit/predictions.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

from src.auditor.boundary.ontology_fusion import (EPS, ontology_energy,
                                                  morphology_logratio)

BOOT = 2000
SEED = 0


# --- A. where does the variance live -------------------------------------
def variance_decomposition(x, rec):
    """One-way random effects: how much of the term is the RECORDING.

    ICC is the share of total variance explained by which recording a
    candidate came from. A score built to discriminate candidates should put
    its variance WITHIN recordings; a high ICC means most of what the score
    says is a property of the video, and a pooled metric will read that as
    discrimination whenever recordings differ in base rate."""
    x = np.asarray(x, float)
    by = defaultdict(list)
    for i, r in enumerate(rec):
        by[r].append(x[i])
    groups = [np.array(v) for v in by.values() if len(v) > 1]
    k, N = len(groups), sum(len(g) for g in groups)
    grand = np.concatenate(groups).mean()
    ssb = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ssw = sum(((g - g.mean()) ** 2).sum() for g in groups)
    msb, msw = ssb / (k - 1), ssw / (N - k)
    n_i = np.array([len(g) for g in groups], float)
    k0 = (N - (n_i ** 2).sum() / N) / (k - 1)
    var_b = max((msb - msw) / k0, 0.0)
    icc = var_b / (var_b + msw) if (var_b + msw) > 0 else float("nan")
    return {"icc": float(icc), "between_var": float(var_b),
            "within_var": float(msw), "n_recordings": k, "n": N,
            "between_share": float(var_b / (var_b + msw))
            if (var_b + msw) > 0 else float("nan")}


# --- B. pair accuracy, per recording -------------------------------------
def pair_counts(sc, ok, fm, rec):
    """Per recording: (weighted wins, pairs). Ties count a half.

    Kept as counts rather than a ratio so the cluster bootstrap can resample
    recordings and re-aggregate without recomputing 81416 comparisons."""
    by = defaultdict(lambda: {"p": [], "n": []})
    for i, r in enumerate(rec):
        if ok[i]:
            by[r]["p"].append(sc[i])
        elif fm[i]:
            by[r]["n"].append(sc[i])
    out = {}
    for r, v in by.items():
        if not v["p"] or not v["n"]:
            continue
        p, n = np.array(v["p"]), np.array(v["n"])
        d = p[:, None] - n[None, :]
        out[r] = (float((d > 0).sum() + 0.5 * (d == 0).sum()), int(d.size),
                  len(p), len(n))
    return out


def cluster_ci(counts, boot=BOOT, seed=SEED):
    """Resample the RECORDINGS. Pairs inside one are not independent."""
    keys = list(counts)
    rng = np.random.default_rng(seed)
    w = np.array([counts[k][0] for k in keys])
    n = np.array([counts[k][1] for k in keys], float)
    acc = w.sum() / n.sum()
    idx = rng.integers(0, len(keys), size=(boot, len(keys)))
    draws = w[idx].sum(1) / np.maximum(n[idx].sum(1), 1)
    return float(acc), float(np.quantile(draws, 0.025)), \
        float(np.quantile(draws, 0.975))


def per_recording_table(cols, ok, fm, rec, top=40):
    counts = {name: pair_counts(np.asarray(s, float), ok, fm, rec)
              for name, s in cols}
    keys = sorted(counts[cols[0][0]],
                  key=lambda r: -counts[cols[0][0]][r][1])
    print("=" * 78)
    print("PAIR ACCURACY, ONE RECORDING AT A TIME")
    print("=" * 78)
    head = "".join(f"{c:>12}" for c, _ in cols)
    print(f"  {'recording':<22}{'true':>6}{'f_mid':>7}{'pairs':>8}{head}")
    rows = []
    for r in keys[:top]:
        _, npair, ntrue, nfm = counts[cols[0][0]][r]
        vals = [counts[c][r][0] / counts[c][r][1] for c, _ in cols]
        rows.append({"recording_id": r, "n_true": ntrue, "n_false_mid": nfm,
                     "n_pairs": npair,
                     **{c: v for (c, _), v in zip(cols, vals)}})
        body = "".join(f"{v:>12.3f}" for v in vals)
        print(f"  {str(r)[:22]:<22}{ntrue:>6}{nfm:>7}{npair:>8}{body}")
    if len(keys) > top:
        print(f"  ... {len(keys) - top} more recordings not shown")

    print(f"\n  {'':<22}{'':>21}{head}")
    for stat, fn in (("median", np.median),
                     ("25th pct", lambda v: np.quantile(v, 0.25)),
                     ("75th pct", lambda v: np.quantile(v, 0.75))):
        vals = [fn([counts[c][r][0] / counts[c][r][1] for r in keys])
                for c, _ in cols]
        print(f"  {stat:<22}{'':>21}" + "".join(f"{v:>12.3f}" for v in vals))
    for tag, test in ((">= 0.60", lambda v: v >= 0.60),
                      ("<= 0.40", lambda v: v <= 0.40)):
        vals = [sum(test(counts[c][r][0] / counts[c][r][1]) for r in keys)
                for c, _ in cols]
        print(f"  {'recordings ' + tag:<22}{'':>21}"
              + "".join(f"{v:>12d}" for v in vals))

    print(f"\n  pooled, with a cluster bootstrap over the "
          f"{len(keys)} recordings:")
    ci = {}
    for c, _ in cols:
        a, lo, hi = cluster_ci(counts[c])
        ci[c] = {"acc": a, "lo": lo, "hi": hi}
        flag = "" if lo > 0.5 else "   <- 0.5 inside the interval"
        print(f"    {c:<14}{a:>8.4f}   95% CI [{lo:.4f}, {hi:.4f}]{flag}")
    print(f"\n  A COLUMN SPLIT ACROSS THE RECORDINGS -- some near 0.8, some\n"
          f"  near 0.2 -- means local signal exists and its SIGN is unstable.\n"
          f"  A column piled on 0.5 in every recording means there is no\n"
          f"  candidate-local signal, and no per-recording normalisation can\n"
          f"  make one, because normalisation preserves within-recording order.")
    return rows, ci


# --- the invariance, shown rather than argued ----------------------------
def transform_invariance(m, ok, fm, rec):
    m = np.asarray(m, float)
    by = defaultdict(list)
    for i, r in enumerate(rec):
        by[r].append(i)
    cent, z, pct = m.copy(), m.copy(), m.copy()
    for idx in by.values():
        j = np.array(idx)
        v = m[j]
        cent[j] = v - np.median(v)
        z[j] = (v - v.mean()) / (v.std() + 1e-12)
        pct[j] = v.argsort().argsort() / max(len(v) - 1, 1)
    print("=" * 78)
    print("RECORDING-RELATIVE TRANSFORMS, ON THE WITHIN-RECORDING METRIC")
    print("=" * 78)
    out = {}
    for name, v in (("raw", m), ("centered", cent), ("z-scored", z),
                    ("rank pctile", pct)):
        c = pair_counts(v, ok, fm, rec)
        acc = sum(x[0] for x in c.values()) / sum(x[1] for x in c.values())
        out[name] = float(acc)
        print(f"  {name:<16}{acc:>10.4f}")
    print(f"\n  These are the same number, and that is the point. Every one of\n"
          f"  these transforms is monotone WITHIN a recording, so none of them\n"
          f"  can reorder a pair that is already inside one recording.\n"
          f"  Recording-relative scoring is a fix for pooled metrics and for\n"
          f"  the scale a term carries into a SUM -- it is not a way to\n"
          f"  recover local signal that the representation does not have.")
    return out


# --- C. what explains the recording-level offset --------------------------
def _spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra = a.argsort().argsort().astype(float)
    rb = b.argsort().argsort().astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d else float("nan")


def nuisance_correlations(cands, m, rec, extra_cov=None):
    """Is the recording-level offset explained by something about the video?

    The one that matters most is `true boundary density`. If a recording's
    mean ontology term tracks how many real boundaries that recording
    contains, then pooling the recordings hands the score a base-rate signal,
    and pooled AUROC rises without any candidate ever being discriminated
    from its neighbour."""
    by = defaultdict(list)
    for i, r in enumerate(rec):
        by[r].append(i)
    rows = {}
    for r, idx in by.items():
        j = np.array(idx)
        t = np.array([cands[i]["candidate_time"] for i in j], float)
        span = float(t.max() - t.min()) if len(t) > 1 else 0.0
        rows[r] = {
            "mean_ontology_term": float(np.mean(m[j])),
            "n_candidates": float(len(j)),
            "span_s": span,
            "candidate_density_per_min": len(j) / (span / 60) if span else 0.0,
            "mean_detector_score": float(np.mean(
                [cands[i]["detector_score"] for i in j])),
            "true_boundary_density": float(np.mean(
                [cands[i]["is_true_boundary"] for i in j])),
        }
        if extra_cov and r in extra_cov:
            for k, v in extra_cov[r].items():
                try:
                    rows[r][k] = float(v)
                except (TypeError, ValueError):
                    pass

    keys = sorted(rows)
    y = np.array([rows[r]["mean_ontology_term"] for r in keys])
    names = [k for k in rows[keys[0]] if k != "mean_ontology_term"]
    print("=" * 78)
    print("WHAT THE RECORDING-LEVEL OFFSET TRACKS")
    print("=" * 78)
    print(f"  Spearman against each recording's MEAN ontology term, "
          f"n = {len(keys)} recordings\n")
    out = {}
    for n in names:
        x = np.array([rows[r].get(n, np.nan) for r in keys])
        if np.isnan(x).any() or np.allclose(x, x[0]):
            continue
        out[n] = _spearman(x, y)
        mark = "   <-" if abs(out[n]) >= 0.4 else ""
        print(f"  {n:<32}{out[n]:>+8.3f}{mark}")
    print(f"\n  `true_boundary_density` is the one to read first. A strong\n"
          f"  correlation there IS the mechanism: pooled metrics reward a\n"
          f"  score for knowing which RECORDINGS are boundary-rich, which is\n"
          f"  not a thing a person reviewing one candidate can use.")
    return {"per_recording": rows, "spearman": out}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--morphology", required=True)
    ap.add_argument("--oracle_audit", required=True)
    ap.add_argument("--recording_covariates",
                    help="optional JSON {recording_id: {name: value}} of "
                         "nuisance covariates measured elsewhere -- coverage, "
                         "feature norms, anything recording-level.")
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--out")
    a = ap.parse_args()

    from src.auditor.boundary.ontology_constitution import Constitution
    C = Constitution()
    C.check_oracle_use("headroom")

    cands = [json.loads(l) for l in open(a.candidates, encoding="utf-8")
             if l.strip()]
    C.check_candidate_pool(cands, a.candidates)
    morph = {}
    for l in open(a.morphology, encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            morph[r["candidate_id"]] = r
    if any(c["candidate_id"] not in morph for c in cands):
        raise SystemExit("morphology does not cover the frozen pool")

    rec = [c["recording_id"] for c in cands]
    det = np.array([c["detector_score"] for c in cands], float)
    pp = np.array([morph[c["candidate_id"]]["p_point"] for c in cands])
    pn = np.array([morph[c["candidate_id"]]["p_no_transition"] for c in cands])
    m = morphology_logratio(pp, pn)
    logit = np.log(det.clip(1e-4, 1 - 1e-4) / (1 - det.clip(1e-4, 1 - 1e-4)))
    energy = ontology_energy(det, pp, pn)
    ok = np.array([c["is_true_boundary"] for c in cands], bool)

    bad = set()
    for l in open(a.oracle_audit, encoding="utf-8"):
        if not l.strip():
            continue
        e = json.loads(l)
        for x in e.get("predicted_peaks", []):
            if x.get("status") == "false_mid_segment":
                bad.add((e["recording_id"], round(x["pred_time"], 1)))
    fm = np.array([(c["recording_id"], round(c["candidate_time"], 1)) in bad
                   for c in cands])
    print(f"{len(cands)} candidates, {len({*rec})} recordings, "
          f"{int(ok.sum())} true, {int(fm.sum())} audited false_mid\n")

    print("=" * 78)
    print("A. WHERE EACH SCORE'S VARIANCE LIVES")
    print("=" * 78)
    print(f"  {'':<22}{'ICC':>10}{'between var':>14}{'within var':>13}")
    vd = {}
    for name, v in (("detector logit", logit), ("ontology term", m),
                    ("E_onto", energy)):
        d = variance_decomposition(v, rec)
        vd[name] = d
        print(f"  {name:<22}{d['icc']:>10.3f}{d['between_var']:>14.3f}"
              f"{d['within_var']:>13.3f}")
    print(f"\n  ICC is the share of a score's variance that is explained by\n"
          f"  WHICH RECORDING the candidate came from. A candidate "
          f"discriminator\n  wants this low. High means the score is largely "
          f"a property of the\n  video, and pooling recordings then lets base "
          f"rate masquerade as skill.")
    print()

    cols = [("detector", logit), ("morph only", m), ("E_onto", energy)]
    rows, ci = per_recording_table(cols, ok, fm, rec, top=a.top)
    print()
    inv = transform_invariance(m, ok, fm, rec)
    print()
    cov = None
    if a.recording_covariates:
        cov = json.load(open(a.recording_covariates, encoding="utf-8"))
    nui = nuisance_correlations(cands, m, rec, cov)

    if a.out:
        json.dump({"variance_decomposition": vd, "per_recording": rows,
                   "pooled_ci": ci, "transform_invariance": inv,
                   "nuisance": nui, "eps": EPS},
                  open(a.out, "w", encoding="utf-8"), indent=2)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
