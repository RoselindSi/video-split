"""Pool B: a model-independent sampling frame over every segment. Builds, does not sample.

WHY A SECOND FRAME AND WHY NOT THE OBVIOUS ONE. Pool A -- the 48 frozen plus
the 41 enrichment rows -- comes from a boundary-error audit and is therefore
enriched for boundary problems. It carries semantic errors only incidentally,
which is why the measured `claim_support = no` yields are what they are.

Naming-model disagreement would find negatives quickly and would make the
sampling mechanism a function of the current naming model's failure modes. A
semantic auditor trained on gold collected that way learns "where this naming
model is weak" and there is afterwards no way to separate that from "semantic
correctness". The convenience is not worth losing the distinction.

Uniform random over 12,446 segments does not depend on any model but, if the
true prevalence of semantic error is low, spends hundreds of audits to find
ten negatives.

So: stratify on SURFACE properties of the annotation, which need no model and
no assumption about which stratum is error-prone. Every feature below is
computed from the stored label and the segment bounds alone:

    duration bucket        short segments and very long ones are different
                           annotation problems
    token count            label length as a crude complexity proxy
    conjunction markers    `and`, `then`, `;`, `/`, `->` -- compound claims
    ordinal markers        `8th`, `second`, `x3` -- repeated-instance labels
    neighbour similarity   how much the label shares with its neighbours,
                           the repeated-instance signature from the boundary
                           side
    recording              so a stratum is not one recording's habit

NO PART-OF-SPEECH TAGGING. "Verb count" and "object count" would need a tagger
or a verb list, and this project has neither frozen; a bad tagger would put a
silent model back into a frame built to exclude one. Conjunction and ordinal
markers are surface strings and mean exactly what they are.

THIS FILE BUILDS THE FRAME AND REPORTS IT. It samples only when given an
explicit deficit, because the deficit is not known until the 41 productive
rows in pool A are exhausted, and sizing a batch against a projection is how a
planning estimate turns into a fact nobody checked.

POOL SEPARATION, recorded here so it survives:

    A   48 frozen + 41 enrichment, boundary-enriched
        ontology development, training, diagnostics
    B   this frame, surface-stratified
        raising NO support, training
    C   later, a frozen representative sample
        the only pool a performance claim may be made on

B raises supervision and cannot carry a performance number, or the boundary
selection bias is simply replaced by a semantic enrichment bias with a better
name. Every row carries `pool` and its stratum so the three never merge by
accident.

Usage:
    python -m src.auditor.semantic.surface_frame --plan \
        --recseg '/workspace/tr1/data_recseg*/recseg_*.json' \
        --exclude data/gold/semantic_ontology_gold_48.json
    python -m src.auditor.semantic.surface_frame --deficit 12 --out ...
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import re
from collections import Counter, defaultdict

from src.auditor.semantic.render_ontology_clips import get_segments, get_video

CONJ = re.compile(r"\b(and|then|after|before|while|plus)\b|[;/]|->", re.I)
ORDINAL = re.compile(r"\b(\d+(st|nd|rd|th)|first|second|third|fourth|fifth|"
                     r"x\d+|\d+x)\b", re.I)


def tokens(s):
    return re.sub(r"[^a-z0-9 ]", " ", (s or "").lower()).split()


def jaccard(a, b):
    A, B = set(tokens(a)), set(tokens(b))
    return len(A & B) / len(A | B) if A and B else 0.0


def duration_bucket(d):
    for hi, name in ((3, "0-3s"), (8, "3-8s"), (15, "8-15s"), (30, "15-30s")):
        if d < hi:
            return name
    return "30s+"


def token_bucket(n):
    return "1-3" if n <= 3 else ("4-6" if n <= 6 else "7+")


def features(label, start, end, prev_label, next_label):
    t = tokens(label)
    return {
        "duration_bucket": duration_bucket(end - start),
        "token_bucket": token_bucket(len(t)),
        "compound": bool(CONJ.search(label or "")),
        "ordinal": bool(ORDINAL.search(label or "")),
        "neighbour_similar": max(jaccard(label, prev_label),
                                 jaccard(label, next_label)) >= 0.5,
    }


def stratum_of(f):
    return (f"{f['duration_bucket']}|{f['token_bucket']}"
            f"|{'C' if f['compound'] else '-'}"
            f"{'O' if f['ordinal'] else '-'}"
            f"{'N' if f['neighbour_similar'] else '-'}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recseg", action="append", required=True,
                    help="file, directory or glob; quote globs")
    ap.add_argument("--exclude", action="append", default=[],
                    help="gold json(s) whose recordings' segments are already "
                         "in pool A and must not reappear in pool B")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--deficit", type=int, default=0,
                    help="how many MORE clean negatives are needed. Sampling "
                         "happens only when this is given, because it is "
                         "unknown until pool A's productive rows are "
                         "exhausted")
    ap.add_argument("--assumed_no_rate", type=float, default=0.10,
                    help="prevalence of claim_support=no in an unselected "
                         "segment. A GUESS with no measurement behind it, "
                         "used only to size the batch, and printed as such")
    ap.add_argument("--per_stratum_cap", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    a = ap.parse_args()

    paths, unresolved = [], []
    for pat in a.recseg:
        hits = (sorted(glob.glob(os.path.join(pat, "*.json")))
                if os.path.isdir(pat) else
                ([pat] if os.path.exists(pat) else sorted(glob.glob(pat))))
        if not hits:
            unresolved.append(pat)
        for h in hits:
            if h not in paths and not h.endswith(".manifest.json"):
                paths.append(h)
    if unresolved:
        print(f"  !! matched nothing: {unresolved}")

    seen, segs = set(), []
    for p in paths:
        blob = json.load(open(p, encoding="utf-8"))
        if isinstance(blob, dict):
            blob = blob.get("recordings") or blob.get("data") or []
        n_new = 0
        for r in blob:
            rid = r.get("recording_id")
            if not rid or rid in seen:
                continue
            seen.add(rid)
            lst = sorted(([str(x[0]), float(x[1]), float(x[2])]
                          for x in get_segments(r)[0]), key=lambda x: x[1])
            vid = get_video(r)
            for i, (lab, st, en) in enumerate(lst):
                segs.append({
                    "recording_id": rid, "seg_index": i, "label": lab,
                    "start": round(st, 2), "end": round(en, 2),
                    "video": vid,
                    "prev_label": lst[i - 1][0] if i else "",
                    "next_label": lst[i + 1][0] if i + 1 < len(lst) else ""})
            n_new += 1
        print(f"  {os.path.join(os.path.basename(os.path.dirname(p)), os.path.basename(p)):<44} "
              f"{n_new:>4} new recordings")

    # The gold rows carry `event_id` and NOT `recording_id`, so reading only
    # recording_id made this filter a no-op: it printed "0 dropped from 0"
    # underneath a frame that still contained every pool-A recording. A
    # silently empty exclusion is worse than none, because the leak it allows
    # is invisible in the output.
    excl, no_key = set(), 0
    for p in a.exclude:
        if not os.path.exists(p):
            print(f"  !! --exclude {p} not found")
            continue
        blob = json.load(open(p, encoding="utf-8-sig"))
        items = (blob.get("events") if isinstance(blob, dict)
                 else blob if isinstance(blob, list) else [])
        n0 = len(excl)
        for e in items:
            if not isinstance(e, dict):
                continue
            rid = e.get("recording_id")
            if not rid:
                m = re.match(r"^(recording_\d+)", str(e.get("event_id") or ""))
                rid = m.group(1) if m else None
            if rid:
                excl.add(rid)
            else:
                no_key += 1
        print(f"  --exclude {os.path.basename(p)}: {len(items)} rows -> "
              f"{len(excl) - n0} recordings")
    if no_key:
        print(f"  !! {no_key} excluded rows had neither recording_id nor a "
              f"parseable event_id")
    if a.exclude and not excl:
        raise SystemExit(
            "--exclude was given and matched no recordings. That would build "
            "pool B on top of pool A and the leak would not show in any "
            "number below. Fix the input rather than proceeding.")
    before = len(segs)
    segs = [s for s in segs if s["recording_id"] not in excl]
    print(f"\n{before} segments over {len(seen)} recordings; "
          f"{before - len(segs)} dropped from {len(excl)} pool-A recordings")

    strata = defaultdict(list)
    for s in segs:
        f = features(s["label"], s["start"], s["end"],
                     s["prev_label"], s["next_label"])
        s.update(f)
        s["stratum"] = stratum_of(f)
        strata[s["stratum"]].append(s)

    print(f"\nFRAME: {len(strata)} strata over {len(segs)} segments")
    print(f"  marginals (each computed on every segment, so they overlap):")
    for key in ("duration_bucket", "token_bucket"):
        print(f"    {key:<20} {dict(Counter(s[key] for s in segs).most_common())}")
    for key in ("compound", "ordinal", "neighbour_similar"):
        n = sum(1 for s in segs if s[key])
        print(f"    {key:<20} {n} / {len(segs)} ({100*n/len(segs):.0f}%)")
    big = sorted(strata.items(), key=lambda kv: -len(kv[1]))
    print(f"\n  largest strata: "
          f"{[(k, len(v)) for k, v in big[:5]]}")
    print(f"  strata with fewer than {a.per_stratum_cap} segments: "
          f"{sum(1 for _k, v in strata.items() if len(v) < a.per_stratum_cap)}")

    if not a.deficit:
        print(f"\n--deficit not given, so nothing is sampled. The deficit is "
              f"unknown until the 41\nproductive rows in pool A are audited, "
              f"and sizing a batch against a projection is\nhow an estimate "
              f"becomes a number nobody checked. Rerun with --deficit N.")
        return

    n_need = int(round(a.deficit / max(a.assumed_no_rate, 1e-6)))
    print(f"\nSIZING: {a.deficit} more negatives at an ASSUMED rate of "
          f"{a.assumed_no_rate:.2f}\n  -> {n_need} segments to audit. That "
          f"rate is a guess with no measurement behind it;\n  the first "
          f"completed batch replaces it and the batch after that is sized "
          f"properly.")

    rng = random.Random(a.seed)
    picked, order = [], sorted(strata, key=lambda k: (-len(strata[k]), k))
    depth = 0
    while len(picked) < n_need and depth < a.per_stratum_cap:
        for k in order:
            v = strata[k]
            if len(v) > depth and len(picked) < n_need:
                if depth == 0:
                    rng.shuffle(v)
                picked.append(v[depth])
        depth += 1
    print(f"  sampled {len(picked)} over "
          f"{len({p['stratum'] for p in picked})} strata, "
          f"{len({p['recording_id'] for p in picked})} recordings")

    if a.out:
        cols = ["pool", "stratum", "recording_id", "seg_index", "start",
                "end", "duration_bucket", "token_bucket", "compound",
                "ordinal", "neighbour_similar", "stored_label", "video"]
        FIELDS = ["claim_support", "granularity", "major_action_missing",
                  "action_presence", "segment_structure",
                  "upstream_timing_issue", "actual_action_summary",
                  "semantic_audit_note"]
        with open(a.out, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(cols + FIELDS)
            for s in picked:
                w.writerow(["B", s["stratum"], s["recording_id"],
                            s["seg_index"], s["start"], s["end"],
                            s["duration_bucket"], s["token_bucket"],
                            s["compound"], s["ordinal"],
                            s["neighbour_similar"], s["label"], s["video"]]
                           + [""] * len(FIELDS))
        print(f"\nwrote {a.out}")
        print(f"  every row carries pool=B and its stratum. Pool B raises "
              f"supervision and cannot\n  carry a performance number -- that "
              f"needs pool C, a frozen representative sample.")


if __name__ == "__main__":
    main()
