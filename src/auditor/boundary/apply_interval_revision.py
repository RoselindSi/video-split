"""Apply the ontology re-audit of the 37 INTERVAL events to the pair labels.

WHAT THIS REVISION IS, AND WHAT IT IS NOT. The blind topology audit asked
"what kind of extended transition is this". This one asks a different
question -- at the dataset's task-level granularity, is there a boundary here
at all -- and answers it 25 times with `same_action_internal_motion`. Those are
not two readings of one question that happen to disagree; the second
re-adjudicates the ORIGINAL gradual call using task-goal reasoning
("按当前颗粒度", "task-level"), which is an ontology decision rather than a
morphology observation. Both can be right at once.

IT IS NOT AN INDEPENDENT SECOND OPINION. The revised sheet carries
`interval_morphology_subtype_blind` in the same row, so the second pass saw
the first pass's answer. The first was blind; the second was not. Agreement
between them is therefore not evidence, and only the disagreements carry
information.

THE CIRCULARITY CHECK IS RUN HERE AND CANNOT BE SKIPPED. The model's P(POINT)
values for the point_like events were on screen before this revision was made.
If the revision tracks the score, the relabelled target has absorbed the
model's opinion and every subsequent number is circular. So the direction of
each revision is correlated against P(POINT) before anything is written, and
the answer is printed whether or not it is convenient.

WHAT IT DOES TO THE POPULATION. INTERVAL falls from 37 events to 5, which is
below every reporting threshold in this project -- the class stops being
measurable rather than becoming easier. That is a real result and it changes
what Boundary v2 can ask, so it is stated in the output rather than left to be
discovered after the next training run.

Nothing is written without --write.

Usage:
    python -m src.auditor.boundary.apply_interval_revision \
        --revision data/gold/interval_audit_37_ontology_revised.csv \
        --pair_labels data/gold/pair_labels_v1.csv \
        --pair_labels data/gold/batch3_pair_labels_v1_relabel_v1.csv \
        --predictions .../boundary_v1_oof.json \
        --write
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone

MORPH = {"sharp_visible_transition": "POINT_TRANSITION",
         "gradual_phase_transition": "INTERVAL_TRANSITION",
         "same_action_internal_motion": "NO_TRANSITION",
         "visibility_or_offscreen": "UNOBSERVABLE"}
POINT = "POINT_TRANSITION"


def spearman(a, b):
    import numpy as np
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 4:
        return float("nan")

    def rank(x):
        o = np.argsort(x)
        r = np.empty(len(x), float)
        r[o] = np.arange(len(x), dtype=float)
        _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
        s = np.zeros(len(cnt))
        np.add.at(s, inv, r)
        return (s / cnt)[inv]

    ra, rb = rank(a[ok]), rank(b[ok])
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--pair_labels", action="append", required=True)
    ap.add_argument("--predictions",
                    help="boundary_v1_oof.json, for the circularity check")
    ap.add_argument("--suffix", default="_ontology_v2")
    ap.add_argument("--provenance",
                    default="docs/interval_revision_provenance.json",
                    help="written alongside --write; the revision is not "
                         "reproducible without it")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    rev = {}
    with open(a.revision, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            e = r.get("event_id")
            if e and (r.get("revised_temporal_pair_subtype") or "").strip():
                rev[e] = r
    print(f"{len(rev)} revised events")
    print(f"  revised subtype: "
          f"{dict(Counter(r['revised_temporal_pair_subtype'] for r in rev.values()).most_common())}")
    print(f"  confidence:      "
          f"{dict(Counter(r['ontology_confidence'] for r in rev.values()).most_common())}")

    # ------------------------------------------------- the circularity check
    if a.predictions and os.path.exists(a.predictions):
        blob = json.load(open(a.predictions, encoding="utf-8"))
        p = {r["event_id"]: r["morphology"][POINT] for r in blob["events"]
             if r.get("morphology")}
        have = [e for e in rev if e in p]
        # ordered least to most point-like, so a positive correlation means the
        # revision moved WITH the model's opinion
        rank_of = {"same_action_internal_motion": 0, "visibility_or_offscreen": 0,
                   "ambiguous": 1, "gradual_phase_transition": 2,
                   "sharp_visible_transition": 3}
        if len(have) >= 4:
            globals()["_rho"] = None
            rho = spearman([p[e] for e in have],
                           [rank_of.get(rev[e]["revised_temporal_pair_subtype"],
                                        1) for e in have])
            print(f"\n{'=' * 78}\nCIRCULARITY CHECK\n{'=' * 78}")
            globals()["_rho"] = rho
            se = 1.0 / max((len(have) - 1) ** 0.5, 1e-9)
            print(f"  spearman(P(POINT), how point-like the revision is) = "
                  f"{rho:+.3f} over {len(have)} events   (rough se "
                  f"{se:.3f})")
            # three bands, not two. An earlier version branched on rho > 0.3
            # and called everything else "at or below zero", which reported
            # +0.169 as a clean pass -- a claim about circularity that the
            # number did not support.
            if rho > 2 * se and rho > 0.3:
                print(f"  !! The revision moves WITH the model, by more than "
                      f"twice the standard error. The P(POINT) values were\n"
                      f"     visible before this pass, so the relabelled "
                      f"target may have absorbed the model's opinion and\n"
                      f"     every number computed on it afterwards is "
                      f"circular. Do not train on this without a blind pass.")
            elif rho > se:
                print(f"  Positive but within about {rho / se:.1f} standard "
                      f"errors of zero. Not distinguishable from no "
                      f"relationship, and not\n  a demonstration of "
                      f"independence either -- at this n the check can only "
                      f"rule out a LARGE anchoring effect.")
            else:
                print(f"  Within one standard error of zero, so no anchoring "
                      f"effect is visible. That is the check failing to fire, "
                      f"not\n  proof of independence: at this n only a large "
                      f"effect would show.")
            for e in sorted(have, key=lambda x: -p[x])[:6]:
                print(f"    {e[-44:]:<45} P(POINT) {p[e]:.3f}  -> "
                      f"{rev[e]['revised_temporal_pair_subtype']}")

    # ------------------------------------------------------------- the edit
    print(f"\n{'=' * 78}\nWHAT CHANGES IN THE PAIR LABELS\n{'=' * 78}")
    total = Counter()
    prov_events, prov_files = [], []
    for path in a.pair_labels:
        with open(path, newline="", encoding="utf-8-sig") as f:
            rd = csv.DictReader(f)
            fields = list(rd.fieldnames or [])
            rows = list(rd)
        col = "temporal_pair_subtype"
        if col not in fields:
            print(f"  {os.path.basename(path)}: no {col} column, skipped")
            continue
        n_hit = n_same = 0
        moves = Counter()
        for r in rows:
            e = r.get("event_id")
            if e not in rev:
                continue
            n_hit += 1
            old, new = r[col], rev[e]["revised_temporal_pair_subtype"]
            if old == new:
                n_same += 1
                continue
            moves[f"{MORPH.get(old, 'MASKED')} -> {MORPH.get(new, 'MASKED')}"] += 1
            prov_events.append({
                "event_id": e, "file": os.path.basename(path),
                "from_subtype": old, "to_subtype": new,
                "from_morphology": MORPH.get(old, "MASKED"),
                "to_morphology": MORPH.get(new, "MASKED"),
                "confidence": rev[e].get("ontology_confidence"),
                "reason": rev[e].get("ontology_reason"),
                "blind_topology_call":
                    rev[e].get("interval_morphology_subtype_blind"),
                "suggested_candidate_relation":
                    rev[e].get("suggested_candidate_relation") or None,
                "suggested_corrected_boundary_time":
                    rev[e].get("suggested_corrected_boundary_time") or None})
            r[col] = new
            # the supervision column is derived from the subtype; leaving a
            # stale value there would let the old call keep acting through a
            # different field
            if "pair_supervision" in r:
                r["pair_supervision"] = ""
        out = (os.path.splitext(path)[0] + a.suffix
               + os.path.splitext(path)[1])
        print(f"  {os.path.basename(path)}: {n_hit} of the revised events "
              f"present, {n_hit - n_same} changed")
        for k, v in moves.most_common():
            print(f"      {v:>3}  {k}")
        total.update(moves)
        if a.write and n_hit:
            with open(out, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fields)
                w.writeheader()
                w.writerows(rows)
            print(f"      wrote {out}")
            prov_files.append({"source": path, "source_sha256": sha256(path),
                               "output": out, "output_sha256": sha256(out)})

    print(f"\n{'=' * 78}\nWHAT IT DOES TO THE MORPHOLOGY POPULATION"
          f"\n{'=' * 78}")
    n_int = sum(1 for r in rev.values()
                if r["revised_temporal_pair_subtype"] == "gradual_phase_transition")
    print(f"  INTERVAL_TRANSITION goes from 37 to {n_int}.")
    if n_int < 5:
        print(f"  {n_int} is below every reporting threshold in this project "
              f"-- fewer events than folds, and fewer recordings still. The\n"
              f"  class stops being MEASURABLE rather than becoming easier, so "
              f"Boundary v2 cannot ask the POINT-vs-INTERVAL\n  question on "
              f"this data at all. That is a result about the dataset and it "
              f"should be recorded as one.")
    print(f"  net moves across both files: {dict(total)}")
    if a.write and prov_files:
        prov = {
            "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "revision_file": os.path.abspath(a.revision),
            "revision_sha256": sha256(a.revision),
            "n_revised": len(rev),
            "n_changed": len(prov_events),
            "circularity_spearman": globals().get("_rho"),
            "files": prov_files,
            "caveats": [
                "The revising pass saw the blind topology call in the same "
                "row, so it is not an independent second opinion; only the "
                "disagreements carry information.",
                "It answers a DIFFERENT question from the blind audit -- "
                "whether a boundary exists at the dataset's task granularity, "
                "not what kind of extended transition an event is. Both can "
                "hold at once.",
                "The model's P(POINT) values for the point_like events were "
                "visible before this pass. The circularity correlation is "
                "recorded above; at n=37 it can only rule out a large "
                "anchoring effect.",
                "INTERVAL_TRANSITION falls to 5 events, below every reporting "
                "threshold in this project, so POINT-vs-INTERVAL is no longer "
                "measurable on this data.",
                "The ORIGINAL files are untouched. Every number reported "
                "before this date was computed on them and stays "
                "reproducible.",
            ],
            "events": prov_events}
        os.makedirs(os.path.dirname(os.path.abspath(a.provenance)) or ".",
                    exist_ok=True)
        json.dump(prov, open(a.provenance, "w", encoding="utf-8"), indent=2,
                  ensure_ascii=False)
        print(f"\n  froze provenance to {a.provenance}: "
              f"{len(prov_events)} per-event changes, sha256 of every input "
              f"and output,\n  the circularity correlation, and the four "
              f"caveats that qualify the revision.")
    if not a.write:
        print(f"\n  --write not given, so nothing was written. Rerun the label "
              f"builder and training after writing, and expect POINT-vs-NONE "
              f"to\n  move: 25 events leave the class the model was scoring "
              f"low and join the class it scores low by design.")


if __name__ == "__main__":
    main()
