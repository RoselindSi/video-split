"""Can naming-vs-stored agreement separate claim_support = yes from no?

A DIAGNOSTIC, AND UNDERPOWERED BY CONSTRUCTION. The contrast is 29 YES against
6 NO. Six negatives cannot support a claim about anything, and the point of
running this now is to see whether the pipeline end-to-end produces usable
features at all -- coverage, join, sane distributions -- before more
annotation is spent. The minimum AUROC this many events could distinguish from
chance is computed and printed BEFORE any result, so a number near 0.7 is read
as "inside the noise" rather than as encouragement.

WHAT THE ARMS ARE.

    video prior       measured elsewhere at 0.601 [0.495, 0.706] and quoted
                      for context ONLY. It reads no label and therefore cannot
                      verify one -- but it was also computed on 186 events
                      with a different target (`correct` against the rest), so
                      it is not a threshold the naming features can be
                      compared against. Making it one means recomputing it on
                      these events and this label.
    naming support    the naming model's own name for the segment against the
                      STORED label. Both are text, but the prediction is
                      video-grounded, so the comparison IS a video-to-text
                      signal about the stored label. This is the arm that has
                      never had coverage: 8 of 186 last time, and the segment
                      windows were rebuilt precisely to fix that.

THE METRIC FUNCTIONS ARE TRANSCRIBED from eval_naming_decoupled rather than
imported, because importing that module loads torch and transformers for what
is string arithmetic. Transcription can drift, so it is CHECKED: the aggregate
recomputed here from `pred_names` and `gt_names` is compared against the
`verb_acc` and `obj_f1` the eval itself stored on the same row. A mismatch
means the copy is stale and is reported rather than assumed away.

`emb_sim` is NOT recomputed -- it needs a sentence encoder -- so the per-row
aggregate the eval already stored is used where a row-level number is enough,
and the per-segment arm runs without it.

COVERAGE IS REPORTED BEFORE DISCRIMINATION, always. An event contributes only
through segments the annotator was SHOWN, and only where naming returned a
prediction at that position. Both filters shrink n and both have bitten this
project already.

Usage:
    python -m src.auditor.semantic.claim_support_diagnostic \
        --gold data/gold/semantic_ontology_gold_48.json \
        --pred /workspace/tr1/results/auditor/naming_run_pred_v2.jsonl \
        --join /workspace/tr1/results/auditor/naming_run_join.json \
        --event_map /workspace/tr1/results/auditor/naming_targets_48_event_map.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
from collections import Counter, defaultdict

# --- transcribed from src/eval/eval_naming_decoupled.py, checked at runtime ---
W = re.compile(r"[a-zA-Z]+")
tok = lambda s: [w.lower() for w in W.findall(s)]
ORD = {"first", "second", "third", "fourth", "fifth", "sixth", "seventh",
       "eighth", "ninth", "tenth", "final", "initial", "re"}
STOP = {"the", "a", "an", "and", "or", "to", "of", "into", "onto", "on", "in",
        "with", "from", "for", "at", "by", "up", "down", "out", "off", "over",
        "then", "all", "it", "its", "this", "that", "these", "those", "each",
        "again", "perform", "performs", "performing"}
GEN = {"object", "objects", "item", "items", "thing", "things", "stuff",
       "something", "task", "tasks", "step", "steps", "part", "parts", "area",
       "surface", "material"}
CLUSTERS = [
    {"open", "unseal", "uncover", "unwrap", "unzip"},
    {"close", "shut", "seal", "cover", "zip"},
    {"remove", "take", "detach", "extract", "pull", "lift", "withdraw"},
    {"put", "place", "set", "return", "store", "replace", "reposition",
     "position", "insert", "mount", "load", "adjust", "align", "reset",
     "arrange", "straighten"},
    {"inspect", "check", "examine", "look", "observe", "view"},
    {"rotate", "turn", "flip", "spin", "rotation"},
    {"tighten", "screw", "fasten", "secure"},
    {"loosen", "unscrew", "undo", "release"},
    {"stack", "pile", "pack", "repack", "gather"},
    {"fold", "bend", "crease"},
    {"grab", "grasp", "pick", "retrieve", "hold", "grip"},
    {"wipe", "clean", "scrub", "wash", "rinse"},
    {"pour", "empty", "dump", "spread", "tip"},
    {"press", "push", "tap", "operate"},
    {"slip", "slide"}, {"tear", "rip"},
]


def cluster_of(v):
    for i, c in enumerate(CLUSTERS):
        if v in c:
            return i
    return -1


def clusters_in(name):
    return {cluster_of(w) for w in tok(name)} - {-1}


def primary_verb(name):
    for w in tok(name):
        if w not in ORD:
            return w
    return ""


def content(name):
    return {w for w in tok(name) if w not in STOP and w not in ORD}


def verb_match(p, g):
    if clusters_in(p) & clusters_in(g):
        return 1.0
    vp, vg = primary_verb(p), primary_verb(g)
    return 1.0 if (vp and vp == vg) else 0.0


def obj_f1(p, g):
    cp, cg = content(p), content(g)
    if not cp or not cg:
        return 0.0
    inter = len(cp & cg)
    if not inter:
        return 0.0
    pr, rc = inter / len(cp), inter / len(cg)
    return 2 * pr * rc / (pr + rc)


def is_generic(name):
    return 1.0 if (GEN & set(tok(name))) else 0.0
# ------------------------------------------------------------------------


def auroc(scores, labels):
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return None
    wins = sum((1.0 if p > n else 0.5 if p == n else 0.0)
               for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def grouped_boot(scores, labels, groups, n_boot, seed):
    by = defaultdict(list)
    for s, y, g in zip(scores, labels, groups):
        by[g].append((s, y))
    keys = list(by)
    rng = random.Random(seed)
    out = []
    for _ in range(n_boot):
        pick = [by[rng.choice(keys)] for _ in keys]
        flat = [x for g in pick for x in g]
        a = auroc([x[0] for x in flat], [x[1] for x in flat])
        if a is not None:
            out.append(a)
    out.sort()
    if not out:
        return None, None
    return out[int(0.025 * len(out))], out[min(int(0.975 * len(out)),
                                               len(out) - 1)]


def min_detectable(n_pos, n_neg, n_boot, seed):
    """The AUROC a random scorer reaches by chance at this n, upper 97.5%.

    Printed before any result. With six negatives the null band is wide, and
    knowing where it ends is what stops 0.70 being read as a signal."""
    rng = random.Random(seed)
    y = [1] * n_pos + [0] * n_neg
    vals = []
    for _ in range(n_boot):
        s = [rng.random() for _ in y]
        a = auroc(s, y)
        if a is not None:
            vals.append(a)
    vals.sort()
    return vals[min(int(0.975 * len(vals)), len(vals) - 1)]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", action="append", required=True,
                    help="semantic gold json, or a filled enrichment csv")
    ap.add_argument("--pred", required=True,
                    help="eval_naming_decoupled output jsonl")
    ap.add_argument("--join", required=True,
                    help="naming_run_join.json from naming_run_spec")
    ap.add_argument("--event_map", action="append", required=True,
                    help="naming_targets_*_event_map.json")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--video_prior_reference", default="0.601 [0.495, 0.706]",
                    help="quoted for context only. It was measured on a "
                         "different population and a different target, so it "
                         "is NOT a threshold these numbers can be read "
                         "against")
    ap.add_argument("--out")
    a = ap.parse_args()

    # ------------------------------------------------------------- gold
    gold = {}
    for p in a.gold:
        if p.lower().endswith(".csv"):
            with open(p, newline="", encoding="utf-8-sig") as f:
                rows = [r for r in csv.DictReader(f)
                        if (r.get("claim_support") or "").strip()]
        else:
            rows = json.load(open(p, encoding="utf-8-sig"))["events"]
        for r in rows:
            k = r.get("audit_key") or r.get("event_id")
            if k:
                gold[k] = r
        print(f"  {os.path.basename(p):<40} {len(rows):>4} audited events")
    lab = {k: v["claim_support"] for k, v in gold.items()}
    print(f"{len(gold)} audited events: "
          f"{dict(Counter(lab.values()).most_common())}")

    # ------------------------------------------------------ naming output
    preds = {}
    for line in open(a.pred, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            preds[r["video"]] = r
    join = json.load(open(a.join, encoding="utf-8"))
    emap = {}
    for p in a.event_map:
        emap.update(json.load(open(p, encoding="utf-8")))
    print(f"{len(preds)} naming rows; {len(join)} joined segments; "
          f"{len(emap)} mapped events")

    # transcription self-check against the eval's own stored aggregates
    bad = 0
    for r in preds.values():
        pn, gn = r.get("pred_names") or [], r.get("gt_names") or []
        k = min(len(pn), len(gn))
        if not k:
            continue
        va = sum(verb_match(pn[i], gn[i]) for i in range(k)) / k
        of = sum(obj_f1(pn[i], gn[i]) for i in range(k)) / k
        if abs(va - r.get("verb_acc", va)) > 1e-6 \
                or abs(of - r.get("obj_f1", of)) > 1e-6:
            bad += 1
    print(f"  metric transcription check: {len(preds) - bad}/{len(preds)} "
          f"rows reproduce the eval's stored verb_acc and obj_f1"
          + ("" if not bad else "  !! the copy has drifted from "
                                "eval_naming_decoupled"))

    # video path -> recording, so a join key can find its prediction
    vid_of = {}
    for r in preds.values():
        vid_of[r["video"]] = r
    rec_pred = {}
    for uid, j in join.items():
        for v, r in vid_of.items():
            if f"/{j['recording_id']}/" in v or j["recording_id"] in v:
                rec_pred[j["recording_id"]] = r
                break

    # --------------------------------------------------------- per event
    rows, no_seg, no_pred, not_shown = [], [], [], 0
    mis, n_empty = set(), [0]
    for key, g in gold.items():
        m = emap.get(key)
        if not m:
            no_seg.append(key)
            continue
        feats = []
        for s in m["segments"]:
            if not s.get("shown_in_sheet", True):
                not_shown += 1
                continue
            j = join.get(s["segment_uid"])
            pr = rec_pred.get(j["recording_id"]) if j else None
            if not j or not pr:
                continue
            # POSITIONAL JOIN NEEDS AN ALIGNED LIST. When naming returns 97
            # names for 46 segments, pred_names[i] is not segment i and the
            # join would attach a confidently wrong name. Only recordings
            # where the counts match can be read positionally at all.
            if pr.get("n_pred") != pr.get("n_gt"):
                mis.add(j["recording_id"])
                continue
            names = pr.get("pred_names") or []
            if j["position"] >= len(names):
                continue
            p, stored = names[j["position"]], j["stored_label"]
            # CHUNKED RUNS PAD SHORT BLOCKS WITH "". That keeps every later
            # segment on its own name instead of shifting them, and it also
            # makes n_pred == n_gt by construction -- so the count guard above
            # can no longer see a hole. An empty name is a hole.
            if not p.strip():
                n_empty[0] += 1
                continue
            feats.append({"verb": verb_match(p, stored),
                          "obj": obj_f1(p, stored),
                          "generic": is_generic(p),
                          "pred_name": p, "stored_label": stored})
        if not feats:
            no_pred.append(key)
            continue
        rows.append({"audit_key": key, "claim_support": lab[key],
                     "recording_id": m["recording_id"],
                     "n_segments_used": len(feats),
                     "verb_min": min(f["verb"] for f in feats),
                     "verb_mean": sum(f["verb"] for f in feats) / len(feats),
                     "obj_min": min(f["obj"] for f in feats),
                     "obj_mean": sum(f["obj"] for f in feats) / len(feats),
                     "generic_any": max(f["generic"] for f in feats),
                     "segments": feats})

    print(f"\nCOVERAGE, before any discrimination:")
    print(f"  events with a usable naming feature  {len(rows)}/{len(gold)}")
    print(f"  no entry in the event map            {len(no_seg)}")
    print(f"  no prediction at any shown segment   {len(no_pred)}")
    print(f"  segments skipped as not shown        {not_shown}")
    n_mis = sum(1 for r in preds.values()
                if r.get("n_pred") != r.get("n_gt"))
    print(f"  segments with an EMPTY predicted name {n_empty[0]}  "
          f"(chunk padding: naming returned fewer names than asked)")
    print(f"  recordings dropped, n_pred != n_gt    {len(mis)} "
          f"(of {n_mis}/{len(preds)} misaligned in the naming output)")
    if n_mis > len(preds) * 0.5:
        print(f"  !! more than half the naming output is misaligned. A "
              f"positional join cannot be\n     trusted there, and the "
              f"numbers below are from the aligned minority -- which is\n"
              f"     a different, easier population (short recordings).")
    used = [r for r in rows if r["claim_support"] in ("yes", "no")]
    n_pos = sum(1 for r in used if r["claim_support"] == "yes")
    n_neg = len(used) - n_pos
    print(f"  YES vs NO with features              {n_pos} vs {n_neg} over "
          f"{len({r['recording_id'] for r in used})} recordings")
    if n_neg < 2 or n_pos < 2:
        raise SystemExit(
            "  not enough of one class to compute anything. That is the "
            "result: the pipeline\n  runs and the supervision does not exist "
            "yet.")

    mdc = min_detectable(n_pos, n_neg, a.n_boot, a.seed)
    print(f"\n  A RANDOM scorer reaches AUROC {mdc:.3f} at the 97.5th "
          f"percentile with {n_pos} vs {n_neg}.\n  Anything below that is "
          f"inside the noise, and it is printed here so it cannot be\n  "
          f"decided after seeing the numbers.")

    print(f"\nDISCRIMINATION (single features, no model fitted):")
    print(f"  {'feature':<14}{'AUROC':>8}{'grouped 95%':>22}")
    res = {}
    y = [1 if r["claim_support"] == "yes" else 0 for r in used]
    grp = [r["recording_id"] for r in used]
    for f in ("verb_min", "verb_mean", "obj_min", "obj_mean", "generic_any"):
        s = [r[f] for r in used]
        au = auroc(s, y)
        lo, hi = grouped_boot(s, y, grp, a.n_boot, a.seed)
        flag = "" if au is None or au <= mdc else "   > chance band"
        print(f"  {f:<14}{au:>8.3f}   [{lo:.3f}, {hi:.3f}]{flag}")
        res[f] = {"auroc": au, "grouped_95": [lo, hi]}
    print(f"\n  video prior, quoted: {a.video_prior_reference}")
    print(f"  NOT COMPARABLE TO THE ROWS ABOVE, and I originally printed it "
          f"as if it were.\n  That number was measured on 186 events with "
          f"the target `correct` against everything\n  else; these are "
          f"{n_pos + n_neg} events with `yes` against `no`. Different "
          f"population, different\n  label, different n. A naming feature "
          f"sitting above it has not been shown to buy\n  anything -- "
          f"establishing that needs the prior recomputed on THESE events "
          f"against\n  THIS label.")

    if a.out:
        json.dump({"coverage": {"with_features": len(rows),
                                "n_yes": n_pos, "n_no": n_neg,
                                "no_event_map": len(no_seg),
                                "no_prediction": len(no_pred)},
                   "min_detectable_auroc": mdc, "features": res,
                   "events": [{k: v for k, v in r.items() if k != "segments"}
                              for r in used]},
                  open(a.out, "w", encoding="utf-8"), indent=2,
                  ensure_ascii=False)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
