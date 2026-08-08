"""Arm 1: what does the student alone achieve on AUTO_REJECT? No API calls.

The teacher has to beat this, and the composition of the target says the bar
may already be high. 35 of the 36 safe-to-delete events are
`same_action_internal_motion` -- the single class the verifier was actually
supervised against, since CLEAN_BINARY is sharp versus same_action and nothing
else. So the teacher is being asked to certify the one discrimination the
student was built for, and measuring that before spending anything is the
cheapest decision available.

NO REFIT. This scores the DEPLOYED student output, taken from the frozen
policy decisions, because that is what a cascade would actually gate on.
Fitting a fresh head against reject_safe would answer a different question --
"could a student be built for this" -- and reporting it as the baseline would
quietly replace a measurement with a search. If this arm is weak, that refit
becomes the next question rather than a footnote to this one.

THE THRESHOLD IS CHOSEN NESTED. The frozen policy proposes no rejects at all
(reject_below is -1.0, added as an action-space option and never turned on),
so there is no threshold to inherit and one has to be picked. It is picked
inside each outer fold's TRAINING recordings and applied only to that fold's
held-out ones. Pooling out-of-fold scores over every event and choosing a
threshold that looks good on them is how every earlier operating point in this
project was selected, and it is why nested selection kept breaching afterwards.

THE SCORES MUST ALREADY BE OUT-OF-FOLD. This file cannot verify that; it reads
whatever the decisions file holds. If those scores came from a model that saw
these recordings in training, every number below is optimistic and the nested
threshold does not repair it.

EVERY NON-REJECT-SAFE EVENT IS A FALSE REJECT, including the ones whose truth
is unknown. Deleting an offscreen candidate destroys something whether or not
the gold can say what, so there is no excluded category -- and each false
reject is printed individually with its subtype and corrected boundary,
because at this precision target they are single events, not a rate.

Usage:
    python -m src.auditor.reject_baseline \
        --reject_safe /workspace/tr1/results/hal/c3/reject_safe.json \
        --decisions .../policy_decisions_v4.primary_transportability_frontier.csv \
        --exclude_review /workspace/tr1/results/hal/c3/teacher_observe_only.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict

import numpy as np

from src.boundary.pairwise_verifier import stratified_grouped_folds
from src.boundary.state_adapter import _auroc
from src.boundary.c3_selective_policy import wilson

MIN_PRECISION = 0.95
MIN_N = 20


def buffered(tp, fp):
    """Precision after one more false reject. Written as a formula because
    'still 0.95 with one extra error' is ambiguous: 19/20 reads as a pass and
    19/(19+1+1) = 0.905 does not."""
    return tp / (tp + fp + 1) if (tp + fp + 1) else float("nan")


def pick_threshold(scores, y, target=MIN_PRECISION):
    """The HIGHEST cut whose prefix of lowest-scoring events still satisfies
    the buffered precision. Reject is the low tail, so the sort is ascending;
    everything else mirrors the keep-side selection."""
    m = np.isfinite(scores)
    if m.sum() < 10 or len(set(y[m].tolist())) < 2:
        return None
    order = np.argsort(scores[m])
    s, yy = scores[m][order], y[m][order]
    tp = fp = 0
    th = None
    for i in range(len(s)):
        tp += int(yy[i] == 1)
        fp += int(yy[i] == 0)
        if buffered(tp, fp) >= target:
            th = float(s[i])
    return th


def grouped_bootstrap(rej, ok, groups, n_boot, seed):
    """Recordings are resampled, not events. Events inside one recording share
    a scene, a camera and an annotator pass, so resampling them independently
    would treat correlated errors as independent and shrink the interval."""
    by = defaultdict(list)
    for r, o, g in zip(rej, ok, groups):
        by[g].append((r, o))
    keys = list(by)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(keys), len(keys))
        t = f = 0
        for i in pick:
            for r, o in by[keys[i]]:
                if r:
                    t += int(o)
                    f += int(not o)
        if t + f:
            out.append(t / (t + f))
    return (float(np.percentile(out, 2.5)),
            float(np.percentile(out, 97.5))) if out else (float("nan"),) * 2


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reject_safe", required=True,
                    help="output of src.auditor.reject_safe --out")
    ap.add_argument("--decisions", required=True)
    ap.add_argument("--gold", action="append",
                    default=["data/gold/audit_188_gold_v2.jsonl"])
    ap.add_argument("--exclude_review", action="append", default=[],
                    help="teacher development events, removed so this arm is "
                         "measured on the same held-out set as the teacher")
    ap.add_argument("--score_col", default=None,
                    help="which arm to threshold. Not guessed when several "
                         "are present: the decisions file carries all three "
                         "scorers side by side and picking whichever wins "
                         "after seeing the result is a search, not a baseline")
    ap.add_argument("--policy", help="policy result json, to read the arm the "
                                     "frozen operating point actually used")
    ap.add_argument("--all_arms", action="store_true",
                    help="report every arm. Only legitimate as a diagnostic: "
                         "the deployed arm is the baseline, and the best of "
                         "three chosen afterwards is not")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    a = ap.parse_args()

    rs = json.load(open(a.reject_safe, encoding="utf-8"))
    safe = {r["event_id"]: r for r in rs["events"]}
    gold = {}
    for p in a.gold:
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    gold[r["event_id"]] = r
    dev = set()
    for p in a.exclude_review:
        if os.path.exists(p):
            for r in json.load(open(p, encoding="utf-8")).get("results", []):
                dev.add(r["event_id"])

    with open(a.decisions, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        fields = list(rd.fieldnames or [])
        raw = list(rd)
    NON_SCORE = {"event_id", "recording_id", "source", "y", "subtype",
                 "reliability", "decision", "reason", "policy_role"}
    numeric = [c for c in fields if c not in NON_SCORE
               and sum(1 for r in raw[:50] if _isnum(r.get(c))) > len(raw[:50]) / 2]
    cols = resolve_cols(a, numeric, raw)
    print(f"{os.path.basename(a.decisions)} carries {len(numeric)} scorers: "
          + ", ".join(f"`{c}`" for c in numeric))
    if len(cols) > 1:
        print("  !! --all_arms: every arm below is a separate baseline. The "
              "deployed one is THE baseline; reading off whichever\n     arm "
              "wins here and comparing the teacher against that would import "
              "a three-way search into the comparison.")

    for col in cols:
        rows = []
        for r in raw:
            e = r["event_id"]
            if e not in safe or not safe[e]["audited"]:
                continue
            try:
                s = float(r[col])
            except (TypeError, ValueError):
                continue
            try:
                rel = float(r.get("reliability"))
            except (TypeError, ValueError):
                rel = float("nan")
            rows.append({"event_id": e, "recording_id": r["recording_id"],
                         "score": s, "y": bool(safe[e]["reject_safe"]),
                         "subtype": safe[e]["subtype"], "dev": e in dev,
                         "reliability": rel, "reason": r.get("reason")})
        run_arm(a, col, rows, safe, gold)
    closing()


def _isnum(x):
    try:
        float(x)
        return True
    except (TypeError, ValueError):
        return False


def resolve_cols(a, numeric, raw):
    """Which arm to threshold. Refuses to guess when several are present."""
    if a.score_col:
        if a.score_col not in numeric:
            raise SystemExit(f"`{a.score_col}` is not one of {numeric}")
        return [a.score_col]
    if a.policy and os.path.exists(a.policy):
        txt = json.dumps(json.load(open(a.policy, encoding="utf-8")))
        hit = [c for c in numeric if f'"{c}"' in txt]
        if len(hit) == 1:
            print(f"  arm read from {os.path.basename(a.policy)}: `{hit[0]}`")
            return hit
        print(f"  !! {os.path.basename(a.policy)} names {len(hit)} of the "
              f"arms, so it does not identify the deployed one")
    if a.all_arms:
        return numeric
    if len(numeric) == 1:
        return numeric
    raise SystemExit(
        "several scorers are present and this file will not pick one for "
        "you:\n  " + "\n  ".join(numeric) + "\n\nThe deployed arm is the "
        "baseline the teacher has to beat. Choosing whichever of the three\n"
        "looks best after the fact would put a three-way search inside the "
        "number.\n\nEither name it with --score_col, point --policy at the "
        "result json that recorded the frozen\noperating point, or pass "
        "--all_arms to see all three as a diagnostic.")


def run_arm(a, col, rows, safe, gold):
    print(f"\n\n{'#' * 78}\n# ARM `{col}`\n{'#' * 78}")
    print(f"  {len(rows)} audited events carry a student score  "
          f"({sum(1 for r in rows if r['y'])} reject-safe)")
    miss = [e for e, v in safe.items()
            if v["audited"] and e not in {r['event_id'] for r in rows}]
    if miss:
        print(f"  !! {len(miss)} audited events have no score and are absent "
              f"from every number below, e.g. {miss[:2]}")

    for tag, sel in (("ALL AUDITED", rows),
                     ("HELD OUT (teacher-development events removed)",
                      [r for r in rows if not r["dev"]])):
        if not sel:
            continue
        y = np.array([r["y"] for r in sel], float)
        sc = np.array([r["score"] for r in sel], float)
        groups = [r["recording_id"] for r in sel]
        print(f"\n{'=' * 78}\n{tag}: {len(sel)} events, "
              f"{int(y.sum())} reject-safe ({y.mean():.3f})\n{'=' * 78}")
        # ranking quality first: a threshold cannot rescue a score that does
        # not order the two classes, and this is independent of any cut
        print(f"  AUROC of the low tail against reject_safe: "
              f"{_auroc(y, -sc):.3f}")

        folds = stratified_grouped_folds(groups, y, a.n_folds, seed=a.seed)
        rej = np.zeros(len(sel), bool)
        print(f"\n  {'fold':>4} {'n_test':>7} {'thr':>8} {'rejected':>9} "
              f"{'TP':>4} {'FP':>4} {'prec':>6} {'buff':>6}")
        for fi, f in enumerate(folds):
            te = np.array([g in f for g in groups])
            tr = ~te
            if te.sum() < 2 or tr.sum() < 10:
                continue
            th = pick_threshold(sc[tr], y[tr])
            if th is None:
                print(f"  {fi:>4} {int(te.sum()):>7} {'none':>8}   "
                      f"no cut on the training recordings reaches the "
                      f"buffered precision")
                continue
            hit = te & (sc <= th)
            rej |= hit
            tp = int(y[hit].sum())
            fp = int(hit.sum() - tp)
            print(f"  {fi:>4} {int(te.sum()):>7} {th:>8.4f} "
                  f"{int(hit.sum()):>9} {tp:>4} {fp:>4} "
                  f"{(tp / max(1, tp + fp)):>6.3f} {buffered(tp, fp):>6.3f}")

        tp = int(y[rej].sum())
        fp = int(rej.sum() - tp)
        # undefined, not zero: rejecting nothing is a refusal to act, and
        # printing 0.000 reads as "every reject was wrong"
        prec = tp / (tp + fp) if (tp + fp) else float("nan")
        lo, hi = wilson(tp, tp + fp) if tp + fp else (float("nan"),) * 2
        blo, bhi = grouped_bootstrap(rej, y.astype(bool), groups,
                                     a.n_boot, a.seed)
        print(f"\n  AUTO_REJECT n {int(rej.sum())}   correct {tp}   "
              f"wrong {fp}")
        print(f"  precision {prec:.3f}   Wilson [{lo:.3f}, {hi:.3f}]   "
              f"recording-grouped bootstrap [{blo:.3f}, {bhi:.3f}]")
        print(f"  one-error buffered {buffered(tp, fp):.3f}   "
              f"(target {MIN_PRECISION})")
        print(f"  review reduction {rej.sum() / len(sel):.1%} of this set   "
              f"recall over reject-safe {tp / max(1, int(y.sum())):.1%}")

        ok = (int(rej.sum()) >= MIN_N and prec >= MIN_PRECISION
              and buffered(tp, fp) >= MIN_PRECISION)
        if not rej.sum():
            print("  No fold found a cut meeting the buffered precision on "
                  "its own training recordings. That is the correct output,\n"
                  "  not a crash: the deployed score has no tail this target "
                  "can be read off at the required precision.")
        print(f"  PRE-REGISTERED BAR (n>={MIN_N}, precision>={MIN_PRECISION}, "
              f"buffered>={MIN_PRECISION}): {'MET' if ok else 'NOT MET'}")
        if ok:
            print("  The student alone already clears the bar on this set. A "
                  "teacher can only add cost unless it raises n\n  at the same "
                  "precision -- so the experiment to run is recall, not "
                  "precision.")

        # the deployed policy sends low-reliability events to REVIEW rather
        # than acting on them. A reject action is a NEW action and nothing has
        # decided whether that gate should apply to it too, so this arm leaves
        # the gate off and shows where the rejects landed instead -- a reject
        # set concentrated at reliability 0.000 is being produced by scores the
        # policy itself declines to trust
        if rej.sum():
            rel = np.array([r["reliability"] for r in sel], float)[rej]
            zero = int(np.sum(rel == 0.0))
            print(f"\n  reliability of the {int(rej.sum())} rejected: "
                  f"min {np.nanmin(rel):.3f}  median {np.nanmedian(rel):.3f}  "
                  f"at 0.000: {zero}")
            byr = Counter(r["reason"] for i, r in enumerate(sel) if rej[i])
            print(f"  the policy's own reason for those events: {dict(byr)}")
            if zero:
                print(f"  !! {zero} of the rejects carry reliability 0.000. "
                      f"The frozen policy routes those to REVIEW precisely "
                      f"because it\n     does not trust the score there, and "
                      f"this arm is deleting them on that same score.")

        if fp:
            print(f"\n  THE {fp} FALSE REJECTS, one by one. At this precision "
                  f"target they are events, not a rate:")
            for i, r in enumerate(sel):
                if not (rej[i] and not r["y"]):
                    continue
                g = gold.get(r["event_id"], {})
                print(f"    {r['event_id'][-46:]:<47} score {r['score']:.4f}")
                print(f"      subtype {r['subtype']}   truth "
                      f"{g.get('temporal_truth')}   validity "
                      f"{g.get('candidate_boundary_validity')}   "
                      f"corrected {g.get('primary_corrected_boundary_time')}")
                print(f"      excluded from reject-safe by: "
                      f"{safe[r['event_id']]['reason']}")
            print(f"\n  false rejects by subtype: "
                  f"{dict(Counter(r['subtype'] for i, r in enumerate(sel) if rej[i] and not r['y']))}")

        # ------------------------------------------------- what the tail holds
        # The nested selection above answers "is there a deployable cut". When
        # the answer is no, the next question is how far off it is and what is
        # standing in the way -- a tail that is 0.90 clean needs a teacher to
        # remove a couple of events, and one that is 0.60 clean needs the
        # teacher to do the whole job. Both look identical as "n 0".
        #
        # THIS IS POOLED AND THEREFORE OPTIMISTIC. Every cut below is read off
        # the same events it is scored on, which is exactly the selection this
        # file refuses to make for the operating point. It is a description of
        # the tail, not a candidate policy, and no threshold here may be
        # deployed without being re-chosen inside training folds.
        order = np.argsort(sc)
        print(f"\n  THE LOW TAIL, POOLED (optimistic by construction -- a "
              f"description of what is there, never an operating point):")
        print(f"    {'k':>4} {'cut':>8} {'safe':>6} {'wrong':>6} "
              f"{'precision':>10} {'buffered':>9}")
        for k in (5, 10, 15, 20, 25, 30, 35, 40, 50):
            if k > len(order):
                break
            idx = order[:k]
            t = int(y[idx].sum())
            fpk = k - t
            print(f"    {k:>4} {sc[order[k - 1]]:>8.4f} {t:>6} {fpk:>6} "
                  f"{t / k:>10.3f} {buffered(t, fpk):>9.3f}")
        best = max(((buffered(int(y[order[:k]].sum()),
                              k - int(y[order[:k]].sum())), k)
                    for k in range(MIN_N, len(order) + 1)), default=(0, 0))
        print(f"    best buffered precision at n>={MIN_N}, pooled: "
              f"{best[0]:.3f} at n={best[1]}   (target {MIN_PRECISION})")

        contam = [sel[i] for i in order[:40] if not sel[i]["y"]]
        print(f"\n  what contaminates the lowest-scoring 40: {len(contam)} of "
              f"40 are not safe to delete")
        for st, n in Counter(c["subtype"] for c in contam).most_common():
            ex = next(c for c in contam if c["subtype"] == st)
            print(f"    {n:>3}  {st:<32} e.g. {ex['event_id'][-40:]}")
        print("    This is the teacher's actual job on this branch: these are "
              "the events it has to recognise and refuse, and\n    the ones "
              "whose subtypes it has already been shown to handle badly are "
              "the reason to expect it to be hard.")

        if a.out and tag.startswith("HELD"):
            # one file per arm, so --all_arms does not silently leave only the
            # last arm's numbers behind the single name the caller gave
            base, ext = os.path.splitext(a.out)
            out = a.out if not a.all_arms else \
                f"{base}.{col.replace(' ', '_').replace(',', '')}{ext}"
            json.dump({"score_col": col, "n": len(sel),
                       "n_reject": int(rej.sum()), "tp": tp, "fp": fp,
                       "precision": prec, "buffered": buffered(tp, fp),
                       "wilson": [lo, hi], "bootstrap": [blo, bhi],
                       "auroc": float(_auroc(y, -sc)),
                       "bar_met": bool(ok),
                       "false_rejects": [sel[i]["event_id"]
                                         for i in range(len(sel))
                                         if rej[i] and not sel[i]["y"]]},
                      open(out, "w", encoding="utf-8"), indent=2)
            print(f"\nwrote {out}")


def closing():
    print(f"\n{'=' * 78}\nAUTO-REJECT, THE THREE ARMS\n{'=' * 78}")
    print(f"  {'arm':<30} {'n':>5} {'correct':>8} {'wrong':>6} {'precision':>10}")
    print(f"  {'student alone (above)':<30}   see the held-out block")
    print(f"  {'teacher certified':<30}     not run")
    print(f"  {'student + teacher':<30}     not run")
    print("  The teacher arms stay empty until this one is read. If the "
          "student already meets the bar, the only question worth an API\n  "
          "call is whether the teacher raises n at the same precision; if it "
          "does not, there is nothing left to test.")


if __name__ == "__main__":
    main()
