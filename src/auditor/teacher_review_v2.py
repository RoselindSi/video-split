"""Teacher reviewer v2: is the transition LOCALISED at the candidate?

v1 is untouched and stays the baseline. This is a different formulation, not a
tuned prompt, and the reason is in v1's own output.

On the 7 gradual_phase_transition events v1 saw, it admitted 5. The evidence
audit found nothing inconsistent -- the descriptions were specific and true --
and the adversarial pass produced the correct objection in four of them
("flows smoothly into rotating it, with no explicit pause", "at the exact
candidate moment the left hand remains in contact with the left faucet
handle"). It approved anyway. The model saw gradual and did not treat it as a
blocker.

v1's guideline invited that: it listed release, recontact and target switch as
evidence OF a sharp transition, and a gradual event contains every one of
them. What separates them is WHERE the change sits in time, and the single v1
case that reasoned about that explicitly got it right.

So four things change, and the model is not one of them:

  the question   H1 is a transition localised at the candidate within the
                 annotation tolerance, not "a change occurred"
  the schema     temporal localisation is asked for as its own fields --
                 pre/post stability, when the change begins and ends, whether
                 a unique transition point is visible, whether it falls within
                 +/-0.5 s -- and "sharp" is gone from the observation stage so
                 it cannot anchor everything after it
  the decision   admission is decided by a deterministic rule over those
                 fields. The model's own label is recorded and is not the
                 verdict, because in v1 it wrote the objection and approved
  the cost       one call per event; the safety pass runs only on candidates
                 that already passed the rule, and may only object by citing a
                 specific observable contradiction

H0 NEVER MEANS DELETE. The product decision is admit-or-show-a-human. Gradual,
annotation-convention, offscreen and ambiguous events all still need a person,
and auto-rejecting them would trade a real uncertainty for a better-looking
review rate.

THE SAMPLE IS THE SAME 32 EVENTS. Comparing a new prompt on new events cannot
say whether an improvement came from the prompt or from easier data, so the
sampling is reproduced from v1 and then VERIFIED against v1's result file when
one is given -- reproducing it and checking it are different, and only the
second catches a drift.

Usage:
    export ARK_API_KEY=...
    python -m src.auditor.teacher_review_v2 \
        --config configs/teacher_review_v2.json \
        --decisions .../policy_decisions_v4.primary_transportability_frontier.csv \
        --pair_labels data/gold/pair_labels_v1.csv \
        --pair_labels data/gold/batch3_pair_labels_v1_relabel_v1.csv \
        --data .../recseg_train.json --data .../recseg_val.json \
        --n_per_class 16 --v1_review .../teacher_review.json \
        --out /workspace/tr1/results/hal/c3/teacher_review_v2.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import Counter

import numpy as np

from src.boundary.pair_taxonomy import load_pair_labels
from src.auditor.teacher_review import event_time, frames_b64, parse_json

SHARP = "sharp_visible_transition"


def sample_same_as_v1(decisions, labels, n_per_class, seed):
    """Reproduce v1's sample exactly: the false keeps, plus the true keeps
    nearest in student score, shuffled with the same seed."""
    keeps = []
    with open(decisions, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("decision") != "AUTO_KEEP":
                continue
            sub = labels.get(r["event_id"])
            if sub is None:
                continue
            keeps.append({"event_id": r["event_id"],
                          "recording_id": r["recording_id"],
                          "subtype": sub, "correct": sub == SHARP,
                          "score": r.get("score") or r.get("fused_score") or ""})
    good = [k for k in keeps if k["correct"]]
    bad = [k for k in keeps if not k["correct"]]
    rng = random.Random(seed)
    n = min(n_per_class, len(bad))
    picked_bad = bad if len(bad) <= n else rng.sample(bad, n)
    fnum = lambda x: float(x) if x not in ("", None) else float("nan")
    pool = sorted(good, key=lambda g: fnum(g["score"]))
    picked_good, used = [], set()
    for b in picked_bad:
        cand = [g for g in pool if g["event_id"] not in used]
        if not cand:
            break
        j = int(np.argmin([abs(fnum(g["score"]) - fnum(b["score"]))
                           if np.isfinite(fnum(g["score"])) else 1e9
                           for g in cand]))
        picked_good.append(cand[j])
        used.add(cand[j]["event_id"])
    sample = [dict(x, arm="false_keep") for x in picked_bad] + \
             [dict(x, arm="true_keep") for x in picked_good]
    rng.shuffle(sample)
    return sample, keeps, good, bad


def eligible(b, rule):
    """The admission rule, applied in code.

    v1 demonstrated the need: the model wrote a correct objection and then
    approved. Generating an objection and letting it count are different
    things, so these fields decide and the model's own label is recorded
    beside them."""
    if not b:
        return False, ["no parseable response"]
    fail = []
    if b.get("decision") != rule["require_decision"]:
        fail.append(f"decision={b.get('decision')}")
    if rule["require_evidence_sufficient"] and not b.get("evidence_sufficient"):
        fail.append("evidence_sufficient=false")
    for k in rule["require_yes"]:
        if b.get(k) != "yes":
            fail.append(f"{k}={b.get(k)}")
    for k in rule["forbid_no"]:
        if b.get(k) == "no":
            fail.append(f"{k}=no")
    for k in rule["forbid_yes"]:
        if b.get(k) == "yes":
            fail.append(f"{k}=yes")
    for k in rule["forbid_insufficient"]:
        if b.get(k) == "insufficient":
            fail.append(f"{k}=insufficient")
    return (not fail), fail


def contradiction(b):
    """The model saying H1 while its own fields say the change is not
    localised. v1's whole failure mode, now countable."""
    if not b or b.get("decision") != "H1":
        return None
    for k in ("change_concentrated_near_candidate",
              "unique_transition_point_visible", "transition_within_tolerance"):
        if b.get(k) == "no":
            return k
    return None


def call(client, model, images, rel_t, instruction, temperature):
    content = []
    for url, dt in zip(images, rel_t):
        content.append({"type": "text", "text": f"[t{dt:+.1f}s]"})
        content.append({"type": "image_url", "image_url": {"url": url}})
    content.append({"type": "text", "text": instruction})
    r = client.chat.completions.create(
        model=model, temperature=temperature,
        messages=[{"role": "user", "content": content}])
    return r.choices[0].message.content


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/teacher_review_v2.json")
    ap.add_argument("--decisions", required=True)
    ap.add_argument("--pair_labels", action="append", required=True)
    ap.add_argument("--data", action="append", required=True)
    ap.add_argument("--n_per_class", type=int, default=16)
    ap.add_argument("--all_proposed", action="store_true",
                    help="review every AUTO_KEEP rather than the balanced 32; "
                         "the 32 are balanced by construction so their "
                         "precision is not a deployment precision")
    ap.add_argument("--v1_review", help="v1 result file, to verify the sample "
                                        "is identical")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=1,
                    help="review each event this many times. Two identical v1 "
                         "runs at temperature 0 disagreed on 9 of 32 events "
                         "while their aggregate counts barely moved, so a "
                         "single run cannot tell a 9/16 from a 12/16 and the "
                         "pre-registered threshold is not measurable without "
                         "this.")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cfg = json.load(open(a.config, encoding="utf-8"))
    key = os.environ.get("ARK_API_KEY")
    if not key and not a.dry_run:
        raise SystemExit("ARK_API_KEY is not set. It is read from the "
                         "environment only -- never a flag, which would land "
                         "in shell history and the process list.")

    labels = {}
    for p in a.pair_labels:
        for e, v in load_pair_labels(p).items():
            labels[e] = v["temporal_pair_subtype"]
    video = {}
    for p in a.data:
        for r in json.load(open(p, encoding="utf-8")):
            video[r["recording_id"]] = r["video"]

    sample, keeps, good, bad = sample_same_as_v1(
        a.decisions, labels, a.n_per_class, a.seed)
    print(f"student proposes {len(keeps)} admissions: {len(good)} correct, "
          f"{len(bad)} wrong")
    if a.all_proposed:
        sample = [dict(k, arm="false_keep" if not k["correct"] else "true_keep")
                  for k in keeps]
        print(f"  --all_proposed: reviewing all {len(sample)}, which IS the "
              f"deployment distribution")
    else:
        print(f"  balanced pilot: {sum(1 for s in sample if s['arm'] == 'false_keep')}"
              f" false + {sum(1 for s in sample if s['arm'] == 'true_keep')} "
              f"score-matched true keeps. Balanced by construction, so its "
              f"precision is NOT a deployment precision.")

    if a.v1_review and os.path.exists(a.v1_review):
        v1 = json.load(open(a.v1_review, encoding="utf-8"))
        v1_ids = {r["event_id"] for r in v1.get("results", [])}
        mine = {s["event_id"] for s in sample}
        if v1_ids and not a.all_proposed:
            miss, extra = v1_ids - mine, mine - v1_ids
            if miss or extra:
                raise SystemExit(
                    f"the sample does not match v1: {len(miss)} of v1's events "
                    f"missing, {len(extra)} new. A prompt compared on different "
                    f"events cannot say whether it improved or got easier data. "
                    f"e.g. {sorted(miss)[:2]} / {sorted(extra)[:2]}")
            print(f"  sample verified identical to v1 ({len(mine)} events)")

    if a.dry_run:
        print(f"\n--dry_run: no call made. prompt {len(cfg['prompt'])} chars, "
              f"schema {len(cfg['schema'])}, "
              f"{cfg['context_window_s']}s@{cfg['context_fps']}fps + "
              f"{cfg['transition_window_s']}s@{cfg['transition_fps']}fps")
        return

    from volcenginesdkarkruntime import Ark
    client = Ark(base_url=cfg["base_url"], api_key=key)
    n_ctx = int(round(2 * cfg["context_window_s"] * cfg["context_fps"])) + 1
    n_tr = int(round(2 * cfg["transition_window_s"] * cfg["transition_fps"])) + 1
    print(f"  {n_ctx} context frames + {n_tr} transition frames per event")

    results, n_safety = [], 0
    for i, s in enumerate(sample):
        vp, t = video.get(s["recording_id"]), event_time(s["event_id"])
        if vp is None or t is None:
            print(f"  !! {s['event_id']}: no video or timestamp")
            continue
        ctx, ct = frames_b64(vp, t, cfg["context_window_s"], n_ctx,
                             cfg["eye"], cfg["long_side"])
        tr, tt = frames_b64(vp, t, cfg["transition_window_s"], n_tr,
                            cfg["eye"], cfg["long_side"])
        imgs, rel = ctx + tr, ct + tt
        # repeated identically; the model is not reproducible at temperature 0
        draws = []
        for _ in range(max(1, a.repeats)):
            try:
                raw = call(client, cfg["model"], imgs, rel,
                           cfg["prompt"] + "\n\n" + cfg["schema"],
                           cfg["temperature"])
            except Exception as ex:
                print(f"  !! {s['event_id']}: {type(ex).__name__}: "
                      f"{str(ex)[:110]}")
                continue
            bb = parse_json(raw)
            draws.append({"review": bb,
                          "unparsed": None if bb else (raw or "")[:400],
                          "eligible": eligible(bb, cfg["eligibility"])[0],
                          "decision": (bb or {}).get("decision")})
        if not draws:
            continue
        # the FIRST draw carries forward, so a single-repeat run is unchanged;
        # the rest are kept to show how far the answer moves
        b = draws[0]["review"]
        raw = draws[0]["unparsed"]
        ok, why = eligible(b, cfg["eligibility"])
        n_el = sum(1 for d in draws if d["eligible"])
        rec = {**s, "review": b, "unparsed": draws[0]["unparsed"],
               "eligible": ok, "eligibility_failures": why,
               "contradiction": contradiction(b),
               "repeats": len(draws), "eligible_draws": n_el,
               # a draw whose reply did not parse has decision None, which is
               # not sortable against strings -- and it is information, not an
               # error: an unparseable answer is one the reviewer failed to
               # give, and it belongs in the spread rather than crashing it
               "decisions_seen": sorted({d["decision"] or "unparsed"
                                         for d in draws}),
               "n_unparsed": sum(1 for d in draws if d["decision"] is None),
               "stable": len(draws) == 1 or n_el in (0, len(draws)),
               # the FULL observation per draw, not just its verdict. Storing
               # decision and eligible alone answers "did it flip" and not
               # "which field flipped", and those point at different problems:
               # unstable object perception is a different failure from stable
               # perception with unstable temporal adjudication.
               "draws": [{"decision": d["decision"], "eligible": d["eligible"],
                          "review": d["review"]} for d in draws]}
        # the safety pass costs a call and only runs where one is warranted
        if ok and cfg["safety_pass"]["enabled"]:
            n_safety += 1
            try:
                sraw = call(client, cfg["model"], imgs, rel,
                            cfg["safety_pass"]["prompt"].format(
                                first=json.dumps(b, ensure_ascii=False)),
                            cfg["temperature"])
                rec["safety"] = parse_json(sraw)
                rec["safety_unparsed"] = None if rec["safety"] else (sraw or "")[:400]
            except Exception as ex:
                print(f"  !! {s['event_id']}: safety {type(ex).__name__}")
                rec["safety"] = None
        rec["route"] = (
            cfg["routing"]["H1_eligible_approved"]
            if ok and (rec.get("safety") or {}).get("final", "approve") == "approve"
            else cfg["routing"]["H1_eligible_challenged"] if ok
            else cfg["routing"].get(
                (b or {}).get("decision", "insufficient_evidence"),
                "requires_human_review"))
        results.append(rec)
        d = (b or {}).get("decision", "?")
        print(f"  [{i+1}/{len(sample)}] {s['event_id'][:42]:<42} "
              f"{s['arm']:<11} {d:<20} eligible={str(ok):<5} "
              f"{rec['route']}", flush=True)

    # ------------------------------------------------------------- report
    fk = [r for r in results if r["arm"] == "false_keep"]
    tk = [r for r in results if r["arm"] == "true_keep"]
    adm = lambda r: r["route"] == cfg["routing"]["H1_eligible_approved"]
    n_ch_bad = sum(1 for r in fk if not adm(r))
    n_ap_good = sum(1 for r in tk if adm(r))
    grad = [r for r in results if r["subtype"] == "gradual_phase_transition"]
    n_grad = sum(1 for r in grad if not adm(r))
    insuf = sum(1 for r in results
                if (r.get("review") or {}).get("decision") == "insufficient_evidence")
    contra = [r for r in results if r["contradiction"]]

    print(f"\n{'=' * 74}\nRESULTS\n{'=' * 74}")
    print(f"  false keeps blocked      {n_ch_bad}/{len(fk)}")
    print(f"  true keeps retained      {n_ap_good}/{len(tk)}")
    print(f"  gradual blocked          {n_grad}/{len(grad)}"
          f"   <- v1 blocked 2/7, the largest single error class")
    print(f"  insufficient_evidence    {insuf}/{len(results)}")
    print(f"  passed the rule          {sum(1 for r in results if r['eligible'])}"
          f"   (safety pass ran {n_safety} times, "
          f"{len(results) + n_safety} calls for {len(results)} events)")
    print(f"  evidence/decision contradictions  {len(contra)}"
          f"   <- said H1 while its own fields said the change is not localised")
    for r in contra:
        print(f"      {r['event_id'][:46]:<46} {r['contradiction']}=no")

    over = [r for r in results if r["eligible"]
            and (r.get("safety") or {}).get("final") == "challenge"]
    print(f"\n  safety pass overturned {len(over)} of {n_safety}: "
          f"{sum(1 for r in over if r['arm'] == 'true_keep')} were real boundaries "
          f"(cost), {sum(1 for r in over if r['arm'] == 'false_keep')} were "
          f"wrong admissions (benefit)")

    if a.repeats > 1:
        unstable = [r for r in results if not r["stable"]]
        print(f"\n{'=' * 74}\nREPRODUCIBILITY over {a.repeats} identical "
              f"calls\n{'=' * 74}")
        print(f"  events whose admission flipped between draws: "
              f"{len(unstable)}/{len(results)}")
        for r in unstable:
            print(f"    {r['event_id'][:46]:<46} {r['arm']:<11} "
                  f"eligible in {r['eligible_draws']}/{r['repeats']} draws, "
                  f"decisions {r['decisions_seen']}"
                  + (f", {r['n_unparsed']} unparsed" if r.get("n_unparsed")
                     else ""))
        print("  An aggregate count can look steady while individual events "
              "move underneath it -- two v1 runs both scored 9/16 on the false "
              "keeps while\n  disagreeing about 9 of 32 events. A threshold of "
              "12/16 cannot be read off one draw when the draw itself has this "
              "much spread.")

    print(f"\n  {'route':<28} {'n':>4}   by arm")
    for rt in sorted({r["route"] for r in results}):
        g = [r for r in results if r["route"] == rt]
        print(f"  {rt:<28} {len(g):>4}   "
              f"false={sum(1 for r in g if r['arm'] == 'false_keep')} "
              f"true={sum(1 for r in g if r['arm'] == 'true_keep')}")

    print(f"\n  {'subtype':<32} {'n':>3} {'admitted':>9} {'to human':>9}")
    for s in sorted({r["subtype"] for r in results}):
        g = [r for r in results if r["subtype"] == s]
        na = sum(1 for r in g if adm(r))
        print(f"  {s:<32} {len(g):>3} {na:>9} {len(g) - na:>9}"
              + ("   <- should be admitted" if s == SHARP else ""))

    if not a.all_proposed:
        print(f"\n  These 32 are balanced by construction. The deployment "
              f"number comes from --all_proposed on all {len(keeps)} proposals, "
              f"and only there\n  do precision and the one-error buffer mean "
              f"anything.")

    blob = {"config": os.path.abspath(a.config), "model": cfg["model"],
            "n": len(results), "n_calls": len(results) + n_safety,
            "false_keeps_blocked": [n_ch_bad, len(fk)],
            "true_keeps_retained": [n_ap_good, len(tk)],
            "gradual_blocked": [n_grad, len(grad)],
            "contradictions": [r["event_id"] for r in contra],
            "results": results}
    txt = json.dumps(blob, ensure_ascii=False, indent=2, default=str)
    if key and key in txt:
        raise SystemExit("refusing to write: the API key appears in the output")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(txt)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
