"""Score visual-auditor output against the frozen Gold v2 labels, field by
field. This is the actual experiment the mentor asked for first: *not* whether
the auditor is generally accurate, but WHERE it is trustworthy -- because we
will only auto-act on the field/slice combinations where it reproduces human
judgment reliably, and route the rest to human review.

Reports, per the auditor design:
  - per-field accuracy + confusion matrix for every categorical head
  - candidate boundary validity (valid vs spurious) accuracy
  - semantic "label is wrong" detection precision / recall / F1
  - granularity classification accuracy
  - corrected primary-verb top-1 (exact + soft) accuracy
  - corrected boundary-time error (MAE, within 0.5s / 1.0s)
  - exclude / auto-proposal decision accuracy
  - three hard-case slices that are the whole reason for the audit:
      (1) true fast boundaries must NOT be dismissed as internal motion
      (2) repetitive/reversing motion must NOT be called a boundary
      (3) correct-but-coarse / compound labels must NOT be flagged "incorrect"
  - schema-violation counts (auditor emitted an out-of-vocabulary value)

Usage:
    python -m src.auditor.eval_auditor --pred /tmp/auditor_pred.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict

from . import gold_schema as S


def _norm(v):
    return None if v is None else str(v).strip().lower()


def load_pred(path):
    with open(path, encoding="utf-8") as f:
        return {json.loads(l)["event_id"]: json.loads(l) for l in f if l.strip()}


def per_field_accuracy(gold_rows, pred, field):
    """Accuracy over events where gold has a value; also count auditor schema
    violations (out-of-vocab) and misses (no prediction)."""
    n = correct = viol = miss = 0
    confusion = defaultdict(Counter)  # gold -> Counter(pred)
    for g in gold_rows:
        gv = _norm(g.get(field))
        if gv is None:
            continue
        n += 1
        p = pred.get(g["event_id"], {})
        pv = _norm(p.get(field))
        if pv is None:
            miss += 1
            confusion[gv]["<none>"] += 1
            continue
        if field in S.ENUM_FIELDS and pv not in S.ENUM_FIELDS[field]:
            viol += 1
            confusion[gv]["<invalid>"] += 1
            continue
        confusion[gv][pv] += 1
        if pv == gv:
            correct += 1
    return {"n": n, "acc": correct / n if n else 0.0, "correct": correct,
            "schema_violations": viol, "missing": miss, "confusion": confusion}


def binary_prf(gold_rows, pred, is_pos_gold, is_pos_pred, label):
    tp = fp = fn = tn = 0
    for g in gold_rows:
        p = pred.get(g["event_id"], {})
        gy, py = is_pos_gold(g), is_pos_pred(p)
        tp += gy and py
        fp += (not gy) and py
        fn += gy and (not py)
        tn += (not gy) and (not py)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"label": label, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": prec, "recall": rec, "f1": f1,
            "n_pos_gold": tp + fn}


_TOK = re.compile(r"[a-z]+")


def _verb_tokens(s):
    return set(_TOK.findall(s.lower())) if s else set()


def verb_match(gold_rows, pred):
    exact = soft = n = 0
    for g in gold_rows:
        gv = g.get("corrected_primary_verb")
        if not gv:
            continue
        n += 1
        pv = pred.get(g["event_id"], {}).get("corrected_primary_verb")
        if not pv:
            continue
        if _norm(pv) == _norm(gv):
            exact += 1
            soft += 1
        elif _verb_tokens(pv) & _verb_tokens(gv):
            soft += 1
    return {"n": n, "top1_exact": exact / n if n else 0.0,
            "top1_soft": soft / n if n else 0.0}


def time_error(gold_rows, pred):
    errs = []
    for g in gold_rows:
        gt = g.get("primary_corrected_boundary_time")
        if gt is None:
            continue
        pt = pred.get(g["event_id"], {}).get("primary_corrected_boundary_time")
        if pt is None:
            continue
        try:
            errs.append(abs(float(pt) - float(gt)))
        except (TypeError, ValueError):
            pass
    if not errs:
        return {"n": 0}
    errs.sort()
    return {"n": len(errs), "mae": sum(errs) / len(errs),
            "median": errs[len(errs) // 2],
            "within_0.5s": sum(e <= 0.5 for e in errs) / len(errs),
            "within_1.0s": sum(e <= 1.0 for e in errs) / len(errs)}


def _confidence_bucket(overall):
    """Match the high/medium/low thresholds run_visual_auditor.py actually
    uses to set auto_proposal_eligible (0.8 / 0.5), so this reads as
    'would auto-acting on this confidence tier have been safe'."""
    if overall is None:
        return "unknown"
    if overall >= 0.8:
        return "high(>=0.8)"
    if overall >= 0.5:
        return "medium(0.5-0.8)"
    return "low(<0.5)"


_CONF_BUCKET_ORDER = ["high(>=0.8)", "medium(0.5-0.8)", "low(<0.5)", "unknown"]


def calibration_buckets(rows, pred, is_correct):
    """rows: subset of gold to evaluate over. is_correct(gold_row, pred_row)
    -> bool. Buckets by the auditor's OWN consistency-based `_confidence`
    score (NOT the review_confidence enum, which is graded elsewhere against
    the human's own separately-written confidence string and is a different,
    much weaker check -- see the note printed above this section)."""
    buckets = defaultdict(lambda: [0, 0])
    for g in rows:
        p = pred.get(g["event_id"])
        if p is None:
            continue
        conf = (p.get("_confidence") or {}).get("overall")
        b = _confidence_bucket(conf)
        buckets[b][1] += 1
        buckets[b][0] += int(bool(is_correct(g, p)))
    return buckets


def _print_calibration(title, buckets):
    print(f"  {title}:")
    any_row = False
    for b in _CONF_BUCKET_ORDER:
        if b in buckets:
            c, n = buckets[b]
            print(f"    {b:<16} acc={c / n:.3f}  (n={n})")
            any_row = True
    if not any_row:
        print("    (no rows)")


def _print_confusion(conf, title):
    print(f"    {title} (gold -> pred):")
    for gv in sorted(conf):
        row = ", ".join(f"{pv}:{c}" for pv, c in conf[gv].most_common())
        print(f"      {gv:<22} -> {row}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True, help="auditor_pred.jsonl from run_visual_auditor")
    ap.add_argument("--gold", help="gold jsonl (default: committed data/gold/...)")
    ap.add_argument("--out", help="write JSON summary here (default: <pred>.eval.json)")
    ap.add_argument("--show_confusion", action="store_true")
    a = ap.parse_args()

    gold_path, _ = S.default_gold_paths()
    gold_path = a.gold or gold_path
    gold = S.load_gold(gold_path)
    pred = load_pred(a.pred)

    try:
        from src.eval.run_manifest import print_manifest_if_exists
        print_manifest_if_exists(a.pred)
    except Exception:
        pass

    covered = sum(1 for g in gold if g["event_id"] in pred)
    print(f"\n=== Visual auditor vs Gold v2 ({covered}/{len(gold)} events predicted) ===\n")

    summary = {"n_gold": len(gold), "n_predicted": covered, "fields": {}}

    print("-- per-field accuracy --")
    for field in S.ENUM_FIELDS:
        r = per_field_accuracy(gold, pred, field)
        summary["fields"][field] = {k: v for k, v in r.items() if k != "confusion"}
        flag = ""
        if r["schema_violations"]:
            flag += f"  !{r['schema_violations']} schema-viol"
        if r["missing"]:
            flag += f"  !{r['missing']} missing"
        print(f"  {field:<28} acc={r['acc']:.3f}  (n={r['n']}, correct={r['correct']}){flag}")
        if a.show_confusion:
            _print_confusion(r["confusion"], field)

    # --- derived / decision metrics ---------------------------------------
    print("\n-- boundary validity (valid vs spurious) --")
    valid_rows = [g for g in gold if _norm(g.get("candidate_boundary_validity")) in ("valid", "invalid")]
    bv = binary_prf(
        valid_rows, pred,
        is_pos_gold=lambda g: _norm(g.get("candidate_boundary_validity")) == "valid",
        is_pos_pred=lambda p: _norm(p.get("candidate_boundary_validity")) == "valid",
        label="candidate_boundary_valid")
    acc = (bv["tp"] + bv["tn"]) / max(1, len(valid_rows))
    print(f"  binary valid-vs-invalid accuracy={acc:.3f} on {len(valid_rows)} decisive rows; "
          f"P={bv['precision']:.3f} R={bv['recall']:.3f} F1={bv['f1']:.3f}")
    summary["boundary_validity_binary"] = {**bv, "accuracy": acc, "n": len(valid_rows)}

    print("\n-- 'label is wrong' detection --")
    def gold_label_wrong(g):
        return (_norm(g.get("label_support")) == "contradicted"
                or _norm(g.get("label_completeness")) in ("incorrect", "wrong_object"))
    def pred_label_wrong(p):
        return (_norm(p.get("label_support")) == "contradicted"
                or _norm(p.get("label_completeness")) in ("incorrect", "wrong_object"))
    lw = binary_prf(gold, pred, gold_label_wrong, pred_label_wrong, "label_wrong")
    print(f"  P={lw['precision']:.3f} R={lw['recall']:.3f} F1={lw['f1']:.3f} "
          f"(gold positives={lw['n_pos_gold']}, tp={lw['tp']} fp={lw['fp']} fn={lw['fn']})")
    summary["label_wrong_detection"] = lw

    print("\n-- corrected primary verb --")
    vm = verb_match(gold, pred)
    print(f"  top1_exact={vm['top1_exact']:.3f}  top1_soft(token-overlap)={vm['top1_soft']:.3f}  (n={vm['n']})")
    print("  note: top-3 not measurable in MVP (auditor emits one verb, not a ranked list)")
    summary["corrected_verb"] = vm

    print("\n-- corrected boundary time --")
    te = time_error(gold, pred)
    if te["n"]:
        print(f"  n={te['n']} MAE={te['mae']:.2f}s median={te['median']:.2f}s "
              f"within0.5s={te['within_0.5s']:.3f} within1.0s={te['within_1.0s']:.3f}")
    else:
        print("  n=0 (no comparable corrected times)")
    summary["corrected_time"] = te

    print("\n-- decisions --")
    excl = binary_prf(
        gold, pred,
        is_pos_gold=lambda g: _norm(g.get("boundary_contrastive_role")) == "exclude",
        is_pos_pred=lambda p: _norm(p.get("boundary_contrastive_role")) == "exclude",
        label="boundary_exclude")
    auto_correct = auto_n = 0
    for g in gold:
        p = pred.get(g["event_id"])
        if p is None:
            continue
        auto_n += 1
        auto_correct += bool(g.get("auto_proposal_eligible")) == bool(p.get("auto_proposal_eligible"))
    print(f"  boundary 'exclude' detection: P={excl['precision']:.3f} R={excl['recall']:.3f} "
          f"F1={excl['f1']:.3f} (gold excludes={excl['n_pos_gold']})")
    print(f"  auto_proposal_eligible agreement: {auto_correct}/{auto_n} = "
          f"{auto_correct / auto_n if auto_n else 0:.3f}")
    summary["boundary_exclude"] = excl
    summary["auto_proposal_agreement"] = {"correct": auto_correct, "n": auto_n}

    # --- the three hard-case slices ---------------------------------------
    print("\n-- hard-case slices (the point of the audit) --")

    s1 = [g for g in gold if _norm(g.get("boundary_contrastive_role")) == "positive"]
    s1_ok = sum(1 for g in s1 if _norm(pred.get(g["event_id"], {}).get("temporal_truth")) == "valid")
    print(f"  (1) true boundaries kept as 'valid': {s1_ok}/{len(s1)} = "
          f"{s1_ok / len(s1) if s1 else 0:.3f}   [want high -- don't dismiss fast real actions]")

    s2 = [g for g in gold if _norm(g.get("boundary_contrastive_role")) == "motion_hard_negative"]
    s2_ok = sum(1 for g in s2 if _norm(pred.get(g["event_id"], {}).get("temporal_truth")) in ("spurious",))
    print(f"  (2) motion-hard-negatives called 'spurious': {s2_ok}/{len(s2)} = "
          f"{s2_ok / len(s2) if s2 else 0:.3f}   [want high -- don't call repetitive motion a boundary]")

    s3 = [g for g in gold if (_norm(g.get("label_granularity")) == "too_coarse"
                              or _norm(g.get("label_completeness")) == "missing_secondary")]
    s3_bad = sum(1 for g in s3 if pred_label_wrong(pred.get(g["event_id"], {})))
    print(f"  (3) correct-but-coarse/compound wrongly flagged 'incorrect': {s3_bad}/{len(s3)} = "
          f"{s3_bad / len(s3) if s3 else 0:.3f}   [want LOW -- coarse != wrong]")

    summary["hard_slices"] = {
        "true_boundary_kept_valid": {"ok": s1_ok, "n": len(s1)},
        "motion_neg_called_spurious": {"ok": s2_ok, "n": len(s2)},
        "coarse_wrongly_flagged_incorrect": {"bad": s3_bad, "n": len(s3)},
    }

    # --- confidence calibration ---------------------------------------
    # This is DIFFERENT from the `review_confidence` row under per-field
    # accuracy above: that row compares two independently-written confidence
    # STRINGS (the human's own self-rated confidence in their gold judgment,
    # vs the auditor's self-rated bucket) and can score high by coincidence
    # if both happen to say "high" most of the time. It says nothing about
    # whether the auditor's confidence tracks whether it was actually RIGHT.
    # This section does that: buckets by the auditor's own consistency score
    # (`_confidence.overall`, from repeats + blind/conditioned agreement) and
    # reports accuracy against gold per bucket. If accuracy climbs sharply in
    # the high bucket, that subset may be safe to auto-act on even when the
    # overall field accuracy is not.
    print("\n-- confidence calibration (does self-consistency track correctness?) --")
    print("  NOT the same check as 'review_confidence' above -- see code comment.")

    def _temporal_ok(g, p):
        gv = _norm(g.get("temporal_truth"))
        return gv is not None and _norm(p.get("temporal_truth")) == gv

    def _label_support_ok(g, p):
        gv = _norm(g.get("label_support"))
        return gv is not None and _norm(p.get("label_support")) == gv

    def _motion_neg_ok(g, p):
        return _norm(p.get("temporal_truth")) == "spurious"

    cal_temporal = calibration_buckets(gold, pred, _temporal_ok)
    cal_label = calibration_buckets(gold, pred, _label_support_ok)
    cal_motion_neg = calibration_buckets(s2, pred, _motion_neg_ok)
    _print_calibration("temporal_truth accuracy by confidence bucket", cal_temporal)
    _print_calibration("label_support accuracy by confidence bucket", cal_label)
    _print_calibration("hard-slice(2) motion_hard_negative->spurious, by confidence bucket", cal_motion_neg)
    summary["confidence_calibration"] = {
        "temporal_truth": dict(cal_temporal),
        "label_support": dict(cal_label),
        "motion_neg_called_spurious": dict(cal_motion_neg),
    }

    out = a.out or (a.pred + ".eval.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=lambda o: dict(o) if isinstance(o, Counter) else str(o))
    print(f"\nwrote eval summary -> {out}")


if __name__ == "__main__":
    main()
