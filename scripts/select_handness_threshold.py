#!/usr/bin/env python3
"""Freeze a Handness threshold on human validation rows and audit test rows."""
from __future__ import annotations

import argparse
import csv
import json
import math

from src.semhand.handness_head import binary_auc


def human_rows(path, exclude_recs=None, only_recs=None):
    exclude_recs = set(exclude_recs or ())
    only_recs = set(only_recs or ())
    rows = []
    with open(path, encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row.get("supervision") != "human" or row.get("target", "") == "":
                continue
            rec = row.get("rec", "")
            if rec in exclude_recs or (only_recs and rec not in only_recs):
                continue
            rows.append((float(row["p_hand"]), int(float(row["target"]) >= .5)))
    return rows


def select_threshold(rows, minimum_recall=.995):
    positives = sorted(probability for probability, target in rows if target)
    if not positives:
        raise ValueError("validation predictions contain no human positives")
    allowed = math.floor(len(positives) * (1 - minimum_recall) + 1e-12)
    allowed = min(allowed, len(positives) - 1)
    return positives[allowed], allowed


def metrics(rows, threshold):
    positive = sum(target for _, target in rows)
    negative = len(rows) - positive
    tp = sum(target and probability >= threshold for probability, target in rows)
    fn = positive - tp
    tn = sum(not target and probability < threshold for probability, target in rows)
    fp = negative - tn
    return {
        "n": len(rows), "positive": positive, "negative": negative,
        "human_auc": binary_auc([target for _, target in rows],
                                [probability for probability, _ in rows]),
        "tp": tp, "fn": fn, "tn": tn, "fp": fp,
        "recall": tp / positive if positive else None,
        "specificity": tn / negative if negative else None,
    }


def build_report(validation_path, test_path, minimum_recall=.995,
                 heldout_recs=None):
    heldout_recs = set(heldout_recs or ())
    validation = human_rows(validation_path, exclude_recs=heldout_recs)
    threshold, allowed = select_threshold(validation, minimum_recall)
    result = {
        "threshold": threshold, "minimum_validation_recall": minimum_recall,
        "allowed_validation_false_negatives": allowed,
        "splits": {"val": metrics(validation, threshold)},
    }
    if heldout_recs:
        result["heldout_recordings"] = sorted(heldout_recs)
        result["splits"]["val_all"] = metrics(
            human_rows(validation_path), threshold)
        result["splits"]["heldout"] = metrics(
            human_rows(validation_path, only_recs=heldout_recs), threshold)
    if test_path:
        result["splits"]["test"] = metrics(human_rows(test_path), threshold)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--test")
    parser.add_argument("--minimum-recall", type=float, default=.995)
    parser.add_argument("--heldout-rec", action="append", default=[])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = build_report(args.validation, args.test, args.minimum_recall,
                          args.heldout_rec)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
