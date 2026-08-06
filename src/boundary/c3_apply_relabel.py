"""Turn a filled relabel sheet into a new pair-label file.

Different from c3_apply_label_corrections, and the difference matters. That
one applies a handful of flips two annotators agreed on, within the clean
binary, leaving the population fixed. This one replaces subtypes outright, and
a replacement can move an event INTO or OUT OF the clean set -- which is the
point. The first 40 rows of the batch3 relabel moved 15 of 40 events across
that boundary while changing only 3 labels inside it. The error being repaired
is mostly about which events belong in the training population, not about how
the ones that belong are signed.

WHAT THIS FILE REFUSES TO DO IS PRETEND A PARTIAL SHEET IS A WHOLE ONE. Rows
left blank keep their existing label, and the count of them is printed at the
top and again at the bottom, because a half-filled sheet silently merged into
a label set produces a file that looks complete and is half old and half new
with nothing recording which is which. Every row this touches is listed in an
audit trail with both values.

The output is written BESIDE the original. Every number this project has
reported was computed against the original, and overwriting it would make
those numbers unreproducible while looking like an improvement.

Usage:
    python -m src.boundary.c3_apply_relabel \
        --pair_labels data/gold/batch3_pair_labels_v1.csv \
        --relabel data/gold/batch3_relabel_claude_partial40.csv \
        --suffix _relabel_v1 --out_dir data/gold
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import Counter

from src.boundary.pair_taxonomy import (
    SUBTYPES, SUBTYPE_TO_SUPERVISION, CLEAN_BINARY,
)
from src.boundary.c3_annotator_agreement import read_csv


def load_relabel(path):
    rows = read_csv(path)
    if not rows:
        raise SystemExit(f"{path} is empty")
    c_sub = next((k for k in rows[0] if k.startswith("subtype")), None)
    c_conf = next((k for k in rows[0] if k.startswith("confidence")), None)
    c_why = next((k for k in rows[0] if k.startswith("why")), None)
    if not c_sub:
        raise SystemExit(f"{path}: no subtype column")
    out, blank, bad = {}, 0, []
    for r in rows:
        v = (r.get(c_sub) or "").strip()
        if not v:
            blank += 1
            continue
        if v not in SUBTYPES:
            bad.append((r["event_id"], v))
            continue
        out[r["event_id"]] = {
            "subtype": v,
            "conf": (r.get(c_conf) or "").strip() if c_conf else "",
            "why": (r.get(c_why) or "").strip() if c_why else "",
        }
    if bad:
        raise SystemExit(f"{len(bad)} rows carry a subtype outside the "
                         f"vocabulary, e.g. {bad[:3]}. Fix the sheet rather "
                         f"than letting an unrecognised value fall through as "
                         f"'no change'.")
    return out, blank, len(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair_labels", required=True)
    ap.add_argument("--relabel", action="append", required=True)
    ap.add_argument("--suffix", default="_relabel_v1")
    ap.add_argument("--out_dir", default="data/gold")
    ap.add_argument("--require_complete", action="store_true",
                    help="refuse to write while any sheet row is blank")
    a = ap.parse_args()

    new, blank, total = {}, 0, 0
    for p in a.relabel:
        n, b, t = load_relabel(p)
        dup = set(n) & set(new)
        if dup:
            print(f"  !! {len(dup)} event(s) appear in more than one sheet; "
                  f"the later file wins: {sorted(dup)[:3]}")
        new.update(n)
        blank += b
        total += t
    print(f"{len(new)} filled of {total} sheet rows ({blank} blank)")
    if blank:
        msg = (f"{blank} rows are still blank and will KEEP their existing "
               f"label. The output will be part old, part new.")
        if a.require_complete:
            raise SystemExit(msg + " --require_complete was given, so nothing "
                                   "was written.")
        print(f"  !! {msg}")

    rows = read_csv(a.pair_labels)
    if not rows or "temporal_pair_subtype" not in rows[0]:
        raise SystemExit(f"{a.pair_labels}: no temporal_pair_subtype column")
    cols = list(rows[0])
    stored = {r["event_id"]: (r.get("temporal_pair_subtype") or "").strip()
              for r in rows}
    absent = set(new) - set(stored)
    if absent:
        print(f"  !! {len(absent)} relabelled events have no row in "
              f"{a.pair_labels} and were skipped: {sorted(absent)[:3]}")

    inclean = lambda s: SUBTYPE_TO_SUPERVISION.get(s, "") in CLEAN_BINARY
    changed, entered, left, flipped = [], [], [], []
    for r in rows:
        e = r["event_id"]
        v = new.get(e)
        if not v or v["subtype"] == stored[e]:
            continue
        old = stored[e]
        r["temporal_pair_subtype"] = v["subtype"]
        if "pair_supervision" in cols:
            r["pair_supervision"] = SUBTYPE_TO_SUPERVISION[v["subtype"]]
        if "notes" in cols:
            r["notes"] = ((r.get("notes") or "") + " | relabelled").strip(" |")
        changed.append((e, old, v["subtype"], v["conf"], v["why"]))
        if inclean(old) and not inclean(v["subtype"]):
            left.append((e, old, v["subtype"]))
        elif not inclean(old) and inclean(v["subtype"]):
            entered.append((e, old, v["subtype"]))
        elif inclean(old) and inclean(v["subtype"]):
            flipped.append((e, old, v["subtype"]))

    kept = len(new) - len(changed) - len(absent & set(new))
    print(f"\n  {kept} relabelled rows confirm the existing subtype, "
          f"{len(changed)} change it")
    print(f"  of the changes: {len(flipped)} stay inside the clean binary and "
          f"flip sign, {len(left)} leave the clean set, {len(entered)} enter it")
    for tag, lst in (("LEAVES", left), ("ENTERS", entered), ("FLIPS", flipped)):
        for e, o, n_ in lst:
            print(f"    {tag:<7} {e:<48} {o} -> {n_}")

    # THE COMPOSITION, not just the counts. The batch3 held-out failure was
    # diagnosed as a base-rate shift (0.310 vs 0.745), so a relabel that moves
    # events in and out of the clean set changes the very quantity that
    # diagnosis rested on, and it has to be visible before anything is rerun.
    def stats(get):
        c = [get(r) for r in rows]
        cl = [s for s in c if inclean(s)]
        pos = sum(1 for s in cl if SUBTYPE_TO_SUPERVISION[s] == "strong_separate")
        return len(cl), pos, (pos / len(cl) if cl else float("nan"))
    n0, p0, r0 = stats(lambda r: stored[r["event_id"]])
    n1, p1, r1 = stats(lambda r: (r.get("temporal_pair_subtype") or "").strip())
    print(f"\n  clean set: {n0} events, {p0} positive, base rate {r0:.3f}")
    print(f"          -> {n1} events, {p1} positive, base rate {r1:.3f}")
    print(f"  subtypes before: "
          f"{dict(Counter(stored[r['event_id']] for r in rows))}")
    print(f"  subtypes after:  "
          f"{dict(Counter((r.get('temporal_pair_subtype') or '').strip() for r in rows))}")

    if not changed:
        print("\nnothing changed; no file written")
        return

    os.makedirs(a.out_dir, exist_ok=True)
    out = os.path.join(a.out_dir, os.path.basename(a.pair_labels).replace(
        ".csv", f"{a.suffix}.csv"))
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        w.writerows(rows)
    trail = os.path.join(a.out_dir, f"relabel_trail{a.suffix}.csv")
    with open(trail, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["event_id", "old_subtype", "new_subtype", "old_supervision",
                    "new_supervision", "clean_membership_change", "confidence",
                    "why"])
        for e, o, n_, conf, why in changed:
            mv = ("left" if inclean(o) and not inclean(n_) else
                  "entered" if not inclean(o) and inclean(n_) else
                  "flipped" if inclean(o) else "outside")
            w.writerow([e, o, n_, SUBTYPE_TO_SUPERVISION[o],
                        SUBTYPE_TO_SUPERVISION[n_], mv, conf, why])
    print(f"\nwrote {out}\n      {trail}")
    if blank:
        print(f"\n!! {blank} sheet rows were blank. This file is part old, "
              f"part new, and the trail names only the rows that moved -- so a "
              f"row absent from the trail is either confirmed or never "
              f"reviewed, and nothing here distinguishes them. Do not read a "
              f"downstream result from this as the relabel's full effect.")


if __name__ == "__main__":
    main()
