"""Score the distilled student on the held-out zone batches, against V1 deployed.

Runs in `/workspace/venv_rig`. The ship rule is the one committed with
`distil_train.py` before any student was trained:

    on the held-out batches, through V1's own post-processing
    (1) M2  foreign-hand frames called self      <= V1 deployed's
    (2) M1  own-hand flips per 100 pairs         <= V1 deployed's
    (3) G   own-hand frames called self          >= V1 deployed's - GUARD
    (4) at least one of M1, M2 strictly better with a 95% recording-bootstrap
        interval of the difference that excludes zero

THE LABELS ARE ZONE-DERIVED, NOT TRACK GOLD. A drawn boundary plus the
render's own lookup table decides each track; on fresh29 that rule agreed with
human track gold on 99.0% of tracks, and tracks whose mean sits in the
ambiguous band are dropped rather than guessed. It is a coarser reference than
`own_gold`, and the foreign-hand count is what it is: these recordings were
drawn blind, so most of them are one person working alone.

BATCHES ARE POOLED AND ALSO PRINTED APART, since they differ in how far they
sit from V1's training days.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os

import numpy as np

from src.semhand import GUARD, STRIDE
from src.semhand.evaluate import boot_diff, per_recording, summarise
from src.semhand.holdreplay import invert_prior, key, replay


def read_zone_gold(path):
    g = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if r.get("label") in ("0", "1"):
            g[(r["rec"], str(r["tid"]))] = "owner" if r["label"] == "1" else "other"
    return g


def read_starts(clips):
    out = {}
    for line in open(clips):
        line = line.strip()
        if line and not line.startswith("#"):
            bag, s, _e = line.rsplit(":", 2)
            out[os.path.basename(bag.rstrip("/")).replace("databag-26_", "R")] = int(s)
    return out


def one_batch(dump_path, labels_path, clips_path, pred_path, strata_path):
    dump = list(csv.DictReader(open(dump_path, encoding="utf-8")))
    prior = invert_prior(dump)
    by_key = {key(r): r for r in dump}
    gold = read_zone_gold(labels_path)
    starts = read_starts(clips_path)
    preds = {}
    if pred_path and os.path.exists(pred_path):
        for r in csv.DictReader(open(pred_path)):
            for arm, v in r.items():
                if arm != "id" and v != "":
                    preds.setdefault(arm, {})[r["id"]] = float(v)
    ident = lambda r: f"{r['rec']}|{r['frame']}|{r['tid']}"
    rows = [r for r in dump
            if (r["rec"], str(r["tid"])) in gold
            and (int(r["frame"]) - starts[r["rec"]]) % STRIDE == 0
            and all(ident(r) in p for p in preds.values())]
    st = {}
    if strata_path and os.path.exists(strata_path):
        for line in open(strata_path):
            p = line.split()
            if len(p) >= 4:
                st[p[0].replace("databag-26_", "R")] = p[3]
    labels = {"V1 deployed": {key(r): by_key[key(r)]["final_owner_post_cap"] == "1" for r in rows}}
    p_v1 = {key(r): float(by_key[key(r)]["p_owner_raw"]) for r in rows}
    labels["V1 raw"] = {k: v >= 0.5 for k, v in p_v1.items()}
    labels["V1 held"] = replay(rows, p_v1, prior)
    for arm, p in preds.items():
        q = {key(r): p[ident(r)] for r in rows}
        labels[f"{arm} raw"] = {k: v >= 0.5 for k, v in q.items()}
        labels[f"{arm} held"] = replay(rows, q, prior)
    return rows, gold, labels, st


def table(rows, gold, labels, recs, title):
    counts = {n: per_recording(rows, lab, gold) for n, lab in labels.items()}
    tab = {n: summarise(c, recs) for n, c in counts.items()}
    tracks = {(r["rec"], str(r["tid"])) for r in rows if r["rec"] in recs} & set(gold)
    print(f"\n=== {title}：{sum(1 for r in rows if r['rec'] in recs)} 只手 / {len(recs)} 段 / "
          f"{len(tracks)} 条轨迹（自己 {sum(gold[t] == 'owner' for t in tracks)} / "
          f"别人 {sum(gold[t] == 'other' for t in tracks)}）===")
    print(f"  {'':<16}{'M1 翻转/100':>12}{'M2 别人帧判自己':>16}{'率':>8}{'别人轨迹有误':>12}{'G 自己帧召回':>14}")
    for n, m in tab.items():
        o = m["other_tracks_any_self"]
        print(f"  {n:<16}{m['M1_own_flips_per100']:>12.2f}{m['M2_other_frames_self']:>16d}"
              f"{m['M2_other_frame_rate']:>8.2%}{f'{o[0]}/{o[1]}':>12}{m['G_own_frames_self']:>14.2%}")
    return tab, counts


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", action="append", required=True,
                    help="name:dump:labels:clips:pred:strata")
    ap.add_argument("--out", default="/workspace/distil/student/report.json")
    a = ap.parse_args()
    all_rows, all_gold, all_labels, all_st = [], {}, {}, {}
    per_batch = {}
    for spec in a.batch:
        name, dump, labels_p, clips, pred, strata = spec.split(":")
        rows, gold, labels, st = one_batch(dump, labels_p, clips, pred, strata)
        per_batch[name] = (rows, gold, labels, sorted({r["rec"] for r in rows}))
        all_rows += rows
        all_gold.update(gold)
        all_st.update(st)
        for n, lab in labels.items():
            all_labels.setdefault(n, {}).update(lab)
    common = sorted(set.intersection(*[set(v) for v in all_labels.values()])) if all_labels else []
    all_labels = {n: {k: v for k, v in lab.items() if k in set(common)} for n, lab in all_labels.items()}
    recs = sorted({r["rec"] for r in all_rows})
    report = {"batches": {}, "pooled": {}, "verdicts": {}}
    for name, (rows, gold, labels, brecs) in per_batch.items():
        tab, _ = table(rows, gold, labels, brecs, name)
        report["batches"][name] = tab
    tab, counts = table(all_rows, all_gold, all_labels, recs, "三批合并")
    report["pooled"] = tab
    for s in sorted(set(all_st.values())):
        rs = [r for r in recs if all_st.get(r) == s]
        if rs:
            t, _ = table(all_rows, all_gold, all_labels, rs, f"分层 {s}")
            report["batches"][f"stratum_{s}"] = t

    base = "V1 deployed"
    print(f"\n=== 出厂判据（事先写定，见 src/semhand/distil_train.py）对 {base} ===")
    for arm in [n for n in tab if n.endswith(" held") and not n.startswith("V1")]:
        d = boot_diff(counts[arm], counts[base], recs)
        s, b = tab[arm], tab[base]
        c1 = s["M2_other_frames_self"] <= b["M2_other_frames_self"]
        c2 = s["M1_own_flips_per100"] <= b["M1_own_flips_per100"]
        c3 = s["G_own_frames_self"] >= b["G_own_frames_self"] - GUARD
        c4 = [m for m in ("M1", "M2") if d[m][2] < 0]
        ok = c1 and c2 and c3 and bool(c4)
        report["verdicts"][arm] = {"ship": ok, "M2_no_worse": c1, "M1_no_worse": c2,
                                   "guard": c3, "significant": c4, "diff": d}
        print(f"  {arm:<16} {'可上线' if ok else '不可上线'}  "
              f"ΔM2 {d['M2'][0]:+d} [{d['M2'][1]:+.0f},{d['M2'][2]:+.0f}]  "
              f"ΔM1 {d['M1'][0]:+.2f} [{d['M1'][1]:+.2f},{d['M1'][2]:+.2f}]  "
              f"ΔG {d['G'][0]:+.3f}  (1 {c1} 2 {c2} 3 {c3} 4 {c4})")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(report, open(a.out, "w"), indent=1, ensure_ascii=False, default=float)
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
