"""Turn the drawn zones into self/other labels on the hand dump, and train on them.

THE ZONE AND THE DETECTIONS LIVE IN DIFFERENT IMAGES. Zones were drawn on raw
cam3 fisheye frames; the hand dump was detected on the constant-depth wide
render. The join does not guess a depth to cross between them: the renderer
filled every panorama pixel by sampling cam3 through `source_maps(..., 0.6)`,
so pushing a detection through that same table lands on the cam3 pixel its
hand was drawn from. Where another module owns the panorama pixel the lookup
still exists but carries the render's parallax, and that share is reported
rather than hidden.

A BOX IS NOT A POINT. A hand straddling the boundary is the interesting case,
so the label is the fraction of a 5x5 grid over the box that falls on the
wearer's side, and a detection counts as self at half or more.

THE ZONE IS A PRIOR, SO IT IS SCORED LIKE ONE. The drawn zone is judged against
the human ownership gold at track level, next to the deployed classifier on
the same tracks, and the foreign-hand recall is the column that matters: an
other-person's hand called self is the one that goes unblurred.

TRAINED WITHOUT THE ZONE AT INFERENCE. The model sees only what exists when a
new recording arrives -- box geometry, the arm and wrist fields, handedness --
and learns the zone's label from them. Folds are whole recordings, because a
wearer's own hands sit in the same place for thirty seconds and a random split
would grade memory of a recording rather than a rule.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import math
import os
import re
import sys

import numpy as np

from src.rig.zone_stability import Eye, S as ZS

GRID = 5
DEPTH_M = 0.6          # what own_dump rendered with; the maps must match it


def load_clips(path):
    out = {}
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = line.rsplit(":", 2)
        out[os.path.basename(p[0].rstrip("/")).replace("databag-26_", "R")] = p[0]
    return out


def page_meta(pages, rec):
    for d in pages:
        f = os.path.join(d, f"zone_{rec}.html")
        if os.path.exists(f):
            s = open(f, encoding="utf-8").read(9000)
            return json.loads(re.search(r"const M = (\{.*?\});", s, re.S).group(1))
    return None


def latest(ann, rec, who):
    fs = sorted(glob.glob(f"{ann}/*/{rec}__{who}__final_*.json"))
    for f in reversed(fs):
        d = json.load(open(f, encoding="utf-8"))
        if any(e.get("keyframes") for e in d["eyes"].values()):
            return d
    return None


def rig_maps(databag):
    """cam3 sampling table and module ownership for the panorama own_dump saw."""
    from src.rig.calibration import RigCalibration
    from src.rig.geometry import VirtualWideCamera, source_maps
    from src.rig.render_wide import render
    rig = RigCalibration(os.path.join(databag, "calibration.yaml"))
    vcam = VirtualWideCamera.from_rig(rig)
    mx, my, ok = source_maps(rig, "cam3", vcam, DEPTH_M)
    dummy = {m.left.name: np.zeros((rig.cameras[m.left.name].height,
                                    rig.cameras[m.left.name].width, 3), np.uint8)
             for m in rig.modules}
    _, owner, _, _ = render(rig, vcam, dummy, DEPTH_M, colour_match=False)
    mid = len(rig.modules) // 2
    return mx, my, ok, owner, mid, vcam.width, vcam.height


def label_rows(a):
    clips = {}
    for c in a.clips:
        clips.update(load_clips(c))
    by_rec = collections.defaultdict(list)
    for r in csv.DictReader(open(a.dump, encoding="utf-8")):
        by_rec[r["rec"]].append(r)
    out, skipped = [], []
    for rec in sorted(by_rec):
        z = latest(a.ann, rec, a.who)
        m = page_meta(a.pages, rec)
        if z is None or m is None or rec not in clips:
            skipped.append((rec, "no zone" if z is None else
                            "no page" if m is None else "no databag"))
            continue
        start = int(m["start"]); nfr = int(round(m["dur"] * m["fps"]))
        eye = Eye(z["eyes"]["cam3"]["keyframes"])
        mx, my, ok, owner, mid, PW, PH = rig_maps(clips[rec])
        cache = {}
        n_in = 0
        for r in by_rec[rec]:
            f = int(r["frame"])
            if not (start <= f < start + nfr):
                continue
            x0, y0, x1, y1 = (float(r[k]) for k in ("x0", "y0", "x1", "y1"))
            if f not in cache:
                cache = {f: eye.mask(f)}          # rows arrive in frame order
            mask = cache[f]
            us = np.linspace(x0, x1, GRID); vs = np.linspace(y0, y1, GRID)
            hit = tot = 0
            for v in vs:
                for u in us:
                    iu = min(max(int(round(u)), 0), PW - 1)
                    iv = min(max(int(round(v)), 0), PH - 1)
                    if not ok[iv, iu]:
                        continue
                    cx, cy = mx[iv, iu], my[iv, iu]
                    mi = min(max(int(cy / ZS), 0), mask.shape[0] - 1)
                    mj = min(max(int(cx / ZS), 0), mask.shape[1] - 1)
                    hit += int(mask[mi, mj]); tot += 1
            cu = min(max(int(round((x0 + x1) / 2)), 0), PW - 1)
            cv_ = min(max(int(round((y0 + y1) / 2)), 0), PH - 1)
            row = dict(r)
            row.update(zone_frac=(hit / tot) if tot else "", zone_n=tot,
                       zone_self=int(hit / tot >= 0.5) if tot else "",
                       pano_owner_is_cam3=int(owner[cv_, cu] == mid),
                       pano_w=PW, pano_h=PH)
            out.append(row)
            n_in += 1
        print(f"  {rec}: {n_in} 个检测在区域片段里", flush=True)
    return out, skipped


def gold(paths):
    g = {}
    for p in paths:
        for r in csv.DictReader(open(p, encoding="utf-8-sig")):
            t = r.get("track_truth") or r.get("human_ownership")
            if t:
                g[(r["rec"], str(r["tid"]))] = t
    return g


def score(name, pred, truth):
    """pred/truth: {track: 'owner'|'other'} on the same tracks."""
    keys = [k for k in truth if k in pred]
    tp = sum(1 for k in keys if truth[k] == "owner" and pred[k] == "owner")
    tn = sum(1 for k in keys if truth[k] == "other" and pred[k] == "other")
    fo = sum(1 for k in keys if truth[k] == "other" and pred[k] == "owner")
    fs = sum(1 for k in keys if truth[k] == "owner" and pred[k] == "other")
    n = len(keys)
    oth = tn + fo
    own = tp + fs
    print(f"  {name:<30} n={n:<4} 准确 {(tp + tn) / n:6.1%}   "
          f"别人的手召回 {tn / oth if oth else float('nan'):6.1%} ({tn}/{oth})   "
          f"自己的手召回 {tp / own if own else float('nan'):6.1%} ({tp}/{own})")


def features(r):
    PW, PH = float(r["pano_w"]), float(r["pano_h"])
    x0, y0, x1, y1 = (float(r[k]) for k in ("x0", "y0", "x1", "y1"))
    num = lambda k: (float(r[k]) if r.get(k) not in (None, "", "nan") else np.nan)
    ang = num("arm_angle")
    edge = r.get("exit_edge", "")
    return [(x0 + x1) / 2 / PW, (y0 + y1) / 2 / PH, (x1 - x0) / PW,
            (y1 - y0) / PH, y1 / PH, x0 / PW, x1 / PW,
            num("wrist_x") / PW, num("wrist_y") / PH,
            math.sin(math.radians(ang)) if not np.isnan(ang) else np.nan,
            math.cos(math.radians(ang)) if not np.isnan(ang) else np.nan,
            num("exit_x") / PW, num("exit_y") / PH,
            float(edge == "bottom"), float(edge == "left"),
            float(edge == "right"), float(edge == "top"),
            float(r.get("side_raw") == "left"), float(r.get("side_raw") == "right"),
            num("side_conf")]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ann", required=True)
    ap.add_argument("--pages", action="append", required=True)
    ap.add_argument("--clips", action="append", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--gold", action="append", default=[])
    ap.add_argument("--who", default="roselindsi")
    ap.add_argument("--out", required=True, help="labelled detections csv")
    a = ap.parse_args()

    rows, skipped = label_rows(a)
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    lab = [r for r in rows if r["zone_self"] != ""]
    print(f"\n{len(rows)} 个检测落在区域片段里，{len(lab)} 个能映射到 cam3 "
          f"（{sum(int(r['pano_owner_is_cam3']) for r in rows) / max(len(rows), 1):.0%} "
          f"的检测中心在 cam3 负责的全景区域）-> {a.out}")
    for rec, why in skipped:
        print(f"  跳过 {rec}: {why}")
    print(f"区域判为自己侧的检测: {np.mean([int(r['zone_self']) for r in lab]):.1%}")

    truth = {k: v for k, v in gold(a.gold).items() if v in ("owner", "other")}
    by_track = collections.defaultdict(list)
    for r in lab:
        by_track[(r["rec"], str(r["tid"]))].append(r)
    maj = lambda xs: "owner" if np.mean(xs) >= 0.5 else "other"
    zone_pred = {k: maj([int(r["zone_self"]) for r in v]) for k, v in by_track.items()}
    v1_pred = {k: maj([int(r["final_owner_post_cap"]) for r in v])
               for k, v in by_track.items()}
    covered = {k: t for k, t in truth.items() if k in by_track}
    print(f"\n人工 gold 里 owner/other 的轨迹 {len(truth)} 条，其中 {len(covered)} 条"
          f"在区域片段里有检测（只在这些上比）:")
    score("区域先验（画的区域，直接判）", zone_pred, covered)
    score("V1（现在部署的归属）", v1_pred, covered)

    # ---------------------------------------------------------- training
    from sklearn.ensemble import HistGradientBoostingClassifier
    X = np.array([features(r) for r in lab], float)
    y = np.array([int(r["zone_self"]) for r in lab])
    p_v1 = np.array([float(r["logit_owner"]) for r in lab])
    recs = np.array([r["rec"] for r in lab])
    print(f"\n训练：{len(lab)} 个检测、{len(set(recs))} 段录像，按录像留一交叉验证，"
          f"标签 = 区域先验（推理时不看区域）")
    for name, XX in (("几何+手臂", X), ("几何+手臂+V1分数", np.c_[X, p_v1])):
        prob = np.zeros(len(y))
        for rec in sorted(set(recs)):
            tr, te = recs != rec, recs == rec
            if len(set(y[tr])) < 2:
                prob[te] = y[tr].mean(); continue
            clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08,
                                                 max_leaf_nodes=15,
                                                 l2_regularization=1.0,
                                                 random_state=0)
            clf.fit(XX[tr], y[tr])
            prob[te] = clf.predict_proba(XX[te])[:, 1]
        det_acc = ((prob >= 0.5) == y).mean()
        pred = collections.defaultdict(list)
        for r, p in zip(lab, prob):
            pred[(r["rec"], str(r["tid"]))].append(p)
        track_pred = {k: ("owner" if np.mean(v) >= 0.5 else "other")
                      for k, v in pred.items()}
        print(f"  [{name}] 逐检测复现区域标签 {det_acc:.1%}")
        score(f"模型: {name}", track_pred, covered)


if __name__ == "__main__":
    main()
