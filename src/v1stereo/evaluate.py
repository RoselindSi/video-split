"""Score both questions on the same hands, own-hand stability first.

PART ONE, testpkg_2, PER HAND. V1's frozen set: labelled frames from 85
recordings that no model here has trained on. Only rows that have a stereo
pair are scored, for every model. It answers Q1 without touching fresh29.

PART TWO, fresh29, PER TRACK AND PER FRAME. The detections the stereo arms
can see, restricted further to those that also have a panorama crop, so every
row -- V1 deployed, V1 raw, V1p13, V1s13 and the four zone arms -- is scored
on one set of hands.

STABILITY IS MEASURED ON THE WEARER'S OWN TRACKS, RAW. For each gold-owner
track, over pairs of consecutive frames:
  own false-blur rate    frames called foreign / own frames
  switches / 100         label changes between consecutive own frames
  tracks touched         own tracks with at least one false-blurred frame
  longest false blur     the longest consecutive run, in seconds
No classifier gets smoothing; `V1 deployed` keeps its own, as a reference.

THE VERDICTS ARE THE ONES IN `src/v1stereo/__init__.py`, applied mechanically
to the pooled fresh29 numbers and printed per stratum beside them.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import os

import numpy as np

from src.v1stereo.train import GEOM13, StereoPairs, common_rows

STRATA = "/workspace/fresh29_strata.txt"


def probs(model, ds, device, batch=64):
    import torch
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(ds), batch):
            h, c, g, _ = zip(*[ds[j] for j in range(i, min(len(ds), i + batch))])
            o = model(torch.stack(h).to(device), torch.stack(c).to(device),
                      torch.stack(g).to(device))
            out.append(torch.softmax(o, 1)[:, 1].cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def load_v1(path, device):
    import torch
    from src.rig import own_ctx
    ck = torch.load(path, map_location=device, weights_only=False)
    m = own_ctx.build(ck.get("arm", "both_geom"),
                      n_geom=ck.get("n_geom", len(own_ctx.GEOM))).to(device)
    m.load_state_dict(ck["state"])
    return m, ck


def hand_scores(p_owner, y):
    pred = (np.asarray(p_owner) >= 0.5).astype(int)
    y = np.asarray(y)
    own = y == 1
    oth = y == 0
    tp = int(((pred == 0) & oth).sum())
    fp = int(((pred == 0) & own).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / int(oth.sum()) if oth.any() else float("nan")
    return {"own_recall": float((pred[own] == 1).mean()) if own.any() else float("nan"),
            "other_recall": rec, "other_prec": prec,
            "other_f1": (2 * prec * rec / (prec + rec)
                         if prec == prec and rec == rec and prec + rec else float("nan")),
            "n_own": int(own.sum()), "n_other": int(oth.sum())}


def g13_from_dump(d, W, H):
    W, H = float(W), float(H)
    x0, y0, x1, y1 = (float(d[k]) for k in ("x0", "y0", "x1", "y1"))
    ang = d.get("arm_angle", "")
    dx, dy = ((math.cos(math.radians(float(ang))), math.sin(math.radians(float(ang))))
              if ang not in ("", None) else (0.0, 0.0))
    ex = float(d["exit_x"]) / W if d.get("exit_x") not in ("", None) else 0.0
    ey = float(d["exit_y"]) / H if d.get("exit_y") not in ("", None) else 0.0
    e = d.get("exit_edge", "")
    v = {"box_cx": (x0 + x1) / 2 / W, "box_cy": (y0 + y1) / 2 / H,
         "box_w": (x1 - x0) / W, "box_h": (y1 - y0) / H, "dir_x": dx, "dir_y": dy,
         "exit_x": ex, "exit_y": ey, "edge_bottom": float(e == "bottom"),
         "edge_left": float(e == "left"), "edge_right": float(e == "right"),
         "edge_top": float(e == "top"), "conf": float(d.get("side_conf") or 0.0)}
    return np.array([v[n] for n in GEOM13], np.float32)


class DetPano:
    def __init__(self, root, rows):
        self.root, self.rows = root, rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        import cv2
        import torch
        from src.selfother.model import to_tensor
        cv2.setNumThreads(1)
        r = self.rows[i]
        h = cv2.imread(os.path.join(self.root, r["hand_path"]))
        c = cv2.imread(os.path.join(self.root, r["ctx_path"]))
        return to_tensor(h), to_tensor(c), torch.from_numpy(r["g"]), 0


def stability(rows, self_det, self_track, gold, recs, fps):
    frames = collections.defaultdict(list)
    for r, s in zip(rows, self_det):
        k = (r["rec"], r["tid"])
        if k in gold and r["rec"] in recs:
            frames[k].append((int(r["frame"]), bool(s)))
    own = [k for k in frames if gold[k] == "owner"]
    oth = [k for k in frames if gold[k] == "other"]
    n_own = fb = pairs = sw = touched = 0
    longest_fb = 0
    for k in own:
        prev = prev_s = None
        run, any_fb = 0, False
        for f, s in sorted(frames[k]):
            n_own += 1
            fb += (not s)
            consecutive = prev is not None and f == prev + 1
            if consecutive:
                pairs += 1
                sw += (s != prev_s)
            run = (run + 1 if consecutive and prev_s is False else 1) if not s else 0
            longest_fb = max(longest_fb, run)
            any_fb |= (not s)
            prev, prev_s = f, s
        touched += any_fb
    n_oth = miss = 0
    longest_exp = 0
    for k in oth:
        prev = prev_s = None
        run = 0
        for f, s in sorted(frames[k]):
            n_oth += 1
            miss += s
            consecutive = prev is not None and f == prev + 1
            run = (run + 1 if consecutive and prev_s else 1) if s else 0
            longest_exp = max(longest_exp, run)
            prev, prev_s = f, s
    return {
        "own_tracks": [sum(1 for k in own if self_track.get(k)), len(own)],
        "other_tracks": [sum(1 for k in oth if self_track.get(k) is False), len(oth)],
        "own_fb_rate": fb / n_own if n_own else float("nan"),
        "own_switch_per100": 100.0 * sw / pairs if pairs else float("nan"),
        "own_tracks_touched": [touched, len(own)],
        "own_longest_fb_s": longest_fb / fps,
        "other_frame_miss_rate": miss / n_oth if n_oth else float("nan"),
        "other_longest_exposure_s": longest_exp / fps}


def better(a, b):
    miss_a = a["other_tracks"][1] - a["other_tracks"][0]
    miss_b = b["other_tracks"][1] - b["other_tracks"][0]
    return (a["own_fb_rate"] < b["own_fb_rate"]
            and a["own_switch_per100"] < b["own_switch_per100"]
            and miss_a <= miss_b + 1)


def fmt(m):
    o, t = m["own_tracks"], m["other_tracks"]
    return (f"自己的手轨迹 {o[0]}/{o[1]}  误糊帧 {m['own_fb_rate']:6.2%}  "
            f"翻转/100帧 {m['own_switch_per100']:5.2f}  受影响轨迹 "
            f"{m['own_tracks_touched'][0]}/{m['own_tracks_touched'][1]}  "
            f"最长误糊 {m['own_longest_fb_s']:.2f}s | 别人的手轨迹 {t[0]}/{t[1]}  "
            f"漏糊帧 {m['other_frame_miss_rate']:5.2%}  最长暴露 "
            f"{m['other_longest_exposure_s']:.2f}s")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="recut root (stereo pkg crops)")
    ap.add_argument("--testpkg", required=True)
    ap.add_argument("--v1", required=True, help="shipped own_ctx_best.pt")
    ap.add_argument("--v1p", action="append", required=True)
    ap.add_argument("--v1s", action="append", required=True)
    ap.add_argument("--zone_ckpt", action="append", required=True)
    ap.add_argument("--stereo_root", required=True, help="selfother crops root")
    ap.add_argument("--split", required=True)
    ap.add_argument("--pano_root", required=True, help="panocrops.py output")
    ap.add_argument("--dump", required=True)
    ap.add_argument("--gold", action="append", required=True)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    from src.rig import own_ctx
    from src.selfother.crops import read_index
    from src.selfother.evaluate import strata
    from src.selfother.model import load_trained
    from src.selfother.train import predict, read_gold
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(8)
    report = {"testpkg_2": {}, "fresh29": {}, "verdicts": {}}

    # ------------------------------------------------ part one: testpkg_2
    rows13, n_all = common_rows([a.testpkg], a.root)
    y = [r["y"] for r in rows13]
    full = {r["stem"]: r for r in own_ctx.load([a.testpkg], verbose=False)}
    rows14 = [full[r["stem"]] for r in rows13]
    print(f"=== testpkg_2 逐手（{len(rows13)}/{n_all} 行有双目对）===")
    m, _ = load_v1(a.v1, device)
    p = probs(m, own_ctx.Pairs(rows14, augment=False), device)
    report["testpkg_2"]["V1 shipped"] = hand_scores(p, y)
    for name, paths, kind in (("V1p13", a.v1p, "pano"), ("V1s13", a.v1s, "stereo")):
        ps = []
        for path in paths:
            mm, _ = load_v1(path, device)
            ds = (own_ctx.Pairs(rows13, augment=False) if kind == "pano"
                  else StereoPairs(a.root, rows13, augment=False))
            ps.append(probs(mm, ds, device))
        report["testpkg_2"][name] = dict(hand_scores(np.mean(ps, 0), y),
                                         seeds=[hand_scores(q, y) for q in ps])
    for name, s in report["testpkg_2"].items():
        print(f"  {name:<11} 自己的手召回 {s['own_recall']:.3f}  别人的手 召回 "
              f"{s['other_recall']:.3f} 精度 {s['other_prec']:.3f} F1 "
              f"{s['other_f1']:.3f}  (自己 {s['n_own']} / 别人 {s['n_other']})"
              + ("  各seed F1 " + " ".join(f"{x['other_f1']:.3f}" for x in s["seeds"])
                 if "seeds" in s else ""))

    # ------------------------------------------------ part two: fresh29
    dump = {(r["rec"], r["frame"], r["tid"]): r
            for r in csv.DictReader(open(a.dump, encoding="utf-8"))}
    pano = {}
    for rec in os.listdir(a.pano_root):
        p_ = os.path.join(a.pano_root, rec, "index.csv")
        if os.path.exists(p_):
            for r in csv.DictReader(open(p_, encoding="utf-8")):
                if r["status"] == "ok":
                    pano[(r["rec"], r["frame"], r["tid"])] = r
    rows = []
    for r in read_index(a.stereo_root, a.split):
        k = (r["rec"], r["frame"], r["tid"])
        if r["status"] == "ok" and k in pano and k in dump:
            rows.append(dict(r, g=g13_from_dump(dump[k], pano[k]["pano_w"],
                                                pano[k]["pano_h"]),
                             hand_path=pano[k]["hand_path"],
                             ctx_path=pano[k]["ctx_path"]))
    gold = read_gold(a.gold)
    st = strata(STRATA)
    print(f"\n=== fresh29（{len(rows)} 个检测、"
          f"{len({(r['rec'], r['tid']) for r in rows} & set(gold))} 条 gold 轨迹，"
          f"双目对和全景裁剪都有）===")

    track_mean = lambda vals: {k: float(np.mean(v)) >= 0.5 for k, v in _group(rows, vals).items()}
    defs = {}
    v1d = [dump[(r["rec"], r["frame"], r["tid"])]["final_owner_post_cap"] == "1" for r in rows]
    defs["V1 deployed"] = (v1d, track_mean([float(x) for x in v1d]))
    v1r = [float(dump[(r["rec"], r["frame"], r["tid"])]["p_owner_raw"]) for r in rows]
    defs["V1 raw"] = ([x >= 0.5 for x in v1r], track_mean(v1r))
    stereo_ds_root = os.path.join(a.stereo_root, a.split)
    for name, paths, kind in (("V1p13", a.v1p, "pano"), ("V1s13", a.v1s, "stereo")):
        ps = []
        for path in paths:
            mm, _ = load_v1(path, device)
            ds = (DetPano(os.path.join(a.pano_root), rows) if kind == "pano"
                  else StereoPairs(stereo_ds_root, rows, augment=False))
            ps.append(probs(mm, ds, device))
            print(f"  {name} {os.path.basename(path)} 预测完", flush=True)
        pm = np.mean(ps, 0)
        defs[name] = ([x >= 0.5 for x in pm], track_mean(pm))
    zone = collections.defaultdict(list)
    for path in a.zone_ckpt:
        net, meta = load_trained(path, device)
        init = "v1s" if "V1s13" in (meta.get("v1_ckpt") or "") else meta["init"]
        zone[f"{meta['input']}/{init}"].append(
            predict(net, a.stereo_root, a.split, rows, meta["input"], device, 256, 4))
        print(f"  {meta['input']}/{init} seed {meta['seed']} 预测完", flush=True)
    for name in sorted(zone):
        pm = np.mean(zone[name], 0)
        defs[name] = ([x >= 0.5 for x in pm], track_mean(pm))

    recs_all = {r["rec"] for r in rows}
    groups = {"all": recs_all}
    for s_ in sorted(set(st.values())):
        groups[s_] = {x for x in recs_all if st.get(x) == s_}
    for g, recs in groups.items():
        print(f"\n--- {g} ({len(recs)} 段) ---")
        for name, (det, trk) in defs.items():
            mtr = stability(rows, det, trk, gold, recs, a.fps)
            report["fresh29"].setdefault(name, {})[g] = mtr
            print(f"  {name:<12} {fmt(mtr)}")

    R = report["fresh29"]
    print("\n=== 判据（事先写定）: 误糊帧更少 且 翻转更少 且 别人的手最多多漏 1 条 ===")
    for q, a_, b_ in (("Q1 双目让 V1 更好", "V1s13", "V1p13"),
                      ("Q2 A: V1s13 初始化 vs ImageNet", "A/v1s", "A/imagenet"),
                      ("Q2 B: V1s13 初始化 vs ImageNet", "B/v1s", "B/imagenet")):
        if a_ in R and b_ in R:
            ok = better(R[a_]["all"], R[b_]["all"])
            report["verdicts"][q] = ok
            print(f"  {q:<34} {'是' if ok else '否'}")
    with open(a.out, "w") as f:
        json.dump(report, f, indent=1, default=float)
    print(f"\n-> {a.out}")


def _group(rows, vals):
    g = collections.defaultdict(list)
    for r, v in zip(rows, vals):
        g[(r["rec"], r["tid"])].append(v)
    return g


if __name__ == "__main__":
    main()
