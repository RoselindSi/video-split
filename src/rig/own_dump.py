"""Is an ownership error a flicker, or is the whole track wrong?

TRACK-LEVEL AGGREGATION CAN DRIVE THE FLIP RATE TO ZERO BY CONSTRUCTION, which
is why the flip rate is the wrong question. A hand's owner does not change
while the hand exists, so ownership is a property of the track and every
within-track flip is an error rather than a trade -- but making the label
constant only helps if the track's evidence is mostly right. If the classifier
sits confidently on the wrong side for the whole track, aggregation does not
repair anything; it makes the mistake tidy. The census below separates those.

    A  stable correct        mostly on the right side, few or no flips
    B  mixed, aggregate correct   flips, but the track's evidence is right
                                  -- THE ONLY CLASS AGGREGATION REPAIRS
    C  mixed, aggregate wrong     flips, and the track's evidence is wrong
                                  -- aggregation would enlarge the error
    D  stable wrong          confidently wrong throughout; nothing temporal
                             can touch it

THE STAGES ARE STORED SEPARATELY BECAUSE THEY DISAGREE. The deployed verdict
is a classifier probability, then a geometric prior and a rule blended into
it, then an EMA, then a Schmitt trigger, and finally a frame-level cap that
writes its decision back into the per-track belief. A flip seen at the output
cannot be charged to any one of those unless they are recorded apart, and the
cap writeback is already a known debt: it puts a fact about the FRAME -- how
many hands are in it -- into a single hand's history.

THE REFERENCE IS A FROZEN RULE, NOT GROUND TRUTH. `exit_y >= 0.55H` reproduced
1028 stored labels over 13 recordings without an exception, and it is
independent of the ownership classifier, which is what makes it usable as a
second opinion here. It is not a human verdict on these frames and nothing
below should be read as accuracy against one; what is measured is agreement
between the frame-level classifier's dynamics and an independent reference.

NOTHING PRODUCTION IS CHANGED. This runs the deployed configuration and writes
down what it did. The three aggregation policies are replayed offline over the
recorded probabilities, at a fixed 0.5, with no sweep -- picking a threshold on
the same errors it is scored against is how a policy gets fitted to its test.
"""
from __future__ import annotations

import argparse
import collections
import csv
import math
import os

FIELDS = ("rec", "frame", "tid", "reference_owner", "p_owner_raw",
          "logit_owner", "ema_owner", "ownhold_pre_cap", "owner_count_pre_cap",
          "cap_triggered", "cap_demoted", "final_owner_post_cap",
          "track_age", "track_lost", "side_raw", "side_conf",
          # The box is here so a later census can rebuild the crop the
          # classifier actually saw. Leaving it out once already meant the
          # only follow-up question worth asking -- what was it looking at --
          # could not be asked without repeating the whole run.
          "x0", "y0", "x1", "y1",
          # THE ARM CUE IN ITS UNTHRESHOLDED FORM. `forearm_exit` already runs
          # every frame -- it is what `rule_owner` is computed from -- and
          # everything except one boolean was being thrown away. The wrist, the
          # direction of the forearm, and where that ray meets the border are
          # the quantities the literature on wearer-arm attachment actually
          # uses; `exit_y >= 0.55H` is one threshold cut through them.
          "wrist_x", "wrist_y", "arm_angle", "exit_edge", "exit_x", "exit_y")


def arm_fields(d):
    """Wrist, forearm direction and border exit, or blanks when absent.

    Blank is not zero and must not become zero: a hand whose keypoints are
    unusable has no arm evidence at all, which is a different state from an
    arm pointing along the x axis."""
    import math as _m
    kp = d.get("kp")
    out = {"wrist_x": "", "wrist_y": "", "arm_angle": "",
           "exit_edge": d.get("edge") or "", "exit_x": "", "exit_y": ""}
    pt = d.get("exit")
    if pt is not None:
        out["exit_x"], out["exit_y"] = int(pt[0]), int(pt[1])
    if kp is None:
        return out
    try:
        import numpy as _np
        kp = _np.asarray(kp, float)
        if kp.shape[0] < 21 or not _np.isfinite(kp).all():
            return out
        w = kp[0]
        v = w - kp[1:21].mean(0)
        out["wrist_x"], out["wrist_y"] = int(w[0]), int(w[1])
        if float(_np.linalg.norm(v)) > 1e-6:
            out["arm_angle"] = round(_m.degrees(_m.atan2(float(v[1]),
                                                         float(v[0]))), 2)
    except Exception:
        pass
    return out


def logit(p, eps=1e-6):
    p = min(1.0 - eps, max(eps, float(p)))
    return math.log(p / (1.0 - p))


def wilson(k, n, z=1.96):
    if not n:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / d, (c + m) / d)


def dump(a):
    import cv2
    from ultralytics import YOLO
    from src.rig.calibration import RigCalibration
    from src.rig.geometry import VirtualWideCamera
    from src.rig.render_wide import render
    from src.rig.seam_fix import ClipReader, Prefetch
    from src.rig.hand_detect import detect, OwnHold
    from src.rig.hand_track import Tracker, MAX_LOST
    from src.rig import own_ctx, geom_prior, demo_video

    clips = {}
    for line in open(a.clips):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = line.rsplit(":", 2)
        tag = os.path.basename(p[0].rstrip("/")).replace("databag-26_", "R")
        clips[tag] = (p[0], int(p[1]))

    model = YOLO(a.weights)
    ctx_model, ctx_device, _ = own_ctx.load_model(a.clf_ctx)
    if ctx_model is None:
        raise SystemExit(f"no ownership checkpoint at {a.clf_ctx}")
    geom = geom_prior.load_model(a.geom)

    rows, failed = [], []
    # RESUME, BECAUSE ONE BAD CLIP USED TO COST THE WHOLE RUN. A four-hour
    # pass over twenty-nine recordings died on the seventh when a reader
    # returned nothing, and the six already decoded went with it.
    done = set()
    if a.resume and os.path.exists(a.out):
        for r in csv.DictReader(open(a.out, encoding="utf-8-sig")):
            rows.append({k: r.get(k, "") for k in FIELDS})
            done.add(r["rec"])
        print(f"  续跑：{a.out} 里已有 {len(done)} 段 / {len(rows)} 行",
              flush=True)
    for tag in (a.rec or sorted(clips)):
        if tag not in clips or tag in done:
            continue
        try:
            dump_one(a, tag, clips, model, ctx_model, ctx_device, geom, rows)
        except Exception as e:
            # A DECODE FAILURE IS NOT A REASON TO LOSE THE OTHER TWENTY-EIGHT.
            # It is also not silent: the clip is named, and a batch that drops
            # recordings has to say which ones before any rate computed on it
            # means anything.
            failed.append((tag, f"{type(e).__name__}: {e}".split("\n")[0]))
            print(f"  !! {tag} 失败并跳过 -- {failed[-1][1]}", flush=True)
    if failed:
        print(f"\n  跳过 {len(failed)} 段：")
        for tag, why in failed:
            print(f"    {tag}  {why}")
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(FIELDS))
        w.writeheader()
        w.writerows(rows)
    print(f"\n  {len(rows)} 行 / "
          f"{len({(r['rec'], r['tid']) for r in rows})} 条轨迹 -> {a.out}")


def dump_one(a, tag, clips, model, ctx_model, ctx_device, geom, rows):
    """One recording, appended to `rows`. Raises on a decode failure."""
    from src.rig.calibration import RigCalibration
    from src.rig.geometry import VirtualWideCamera
    from src.rig.render_wide import render
    from src.rig.seam_fix import ClipReader, Prefetch
    from src.rig.hand_detect import detect, OwnHold
    from src.rig.hand_track import Tracker, MAX_LOST
    from src.rig import own_ctx, demo_video
    if True:
        databag, start = clips[tag]
        rig = RigCalibration(os.path.join(databag, "calibration.yaml"))
        vcam = VirtualWideCamera.from_rig(rig)
        vids = {k: os.path.join(databag, f"{k}.mp4")
                for k in ("cam12", "cam34", "cam56")}
        # The deployed configuration, unchanged.
        tracker = Tracker(max_lost=max(MAX_LOST,
                                       demo_video.MAX_PREDICTION_AGE))
        ownhold = OwnHold(geom=geom, geom_w=a.geom_w, max_owner=a.max_owner,
                          state_ttl=tracker.max_lost)
        rd = Prefetch(ClipReader(rig, vids, start), skip=0)
        mc = {}
        print(f"  {tag}: {start}-{start + a.n - 1}", flush=True)
        for k in range(a.n):
            src = rd.next()
            if not src:
                break
            try:
                rgb, _, _, _ = render(rig, vcam, src, 0.6, map_cache=mc)
            except TypeError:
                rgb, _, _, _ = render(rig, vcam, src, 0.6)
            raw_dets = detect(model, rgb, min_conf=a.conf)
            raw_ids = tracker.update(raw_dets, rgb.shape,
                                     new_track_conf=a.new_track_conf,
                                     # SEPARATED FROM THE DETECTOR'S FLOOR ON
                                     # PURPOSE. Raising `--conf` to disable
                                     # track-conditioned admission would also
                                     # delete those detections from the frame,
                                     # which changes what `max_owner` ranks and
                                     # what the geometric prior sees as
                                     # context. The ablation has to move one
                                     # thing.
                                     continue_conf=(a.continue_conf
                                                    if a.continue_conf
                                                    is not None else a.conf))
            keep = [i for i, t in enumerate(raw_ids) if t is not None]
            dets = [raw_dets[i] for i in keep]
            tids = [raw_ids[i] for i in keep]
            if not dets:
                continue
            flags = own_ctx.predict(ctx_model, ctx_device, rgb, dets)
            raw_p = [float(p) for _, p in flags]
            out = ownhold.update(dets, flags, shape=rgb.shape, ids=tids)
            pre = getattr(ownhold, "last_pre_cap", list(out))
            demoted = {t for t, _ in getattr(ownhold, "last_demoted", [])}
            n_pre = sum(1 for o, _ in pre if o)
            capped = int(a.max_owner is not None and n_pre > a.max_owner)
            for i, (d, tid) in enumerate(zip(dets, tids)):
                tr = tracker.tracks.get(tid, {})
                rows.append({
                    "rec": tag, "frame": start + k, "tid": tid,
                    # A SECOND OPINION, NOT A VERDICT: the frozen exit-height
                    # rule, independent of the classifier being examined.
                    "reference_owner": int(bool(d.get("rule_owner"))),
                    "p_owner_raw": round(raw_p[i], 5),
                    "logit_owner": round(logit(raw_p[i]), 4),
                    "ema_owner": round(float(pre[i][1]), 5),
                    "ownhold_pre_cap": int(bool(pre[i][0])),
                    "owner_count_pre_cap": n_pre,
                    "cap_triggered": capped,
                    "cap_demoted": int(tid in demoted),
                    "final_owner_post_cap": int(bool(out[i][0])),
                    "track_age": int(tr.get("age", 0)),
                    "track_lost": int(tr.get("lost", 0)),
                    "side_raw": d.get("side", ""),
                    "side_conf": round(float(d.get("conf", 0.0)), 4),
                    "x0": int(d["box"][0]), "y0": int(d["box"][1]),
                    "x1": int(d["box"][2]), "y1": int(d["box"][3]),
                    **arm_fields(d)})
        rd.close()
        # FLUSHED AFTER EVERY RECORDING. A four-hour run that writes once at
        # the end has a four-hour blast radius, and the sheets downstream can
        # start on what has landed instead of waiting for the last clip.
        with open(a.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(FIELDS))
            w.writeheader()
            w.writerows(rows)
        print(f"    -> {len(rows)} 行已落盘", flush=True)
    print(f"\n  {len(rows)} 行 / "
          f"{len({(r['rec'], r['tid']) for r in rows})} 条轨迹 -> {a.out}")


# ---------------------------------------------------------------- replay

def replay(rows):
    """O0 deployed, O1 track majority, O2 mean logit. -> per-frame verdicts

    Both aggregates decide at a FIXED 0.5 -- majority of `p >= 0.5`, and mean
    logit against 0. Sweeping either on the errors they are scored against
    would fit the policy to its own test set."""
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["rec"], r["tid"])].append(r)
    out = {}
    for key, v in by.items():
        ps = [float(r["p_owner_raw"]) for r in v]
        maj = sum(1 for p in ps if p >= 0.5) * 2 >= len(ps)
        mean = sum(logit(p) for p in ps) / len(ps) >= 0.0
        out[key] = {"O1": maj, "O2": mean, "rows": v}
    return out


def census(rows, tracks):
    """A/B/C/D over tracks, against the frozen reference rule."""
    tab = collections.Counter()
    detail = collections.defaultdict(list)
    for key, t in tracks.items():
        v = t["rows"]
        ref = [int(r["reference_owner"]) for r in v]
        # A track whose reference answer is not itself constant cannot be
        # scored this way; the rule disagreeing with itself over one hand
        # means the rule, not the classifier, is the thing in question.
        if len(set(ref)) > 1:
            tab["reference 自己不一致"] += 1
            continue
        truth = bool(ref[0])
        raw = [float(r["p_owner_raw"]) >= 0.5 for r in v]
        mixed = len(set(raw)) > 1
        agg_ok = t["O1"] == truth
        cls = ("B" if agg_ok else "C") if mixed else ("A" if agg_ok else "D")
        tab[cls] += 1
        detail[cls].append((key, len(v)))
    return tab, detail


def report(rows):
    tracks = replay(rows)
    n_row = len(rows)
    print(f"\n  {n_row} 帧-手  /  {len(tracks)} 条轨迹")

    tab, detail = census(rows, tracks)
    named = {"A": "A 稳定正确", "B": "B 会翻但整体正确 ← 聚合唯一能救的",
             "C": "C 会翻且整体错   ← 聚合会放大",
             "D": "D 稳定地错       ← 聚合完全无用"}
    scored = sum(tab[k] for k in "ABCD")
    print(f"\n  === 轨迹普查（参照：冻结出口规则，不是人工真值）===")
    for k in "ABCD":
        f = sum(n for _, n in detail[k])
        print(f"  {named[k]:<34} {tab[k]:>4} 条 {tab[k]/max(1,scored):>6.1%}"
              f"   {f:>6} 帧")
    if tab["reference 自己不一致"]:
        print(f"  {'（参照规则在轨迹内自相矛盾，未计）':<34} "
              f"{tab['reference 自己不一致']:>4} 条")

    print(f"\n  === 三种策略对参照规则的一致率 ===")
    print(f"  {'':<26} {'帧级一致':>10} {'轨迹级一致':>12} {'翻转/百帧':>11}")
    for pol in ("O0", "O1", "O2"):
        ok = tot = flips = seen = 0
        t_ok = t_n = 0
        for key, t in tracks.items():
            v = t["rows"]
            ref = [int(r["reference_owner"]) for r in v]
            if len(set(ref)) > 1:
                continue
            truth = bool(ref[0])
            if pol == "O0":
                lab = [bool(int(r["final_owner_post_cap"])) for r in v]
            else:
                lab = [t[pol]] * len(v)
            ok += sum(1 for x in lab if x == truth)
            tot += len(lab)
            flips += sum(1 for x, y in zip(lab, lab[1:]) if x != y)
            seen += len(lab)
            t_ok += (lab[0] == truth) if len(set(lab)) == 1 else 0
            t_n += 1
        print(f"  {pol:<26} {ok/max(1,tot):>10.3f} {t_ok/max(1,t_n):>12.3f} "
              f"{100.0*flips/max(1,seen):>11.1f}")

    print(f"\n  === 聚合修好了多少、弄坏了多少（O0 -> O1）===")
    rep = brk = 0
    for key, t in tracks.items():
        v = t["rows"]
        ref = [int(r["reference_owner"]) for r in v]
        if len(set(ref)) > 1:
            continue
        truth = bool(ref[0])
        for r in v:
            was = bool(int(r["final_owner_post_cap"])) == truth
            now = t["O1"] == truth
            rep += (not was) and now
            brk += was and (not now)
    print(f"  修好 {rep} 帧   弄坏 {brk} 帧   净 {rep - brk:+d}")
    print("  修好的是 flicker，弄坏的是整条轨迹被统一到错的一侧 —— 后者更难发现，")
    print("  因为它不再闪烁。净值为正也不代表该上，要看弄坏的那些是不是别人的手。")

    print(f"\n  === cap 干预了多少 ===")
    cap = sum(1 for r in rows if int(r["cap_triggered"]))
    dem = sum(1 for r in rows if int(r["cap_demoted"]))
    dis = sum(1 for r in rows
              if int(r["ownhold_pre_cap"]) != int(r["final_owner_post_cap"]))
    print(f"  cap 触发 {cap} 帧-手 ({cap/n_row:.1%})   实际降级 {dem}   "
          f"cap 改变了结论 {dis}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("dump", "report"), required=True)
    ap.add_argument("--clips", default="/workspace/e2e_main2.txt")
    ap.add_argument("--rec", action="append")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--new_track_conf", type=float, default=0.60)
    ap.add_argument("--continue_conf", type=float, default=None,
                    help="admission floor for continuing an existing track; "
                         "defaults to --conf, and setting it equal to "
                         "--new_track_conf turns track-conditioned admission "
                         "off without changing what the detector reports")
    ap.add_argument("--geom_w", type=float, default=0.5)
    ap.add_argument("--max_owner", type=int, default=2)
    ap.add_argument("--clf_ctx", default="/workspace/own_ctx_best.pt")
    ap.add_argument("--geom", default="/workspace/geom_inv2.json")
    ap.add_argument("--weights",
                    default="/shared/models/HaWoR/weights/external/detector.pt")
    ap.add_argument("--out", default="/workspace/own_dump.csv")
    ap.add_argument("--resume", action="store_true",
                    help="keep the recordings already in --out and only run "
                         "the ones missing")
    ap.add_argument("--rows", help="report mode: the dumped csv")
    ap.add_argument("--hands", nargs="*",
                    help="hand_precision label csvs; keep only the tracks a "
                         "person confirmed are hands")
    a = ap.parse_args()

    if a.mode == "dump":
        dump(a)
        return
    rows = list(csv.DictReader(open(a.rows, encoding="utf-8-sig")))
    if a.hands:
        # THE DENOMINATOR WAS CONTAMINATED. 18.2% of the tracks handed to the
        # ownership classifier are not hands at all, and a detector proposing
        # a bag of peppers is not an ownership failure -- charging it to this
        # stage sends the repair to the wrong place. Everything below is
        # re-derived on the tracks a person confirmed, so the census counts
        # only errors ownership could have made.
        keep = set()
        for f in a.hands:
            for r in csv.DictReader(open(f, encoding="utf-8-sig")):
                if r["verdict"] == "hand":
                    keep.add((r["rec"], r["tid"]))
        n0 = len(rows)
        rows = [r for r in rows if (r["rec"], r["tid"]) in keep]
        print(f"  只保留人工确认是手的轨迹：{len(keep)} 条，"
              f"{n0} -> {len(rows)} 帧-手")
    report(rows)


if __name__ == "__main__":
    main()
