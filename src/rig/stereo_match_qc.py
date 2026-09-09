"""Are the front module's odd pairs two different hands, or one hand twice?

THIS SHEET WAS BUILT TO EXPLAIN 375 IMPOSSIBLE MATCHES AND THEY TURNED OUT NOT
TO BE IMPOSSIBLE. A grid back-projected through the KB4 model gave 2145
positive disparities and no negative one, which read as proof that a negative
disparity meant a wrong pair -- but it was one databag's calibration. Module
A's far-field disparity runs from +88 px to -128 px across the twenty-nine
recordings, and where it is -128 a hand half a metre away is SUPPOSED to come
out near -50. Judged against its own pixel's depth band, 80% of the 150
negative pairs are perfectly ordinary hands at 0.36 m.

SO THE SHEET NOW ANSWERS THE QUESTION THAT SURVIVED. Two hypotheses make
opposite predictions and no amount of geometry can separate them, because
every geometric statistic here has already failed:

    H1  the pair is two different hands -- an association failure
    H2  it is one hand, and the box CENTRE is not a stable stereo landmark,
        because two views of a hand do not crop it the same way

Reprojection and epipolar error point the wrong way (the odd pairs score
BETTER on both), the triangulation angle does not separate, handedness agrees
96% either way, and the 21 keypoints add nothing: their epipolar residual is
3.13 against 2.97 and their depth spread is TIGHTER on the odd ones. A person
looking at two crops settles it in a second, and nothing else does.

WHAT REMAINS TRUE WITHOUT ANY MEASUREMENT is how the matcher works. It
projects the panorama box into each eye and takes, in each eye INDEPENDENTLY,
the detection nearest that point within 180 px. Left and right are never
compared. So a pair cost does not exist to be blamed or improved -- which is
also why the chosen pair is the cost minimum in 300 cases out of 300, and why
`best minus second best` was never going to say anything.

THE SHEET IS BLIND AND MIXES BOTH SIGNS. No geometry is shown, and the
positive half is judged too: a labeller who can tell which arm a case is in
turns the control into a second treatment. That is the same mistake that let
one defect read 2.2% and 46.9% in two sampling frames.

ONE REPAIR GETS TESTED, NOT THREE. M1 is M0 with the depth band as a
feasibility filter -- same seed, same radius, same cost -- and it differs from
M0 on 30 of the 300: it drops 22 and swaps 8. If that cannot remove the
confirmed mismatches while keeping the confirmed good pairs, the failure is in
the landmark rather than the association, and the next thing to try is a wrist
rather than a better cost.
"""
from __future__ import annotations

import argparse
import base64
import collections
import csv
import json
import math
import os
import random

# The deployed matcher's only parameter: how far from the predicted pixel an
# eye detection may sit and still be taken. Reproduced here rather than
# imported so the QC keeps working if coverage's copy is changed.
SEED_RADIUS = 180.0
# Candidates considered by the QC. Wider than the matcher's own radius on
# purpose: the pair it SHOULD have taken may be outside the radius it used,
# and that difference is one of the answers being looked for.
QC_RADIUS = 420.0

PATTERNS = [("same", "同一只手（配对正确）"),
            ("wearer_lr", "佩戴者左手 ↔ 佩戴者右手"),
            ("wearer_other", "佩戴者的手 ↔ 别人的手"),
            ("two_others", "两个不同的别人的手"),
            ("not_hand", "有一边根本不是手"),
            ("unclear", "看不清")]

SHEET = """<meta charset=utf-8><title>module_A pair QC</title><style>
body{font:13px/1.5 system-ui;margin:0;background:#111;color:#ddd}
#bar{position:sticky;top:0;background:#181818;padding:9px 14px;z-index:9;
  border-bottom:1px solid #333;display:flex;gap:14px;align-items:center;
  flex-wrap:wrap}
button{font:13px system-ui;padding:5px 10px;cursor:pointer}
b{color:#ffd33d}
.t{padding:9px 14px;border-bottom:1px solid #262626;display:flex;gap:10px;
  align-items:flex-start}
.t.cur{background:#1d2430;outline:2px solid #4a8}
.t.done{border-left:5px solid #2a6}
.meta{min-width:170px;color:#9ab;font-size:12px}
.num{color:#8ab4c8;font-family:ui-monospace,monospace;font-size:11px}
.strip{display:flex;gap:5px;flex:1;align-items:flex-start}
.strip figure{margin:0;flex:1}
.strip figure.pano{flex:2.2}
.strip img{width:100%;border-radius:3px;display:block}
.strip figcaption{font-size:10px;color:#888;text-align:center}
.key{font-size:11px;color:#999}
</style>
<div id=bar><span id=prog></span><span id=keys></span>
<button onclick="dl()">download CSV</button>
<span class=key>左眼和右眼各一张裁剪，右边是同一时刻的全景（绿框=全景检测）。
问的是：这两张裁剪里的，是不是同一只手。故意不显示任何几何数字。</span></div>
<div id=list></div>
<script>
const D = __PAYLOAD__;
const P = __PATTERNS__;
const KEY = "pairqc:" + D.tag;
let lab = {}, cur = 0, hist = [];
try { lab = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch(e) { lab={}; }
const list = document.getElementById("list");
D.pairs.forEach((t, i) => {
  const d = document.createElement("div");
  d.className = "t"; d.id = "t" + i;
  d.innerHTML = '<div class=meta>' + t.rec + ' f' + t.frame +
    ' <span id=v' + i + '></span></div>' +
    '<div class=strip>' +
    '<figure><img src="' + t.l + '"><figcaption>左眼</figcaption></figure>' +
    '<figure><img src="' + t.r + '"><figcaption>右眼</figcaption></figure>' +
    '<figure class=pano><img src="' + t.p +
    '"><figcaption>全景</figcaption></figure></div>';
  d.onclick = () => { cur = i; draw(); };
  list.appendChild(d);
});
function draw(){
  D.pairs.forEach((t,i)=>{
    const v = lab[t.key];
    document.getElementById("t"+i).className =
      "t" + (v ? " done" : "") + (i===cur ? " cur" : "");
    document.getElementById("v"+i).innerHTML = v ?
      '<b>' + (P.find(p => p[0] === v) || ["",v])[1] + '</b>' : '';
  });
  document.getElementById("keys").innerHTML =
    P.map((p,k) => '<b>' + (k+1) + '</b> ' + p[1]).join(" &nbsp; ") +
    ' &nbsp;&nbsp; <b>&uarr;&darr;</b> 换 &nbsp; <b>u</b> undo';
  const n = D.pairs.filter(t => lab[t.key]).length;
  document.getElementById("prog").innerHTML =
    "<b>" + D.tag + "</b> &nbsp; " + n + "/" + D.pairs.length + " 已判";
  localStorage.setItem(KEY, JSON.stringify(lab));
  const el = document.getElementById("t"+cur);
  if(el) el.scrollIntoView({block:"nearest"});
}
document.onkeydown = ev => {
  const n = +ev.key;
  if(n >= 1 && n <= P.length){
    const t = D.pairs[cur]; if(!t) return;
    hist.push([t.key, lab[t.key]]);
    lab[t.key] = P[n-1][0];
    cur = Math.min(cur+1, D.pairs.length-1); draw();
  }
  else if(ev.key==="ArrowDown"){cur=Math.min(cur+1,D.pairs.length-1);draw();}
  else if(ev.key==="ArrowUp"){cur=Math.max(cur-1,0);draw();}
  else if(ev.key==="u"){const h=hist.pop(); if(h){ if(h[1]===undefined)
    delete lab[h[0]]; else lab[h[0]]=h[1]; draw(); }}
  else return;
  ev.preventDefault();
};
draw();
function dl(){
  let s = "key,rec,frame,pattern\\n";
  for(const t of D.pairs) if(lab[t.key])
    s += t.key + "," + t.rec + "," + t.frame + "," + lab[t.key] + "\\n";
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([s],{type:"text/csv"}));
  a.download = "pair_qc_" + D.tag + ".csv"; a.click();
}
</script>
"""

FIELDS = ("key", "rec", "frame", "tid", "module", "reference_owner",
          "pano_u", "pano_v", "pano_w_frac",
          "li", "ri", "n_left", "n_right", "n_pairs", "n_feasible",
          "chosen", "chosen_by_matcher",
          "seed_d_l", "seed_d_r", "cur_cost", "cost_rank", "cost_margin",
          "side_l", "side_r", "side_agree", "conf_l", "conf_r",
          "size_ratio", "disparity_px", "epipolar_px", "angle_deg",
          "reproj_px", "range_m", "cheiral_ok",
          "ul", "vl", "ur", "vr", "d_at_near", "d_at_far", "band_ok",
          "z_implied",
          "kp_n", "kp_epi_med", "kp_disp_pos", "kp_depth_med", "kp_depth_iqr")

# The depth range a hand in this corpus can plausibly be at. Deliberately
# generous at both ends: this is a feasibility bound, not a prior.
Z_NEAR, Z_FAR = 0.15, 2.0
# How far outside the predicted band a correct pair may still fall. Two views
# of the same hand do not put their box centres on the same 3D point -- the
# silhouette differs -- so a few pixels of slack is physics, not fudge.
BAND_SLACK_PX = 20.0


def disparity_at(pix, Z, m, cv2, np, s3):
    """Predicted uL-uR for a left-eye pixel if its point were Z metres away.

    THIS IS THE FEASIBILITY CONSTRAINT `disparity > 0` WAS STANDING IN FOR.
    Module A's own predicted disparity runs from +9 px to +180 at 0.3 m
    depending on where in the fisheye the point sits, so a global sign test
    has almost no margin there while a global magnitude test would be wrong
    everywhere. Per pixel, the band is narrow and the test is sharp."""
    ray = s3.undistort([pix], m.left, cv2, np)[0]
    R = np.asarray(m.left.R, np.float64)
    t = np.asarray(m.left.t, np.float64).reshape(3)
    P1 = R @ np.array([ray[0] * Z, ray[1] * Z, Z]) + t

    def to_pix(cam):
        q = s3.proj_matrix(cam, np) @ np.append(P1, 1.0)
        if q[2] <= 0:
            return None
        n = (q[:2] / q[2]).reshape(1, 1, 2)
        p = cv2.fisheye.distortPoints(
            np.asarray(n, np.float64), np.asarray(cam.K, np.float64),
            np.asarray(cam.D, np.float64).reshape(4, 1))
        return p.reshape(2)

    pl, pr = to_pix(m.left), to_pix(m.right)
    if pl is None or pr is None:
        return None
    return float(pl[0] - pr[0])


def pair_features(dl, dr, m, cv2, np, s3):
    """Everything about one candidate (left det, right det) pair."""
    def ctr(d):
        b = d["box"]
        return ((float(b[0]) + float(b[2])) / 2.0,
                (float(b[1]) + float(b[3])) / 2.0)

    def area(d):
        b = d["box"]
        return max(1.0, (float(b[2]) - float(b[0]))
                   * (float(b[3]) - float(b[1])))

    cl, cr = ctr(dl), ctr(dr)
    pL = s3.undistort([cl], m.left, cv2, np)[0]
    pR = s3.undistort([cr], m.right, cv2, np)[0]
    X, err = s3.triangulate(pL, pR, m.left, m.right, cv2, np)
    R = np.asarray(m.left.R, np.float64)
    t = np.asarray(m.left.t, np.float64).reshape(3)
    XL = R.T @ (X - t)
    R2 = np.asarray(m.right.R, np.float64)
    t2 = np.asarray(m.right.t, np.float64).reshape(3)
    XR = R2.T @ (X - t2)
    out = {
        "disparity_px": round(cl[0] - cr[0], 1),
        "epipolar_px": round(s3.epipolar_px(pL, pR, m.left, m.right, np), 2),
        "angle_deg": round(s3.tri_angle_deg(X, m.left, m.right, np), 2),
        "reproj_px": round(float(err), 2),
        "range_m": round(float(np.linalg.norm(XL)), 4),
        "cheiral_ok": int(XL[2] > 0 and XR[2] > 0),
        "side_l": dl.get("side", ""), "side_r": dr.get("side", ""),
        "side_agree": int(dl.get("side") == dr.get("side")),
        "conf_l": round(float(dl["conf"]), 3),
        "conf_r": round(float(dr["conf"]), 3),
        "size_ratio": round(min(area(dl), area(dr))
                            / max(area(dl), area(dr)), 3),
        "ul": round(cl[0], 1), "vl": round(cl[1], 1),
        "ur": round(cr[0], 1), "vr": round(cr[1], 1),
    }
    d_near = disparity_at(cl, Z_NEAR, m, cv2, np, s3)
    d_far = disparity_at(cl, Z_FAR, m, cv2, np, s3)
    obs = out["disparity_px"]
    out["d_at_near"] = round(d_near, 1) if d_near is not None else ""
    out["d_at_far"] = round(d_far, 1) if d_far is not None else ""
    if d_near is None or d_far is None:
        out["band_ok"] = ""
        out["z_implied"] = ""
    else:
        lo, hi = min(d_near, d_far), max(d_near, d_far)
        out["band_ok"] = int(lo - BAND_SLACK_PX <= obs <= hi + BAND_SLACK_PX)
        # Disparity is monotone in 1/Z, so one bisection reads the depth the
        # observed disparity is claiming. A hand at eight metres is a verdict.
        a_, b_ = Z_NEAR, Z_FAR
        z = ""
        if lo <= obs <= hi:
            for _ in range(40):
                mid = 0.5 * (a_ + b_)
                dm = disparity_at(cl, mid, m, cv2, np, s3)
                if dm is None:
                    break
                if (dm > obs) == (d_near > d_far):
                    a_ = mid
                else:
                    b_ = mid
            z = round(0.5 * (a_ + b_), 3)
        out["z_implied"] = z
    # THE 21-POINT TEST. Two arbitrary rays triangulate to something and
    # reproject well; twenty-one correspondences agreeing on one epipolar
    # geometry AND one depth is what a wrong pair cannot fake.
    kl, kr = dl.get("kp"), dr.get("kp")
    epi, dep, pos, n = [], [], 0, 0
    if kl is not None and kr is not None and len(kl) == len(kr):
        for a_, b_ in zip(np.asarray(kl, np.float64),
                          np.asarray(kr, np.float64)):
            if not (a_[0] > 0 and a_[1] > 0 and b_[0] > 0 and b_[1] > 0):
                continue
            qL = s3.undistort([(a_[0], a_[1])], m.left, cv2, np)[0]
            qR = s3.undistort([(b_[0], b_[1])], m.right, cv2, np)[0]
            epi.append(s3.epipolar_px(qL, qR, m.left, m.right, np))
            Y, _ = s3.triangulate(qL, qR, m.left, m.right, cv2, np)
            dep.append(float((R.T @ (Y - t))[2]))
            pos += int(a_[0] - b_[0] > 0)
            n += 1
    out["kp_n"] = n
    if n >= 3:
        epi.sort()
        dep.sort()
        q1, q3 = dep[len(dep) // 4], dep[(3 * len(dep)) // 4]
        out["kp_epi_med"] = round(epi[len(epi) // 2], 2)
        out["kp_disp_pos"] = round(pos / n, 3)
        out["kp_depth_med"] = round(dep[len(dep) // 2], 4)
        out["kp_depth_iqr"] = round(q3 - q1, 4)
    else:
        out["kp_epi_med"] = out["kp_disp_pos"] = ""
        out["kp_depth_med"] = out["kp_depth_iqr"] = ""
    return out, (cl, cr)


def build(a):
    import cv2
    import numpy as np
    from ultralytics import YOLO
    from src.rig.calibration import RigCalibration
    from src.rig.geometry import VirtualWideCamera, source_maps
    from src.rig.render_wide import render
    from src.rig.seam_fix import ClipReader
    from src.rig.hand_detect import detect
    from src.rig import stereo3d as s3

    cov = list(csv.DictReader(open(a.cov, encoding="utf-8-sig")))
    want = [r for r in cov
            if r["module"] == a.module and r["matched"] == "1"]
    neg = [r for r in want if float(r["disparity_px"]) <= 0]
    pos = [r for r in want if float(r["disparity_px"]) > 0]
    rnd = random.Random(a.seed)
    rnd.shuffle(neg)
    rnd.shuffle(pos)
    sel = neg[:a.n_neg] + pos[:a.n_pos]
    print(f"  {a.module}: 负视差 {len(neg)} 取 {min(a.n_neg, len(neg))}   "
          f"正视差 {len(pos)} 取 {min(a.n_pos, len(pos))}")

    by_rec = collections.defaultdict(list)
    for r in sel:
        by_rec[r["rec"]].append(r)

    clips = {}
    for line in open(a.clips):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = line.rsplit(":", 2)
        clips[os.path.basename(p[0].rstrip("/")).replace("databag-26_", "R")] \
            = (p[0], int(p[1]))

    model = YOLO(a.weights)
    dump, cards = [], []
    for tag in sorted(by_rec):
        databag, start = clips[tag]
        rig = RigCalibration(os.path.join(databag, "calibration.yaml"))
        vcam = VirtualWideCamera.from_rig(rig)
        mods = rig.modules() if callable(rig.modules) else rig.modules
        m = next(x for x in mods if x.name == a.module)
        vids = {k: os.path.join(databag, f"{k}.mp4")
                for k in ("cam12", "cam34", "cam56")}
        need = collections.defaultdict(list)
        for r in by_rec[tag]:
            need[int(r["frame"])].append(r)
        lo, hi = min(need), max(need)
        rd = ClipReader(rig, vids, lo)
        mc, maps = {}, {}
        print(f"  {tag}: {len(by_rec[tag])} 个检测, 读 {lo}-{hi}", flush=True)
        for k in range(hi - lo + 1):
            src = rd.next()
            if not src:
                break
            f = lo + k
            if f not in need:
                continue
            try:
                rgb, _, _, _ = render(rig, vcam, src, 0.6, map_cache=mc)
            except TypeError:
                rgb, _, _, _ = render(rig, vcam, src, 0.6)
            H, W = rgb.shape[:2]
            eye = {}
            for cam in (m.left, m.right):
                if cam.name not in src:
                    continue
                eye[cam.name] = detect(model, src[cam.name], min_conf=a.conf)
                if cam.name not in maps:
                    # `source_maps` looks the camera up by name; handing it
                    # the object makes rig.cameras[...] hash an ndarray.
                    maps[cam.name] = source_maps(rig, cam.name, vcam, 0.6)

            for r in need[f]:
                u = int(round(float(r["u_frac"]) * W))
                v = int(round(float(r["v_frac"]) * H))
                u, v = min(max(u, 0), W - 1), min(max(v, 0), H - 1)
                seeds = {}
                for side, cam in (("l", m.left), ("r", m.right)):
                    mp = maps.get(cam.name)
                    if mp is None or not mp[2][v, u]:
                        continue
                    seeds[side] = (float(mp[0][v, u]), float(mp[1][v, u]))
                if len(seeds) < 2:
                    continue
                L = eye.get(m.left.name, [])
                R = eye.get(m.right.name, [])

                def near(dets, seed, rad):
                    out = []
                    for i, d in enumerate(dets):
                        cx = (float(d["box"][0]) + float(d["box"][2])) / 2.0
                        cy = (float(d["box"][1]) + float(d["box"][3])) / 2.0
                        dd = math.hypot(cx - seed[0], cy - seed[1])
                        if dd < rad:
                            out.append((i, d, dd))
                    return out

                cl = near(L, seeds["l"], a.radius)
                cr = near(R, seeds["r"], a.radius)
                if not cl or not cr:
                    continue
                # The deployed matcher: nearest in each eye INDEPENDENTLY,
                # inside 180 px. Reproduced, not imported, so the QC keeps
                # meaning the same thing if coverage's copy moves.
                pick_l = min((x for x in cl if x[2] < SEED_RADIUS),
                             key=lambda x: x[2], default=None)
                pick_r = min((x for x in cr if x[2] < SEED_RADIUS),
                             key=lambda x: x[2], default=None)
                rowset = []
                for il, dl, ddl in cl:
                    for ir, dr, ddr in cr:
                        feat, (pl, pr) = pair_features(dl, dr, m, cv2, np, s3)
                        row = {"key": f"{tag}:{f}:{r['tid']}",
                               "rec": tag, "frame": f, "tid": r["tid"],
                               "module": a.module,
                               "reference_owner": r["reference_owner"],
                               "pano_u": u, "pano_v": v,
                               "pano_w_frac": r["w_frac"],
                               "li": il, "ri": ir,
                               "n_left": len(cl), "n_right": len(cr),
                               "seed_d_l": round(ddl, 1),
                               "seed_d_r": round(ddr, 1),
                               "cur_cost": round(ddl + ddr, 1),
                               "chosen": 0, "chosen_by_matcher": 0}
                        row.update(feat)
                        row["_dl"], row["_dr"] = dl, dr
                        row["_pl"], row["_pr"] = pl, pr
                        rowset.append(row)
                if not rowset:
                    continue
                feas = [x for x in rowset
                        if x["band_ok"] == 1 and x["cheiral_ok"]]
                order = sorted(rowset, key=lambda x: x["cur_cost"])
                for rank, x in enumerate(order):
                    x["cost_rank"] = rank
                    x["cost_margin"] = round(
                        order[1]["cur_cost"] - order[0]["cur_cost"], 1) \
                        if len(order) > 1 else ""
                    x["n_pairs"] = len(rowset)
                    x["n_feasible"] = len(feas)
                chosen = None
                if pick_l is not None and pick_r is not None:
                    for x in rowset:
                        if x["li"] == pick_l[0] and x["ri"] == pick_r[0]:
                            x["chosen"] = x["chosen_by_matcher"] = 1
                            chosen = x
                if chosen is None:
                    chosen = order[0]
                    chosen["chosen"] = 1
                for x in rowset:
                    dump.append({k2: x.get(k2, "") for k2 in FIELDS})
                card_of = chosen
                if a.m1_diff:
                    # THE EIGHT PAIRS M1 SWAPS TO. They are the only new
                    # thing it produces that has never been looked at, and
                    # without a verdict on them the ablation cannot say
                    # whether a swap was a repair or a second mistake.
                    p1 = m1_pick(rowset)
                    if p1 is None or (p1["li"], p1["ri"]) == (chosen["li"],
                                                              chosen["ri"]):
                        continue
                    card_of = p1
                if a.out:
                    # ENCODED HERE, NOT LATER. Holding the panorama and two
                    # 1920x1520 eyes for three hundred cards is six gigabytes;
                    # the three JPEGs are twenty kilobytes.
                    cards.append(card(a, tag, f, r, card_of,
                                      rgb, src, m, cv2))

        rd.close()

    with open(a.out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(FIELDS))
        w.writeheader()
        w.writerows(dump)
    print(f"  {len(dump)} 个候选对 -> {a.out_csv}")
    report(dump)

    if a.out:
        write_sheet(a, cards, cv2)


def crop(img, cx, cy, half, cv2, box=None, w=200, q=82):
    H, W = img.shape[:2]
    x0, y0 = max(0, int(cx - half)), max(0, int(cy - half))
    x1, y1 = min(W, int(cx + half)), min(H, int(cy + half))
    c = img[y0:y1, x0:x1].copy()
    if c.size == 0:
        return ""
    if box is not None:
        cv2.rectangle(c, (int(box[0]) - x0, int(box[1]) - y0),
                      (int(box[2]) - x0, int(box[3]) - y0), (60, 255, 60), 2)
    h = int(round(c.shape[0] * w / max(1, c.shape[1])))
    ok, buf = cv2.imencode(".jpg", cv2.resize(c, (w, max(1, h))),
                           [int(cv2.IMWRITE_JPEG_QUALITY), q])
    return ("data:image/jpeg;base64," +
            base64.b64encode(buf).decode()) if ok else ""


def m1_pick(rowset):
    """M0 plus one thing: the candidate must lie in its own depth band.

    NOT A NEW MATCHER. Same seed projection, same 180 px radius, same
    nearest-seed cost; the band only removes candidates that no real point at
    that pixel could have produced. If it cannot separate the confirmed
    mismatches from the confirmed correct pairs, the analytic repair is out of
    moves and the failure is in the landmark, not in the association."""
    cand = [x for x in rowset if x["seed_d_l"] < SEED_RADIUS
            and x["seed_d_r"] < SEED_RADIUS]
    feas = [x for x in cand if x["band_ok"] == 1 and x["cheiral_ok"]]
    return min(feas, key=lambda x: x["cur_cost"]) if feas else None


def ablate(a):
    """M0 vs M1 on the pairs a person has actually judged.

    THE LABELS ARE ON M0'S OUTPUT, so M1 can only be scored where it agrees
    with M0 or where its own pick has been judged too. It agrees on 270 of the
    300, rejects outright on 22 -- which needs no new label, since the
    question there is only whether what it dropped was wrong -- and swaps the
    pair on 8, which is why there is a second small sheet."""
    rows = list(csv.DictReader(open(a.cov, encoding="utf-8-sig")))
    for r in rows:
        for k in ("seed_d_l", "seed_d_r", "cur_cost", "disparity_px"):
            r[k] = float(r[k])
        for k in ("band_ok", "cheiral_ok"):
            r[k] = int(r[k]) if r[k] not in ("", None) else 0
    lab = {}
    for p in (a.labels or []) + (a.m1_labels or []):
        for r in csv.DictReader(open(p, encoding="utf-8-sig")):
            lab[r["key"]] = r["pattern"]
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["rec"], r["frame"], r["tid"])].append(r)

    def key(k, x):
        return f"{k[0]}:{k[1]}:{k[2]}:{x['li']}:{x['ri']}"

    kept_ok = kept_bad = dropped_ok = dropped_bad = 0
    swap = collections.Counter()
    unjudged = 0
    for k, v in by.items():
        ch = next((x for x in v if int(x["chosen"])), None)
        if ch is None:
            continue
        v0 = lab.get(key(k, ch))
        if v0 is None:
            continue
        if v0 == "unclear":
            unjudged += 1
            continue
        good0 = v0 == "same"
        p = m1_pick(v)
        if p is None:
            if good0:
                dropped_ok += 1
            else:
                dropped_bad += 1
        elif (p["li"], p["ri"]) == (ch["li"], ch["ri"]):
            if good0:
                kept_ok += 1
            else:
                kept_bad += 1
        else:
            v1 = lab.get(key(k, p))
            swap[(v0, v1 or "未标")] += 1

    n_ok = kept_ok + dropped_ok + sum(n for (a_, b_), n in swap.items()
                                      if a_ == "same")
    n_bad = kept_bad + dropped_bad + sum(n for (a_, b_), n in swap.items()
                                         if a_ != "same")
    print(f"\n=== M0 vs M1（人工确认集，n={n_ok + n_bad}，"
          f"另有 {unjudged} 个『看不清』不计）===")
    print(f"  M0 判对 {n_ok}   M0 判错 {n_bad}   "
          f"M0 correct-match rate {n_ok / max(1, n_ok + n_bad):.1%}")
    print("\n  M1 对 M0 的每一类做了什么")
    print(f"    对的留下   {kept_ok:>4}      对的被丢   {dropped_ok:>4}")
    print(f"    错的留下   {kept_bad:>4}      错的被丢   {dropped_bad:>4}")
    for (v0, v1), n in sorted(swap.items(), key=lambda x: -x[1]):
        print(f"    换了对     M0={v0} -> M1={v1}   {n}")
    swap_ok = sum(n for (a_, b_), n in swap.items() if b_ == "same")
    swap_from_bad = sum(n for (a_, b_), n in swap.items() if a_ != "same")
    rej = dropped_bad + swap_from_bad
    print(f"\n  false-match rejection  {rej}/{n_bad} = "
          f"{rej / max(1, n_bad):.1%}   （丢掉或换掉的错配）")
    keep = kept_ok + sum(n for (a_, b_), n in swap.items()
                         if a_ == "same" and b_ == "same")
    print(f"  correct-match recall   {keep}/{n_ok} = "
          f"{keep / max(1, n_ok):.1%}   （原本判对且 M1 仍给出正确对）")
    print(f"  换对后变成正确的        {swap_ok}")
    print("\n  判据（事前写死，不调阈值）：rejection 明显提高且 recall 基本不掉"
          "\n  → matcher 可修，重跑 Gate A/B/C；否则 bbox 中心不是可靠的"
          "\n  stereo landmark，只值得试一次 wrist/稳定关键点三角化。")


def card(a, tag, f, r, ch, rgb, src, m, cv2):
    dl, dr = ch["_dl"], ch["_dr"]
    pl, pr = ch["_pl"], ch["_pr"]
    iml, imr = src.get(m.left.name), src.get(m.right.name)
    if iml is None or imr is None:
        return None

    def half(d):
        return max(60, int(max(float(d["box"][2]) - float(d["box"][0]),
                               float(d["box"][3]) - float(d["box"][1]))
                           * a.ctx_scale / 2))

    W = rgb.shape[1]
    wf = float(r["w_frac"]) * W
    pano = rgb.copy()
    cv2.rectangle(pano, (int(ch["pano_u"] - wf / 2),
                         int(ch["pano_v"] - wf / 2)),
                  (int(ch["pano_u"] + wf / 2),
                   int(ch["pano_v"] + wf / 2)), (60, 255, 60), 3)
    return {"key": f"{tag}:{f}:{r['tid']}:{ch['li']}:{ch['ri']}",
            "rec": tag, "frame": f,
            "l": crop(iml, pl[0], pl[1], half(dl), cv2, dl["box"], a.crop_w),
            "r": crop(imr, pr[0], pr[1], half(dr), cv2, dr["box"], a.crop_w),
            "p": crop(pano, ch["pano_u"], ch["pano_v"],
                      max(220, wf * 2.5), cv2, None, a.pano_w)}


def write_sheet(a, cards, cv2):
    out = [c for c in cards if c]
    # BLIND. Negative and positive disparity mixed, no geometry shown: a
    # labeller who can tell which arm a case is in makes the control useless.
    random.Random(a.seed + 7).shuffle(out)
    stem, ext = os.path.splitext(a.out)
    n_pg = max(1, (len(out) + a.page - 1) // a.page)
    for i in range(n_pg):
        part = out[i * a.page:(i + 1) * a.page]
        path = a.out if n_pg == 1 else f"{stem}_{i + 1}{ext}"
        tag = "all" if n_pg == 1 else f"p{i + 1}"
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(SHEET.replace("__PAYLOAD__",
                                   json.dumps({"tag": tag, "pairs": part}))
                     .replace("__PATTERNS__",
                              json.dumps([[k, v] for k, v in PATTERNS],
                                         ensure_ascii=False)))
        print(f"  {len(part):>3} 对 -> {path} "
              f"({os.path.getsize(path) / 1e6:.1f} MB)")
    for i, (k, v) in enumerate(PATTERNS):
        print(f"  {i + 1} {v}")


def report(dump):
    ch = [r for r in dump if int(r["chosen"])]
    if not ch:
        return
    neg = [r for r in ch if float(r["disparity_px"]) <= 0]
    pos = [r for r in ch if float(r["disparity_px"]) > 0]
    print(f"\n=== 被选中的对 {len(ch)}（负 {len(neg)} / 正 {len(pos)}）===")

    print("\n  被选中的对落在自己那条深度带里吗"
          f"（每个左眼像素单独算 {Z_NEAR}-{Z_FAR}m 的预测视差区间）")
    for name, s in (("负视差", neg), ("正视差", pos)):
        if not s:
            continue
        ok = sum(1 for r in s if str(r["band_ok"]) == "1")
        zs = sorted(float(r["z_implied"]) for r in s
                    if r["z_implied"] not in ("", None))
        zt = f"   带内那些的隐含深度中位 {zs[len(zs)//2]:.2f}m" if zs else ""
        print(f"    {name}  n={len(s)}   在带内 {ok} ({ok/len(s):.0%}){zt}")
    print("    视差为正只是这条带的一个很松的下界。module_A 的预测视差"
          "在 0.3m 处随像素从 +9 到 +180，所以『正』几乎没有余量。")

    print("\n  这到底是配错还是漏检？"
          "（n_feasible = 该检测的所有候选对里落在深度带内且在两相机前方的个数）")
    for name, s in (("负视差", neg), ("正视差", pos)):
        if not s:
            continue
        z = sum(1 for r in s if int(r["n_feasible"]) == 0)
        one = sum(1 for r in s if int(r["n_feasible"]) == 1)
        more = len(s) - z - one
        print(f"    {name}  n={len(s)}   没有可行对 {z} ({z/len(s):.0%})"
              f"   恰好一个 {one} ({one/len(s):.0%})"
              f"   多于一个 {more} ({more/len(s):.0%})")
    print("    『没有可行对』= 有一只眼睛根本没检到这只手，那是检测问题，"
          "换 cost function 救不了。")

    print("\n  同帧候选密度（组合歧义的直接证据）")
    for name, s in (("负视差", neg), ("正视差", pos)):
        if not s:
            continue
        nl = sorted(int(r["n_left"]) for r in s)
        nr = sorted(int(r["n_right"]) for r in s)
        np_ = sorted(int(r["n_pairs"]) for r in s)
        multi = sum(1 for r in s if int(r["n_pairs"]) > 1)
        print(f"    {name}  左眼候选中位 {nl[len(nl)//2]}  "
              f"右眼 {nr[len(nr)//2]}  候选对中位 {np_[len(np_)//2]}  "
              f"不止一个候选对 {multi}/{len(s)} = {multi/len(s):.0%}")

    print("\n  现有 cost（两眼各自到种子点的距离之和）能不能分开")
    for name, s in (("负视差", neg), ("正视差", pos)):
        if not s:
            continue
        c = sorted(float(r["cur_cost"]) for r in s)
        rk = collections.Counter(int(r["cost_rank"]) for r in s)
        print(f"    {name}  cost 中位 {c[len(c)//2]:.0f}px   "
              f"被选中的对在 cost 排序里排第一的比例 "
              f"{rk[0]}/{len(s)} = {rk[0]/len(s):.0%}")

    print("\n  各特征在两组上的中位（能不能当 feasibility 用）")
    keys = ("epipolar_px", "angle_deg", "reproj_px", "range_m",
            "size_ratio", "kp_epi_med", "kp_disp_pos", "kp_depth_iqr")
    print(f"    {'':<14}{'负视差':>12}{'正视差':>12}")
    for k in keys:
        def med(s):
            v = sorted(float(r[k]) for r in s if r[k] not in ("", None))
            return f"{v[len(v)//2]:.3g}" if v else "—"
        print(f"    {k:<14}{med(neg):>12}{med(pos):>12}")
    for name, s in (("负视差", neg), ("正视差", pos)):
        if s:
            ag = sum(int(r["side_agree"]) for r in s)
            print(f"    {name} 左右眼手别一致 {ag}/{len(s)} = {ag/len(s):.0%}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cov", required=True, help="stereo_coverage csv")
    ap.add_argument("--clips", default="/workspace/e2e_main2.txt")
    ap.add_argument("--module", default="module_A")
    ap.add_argument("--n_neg", type=int, default=150)
    ap.add_argument("--n_pos", type=int, default=150)
    ap.add_argument("--radius", type=float, default=QC_RADIUS)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--weights",
                    default="/shared/models/HaWoR/weights/external/detector.pt")
    ap.add_argument("--crop_w", type=int, default=190)
    ap.add_argument("--pano_w", type=int, default=380)
    ap.add_argument("--ctx_scale", type=float, default=3.0)
    ap.add_argument("--page", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_csv", help="candidate-pair dump")
    ap.add_argument("--out", help="QC sheet html")
    ap.add_argument("--m1_diff", action="store_true",
                    help="sheet only the pairs M1 swaps to")
    ap.add_argument("--ablate", action="store_true",
                    help="score M0 vs M1 on the judged pairs")
    ap.add_argument("--labels", nargs="*", help="pair_qc csvs")
    ap.add_argument("--m1_labels", nargs="*", help="the M1-swap sheet's csv")
    a = ap.parse_args()
    if a.ablate:
        ablate(a)
    else:
        if not a.out_csv:
            ap.error("--out_csv is required when building")
        build(a)


if __name__ == "__main__":
    main()
