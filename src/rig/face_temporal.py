"""How many blurred frames does one face proposal turn into?

THE OVER-BLUR ATTRIBUTION TRACED TWENTY-SIX OF THIRTY UNEXPLAINED COVERS TO A
FACE DETECTION ONE TO ELEVEN FRAMES EARLIER, and none to nothing at all. So
`residue` was never a separate defect: `Hold` keeps a face covered for
HOLD_FRAMES = 12 after the detector stops finding it, which turns a single
frame of detector output into four tenths of a second of mosaic. A detector
whose candidate precision is already 0.974 then contributes about half of all
false blur, because every false proposal is multiplied twelve-fold, and that
is why tightening the detector kept costing more than it returned.

THE QUANTITY IS AN AMPLIFICATION, NOT A RATE.

    amplification = frames finally blurred / frames the detector fired

A temporal mechanism worth having amplifies a TRUE face above one -- that is
what bridging a dropout means -- and a FALSE one at about one, because copying
a mistake forward is all that does. The two numbers have to be reported apart
or the good half pays for the bad half.

THE POLICIES ARE REPLAYED ON FROZEN CANDIDATES. The detector runs once and
every proposal is written down; T0, T1 and T2 are then simulated in pure
Python over identical boxes and identical scores, so nothing but the policy
can differ between arms. It also means the gap bound is not swept: G is fixed
at 12 to match the hold it replaces, because choosing it on the same errors it
is measured against is how a policy gets fitted to its own test set.

WHY NOT SIMPLY DELETE ISOLATED DETECTIONS. Because a real face is sometimes
seen for one frame too, and a rule that drops single observations drops both.
That asymmetry is the whole reason the true arm exists: a change that removes
false blur by removing coverage is not an improvement, it is a different
setting of the same dial the size cap and the confidence floor already turn.
"""
from __future__ import annotations

import argparse
import base64
import collections
import csv
import json
import math
import os

# Matching for both the hold and the run grouping. Same value as face_mask.Hold
# so T0 here is the deployed behaviour and not an approximation of it.
MIN_IOU = 0.2
# The gap bound for T2, fixed to the hold it replaces rather than tuned.
GAP = 12


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1])
    ub = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (ua + ub - inter)


def wilson(k, n, z=1.96):
    if not n:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / d, (c + m) / d)


# ---------------------------------------------------------------- dump

def dump(a):
    """Every face proposal the pipeline would act on, frame by frame."""
    import cv2
    from ultralytics import YOLO
    from src.rig import face_mask
    from src.rig.calibration import RigCalibration
    from src.rig.geometry import VirtualWideCamera
    from src.rig.render_wide import render
    from src.rig.seam_fix import ClipReader, Prefetch
    from src.rig.hand_detect import detect as hand_detect

    clips = {}
    for line in open(a.clips):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = line.rsplit(":", 2)
        tag = os.path.basename(p[0].rstrip("/")).replace("databag-26_", "R")
        clips[tag] = (p[0], int(p[1]))

    det = face_mask.load_detector(a.face_model, a.face_conf)
    hands = YOLO(a.weights)
    rows = []
    for tag in (a.rec or sorted(clips)):
        if tag not in clips:
            continue
        databag, start = clips[tag]
        rig = RigCalibration(os.path.join(databag, "calibration.yaml"))
        vcam = VirtualWideCamera.from_rig(rig)
        vids = {k: os.path.join(databag, f"{k}.mp4")
                for k in ("cam12", "cam34", "cam56")}
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
            faces = face_mask.detect_faces(det, rgb)
            # The veto ships, so a candidate it removes is not a candidate.
            hd = hand_detect(hands, rgb, min_conf=0.25)
            keep, vetoed = face_mask.split_on_hands(faces, hd)
            for f, v in [(x, 0) for x in keep] + [(x, 1) for x in vetoed]:
                rows.append({"rec": tag, "frame": start + k,
                             "x0": int(f[0]), "y0": int(f[1]),
                             "x1": int(f[2]), "y1": int(f[3]),
                             "score": round(float(f[4]), 4), "vetoed": v})
        rd.close()
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n  {len(rows)} 个人脸候选 -> {a.out}")


# ---------------------------------------------------------------- runs

def runs_of(props):
    """Chain per-frame proposals into runs. -> [ {rec, frames:{f:box}} ]

    A RUN IS A PROPERTY OF THE DETECTIONS, NOT OF ANY POLICY, so the same
    grouping is used for all three arms and a run means the same object in
    each. Chaining is by overlap with the most recent box, allowing a gap of
    up to GAP frames -- the same window the hold spans, so nothing is grouped
    that the hold would not itself have joined."""
    by = collections.defaultdict(list)
    for p in props:
        by[p["rec"]].append(p)
    out = []
    for rec, ps in sorted(by.items()):
        ps.sort(key=lambda r: (int(r["frame"]), -float(r["score"])))
        live = []                          # [ {last, frames} ]
        for p in ps:
            f = int(p["frame"])
            box = (int(p["x0"]), int(p["y0"]), int(p["x1"]), int(p["y1"]))
            best, best_v = None, MIN_IOU
            for r in live:
                if f - r["last"] > GAP:
                    continue
                v = iou(box, r["frames"][r["last"]])
                if v >= best_v:
                    best, best_v = r, v
            if best is None:
                live.append({"rec": rec, "last": f, "frames": {f: box}})
            else:
                best["frames"][f] = box
                best["last"] = f
        out += live
    for r in out:
        ks = sorted(r["frames"])
        r["first"], r["last"] = ks[0], ks[-1]
        r["n_det"] = len(ks)
        r["span"] = ks[-1] - ks[0] + 1
    return out


def covered_frames(run, policy, hold=12, gap=GAP):
    """Which frames a policy leaves a mosaic on. -> set of frame numbers

    T0/T1 are forward holds: every detection starts a countdown, and the cover
    survives until it expires. T2 never looks past the last observation -- it
    only bridges a gap that is closed at both ends."""
    det = sorted(run["frames"])
    out = set()
    if policy in ("T0", "T1"):
        h = hold if policy == "T0" else 2
        for f in det:
            out.update(range(f, f + h))
        return out
    if policy == "T2":
        out.update(det)
        for a, b in zip(det, det[1:]):
            if b - a <= gap:
                out.update(range(a, b + 1))
        return out
    raise ValueError(policy)


def report(props, labels, hold, gap):
    runs = runs_of(props)
    lab = {k: v for k, v in labels.items()}
    print(f"\n  {len(props)} 个候选  ->  {len(runs)} 条候选轨迹"
          f"   已判真假 {sum(1 for r in runs if key_of(r) in lab)}")

    print(f"\n  {'策略':<6} {'孤立候选(n_det=1)的糊帧':>22} {'总糊帧':>10} "
          f"{'放大倍数':>10} {'尾部帧':>9}")
    for pol in ("T0", "T1", "T2"):
        tot = iso = tail = ndet = 0
        for r in runs:
            cov = covered_frames(r, pol, hold, gap)
            tot += len(cov)
            ndet += r["n_det"]
            tail += sum(1 for f in cov if f > r["last"])
            if r["n_det"] == 1:
                iso += len(cov)
        print(f"  {pol:<6} {iso:>22} {tot:>10} {tot / ndet:>10.2f} "
              f"{tail:>9}")

    if not lab:
        print("\n  还没有真假标注，所以只有合计。真脸和误检必须分开报 —— "
              "一个把两者一起\n  压下去的策略不是改进，是把阈值往回调。")
        return
    print(f"\n  {'':<6} {'真脸放大':>10} {'真脸跨接':>10} "
          f"{'误检放大':>10} {'误检糊帧':>10}")
    for pol in ("T0", "T1", "T2"):
        t_cov = t_det = f_cov = f_det = 0
        bridged = span = 0
        for r in runs:
            v = lab.get(key_of(r))
            if v is None:
                continue
            cov = covered_frames(r, pol, hold, gap)
            if v:
                t_cov += len(cov); t_det += r["n_det"]
                # Bridging is only claimable inside the observed span; before
                # the first detection and after the last there is no evidence
                # the face was there at all.
                bridged += len([f for f in cov
                                if r["first"] <= f <= r["last"]])
                span += r["span"]
            else:
                f_cov += len(cov); f_det += r["n_det"]
        print(f"  {pol:<6} {t_cov / t_det if t_det else float('nan'):>10.2f} "
              f"{bridged / span if span else float('nan'):>10.3f} "
              f"{f_cov / f_det if f_det else float('nan'):>10.2f} "
              f"{f_cov:>10}")
    print("\n  真脸跨接 = 首末检测之间被盖住的比例。首检之前、末检之后没有任何")
    print("  证据说明脸还在，所以那些帧不计入跨接，只计入放大。")
    print("  理想的机制：真脸放大 > 1（补上漏检），误检放大 ≈ 1（不把错误复制下去）。")


def key_of(r):
    return f"{r['rec']}:{r['first']}"


# ---------------------------------------------------------------- sheet

SHEET = """<meta charset=utf-8><title>face runs</title><style>
body{font:13px/1.5 system-ui;margin:0;background:#111;color:#ddd}
#bar{position:sticky;top:0;background:#181818;padding:9px 14px;z-index:9;
  border-bottom:1px solid #333;display:flex;gap:14px;align-items:center}
button{font:13px system-ui;padding:5px 10px;cursor:pointer}
b{color:#ffd33d}
.r{padding:10px 14px;border-bottom:1px solid #262626;display:flex;gap:12px;
  align-items:flex-start}
.r.cur{background:#1d2430;outline:2px solid #4a8}
.r.yes{border-left:5px solid #2a6}
.r.no{border-left:5px solid #d33}
.meta{min-width:200px;color:#9ab;font-size:12px}
.num{color:#8ab4c8;font-family:ui-monospace,monospace;font-size:11px}
.strip{display:flex;gap:4px;flex:1}
.strip figure{margin:0;flex:1}
.strip img{width:100%;border-radius:3px;display:block}
.strip figcaption{font-size:10px;color:#888;text-align:center}
</style>
<div id=bar><span id=prog></span>
<span><b>1</b> 是脸 &nbsp; <b>2</b> 不是脸 &nbsp; <b>&uarr;&darr;</b> move
 &nbsp; <b>u</b> undo</span>
<button onclick="dl()">download CSV</button>
<span class=num>一条候选轨迹判一次。青框是检测器提出的位置。</span></div>
<div id=list></div>
<script>
const D = __PAYLOAD__;
const KEY = "ftmp:" + D.tag;
let lab = {}, cur = 0, hist = [];
try { lab = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch(e) { lab={}; }
const list = document.getElementById("list");
D.runs.forEach((c, i) => {
  const d = document.createElement("div");
  d.className = "r"; d.id = "r" + i;
  d.innerHTML = '<div class=meta>' + c.rec + '<br><span class=num>f' +
    c.first + '-' + c.last + '  检测 ' + c.n_det + ' 帧 / 跨度 ' + c.span +
    '<br>分数 ' + c.score + '</span> <span id=v' + i + '></span></div>' +
    '<div class=strip>' + c.frames.map(f =>
      '<figure><img src="' + f.img + '"><figcaption>f' + f.f +
      '</figcaption></figure>').join('') + '</div>';
  d.onclick = () => { cur = i; draw(); };
  list.appendChild(d);
});
function draw(){
  D.runs.forEach((c,i)=>{
    const v = lab[c.key];
    document.getElementById("r"+i).className =
      "r " + (v===undefined ? "" : (v ? "yes" : "no")) + (i===cur?" cur":"");
    document.getElementById("v"+i).innerHTML = v===undefined ? "" :
      (v ? '<b style="color:#5c5">是脸</b>' : '<b style="color:#d55">不是</b>');
  });
  document.getElementById("prog").innerHTML =
    "<b>" + D.tag + "</b> &nbsp; " + Object.keys(lab).length + "/" +
    D.runs.length + " 已判";
  localStorage.setItem(KEY, JSON.stringify(lab));
  const el = document.getElementById("r"+cur);
  if(el) el.scrollIntoView({block:"nearest"});
}
document.onkeydown = ev => {
  if(ev.key==="1"||ev.key==="2"){
    const c = D.runs[cur]; if(!c) return;
    hist.push([c.key, lab[c.key]]);
    lab[c.key] = ev.key==="1";
    cur = Math.min(cur+1, D.runs.length-1); draw();
  }
  else if(ev.key==="ArrowDown"){cur=Math.min(cur+1,D.runs.length-1);draw();}
  else if(ev.key==="ArrowUp"){cur=Math.max(cur-1,0);draw();}
  else if(ev.key==="u"){const h=hist.pop(); if(h){ if(h[1]===undefined)
    delete lab[h[0]]; else lab[h[0]]=h[1]; draw(); }}
  else return;
  ev.preventDefault();
};
draw();
function dl(){
  let s = "key,rec,first,last,n_det,is_face\\n";
  for(const c of D.runs) if(c.key in lab)
    s += c.key + "," + c.rec + "," + c.first + "," + c.last + "," +
         c.n_det + "," + (lab[c.key]?1:0) + "\\n";
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([s],{type:"text/csv"}));
  a.download = "face_runs.csv"; a.click();
}
</script>
"""


def sheet(a, props):
    import cv2
    from src.rig.calibration import RigCalibration
    from src.rig.geometry import VirtualWideCamera
    from src.rig.render_wide import render
    from src.rig.seam_fix import ClipReader, Prefetch

    clips = {}
    for line in open(a.clips):
        line = line.strip()
        if not line:
            continue
        p = line.rsplit(":", 2)
        clips[os.path.basename(p[0].rstrip("/")).replace("databag-26_", "R")] \
            = (p[0], int(p[1]))
    runs = runs_of(props)
    score = {}
    for p in props:
        score.setdefault((p["rec"], int(p["frame"])), float(p["score"]))
    by_rec = collections.defaultdict(list)
    for r in runs:
        by_rec[r["rec"]].append(r)

    out = []
    for rec, rs in sorted(by_rec.items()):
        if rec not in clips:
            continue
        databag, start = clips[rec]
        rig = RigCalibration(os.path.join(databag, "calibration.yaml"))
        vcam = VirtualWideCamera.from_rig(rig)
        vids = {k: os.path.join(databag, f"{k}.mp4")
                for k in ("cam12", "cam34", "cam56")}
        want = {}
        for r in rs:
            ks = sorted(r["frames"])
            pick = [ks[0], ks[len(ks) // 2], ks[-1]][:max(1, min(3, len(ks)))]
            for f in dict.fromkeys(pick):
                want.setdefault(f, []).append(r)
        lo = min(want)
        rd = Prefetch(ClipReader(rig, vids, lo), skip=0)
        mc, imgs = {}, {}
        print(f"  {rec}: {len(rs)} 条轨迹, 读 {lo}-{max(want)}", flush=True)
        for k in range(max(want) - lo + 1):
            src = rd.next()
            if not src:
                break
            f = lo + k
            if f not in want:
                continue
            try:
                rgb, _, _, _ = render(rig, vcam, src, 0.6, map_cache=mc)
            except TypeError:
                rgb, _, _, _ = render(rig, vcam, src, 0.6)
            imgs[f] = rgb
        rd.close()
        for r in rs:
            ks = sorted(r["frames"])
            pick = list(dict.fromkeys(
                [ks[0], ks[len(ks) // 2], ks[-1]]))
            frames = []
            for f in pick:
                img = imgs.get(f)
                if img is None:
                    continue
                x0, y0, x1, y1 = r["frames"][f]
                pad = int(max(x1 - x0, y1 - y0) * 1.2)
                cx0, cy0 = max(0, x0 - pad), max(0, y0 - pad)
                cx1 = min(img.shape[1], x1 + pad)
                cy1 = min(img.shape[0], y1 + pad)
                crop = img[cy0:cy1, cx0:cx1].copy()
                cv2.rectangle(crop, (x0 - cx0, y0 - cy0),
                              (x1 - cx0, y1 - cy0), (255, 220, 40), 2)
                h = int(round(crop.shape[0] * a.width / max(1, crop.shape[1])))
                ok, buf = cv2.imencode(".jpg",
                                       cv2.resize(crop, (a.width, max(1, h))),
                                       [int(cv2.IMWRITE_JPEG_QUALITY), 82])
                if ok:
                    frames.append({"f": f, "img": "data:image/jpeg;base64,"
                                   + base64.b64encode(buf).decode()})
            if frames:
                out.append({"key": key_of(r), "rec": rec, "first": r["first"],
                            "last": r["last"], "n_det": r["n_det"],
                            "span": r["span"],
                            "score": round(max(
                                score.get((rec, f), 0.0) for f in ks), 2),
                            "frames": frames})
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(SHEET.replace("__PAYLOAD__",
                              json.dumps({"tag": "faceruns", "runs": out})))
    print(f"\n  {len(out)} 条轨迹 -> {a.out} "
          f"({os.path.getsize(a.out) / 1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("dump", "sheet", "report"),
                    required=True)
    ap.add_argument("--clips", default="/workspace/e2e_main2.txt")
    ap.add_argument("--rec", action="append")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--props", help="the dumped candidate csv")
    ap.add_argument("--labels", help="a labelled face_runs csv")
    ap.add_argument("--hold", type=int, default=12)
    ap.add_argument("--gap", type=int, default=GAP)
    ap.add_argument("--width", type=int, default=180)
    ap.add_argument("--out")
    ap.add_argument("--face_model",
                    default="/workspace/models/yolov8n-face-lindevs.onnx")
    ap.add_argument("--face_conf", type=float, default=None)
    ap.add_argument("--weights",
                    default="/shared/models/HaWoR/weights/external/detector.pt")
    a = ap.parse_args()

    if a.mode == "dump":
        dump(a)
        return
    props = [r for r in csv.DictReader(open(a.props, encoding="utf-8-sig"))
             if not int(r.get("vetoed", 0))]
    if a.mode == "sheet":
        sheet(a, props)
        return
    labels = {}
    if a.labels:
        for r in csv.DictReader(open(a.labels, encoding="utf-8-sig")):
            labels[r["key"]] = bool(int(r["is_face"]))
    report(props, labels, a.hold, a.gap)


if __name__ == "__main__":
    main()
