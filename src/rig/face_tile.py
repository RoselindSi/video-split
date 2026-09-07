"""What slicing the frame buys in recall and what it costs in false covers.

93.5% OF THE FACES THIS PIPELINE DELIVERED UNCOVERED WERE NEVER PROPOSED.
Not vetoed, not mis-tracked, not missed by the mask: the detector put no box
there at all. So the only stage worth changing is the proposer, and the
proposer is running a static 640x640 ONNX on a 1.78:1 frame with a straight
resize -- every face is squeezed to 56% of its width before the network sees
it, on top of the downscale. The hand branch had exactly this disease and
IMGSZ 512 -> 1024 cured it; the face ONNX refuses every input size but 640,
so the resolution has to come from tiles.

TWO ARMS, BECAUSE RECALL ALONE WOULD LIE. Over-blur is already the most
frequent defect in the end-to-end audit -- more frequent than either kind of
miss -- so a change that finds the distant colleague and doubles the mosaics
on the workbench is not an improvement, and measuring only the faces it
rescues would hide that completely.

    RECALL     the frames a person marked as `a face was visible and it was
               not covered`, attributed to `detector never proposed`. How
               many of them does the sliced pass now propose?
    PRECISION  frames drawn at random from the same recordings, with no
               regard to whether anything was missed in them. How many extra
               proposals appear, and are they faces?

BOTH ARMS NEED THE SAME LABEL AND IT IS NOT AUTOMATABLE. A new box is a win
or a false alarm depending on what is inside it, and nothing in this project
knows which. So this harvests the boxes the sliced pass ADDS -- the ones the
whole-frame pass did not already have -- and asks a person about each, once,
with the arm recorded so the two numbers stay separate.

WHAT THIS DOES NOT MEASURE. The hand veto is not applied, and neither is the
tracker or the mask: this is a detector experiment and mixing in the stages
downstream of it would confound the answer. A proposal this rescues still has
to survive `drop_on_hands` to become a cover, so the recall number here is an
upper bound on the end-to-end gain, not the gain itself.
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import random

# A new box is `the same thing` as a baseline box above this overlap. Two
# passes over one face rarely agree to the pixel and the merge already ran
# NMS, so this only has to separate `the same face` from `somewhere else`.
SAME_IOU = 0.30

SHEET = """<meta charset=utf-8><title>tiled face proposals</title><style>
body{font:13px/1.5 system-ui;margin:0;background:#111;color:#ddd}
#bar{position:sticky;top:0;background:#181818;padding:9px 14px;z-index:9;
  border-bottom:1px solid #333;display:flex;gap:16px;align-items:center;
  flex-wrap:wrap}
button{font:13px system-ui;padding:5px 10px;cursor:pointer}
b{color:#ffd33d}
.c{padding:10px 14px;border-bottom:1px solid #262626;display:flex;gap:12px;
  align-items:flex-start}
.c.cur{background:#1d2430;outline:2px solid #4a8}
.c.face{border-left:5px solid #2a6}
.c.notface{border-left:5px solid #d33}
.c img.crop{width:190px;border-radius:3px}
.c img.ctx{width:560px;border-radius:3px}
.meta{color:#9ab;font-size:12px;min-width:210px}
.num{color:#8ab4c8;font-family:ui-monospace,monospace;font-size:11px}
.arm{padding:1px 7px;border-radius:3px;font-size:11px;margin-left:6px}
.arm.recall{background:#264}
.arm.precision{background:#436}
</style>
<div id=bar>
 <span id=prog></span>
 <span><b>1</b> 是脸 &nbsp; <b>2</b> 不是脸 &nbsp;
   <b>&uarr;&darr;</b> move &nbsp; <b>u</b> undo</span>
 <button onclick="dl()">download CSV</button>
 <span class=num>只显示切块后<b>新增</b>的框；整帧那一遍已经找到的不在这里</span>
</div>
<div id=list></div>
<script>
const D = __PAYLOAD__;
const KEY = "tile:" + D.tag;
let lab = {}, cur = 0, hist = [];
try { lab = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch(e) { lab={}; }
const list = document.getElementById("list");
D.cands.forEach((c, i) => {
  const d = document.createElement("div");
  d.className = "c"; d.id = "c" + i;
  d.innerHTML = '<div class=meta>' + c.rec + ' f' + c.frame +
    '<span class="arm ' + c.arm + '">' + (c.arm === "recall" ?
      "召回臂 · 这帧漏过脸" : "精确臂 · 随机帧") + '</span>' +
    '<br><span class=num>score ' + c.score.toFixed(2) +
    '  w ' + (c.w_frac * 100).toFixed(1) + '% 画面宽</span>' +
    ' <span id=v' + i + '></span></div>' +
    '<img class=crop src="' + c.crop + '">' +
    '<img class=ctx src="' + c.ctx + '">';
  d.onclick = () => { cur = i; draw(); };
  list.appendChild(d);
});
function draw(){
  D.cands.forEach((c,i)=>{
    const v = lab[c.key];
    document.getElementById("c"+i).className =
      "c " + (v === undefined ? "" : (v ? "face" : "notface")) +
      (i===cur ? " cur" : "");
    document.getElementById("v"+i).innerHTML = v === undefined ? "" :
      (v ? '<b style="color:#5c5">是脸</b>' : '<b style="color:#d55">不是</b>');
  });
  const n = Object.keys(lab).length;
  document.getElementById("prog").innerHTML =
    "<b>" + D.tag + "</b> &nbsp; " + n + "/" + D.cands.length + " 已判";
  localStorage.setItem(KEY, JSON.stringify(lab));
  const el = document.getElementById("c"+cur);
  if(el) el.scrollIntoView({block:"nearest"});
}
document.onkeydown = ev => {
  if(ev.key==="1" || ev.key==="2"){
    const c = D.cands[cur]; if(!c) return;
    hist.push([c.key, lab[c.key]]);
    lab[c.key] = ev.key === "1";
    cur = Math.min(cur+1, D.cands.length-1);
    draw();
  }
  else if(ev.key==="ArrowDown"){ cur=Math.min(cur+1,D.cands.length-1); draw(); }
  else if(ev.key==="ArrowUp"){ cur=Math.max(cur-1,0); draw(); }
  else if(ev.key==="u"){ const h=hist.pop(); if(h){ if(h[1]===undefined)
      delete lab[h[0]]; else lab[h[0]]=h[1]; draw(); } }
  else return;
  ev.preventDefault();
};
draw();
function dl(){
  let s = "recording,frame,arm,score,w_frac,is_face\\n";
  for(const c of D.cands) if(c.key in lab)
    s += c.rec + "," + c.frame + "," + c.arm + "," + c.score.toFixed(4) +
         "," + c.w_frac.toFixed(5) + "," + (lab[c.key] ? 1 : 0) + "\\n";
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([s], {type:"text/csv"}));
  a.download = "tile_" + D.tag + ".csv";
  a.click();
}
</script>
"""


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1])
    ub = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (ua + ub - inter)


def load_clips(path):
    """-> {recording tag: (databag, start, n)}"""
    out = {}
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.rsplit(":", 2)
        if len(parts) == 3:
            d, s, n = parts[0], int(parts[1]), int(parts[2])
        else:
            d, s, n = parts[0], int(parts[1]), 1000
        tag = os.path.basename(d.rstrip("/")).replace("databag-26_", "R")
        out[tag] = (d, s, n)
    return out


def wilson(k, n, z=1.96):
    import math
    if not n:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / d, (c + m) / d)


def report(diag_path, labels_path, fps=30.0):
    """The two arms, kept apart.

    THE DENOMINATORS COME FROM DIFFERENT PLACES ON PURPOSE. A rescued face is
    counted against the frames that HAD a miss, because that is the population
    the change is supposed to help. A false alarm is counted against RANDOM
    frames, because that is the population that pays for it, and a rate
    estimated over frames selected for being broken would flatter it."""
    recall_frames = set()
    for r in csv.DictReader(open(diag_path, encoding="utf-8-sig")):
        if r.get("cause") == "detector":
            recall_frames.add((r["recording"], int(r["frame"])))
    rows = list(csv.DictReader(open(labels_path, encoding="utf-8-sig")))
    if not rows:
        raise SystemExit(f"no labelled rows in {labels_path}")

    rescued, arm_frames = set(), {"recall": set(), "precision": set()}
    false_by_frame, true_pre = {}, set()
    for r in rows:
        key = (r["recording"], int(r["frame"]))
        arm = r["arm"]
        arm_frames[arm].add(key)
        if int(r["is_face"]):
            (rescued if arm == "recall" else true_pre).add(key)
        elif arm == "precision":
            false_by_frame[key] = false_by_frame.get(key, 0) + 1

    print("\n=== 召回臂：切块能救回多少漏掉的脸 ===")
    n = len(recall_frames)
    k = len(rescued & recall_frames)
    lo, hi = wilson(k, n)
    print(f"  原本『检测器没提出』的帧   {n}")
    print(f"  切块后提出了真脸的帧       {k}   {k / n if n else 0:.1%} "
          f"[{lo:.3f}, {hi:.3f}]")
    print(f"  （其余 {n - k} 帧切块也没提出来，或提出的不是脸）")

    print("\n=== 精确臂：随机帧上多出来的误检 ===")
    # Frames that produced no new box at all never reach the sheet, so the
    # denominator is the frames SAMPLED, not the frames labelled.
    print(f"  被判过的随机帧             {len(arm_frames['precision'])}")
    print(f"  其中多出至少一个假框的帧    {len(false_by_frame)}")
    print(f"  多出来的假框总数           {sum(false_by_frame.values())}")
    print(f"  多出来的真脸（额外收获）    {len(true_pre)}")

    print("\n  召回臂的分母是『出过问题的帧』，精确臂的分母是『随机帧』——"
          "两个分母\n  不能互换：在挑出来的坏帧上估误检率会低估，"
          "在随机帧上估召回增益会\n  没有信号。")
    print("  这里只测检测器。救回来的候选还要过 drop_on_hands 才会变成马赛克，"
          "\n  所以召回数是端到端增益的上界，不是增益本身。")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--diag", required=True,
                    help="the attributed face_miss csv; rows whose cause is "
                         "`detector` are the recall arm")
    ap.add_argument("--clips", default="/workspace/e2e_main2.txt")
    ap.add_argument("--sample", type=int, default=8,
                    help="random frames per recording for the precision arm. "
                         "Drawn from the audited window with no regard to "
                         "whether anything was missed there.")
    ap.add_argument("--cols", type=int, default=2)
    ap.add_argument("--rows", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    ap.add_argument("--report",
                    help="a labelled tile_*.csv; print the two arms and exit "
                         "instead of harvesting")
    ap.add_argument("--face_model",
                    default="/workspace/models/yolov8n-face-lindevs.onnx")
    ap.add_argument("--face_conf", type=float, default=None)
    ap.add_argument("--ctx_w", type=int, default=760)
    a = ap.parse_args()

    if a.report:
        report(a.diag, a.report)
        return
    if not a.out:
        ap.error("--out is required unless --report is given")

    import cv2
    from src.rig import face_mask
    from src.rig.calibration import RigCalibration
    from src.rig.geometry import VirtualWideCamera
    from src.rig.render_wide import render
    from src.rig.seam_fix import ClipReader, Prefetch

    clips = load_clips(a.clips)
    want = {}
    for r in csv.DictReader(open(a.diag, encoding="utf-8-sig")):
        if r.get("cause") == "detector":
            want.setdefault(r["recording"], {})[int(r["frame"])] = "recall"
    if not want:
        raise SystemExit(f"no `detector` rows in {a.diag}")

    # THE RANDOM FRAMES ARE DRAWN FROM THE SPAN THE MISSES ALREADY FORCE US TO
    # DECODE, not from the whole window. The reader is sequential and a frame
    # step costs about a second, so sampling across the full 1000-frame window
    # would multiply the decode by eight to gain nothing: what the precision
    # arm needs is frames NOT SELECTED FOR BEING BROKEN, and any frame in the
    # span that nobody flagged is that. It does mean the estimate describes
    # these stretches of footage rather than the recordings at large.
    rng = random.Random(a.seed)
    for rec in list(want):
        if rec not in clips:
            continue
        miss = sorted(want[rec])
        pool = [f for f in range(miss[0], miss[-1] + 1) if f not in want[rec]]
        for f in rng.sample(pool, min(a.sample, len(pool))):
            want[rec][f] = "precision"

    det = face_mask.load_detector(a.face_model, a.face_conf)
    if det is None or not hasattr(det, "detect_tiled"):
        raise SystemExit(f"no sliceable face detector at {a.face_model}")

    n_rec = sum(1 for v in want.values() for x in v.values() if x == "recall")
    n_pre = sum(1 for v in want.values() for x in v.values() if x == "precision")
    print(f"  {len(want)} 段录像   召回臂 {n_rec} 帧   精确臂 {n_pre} 帧   "
          f"切块 {a.cols}x{a.rows}")

    cands, seen = [], {"base": 0, "tiled": 0, "frames": 0}
    for rec in sorted(want):
        if rec not in clips:
            print(f"  !! {rec} 不在 clips 里，跳过 {len(want[rec])} 帧")
            continue
        databag, _, _ = clips[rec]
        rig = RigCalibration(os.path.join(databag, "calibration.yaml"))
        vcam = VirtualWideCamera.from_rig(rig)
        vids = {k: os.path.join(databag, f"{k}.mp4")
                for k in ("cam12", "cam34", "cam56")}
        frames = sorted(want[rec])
        lo, hi = frames[0], frames[-1]
        print(f"  {rec}: {len(frames)} 帧, 读 {lo}-{hi}")
        # Face detection carries no state between frames, so unlike the
        # attribution pass this can skip: only the wanted frames are decoded
        # into a panorama, and the reader jumps the rest.
        rd = Prefetch(ClipReader(rig, vids, lo), skip=0)
        mc, pos = {}, lo
        for target in frames:
            src = None
            while pos <= target:
                src = rd.next()
                if not src:
                    break
                pos += 1
            if not src:
                print(f"    !! 读到 {pos} 就断了，剩下的跳过")
                break
            try:
                rgb, _, _, _ = render(rig, vcam, src, 0.6, map_cache=mc)
            except TypeError:
                rgb, _, _, _ = render(rig, vcam, src, 0.6)
            H, W = rgb.shape[:2]
            base = face_mask.detect_faces(det, rgb)
            tiled = face_mask.detect_faces(det, rgb, tiles=(a.cols, a.rows))
            seen["frames"] += 1
            seen["base"] += len(base)
            seen["tiled"] += len(tiled)
            new = [t for t in tiled
                   if all(iou(t[:4], b[:4]) < SAME_IOU for b in base)]
            ctx = cv2.resize(rgb, (a.ctx_w, int(a.ctx_w * H / W)))
            sx = a.ctx_w / float(W)
            for j, (x0, y0, x1, y1, sc) in enumerate(new):
                px = int(max(x1 - x0, y1 - y0) * 0.8)
                cx0, cy0 = max(0, int(x0) - px), max(0, int(y0) - px)
                cx1, cy1 = min(W, int(x1) + px), min(H, int(y1) + px)
                if cx1 <= cx0 + 4 or cy1 <= cy0 + 4:
                    continue
                marked = ctx.copy()
                cv2.rectangle(marked,
                              (int(x0 * sx), int(y0 * sx)),
                              (int(x1 * sx), int(y1 * sx)), (60, 255, 255), 2)

                def b64(img, w=None):
                    if w:
                        img = cv2.resize(img, (w, max(
                            1, int(img.shape[0] * w / img.shape[1]))))
                    ok, buf = cv2.imencode(".jpg", img,
                                           [int(cv2.IMWRITE_JPEG_QUALITY), 88])
                    return ("data:image/jpeg;base64,"
                            + base64.b64encode(buf).decode()) if ok else ""
                cands.append({
                    "key": f"{rec}:{target}:{j}", "rec": rec, "frame": target,
                    "arm": want[rec][target], "score": float(sc),
                    "w_frac": float(x1 - x0) / W,
                    "crop": b64(rgb[cy0:cy1, cx0:cx1], 260),
                    "ctx": b64(marked)})

    tag = f"{a.cols}x{a.rows}"
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(SHEET.replace("__PAYLOAD__",
                              json.dumps({"tag": tag, "cands": cands})))
    print(f"\n  {seen['frames']} 帧：整帧提出 {seen['base']}，"
          f"切块提出 {seen['tiled']}，新增待判 {len(cands)}")
    print(f"  -> {a.out} ({os.path.getsize(a.out) / 1e6:.1f} MB)")
    print("  1 是脸   2 不是脸。判完导 CSV，用 face_tile_report 算两条臂。")


if __name__ == "__main__":
    main()
