"""What the hand detector actually saw, before any threshold discarded it.

THE FAILURE THIS EXISTS TO EXPLAIN. A colleague's hand can be on screen for
four seconds, be found by the detector on every one of those frames, and
render with nothing covered. It is not a classifier error and not a tracking
error: the detection never scored 0.60, `Tracker` refuses to start a track
below that, and `demo_video` drops any detection without an id. Nothing
downstream is ever asked about it. Measured over three clips, the longest such
run was 52 frames with a peak score of 0.55 -- five hundredths under the bar,
for four and a third seconds.

TWO FAILURES WEAR THE SAME FACE AND HAVE DIFFERENT FIXES. A hand can be
missing because the detector scored it too low, or because the detector
returned nothing for it at all. The first is a policy about thresholds and
temporal evidence; the second is the detector's own recall, and no threshold
recovers it. The census below separates them, which the render trace cannot:
that trace only ever sees what survived a floor of 0.25.

WHY RUNS AND NOT FRAMES. A single 0.55 detection is as likely to be a wrench
as a hand -- at this score the detector fires on plenty that is not a hand.
What separates them is persistence: junk does not stay coherently in one place
for a second while moving like a limb. So detections are chained frame to
frame by overlap, and the unit of both the census and the audit is the run.
The distribution measured so far is bimodal in exactly the way that argues
for this: a median run is 1-2 frames, and the tail runs 25 to 52.

THE SHEET DOES NOT ASSUME THE ANSWER. It shows the long runs in the band
under the threshold and asks a person whether they are hands. If they are
mostly hands, the bar is too high and temporal evidence should promote them.
If they are mostly not, lowering the bar would fill the render with covered
wrenches, and the fix is elsewhere. Nothing here decides that; it is a
measurement instrument and it changes no pipeline behaviour.
"""
from __future__ import annotations

import argparse
import base64
import json
import os

import numpy as np

from src.rig.hand_track import box_iou

# The census bands. The two that matter are the last two: 0.50-0.60 is the
# band a real hand sits in while the system pretends it is not there, and
# `>=0.60` is what the pipeline currently acts on at all.
BANDS = ((0.00, 0.25), (0.25, 0.50), (0.50, 0.60), (0.60, 1.01))
BAND_NAMES = ("<0.25", "0.25-0.50", "0.50-0.60", ">=0.60")

# Chaining threshold. Deliberately loose: a hand moving fast between two
# frames of a 30 fps source overlaps its own previous box by less than a
# tracker's gate would like, and breaking a run in the middle would report
# one four-second failure as four one-second ones.
CHAIN_IOU = 0.25

# A run shorter than this is not offered for audit however high it scored.
MIN_AUDIT_LEN = 6


def _pct(v, q):
    if not v:
        return float("nan")
    s = sorted(v)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


def band_of(conf):
    for i, (lo, hi) in enumerate(BANDS):
        if lo <= conf < hi:
            return i
    return len(BANDS) - 1


class Runs:
    """Detections chained across frames by overlap. -> [run]

    A run carries every frame's box and score, so the audit sheet can show
    what the detector was looking at and the census can ask how close the run
    came to the bar without collapsing it to one number first."""

    def __init__(self, chain_iou=CHAIN_IOU):
        self.chain_iou = float(chain_iou)
        self.open, self.done = [], []

    def update(self, k, dets):
        live = [r for r in self.open if r["last_k"] == k - 1]
        used, fresh = set(), []
        for d in dets:
            box, conf = list(d["box"]), float(d.get("conf", 1.0))
            best, best_v = None, self.chain_iou
            for r in live:
                if id(r) in used:
                    continue
                v = box_iou(r["boxes"][-1], box)
                if v >= best_v:
                    best, best_v = r, v
            if best is None:
                fresh.append({"first_k": k, "last_k": k, "boxes": [box],
                              "confs": [conf], "frames": [k]})
            else:
                used.add(id(best))
                best["last_k"] = k
                best["boxes"].append(box)
                best["confs"].append(conf)
                best["frames"].append(k)
        for r in list(self.open):
            if r["last_k"] < k:
                self.open.remove(r)
                self.done.append(r)
        self.open.extend(fresh)

    def all(self):
        out = self.done + self.open
        for r in out:
            r["n"] = len(r["frames"])
            r["peak"] = max(r["confs"])
            r["median_conf"] = _pct(r["confs"], 0.5)
            r["min_conf"] = min(r["confs"])
            c = [((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) for b in r["boxes"]]
            step = [float(np.hypot(c[i + 1][0] - c[i][0],
                                   c[i + 1][1] - c[i][1]))
                    for i in range(len(c) - 1)]
            r["median_step_px"] = _pct(step, 0.5) if step else 0.0
        return out


def census(runs, n_frames, blind, new_track_conf=0.60):
    """Print where the detector's evidence sat. -> None

    THE FRAME COUNT AND THE RUN COUNT ANSWER DIFFERENT QUESTIONS. How many
    frames had no candidate at all is about the detector's recall. How many
    RUNS peaked in each band is about the threshold: a run that peaked at 0.55
    is a thing the detector saw the whole time and the pipeline never met."""
    print(f"\n  DETECTOR CENSUS over {n_frames} frames")
    print(f"    frames with NO candidate at any score: {blind}"
          f"  ({blind / max(n_frames, 1):.1%})")
    print("    Those cannot be recovered by any threshold. Everything below "
          "can.")
    by = {}
    for r in runs:
        by.setdefault(band_of(r["peak"]), []).append(r)
    print(f"\n  {'peak band':>12} {'runs':>6} {'>=6f':>6} {'>=12f':>6} "
          f"{'>=24f':>6} {'longest':>8} {'med step px':>12}")
    for i, name in enumerate(BAND_NAMES):
        v = by.get(i, [])
        if not v:
            print(f"  {name:>12} {0:>6}")
            continue
        n = [r["n"] for r in v]
        print(f"  {name:>12} {len(v):>6} {sum(1 for x in n if x >= 6):>6} "
              f"{sum(1 for x in n if x >= 12):>6} "
              f"{sum(1 for x in n if x >= 24):>6} {max(n):>8} "
              f"{_pct([r['median_step_px'] for r in v], 0.5):>12.1f}")
    band = [r for r in runs if 0.50 <= r["peak"] < new_track_conf]
    long_band = [r for r in band if r["n"] >= MIN_AUDIT_LEN]
    print(f"\n    {len(band)} runs peaked in [0.50, {new_track_conf:.2f}) -- "
          f"seen the whole time, never admitted.\n    {len(long_band)} of "
          f"them lasted {MIN_AUDIT_LEN} frames or more. Those are what the "
          f"sheet asks about.")


SHEET_SCALE = 0.5


def _tile(img, box, pad=0.5, width=150, scale=SHEET_SCALE):
    """-> base64 <img> src of the box with context round it, or None."""
    import cv2
    x0, y0, x1, y1 = [int(v * scale) for v in box]
    p = int(max(x1 - x0, y1 - y0) * pad)
    H, W = img.shape[:2]
    cx0, cy0 = max(0, x0 - p), max(0, y0 - p)
    cx1, cy1 = min(W, x1 + p), min(H, y1 + p)
    if cx1 <= cx0 or cy1 <= cy0:
        return None
    crop = img[cy0:cy1, cx0:cx1].copy()
    cv2.rectangle(crop, (x0 - cx0, y0 - cy0), (x1 - cx0, y1 - cy0),
                  (60, 220, 255), 2)
    if crop.shape[1] > width:
        h = int(round(crop.shape[0] * width / crop.shape[1]))
        crop = cv2.resize(crop, (width, h), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 78])
    return ("data:image/jpeg;base64," + base64.b64encode(buf).decode()
            if ok else None)


SHEET = """<meta charset=utf-8><title>low-confidence runs</title><style>
body{font:13px/1.5 system-ui;margin:0;background:#111;color:#ddd}
#bar{position:sticky;top:0;background:#181818;padding:9px 14px;z-index:9;
  border-bottom:1px solid #333;display:flex;gap:18px;align-items:center;
  flex-wrap:wrap}
button{font:13px system-ui;padding:5px 10px;cursor:pointer}
b{color:#ffd33d}
.run{padding:12px 14px;border-bottom:1px solid #262626}
.run.cur{background:#1d2430;outline:2px solid #4a8}
.run.hand_owner{border-left:5px solid #2a6}
.run.hand_other{border-left:5px solid #d33}
.run.nothand{border-left:5px solid #666}
.run.skip{border-left:5px solid #444}
.strip{display:flex;gap:6px;flex-wrap:wrap;margin-top:6px}
.strip figure{margin:0;text-align:center}
.strip img{border-radius:3px;display:block}
.strip figcaption{font-size:10px;color:#999;margin-top:2px}
.meta{color:#9ab;font-size:12px}
</style>
<div id=bar>
 <span id=prog></span>
 <span><b>1</b> 真手·佩戴者 &nbsp; <b>2</b> 真手·别人 &nbsp;
   <b>3</b> 不是手 &nbsp; <b>4</b> 说不准 &nbsp;
   <b>&uarr; &darr;</b> move &nbsp; <b>u</b> undo</span>
 <button onclick="dl()">download CSV</button>
</div>
<div id=list></div>
<script>
const D = __PAYLOAD__;
const KEY = "detaudit:" + D.tag;
let lab = {}, cur = 0, hist = [];
try { lab = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch(e) { lab={}; }
const list = document.getElementById("list");
D.runs.forEach((r, i) => {
  const d = document.createElement("div");
  d.className = "run"; d.id = "r" + i;
  d.innerHTML = '<div class=meta><b>' + r.id + '</b> &nbsp; frames ' +
    r.first + '-' + r.last + ' (' + r.n + ') &nbsp; conf ' + r.min +
    ' / ' + r.med + ' / <b>' + r.peak + '</b> &nbsp; med step ' +
    r.step + 'px</div><div class=strip>' +
    r.tiles.map(t => '<figure><img src="' + t.img + '">' +
      '<figcaption>f' + t.frame + ' &middot; ' + t.conf +
      '</figcaption></figure>').join('') + '</div>';
  d.onclick = () => { cur = i; draw(); };
  list.appendChild(d);
});
function draw(){
  D.runs.forEach((r,i)=>{
    document.getElementById("r"+i).className =
      "run " + (lab[r.id] || "") + (i===cur ? " cur" : "");
  });
  const n = Object.keys(lab).length;
  const h = Object.values(lab).filter(v=>v.startsWith("hand")).length;
  document.getElementById("prog").innerHTML =
    "<b>" + D.tag + "</b> &nbsp; " + n + "/" + D.runs.length +
    " judged, " + h + "真手";
  localStorage.setItem(KEY, JSON.stringify(lab));
  const e = document.getElementById("r"+cur);
  if(e) e.scrollIntoView({block:"nearest"});
}
function set(v){
  const r = D.runs[cur]; if(!r) return;
  hist.push([r.id, lab[r.id]]);
  lab[r.id] = v; cur = Math.min(cur+1, D.runs.length-1);
  draw();
}
document.onkeydown = e => {
  if(e.key==="1") set("hand_owner");
  else if(e.key==="2") set("hand_other");
  else if(e.key==="3") set("nothand");
  else if(e.key==="4") set("skip");
  else if(e.key==="ArrowDown") { cur=Math.min(cur+1,D.runs.length-1); draw(); }
  else if(e.key==="ArrowUp") { cur=Math.max(cur-1,0); draw(); }
  else if(e.key==="u") { const h=hist.pop(); if(h){ if(h[1]===undefined)
      delete lab[h[0]]; else lab[h[0]]=h[1]; draw(); } }
  else return;
  e.preventDefault();
};
draw();
function dl(){
  let s = "run_id,label,n_frames,peak_conf\\n";
  for(const r of D.runs) if(lab[r.id])
    s += r.id + "," + lab[r.id] + "," + r.n + "," + r.peak + "\\n";
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([s], {type:"text/csv"}));
  a.download = "detaudit_" + D.tag + ".csv";
  a.click();
}
</script>
"""


def sheet(runs, keep, out, tag, max_tiles=8):
    """Write the audit page. -> number of runs on it."""
    items = []
    for r in sorted(keep, key=lambda x: -x["n"]):
        idx = (list(range(r["n"])) if r["n"] <= max_tiles else
               [int(round(i * (r["n"] - 1) / (max_tiles - 1)))
                for i in range(max_tiles)])
        tiles = []
        for i in idx:
            img = r["_frames"][i]
            t = _tile(img, r["boxes"][i]) if img is not None else None
            if t:
                tiles.append({"img": t, "frame": r["frames"][i],
                              "conf": round(r["confs"][i], 2)})
        if not tiles:
            continue
        # THE ID CARRIES WHERE THE RUN STARTED, NOT ONLY WHEN. Two hands
        # entering on the same frame produced two runs with the same id, and
        # the sheet keys its labels on the id: judging one silently labelled
        # the other, and the CSV wrote both. Two of thirty-six rows in the
        # first batch were never actually looked at.
        x0, y0 = int(r["boxes"][0][0]), int(r["boxes"][0][1])
        items.append({"id": f"{tag}_r{r['first_k']:05d}_{x0:04d}x{y0:04d}",
                      "first": r["first_k"], "last": r["last_k"],
                      "n": r["n"], "min": round(r["min_conf"], 2),
                      "med": round(r["median_conf"], 2),
                      "peak": round(r["peak"], 2),
                      "step": round(r["median_step_px"], 1),
                      "tiles": tiles})
    html = SHEET.replace("__PAYLOAD__",
                         json.dumps({"tag": tag, "runs": items}))
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n  {len(items)} runs -> {out} "
          f"({os.path.getsize(out) / 1e6:.1f} MB)")
    print("  1 真手·佩戴者   2 真手·别人   3 不是手   4 说不准")
    print("  The question is only whether these are hands. If most are, the "
          "0.60 bar is\n  keeping real hands out of the pipeline entirely. "
          "If most are not, lowering it\n  would cover the bench instead, "
          "and the missing hands are missing for another\n  reason.")
    return len(items)


def write_clip(run, frames, a, tag):
    """An mp4 of one run, its box drawn, padded either side. -> path or None

    WHY A CLIP AND NOT MORE STILLS. Two of the first thirty-six runs came
    back `uncertain`, and both were long ones -- sixteen and twenty-five
    frames. A filmstrip shows what a thing looks like; whether it moves like
    a hand needs the motion itself, and those are exactly the runs where the
    appearance did not settle it."""
    import cv2
    lo = max(0, run["first_k"] - a.clip_pad)
    hi = run["last_k"] + a.clip_pad
    have = [k for k in range(lo, hi + 1) if k in frames]
    if not have:
        print(f"    !! no frames kept for run at {run['first_k']}")
        return None
    h, w = frames[have[0]].shape[:2]
    out = os.path.join(a.clip_dir,
                       f"clip_{tag}_r{run['first_k']:05d}.mp4")
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"),
                         float(a.clip_fps), (w, h))
    at = dict(zip(run["frames"], run["boxes"]))
    for k in have:
        img = frames[k].copy()
        box = at.get(k)
        if box is not None:
            x0, y0, x1, y1 = [int(v) for v in box]
            cv2.rectangle(img, (x0, y0), (x1, y1), (60, 220, 255), 3)
            i = run["frames"].index(k)
            cv2.putText(img, f"{run['confs'][i]:.2f}", (x0, max(14, y0 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 220, 255), 2)
        cv2.putText(img, f"src frame {a.start + k}", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 2)
        vw.write(img)
    vw.release()
    print(f"    run at {run['first_k']} ({run['n']} frames, peak "
          f"{run['peak']:.2f}) -> {out}")
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--databag", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--weights",
                    default="/shared/models/HaWoR/weights/external/detector.pt")
    ap.add_argument("--floor", type=float, default=0.05,
                    help="the detector's own floor. Below the pipeline's "
                         "0.25 on purpose: what never reached 0.25 is the "
                         "difference between a threshold problem and a "
                         "recall problem.")
    ap.add_argument("--new_track_conf", type=float, default=0.60)
    ap.add_argument("--min_len", type=int, default=MIN_AUDIT_LEN)
    ap.add_argument("--sheet")
    ap.add_argument("--clip_at", type=int, action="append", default=[],
                    help="write an mp4 around the run starting at this "
                         "frame index, boxes drawn. For the runs a filmstrip "
                         "cannot settle: a still cannot show whether a shape "
                         "moves like a limb.")
    ap.add_argument("--clip_pad", type=int, default=12)
    ap.add_argument("--clip_fps", type=float, default=6.0)
    ap.add_argument("--clip_dir", default="/workspace")
    ap.add_argument("--panorama", choices=("depth", "baseline"),
                    default="baseline")
    a = ap.parse_args()

    from ultralytics import YOLO
    from src.rig.hand_detect import detect
    from src.rig.calibration import RigCalibration
    from src.rig.geometry import VirtualWideCamera
    from src.rig.render_wide import render
    from src.rig.seam_fix import ClipReader, Prefetch
    from src.rig.panorama import DepthAwarePanorama

    cal = os.path.join(a.databag, "calibration.yaml")
    vids = {k: os.path.join(a.databag, f"{k}.mp4")
            for k in ("cam12", "cam34", "cam56")}
    rig = RigCalibration(cal)
    vcam = VirtualWideCamera.from_rig(rig)
    model = YOLO(a.weights)
    # Baseline by default, matching what ships. The census is about the
    # detector, and changing the renderer under it would change what the
    # detector is looking at.
    pano = (DepthAwarePanorama(rig, vcam) if a.panorama == "depth" else None)
    mc = {}
    tag = os.path.basename(a.databag).replace("databag-26_", "R")

    print(f"  {a.n} frames from {a.start}, stride {a.stride}, "
          f"detector floor {a.floor:.2f}, bar {a.new_track_conf:.2f}")
    rd = Prefetch(ClipReader(rig, vids, a.start), skip=max(0, a.stride - 1))
    runs, blind, seen = Runs(), 0, 0
    frames = {}
    # Full-resolution frames are kept only for the windows a clip was asked
    # for. Keeping every frame at full size costs most of a gigabyte.
    want_clip = set()
    for c in a.clip_at:
        want_clip.update(range(c - a.clip_pad, c + 200 + a.clip_pad))
    clip_frames = {}
    for k in range(a.n):
        src = rd.next()
        if not src:
            break
        if pano is not None:
            rgb, _, _, _ = pano.render(src)
        else:
            try:
                rgb, _, _, _ = render(rig, vcam, src, 0.6, map_cache=mc)
            except TypeError:
                rgb, _, _, _ = render(rig, vcam, src, 0.6)
        seen += 1
        dets = detect(model, rgb, min_conf=a.floor)
        if not dets:
            blind += 1
        runs.update(k, dets)
        # HALF SIZE, AND ONLY WHEN A SHEET IS WANTED. Two hundred panorama
        # frames at full resolution is most of a gigabyte held for the sake
        # of 150px thumbnails; the tiles are downscaled anyway.
        if a.sheet:
            import cv2
            frames[k] = cv2.resize(rgb, None, fx=SHEET_SCALE, fy=SHEET_SCALE,
                                   interpolation=cv2.INTER_AREA)
        if k in want_clip:
            clip_frames[k] = rgb.copy()
        if (k + 1) % 25 == 0:
            print(f"    [{k + 1}/{a.n}]", flush=True)
    rd.close()

    all_runs = runs.all()
    census(all_runs, seen, blind, a.new_track_conf)
    for c in a.clip_at:
        for r in all_runs:
            if r["first_k"] == c:
                write_clip(r, clip_frames, a, tag)
    if a.sheet:
        for r in all_runs:
            r["_frames"] = [frames.get(f) for f in r["frames"]]
        keep = [r for r in all_runs
                if r["n"] >= a.min_len
                and 0.50 <= r["peak"] < a.new_track_conf]
        sheet(all_runs, keep, a.sheet, tag)


if __name__ == "__main__":
    main()
