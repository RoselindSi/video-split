"""Ownership labelled once per track, not once per frame.

WHAT THIS UNBLOCKS AND WHY IT COULD NOT BE DONE WITH WHAT EXISTS. Ownership is
a property of a hand that lasts as long as the hand is in shot, and the
pipeline currently re-decides it every frame. Pooling the frames of a track
should be strictly better -- but there was no way to measure it, because every
labelled package in this project was sampled at a stride of thirteen or
seventeen frames. Two labelled hands from one recording are seconds apart and
belong to different tracks, so nothing in the corpus says "these forty frames
are one hand, and it was a colleague's".

So this runs the real pipeline -- the detector at the deployment floor, the
real tracker at the real 0.60 admission bar -- and asks for one judgement per
track. A track is one decision covering thirty to sixty frames, which makes
this the cheapest ground truth per frame in the project by a wide margin.

IT IS THE PIPELINE'S TRACKS, DELIBERATELY, NOT IDEAL ONES. The question is
whether pooling helps the tracks the system actually has, including the
fragmented ones. Building better tracks first and then measuring pooling on
those would answer a question about a system that does not exist.

THE STORED FRAMES CARRY NO DRAWN MARKS. `own_label._write_sample` paints the
hand box and a wrist-to-exit line onto the context frame it saves, which turns
the geometry features into pixels and quietly feeds them to any model that
reads that frame. The frames written here are untouched; the box lives in the
csv, where a model can be denied it.
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import os

from src.rig.hand_track import box_iou

# Frames of a track between saved crops. Every frame's box goes in the csv;
# crops are what cost disk and decode time, and a verdict pooled over every
# third frame of a sixty-frame track has twenty samples, which is plenty.
CROP_STRIDE = 3

# A track shorter than this is not offered. It is not that short tracks do not
# matter -- they are most of the fragmentation problem -- but a filmstrip of
# three frames cannot be judged, and an unjudgeable row in a sheet becomes a
# `skip` that looks like uncertainty about the hand.
MIN_TRACK = 5

CROP_PX = 192
TILES = 10

# Overlap at which the comparison input is credited with having admitted the
# same hand. Loose on purpose: a box found at two detector resolutions moves
# a little, and calling that a miss would inflate the rescue set.
CMP_IOU = 0.30


SHEET = """<meta charset=utf-8><title>track ownership</title><style>
body{font:13px/1.5 system-ui;margin:0;background:#111;color:#ddd}
#bar{position:sticky;top:0;background:#181818;padding:9px 14px;z-index:9;
  border-bottom:1px solid #333;display:flex;gap:18px;align-items:center;
  flex-wrap:wrap}
button{font:13px system-ui;padding:5px 10px;cursor:pointer}
b{color:#ffd33d}
.tk{padding:12px 14px;border-bottom:1px solid #262626}
.tk.cur{background:#1d2430;outline:2px solid #4a8}
.tk.owner{border-left:5px solid #2a6}
.tk.other{border-left:5px solid #d33}
.tk.nothand{border-left:5px solid #666}
.tk.mixed{border-left:5px solid #c8a}
.tk.skip{border-left:5px solid #444}
.badge{background:#733;color:#fff;padding:1px 6px;border-radius:3px;
  font-size:11px;margin-left:8px}
.badge.part{background:#763}
.strip{display:flex;gap:6px;flex-wrap:wrap;margin-top:6px}
.strip figure{margin:0;text-align:center}
.strip img.ctx{border-radius:3px;display:block}
.strip .wrap{position:relative;display:inline-block}
.strip img.zoom{position:absolute;right:3px;bottom:3px;width:58px;
  border:2px solid #000;border-radius:3px}
.strip figcaption{font-size:10px;color:#999;margin-top:2px}
.meta{color:#9ab;font-size:12px}
</style>
<div id=bar>
 <span id=prog></span>
 <span><b>1</b> 佩戴者 &nbsp; <b>2</b> 别人 &nbsp; <b>3</b> 不是手 &nbsp;
   <b>4</b> 中途换手/混了 &nbsp; <b>5</b> 说不准 &nbsp;
   <b>&uarr;&darr;</b> move &nbsp; <b>u</b> undo</span>
 <button onclick="dl()">download CSV</button>
</div>
<div id=list></div>
<script>
const D = __PAYLOAD__;
const KEY = "trackaudit:" + D.tag;
let lab = {}, cur = 0, hist = [];
try { lab = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch(e) { lab={}; }
const list = document.getElementById("list");
D.tracks.forEach((t, i) => {
  const d = document.createElement("div");
  d.className = "tk"; d.id = "k" + i;
  const badge = (t.cmp === null || t.cmp === undefined) ? '' :
    (t.cmp === 0 ? '<span class=badge>小图完全没看到</span>' :
     (t.cmp < 0.5 ? '<span class="badge part">小图只看到 ' +
        Math.round(t.cmp*100) + '%</span>' : ''));
  d.innerHTML = '<div class=meta><b>track ' + t.tid + '</b>' + badge +
    ' &nbsp; frames ' +
    t.first + '-' + t.last + ' (' + t.n + ') &nbsp; conf ' + t.cmin + '/' +
    t.cmed + '/' + t.cmax + '</div><div class=strip>' +
    t.tiles.map(x => '<figure><span class=wrap><img class=ctx src="' +
      x.img + '">' + (x.zoom ? '<img class=zoom src="' + x.zoom + '">' : '') +
      '</span><figcaption>f' + x.frame + ' &middot; ' + x.conf +
      '</figcaption></figure>').join('') + '</div>';
  d.onclick = () => { cur = i; draw(); };
  list.appendChild(d);
});
function draw(){
  D.tracks.forEach((t,i)=>{
    document.getElementById("k"+i).className =
      "tk " + (lab[t.tid] || "") + (i===cur ? " cur" : "");
  });
  const n = Object.keys(lab).length;
  const o = Object.values(lab).filter(v=>v==="other").length;
  document.getElementById("prog").innerHTML =
    "<b>" + D.tag + "</b> &nbsp; " + n + "/" + D.tracks.length +
    " judged, " + o + " 别人的手";
  localStorage.setItem(KEY, JSON.stringify(lab));
  const e = document.getElementById("k"+cur);
  if(e) e.scrollIntoView({block:"nearest"});
}
function set(v){
  const t = D.tracks[cur]; if(!t) return;
  hist.push([t.tid, lab[t.tid]]);
  lab[t.tid] = v; cur = Math.min(cur+1, D.tracks.length-1);
  draw();
}
document.onkeydown = e => {
  if(e.key==="1") set("owner");
  else if(e.key==="2") set("other");
  else if(e.key==="3") set("nothand");
  else if(e.key==="4") set("mixed");
  else if(e.key==="5") set("skip");
  else if(e.key==="ArrowDown") { cur=Math.min(cur+1,D.tracks.length-1); draw(); }
  else if(e.key==="ArrowUp") { cur=Math.max(cur-1,0); draw(); }
  else if(e.key==="u") { const h=hist.pop(); if(h){ if(h[1]===undefined)
      delete lab[h[0]]; else lab[h[0]]=h[1]; draw(); } }
  else return;
  e.preventDefault();
};
draw();
function dl(){
  let s = "tid,label,recording,first_frame,last_frame,n_frames,conf_max," +
          "cmp_seen_frac\\n";
  for(const t of D.tracks) if(lab[t.tid])
    s += t.tid + "," + lab[t.tid] + "," + D.tag + "," + t.first + "," +
         t.last + "," + t.n + "," + t.cmax + "," +
         (t.cmp === null || t.cmp === undefined ? "" : t.cmp) + "\\n";
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([s], {type:"text/csv"}));
  a.download = "trackaudit_" + D.tag + ".csv";
  a.click();
}
</script>
"""


# WHOSE HAND IT IS IS NOT VISIBLE IN A CROP OF THE HAND. The first version of
# this sheet showed the 192px crops and nothing else, which is the same
# mistake the ownership classifier makes: a colleague's hand and the wearer's
# hand look identical close up. What separates them is where the forearm goes
# -- off the bottom edge with nothing attached, or across the bench to a
# torso -- and that is only in the whole frame. So the tile is the CONTEXT
# frame with the box drawn, and the crop rides along as an inset for detail.
CTX_TILE_W = 300
ZOOM_W = 116


def _b64(img, quality=78):
    import cv2
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY),
                                         quality])
    return ("data:image/jpeg;base64," + base64.b64encode(buf).decode()
            if ok else None)


def _context_tile(path, box, width=CTX_TILE_W):
    """The whole frame with this hand boxed. -> data URI, or None.

    `box` is (cx, cy, w, h) as fractions of the frame, so it survives the
    resize the context frame was stored at."""
    import cv2
    img = cv2.imread(path)
    if img is None:
        return None
    H, W = img.shape[:2]
    cx, cy, bw, bh = box
    x0, y0 = int((cx - bw / 2) * W), int((cy - bh / 2) * H)
    x1, y1 = int((cx + bw / 2) * W), int((cy + bh / 2) * H)
    img = img.copy()
    cv2.rectangle(img, (x0, y0), (x1, y1), (60, 220, 255), 3)
    h = int(round(H * width / W))
    return _b64(cv2.resize(img, (width, h), interpolation=cv2.INTER_AREA))


def rebuild_sheet(pkg, tiles=6, min_track=MIN_TRACK):
    """Regenerate sheet.html from a package already on disk. -> n tracks

    Detection is the expensive part and it is already done; only the page was
    wrong. This re-reads `hands.csv` and the stored frames and writes a new
    sheet, so a layout mistake costs a second rather than another pass over
    the video."""
    import cv2
    rows = list(csv.DictReader(open(os.path.join(pkg, "hands.csv"),
                                    encoding="utf-8-sig")))
    by = {}
    for r in rows:
        by.setdefault(int(r["tid"]), []).append(r)
    meta = {int(r["tid"]): r
            for r in csv.DictReader(open(os.path.join(pkg, "tracks.csv"),
                                         encoding="utf-8-sig"))}
    tag = os.path.basename(pkg).replace("trackpkg_", "")
    items = []
    for tid, rs in sorted(by.items()):
        m = meta.get(tid)
        if m is None:
            continue
        rs.sort(key=lambda r: int(r["frame"]))
        idx = (list(range(len(rs))) if len(rs) <= tiles else
               [int(round(i * (len(rs) - 1) / (tiles - 1)))
                for i in range(tiles)])
        got = []
        for i in idx:
            r = rs[i]
            box = (float(r["box_cx"]), float(r["box_cy"]),
                   float(r["box_w"]), float(r["box_h"]))
            ctx = _context_tile(os.path.join(pkg, "context",
                                             r["stem"] + ".jpg"), box)
            if ctx is None:
                continue
            crop = cv2.imread(os.path.join(pkg, "crops", r["stem"] + ".jpg"))
            got.append({"img": ctx,
                        "zoom": (_b64(cv2.resize(crop, (ZOOM_W, ZOOM_W)))
                                 if crop is not None else None),
                        "frame": int(r["frame"]),
                        "conf": round(float(r["conf"]), 2)})
        if not got:
            continue
        cmp = m.get("cmp_seen_frac", "")
        items.append({"tid": tid, "n": int(m["n_frames"]),
                      "first": int(m["first_frame"]),
                      "last": int(m["last_frame"]),
                      "cmin": float(m["conf_min"]),
                      "cmed": float(m["conf_med"]),
                      "cmax": float(m["conf_max"]),
                      "cmp": (None if cmp in ("", "None") else float(cmp)),
                      "tiles": got})
    # Rescues first: they are the population whose precision decides the
    # question, and a labeller reaching them last labels them tired.
    items.sort(key=lambda it: (1.0 if it["cmp"] is None else it["cmp"],
                               it["tid"]))
    out = os.path.join(pkg, "sheet.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(SHEET.replace("__PAYLOAD__",
                              json.dumps({"tag": tag, "tracks": items})))
    print(f"  {tag}: {len(items)} tracks -> {out} "
          f"({os.path.getsize(out) / 1e6:.1f} MB)")
    return len(items)


def _med(v):
    s = sorted(v)
    return s[len(s) // 2] if s else float("nan")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rebuild_sheet", action="append", default=[],
                    help="regenerate an existing package's sheet with the "
                         "context frames, without re-running the detector")
    ap.add_argument("--databag")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--weights",
                    default="/shared/models/HaWoR/weights/external/detector.pt")
    ap.add_argument("--out",
                    help="package directory: crops/, context/, hands.csv, "
                         "tracks.csv and the sheet")
    ap.add_argument("--imgsz", type=int, default=512,
                    help="the detector input this package's tracks come from")
    ap.add_argument("--compare_imgsz", type=int,
                    help="also detect at this size and record, per frame, "
                         "whether it admitted an overlapping hand. A track "
                         "the smaller input never admitted is the RESCUE "
                         "population, and its precision is the number the "
                         "resolution change actually turns on -- the "
                         "negatives audited so far were all selected at 512 "
                         "and cannot say anything about detections only the "
                         "larger input proposes.")
    ap.add_argument("--new_track_conf", type=float, default=0.60)
    ap.add_argument("--continue_conf", type=float, default=0.25)
    ap.add_argument("--crop_stride", type=int, default=CROP_STRIDE)
    ap.add_argument("--min_track", type=int, default=MIN_TRACK)
    a = ap.parse_args()
    if a.rebuild_sheet:
        for pkg in a.rebuild_sheet:
            rebuild_sheet(pkg)
        raise SystemExit(0)
    if not a.databag or not a.out:
        ap.error("give --databag and --out, or --rebuild_sheet on an "
                 "existing package")

    import cv2
    from ultralytics import YOLO
    from src.rig.calibration import RigCalibration
    from src.rig.geometry import VirtualWideCamera
    from src.rig.render_wide import render
    from src.rig.seam_fix import ClipReader, Prefetch
    from src.rig.hand_detect import detect
    from src.rig.hand_track import Tracker
    from src.rig import own_label

    cal = os.path.join(a.databag, "calibration.yaml")
    vids = {k: os.path.join(a.databag, f"{k}.mp4")
            for k in ("cam12", "cam34", "cam56")}
    rig = RigCalibration(cal)
    vcam = VirtualWideCamera.from_rig(rig)
    model = YOLO(a.weights)
    tag = os.path.basename(a.databag).replace("databag-26_", "R")
    for sub in ("crops", "context"):
        os.makedirs(os.path.join(a.out, sub), exist_ok=True)

    print(f"  {a.n} frames from {a.start}, real pipeline settings "
          f"(new>={a.new_track_conf:.2f}, continue>={a.continue_conf:.2f})")
    tracker = Tracker()
    cmp_tracker = Tracker() if a.compare_imgsz else None
    rd = Prefetch(ClipReader(rig, vids, a.start), skip=max(0, a.stride - 1))
    mc = {}
    tracks, rows = {}, []
    for k in range(a.n):
        src = rd.next()
        if not src:
            break
        try:
            rgb, _, _, _ = render(rig, vcam, src, 0.6, map_cache=mc)
        except TypeError:
            rgb, _, _, _ = render(rig, vcam, src, 0.6)
        dets = detect(model, rgb, imgsz=a.imgsz, min_conf=a.continue_conf)
        ids = tracker.update(dets, rgb.shape,
                             new_track_conf=a.new_track_conf,
                             continue_conf=a.continue_conf)
        # The comparison input runs the SAME admission policy on its own
        # detections, so what differs between them is evidence quality and
        # nothing else.
        cmp_boxes = []
        if cmp_tracker is not None:
            cd = detect(model, rgb, imgsz=a.compare_imgsz,
                        min_conf=a.continue_conf)
            cid = cmp_tracker.update(cd, rgb.shape,
                                     new_track_conf=a.new_track_conf,
                                     continue_conf=a.continue_conf)
            cmp_boxes = [d["box"] for d, t in zip(cd, cid) if t is not None]
        for d, tid in zip(dets, ids):
            if tid is None:
                continue
            t = tracks.setdefault(tid, {"frames": [], "boxes": [],
                                        "confs": []})
            t["frames"].append(k)
            t["boxes"].append([int(v) for v in d["box"]])
            t["confs"].append(float(d.get("conf", 1.0)))
            t.setdefault("cmp", []).append(
                1 if any(box_iou(d["box"], b) >= CMP_IOU for b in cmp_boxes)
                else 0)
            # Crops on a stride, and the frame is written UNMARKED.
            if len(t["frames"]) % a.crop_stride == 1:
                stem = f"{tag}_f{a.start + k * a.stride:06d}_h{tid}"
                x0, y0, x1, y1 = [int(v) for v in d["box"]]
                pad = int(max(x1 - x0, y1 - y0) * 0.6)
                H, W = rgb.shape[:2]
                cx0, cy0 = max(0, x0 - pad), max(0, y0 - pad)
                cx1, cy1 = min(W, x1 + pad), min(H, y1 + pad)
                cv2.imwrite(os.path.join(a.out, "crops", stem + ".jpg"),
                            cv2.resize(rgb[cy0:cy1, cx0:cx1],
                                       (CROP_PX, CROP_PX)),
                            [cv2.IMWRITE_JPEG_QUALITY, 90])
                cv2.imwrite(os.path.join(a.out, "context", stem + ".jpg"),
                            cv2.resize(rgb, (900, int(900 * H / W))),
                            [cv2.IMWRITE_JPEG_QUALITY, 82])
                f = own_label.features(d, rgb.shape)
                rows.append({"stem": stem, "tid": tid,
                             "cmp_admitted": (t["cmp"][-1]
                                              if cmp_tracker else ""),
                             "frame": a.start + k * a.stride,
                             **{c: float(v) for c, v in
                                zip(own_label.FEATURES, f)}})
        if (k + 1) % 25 == 0:
            print(f"    [{k + 1}/{a.n}]", flush=True)
    rd.close()

    # ONE SHEET BUILDER, NOT TWO. This function used to assemble its own
    # tiles from the 192px crops while `rebuild_sheet` assembled them from
    # the context frames. The two drifted, and the run that mattered shipped
    # the version a person cannot label from: whose hand it is lives in where
    # the forearm goes, and a crop of the hand does not contain it. The csv
    # is written here and the page is built by the one function that builds
    # pages.
    items = []
    for tid, t in sorted(tracks.items()):
        if len(t["frames"]) < a.min_track:
            continue
        cmp = t.get("cmp", [])
        items.append({"tid": int(tid), "n": len(t["frames"]),
                      "first": a.start + t["frames"][0] * a.stride,
                      "last": a.start + t["frames"][-1] * a.stride,
                      "cmin": round(min(t["confs"]), 2),
                      "cmed": round(_med(t["confs"]), 2),
                      "cmax": round(max(t["confs"]), 2),
                      "cmp": (None if not cmp
                              else round(sum(cmp) / len(cmp), 3))})

    with open(os.path.join(a.out, "tracks.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["tid", "recording", "first_frame",
                                          "last_frame", "n_frames",
                                          "conf_min", "conf_med", "conf_max",
                                          "cmp_seen_frac", "label"])
        w.writeheader()
        for it in items:
            w.writerow({"tid": it["tid"], "recording": tag,
                        "cmp_seen_frac": it.get("cmp"),
                        "first_frame": it["first"], "last_frame": it["last"],
                        "n_frames": it["n"], "conf_min": it["cmin"],
                        "conf_med": it["cmed"], "conf_max": it["cmax"],
                        "label": ""})
    if rows:
        cols = ["stem", "tid", "cmp_admitted", "frame"] \
            + list(own_label.FEATURES)
        with open(os.path.join(a.out, "hands.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols + ["label"])
            w.writeheader()
            for r in rows:
                w.writerow({**r, "label": ""})
    print(f"\n  {len(tracks)} tracks, {len(items)} at least "
          f"{a.min_track} frames, {len(rows)} crops")
    rebuild_sheet(a.out)
    print("  1 佩戴者  2 别人  3 不是手  4 中途换手/混了  5 说不准")
    print("\n  `mixed` is a real answer and not a cop-out: a track that "
          "starts on one hand\n  and ends on another is an identity failure, "
          "and pooling a verdict over it would\n  be pooling over two hands. "
          "Those rows are the tracker's error rate, measured\n  by the same "
          "pass that measures ownership.")


if __name__ == "__main__":
    main()
