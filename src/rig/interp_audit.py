"""Where the detector actually blinked, was the interpolated box on the hand?

SYNTHETIC MASKING ANSWERED A DIFFERENT QUESTION AND SAID SO. Hiding frames the
detector found gives a straight line an easy target: those frames were easy
enough to detect. Real gaps are the opposite by construction -- motion blur,
occlusion, a hand too small or too far -- so the errors that matter are
missing not at random, and nothing in the synthetic result carries across.
Only a person looking at the frames the detector lost can close it.

TWO QUESTIONS, AND THE SECOND IS THE ONE NO OFFLINE TEST CAN REACH. Drawing
the true box gives the geometry: IoU and centre error against the fill. But a
gap is also the only place a tracker can join two different hands, and if it
does, the fill paints a box along the path between them and blurs whatever is
underneath. That failure has no instance in the first corpus -- mixed_identity
came back 0 of 274 -- so its absence there is a fact about that data, not
about the method. `endpoint_identity` asks it directly, per gap, and a short
switch inside one track would never show up in a whole-track label.

THE FILL IS SHOWN, WHICH BIASES THE DRAWN BOX TOWARDS IT. A person given a
rectangle tends to agree with it, so the IoU that comes out is an upper bound
rather than an unbiased estimate. The alternative -- draw first, reveal after
-- costs a second pass over every frame, and the number wanted here is whether
the cover lands on the hand at all, which an upper bound still answers when it
comes out low. When it comes out high the bias is the thing to remember.

NO THRESHOLD IS BEING FITTED. `endpoint_continuity` is recorded and not acted
on. If the drawn boxes show the fill failing where continuity is large, that
is the evidence a gate would be designed from -- in the next round, on other
data.
"""
from __future__ import annotations

import argparse
import base64
import collections
import csv
import json
import math
import os

from src.rig.track_fill import (INTERP_MAX_GAP, box, interpolate, gaps_of,
                                continuity)

IDENTITY = [("same_hand", "两端是同一只手"), ("id_switch", "两端不是同一只手"),
            ("uncertain", "说不准")]

SHEET = """<meta charset=utf-8><title>interpolation audit</title><style>
body{font:13px/1.5 system-ui;margin:0;background:#111;color:#ddd}
#bar{position:sticky;top:0;background:#181818;padding:9px 14px;z-index:9;
  border-bottom:1px solid #333;display:flex;gap:14px;align-items:center;
  flex-wrap:wrap}
button{font:13px system-ui;padding:5px 10px;cursor:pointer}
b{color:#ffd33d}
.g{padding:10px 14px;border-bottom:1px solid #262626}
.g.cur{background:#1d2430;outline:2px solid #4a8}
.g.done{border-left:5px solid #2a6}
.hd{color:#9ab;font-size:12px;margin-bottom:6px}
.num{color:#8ab4c8;font-family:ui-monospace,monospace;font-size:11px}
.row{display:flex;gap:6px;align-items:flex-start;flex-wrap:wrap}
figure{margin:0}
figcaption{font-size:10px;color:#888;text-align:center}
canvas{border-radius:3px;display:block;cursor:crosshair}
.end img{border-radius:3px;display:block;opacity:.9}
.key{font-size:11px;color:#999}
.chip{display:inline-block;padding:1px 8px;margin-right:6px;border-radius:3px;
  font-size:11px;background:#222;color:#888;border:1px solid #333}
.chip.on{background:#25303a;color:#cde;border-color:#3a5}
</style>
<div id=bar><span id=prog></span>
<span class=key>洋红=插值补出来的框。<b>拖拽</b>在每张图上画出手的真实位置；
<b>x</b> 表示这张裁剪里根本没有手；<b>z</b> 撤销当前图。
两端身份：<b>1</b> 同一只手 <b>2</b> 不是同一只手 <b>3</b> 说不准。
<b>&uarr;&darr;</b> 换空洞</span>
<button onclick="dl()">download CSV</button></div>
<div id=list></div>
<script>
const D = __PAYLOAD__;
const ID = __IDENTITY__;
const KEY = "interpaudit:" + D.tag;
let st = {}, cur = 0;
try { st = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch(e) { st={}; }
const list = document.getElementById("list");
function save(){ localStorage.setItem(KEY, JSON.stringify(st)); }
D.gaps.forEach((g, i) => {
  const d = document.createElement("div");
  d.className = "g"; d.id = "g" + i;
  d.innerHTML = '<div class=hd>' + g.rec + '  track ' + g.tid +
    '  <span class=num>gap ' + g.n + ' 帧  f' + g.f0 + '-' + g.f1 +
    '  端点连续性 ' + g.cont + '</span> <span id=v' + i + '></span></div>' +
    '<div class=row>' +
    '<figure class=end><img src="' + g.pre + '" width="150">' +
    '<figcaption>gap 前（真实检测）</figcaption></figure>' +
    g.frames.map((f, j) =>
      '<figure><canvas id="c' + i + '_' + j + '" width="' + f.w +
      '" height="' + f.h + '"></canvas><figcaption>f' + f.f +
      '</figcaption></figure>').join('') +
    '<figure class=end><img src="' + g.post + '" width="150">' +
    '<figcaption>gap 后（真实检测）</figcaption></figure></div>';
  d.onclick = () => { cur = i; draw(); };
  list.appendChild(d);
});
const imgs = {};
D.gaps.forEach((g, i) => g.frames.forEach((f, j) => {
  const im = new Image();
  im.onload = () => { imgs[i + "_" + j] = im; paint(i, j); };
  im.src = f.img;
  const cv = document.getElementById("c" + i + "_" + j);
  let drag = null;
  cv.onmousedown = e => {
    const r = cv.getBoundingClientRect();
    drag = [e.clientX - r.left, e.clientY - r.top];
  };
  cv.onmousemove = e => {
    if (!drag) return;
    const r = cv.getBoundingClientRect();
    st[f.key] = {b: [Math.min(drag[0], e.clientX - r.left),
                     Math.min(drag[1], e.clientY - r.top),
                     Math.max(drag[0], e.clientX - r.left),
                     Math.max(drag[1], e.clientY - r.top)]};
    paint(i, j);
  };
  cv.onmouseup = () => { drag = null; save(); draw(); };
}));
function paint(i, j){
  const f = D.gaps[i].frames[j];
  const cv = document.getElementById("c" + i + "_" + j);
  const ctx = cv.getContext("2d");
  const im = imgs[i + "_" + j];
  if (im) ctx.drawImage(im, 0, 0);
  ctx.lineWidth = 2;
  ctx.strokeStyle = "#e0e";
  ctx.strokeRect(f.pb[0], f.pb[1], f.pb[2]-f.pb[0], f.pb[3]-f.pb[1]);
  const s = st[f.key];
  if (s && s.b){
    ctx.strokeStyle = "#3f6";
    ctx.strokeRect(s.b[0], s.b[1], s.b[2]-s.b[0], s.b[3]-s.b[1]);
  } else if (s && s.none){
    ctx.strokeStyle = "#f44"; ctx.lineWidth = 4;
    ctx.beginPath(); ctx.moveTo(0,0); ctx.lineTo(cv.width, cv.height);
    ctx.moveTo(cv.width,0); ctx.lineTo(0, cv.height); ctx.stroke();
  }
}
function done(i){
  const g = D.gaps[i];
  return st["id:" + g.key] && g.frames.every(f => st[f.key]);
}
function draw(){
  D.gaps.forEach((g,i)=>{
    document.getElementById("g"+i).className =
      "g" + (done(i) ? " done" : "") + (i===cur ? " cur" : "");
    const v = st["id:" + g.key];
    document.getElementById("v"+i).innerHTML = ID.map(o =>
      '<span class="chip' + (v===o[0] ? " on" : "") + '">' + o[1] +
      '</span>').join("");
  });
  const n = D.gaps.filter((_,i)=>done(i)).length;
  document.getElementById("prog").innerHTML =
    "<b>" + D.tag + "</b> &nbsp; " + n + "/" + D.gaps.length + " 个空洞已判完";
  save();
  const el = document.getElementById("g"+cur);
  if(el) el.scrollIntoView({block:"nearest"});
}
document.onkeydown = ev => {
  const g = D.gaps[cur]; if(!g) return;
  if(ev.key>="1" && ev.key<="3"){ st["id:"+g.key] = ID[+ev.key-1][0]; }
  else if(ev.key==="x"){ g.frames.forEach(f => { if(!st[f.key])
      st[f.key] = {none:1}; }); g.frames.forEach((f,j)=>paint(cur,j)); }
  else if(ev.key==="z"){ g.frames.forEach((f,j)=>{ delete st[f.key];
      paint(cur,j); }); }
  else if(ev.key==="ArrowDown"){cur=Math.min(cur+1,D.gaps.length-1);}
  else if(ev.key==="ArrowUp"){cur=Math.max(cur-1,0);}
  else return;
  draw(); ev.preventDefault();
};
draw();
function dl(){
  let s = "key,rec,tid,frame,gap_len,endpoint_continuity,endpoint_identity," +
          "pred_x0,pred_y0,pred_x1,pred_y1,true_x0,true_y0,true_x1,true_y1,"+
          "no_hand\\n";
  for(const g of D.gaps){
    const idv = st["id:" + g.key] || "";
    for(const f of g.frames){
      const v = st[f.key];
      if(!v) continue;
      const b = v.b ? f.o.map((o,k)=> Math.round(o + v.b[k] / f.s))
                    : ["","","",""];
      s += f.key + "," + g.rec + "," + g.tid + "," + f.f + "," + g.n + "," +
           g.cont + "," + idv + "," + f.tb.join(",") + "," + b.join(",") +
           "," + (v.none ? 1 : 0) + "\\n";
    }
  }
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([s],{type:"text/csv"}));
  a.download = "interp_audit_" + D.tag + ".csv"; a.click();
}
</script>
"""


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--clips", required=True)
    ap.add_argument("--ctx", type=float, default=4.0)
    ap.add_argument("--crop_w", type=int, default=210)
    ap.add_argument("--page", type=int, default=40)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import cv2
    from src.rig.calibration import RigCalibration
    from src.rig.geometry import VirtualWideCamera
    from src.rig.render_wide import render
    from src.rig.seam_fix import ClipReader, Prefetch

    by = collections.defaultdict(list)
    for r in csv.DictReader(open(a.rows, encoding="utf-8-sig")):
        by[(r["rec"], r["tid"])].append(r)
    for v in by.values():
        v.sort(key=lambda r: int(r["frame"]))
    # THE PIPELINE'S OWN VERDICT PICKS THE GAPS, not a person's. This audit
    # runs beside the ownership labelling rather than after it, and what gets
    # filled in production is decided by the majority of the deployed labels.
    todo = {}
    for k, v in by.items():
        maj_owner = sum(1 for r in v
                        if int(r["final_owner_post_cap"])) * 2 > len(v)
        if maj_owner:
            continue
        g = gaps_of(v)
        if g:
            todo[k] = (v, g)
    n_gap = sum(len(g) for _, g in todo.values())
    n_fr = sum(sum(x[2] for x in g) for _, g in todo.values())
    print(f"  {len(todo)} 条整轨判 other 的轨迹里有 {n_gap} 个内部空洞 / "
          f"{n_fr} 帧要画")

    clips = {}
    for line in open(a.clips):
        line = line.strip()
        if not line:
            continue
        p = line.rsplit(":", 2)
        clips[os.path.basename(p[0].rstrip("/")).replace("databag-26_", "R")] \
            = (p[0], int(p[1]))

    per_rec = collections.defaultdict(list)
    for k in todo:
        per_rec[k[0]].append(k)

    def b64(img, w, q=85):
        h = int(round(img.shape[0] * w / max(1, img.shape[1])))
        ok, buf = cv2.imencode(".jpg", cv2.resize(img, (w, max(1, h))),
                               [int(cv2.IMWRITE_JPEG_QUALITY), q])
        return ("data:image/jpeg;base64," +
                base64.b64encode(buf).decode()) if ok else ""

    out = []
    for rec in sorted(per_rec):
        if rec not in clips:
            continue
        databag, _ = clips[rec]
        rig = RigCalibration(os.path.join(databag, "calibration.yaml"))
        vcam = VirtualWideCamera.from_rig(rig)
        vids = {kk: os.path.join(databag, f"{kk}.mp4")
                for kk in ("cam12", "cam34", "cam56")}
        want = set()
        for k in per_rec[rec]:
            v, gs = todo[k]
            for i, j, g in gs:
                want.add(int(v[i]["frame"]))
                want.add(int(v[j]["frame"]))
                want.update(range(int(v[i]["frame"]) + 1, int(v[j]["frame"])))
        lo, hi = min(want), max(want)
        rd = Prefetch(ClipReader(rig, vids, lo), skip=0)
        mc, imgs = {}, {}
        print(f"  {rec}: 读 {lo}-{hi}", flush=True)
        for t in range(hi - lo + 1):
            src = rd.next()
            if not src:
                break
            f = lo + t
            if f not in want:
                continue
            try:
                rgb, _, _, _ = render(rig, vcam, src, 0.6, map_cache=mc)
            except TypeError:
                rgb, _, _, _ = render(rig, vcam, src, 0.6)
            imgs[f] = rgb
        rd.close()

        for k in per_rec[rec]:
            v, gs = todo[k]
            for i, j, g in gs:
                f0, f1 = int(v[i]["frame"]), int(v[j]["frame"])
                b0, b1 = box(v[i]), box(v[j])
                cont = round(continuity(v, i, j), 3)
                frames = []
                for t in range(1, g + 1):
                    img = imgs.get(f0 + t)
                    if img is None:
                        continue
                    cx, cy, bw, bh = interpolate(b0, b1, t / (g + 1.0))
                    H, W = img.shape[:2]
                    half = max(40, int(max(bw, bh) * a.ctx / 2))
                    x0, y0 = max(0, int(cx - half)), max(0, int(cy - half))
                    x1, y1 = min(W, int(cx + half)), min(H, int(cy + half))
                    crop = img[y0:y1, x0:x1]
                    if crop.size == 0:
                        continue
                    s = a.crop_w / float(crop.shape[1])
                    cw = a.crop_w
                    ch = max(1, int(round(crop.shape[0] * s)))
                    frames.append({
                        "key": f"{rec}:{k[1]}:{f0 + t}", "f": f0 + t,
                        "w": cw, "h": ch, "s": round(s, 5),
                        "o": [x0, y0, x0, y0],
                        # the fill, in crop pixels, for the canvas to draw
                        "pb": [round((cx - bw / 2 - x0) * s, 1),
                               round((cy - bh / 2 - y0) * s, 1),
                               round((cx + bw / 2 - x0) * s, 1),
                               round((cy + bh / 2 - y0) * s, 1)],
                        # and in panorama pixels, for the csv
                        "tb": [int(cx - bw / 2), int(cy - bh / 2),
                               int(cx + bw / 2), int(cy + bh / 2)],
                        "img": b64(crop, cw)})
                if not frames:
                    continue

                def end(idx, b):
                    img = imgs.get(int(v[idx]["frame"]))
                    if img is None:
                        return ""
                    H, W = img.shape[:2]
                    half = max(40, int(max(b[2], b[3]) * a.ctx / 2))
                    c = img[max(0, int(b[1] - half)):min(H, int(b[1] + half)),
                            max(0, int(b[0] - half)):min(W, int(b[0] + half))]
                    return b64(c, 150) if c.size else ""

                out.append({"key": f"{rec}:{k[1]}:{f0}", "rec": rec,
                            "tid": k[1], "n": g, "f0": f0, "f1": f1,
                            "cont": cont, "frames": frames,
                            "pre": end(i, b0), "post": end(j, b1)})

    stem, ext = os.path.splitext(a.out)
    n_pg = max(1, (len(out) + a.page - 1) // a.page)
    for i in range(n_pg):
        part = out[i * a.page:(i + 1) * a.page]
        path = a.out if n_pg == 1 else f"{stem}_{i + 1}{ext}"
        tag = "all" if n_pg == 1 else f"p{i + 1}"
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(SHEET.replace("__PAYLOAD__",
                                   json.dumps({"tag": tag, "gaps": part}))
                     .replace("__IDENTITY__",
                              json.dumps([[x, y] for x, y in IDENTITY],
                                         ensure_ascii=False)))
        print(f"  {len(part):>3} 个空洞 -> {path} "
              f"({os.path.getsize(path) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
