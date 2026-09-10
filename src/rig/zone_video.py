"""Cut the frame in two with a freehand curve, and watch whether it holds.

A REGION THAT ONLY HOLDS FOR ONE FRAME IS NOT A REGION. The wearer leans, the
head turns, the bench swings through the view -- whether `the wearer's side`
means the same pixels ten seconds later is the whole question, and a sheet of
stills cannot ask it. So the clip plays with the line on it, and a keyframe
goes in only where it stops being right.

A CUT, NOT AN OUTLINE. What is being annotated is a boundary, not an object:
the curve crosses the whole frame and everything on one side is the wearer's.
An outline drawn loosely around some hands is hard to be wrong about; a cut
either leaves the colleague's hands on the far side or it does not.

THE CURVE IS RESAMPLED TO A FIXED NUMBER OF POINTS, and that is what makes
keyframes work. Two freehand strokes have no correspondence between their
points, so interpolating raw ones would blend a fast stroke into a slow one
and produce a shape nobody drew. Resampled by arc length to the same count,
point k of one curve means the same fraction along as point k of the other,
and the boundary moves smoothly between any two keyframes.

CLOSED AGAINST THE FRAME BORDER, NOT AGAINST ITSELF. The ends are extended
along their own direction until they hit an edge, and the filled side is
completed by walking the perimeter from one exit to the other. Which way round
that walk goes is the only thing `f` changes.

cam34, LEFT EYE, NOTHING STITCHED. The file is 3840x1520 side by side and the
left half is cam3, the middle of the fan at 29.5 degrees of yaw, with cam1 at
0 and cam5 at 54.3. The panorama is a render with a depth assumption baked
into it, so a line drawn there measures the render; mapping out to cam12 and
cam56 is a later pass through the calibration, not part of annotating.

THE SIDE IS RECORDED AS A VECTOR, NOT AS A WORD. `left of a to b` depends on
which end got clicked first, so the export carries a unit normal pointing into
the wearer's half as well. Nothing downstream should have to reconstruct the
convention from the order of two points.

THE CLIP SHIPS BESIDE THE PAGE. Thirty seconds of 960-wide H.264 is five
megabytes; the same seconds inlined as a data URI is not, and a self-contained
file that takes a minute to open is the worse tool.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess

SHEET = """<meta charset=utf-8><title>wearer zone — __REC__</title><style>
body{font:13px/1.5 system-ui;margin:0;background:#111;color:#ddd}
#bar{position:sticky;top:0;background:#181818;padding:8px 14px;z-index:9;
  border-bottom:1px solid #333;display:flex;gap:12px;align-items:center;
  flex-wrap:wrap}
button{font:13px system-ui;padding:5px 10px;cursor:pointer}
b{color:#ffd33d}
#stage{position:relative;margin:12px 14px;width:__W__px;height:__H__px}
video,canvas{position:absolute;left:0;top:0;width:__W__px;height:__H__px;
  border-radius:4px}
canvas{cursor:crosshair}
#tl{margin:0 14px 18px;height:34px;position:relative;background:#1b1b1b;
  border-radius:4px;cursor:pointer;touch-action:none;user-select:none}
#tl.drag{cursor:grabbing}
#play{position:absolute;top:0;bottom:0;background:#2a3a4a;border-radius:4px}
#head{position:absolute;top:0;bottom:0;width:2px;background:#ffd33d}
#grip{position:absolute;top:50%;width:14px;height:14px;margin:-7px 0 0 -7px;
  border-radius:50%;background:#ffd33d;box-shadow:0 0 0 3px rgba(0,0,0,.35);
  pointer-events:none}
.kf{position:absolute;top:0;bottom:0;width:3px;background:#3f6}
.key{font-size:11px;color:#999;max-width:920px}
</style>
<div id=bar>
 <button onclick="tog()">播放/暂停 (空格)</button>
 <span id=info></span>
 <span class=key><b>按住拖拽</b>=随手画一条分割曲线（画到画面边缘就自动收笔；
 松开鼠标也收笔，两端自动延到边缘）
 <b>f</b> 翻转哪一侧是自己的 <b>k</b> 打关键帧 <b>c</b>/<b>d</b> 删掉当前关键帧
 <b>,</b>/<b>.</b> 前后一帧 <b>&larr;</b>/<b>&rarr;</b> 前后一秒。
 底下的进度条可以<b>拖着拉</b>，绿色竖线是关键帧。松手后如果原来在播就继续播。
 绿色一侧=佩戴者自己的手所在的区域。</span>
 <button onclick="dl()">download JSON</button>
</div>
<div id=stage><video id=v src="__VIDEO__" playsinline></video>
<canvas id=c width=__W__ height=__H__></canvas></div>
<div id=tl><div id=play></div><div id=head></div><div id=grip></div></div>
<script>
const M = __META__;
const KEY = "zonecurve:" + M.rec;
const N = 64;                    // points every stored curve is resampled to
let kfs = {};                    // frame -> {c: [[x,y] x N], flip: 0|1}
try { kfs = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch(e) { kfs={}; }
const v = document.getElementById("v"), c = document.getElementById("c");
const ctx = c.getContext("2d"), tl = document.getElementById("tl");
const W = c.width, H = c.height, P = 2*(W+H);
let live = null, raw = null;
const FR = 1 / M.fps;
function times(){ return Object.keys(kfs).map(Number).sort((a,b)=>a-b); }
function frameOf(t){ return Math.round(t * M.fps); }
function keyOf(t){ return String(frameOf(t)); }
// ARC-LENGTH RESAMPLE. Two freehand strokes have no correspondence between
// their raw points -- a slow stroke leaves twenty where a fast one leaves
// four -- so interpolating them directly blends nothing meaningful. At a fixed
// count by arc length, point k of one curve is the same fraction along as
// point k of the other, and keyframes become total.
function resample(pts, n){
  const d = [0];
  for (let i=1;i<pts.length;i++)
    d.push(d[i-1] + Math.hypot(pts[i][0]-pts[i-1][0], pts[i][1]-pts[i-1][1]));
  const L = d[d.length-1] || 1, out = [];
  let j = 0;
  for (let k=0;k<n;k++){
    const target = L * k / (n-1);
    while (j < d.length-2 && d[j+1] < target) j++;
    const seg = (d[j+1]-d[j]) || 1, a = Math.min(1, (target-d[j])/seg);
    out.push([pts[j][0] + (pts[j+1][0]-pts[j][0])*a,
              pts[j][1] + (pts[j+1][1]-pts[j][1])*a]);
  }
  return out;
}
// Extend a point along a direction until it meets the frame border.
function snap(p, d){
  const ts = [];
  if (Math.abs(d[0]) > 1e-9){ ts.push((0-p[0])/d[0], (W-p[0])/d[0]); }
  if (Math.abs(d[1]) > 1e-9){ ts.push((0-p[1])/d[1], (H-p[1])/d[1]); }
  const good = ts.filter(t => t > 0).map(t => [p[0]+d[0]*t, p[1]+d[1]*t])
    .filter(q => q[0] >= -1 && q[0] <= W+1 && q[1] >= -1 && q[1] <= H+1);
  if (!good.length) return [Math.min(Math.max(p[0],0),W),
                            Math.min(Math.max(p[1],0),H)];
  return good.reduce((a,b) => Math.hypot(a[0]-p[0],a[1]-p[1]) <
                              Math.hypot(b[0]-p[0],b[1]-p[1]) ? a : b);
}
function perimT(p){
  const e = 1e-6, cl = (x,m) => Math.min(Math.max(x,0),m);
  if (p[1] <= e) return cl(p[0],W);
  if (p[0] >= W-e) return W + cl(p[1],H);
  if (p[1] >= H-e) return W + H + (W - cl(p[0],W));
  return 2*W + H + (H - cl(p[1],H));
}
const CORNERS = [[[W,0],W],[[W,H],W+H],[[0,H],2*W+H],[[0,0],2*W+2*H]];
function cornersBetween(tA, tB, dir){
  const span = dir > 0 ? ((tB-tA+P)%P) : ((tA-tB+P)%P), out = [];
  for (const [pt,tc] of CORNERS){
    const d = dir > 0 ? ((tc-tA+P)%P) : ((tA-tc+P)%P);
    if (d > 1e-9 && d < span-1e-9) out.push([pt,d]);
  }
  out.sort((a,b)=>a[1]-b[1]);
  return out.map(o=>o[0]);
}
// CLOSED AGAINST THE BORDER, NOT AGAINST ITSELF: from where the curve leaves
// the frame, walk the perimeter back to where it entered. Which way round is
// the only thing the side flag changes.
function ownPoly(curve, flip){
  const a = snap(curve[0], [curve[0][0]-curve[1][0], curve[0][1]-curve[1][1]]);
  const n = curve.length;
  const b = snap(curve[n-1], [curve[n-1][0]-curve[n-2][0],
                              curve[n-1][1]-curve[n-2][1]]);
  const mid = [a].concat(curve.slice(1, n-1), [b]);
  return mid.concat(cornersBetween(perimT(b), perimT(a), flip ? -1 : 1));
}
function curveAt(t){
  if (live) return live;
  const f = frameOf(t), ts = times();
  if (!ts.length) return null;
  let lo = null, hi = null;
  for (const k of ts){ if (k <= f) lo = k; else { hi = k; break; } }
  if (lo === null) return kfs[hi];
  if (hi === null) return kfs[lo];
  const s = (f-lo)/(hi-lo), A = kfs[lo], B = kfs[hi];
  return {c: A.c.map((p,i) => [p[0]+(B.c[i][0]-p[0])*s,
                               p[1]+(B.c[i][1]-p[1])*s]), flip: A.flip};
}
function paint(){
  ctx.clearRect(0,0,W,H);
  const L = curveAt(v.currentTime);
  if (L && L.c.length > 1){
    const poly = ownPoly(L.c, L.flip);
    ctx.beginPath(); ctx.moveTo(poly[0][0], poly[0][1]);
    for (let i=1;i<poly.length;i++) ctx.lineTo(poly[i][0], poly[i][1]);
    ctx.closePath();
    ctx.fillStyle = "rgba(60,220,120,.16)"; ctx.fill();
    ctx.strokeStyle = "#ffd33d"; ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(poly[0][0], poly[0][1]);
    for (let i=1;i<L.c.length;i++) ctx.lineTo(L.c[i][0], L.c[i][1]);
    ctx.stroke();
    let cx=0, cy=0;
    for (const p of poly){ cx += p[0]; cy += p[1]; }
    ctx.fillStyle = "#3fc"; ctx.font = "13px system-ui";
    ctx.fillText("\u81ea\u5df1\u7684\u624b\u8fd9\u4e00\u4fa7",
                 cx/poly.length-45, cy/poly.length);
  }
  const on = kfs[keyOf(v.currentTime)];
  document.getElementById("info").innerHTML =
    "<b>" + M.rec + "</b> \u5e27 " + (M.start + frameOf(v.currentTime)) +
    "  " + v.currentTime.toFixed(2) + "s / " + M.dur.toFixed(1) +
    "s &nbsp; \u5173\u952e\u5e27 " + times().length +
    (on ? " <b>\uff08\u5f53\u524d\u662f\u5173\u952e\u5e27\uff09</b>" : "");
  const pct = (v.currentTime/M.dur*100) + "%";
  document.getElementById("head").style.left = pct;
  document.getElementById("grip").style.left = pct;
  document.getElementById("play").style.width = pct;
  // REBUILT ONLY WHEN THE SET CHANGES. paint() runs on every animation frame,
  // and tearing down and recreating these marks sixty times a second made the
  // bar fight the pointer that was dragging it.
  const sig = times().join(",");
  if (sig !== lastKfSig){
    lastKfSig = sig;
    tl.querySelectorAll(".kf").forEach(e => e.remove());
    for (const k of times()){
      const d = document.createElement("div");
      d.className = "kf"; d.style.left = (k / M.fps / M.dur * 100) + "%";
      tl.appendChild(d);
    }
  }
}
let lastKfSig = null;
function save(){ localStorage.setItem(KEY, JSON.stringify(kfs)); paint(); }
// POINTER CAPTURE, AND THE STROKE ENDS AT THE EDGE. Without capture a mouseup
// outside the canvas never arrives, so the stroke stays live and keeps
// following the cursor -- there is no way to stop drawing. And a boundary is
// finished once it reaches the border, so leaving the frame commits it rather
// than dragging the curve around outside.
c.onpointerdown = e => {
  const r = c.getBoundingClientRect();
  c.setPointerCapture(e.pointerId);
  raw = [[e.clientX-r.left, e.clientY-r.top]];
  e.preventDefault();
};
c.onpointermove = e => {
  if (!raw) return;
  const r = c.getBoundingClientRect();
  let p = [e.clientX-r.left, e.clientY-r.top];
  const out = p[0] < 0 || p[0] > W || p[1] < 0 || p[1] > H;
  p = [Math.min(Math.max(p[0], 0), W), Math.min(Math.max(p[1], 0), H)];
  const q = raw[raw.length-1];
  if (!out && Math.hypot(p[0]-q[0], p[1]-q[1]) < 3) return;
  raw.push(p);
  if (raw.length >= 3) refresh();
  if (out) finish();                    // reached the border: that is the end
};
c.onpointerup = c.onpointercancel = () => finish();
function refresh(){
  const prev = kfs[keyOf(v.currentTime)] || curveAt(v.currentTime);
  live = {c: resample(raw, N), flip: prev ? prev.flip : 0};
  // DEFAULT THE SIDE TO THE ONE THE ARMS COME FROM. On a head-mounted rig the
  // wearer's own hands enter from below, so the half holding the bottom centre
  // starts as theirs; `f` is there for when it does not.
  if (!prev){
    const poly = ownPoly(live.c, 0);
    if (!inside(poly, [W/2, H-2])) live.flip = 1;
  }
  paint();
}
function finish(){
  if (live){
    kfs[keyOf(v.currentTime)] =
      {c: live.c.map(p => [Math.round(p[0]*10)/10, Math.round(p[1]*10)/10]),
       flip: live.flip};
    live = null; save();
  }
  raw = null;
}
function inside(poly, p){
  let win = false;
  for (let i=0, j=poly.length-1; i<poly.length; j=i++){
    if (((poly[i][1] > p[1]) !== (poly[j][1] > p[1])) &&
        (p[0] < (poly[j][0]-poly[i][0]) * (p[1]-poly[i][1]) /
                ((poly[j][1]-poly[i][1]) || 1e-9) + poly[i][0])) win = !win;
  }
  return win;
}
// DRAG, NOT JUST CLICK. A click that jumps is fine for finding a moment you
// already know the time of; scrubbing is how you find the frame where the
// boundary stops being right, which is the only reason to keyframe at all.
// Pointer capture keeps the drag alive when the cursor leaves the bar.
let scrub = false, wasPlaying = false;
function seekTo(e){
  const r = tl.getBoundingClientRect();
  const x = Math.max(0, Math.min(r.width, e.clientX - r.left));
  v.currentTime = Math.max(0, Math.min(M.dur - FR, x / r.width * M.dur));
  paint();
}
tl.onpointerdown = e => {
  scrub = true; wasPlaying = !v.paused; v.pause();
  tl.classList.add("drag");
  tl.setPointerCapture(e.pointerId);
  seekTo(e); e.preventDefault();
};
tl.onpointermove = e => { if (scrub) seekTo(e); };
tl.onpointerup = tl.onpointercancel = () => {
  if (!scrub) return;
  scrub = false; tl.classList.remove("drag");
  if (wasPlaying) v.play();
};
function tog(){ v.paused ? v.play() : v.pause(); }
function step(dt){ v.pause();
  v.currentTime = Math.max(0, Math.min(M.dur-FR, v.currentTime+dt)); }
document.onkeydown = ev => {
  if (ev.key === " ") tog();
  else if (ev.key === ",") step(-FR);
  else if (ev.key === ".") step(FR);
  else if (ev.key === "ArrowLeft") step(-1);
  else if (ev.key === "ArrowRight") step(1);
  else if (ev.key === "f" || ev.key === "k"){
    const L = curveAt(v.currentTime);
    if (L) kfs[keyOf(v.currentTime)] =
      {c: L.c.map(p => [Math.round(p[0]*10)/10, Math.round(p[1]*10)/10]),
       flip: ev.key === "f" ? (L.flip ? 0 : 1) : L.flip};
    save();
  }
  else if (ev.key === "c" || ev.key === "d"){
    delete kfs[keyOf(v.currentTime)]; save();
  }
  else return;
  ev.preventDefault();
};
v.addEventListener("loadedmetadata", paint);
(function loop(){ paint(); requestAnimationFrame(loop); })();
function dl(){
  // In cam3's own pixels, not the ones on screen. The closed polygon goes in
  // as well as the curve: a curve plus a side is a convention someone has to
  // reimplement, and a polygon is a thing you can test a point against.
  const s = M.full_w / M.w;
  const up = p => [Math.round(p[0]*s), Math.round(p[1]*s)];
  const out = {recording: M.rec, camera: "cam3", source: M.source,
    clip_start_frame: M.start, fps: M.fps, cam3_size: [M.full_w, M.full_h],
    resampled_points: N,
    interpolation: "linear point-by-point between keyframes; side held",
    keyframes: times().map(f => {
      const L = kfs[String(f)], poly = ownPoly(L.c, L.flip);
      let cx=0, cy=0;
      for (const p of poly){ cx += p[0]; cy += p[1]; }
      return {frame: M.start + f, t: +(f/M.fps).toFixed(3),
              curve_cam3: L.c.map(up),
              own_polygon_cam3: poly.map(up),
              own_side_point_cam3: up([cx/poly.length, cy/poly.length])};
    })};
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([JSON.stringify(out, null, 1)],
                                        {type:"application/json"}));
  a.download = "wearer_zone_" + M.rec + ".json"; a.click();
}
</script>
"""


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", required=True)
    ap.add_argument("--rec", action="append", required=True)
    ap.add_argument("--start", type=int, default=None,
                    help="clip start frame; default the clips file's own")
    ap.add_argument("--n", type=int, default=900, help="frames to extract")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--width", type=int, default=960,
                    help="displayed width; the export is in cam3 pixels")
    ap.add_argument("--crf", type=int, default=26)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()

    clips = {}
    for line in open(a.clips):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = line.rsplit(":", 2)
        clips[os.path.basename(p[0].rstrip("/")).replace("databag-26_", "R")] \
            = (p[0], int(p[1]))
    os.makedirs(a.outdir, exist_ok=True)

    for tag in a.rec:
        if tag not in clips:
            print(f"  !! {tag} 不在 clips 里")
            continue
        databag, start0 = clips[tag]
        start = start0 if a.start is None else a.start
        src = os.path.join(databag, "cam34.mp4")
        mp4 = os.path.join(a.outdir, f"zone_{tag}.mp4")
        # LEFT HALF ONLY. cam34.mp4 is the two eyes side by side and the left
        # one is cam3; annotating both would be the same line drawn twice.
        cmd = ["ffmpeg", "-y", "-loglevel", "error",
               "-ss", f"{start / a.fps:.3f}", "-i", src,
               "-frames:v", str(a.n),
               "-vf", f"crop=iw/2:ih:0:0,scale={a.width}:-2",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", str(a.crf),
               "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart", mp4]
        print(f"  {tag}: 抽 {a.n} 帧（起点 {start}）-> {mp4}", flush=True)
        subprocess.run(cmd, check=True)

        import cv2
        cap = cv2.VideoCapture(src)
        full_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 3840) // 2
        full_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1520)
        cap.release()
        cap = cv2.VideoCapture(mp4)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or a.width)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1)
        nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or a.n)
        cap.release()

        meta = {"rec": tag, "source": src, "start": start, "fps": a.fps,
                "dur": round(nf / a.fps, 3), "w": w, "h": h,
                "full_w": full_w, "full_h": full_h}
        html = (SHEET.replace("__META__", json.dumps(meta, ensure_ascii=False))
                .replace("__VIDEO__", os.path.basename(mp4))
                .replace("__REC__", tag)
                .replace("__W__", str(w)).replace("__H__", str(h)))
        path = os.path.join(a.outdir, f"zone_{tag}.html")
        open(path, "w", encoding="utf-8").write(html)
        print(f"    {nf} 帧 / {nf / a.fps:.1f}s   {w}x{h}   -> {path} "
              f"(mp4 {os.path.getsize(mp4) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
