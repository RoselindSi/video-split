"""Cut the frame in two with a freehand curve, in both eyes, and watch it hold.

A REGION THAT ONLY HOLDS FOR ONE FRAME IS NOT A REGION. The wearer leans, the
head turns, the bench swings through the view -- whether `the wearer's side`
means the same pixels ten seconds later is the whole question, and a sheet of
stills cannot ask it. So the clip plays with the line on it, and a keyframe
goes in only where it stops being right.

A CUT, NOT AN OUTLINE. What is being annotated is a boundary, not an object:
the curve crosses the whole frame and everything on one side is the wearer's.
An outline drawn loosely around some hands is hard to be wrong about; a cut
either leaves the colleague's hands on the far side or it does not.

BOTH EYES, BECAUSE ONE EYE'S CURVE DOES NOT TRANSFER. cam34.mp4 is cam3 and
cam4 side by side on a 9.3 cm baseline. The wearer's own forearms sit at
0.2-0.3 m, where the disparity is tens to a hundred-odd pixels, while the
bench across the aisle is near zero -- so a boundary that sweeps from the
near foreground to the far field is displaced by a DIFFERENT amount at each
point along it, and no single shift maps it across. Annotating only cam3
leaves the other eye with no zone at all, and the filter runs on both.

THE SECOND EYE STARTS AS A COPY, AND SAYS SO. Drawing the same boundary twice
by hand would double the cost of every keyframe, so finishing a stroke in one
eye plants a shifted copy in the other; the annotator checks it and nudges it
with the bracket keys, and the nudge teaches the tool the disparity it uses
next time. Every keyframe carries how its shape got there -- `drawn` from a
stroke, `held` from watching an earlier one stay right, `copied` from the
other eye -- so a consumer can ask for only the curves a person drew, and
nobody can mistake a shifted duplicate for a second observation.

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

NOTHING STITCHED. cam3 is the middle of the fan at 29.5 degrees of yaw, cam1
at 0 and cam5 at 54.3. The panorama is a render with a depth assumption baked
into it, so a line drawn there measures the render; mapping out to cam12 and
cam56 is a later pass through the calibration, not part of annotating.

THE SIDE IS RECORDED AS A POLYGON, NOT AS A WORD. `left of a to b` depends on
which end got clicked first, so the export carries the closed own-side polygon
as well as the curve. Nothing downstream should have to reconstruct the
convention from the order of two points.

THE PAGE POSTS ITSELF BACK WHEN IT IS SERVED. Handing the tool to four people
and collecting four zip files by hand is how annotations go missing; run
`zone_server` beside the same directory and every page saves a draft to one
folder as it is drawn, with whoever drew it named in the file. Opened straight
off the disk it still works, and falls back to the download button.
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
input{font:13px system-ui;padding:4px 7px;background:#222;color:#ddd;
  border:1px solid #444;border-radius:3px;width:130px}
b{color:#ffd33d}
#stage{position:relative;margin:12px 14px;width:__SW__px;height:__H__px}
video,canvas{position:absolute;left:0;top:0;width:__SW__px;height:__H__px;
  border-radius:4px}
canvas{cursor:crosshair}
/* A MISSING VIDEO USED TO BE A BLACK RECTANGLE. The page loads, the canvas
   draws nothing over nothing, and there is no way to tell that from a clip
   that starts on a dark frame -- so the one failure a person will actually
   hit says so in words. */
#err{display:none;position:absolute;left:0;top:0;width:__SW__px;height:__H__px;
  background:#241a1a;border:1px solid #633;border-radius:4px;color:#f99;
  align-items:center;justify-content:center;text-align:center;
  font:14px/1.8 system-ui;padding:0 40px;box-sizing:border-box}
#err span{color:#c88;font-size:12px}
#tl{margin:0 14px 18px;height:34px;position:relative;background:#1b1b1b;
  border-radius:4px;cursor:pointer;touch-action:none;user-select:none}
#tl.drag{cursor:grabbing}
#play{position:absolute;top:0;bottom:0;background:#2a3a4a;border-radius:4px}
#head{position:absolute;top:0;bottom:0;width:2px;background:#ffd33d}
#grip{position:absolute;top:50%;width:14px;height:14px;margin:-7px 0 0 -7px;
  border-radius:50%;background:#ffd33d;box-shadow:0 0 0 3px rgba(0,0,0,.35);
  pointer-events:none}
/* TWO ROWS OF MARKS, ONE PER EYE. Stacking both eyes' keyframes on one row
   would hide exactly the thing worth seeing: a frame where only one eye has
   been fixed. */
.kf{position:absolute;width:3px}
.kf.L{top:2px;height:13px;background:#3f6}
.kf.R{bottom:2px;height:13px;background:#6cf}
.kf.copied{opacity:.42}
#sub{font-size:12px;color:#8b8}
.key{font-size:11px;color:#999;max-width:960px}
</style>
<div id=bar>
 <button onclick="tog()">播放/暂停 (空格)</button>
 <span id=info></span>
 <span class=key><b>按住拖拽</b>=在某一只眼里画分割曲线（画到该眼画面边缘就收笔）。
 画完<b>另一只眼会自动得到一份平移过来的拷贝</b>（半透明标记=拷贝的），
 你只要核对、用 <b>[</b>/<b>]</b> 左右推（按住 shift 推得快），推的量会被记住。
 <b>x</b> 从另一只眼重新拷一份 <b>f</b> 翻转这只眼哪侧是自己的
 <b>k</b> 两只眼一起打关键帧 <b>d</b> 删掉管着这一刻的关键帧（两只眼一起）
 <b>u</b>/<b>ctrl+z</b> 撤销 <b>shift+C</b> 全清
 <b>,</b>/<b>.</b> 前后一帧 <b>&larr;</b>/<b>&rarr;</b> 前后一秒。
 进度条可以拖；绿色=左眼(cam3)关键帧，蓝色=右眼(cam4)。
 绿色填充一侧=佩戴者自己的手所在的区域。</span>
 <input id=who placeholder="你的名字">
 <button id=subbtn onclick="submit()">提交</button>
 <span id=sub></span>
 <button onclick="dl()">download JSON</button>
</div>
<div id=stage><video id=v src="__VIDEO__" playsinline></video>
<canvas id=c width=__SW__ height=__H__></canvas>
<div id=err>找不到 <b>__VIDEO__</b><br><span></span></div></div>
<div id=tl><div id=play></div><div id=head></div><div id=grip></div></div>
<script>
const M = __META__;
const KEY = "zonestereo:" + M.rec;
const N = 64;                    // points every stored curve is resampled to
const EYE_NAME = {L: M.eyes[0], R: M.eyes[1]};
// frame -> {c: [[x,y] x N], flip: 0|1, src: "drawn"|"copied"}, per eye, in
// that eye's OWN pixels: x runs 0..W inside each half, never across the pair.
let kfs = {L: {}, R: {}};
try {
  const s = JSON.parse(localStorage.getItem(KEY) || "null");
  if (s && s.L && s.R) kfs = s;
} catch(e) { kfs = {L:{}, R:{}}; }
let eye = "L";
// WHAT THE TOOL THINKS THE DISPARITY IS. Not a calibration number -- it starts
// at zero and every nudge updates it, so after the first keyframe the copies
// land roughly right on their own.
let shift = +(localStorage.getItem(KEY + ":shift") || 0);
const v = document.getElementById("v"), c = document.getElementById("c");
const ctx = c.getContext("2d"), tl = document.getElementById("tl");
const W = M.w, H = M.h, P = 2*(W+H);
let live = null, raw = null, liveEye = "L";
const FR = 1 / M.fps;
const OX = {L: 0, R: W};
const other = e => e === "L" ? "R" : "L";
// UNDO EXISTS BECAUSE DELETE WAS THE ONLY WAY OUT AND DELETE DID NOT WORK.
// `c` removed the keyframe AT the current frame, and the current frame is
// almost never exactly a keyframe -- you draw at 0, scrub to 400 to check it,
// press c, and nothing happens because there is no keyframe at 400. Snapshots
// are small (a few dozen curves of 64 points) so the whole map is copied.
let hist = [];
function push(){
  hist.push(JSON.stringify(kfs));
  if (hist.length > 60) hist.shift();
}
function undo(){
  const h = hist.pop();
  if (h === undefined) return;
  kfs = JSON.parse(h);
  save();
}
// The keyframe that DECIDES what is on screen right now: the nearest at or
// before this frame, or the first one after it when the playhead sits before
// every keyframe. This is what `d` removes, and the header names it so the
// key is never a guess.
function governing(e, t){
  const f = frameOf(t), ts = times(e);
  if (!ts.length) return null;
  let lo = null;
  for (const k of ts){ if (k <= f) lo = k; else break; }
  return lo === null ? ts[0] : lo;
}
function times(e){ return Object.keys(kfs[e]).map(Number).sort((a,b)=>a-b); }
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
function curveAt(e, t){
  if (live && liveEye === e) return live;
  const f = frameOf(t), ts = times(e);
  if (!ts.length) return null;
  let lo = null, hi = null;
  for (const k of ts){ if (k <= f) lo = k; else { hi = k; break; } }
  if (lo === null) return kfs[e][hi];
  if (hi === null) return kfs[e][lo];
  const s = (f-lo)/(hi-lo), A = kfs[e][lo], B = kfs[e][hi];
  return {c: A.c.map((p,i) => [p[0]+(B.c[i][0]-p[0])*s,
                               p[1]+(B.c[i][1]-p[1])*s]),
          flip: A.flip, src: A.src};
}
function round1(cv){ return cv.map(p => [Math.round(p[0]*10)/10,
                                         Math.round(p[1]*10)/10]); }
function put(e, f, cv, flip, src){
  kfs[e][String(f)] = {c: round1(cv), flip: flip, src: src};
}
// A COPY IS SHIFTED, CLAMPED, AND LABELLED. Clamping matters: a curve pushed
// off the side of the frame would come back as a degenerate polygon covering
// everything, which is worse than no annotation because it looks like one.
function shifted(cv, dx){
  return cv.map(p => [Math.min(Math.max(p[0] + dx, 0), W), p[1]]);
}
function paintEye(e, t){
  const L = curveAt(e, t);
  if (!L || L.c.length < 2) return;
  const ox = OX[e], poly = ownPoly(L.c, L.flip);
  ctx.save(); ctx.translate(ox, 0);
  ctx.beginPath(); ctx.moveTo(poly[0][0], poly[0][1]);
  for (let i=1;i<poly.length;i++) ctx.lineTo(poly[i][0], poly[i][1]);
  ctx.closePath();
  ctx.fillStyle = "rgba(60,220,120,.16)"; ctx.fill();
  ctx.strokeStyle = (L.src === "copied") ? "#8bd" : "#ffd33d";
  ctx.setLineDash(L.src === "copied" ? [9,6] : []);
  ctx.lineWidth = 3;
  ctx.beginPath(); ctx.moveTo(poly[0][0], poly[0][1]);
  for (let i=1;i<L.c.length;i++) ctx.lineTo(L.c[i][0], L.c[i][1]);
  ctx.stroke(); ctx.setLineDash([]);
  let cx=0, cy=0;
  for (const p of poly){ cx += p[0]; cy += p[1]; }
  ctx.fillStyle = "#3fc"; ctx.font = "13px system-ui";
  ctx.fillText("自己的手这一侧", cx/poly.length-45, cy/poly.length);
  ctx.restore();
}
function paint(){
  ctx.clearRect(0,0,2*W,H);
  for (const e of ["L","R"]) paintEye(e, v.currentTime);
  // The seam, and which half is taking the keys.
  ctx.strokeStyle = "#555"; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(W+.5,0); ctx.lineTo(W+.5,H); ctx.stroke();
  ctx.strokeStyle = "#ffd33d"; ctx.lineWidth = 2;
  ctx.strokeRect(OX[eye]+1, 1, W-2, H-2);
  ctx.fillStyle = "#ffd33d"; ctx.font = "12px system-ui";
  ctx.fillText(EYE_NAME[eye], OX[eye]+8, 18);
  ctx.fillStyle = "#777";
  ctx.fillText(EYE_NAME[other(eye)], OX[other(eye)]+8, 18);
  const g = governing(eye, v.currentTime);
  const on = kfs[eye][keyOf(v.currentTime)];
  document.getElementById("info").innerHTML =
    "<b>" + M.rec + "</b> 帧 " + (M.start + frameOf(v.currentTime)) + "  " +
    v.currentTime.toFixed(2) + "s / " + M.dur.toFixed(1) + "s &nbsp; " +
    "关键帧 " + EYE_NAME.L + " " + times("L").length + " / " +
    EYE_NAME.R + " " + times("R").length +
    " &nbsp; 视差 " + shift.toFixed(0) + "px" +
    (on ? " <b>（当前就是关键帧" + (on.src === "copied" ? "·拷贝的" : "") + "）</b>"
        : (g === null ? "" : " &nbsp; d 将删掉帧 " + (M.start + g))) +
    (hist.length ? " &nbsp; 可撤销 " + hist.length + " 步" : "");
  const pct = (v.currentTime/M.dur*100) + "%";
  document.getElementById("head").style.left = pct;
  document.getElementById("grip").style.left = pct;
  document.getElementById("play").style.width = pct;
  // REBUILT ONLY WHEN THE SET CHANGES. paint() runs on every animation frame,
  // and tearing down and recreating these marks sixty times a second made the
  // bar fight the pointer that was dragging it.
  const sig = ["L","R"].map(e =>
    times(e).map(k => k + (kfs[e][String(k)].src === "copied" ? "c" : "")).join(","))
    .join("|");
  if (sig !== lastKfSig){
    lastKfSig = sig;
    tl.querySelectorAll(".kf").forEach(el => el.remove());
    for (const e of ["L","R"]) for (const k of times(e)){
      const d = document.createElement("div");
      d.className = "kf " + e +
        (kfs[e][String(k)].src === "copied" ? " copied" : "");
      d.style.left = (k / M.fps / M.dur * 100) + "%";
      tl.appendChild(d);
    }
  }
}
let lastKfSig = null;
function save(){
  localStorage.setItem(KEY, JSON.stringify(kfs));
  localStorage.setItem(KEY + ":shift", String(shift));
  queueAutosave();
  paint();
}
// POINTER CAPTURE, AND THE STROKE ENDS AT THE EDGE. Without capture a mouseup
// outside the canvas never arrives, so the stroke stays live and keeps
// following the cursor -- there is no way to stop drawing. And a boundary is
// finished once it reaches the border, so leaving the frame commits it rather
// than dragging the curve around outside.
//
// WHICH EYE IS DECIDED BY WHERE THE STROKE STARTED, and the stroke is confined
// there. A drag that wanders across the seam would otherwise produce a curve
// spanning both eyes, which is not a thing that exists.
c.onpointerdown = e => {
  const r = c.getBoundingClientRect();
  const x = (e.clientX - r.left) * (2*W) / r.width;
  const y = (e.clientY - r.top) * H / r.height;
  eye = liveEye = x < W ? "L" : "R";
  c.setPointerCapture(e.pointerId);
  raw = [[x - OX[liveEye], y]];
  e.preventDefault();
};
c.onpointermove = e => {
  if (!raw) return;
  const r = c.getBoundingClientRect();
  let p = [(e.clientX - r.left) * (2*W) / r.width - OX[liveEye],
           (e.clientY - r.top) * H / r.height];
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
  const prev = kfs[liveEye][keyOf(v.currentTime)] || curveAt(liveEye, v.currentTime);
  live = {c: resample(raw, N), flip: prev ? prev.flip : 0, src: "drawn"};
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
  // A FAST STROKE USED TO DRAW NOTHING AT ALL. refresh() only ran once three
  // raw points had arrived, so a quick flick across the frame -- which emits
  // two -- ended with live still null and the stroke silently discarded. Two
  // points are a straight cut, which is a perfectly good answer.
  if (!live && raw && raw.length >= 2) refresh();
  if (live){
    push();
    const f = frameOf(v.currentTime);
    put(liveEye, f, live.c, live.flip, "drawn");
    // THE OTHER EYE GETS THE SAME MOMENT, NOT THE SAME PIXELS. Copying keeps
    // the two eyes on the same keyframe times, which is what makes them
    // comparable at all; a keyframe already sitting on this exact frame was
    // put there on purpose and is left alone.
    if (!kfs[other(liveEye)][String(f)]){
      const dx = liveEye === "L" ? -shift : shift;
      put(other(liveEye), f, shifted(live.c, dx), live.flip, "copied");
    }
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
// NUDGING IS THE DISPARITY MEASUREMENT. Pushing cam4's copy until it lines up
// is the annotator saying how far apart the eyes see this boundary, so the
// amount is kept and the next copy starts there instead of at zero.
function nudge(dx){
  const L = curveAt(eye, v.currentTime);
  if (!L) return;
  push();
  const f = frameOf(v.currentTime);
  const was = kfs[eye][String(f)];
  put(eye, f, shifted(L.c, dx), L.flip, was ? was.src : "copied");
  shift += (eye === "R" ? dx : -dx);
  save();
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
  if (ev.target && ev.target.tagName === "INPUT") return;   // the name box
  const f = frameOf(v.currentTime);
  if (ev.key === " ") tog();
  else if (ev.key === ",") step(-FR);
  else if (ev.key === ".") step(FR);
  else if (ev.key === "ArrowLeft") step(-1);
  else if (ev.key === "ArrowRight") step(1);
  else if (ev.key === "1"){ eye = "L"; paint(); }
  else if (ev.key === "2"){ eye = "R"; paint(); }
  else if (ev.key === "[") nudge(ev.shiftKey ? -10 : -2);
  else if (ev.key === "]") nudge(ev.shiftKey ? 10 : 2);
  else if (ev.key === "{") nudge(-10);
  else if (ev.key === "}") nudge(10);
  else if (ev.key === "x"){
    const S = curveAt(other(eye), v.currentTime);
    if (!S) return;
    push();
    put(eye, f, shifted(S.c, eye === "R" ? shift : -shift), S.flip, "copied");
    save();
  }
  else if (ev.key === "f"){
    const L = curveAt(eye, v.currentTime);
    if (!L) return;
    push();
    const was = kfs[eye][String(f)];
    put(eye, f, L.c, L.flip ? 0 : 1, was ? was.src : L.src);
    save();
  }
  // `k` IS A THIRD PROVENANCE, NOT THE FIRST ONE REPEATED. Freezing the held
  // curve at a later frame says a person watched it to here and it still fits
  // -- worth more than an interpolation and less than a stroke -- so it is
  // recorded as `held` rather than inheriting `drawn` from the keyframe it
  // came from. A curve that reached this eye from the other one stays
  // `copied`: holding a copy does not make it observed.
  else if (ev.key === "k"){
    push();
    for (const e of ["L","R"]){
      const L = curveAt(e, v.currentTime);
      if (!L) continue;
      const was = kfs[e][String(f)];
      put(e, f, L.c, L.flip,
          was ? was.src : (L.src === "copied" ? "copied" : "held"));
    }
    save();
  }
  else if (ev.key === "u" || (ev.key === "z" && (ev.ctrlKey || ev.metaKey))){
    undo();
  }
  // BOTH EYES, because they are kept on the same keyframe times on purpose and
  // dropping one of a pair leaves the other interpolating across a moment the
  // annotator decided was wrong.
  else if (ev.key === "d" || ev.key === "c"){
    const gs = ["L","R"].map(e => [e, governing(e, v.currentTime)])
                        .filter(g => g[1] !== null);
    if (!gs.length) return;
    push();
    for (const [e, g] of gs) delete kfs[e][String(g)];
    save();
  }
  else if (ev.key === "C" || (ev.key === "Delete" && ev.shiftKey)){
    if (!times("L").length && !times("R").length) return;
    push(); kfs = {L:{}, R:{}}; save();
  }
  else return;
  ev.preventDefault();
};
v.addEventListener("loadedmetadata", paint);
v.addEventListener("error", () => {
  const e = document.getElementById("err");
  e.style.display = "flex";
  e.querySelector("span").innerHTML =
    "html 和 mp4 必须在<b>同一个文件夹</b>里。<br>" +
    "把这个 html 移到 mp4 旁边，或者反过来。<br>" +
    "已标的内容不会丢（存在浏览器里）。";
});
(function loop(){ paint(); requestAnimationFrame(loop); })();
// In each eye's own pixels, not the ones on screen. The closed polygon goes in
// as well as the curve: a curve plus a side is a convention someone has to
// reimplement, and a polygon is a thing you can test a point against.
function build(){
  const s = M.full_w / W;
  const up = p => [Math.round(p[0]*s), Math.round(p[1]*s)];
  const eyes = {};
  for (const e of ["L","R"]){
    eyes[EYE_NAME[e]] = {
      size: [M.full_w, M.full_h],
      keyframes: times(e).map(f => {
        const L = kfs[e][String(f)], poly = ownPoly(L.c, L.flip);
        let cx=0, cy=0;
        for (const p of poly){ cx += p[0]; cy += p[1]; }
        return {frame: M.start + f, t: +(f/M.fps).toFixed(3), src: L.src,
                curve: L.c.map(up), own_polygon: poly.map(up),
                own_side_point: up([cx/poly.length, cy/poly.length])};
      })};
  }
  return {recording: M.rec, source: M.source, clip_start_frame: M.start,
          fps: M.fps, resampled_points: N, eyes: eyes,
          copy_shift_px: Math.round(shift * s),
          annotator: (document.getElementById("who").value || "").trim(),
          interpolation: "linear point-by-point between keyframes; side held",
          note: "curve/polygon are in each eye's own full-resolution pixels. " +
                "src: drawn=a stroke here, held=an earlier stroke in this eye " +
                "confirmed at this frame, copied=shifted from the other eye " +
                "and never drawn"};
}
function dl(){
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([JSON.stringify(build(), null, 1)],
                                        {type:"application/json"}));
  a.download = "wearer_zone_" + M.rec + ".json"; a.click();
}
// SERVED OR NOT, THE PAGE STILL WORKS. Opened off the disk there is nowhere to
// post to, so the button says so instead of failing silently every few seconds.
const SERVED = location.protocol.startsWith("http");
const who = document.getElementById("who");
who.value = localStorage.getItem("zoneannotator") || "";
who.onchange = () => localStorage.setItem("zoneannotator", who.value.trim());
function note(s, bad){
  const el = document.getElementById("sub");
  el.textContent = s; el.style.color = bad ? "#e88" : "#8b8";
}
if (!SERVED){
  document.getElementById("subbtn").disabled = true;
  note("离线打开：用 download JSON 交回");
}
function post(status){
  const b = build();
  if (!b.annotator){ note("先填名字", true); return Promise.resolve(false); }
  return fetch("submit", {method:"POST",
      headers:{"Content-Type":"application/json"},
      body: JSON.stringify({rec: M.rec, annotator: b.annotator,
                            status: status, data: b})})
    .then(r => r.ok ? r.json() : Promise.reject(r.status))
    .then(() => { note(status === "final" ? "已提交 ✓" :
      "草稿已存 " + new Date().toLocaleTimeString()); return true; })
    .catch(e => { note("存不上（" + e + "），请 download JSON", true);
                  return false; });
}
function submit(){ post("final"); }
// AUTOSAVE SO A CLOSED TAB IS NOT A LOST AFTERNOON. Debounced, and silent
// about failures after the first: an annotator with no network should see the
// warning once, not once every edit.
let saveTimer = null;
function queueAutosave(){
  if (!SERVED || !who.value.trim()) return;
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => post("draft"), 4000);
}
</script>
"""


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    # EITHER a clips file from this project, OR a video path. The second form
    # exists so the tool can be handed to someone who has a side-by-side mp4
    # and no idea what a databag is.
    ap.add_argument("--clips")
    ap.add_argument("--rec", action="append")
    ap.add_argument("--video", help="a side-by-side stereo mp4 directly")
    ap.add_argument("--tag", help="name for --video's output files")
    ap.add_argument("--eyes", default="cam3,cam4",
                    help="names of the left and right halves")
    ap.add_argument("--start", type=int, default=None,
                    help="clip start frame; default the clips file's own")
    ap.add_argument("--n", type=int, default=900, help="frames to extract")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--width", type=int, default=640,
                    help="displayed width PER EYE; the export is in full "
                         "per-eye pixels")
    ap.add_argument("--crf", type=int, default=26)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()

    eyes = [s.strip() for s in a.eyes.split(",")]
    if len(eyes) != 2:
        ap.error("--eyes 需要两个名字，比如 cam3,cam4")

    jobs = []
    if a.video:
        tag = a.tag or os.path.splitext(os.path.basename(a.video))[0]
        jobs.append((tag, a.video, a.start or 0))
    if a.clips:
        clips = {}
        for line in open(a.clips):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.rsplit(":", 2)
            clips[os.path.basename(p[0].rstrip("/")).replace("databag-26_",
                                                             "R")] \
                = (p[0], int(p[1]))
        for tag in (a.rec or sorted(clips)):
            if tag not in clips:
                print(f"  !! {tag} 不在 clips 里")
                continue
            databag, start0 = clips[tag]
            jobs.append((tag, os.path.join(databag, "cam34.mp4"),
                         start0 if a.start is None else a.start))
    if not jobs:
        ap.error("需要 --video 或者 --clips/--rec")
    os.makedirs(a.outdir, exist_ok=True)

    for tag, src, start in jobs:
        mp4 = os.path.join(a.outdir, f"zone_{tag}.mp4")
        # BOTH EYES KEPT. The earlier version cropped to the left half because
        # the zone was only ever drawn once; now the pair is the thing being
        # annotated, and each half is scaled to the same displayed width so a
        # pixel means the same distance in both.
        cmd = ["ffmpeg", "-y", "-loglevel", "error",
               "-ss", f"{start / a.fps:.3f}", "-i", src,
               "-frames:v", str(a.n),
               "-vf", f"scale={2 * a.width}:-2",
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
        sw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 2 * a.width)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1)
        nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or a.n)
        cap.release()
        w = sw // 2

        meta = {"rec": tag, "source": src, "start": start, "fps": a.fps,
                "dur": round(nf / a.fps, 3), "w": w, "h": h,
                "full_w": full_w, "full_h": full_h, "eyes": eyes}
        html = (SHEET.replace("__META__", json.dumps(meta, ensure_ascii=False))
                .replace("__VIDEO__", os.path.basename(mp4))
                .replace("__REC__", tag)
                .replace("__SW__", str(2 * w)).replace("__H__", str(h)))
        path = os.path.join(a.outdir, f"zone_{tag}.html")
        open(path, "w", encoding="utf-8").write(html)
        print(f"    {nf} 帧 / {nf / a.fps:.1f}s   每眼 {w}x{h} "
              f"(原始每眼 {full_w}x{full_h})   -> {path} "
              f"(mp4 {os.path.getsize(mp4) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
