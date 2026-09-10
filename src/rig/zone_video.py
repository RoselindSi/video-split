"""Split the frame in two with a line, and see whether it holds through the clip.

A REGION THAT ONLY HOLDS FOR ONE FRAME IS NOT A REGION. The wearer leans, the
head turns, the bench swings through the view -- whether `the wearer's side`
means the same pixels ten seconds later is the whole question, and a sheet of
stills cannot ask it. So the clip plays with the line on it, and a keyframe
goes in only where it stops being right.

A LINE, NOT A SHAPE. One cut across the frame is the cheapest thing that can
carry the claim, and it is also the easiest to be wrong about in a way anyone
can see: an outline drawn loosely around some hands is hard to falsify, while
a line either has the colleague's hands on the far side or it does not. Two
endpoints also make the interpolation between keyframes total -- there is no
vertex correspondence to invent, so the line moves smoothly wherever two
keyframes bracket it.

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
#tl{margin:0 14px;height:26px;position:relative;background:#1b1b1b;
  border-radius:4px;cursor:pointer}
#play{position:absolute;top:0;bottom:0;background:#2a3a4a;border-radius:4px}
#head{position:absolute;top:0;bottom:0;width:2px;background:#ffd33d}
.kf{position:absolute;top:0;bottom:0;width:3px;background:#3f6}
.key{font-size:11px;color:#999;max-width:900px}
</style>
<div id=bar>
 <button onclick="tog()">播放/暂停 (空格)</button>
 <span id=info></span>
 <span class=key><b>拖拽</b>=画一条分割线（自动延长到画面边缘）
 <b>f</b> 翻转哪一侧是自己的 <b>k</b> 打关键帧 <b>c</b>/<b>d</b> 删掉当前关键帧
 <b>,</b>/<b>.</b> 前后一帧 <b>&larr;</b>/<b>&rarr;</b> 前后一秒。
 绿色一侧=佩戴者自己的手所在的区域。</span>
 <button onclick="dl()">download JSON</button>
</div>
<div id=stage><video id=v src="__VIDEO__" playsinline></video>
<canvas id=c width=__W__ height=__H__></canvas></div>
<div id=tl><div id=play></div><div id=head></div></div>
<script>
const M = __META__;
const KEY = "zoneline:" + M.rec;
let kfs = {};                    // frame -> {a:[x,y], b:[x,y], flip:0|1}
try { kfs = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch(e) { kfs={}; }
const v = document.getElementById("v"), c = document.getElementById("c");
const ctx = c.getContext("2d"), tl = document.getElementById("tl");
let live = null;
const FR = 1 / M.fps;
function times(){ return Object.keys(kfs).map(Number).sort((a,b)=>a-b); }
function frameOf(t){ return Math.round(t * M.fps); }
function keyOf(t){ return String(frameOf(t)); }
// Two endpoints interpolate totally: unlike a polygon there is no vertex
// correspondence to invent, so the line is defined at every frame between any
// two keyframes. The side is held rather than blended -- it is a choice, not
// a coordinate.
function lineAt(t){
  if (live) return live;
  const f = frameOf(t), ts = times();
  if (!ts.length) return null;
  let lo = null, hi = null;
  for (const k of ts){ if (k <= f) lo = k; else { hi = k; break; } }
  if (lo === null) return kfs[hi];
  if (hi === null) return kfs[lo];
  const s = (f - lo) / (hi - lo), A = kfs[lo], B = kfs[hi];
  const mix = (p, q) => [p[0]+(q[0]-p[0])*s, p[1]+(q[1]-p[1])*s];
  return {a: mix(A.a, B.a), b: mix(A.b, B.b), flip: A.flip};
}
function side(a, b, p){
  return (b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0]);
}
// Sutherland-Hodgman against the half-plane, so the tint is exact at any
// angle including lines that leave through two adjacent corners.
function clipHalf(poly, a, b, keep){
  const out = [];
  for (let i = 0; i < poly.length; i++){
    const P = poly[i], Q = poly[(i+1) % poly.length];
    const sp = side(a,b,P) * keep, sq = side(a,b,Q) * keep;
    if (sp >= 0) out.push(P);
    if (sp * sq < 0){
      const t = sp / (sp - sq);
      out.push([P[0]+(Q[0]-P[0])*t, P[1]+(Q[1]-P[1])*t]);
    }
  }
  return out;
}
function extend(a, b){
  const dx = b[0]-a[0], dy = b[1]-a[1], ts = [];
  if (Math.abs(dx) > 1e-9){ ts.push((0-a[0])/dx, (c.width-a[0])/dx); }
  if (Math.abs(dy) > 1e-9){ ts.push((0-a[1])/dy, (c.height-a[1])/dy); }
  const pts = ts.map(t => [a[0]+dx*t, a[1]+dy*t])
    .filter(p => p[0] >= -1 && p[0] <= c.width+1
               && p[1] >= -1 && p[1] <= c.height+1);
  return pts.length >= 2 ? [pts[0], pts[pts.length-1]] : [a, b];
}
function paint(){
  ctx.clearRect(0,0,c.width,c.height);
  const L = lineAt(v.currentTime);
  if (L){
    const keep = L.flip ? -1 : 1;
    const rect = [[0,0],[c.width,0],[c.width,c.height],[0,c.height]];
    const own = clipHalf(rect, L.a, L.b, keep);
    if (own.length > 2){
      ctx.beginPath(); ctx.moveTo(own[0][0], own[0][1]);
      for (let i=1;i<own.length;i++) ctx.lineTo(own[i][0], own[i][1]);
      ctx.closePath();
      ctx.fillStyle = "rgba(60,220,120,.16)"; ctx.fill();
    }
    const e = extend(L.a, L.b);
    ctx.strokeStyle = "#ffd33d"; ctx.lineWidth = 3;
    ctx.beginPath(); ctx.moveTo(e[0][0], e[0][1]);
    ctx.lineTo(e[1][0], e[1][1]); ctx.stroke();
    ctx.fillStyle = "#3fc";
    ctx.font = "13px system-ui";
    const cx = own.reduce((s,p)=>s+p[0],0)/Math.max(1,own.length);
    const cy = own.reduce((s,p)=>s+p[1],0)/Math.max(1,own.length);
    ctx.fillText("自己的手这一侧", cx-45, cy);
  }
  const on = kfs[keyOf(v.currentTime)];
  document.getElementById("info").innerHTML =
    "<b>" + M.rec + "</b> 帧 " + (M.start + frameOf(v.currentTime)) + "  " +
    v.currentTime.toFixed(2) + "s / " + M.dur.toFixed(1) + "s &nbsp; 关键帧 " +
    times().length + (on ? " <b>（当前是关键帧）</b>" : "");
  document.getElementById("head").style.left = (v.currentTime/M.dur*100) + "%";
  document.getElementById("play").style.width = (v.currentTime/M.dur*100) + "%";
  tl.querySelectorAll(".kf").forEach(e => e.remove());
  for (const k of times()){
    const d = document.createElement("div");
    d.className = "kf"; d.style.left = (k / M.fps / M.dur * 100) + "%";
    tl.appendChild(d);
  }
}
function save(){ localStorage.setItem(KEY, JSON.stringify(kfs)); paint(); }
let drag = null;
c.onmousedown = e => { const r = c.getBoundingClientRect();
  drag = [e.clientX-r.left, e.clientY-r.top]; };
c.onmousemove = e => {
  if (!drag) return;
  const r = c.getBoundingClientRect(), x = e.clientX-r.left, y = e.clientY-r.top;
  if (Math.abs(x-drag[0]) < 8 && Math.abs(y-drag[1]) < 8) return;
  const prev = kfs[keyOf(v.currentTime)] || lineAt(v.currentTime);
  live = {a: [drag[0], drag[1]], b: [x, y], flip: prev ? prev.flip : 0};
  // DEFAULT THE SIDE TO THE ONE THE ARMS COME FROM. The wearer's own hands
  // enter from below on a head-mounted rig, so the half containing the bottom
  // centre starts as theirs; `f` is there for when it does not.
  if (!prev){
    const bc = [c.width/2, c.height-1];
    if (side(live.a, live.b, bc) < 0) live.flip = 1;
  }
  paint();
};
c.onmouseup = () => {
  if (live){ kfs[keyOf(v.currentTime)] = live; live = null; save(); }
  drag = null;
};
tl.onclick = e => { const r = tl.getBoundingClientRect();
  v.currentTime = Math.max(0, Math.min(M.dur,
    (e.clientX-r.left)/r.width*M.dur)); paint(); };
function tog(){ v.paused ? v.play() : v.pause(); }
function step(dt){ v.pause();
  v.currentTime = Math.max(0, Math.min(M.dur-FR, v.currentTime+dt)); }
document.onkeydown = ev => {
  if (ev.key === " ") tog();
  else if (ev.key === ",") step(-FR);
  else if (ev.key === ".") step(FR);
  else if (ev.key === "ArrowLeft") step(-1);
  else if (ev.key === "ArrowRight") step(1);
  else if (ev.key === "f"){
    const k = keyOf(v.currentTime), L = lineAt(v.currentTime);
    if (L){ kfs[k] = {a: L.a.map(Math.round), b: L.b.map(Math.round),
                      flip: L.flip ? 0 : 1}; save(); }
  }
  else if (ev.key === "k"){
    const L = lineAt(v.currentTime);
    if (L){ kfs[keyOf(v.currentTime)] = {a: L.a.map(Math.round),
                                         b: L.b.map(Math.round),
                                         flip: L.flip}; save(); }
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
  // In cam3's own pixels, not the ones on screen: the clip was scaled to fit
  // a browser and nothing downstream should have to know by how much.
  const s = M.full_w / M.w;
  const out = {recording: M.rec, camera: "cam3", source: M.source,
    clip_start_frame: M.start, fps: M.fps,
    cam3_size: [M.full_w, M.full_h],
    interpolation: "linear on both endpoints between keyframes; side held",
    keyframes: times().map(f => {
      const L = kfs[String(f)];
      const a = [Math.round(L.a[0]*s), Math.round(L.a[1]*s)];
      const b = [Math.round(L.b[0]*s), Math.round(L.b[1]*s)];
      // The normal points into the wearer's half. `left of a to b` depends on
      // which end was clicked first, so the vector goes in the file too.
      const dx = b[0]-a[0], dy = b[1]-a[1];
      const n = Math.hypot(dx, dy) || 1;
      const sgn = L.flip ? -1 : 1;
      return {frame: M.start + f, t: +(f/M.fps).toFixed(3),
              line_cam3: [a, b],
              own_side: L.flip ? "right_of_a_to_b" : "left_of_a_to_b",
              own_side_normal_cam3: [ +( sgn*(-dy)/n ).toFixed(4),
                                      +( sgn*( dx)/n ).toFixed(4) ]};
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
