"""Draw the wearer's own region on the video, not on a still.

A REGION THAT ONLY HOLDS FOR ONE FRAME IS NOT A REGION. The wearer leans, the
head turns, the bench moves through the view -- whether `around the wearer`
means the same pixels ten seconds later is the question, and a sheet of stills
cannot ask it. So this plays the clip, the outline stays on screen while it
plays, and a keyframe is added only where it stops being right.

cam34, LEFT EYE, NOTHING STITCHED. The file is 3840x1520 side by side; the
left half is cam3, the middle module of the fan. The panorama is a render with
a depth assumption baked into it, and a region drawn on a render measures the
render. Mapping to cam12 and cam56 is a later step through the calibration,
not something to do while annotating.

HOLD, THEN INTERPOLATE ONLY WHERE IT CAN. Between two keyframes with the same
number of vertices the outline moves vertex by vertex; where the counts differ
-- an ellipse replaced by a hand-drawn polygon -- it holds the earlier one
until the later one starts, because a made-up correspondence between vertices
would draw a shape nobody meant.

THE VIDEO SHIPS BESIDE THE PAGE. A minute of 960-wide H.264 is a few
megabytes; the same minute as a data URI inside the html is not, and a
self-contained file that takes a minute to open is a worse tool than two files
in one folder.
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
 <span class=key><b>拖拽</b>=椭圆（半圆就拖出画面下沿） <b>点击</b>=多边形顶点
 <b>c</b> 清空 <b>u</b> 撤销顶点 <b>k</b> 在当前时刻打关键帧 <b>d</b> 删除当前关键帧
 <b>,</b>/<b>.</b> 前后一帧 <b>&larr;</b>/<b>&rarr;</b> 前后一秒</span>
 <button onclick="dl()">download JSON</button>
</div>
<div id=stage><video id=v src="__VIDEO__" playsinline></video>
<canvas id=c width=__W__ height=__H__></canvas></div>
<div id=tl><div id=play></div><div id=head></div></div>
<script>
const M = __META__;
const KEY = "zonevid:" + M.rec;
let kfs = {};                       // time (ms, rounded to frame) -> polygon
try { kfs = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch(e) { kfs={}; }
const v = document.getElementById("v"), c = document.getElementById("c");
const ctx = c.getContext("2d"), tl = document.getElementById("tl");
let live = null;                    // polygon being drawn right now
const FR = 1 / M.fps;
function times(){ return Object.keys(kfs).map(Number).sort((a,b)=>a-b); }
function frameOf(t){ return Math.round(t * M.fps); }
function keyOf(t){ return String(frameOf(t)); }
function ellipse(x0,y0,x1,y1){
  const cx=(x0+x1)/2, cy=(y0+y1)/2, rx=Math.abs(x1-x0)/2, ry=Math.abs(y1-y0)/2;
  const out=[];
  for(let k=0;k<48;k++){ const a=2*Math.PI*k/48;
    out.push([Math.round(cx+rx*Math.cos(a)), Math.round(cy+ry*Math.sin(a))]); }
  return out;
}
// The outline at an arbitrary time: interpolate between neighbouring keyframes
// when their vertex counts agree, hold the earlier one when they do not.
function polyAt(t){
  if (live) return live;
  const f = frameOf(t), ts = times();
  if (!ts.length) return null;
  let lo = null, hi = null;
  for (const k of ts){ if (k <= f) lo = k; else { hi = k; break; } }
  if (lo === null) return kfs[hi];
  if (hi === null || kfs[lo].length !== kfs[hi].length) return kfs[lo];
  const a = (f - lo) / (hi - lo), A = kfs[lo], B = kfs[hi];
  return A.map((p,i)=>[p[0]+(B[i][0]-p[0])*a, p[1]+(B[i][1]-p[1])*a]);
}
function paint(){
  ctx.clearRect(0,0,c.width,c.height);
  const p = polyAt(v.currentTime);
  if (p && p.length){
    ctx.strokeStyle = "#ffd33d"; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(p[0][0], p[0][1]);
    for (let i=1;i<p.length;i++) ctx.lineTo(p[i][0], p[i][1]);
    ctx.closePath(); ctx.stroke();
    ctx.fillStyle = "rgba(255,211,61,.10)"; ctx.fill();
    if (p.length < 20) p.forEach(q => ctx.fillRect(q[0]-3,q[1]-3,6,6));
  }
  const f = frameOf(v.currentTime), on = kfs[keyOf(v.currentTime)];
  document.getElementById("info").innerHTML =
    "<b>" + M.rec + "</b> 帧 " + (M.start + f) + "  " +
    v.currentTime.toFixed(2) + "s / " + M.dur.toFixed(1) + "s &nbsp; 关键帧 " +
    times().length + (on ? " <b>（当前是关键帧）</b>" : "");
  document.getElementById("head").style.left =
    (v.currentTime / M.dur * 100) + "%";
  document.getElementById("play").style.width =
    (v.currentTime / M.dur * 100) + "%";
  tl.querySelectorAll(".kf").forEach(e => e.remove());
  for (const k of times()){
    const d = document.createElement("div");
    d.className = "kf"; d.style.left = (k / M.fps / M.dur * 100) + "%";
    tl.appendChild(d);
  }
}
function save(){ localStorage.setItem(KEY, JSON.stringify(kfs)); paint(); }
let drag = null;
c.onmousedown = e => { const r=c.getBoundingClientRect();
  drag=[e.clientX-r.left, e.clientY-r.top, false]; };
c.onmousemove = e => {
  if(!drag) return;
  const r=c.getBoundingClientRect(), x=e.clientX-r.left, y=e.clientY-r.top;
  if (Math.abs(x-drag[0])<6 && Math.abs(y-drag[1])<6) return;
  drag[2]=true; live = ellipse(drag[0],drag[1],x,y); paint();
};
c.onmouseup = e => {
  const r=c.getBoundingClientRect();
  if (drag && !drag[2]){
    const base = live || polyAt(v.currentTime) || [];
    live = (base.length && base.length < 20) ? base.slice() : [];
    live.push([Math.round(e.clientX-r.left), Math.round(e.clientY-r.top)]);
  }
  drag = null;
  if (live){ kfs[keyOf(v.currentTime)] = live; live = null; save(); }
};
tl.onclick = e => { const r = tl.getBoundingClientRect();
  v.currentTime = Math.max(0, Math.min(M.dur,
    (e.clientX - r.left) / r.width * M.dur)); paint(); };
function tog(){ v.paused ? v.play() : v.pause(); }
function step(dt){ v.pause(); v.currentTime =
  Math.max(0, Math.min(M.dur - FR, v.currentTime + dt)); }
document.onkeydown = ev => {
  if (ev.key === " ") tog();
  else if (ev.key === ",") step(-FR);
  else if (ev.key === ".") step(FR);
  else if (ev.key === "ArrowLeft") step(-1);
  else if (ev.key === "ArrowRight") step(1);
  else if (ev.key === "c"){ live=null; delete kfs[keyOf(v.currentTime)]; save(); }
  else if (ev.key === "u"){
    const k = keyOf(v.currentTime);
    if (kfs[k] && kfs[k].length){ kfs[k].pop();
      if (!kfs[k].length) delete kfs[k]; save(); }
  }
  else if (ev.key === "k"){
    const p = polyAt(v.currentTime);
    if (p) { kfs[keyOf(v.currentTime)] = p.map(q=>[Math.round(q[0]),
                                                   Math.round(q[1])]); save(); }
  }
  else if (ev.key === "d"){ delete kfs[keyOf(v.currentTime)]; save(); }
  else return;
  ev.preventDefault();
};
v.addEventListener("loadedmetadata", paint);
(function loop(){ paint(); requestAnimationFrame(loop); })();
function dl(){
  // Exported in cam3's own pixels, not the ones on screen: the clip was
  // scaled to fit a browser and nothing downstream should have to know by
  // how much.
  const s = M.full_w / M.w;
  const out = {recording: M.rec, camera: "cam3",
               source: M.source, clip_start_frame: M.start, fps: M.fps,
               cam3_size: [M.full_w, M.full_h],
               interpolation: "linear between keyframes with equal vertex "
                              + "counts, hold otherwise",
               keyframes: times().map(f => ({
                 frame: M.start + f, t: +(f / M.fps).toFixed(3),
                 polygon_cam3: kfs[String(f)].map(p =>
                   [Math.round(p[0]*s), Math.round(p[1]*s)])}))};
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
        # one is cam3; annotating both would be the same region drawn twice.
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
