"""Draw the wearer's own region once, and see it in every camera.

A BINARY REGION IS A DIFFERENT KIND OF LABEL FROM A BOX PER HAND. Everything
this project has annotated so far names an object -- is this a hand, whose is
it, are these two the same one -- and every one of those took a person a
second per track. A region says something about the SCENE instead: this part
of the view is where the wearer's own hands can be, and the rest is not. One
drag covers a recording.

IT IS DRAWN ON cam3 BECAUSE THAT IS THE MIDDLE OF THE FAN. The three modules
sit at 0, 29.5 and 54.3 degrees of yaw, so cam1 and cam5 are the outer ones
and cam3 sees the overlap with both. A region drawn there has somewhere to be
mapped to in either direction.

MAPPING NEEDS A DEPTH AND THAT IS NOT A DETAIL. A region in an image is a cone
of rays, not a set of points, and the modules are 9.3 cm apart (cam1-cam3),
9.4 (cam3-cam5) and 17.3 end to end. At half a metre that separation subtends
about eleven degrees, so `the same region` in cam1 depends on how far away the
thing in it is. The depth is therefore explicit and the tool shows the region
at several of them at once: if the outline barely moves between 0.3 m and 1 m,
the choice does not matter here; if it swings, the region is not
depth-independent and nothing downstream should pretend it is.

THE PANORAMA IS THE EXCEPTION. Its own render already fixes a depth, so a
region mapped into it inherits that assumption rather than adding one, and
that is the frame the pipeline actually runs in.

DETECTIONS ARE SHOWN IN ONE COLOUR, DELIBERATELY. Seeing where the hands are
helps; seeing which ones the model called the wearer's would turn the region
into a drawing of the model's current decision boundary, which is the one
thing it must not be.
"""
from __future__ import annotations

import argparse
import base64
import collections
import csv
import json
import math
import os

SHEET = """<meta charset=utf-8><title>wearer zone</title><style>
body{font:13px/1.5 system-ui;margin:0;background:#111;color:#ddd}
#bar{position:sticky;top:0;background:#181818;padding:9px 14px;z-index:9;
  border-bottom:1px solid #333;display:flex;gap:14px;align-items:center;
  flex-wrap:wrap}
button{font:13px system-ui;padding:5px 10px;cursor:pointer}
select{font:13px system-ui;padding:4px}
b{color:#ffd33d}
.f{padding:12px 14px;border-bottom:1px solid #262626}
.f.cur{background:#1d2430;outline:2px solid #4a8}
.f.done{border-left:5px solid #2a6}
.hd{color:#9ab;font-size:12px;margin-bottom:6px}
.row{display:flex;gap:8px;align-items:flex-start;flex-wrap:wrap}
figure{margin:0}
figcaption{font-size:10px;color:#888;text-align:center}
canvas{border-radius:3px;display:block}
canvas.main{cursor:crosshair}
.key{font-size:11px;color:#999;max-width:760px}
</style>
<div id=bar><span id=prog></span>
<span class=key>在 <b>cam3</b> 上画：<b>拖拽</b>=椭圆（半圆就把它拖出画面下沿），
<b>点击</b>=多边形顶点，<b>c</b> 清空，<b>u</b> 撤销一个顶点。
右边三张是同一个区域映射过去的样子。深度：<select id=depth></select>
<b>&uarr;&darr;</b> 换帧</span>
<button onclick="dl()">download JSON</button></div>
<div id=list></div>
<script>
const D = __PAYLOAD__;
const KEY = "zone:" + D.tag;
let st = {}, cur = 0, depth = 0;
try { st = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch(e) { st={}; }
const sel = document.getElementById("depth");
D.depths.forEach((z, i) => {
  const o = document.createElement("option");
  o.value = i; o.textContent = z + " m"; sel.appendChild(o);
});
sel.onchange = () => { depth = +sel.value; D.frames.forEach((_,i)=>paint(i)); };
const list = document.getElementById("list");
D.frames.forEach((fr, i) => {
  const d = document.createElement("div");
  d.className = "f"; d.id = "f" + i;
  d.innerHTML = '<div class=hd>' + fr.rec + '  f' + fr.frame +
    ' <span id=v' + i + '></span></div><div class=row>' +
    '<figure><canvas class=main id="m' + i + '" width="' + fr.w +
    '" height="' + fr.h + '"></canvas><figcaption>cam3（在这里画）' +
    '</figcaption></figure>' +
    fr.views.map((v, j) =>
      '<figure><canvas id="v' + i + '_' + j + '" width="' + v.w +
      '" height="' + v.h + '"></canvas><figcaption>' + v.name +
      '</figcaption></figure>').join('') + '</div>';
  d.onclick = () => { cur = i; draw(); };
  list.appendChild(d);
});
const imgs = {};
function load(key, src, cb){
  const im = new Image(); im.onload = () => { imgs[key] = im; cb(); };
  im.src = src;
}
D.frames.forEach((fr, i) => {
  load("m" + i, fr.img, () => paint(i));
  fr.views.forEach((v, j) => load("v" + i + "_" + j, v.img, () => paint(i)));
  const cv = document.getElementById("m" + i);
  let drag = null;
  cv.onmousedown = e => {
    const r = cv.getBoundingClientRect();
    drag = [e.clientX - r.left, e.clientY - r.top, false];
  };
  cv.onmousemove = e => {
    if (!drag) return;
    const r = cv.getBoundingClientRect();
    const x = e.clientX - r.left, y = e.clientY - r.top;
    if (Math.abs(x-drag[0]) < 6 && Math.abs(y-drag[1]) < 6) return;
    drag[2] = true;
    st[fr.key] = {poly: ellipse(drag[0], drag[1], x, y), kind: "ellipse"};
    paint(i);
  };
  cv.onmouseup = e => {
    const r = cv.getBoundingClientRect();
    if (drag && !drag[2]){            // a click, not a drag: polygon vertex
      const s = st[fr.key];
      const pts = (s && s.kind === "poly") ? s.poly : [];
      pts.push([Math.round(e.clientX - r.left), Math.round(e.clientY - r.top)]);
      st[fr.key] = {poly: pts, kind: "poly"};
      paint(i);
    }
    drag = null; save(); draw();
  };
});
function ellipse(x0, y0, x1, y1){
  const cx = (x0+x1)/2, cy = (y0+y1)/2;
  const rx = Math.abs(x1-x0)/2, ry = Math.abs(y1-y0)/2, out = [];
  for (let k = 0; k < 48; k++){
    const t = 2*Math.PI*k/48;
    out.push([Math.round(cx + rx*Math.cos(t)), Math.round(cy + ry*Math.sin(t))]);
  }
  return out;
}
function save(){ localStorage.setItem(KEY, JSON.stringify(st)); }
// Bilinear lookup in the precomputed cam3 -> target grid. Null anywhere in the
// cell means the point leaves that camera, and the segment is dropped rather
// than drawn to a guessed place.
function mapPt(fr, j, p){
  const G = fr.views[j].grid[depth], gx = fr.gx, gy = fr.gy;
  const fx = p[0] / fr.w * (gx - 1), fy = p[1] / fr.h * (gy - 1);
  const x0 = Math.max(0, Math.min(gx-2, Math.floor(fx)));
  const y0 = Math.max(0, Math.min(gy-2, Math.floor(fy)));
  const ax = fx - x0, ay = fy - y0;
  const c = [G[y0*gx+x0], G[y0*gx+x0+1], G[(y0+1)*gx+x0], G[(y0+1)*gx+x0+1]];
  if (c.some(v => v === null)) return null;
  return [ (c[0][0]*(1-ax)+c[1][0]*ax)*(1-ay) + (c[2][0]*(1-ax)+c[3][0]*ax)*ay,
           (c[0][1]*(1-ax)+c[1][1]*ax)*(1-ay) + (c[2][1]*(1-ax)+c[3][1]*ax)*ay ];
}
function outline(ctx, pts, colour){
  ctx.strokeStyle = colour; ctx.lineWidth = 2;
  ctx.beginPath();
  let pen = false;
  for (let k = 0; k <= pts.length; k++){
    const p = pts[k % pts.length];
    if (p === null){ pen = false; continue; }
    if (!pen){ ctx.moveTo(p[0], p[1]); pen = true; } else ctx.lineTo(p[0], p[1]);
  }
  ctx.stroke();
}
function paint(i){
  const fr = D.frames[i];
  const cv = document.getElementById("m" + i), ctx = cv.getContext("2d");
  if (imgs["m"+i]) ctx.drawImage(imgs["m"+i], 0, 0); else ctx.clearRect(0,0,cv.width,cv.height);
  const s = st[fr.key];
  if (s) {
    outline(ctx, s.poly, "#ffd33d");
    if (s.kind === "poly")
      s.poly.forEach(p => { ctx.fillStyle="#ffd33d";
                            ctx.fillRect(p[0]-2, p[1]-2, 4, 4); });
  }
  fr.views.forEach((v, j) => {
    const c2 = document.getElementById("v"+i+"_"+j), x2 = c2.getContext("2d");
    if (imgs["v"+i+"_"+j]) x2.drawImage(imgs["v"+i+"_"+j], 0, 0);
    else x2.clearRect(0,0,c2.width,c2.height);
    if (s) outline(x2, s.poly.map(p => mapPt(fr, j, p)), "#ffd33d");
  });
}
function draw(){
  D.frames.forEach((fr,i)=>{
    document.getElementById("f"+i).className =
      "f" + (st[fr.key] ? " done" : "") + (i===cur ? " cur" : "");
    document.getElementById("v"+i).textContent =
      st[fr.key] ? ("已画 " + st[fr.key].poly.length + " 点") : "";
  });
  document.getElementById("prog").innerHTML = "<b>" + D.tag + "</b> &nbsp; " +
    D.frames.filter(f => st[f.key]).length + "/" + D.frames.length + " 帧已画";
  save();
  const el = document.getElementById("f"+cur);
  if (el) el.scrollIntoView({block:"nearest"});
}
document.onkeydown = ev => {
  const fr = D.frames[cur]; if(!fr) return;
  if (ev.key === "c"){ delete st[fr.key]; paint(cur); }
  else if (ev.key === "u"){
    const s = st[fr.key];
    if (s && s.kind === "poly" && s.poly.length) { s.poly.pop(); paint(cur); }
  }
  else if (ev.key === "ArrowDown"){ cur = Math.min(cur+1, D.frames.length-1); }
  else if (ev.key === "ArrowUp"){ cur = Math.max(cur-1, 0); }
  else return;
  draw(); ev.preventDefault();
};
draw();
function dl(){
  const out = [];
  for (const fr of D.frames){
    const s = st[fr.key];
    if (!s) continue;
    out.push({rec: fr.rec, frame: fr.frame, camera: "cam3",
              cam3_size: [fr.full_w, fr.full_h], shown_size: [fr.w, fr.h],
              kind: s.kind, depth_m: D.depths[depth],
              polygon_cam3: s.poly.map(p => [Math.round(p[0]*fr.full_w/fr.w),
                                             Math.round(p[1]*fr.full_h/fr.h)])});
  }
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([JSON.stringify(out, null, 1)],
                                        {type:"application/json"}));
  a.download = "wearer_zone_" + D.tag + ".json"; a.click();
}
</script>
"""


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", required=True)
    ap.add_argument("--rows", help="own_dump csv, to show where hands are")
    ap.add_argument("--rec", action="append",
                    help="recording tag; repeatable, default all in --clips")
    ap.add_argument("--frames", type=int, default=3,
                    help="how many moments per recording")
    ap.add_argument("--depths", default="0.3,0.5,1.0")
    ap.add_argument("--grid", type=int, default=33)
    ap.add_argument("--main_w", type=int, default=560)
    ap.add_argument("--view_w", type=int, default=300)
    ap.add_argument("--page", type=int, default=30)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import cv2
    import numpy as np
    from src.rig.calibration import RigCalibration
    from src.rig.geometry import VirtualWideCamera
    from src.rig.render_wide import render
    from src.rig.seam_fix import ClipReader, Prefetch

    depths = [float(x) for x in a.depths.split(",")]
    clips = {}
    for line in open(a.clips):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = line.rsplit(":", 2)
        clips[os.path.basename(p[0].rstrip("/")).replace("databag-26_", "R")] \
            = (p[0], int(p[1]))
    tags = [t for t in (a.rec or sorted(clips)) if t in clips]

    dets = collections.defaultdict(list)
    if a.rows:
        for r in csv.DictReader(open(a.rows, encoding="utf-8-sig")):
            dets[(r["rec"], int(r["frame"]))].append(
                (int(r["x0"]), int(r["y0"]), int(r["x1"]), int(r["y1"])))

    def b64(img, w, q=86):
        h = int(round(img.shape[0] * w / max(1, img.shape[1])))
        ok, buf = cv2.imencode(".jpg", cv2.resize(img, (w, max(1, h))),
                               [int(cv2.IMWRITE_JPEG_QUALITY), q])
        return (("data:image/jpeg;base64,"
                 + base64.b64encode(buf).decode()) if ok else "", w, h)

    out = []
    for tag in tags:
        databag, start = clips[tag]
        rig = RigCalibration(os.path.join(databag, "calibration.yaml"))
        vcam = VirtualWideCamera.from_rig(rig)
        src_cam, others = "cam3", ["cam1", "cam5"]
        cams = {n: rig.cameras[n] for n in [src_cam] + others}
        SW, SH = cams[src_cam].width, cams[src_cam].height

        # ONE GRID PER RECORDING, NOT PER FRAME: it depends only on the
        # calibration. cam3 pixel -> ray -> a point at depth Z -> the pixel it
        # lands on in each of the other views.
        gx = gy = a.grid
        us = np.linspace(0, SW - 1, gx)
        vs = np.linspace(0, SH - 1, gy)
        uu, vv = np.meshgrid(us, vs)
        pix = np.stack([uu.ravel(), vv.ravel()], -1).astype(np.float64)
        rays = cv2.fisheye.undistortPoints(
            pix.reshape(-1, 1, 2), np.asarray(cams[src_cam].K, np.float64),
            np.asarray(cams[src_cam].D, np.float64).reshape(4, 1)
        ).reshape(-1, 2)
        rays = np.concatenate([rays, np.ones((len(rays), 1))], 1)
        R3 = np.asarray(cams[src_cam].R, np.float64)
        t3 = np.asarray(cams[src_cam].t, np.float64).reshape(3)

        grids = {n: [] for n in others + ["pano"]}
        for Z in depths:
            p_ref = (R3 @ (rays * Z).T).T + t3          # reference frame
            for n in others:
                cam = cams[n]
                Rn = np.asarray(cam.R, np.float64)
                tn = np.asarray(cam.t, np.float64).reshape(3)
                p_cam = (Rn.T @ (p_ref - tn).T).T
                ok = p_cam[:, 2] > 1e-6
                uv = np.full((len(p_cam), 2), np.nan)
                if ok.any():
                    q, _ = cv2.fisheye.projectPoints(
                        p_cam[ok].reshape(-1, 1, 3), np.zeros(3), np.zeros(3),
                        np.asarray(cam.K, np.float64),
                        np.asarray(cam.D, np.float64).reshape(4, 1))
                    uv[ok] = q.reshape(-1, 2)
                bad = (~np.isfinite(uv).all(1)) | (uv[:, 0] < 0) \
                    | (uv[:, 0] >= cam.width) | (uv[:, 1] < 0) \
                    | (uv[:, 1] >= cam.height)
                grids[n].append([None if b else [round(x, 1), round(y, 1)]
                                 for b, (x, y) in zip(bad, uv)])
            # The panorama's own render already fixes a depth, so mapping into
            # it adds no assumption beyond the one the renderer made.
            d = p_ref - vcam.eye
            w = (np.asarray(vcam.R, np.float64).T @ d.T).T
            nrm = np.linalg.norm(w, axis=1)
            az = np.arctan2(w[:, 0], w[:, 2])
            el = np.arcsin(np.clip(-w[:, 1] / np.maximum(nrm, 1e-9), -1, 1))
            px = (az / vcam.hfov + 0.5) * vcam.width
            py = (0.5 - el / vcam.vfov) * vcam.height
            bad = (px < 0) | (px >= vcam.width) | (py < 0) \
                | (py >= vcam.height) | (w[:, 2] <= 0)
            grids["pano"].append([None if b else [round(x, 1), round(y, 1)]
                                  for b, (x, y) in zip(bad, zip(px, py))])

        # Spread across the same 400-frame window every other sheet uses, so
        # a zone can be checked against more than one moment of the recording.
        picks = sorted({start + int(round(i * 399 / max(1, a.frames - 1)))
                        for i in range(a.frames)})
        lo, hi = min(picks), max(picks)
        rd = Prefetch(ClipReader(rig, {k: os.path.join(databag, f"{k}.mp4")
                                       for k in ("cam12", "cam34", "cam56")},
                                 lo), skip=0)
        mc = {}
        print(f"  {tag}: 读 {lo}-{hi}", flush=True)
        for i in range(hi - lo + 1):
            s = rd.next()
            if not s:
                break
            f = lo + i
            if f not in picks:
                continue
            try:
                rgb, _, _, _ = render(rig, vcam, s, 0.6, map_cache=mc)
            except TypeError:
                rgb, _, _, _ = render(rig, vcam, s, 0.6)
            pano = rgb.copy()
            for (x0, y0, x1, y1) in dets.get((tag, f), []):
                cv2.rectangle(pano, (x0, y0), (x1, y1), (170, 170, 170), 2)
            m_img, mw, mh = b64(s[src_cam], a.main_w)
            views = []
            for n in others:
                im, w_, h_ = b64(s[n], a.view_w)
                views.append({"name": n, "img": im, "w": w_, "h": h_,
                              "grid": [[None if p is None else
                                        [p[0] * w_ / cams[n].width,
                                         p[1] * h_ / cams[n].height]
                                        for p in g] for g in grids[n]]})
            im, w_, h_ = b64(pano, a.view_w)
            views.append({"name": "全景（灰框=检测到的手）", "img": im,
                          "w": w_, "h": h_,
                          "grid": [[None if p is None else
                                    [p[0] * w_ / vcam.width,
                                     p[1] * h_ / vcam.height]
                                    for p in g] for g in grids["pano"]]})
            out.append({"key": f"{tag}:{f}", "rec": tag, "frame": f,
                        "img": m_img, "w": mw, "h": mh,
                        "full_w": SW, "full_h": SH, "gx": gx, "gy": gy,
                        "views": views})
        rd.close()

    stem, ext = os.path.splitext(a.out)
    n_pg = max(1, (len(out) + a.page - 1) // a.page)
    for i in range(n_pg):
        part = out[i * a.page:(i + 1) * a.page]
        path = a.out if n_pg == 1 else f"{stem}_{i + 1}{ext}"
        tag = "all" if n_pg == 1 else f"p{i + 1}"
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(SHEET.replace("__PAYLOAD__",
                                   json.dumps({"tag": tag, "frames": part,
                                               "depths": depths})))
        print(f"  {len(part):>3} 帧 -> {path} "
              f"({os.path.getsize(path) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
