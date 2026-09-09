"""Why does a whole track sit on the wrong side of the reference?

FORTY-NINE TRACKS CARRY 58.9% OF THE OWNERSHIP ERROR MASS. The aggregation
census split 343 tracks four ways and the two that no temporal method can
touch -- C, mixed but wrong in aggregate, and D, confidently wrong throughout
-- hold 737 of 1251 wrong frames between them. C is fourteen tracks and is the
largest single block at 35.5%. Nothing about thresholds or smoothing reaches
them: the classifier is not undecided, it is decided and wrong for the length
of a hand's appearance.

THE QUESTION IS NOT WHERE TO PUT A THRESHOLD. Sorting these by model score
would answer a question already known to have no answer -- the errors are
confident. What is being asked is what the model was looking at, so the sheet
shows the crop it classified, the context window it was given, and the frame
around them, and asks which of eight mechanisms explains the verdict.

C AND D ARE SEPARATED BECAUSE THEY FAIL DIFFERENTLY. C flips, so some frames
were read correctly and the representation demonstrably carries something; the
question there is what changed -- viewpoint, occlusion, how much context the
crop caught. D never wavers, which points at a confound rather than a regime
change, and D is where a missing cue would show itself.

THE REFERENCE GETS A CATEGORY OF ITS OWN. It is a frozen rule, not a human
verdict, and 44 tracks in the same run had the rule contradicting itself within
one hand. Some share of these 49 will be the rule's boundary rather than the
classifier's mistake, and folding those into `the model was wrong` would
manufacture a failure mode that does not exist.
"""
from __future__ import annotations

import argparse
import base64
import collections
import csv
import json
import os

CAUSES = [
    ("appearance", "外观混淆", "别人的手看着像自己的，或反过来"),
    ("body_relation", "缺身体关联", "裁剪合理，但要看手臂连到谁才判得了"),
    ("truncation", "第一视角截断", "自己的手过大／贴边／极端视角"),
    ("overlap", "重叠遮挡", "两只手或前臂叠在一起"),
    ("context", "上下文不够", "2.5× 窗口里根本没有能判归属的信息"),
    ("crop_bad", "框或轨迹有问题", "框偏、混进手臂或物体、轨迹太短"),
    ("ref_bad", "参照规则可疑", "出口高度规则在这一例上可能不适用"),
    ("ambiguous", "真的说不清", ""),
]

SHEET = """<meta charset=utf-8><title>ownership mechanism census</title><style>
body{font:13px/1.5 system-ui;margin:0;background:#111;color:#ddd}
#bar{position:sticky;top:0;background:#181818;padding:9px 14px;z-index:9;
  border-bottom:1px solid #333;display:flex;gap:14px;align-items:center;
  flex-wrap:wrap}
button{font:13px system-ui;padding:5px 10px;cursor:pointer}
b{color:#ffd33d}
.t{padding:12px 14px;border-bottom:1px solid #262626}
.t.cur{background:#1d2430;outline:2px solid #4a8}
.t.done{border-left:5px solid #2a6}
.hd{color:#9ab;font-size:12px;margin-bottom:6px}
.grp{padding:1px 8px;border-radius:3px;font-size:11px;margin-right:6px}
.grp.C{background:#a63}
.grp.D{background:#a33}
.dir{padding:1px 8px;border-radius:3px;font-size:11px;background:#356}
.num{color:#8ab4c8;font-family:ui-monospace,monospace;font-size:11px}
.strip{display:flex;gap:5px;align-items:flex-start}
.strip figure{margin:0;flex:1}
.strip img{width:100%;border-radius:3px;display:block}
.strip figcaption{font-size:10px;color:#888;text-align:center}
.key{font-size:11px;color:#aaa}
</style>
<div id=bar>
 <span id=prog></span>
 <span id=keys></span>
 <button onclick="dl()">download CSV</button>
 <span class=key>上排=分类器看到的裁剪窗口，下排=同一时刻的整帧。
 问的是：为什么整条轨迹站在参照的错误一侧。</span>
</div>
<div id=list></div>
<script>
const D = __PAYLOAD__;
const KEY = "census:" + D.tag;
const CLS = __CLS__;
const CN = __CN__;
let lab = {}, cur = 0, hist = [];
try { lab = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch(e) { lab={}; }
document.getElementById("keys").innerHTML = CLS.map((c,i) =>
  '<b>' + (i+1) + '</b> ' + CN[c]).join(' &nbsp; ') +
  ' &nbsp; <b>&uarr;&darr;</b> move &nbsp; <b>u</b> undo';
const list = document.getElementById("list");
D.tracks.forEach((t, i) => {
  const d = document.createElement("div");
  d.className = "t"; d.id = "t" + i;
  d.innerHTML = '<div class=hd><span class="grp ' + t.grp + '">' + t.grp +
    '</span><span class=dir>' + t.dir + '</span> ' + t.rec +
    ' &nbsp;<span class=num>track ' + t.tid + '  f' + t.first + '-' + t.last +
    '  ' + t.n + ' 帧  p_raw ' + t.p_lo + '-' + t.p_hi + ' (中位 ' + t.p_med +
    ')  错 ' + t.wrong + '/' + t.n + '</span> <span id=v' + i + '></span></div>' +
    '<div class=strip>' + t.crops.map(f =>
      '<figure><img src="' + f.c + '"><figcaption>f' + f.f + '  p=' + f.p +
      '</figcaption></figure>').join('') + '</div>' +
    '<div class=strip style="margin-top:4px">' + t.crops.map(f =>
      '<figure><img src="' + f.w + '"></figure>').join('') + '</div>';
  d.onclick = () => { cur = i; draw(); };
  list.appendChild(d);
});
function draw(){
  D.tracks.forEach((t,i)=>{
    const v = lab[t.key];
    document.getElementById("t"+i).className =
      "t" + (v ? " done" : "") + (i===cur ? " cur" : "");
    document.getElementById("v"+i).innerHTML =
      v ? '<span class=grp style="background:#356">' + CN[v] + '</span>' : '';
  });
  document.getElementById("prog").innerHTML =
    "<b>" + D.tag + "</b> &nbsp; " + Object.keys(lab).length + "/" +
    D.tracks.length + " 已判";
  localStorage.setItem(KEY, JSON.stringify(lab));
  const el = document.getElementById("t"+cur);
  if(el) el.scrollIntoView({block:"nearest"});
}
document.onkeydown = ev => {
  const n = +ev.key;
  if(n >= 1 && n <= CLS.length){
    const t = D.tracks[cur]; if(!t) return;
    hist.push([t.key, lab[t.key]]);
    lab[t.key] = CLS[n-1];
    cur = Math.min(cur+1, D.tracks.length-1); draw();
  }
  else if(ev.key==="ArrowDown"){cur=Math.min(cur+1,D.tracks.length-1);draw();}
  else if(ev.key==="ArrowUp"){cur=Math.max(cur-1,0);draw();}
  else if(ev.key==="u"){const h=hist.pop(); if(h){ if(h[1]===undefined)
    delete lab[h[0]]; else lab[h[0]]=h[1]; draw(); }}
  else return;
  ev.preventDefault();
};
draw();
function dl(){
  let s = "key,rec,tid,group,direction,frames,wrong,p_median,cause\\n";
  for(const t of D.tracks) if(lab[t.key])
    s += t.key + "," + t.rec + "," + t.tid + "," + t.grp + "," + t.dir + "," +
         t.n + "," + t.wrong + "," + t.p_med + "," + lab[t.key] + "\\n";
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([s],{type:"text/csv"}));
  a.download = "own_census.csv"; a.click();
}
</script>
"""


def groups(rows):
    """-> {(rec, tid): (group, reference_owner, rows)} for C and D only."""
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["rec"], r["tid"])].append(r)
    out = {}
    for key, v in by.items():
        v.sort(key=lambda r: int(r["frame"]))
        ref = [int(r["reference_owner"]) for r in v]
        if len(set(ref)) > 1:
            continue                       # the rule's own boundary, not this
        truth = bool(ref[0])
        ps = [float(r["p_owner_raw"]) for r in v]
        maj = sum(1 for p in ps if p >= 0.5) * 2 >= len(ps)
        if maj == truth:
            continue                       # A or B
        mixed = len({p >= 0.5 for p in ps}) > 1
        out[key] = ("C" if mixed else "D", truth, v)
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", required=True, help="own_dump.csv")
    ap.add_argument("--clips", default="/workspace/e2e_main2.txt")
    ap.add_argument("--shots", type=int, default=4)
    ap.add_argument("--crop_w", type=int, default=170)
    ap.add_argument("--wide_w", type=int, default=170)
    ap.add_argument("--ctx_scale", type=float, default=2.5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

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

    rows = list(csv.DictReader(open(a.rows, encoding="utf-8-sig")))
    # The dump does not store boxes, so the crop is rebuilt from the frame and
    # the track's own extent; what matters for the census is what the region
    # looked like, not the box to the pixel.
    has_box = "x0" in rows[0]
    g = groups(rows)
    print(f"  C+D {len(g)} 条轨迹  "
          f"{collections.Counter(v[0] for v in g.values())}")

    by_rec = collections.defaultdict(list)
    for (rec, tid), v in g.items():
        by_rec[rec].append((tid, v))

    out = []
    for rec in sorted(by_rec):
        if rec not in clips:
            continue
        databag, _ = clips[rec]
        rig = RigCalibration(os.path.join(databag, "calibration.yaml"))
        vcam = VirtualWideCamera.from_rig(rig)
        vids = {k: os.path.join(databag, f"{k}.mp4")
                for k in ("cam12", "cam34", "cam56")}
        want = collections.defaultdict(list)
        for tid, (grp, truth, v) in by_rec[rec]:
            ks = [int(r["frame"]) for r in v]
            step = max(1, len(ks) // a.shots)
            pick = list(dict.fromkeys(ks[::step][:a.shots] or ks[:1]))
            for f in pick:
                want[f].append((tid, grp, truth, v))
        lo, hi = min(want), max(want)
        rd = Prefetch(ClipReader(rig, vids, lo), skip=0)
        mc, imgs = {}, {}
        print(f"  {rec}: {len(by_rec[rec])} 条, 读 {lo}-{hi}", flush=True)
        for k in range(hi - lo + 1):
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

        def b64(img, w):
            h = int(round(img.shape[0] * w / max(1, img.shape[1])))
            ok, buf = cv2.imencode(".jpg", cv2.resize(img, (w, max(1, h))),
                                   [int(cv2.IMWRITE_JPEG_QUALITY), 84])
            return ("data:image/jpeg;base64,"
                    + base64.b64encode(buf).decode()) if ok else ""

        for tid, (grp, truth, v) in by_rec[rec]:
            ks = [int(r["frame"]) for r in v]
            step = max(1, len(ks) // a.shots)
            pick = list(dict.fromkeys(ks[::step][:a.shots] or ks[:1]))
            pmap = {int(r["frame"]): float(r["p_owner_raw"]) for r in v}
            shots = []
            for f in pick:
                img = imgs.get(f)
                if img is None:
                    continue
                H, W = img.shape[:2]
                if has_box:
                    r0 = next(r for r in v if int(r["frame"]) == f)
                    x0, y0 = int(r0["x0"]), int(r0["y0"])
                    x1, y1 = int(r0["x1"]), int(r0["y1"])
                else:
                    x0 = y0 = x1 = y1 = 0
                if x1 > x0:
                    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
                    half = int(max(x1 - x0, y1 - y0) * a.ctx_scale / 2)
                    cx0, cy0 = max(0, cx - half), max(0, cy - half)
                    cx1, cy1 = min(W, cx + half), min(H, cy + half)
                    crop = img[cy0:cy1, cx0:cx1].copy()
                    cv2.rectangle(crop, (x0 - cx0, y0 - cy0),
                                  (x1 - cx0, y1 - cy0), (60, 255, 60), 2)
                else:
                    crop = img
                wide = img.copy()
                if x1 > x0:
                    cv2.rectangle(wide, (x0, y0), (x1, y1), (60, 255, 60), 3)
                shots.append({"f": f, "p": f"{pmap.get(f, 0.0):.2f}",
                              "c": b64(crop, a.crop_w),
                              "w": b64(wide, a.wide_w)})
            if not shots:
                continue
            ps = sorted(pmap.values())
            wrong = sum(1 for r in v
                        if bool(int(r["final_owner_post_cap"])) != truth)
            out.append({
                "key": f"{rec}:{tid}", "rec": rec, "tid": tid, "grp": grp,
                "dir": ("false other 参照=自己" if truth
                        else "false owner 参照=别人"),
                "first": ks[0], "last": ks[-1], "n": len(v), "wrong": wrong,
                "p_lo": f"{ps[0]:.2f}", "p_hi": f"{ps[-1]:.2f}",
                "p_med": f"{ps[len(ps) // 2]:.2f}", "crops": shots})

    # C first: it holds the most error mass and its flips say the
    # representation carries something, which is the more actionable half.
    out.sort(key=lambda t: (t["grp"] != "C", -t["wrong"]))
    cls = [c for c, _, _ in CAUSES]
    cn = {c: n for c, n, _ in CAUSES}
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(SHEET.replace("__PAYLOAD__",
                              json.dumps({"tag": "CD", "tracks": out}))
                .replace("__CLS__", json.dumps(cls))
                .replace("__CN__", json.dumps(cn, ensure_ascii=False)))
    print(f"\n  {len(out)} 条 -> {a.out} "
          f"({os.path.getsize(a.out) / 1e6:.1f} MB)")
    for i, (c, n, why) in enumerate(CAUSES, 1):
        print(f"  {i} {n:<12} {why}")


if __name__ == "__main__":
    main()
