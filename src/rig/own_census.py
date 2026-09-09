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

QUESTIONS = [
    ("human_ownership", "这只手是谁的",
     # `not_a_hand` IS NOT AN OWNERSHIP ANSWER AND THAT IS THE POINT. A red
     # cloth, a bag of peppers, a basket of greens and a box degenerated to a
     # vertical line all arrived here as ownership errors; the detector
     # proposed a non-hand and the classifier then dutifully assigned it an
     # owner. Nothing about ownership caused that, so folding those into
     # `the classifier was wrong` would send the repair to the wrong stage --
     # the same mistake the cover taxonomy already had to fix once.
     [("owner", "自己"), ("other", "别人"),
      ("not_a_hand", "根本不是手"), ("unsure", "说不准")], False),
    ("reference_rule", "冻结出口规则判得对吗",
     [("agrees", "规则对"), ("suspicious", "规则可疑"),
      ("contradicted", "规则相反")], False),
    ("mechanism", "整条为什么站错侧",
     [("appearance", "外观混淆"), ("body_relation", "缺身体关联"),
      ("truncation", "第一视角截断"), ("overlap", "重叠遮挡"),
      ("context", "上下文不够"), ("crop_bad", "框或轨迹有问题"),
      ("ref_bad", "参照规则可疑"), ("ambiguous", "真的说不清")], False),
    ("entry_distinct", "进入方向和自己的手明显不同吗",
     [("yes", "是"), ("no", "否"), ("uncertain", "不确定")], True),
    ("same_depth", "和自己的手可能同深度吗",
     [("yes", "是"), ("no", "否"), ("uncertain", "不确定")], True),
]

# `human_ownership` IS THE LABEL EVERYTHING DOWNSTREAM SHOULD USE. The frozen
# exit-height rule is what selected these 49 tracks, so scoring a later
# experiment against that same rule would be circular; and 44 tracks in the
# same run had the rule contradicting itself, which is why it gets a question
# of its own rather than being assumed correct. The last two are not training
# targets -- they are there to explain, afterwards, why a depth scalar might
# fail where a trajectory does not.

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
.chip{display:inline-block;padding:1px 8px;margin:2px 5px 0 0;border-radius:3px;
  font-size:11px;background:#222;color:#888;border:1px solid #333}
.chip.has{background:#25303a;color:#cde;border-color:#3a5}
.chip.on{outline:2px solid #ffd33d;color:#ffd33d}
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
const Q = __QUESTIONS__;
const KEY = "census5:" + D.tag;
let lab = {}, cur = 0, qi = 0, hist = [];
try { lab = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch(e) { lab={}; }
const list = document.getElementById("list");
D.tracks.forEach((t, i) => {
  const d = document.createElement("div");
  d.className = "t"; d.id = "t" + i;
  d.innerHTML = '<div class=hd><span class="grp ' + t.grp + '">' + t.grp +
    '</span><span class=dir>' + t.dir + '</span> ' + t.rec +
    ' &nbsp;<span class=num>track ' + t.tid + '  f' + t.first + '-' + t.last +
    '  ' + t.n + ' 帧  p_raw ' + t.p_lo + '-' + t.p_hi + ' (中位 ' + t.p_med +
    ')  错 ' + t.wrong + '/' + t.n + '</span><br><span id=v' + i +
    '></span></div>' +
    '<div class=strip>' + t.crops.map(f =>
      '<figure><img src="' + f.c + '"><figcaption>f' + f.f + '  p=' + f.p +
      '</figcaption></figure>').join('') + '</div>' +
    '<div class=strip style="margin-top:4px">' + t.crops.map(f =>
      '<figure><img src="' + f.w + '"></figure>').join('') + '</div>';
  d.onclick = () => { cur = i; qi = firstUnanswered(i); draw(); };
  list.appendChild(d);
});
function ans(i){ return lab[D.tracks[i].key] || {}; }
function firstUnanswered(i){
  const a = ans(i);
  for (let j = 0; j < Q.length; j++) if (!(Q[j].key in a)) return j;
  return Q.length - 1;
}
function done(i){
  const a = ans(i);
  return Q.every(q => q.optional || (q.key in a));
}
function chips(i){
  const a = ans(i);
  return Q.map((q, j) => {
    const v = a[q.key];
    const on = (i === cur && j === qi);
    const txt = v === undefined ? "—"
      : (v === "" ? "跳过" : (q.opts.find(o => o[0] === v) || ["", v])[1]);
    return '<span class="chip' + (on ? " on" : "") + (v ? " has" : "") + '">' +
           q.label + ': ' + txt + '</span>';
  }).join("");
}
function draw(){
  D.tracks.forEach((t,i)=>{
    document.getElementById("t"+i).className =
      "t" + (done(i) ? " done" : "") + (i===cur ? " cur" : "");
    document.getElementById("v"+i).innerHTML = chips(i);
  });
  const q = Q[qi];
  document.getElementById("keys").innerHTML =
    '<b style="color:#8f8">' + q.label + '</b> &nbsp; ' +
    q.opts.map((o,k) => '<b>' + (k+1) + '</b> ' + o[1]).join(" &nbsp; ") +
    (q.optional ? ' &nbsp; <b>空格</b> 跳过' : '') +
    ' &nbsp;&nbsp; <b>&uarr;&darr;</b> 换轨迹 &nbsp; <b>u</b> undo';
  const nd = D.tracks.filter((_,i)=>done(i)).length;
  document.getElementById("prog").innerHTML =
    "<b>" + D.tag + "</b> &nbsp; " + nd + "/" + D.tracks.length + " 已判完";
  localStorage.setItem(KEY, JSON.stringify(lab));
  const el = document.getElementById("t"+cur);
  if(el) el.scrollIntoView({block:"nearest"});
}
function set(v){
  const t = D.tracks[cur]; if(!t) return;
  hist.push([t.key, JSON.stringify(lab[t.key] || null), qi]);
  lab[t.key] = Object.assign({}, lab[t.key] || {});
  lab[t.key][Q[qi].key] = v;
  if (qi + 1 < Q.length) qi += 1;
  else { cur = Math.min(cur + 1, D.tracks.length - 1); qi = 0; }
  draw();
}
document.onkeydown = ev => {
  const n = +ev.key;
  if(n >= 1 && n <= Q[qi].opts.length){ set(Q[qi].opts[n-1][0]); }
  else if(ev.key===" " && Q[qi].optional){ set(""); }
  else if(ev.key==="ArrowDown"){cur=Math.min(cur+1,D.tracks.length-1);qi=firstUnanswered(cur);draw();}
  else if(ev.key==="ArrowUp"){cur=Math.max(cur-1,0);qi=firstUnanswered(cur);draw();}
  else if(ev.key==="u"){const h=hist.pop(); if(h){
      if(h[1]==="null") delete lab[h[0]]; else lab[h[0]]=JSON.parse(h[1]);
      qi = h[2]; draw(); }}
  else return;
  ev.preventDefault();
};
draw();
function dl(){
  let s = "key,rec,tid,group,ref_direction,frames,wrong,p_median," +
          Q.map(q=>q.key).join(",") + "\\n";
  for(const t of D.tracks){
    const a = lab[t.key]; if(!a) continue;
    s += t.key + "," + t.rec + "," + t.tid + "," + t.grp + ',"' + t.dir +
         '",' + t.n + "," + t.wrong + "," + t.p_med + "," +
         Q.map(q => (q.key in a ? a[q.key] : "")).join(",") + "\\n";
  }
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



def repage(a):
    """Re-render an existing sheet with the current questions.

    The pictures cost hours of video decoding and the schema does not. When
    the questions change -- as they did once the frozen rule turned out to
    need a question of its own -- the payload is lifted out of the old page
    and wrapped in the new one, so nothing is re-read and any answers already
    stored in the browser stay under their own key."""
    import re
    s = open(a.repage, encoding="utf-8").read()
    m = re.search(r"const D = (\{.*?\});\nconst ", s, re.S)
    if not m:
        raise SystemExit(f"{a.repage} 里找不到 payload")
    payload = m.group(1)
    qs = [{"key": k, "label": lab, "optional": bool(opt),
           "opts": [[x, y] for x, y in opts]}
          for k, lab, opts, opt in QUESTIONS]
    out = (SHEET.replace("__PAYLOAD__", payload)
           .replace("__QUESTIONS__", json.dumps(qs, ensure_ascii=False)))
    open(a.out, "w", encoding="utf-8").write(out)
    n = len(json.loads(payload)["tracks"])
    print(f"  {n} 条 -> {a.out} ({os.path.getsize(a.out) / 1e6:.1f} MB)")
    for k, lab, opts, opt in QUESTIONS:
        line = "  ".join(f"{i+1} {b}" for i, (x, b) in enumerate(opts))
        print(f"  {lab:<28} {line}" + ("   空格=跳过" if opt else ""))

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", help="own_dump.csv")
    ap.add_argument("--repage",
                    help="an existing sheet; re-wrap its payload "
                         "with the current questions and exit")
    ap.add_argument("--clips", default="/workspace/e2e_main2.txt")
    ap.add_argument("--shots", type=int, default=4)
    ap.add_argument("--crop_w", type=int, default=170)
    ap.add_argument("--wide_w", type=int, default=170)
    ap.add_argument("--ctx_scale", type=float, default=2.5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if a.repage:
        repage(a)
        return
    if not a.rows:
        ap.error('--rows is required unless --repage')

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
    qs = [{"key": k, "label": lab, "optional": bool(opt),
           "opts": [[a, b] for a, b in opts]}
          for k, lab, opts, opt in QUESTIONS]
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(SHEET.replace("__PAYLOAD__",
                              json.dumps({"tag": "CD", "tracks": out}))
                .replace("__QUESTIONS__", json.dumps(qs, ensure_ascii=False)))
    print(f"\n  {len(out)} 条 -> {a.out} "
          f"({os.path.getsize(a.out) / 1e6:.1f} MB)")
    for k, lab, opts, opt in QUESTIONS:
        line = "  ".join(f"{i+1} {b}" for i, (a, b) in enumerate(opts))
        print(f"  {lab:<28} {line}" + ("   空格=跳过" if opt else ""))


if __name__ == "__main__":
    main()
