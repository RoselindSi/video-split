"""What fraction of the things the ownership classifier is handed are hands?

THE SAME DEFECT MEASURED 2.2% AND 46.9% IN TWO SAMPLING FRAMES. Asked `which
of these mosaics should not be there`, non-hand proposals were 2 of 92; asked
`why is this track's ownership wrong`, they were 23 of 49. Neither number is
the rate. The first sampled frames a person had already called over-blurred,
and a mosaic sitting on a basket of greens does not look like a mistake worth
flagging; the second sampled tracks selected for disagreeing with a rule. Both
are conditioned on an error having been noticed first.

SO THIS SAMPLES NOTHING AT ALL. There are 343 tracks in the twenty-nine
audited windows and all 343 are judged, so there is no sampling frame left to
bias: the number that comes out is the corpus rate, and the only inference
left is from these recordings to other recordings.

THE POPULATION IS WHAT REACHES OWNERSHIP, not what the detector emits. A
detection that never became a track is never classified and never blurs
anything, so its being wrong costs nothing here; the question is what the
classifier was actually asked to judge.

A TRACK IS THE UNIT BECAUSE A HAND DOES NOT STOP BEING ONE. Judging frames
would let one 400-frame false track outweigh forty short true ones, so the
verdict is per track. Both rates are then reported, because they answer
different questions -- how often the classifier is handed rubbish, and how
much of its input is rubbish -- and only the track rate gets an interval,
since frames inside one track are one observation repeated.

ONE BINARY QUESTION. Not `whose hand is it`: that question is what produced
the 46.9%, and answering it requires settling an ownership case first. Here a
person only says whether the green box is on a hand.
"""
from __future__ import annotations

import argparse
import base64
import collections
import csv
import json
import math
import os
import random
import statistics

SHEET = """<meta charset=utf-8><title>hand precision</title><style>
body{font:13px/1.5 system-ui;margin:0;background:#111;color:#ddd}
#bar{position:sticky;top:0;background:#181818;padding:9px 14px;z-index:9;
  border-bottom:1px solid #333;display:flex;gap:14px;align-items:center;
  flex-wrap:wrap}
button{font:13px system-ui;padding:5px 10px;cursor:pointer}
b{color:#ffd33d}
.t{padding:9px 14px;border-bottom:1px solid #262626;display:flex;gap:10px;
  align-items:flex-start}
.t.cur{background:#1d2430;outline:2px solid #4a8}
.t.hand{border-left:5px solid #2a6}
.t.nohand{border-left:5px solid #d33}
.t.unsure{border-left:5px solid #777}
.meta{min-width:195px;color:#9ab;font-size:12px}
.num{color:#8ab4c8;font-family:ui-monospace,monospace;font-size:11px}
.strip{display:flex;gap:4px;flex:1;align-items:flex-start}
.strip figure{margin:0;flex:1}
.strip img{width:100%;border-radius:3px;display:block}
.strip figcaption{font-size:10px;color:#888;text-align:center}
.wide{flex:1.7;opacity:.9}
.key{font-size:11px;color:#999}
</style>
<div id=bar><span id=prog></span>
<span><b>1</b> 是手 &nbsp; <b>2</b> 不是手 &nbsp; <b>3</b> 说不准
 &nbsp; <b>&uarr;&darr;</b> 换 &nbsp; <b>u</b> undo</span>
<button onclick="dl()">download CSV</button>
<span class=key>全部轨迹，没有按任何错误筛选。绿框=检测框，最右是整帧。
问的只有一件事：框里是不是一只手。不问是谁的。</span></div>
<div id=list></div>
<script>
const D = __PAYLOAD__;
const KEY = "handprec:" + D.tag;
const V = ["hand", "nohand", "unsure"];
const CN = {hand: "是手", nohand: "不是手", unsure: "说不准"};
const CL = {hand: "#5c5", nohand: "#d55", unsure: "#999"};
let lab = {}, cur = 0, hist = [];
try { lab = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch(e) { lab={}; }
const list = document.getElementById("list");
D.tracks.forEach((t, i) => {
  const d = document.createElement("div");
  d.className = "t"; d.id = "t" + i;
  d.innerHTML = '<div class=meta>' + t.rec + ' <span id=v' + i + '></span>' +
    '<br><span class=num>track ' + t.tid + '  f' + t.first + '-' + t.last +
    '<br>' + t.n + ' 帧   conf ' + t.conf + '   宽 ' + t.w + '%</span></div>' +
    '<div class=strip>' + t.shots.map(f =>
      '<figure><img src="' + f.img + '"><figcaption>f' + f.f +
      '</figcaption></figure>').join('') +
    '<figure class=wide><img src="' + t.wide +
    '"><figcaption>整帧</figcaption></figure></div>';
  d.onclick = () => { cur = i; draw(); };
  list.appendChild(d);
});
function draw(){
  D.tracks.forEach((t,i)=>{
    const v = lab[t.key];
    document.getElementById("t"+i).className =
      "t " + (v || "") + (i===cur ? " cur" : "");
    document.getElementById("v"+i).innerHTML = v ?
      '<b style="color:' + CL[v] + '">' + CN[v] + '</b>' : '';
  });
  const n = D.tracks.filter(t => lab[t.key]).length;
  document.getElementById("prog").innerHTML =
    "<b>" + D.tag + "</b> &nbsp; " + n + "/" + D.tracks.length + " 已判";
  localStorage.setItem(KEY, JSON.stringify(lab));
  const el = document.getElementById("t"+cur);
  if(el) el.scrollIntoView({block:"nearest"});
}
document.onkeydown = ev => {
  if(ev.key>="1" && ev.key<="3"){
    const t = D.tracks[cur]; if(!t) return;
    hist.push([t.key, lab[t.key]]);
    lab[t.key] = V[+ev.key-1];
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
  let s = "key,rec,tid,frames,det_conf,w_frac,verdict\\n";
  for(const t of D.tracks) if(lab[t.key])
    s += t.key + "," + t.rec + "," + t.tid + "," + t.n + "," + t.conf + "," +
         t.w + "," + lab[t.key] + "\\n";
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([s],{type:"text/csv"}));
  a.download = "hand_precision_" + D.tag + ".csv"; a.click();
}
</script>
"""


def wilson(k, n, z=1.96):
    if not n:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / d, (c + m) / d)


def by_track(rows):
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["rec"], r["tid"])].append(r)
    for v in by.values():
        v.sort(key=lambda r: int(r["frame"]))
    return by


def report(a):
    rows = list(csv.DictReader(open(a.rows, encoding="utf-8-sig")))
    by = by_track(rows)
    lab = {}
    for path in a.labels:
        for r in csv.DictReader(open(path, encoding="utf-8-sig")):
            lab[r["key"]] = r["verdict"]
    judged = {k: v for k, v in by.items() if f"{k[0]}:{k[1]}" in lab}
    if not judged:
        raise SystemExit("no labelled tracks")
    ver = {k: lab[f"{k[0]}:{k[1]}"] for k in judged}

    n_t = len(judged)
    bad = [k for k in judged if ver[k] == "nohand"]
    uns = [k for k in judged if ver[k] == "unsure"]
    lo, hi = wilson(len(bad), n_t)
    print("\n  === 手部检测器精确度（全体轨迹，未按任何错误筛选）===")
    print(f"  判了 {n_t}/{len(by)} 条轨迹")
    print(f"  轨迹级   非手 {len(bad)}/{n_t} = {len(bad) / n_t:.1%} "
          f"[{lo:.3f}, {hi:.3f}]   说不准 {len(uns)}")
    f_all = sum(len(v) for v in judged.values())
    f_bad = sum(len(judged[k]) for k in bad)
    print(f"  帧级     非手 {f_bad}/{f_all} = {f_bad / f_all:.1%}  "
          f"（不给区间：一条轨迹里的帧是同一个观测重复，不是独立样本）")
    print("  两个数回答不同问题：轨迹级=分类器多久被塞一次垃圾，"
          "帧级=它的输入里有多少是垃圾。")

    # MACRO IS THE METRIC. Pooling lets one busy recording set the corpus
    # rate; the spread across recordings is what says whether the number
    # travels at all -- pooled-vs-macro is the same trap the boundary line
    # already fell into once.
    per = collections.defaultdict(lambda: [0, 0])
    for k in judged:
        per[k[0]][1] += 1
        per[k[0]][0] += ver[k] == "nohand"
    rates = [b / n for b, n in per.values() if n >= a.min_rec]
    if len(rates) > 1:
        print(f"\n  录像级（>= {a.min_rec} 条的 {len(rates)} 段）"
              f"  macro {statistics.mean(rates):.1%}"
              f"   中位 {statistics.median(rates):.1%}"
              f"   全距 {min(rates):.0%}-{max(rates):.0%}")
        for rec in sorted(per, key=lambda r: -per[r][0] / per[r][1]):
            b, n = per[rec]
            if n >= a.min_rec:
                print(f"    {rec}  {b:>3}/{n:<3} = {b / n:>6.1%}")

    print("\n  按轨迹长度（长度是这里唯一免费的先验）")
    for name, a_, b_ in (("1-3 帧", 1, 3), ("4-20 帧", 4, 20),
                         ("21-100 帧", 21, 100), (">100 帧", 101, 10 ** 9)):
        s = [k for k in judged if a_ <= len(judged[k]) <= b_]
        if s:
            n = sum(1 for k in s if ver[k] == "nohand")
            fr = sum(len(judged[k]) for k in s)
            fb = sum(len(judged[k]) for k in s if ver[k] == "nohand")
            print(f"    {name:<10} 轨迹 {n:>3}/{len(s):<4} = {n / len(s):>6.1%}"
                  f"    帧 {fb:>5}/{fr:<5} = {fb / fr:>6.1%}")

    print("\n  按检测置信度中位（能不能靠阈值切掉）")
    for a_, b_ in ((0.0, 0.65), (0.65, 0.75), (0.75, 0.85), (0.85, 1.01)):
        s = [k for k in judged
             if a_ <= statistics.median(
                 float(r["side_conf"]) for r in judged[k]) < b_]
        if s:
            n = sum(1 for k in s if ver[k] == "nohand")
            print(f"    {a_:.2f}-{b_:.2f}   {n:>3}/{len(s):<4} = "
                  f"{n / len(s):>6.1%}")

    if a.census:
        # THE POINT OF THE OVERLAP. If the same tracks get the same verdict
        # here, the 2.2% / 46.9% gap was the sampling frame and not the
        # question's wording -- which is exactly what is being claimed.
        cen = {r["key"]: r["human_ownership"]
               for r in csv.DictReader(open(a.census, encoding="utf-8-sig"))
               if r.get("human_ownership")}
        both = [(cen[f"{k[0]}:{k[1]}"], ver[k]) for k in judged
                if f"{k[0]}:{k[1]}" in cen]
        if both:
            agree = sum(1 for c, h in both
                        if (c == "not_a_hand") == (h == "nohand"))
            print(f"\n  与 C+D census 重叠 {len(both)} 条   "
                  f"「是不是手」一致 {agree}/{len(both)}")
            for (c, h), n in sorted(collections.Counter(both).items(),
                                    key=lambda x: -x[1]):
                print(f"    census {c:<11} -> {h:<7} {n}")


def build(a):
    import cv2
    from src.rig.calibration import RigCalibration
    from src.rig.geometry import VirtualWideCamera
    from src.rig.render_wide import render
    from src.rig.seam_fix import ClipReader, Prefetch

    rows = list(csv.DictReader(open(a.rows, encoding="utf-8-sig")))
    by = by_track(rows)
    print(f"  {len(rows)} 帧-手   {len(by)} 条轨迹   "
          f"{len({k[0] for k in by})} 段录像 —— 全判，不抽样")

    clips = {}
    for line in open(a.clips):
        line = line.strip()
        if not line:
            continue
        p = line.rsplit(":", 2)
        clips[os.path.basename(p[0].rstrip("/")).replace("databag-26_", "R")] \
            = (p[0], int(p[1]))

    per_rec = collections.defaultdict(list)
    for k in by:
        per_rec[k[0]].append(k)

    def b64(img, w, q=84):
        h = int(round(img.shape[0] * w / max(1, img.shape[1])))
        ok, buf = cv2.imencode(".jpg", cv2.resize(img, (w, max(1, h))),
                               [int(cv2.IMWRITE_JPEG_QUALITY), q])
        return ("data:image/jpeg;base64," +
                base64.b64encode(buf).decode()) if ok else ""

    out = []
    for rec in sorted(per_rec):
        if rec not in clips:
            print(f"  {rec}: 没有 clip，跳过")
            continue
        databag, _ = clips[rec]
        rig = RigCalibration(os.path.join(databag, "calibration.yaml"))
        vcam = VirtualWideCamera.from_rig(rig)
        vids = {k: os.path.join(databag, f"{k}.mp4")
                for k in ("cam12", "cam34", "cam56")}
        picks, want = {}, set()
        for k in per_rec[rec]:
            v = by[k]
            step = max(1, len(v) // a.shots)
            p = list(dict.fromkeys(int(r["frame"])
                                   for r in v[::step]))[:a.shots]
            picks[k] = p or [int(v[0]["frame"])]
            want.update(picks[k])
        lo, hi = min(want), max(want)
        rd = Prefetch(ClipReader(rig, vids, lo), skip=0)
        mc, imgs = {}, {}
        print(f"  {rec}: {len(per_rec[rec])} 条, 读 {lo}-{hi}", flush=True)
        for i in range(hi - lo + 1):
            src = rd.next()
            if not src:
                break
            f = lo + i
            if f not in want:
                continue
            try:
                rgb, _, _, _ = render(rig, vcam, src, 0.6, map_cache=mc)
            except TypeError:
                rgb, _, _, _ = render(rig, vcam, src, 0.6)
            imgs[f] = rgb
        rd.close()

        for k in per_rec[rec]:
            v = by[k]
            bmap = {int(r["frame"]): r for r in v}
            mid = picks[k][len(picks[k]) // 2]
            shots, wide = [], ""
            for f in picks[k]:
                img = imgs.get(f)
                if img is None:
                    continue
                H, W = img.shape[:2]
                r0 = bmap[f]
                x0, y0 = int(r0["x0"]), int(r0["y0"])
                x1, y1 = int(r0["x1"]), int(r0["y1"])
                cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
                half = max(12, int(max(x1 - x0, y1 - y0) * a.ctx_scale / 2))
                cx0, cy0 = max(0, cx - half), max(0, cy - half)
                cx1, cy1 = min(W, cx + half), min(H, cy + half)
                crop = img[cy0:cy1, cx0:cx1].copy()
                cv2.rectangle(crop, (x0 - cx0, y0 - cy0),
                              (x1 - cx0, y1 - cy0), (60, 255, 60), 2)
                shots.append({"f": f, "img": b64(crop, a.crop_w)})
                if not wide or f == mid:
                    w = img.copy()
                    cv2.rectangle(w, (x0, y0), (x1, y1), (60, 255, 60), 3)
                    wide = b64(w, a.wide_w, 80)
            if not shots:
                continue
            cs = sorted(float(r["side_conf"]) for r in v)
            ws = sorted((int(r["x1"]) - int(r["x0"])) / 1600.0 for r in v)
            out.append({"key": f"{k[0]}:{k[1]}", "rec": k[0], "tid": k[1],
                        "first": int(v[0]["frame"]),
                        "last": int(v[-1]["frame"]), "n": len(v),
                        "conf": round(cs[len(cs) // 2], 2),
                        "w": round(100 * ws[len(ws) // 2], 1),
                        "shots": shots, "wide": wide})

    # ORDER CARRIES NO SIGNAL. Grouped by recording, the judge would learn a
    # scene and answer from it rather than from the box; sorted by length or
    # by confidence, the judge would learn the very covariate being measured.
    random.Random(a.seed).shuffle(out)

    stem, ext = os.path.splitext(a.out)
    n_pg = max(1, (len(out) + a.page - 1) // a.page)
    for i in range(n_pg):
        part = out[i * a.page:(i + 1) * a.page]
        path = a.out if n_pg == 1 else f"{stem}_{i + 1}{ext}"
        tag = "all" if n_pg == 1 else f"p{i + 1}"
        with open(path, "w", encoding="utf-8") as f:
            f.write(SHEET.replace("__PAYLOAD__",
                                  json.dumps({"tag": tag, "tracks": part})))
        print(f"  {len(part):>3} 条 -> {path} "
              f"({os.path.getsize(path) / 1e6:.1f} MB)")
    print("  1 是手   2 不是手   3 说不准")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", required=True, help="own_dump csv")
    ap.add_argument("--clips", default="/workspace/e2e_main2.txt")
    ap.add_argument("--shots", type=int, default=4)
    ap.add_argument("--crop_w", type=int, default=150)
    ap.add_argument("--wide_w", type=int, default=220)
    ap.add_argument("--ctx_scale", type=float, default=2.5)
    ap.add_argument("--page", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--labels", nargs="*", help="report mode: labelled csvs")
    ap.add_argument("--census", help="own_census.csv, for the overlap check")
    ap.add_argument("--min_rec", type=int, default=6)
    ap.add_argument("--out")
    a = ap.parse_args()

    if a.labels:
        report(a)
    elif a.out:
        build(a)
    else:
        ap.error("--out (build) or --labels (report)")


if __name__ == "__main__":
    main()
