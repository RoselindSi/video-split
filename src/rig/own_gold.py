"""Whose hand is it, decided by a person, for every hand in the corpus.

THE REFERENCE WAS THE PROBLEM. Every ownership number this project has
reported was scored against `exit_y >= 0.55H`, a rule that reproduced 1028
stored labels without an exception and was therefore treated as a second
opinion good enough to measure against. On the 26 real hands in the C+D set a
person judged the rule wrong on 17 of them, and those 17 carry 412 of the 737
wrong frames there -- more than the classifier's own confirmed mistakes. A
metric whose largest error term is its own reference is not measuring the
model.

SO THE RULE STOPS BEING THE REFERENCE. There are 274 tracks that a person has
confirmed are hands; that is small enough to judge outright, and once it is
judged the rule becomes one more thing being scored rather than the thing
scoring. The same argument that retired the stored boundary labels applies
here, and for once the corpus is small enough to act on it immediately.

BLIND, AND NOTHING NUMERIC IS SHOWN. Not the classifier's probability, not the
rule's verdict, not the box width -- apparent size is a distance proxy and
distance is most of what separates the wearer's hands from everyone else's, so
showing it would feed the answer back into the question. Recording, frame
range and track length are all that appear.

FOUR ANSWERS, AND TWO OF THEM ARE ESCAPES. `ambiguous` is for a hand whose
owner genuinely cannot be told, and `mixed_identity` for a track that is more
than one hand -- a tracker handoff produces exactly that, and folding one into
`owner` or `other` would charge the ownership classifier for an association
failure. Both are reported separately and neither enters an accuracy
denominator.

THE FOREARM IS THE EVIDENCE, so the crop is wide enough to hold it and two
full frames come with each track. The wearer's arm leaves through the bottom
of the frame; a colleague's does not. That is the same cue the rule uses, and
a person can apply it where a threshold on one number cannot.
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

OPTIONS = [("owner", "自己的手"), ("other", "别人的手"),
           ("ambiguous", "说不准是谁的"), ("mixed_identity", "这条轨迹不止一只手")]

SHEET = """<meta charset=utf-8><title>ownership gold</title><style>
body{font:13px/1.5 system-ui;margin:0;background:#111;color:#ddd}
#bar{position:sticky;top:0;background:#181818;padding:9px 14px;z-index:9;
  border-bottom:1px solid #333;display:flex;gap:14px;align-items:center;
  flex-wrap:wrap}
button{font:13px system-ui;padding:5px 10px;cursor:pointer}
b{color:#ffd33d}
.t{padding:10px 14px;border-bottom:1px solid #262626;display:flex;gap:10px;
  align-items:flex-start}
.t.cur{background:#1d2430;outline:2px solid #4a8}
.t.owner{border-left:5px solid #2a6}
.t.other{border-left:5px solid #c62}
.t.ambiguous{border-left:5px solid #777}
.t.mixed_identity{border-left:5px solid #a3c}
.meta{min-width:150px;color:#9ab;font-size:12px}
.num{color:#8ab4c8;font-family:ui-monospace,monospace;font-size:11px}
.strip{display:flex;gap:4px;flex:1;align-items:flex-start}
.strip figure{margin:0;flex:1}
.strip figure.wide{flex:2.1}
.strip img{width:100%;border-radius:3px;display:block}
.strip figcaption{font-size:10px;color:#888;text-align:center}
.key{font-size:11px;color:#999}
</style>
<div id=bar><span id=prog></span><span id=keys></span>
<button onclick="dl()">download CSV</button>
<span class=key>绿框=同一条轨迹在不同时刻，最右两张是整帧。
看前臂往哪走：自己的手从画面下沿出去。</span></div>
<div id=list></div>
<script>
const D = __PAYLOAD__;
const O = __OPTIONS__;
const KEY = "owngold:" + D.tag;
let lab = {}, cur = 0, hist = [];
try { lab = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch(e) { lab={}; }
const list = document.getElementById("list");
D.tracks.forEach((t, i) => {
  const d = document.createElement("div");
  d.className = "t"; d.id = "t" + i;
  d.innerHTML = '<div class=meta>' + t.rec + ' <span id=v' + i + '></span>' +
    '<br><span class=num>f' + t.first + '-' + t.last + '   ' + t.n +
    ' 帧</span></div><div class=strip>' + t.shots.map(f =>
      '<figure><img src="' + f.img + '"><figcaption>f' + f.f +
      '</figcaption></figure>').join('') + t.wide.map(w =>
      '<figure class=wide><img src="' + w + '"></figure>').join('') +
    '</div>';
  d.onclick = () => { cur = i; draw(); };
  list.appendChild(d);
});
function draw(){
  D.tracks.forEach((t,i)=>{
    const v = lab[t.key];
    document.getElementById("t"+i).className =
      "t " + (v || "") + (i===cur ? " cur" : "");
    document.getElementById("v"+i).innerHTML = v ?
      '<b>' + (O.find(o => o[0] === v) || ["",v])[1] + '</b>' : '';
  });
  document.getElementById("keys").innerHTML =
    O.map((o,k) => '<b>' + (k+1) + '</b> ' + o[1]).join(" &nbsp; ") +
    ' &nbsp;&nbsp; <b>&uarr;&darr;</b> 换 &nbsp; <b>u</b> undo';
  const n = D.tracks.filter(t => lab[t.key]).length;
  document.getElementById("prog").innerHTML =
    "<b>" + D.tag + "</b> &nbsp; " + n + "/" + D.tracks.length + " 已判";
  localStorage.setItem(KEY, JSON.stringify(lab));
  const el = document.getElementById("t"+cur);
  if(el) el.scrollIntoView({block:"nearest"});
}
document.onkeydown = ev => {
  const n = +ev.key;
  if(n >= 1 && n <= O.length){
    const t = D.tracks[cur]; if(!t) return;
    hist.push([t.key, lab[t.key]]);
    lab[t.key] = O[n-1][0];
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
  let s = "key,rec,tid,frames,human_ownership\\n";
  for(const t of D.tracks) if(lab[t.key])
    s += t.key + "," + t.rec + "," + t.tid + "," + t.n + "," + lab[t.key] + "\\n";
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([s],{type:"text/csv"}));
  a.download = "own_gold_" + D.tag + ".csv"; a.click();
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


def load(a):
    rows = list(csv.DictReader(open(a.rows, encoding="utf-8-sig")))
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["rec"], r["tid"])].append(r)
    for v in by.values():
        v.sort(key=lambda r: int(r["frame"]))
    if a.hands:
        keep = set()
        for f in a.hands:
            for r in csv.DictReader(open(f, encoding="utf-8-sig")):
                if r["verdict"] == "hand":
                    keep.add((r["rec"], r["tid"]))
        by = {k: v for k, v in by.items() if k in keep}
    return by


def report(a):
    by = load(a)
    gold = {}
    for p in a.labels:
        for r in csv.DictReader(open(p, encoding="utf-8-sig")):
            gold[(r["rec"], r["tid"])] = r["human_ownership"]
    J = {k: v for k, v in by.items() if k in gold}
    if not J:
        raise SystemExit("no gold")
    esc = collections.Counter(gold[k] for k in J
                              if gold[k] in ("ambiguous", "mixed_identity"))
    D = {k: v for k, v in J.items() if gold[k] in ("owner", "other")}

    def owner(k):
        return gold[k] == "owner"

    def fin(r):
        return bool(int(r["final_owner_post_cap"]))

    def rule(r):
        return bool(int(r["reference_owner"]))

    print(f"\n  === 人工 ownership gold ===")
    print(f"  判了 {len(J)}/{len(by)} 条   其中可用于打分 {len(D)} 条"
          f"   说不准 {esc['ambiguous']}   一条轨迹不止一只手 "
          f"{esc['mixed_identity']}")

    print(f"\n  {'':<20}{'轨迹':>7}{'帧':>9}")
    for name, want in (("human OWNER", True), ("human OTHER", False)):
        s = [k for k in D if owner(k) == want]
        print(f"  {name:<20}{len(s):>7}{sum(len(D[k]) for k in s):>9}")
    fc = ffo = ffe = 0
    for k, v in D.items():
        for r in v:
            if fin(r) == owner(k):
                fc += 1
            elif fin(r):
                ffo += 1        # called the wearer's when it is not: leaked
            else:
                ffe += 1        # called foreign when it is not: blurred
    n_f = fc + ffo + ffe
    tc = tfo = tfe = 0
    for k, v in D.items():
        maj = sum(1 for r in v if fin(r)) * 2 >= len(v)
        if maj == owner(k):
            tc += 1
        elif maj:
            tfo += 1
        else:
            tfe += 1
    print(f"  {'final correct':<20}{tc:>7}{fc:>9}")
    print(f"  {'final false-owner':<20}{tfo:>7}{ffo:>9}   ← 别人的手被当成自己"
          f"的，没糊")
    print(f"  {'final false-other':<20}{tfe:>7}{ffe:>9}   ← 自己的手被当成别人"
          f"的，糊了")
    rc = rw = 0
    trc = trw = 0
    for k, v in D.items():
        rc += sum(1 for r in v if rule(r) == owner(k))
        rw += sum(1 for r in v if rule(r) != owner(k))
        maj = sum(1 for r in v if rule(r)) * 2 >= len(v)
        if maj == owner(k):
            trc += 1
        else:
            trw += 1
    print(f"  {'rule correct':<20}{trc:>7}{rc:>9}")
    print(f"  {'rule wrong':<20}{trw:>7}{rw:>9}")

    print(f"\n  === 逐帧 ===")
    lo, hi = wilson(fc, n_f)
    print(f"  final 帧级准确率      {fc}/{n_f} = {fc/n_f:.1%} "
          f"（区间不给：帧非独立）")
    oth = [k for k in D if not owner(k)]
    own = [k for k in D if owner(k)]
    of = sum(len(D[k]) for k in oth)
    ob = sum(1 for k in oth for r in D[k] if not fin(r))
    wf = sum(len(D[k]) for k in own)
    wb = sum(1 for k in own for r in D[k] if not fin(r))
    print(f"  别人的手被糊的帧比例   {ob}/{of} = {ob/max(1,of):.1%}"
          f"   ← other-hand frame recall，隐私要的就是这个")
    print(f"  自己的手被糊的帧比例   {wb}/{wf} = {wb/max(1,wf):.1%}"
          f"   ← owner false-blur rate")
    print(f"  rule 帧级准确率       {rc}/{rc+rw} = {rc/max(1,rc+rw):.1%}")

    print(f"\n  === 逐轨迹 ===")
    lo, hi = wilson(tc, len(D))
    print(f"  final 轨迹级准确率     {tc}/{len(D)} = {tc/len(D):.1%} "
          f"[{lo:.3f}, {hi:.3f}]")
    lo, hi = wilson(trc, len(D))
    print(f"  rule  轨迹级准确率     {trc}/{len(D)} = {trc/len(D):.1%} "
          f"[{lo:.3f}, {hi:.3f}]")
    whole = sum(1 for k, v in D.items()
                if all(fin(r) != owner(k) for r in v))
    print(f"  整条都判错的轨迹       {whole}/{len(D)} = {whole/len(D):.1%}"
          f"   ← 温度计不会抖，这些是真正的 hard set")

    # OTHER IS THE POSITIVE CLASS. Missing a colleague's hand is the privacy
    # failure; blurring the wearer's is a usability cost. They are not
    # interchangeable, so one F1 is reported and it is the foreign one.
    tp = sum(1 for k in oth for r in D[k] if not fin(r))
    fp = sum(1 for k in own for r in D[k] if not fin(r))
    fn = sum(1 for k in oth for r in D[k] if fin(r))
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    print(f"\n  以『别人的手』为正类（帧级）  精确 {prec:.3f}  召回 {rec:.3f}  "
          f"F1 {2*prec*rec/max(1e-9, prec+rec):.3f}")

    print(f"\n  === 录像级 macro（>= {a.min_rec} 条）===")
    per = collections.defaultdict(lambda: [0, 0, 0])
    for k in D:
        maj = sum(1 for r in D[k] if fin(r)) * 2 >= len(D[k])
        rmaj = sum(1 for r in D[k] if rule(r)) * 2 >= len(D[k])
        p = per[k[0]]
        p[0] += maj == owner(k)
        p[1] += rmaj == owner(k)
        p[2] += 1
    fs = [a_ / n for a_, b_, n in per.values() if n >= a.min_rec]
    rs = [b_ / n for a_, b_, n in per.values() if n >= a.min_rec]
    if fs:
        print(f"  final macro {statistics.mean(fs):.1%}  中位 "
              f"{statistics.median(fs):.1%}  全距 {min(fs):.0%}-{max(fs):.0%}"
              f"   ({len(fs)} 段)")
        print(f"  rule  macro {statistics.mean(rs):.1%}  中位 "
              f"{statistics.median(rs):.1%}  全距 {min(rs):.0%}-{max(rs):.0%}")
        for rec_ in sorted(per, key=lambda r: per[r][0] / per[r][2]):
            a_, b_, n = per[rec_]
            if n >= a.min_rec:
                print(f"    {rec_}  final {a_}/{n} = {a_/n:>6.1%}"
                      f"   rule {b_}/{n} = {b_/n:>6.1%}")

    print(f"\n  === 规则错在哪一边（这决定旧指标是被抬高还是压低）===")
    for name, want in (("人判=自己", True), ("人判=别人", False)):
        s = [k for k in D if owner(k) == want]
        w = sum(1 for k in s
                if (sum(1 for r in D[k] if rule(r)) * 2 >= len(D[k])) != want)
        print(f"    {name}  规则判错 {w}/{len(s)} = {w/max(1,len(s)):.1%}")

    print(f"\n  === 分类器和规则的四格（轨迹级）===")
    g = collections.Counter()
    for k in D:
        fm = sum(1 for r in D[k] if fin(r)) * 2 >= len(D[k])
        rm = sum(1 for r in D[k] if rule(r)) * 2 >= len(D[k])
        g[(fm == owner(k), rm == owner(k))] += 1
    print(f"    都对 {g[(True,True)]}   只有模型对 {g[(True,False)]}   "
          f"只有规则对 {g[(False,True)]}   都错 {g[(False,False)]}")
    print("    『只有模型对』那一格，在旧口径里全部被记成模型的错。")


FPS = 30.0


def arms(a):
    """The privacy metrics that matter, for each candidate arm.

    FRAME RECALL IS THE WRONG HEADLINE. A frame is 33 ms and cannot be seen;
    twenty-eight consecutive frames are nine tenths of a second and are enough
    to recognise a person. Twenty-eight scattered single-frame misses and one
    0.93 s hole are the same number and not the same risk, so what gets
    reported is the exposure event: a contiguous run of frames in which a hand
    a person judged foreign was not covered. Frame recall stays, in small
    print, because it is comparable with everything measured before.

    AND THE MISS THAT MATTERS MOST IS THE ONE THAT NEVER GETS COVERED AT ALL.
    A track whose majority verdict is wrong is a person the system does not
    know is there; a hole inside a track is a person it covers imperfectly.
    Those are different failures with different repairs, so they are counted
    apart and the whole-track miss goes first.

    WHAT THIS DOES NOT COVER, and the number must not be read as if it did:
    a foreign hand that never became a track is invisible to every arm here,
    the eight tracks that produced no images were never judged, and all of it
    is twenty-nine windows of one corpus."""
    by = load(a)
    gold = {}
    for p in a.labels:
        for r in csv.DictReader(open(p, encoding="utf-8-sig")):
            gold[(r["rec"], r["tid"])] = r["human_ownership"]
    D = {k: v for k, v in by.items()
         if gold.get(k) in ("owner", "other")}

    def P0(r):
        return float(r["p_owner_raw"]) >= 0.5

    def P1(r):
        return bool(int(r["ownhold_pre_cap"]))

    def P2(r):
        return bool(int(r["final_owner_post_cap"]))

    def R(r):
        return bool(int(r["reference_owner"]))

    def majority(fn):
        # THE TIE RULE IS FIXED HERE AND NOT AT SCORING TIME. An even split
        # resolves to `other`, because the two errors are not equal: a tie
        # broken towards the wearer leaks, a tie broken towards a colleague
        # blurs. Same asymmetry the Schmitt trigger is built on.
        cache = {}
        for k, v in D.items():
            cache[k] = sum(1 for r in v if fn(r)) * 2 > len(v)
        return lambda r: cache[(r["rec"], r["tid"])]

    def runs_of(v, fn, want_other):
        """Contiguous frames where a foreign hand was left uncovered."""
        out, cur = [], 0
        for r in v:
            if fn(r):                      # called the wearer's -> not blurred
                cur += 1
            elif cur:
                out.append(cur)
                cur = 0
        if cur:
            out.append(cur)
        return out

    print(f"\n  {'':<26}{'整轨漏':>7}{'暴露事件':>9}{'中位s':>7}{'最长s':>7}"
          f"{'>=0.4s':>7}{'录像':>6}{'误糊帧':>7}{'帧召回':>8}{'F1':>7}")
    for name, fn in (("P0 分类器 raw", P0),
                     ("P1 cap 前", P1),
                     ("P2 部署", P2),
                     ("R  冻结出口规则", R),
                     ("P0 + 整轨多数票", majority(P0)),
                     ("P1 + 整轨多数票", majority(P1)),
                     ("P2 + 整轨多数票", majority(P2))):
        miss = ev = 0
        lens, recs = [], set()
        for k, v in D.items():
            if gold[k] != "other":
                continue
            lab = [fn(r) for r in v]
            if sum(lab) * 2 >= len(lab):
                miss += 1
            rr = runs_of(v, fn, True)
            ev += len(rr)
            lens += rr
            if rr:
                recs.add(k[0])
        blur = sum(1 for k, v in D.items() if gold[k] == "owner"
                   for r in v if not fn(r))
        of = sum(len(v) for k, v in D.items() if gold[k] == "other")
        cov = of - sum(lens)
        tp, fp = cov, blur
        rec_ = tp / max(1, of)
        prec = tp / max(1, tp + fp)
        f1 = 2 * prec * rec_ / max(1e-9, prec + rec_)
        lens.sort()
        med = lens[len(lens) // 2] / FPS if lens else 0.0
        mx = lens[-1] / FPS if lens else 0.0
        big = sum(1 for x in lens if x / FPS >= 0.4)
        print(f"  {name:<26}{miss:>7}{ev:>9}{med:>7.2f}{mx:>7.2f}"
              f"{big:>7}{len(recs):>6}{blur:>7}{rec_:>8.3f}{f1:>7.3f}")
    n_oth = sum(1 for k in D if gold[k] == "other")
    print(f"\n  分母：别人的手 {n_oth} 条 / "
          f"{sum(len(v) for k, v in D.items() if gold[k] == 'other')} 帧；"
          f"自己的手 {sum(1 for k in D if gold[k] == 'owner')} 条 / "
          f"{sum(len(v) for k, v in D.items() if gold[k] == 'owner')} 帧")
    print("  『整轨漏』= 多数票判成自己的手的别人的手轨迹数（平局归 other）。")
    print("  不覆盖：从未形成轨迹的别人的手、8 条没渲染出图的轨迹、"
          "这 29 段以外的录像。")


def build(a):
    import cv2
    from src.rig.calibration import RigCalibration
    from src.rig.geometry import VirtualWideCamera
    from src.rig.render_wide import render
    from src.rig.seam_fix import ClipReader, Prefetch

    by = load(a)
    print(f"  {len(by)} 条确认是手的轨迹   "
          f"{sum(len(v) for v in by.values())} 帧-手 —— 全判")

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
            shots, wides = [], []
            # The first and last sampled moment get a full frame. A handoff
            # shows itself between them, and the arm that leaves the bottom
            # edge is often only visible at full width.
            ends = {picks[k][0], picks[k][-1]}
            for f in picks[k]:
                img = imgs.get(f)
                if img is None:
                    continue
                H, W = img.shape[:2]
                r0 = bmap[f]
                x0, y0 = int(r0["x0"]), int(r0["y0"])
                x1, y1 = int(r0["x1"]), int(r0["y1"])
                cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
                half = max(20, int(max(x1 - x0, y1 - y0) * a.ctx_scale / 2))
                cx0, cy0 = max(0, cx - half), max(0, cy - half)
                cx1, cy1 = min(W, cx + half), min(H, cy + half)
                crop = img[cy0:cy1, cx0:cx1].copy()
                cv2.rectangle(crop, (x0 - cx0, y0 - cy0),
                              (x1 - cx0, y1 - cy0), (60, 255, 60), 2)
                shots.append({"f": f, "img": b64(crop, a.crop_w)})
                if f in ends:
                    w = img.copy()
                    cv2.rectangle(w, (x0, y0), (x1, y1), (60, 255, 60), 3)
                    wides.append(b64(w, a.wide_w, 80))
            if not shots:
                continue
            out.append({"key": f"{k[0]}:{k[1]}", "rec": k[0], "tid": k[1],
                        "first": int(v[0]["frame"]),
                        "last": int(v[-1]["frame"]), "n": len(v),
                        "shots": shots, "wide": wides})

    random.Random(a.seed).shuffle(out)
    stem, ext = os.path.splitext(a.out)
    n_pg = max(1, (len(out) + a.page - 1) // a.page)
    for i in range(n_pg):
        part = out[i * a.page:(i + 1) * a.page]
        path = a.out if n_pg == 1 else f"{stem}_{i + 1}{ext}"
        tag = "all" if n_pg == 1 else f"p{i + 1}"
        with open(path, "w", encoding="utf-8") as f:
            f.write(SHEET.replace("__PAYLOAD__",
                                  json.dumps({"tag": tag, "tracks": part}))
                    .replace("__OPTIONS__",
                             json.dumps([[k, v] for k, v in OPTIONS],
                                        ensure_ascii=False)))
        print(f"  {len(part):>3} 条 -> {path} "
              f"({os.path.getsize(path) / 1e6:.1f} MB)")
    for i, (k, v) in enumerate(OPTIONS):
        print(f"  {i + 1} {v}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", required=True, help="own_dump csv")
    ap.add_argument("--hands", nargs="*", help="hand_precision label csvs")
    ap.add_argument("--clips", default="/workspace/e2e_main2.txt")
    ap.add_argument("--shots", type=int, default=4)
    ap.add_argument("--crop_w", type=int, default=185)
    ap.add_argument("--wide_w", type=int, default=330)
    ap.add_argument("--ctx_scale", type=float, default=3.5)
    ap.add_argument("--page", type=int, default=95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min_rec", type=int, default=6)
    ap.add_argument("--labels", nargs="*", help="report mode: gold csvs")
    ap.add_argument("--arms", action="store_true",
                    help="compare the candidate arms on the privacy metrics")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.labels:
        arms(a) if a.arms else report(a)
    elif a.out:
        build(a)
    else:
        ap.error("--out (build) or --labels (report)")


if __name__ == "__main__":
    main()
