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

# ONE PASS, FIVE ANSWERS. The first batch needed two sheets -- is it a hand,
# then whose is it -- and the census showed the two framings agree 48 times in
# 49 on the first question, judged from the same crops. Folding it in halves
# the labelling without touching the sampling frame, which is what made the
# measurement unbiased in the first place.
#
# `not_a_hand` sits at 3 because that is the key a person already reached for:
# asked to judge only ownership, the first batch's non-hands were entered as 3.
OPTIONS = [("owner", "自己的手"), ("other", "别人的手"),
           ("not_a_hand", "根本不是手"), ("ambiguous", "说不准是谁的"),
           ("mixed_identity", "这条轨迹不止一只手")]

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
先看框里是不是一只手；是的话，再判它是谁的。</span></div>
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
    NH = {k: v for k, v in J.items() if gold[k] == "not_a_hand"}
    if NH:
        # THE UPSTREAM NUMBER, MEASURED IN THE SAME PASS. A detector proposing
        # a bag of peppers is not an ownership failure, so it is counted here
        # and then removed from every denominator below.
        nf = sum(len(v) for v in NH.values())
        tf = sum(len(v) for v in J.values())
        lo, hi = wilson(len(NH), len(J))
        print(f"\n  === 上游：到达归属分类器的东西里有多少不是手 ===")
        print(f"  轨迹级 {len(NH)}/{len(J)} = {len(NH)/len(J):.1%} "
              f"[{lo:.3f}, {hi:.3f}]   帧级 {nf}/{tf} = {nf/tf:.1%}")
        blur = sum(1 for v in NH.values() for r in v
                   if not int(r["final_owner_post_cap"]))
        print(f"  其中最终被判 other（=真的糊了）{blur} 帧 = "
              f"{blur/max(1,nf):.1%}，占语料全部糊帧的 "
              f"{blur/max(1, sum(1 for v in J.values() for r in v if not int(r['final_owner_post_cap']))):.1%}")
        for name, a_, b_ in (("1-3 帧", 1, 3), ("4-20 帧", 4, 20),
                             ("21-100 帧", 21, 100), (">100 帧", 101, 10 ** 9)):
            sel = [k for k in J if a_ <= len(J[k]) <= b_]
            if sel:
                n = sum(1 for k in sel if gold[k] == "not_a_hand")
                print(f"    {name:<10} {n:>3}/{len(sel):<4} = {n/len(sel):>6.1%}")

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


def _logit(p, eps=1e-6):
    p = min(1.0 - eps, max(eps, float(p)))
    return math.log(p / (1.0 - p))


def consolidate(a):
    """Track-level ownership consolidation: four estimators of one latent state.

    A HAND DOES NOT CHANGE OWNER WHILE IT EXISTS, so the thing to estimate is
    one state per physical track and not a decision per frame. `mixed_identity`
    came back 0 of 274, which is the assumption's own test: a track is a hand.
    Majority is only the crudest estimator of that state -- it counts 0.51 and
    0.99 as the same vote -- so the alternatives that use the magnitude of the
    evidence get measured beside it rather than assumed better.

    WHICH QUANTITY GETS ACCUMULATED IS THE WHOLE QUESTION. `p_owner_raw` is
    the classifier alone: clean per-frame evidence, but without the geometric
    prior or the frame-level cap, and those are what take foreign-hand recall
    from 0.786 to 0.963. `ema_owner` carries them but is already a temporal
    average, so accumulating it across the track averages an average and the
    frames stop being independent evidence in any sense. Neither is the right
    input; the right one -- blended, pre-smoothing -- is not stored, and that
    is a one-field change to the dump rather than an analysis to fudge here.
    Both are reported so the gap between them is visible.

    THE MEAN LOG-ODDS IS THE ONE WITH A STORY. Each frame contributes evidence
    and the track sums it, which is what a per-frame probability is FOR. It
    was measured once before and lost -- but against the exit rule, which is
    wrong on 37% of foreign tracks, so that result says nothing and this one
    replaces it.

    SEEN GOLD. Every number here comes from the 274 tracks that produced the
    majority rule in the first place. Anything that wins is a DEV candidate."""
    by = load(a)
    gold = {}
    for p in a.labels:
        for r in csv.DictReader(open(p, encoding="utf-8-sig")):
            gold[(r["rec"], r["tid"])] = r["human_ownership"]
    D = {k: v for k, v in by.items() if gold.get(k) in ("owner", "other")}

    def stats(v):
        lab2 = [bool(int(r["final_owner_post_cap"])) for r in v]
        lab1 = [bool(int(r["ownhold_pre_cap"])) for r in v]
        raw = [_logit(r["p_owner_raw"]) for r in v]
        ema = [_logit(r["ema_owner"]) for r in v]
        return lab1, lab2, raw, ema

    def est(name, fn):
        out = {}
        for k, v in D.items():
            out[k] = fn(*stats(v))
        return name, out

    ARMS = [
        est("T0 多数票 (P2 标签)",
            lambda l1, l2, raw, ema: sum(l2) * 2 > len(l2)),
        est("T1 平均概率 (ema)",
            lambda l1, l2, raw, ema: statistics.mean(
                1 / (1 + math.exp(-x)) for x in ema) > 0.5),
        est("T2 平均 log-odds (ema)",
            lambda l1, l2, raw, ema: statistics.mean(ema) > 0),
        est("T3 中位 log-odds (ema)",
            lambda l1, l2, raw, ema: statistics.median(ema) > 0),
        est("T4 平均 log-odds (raw)",
            lambda l1, l2, raw, ema: statistics.mean(raw) > 0),
        est("T5 中位 log-odds (raw)",
            lambda l1, l2, raw, ema: statistics.median(raw) > 0),
        est("T6 多数票 (P1 标签)",
            lambda l1, l2, raw, ema: sum(l1) * 2 > len(l1)),
    ]
    of = sum(len(v) for k, v in D.items() if gold[k] == "other")
    print(f"\n  === 整轨归属整合：同一个隐状态的几种估计 ===")
    print(f"  {'':<24}{'整轨漏':>7}{'误糊帧':>8}{'误糊s':>8}{'帧召回':>9}"
          f"{'与T0不同':>9}")
    base = ARMS[0][1]
    for name, out in ARMS:
        miss = sum(1 for k in D if gold[k] == "other" and out[k])
        blur = sum(len(D[k]) for k in D if gold[k] == "owner" and not out[k])
        cov = sum(len(D[k]) for k in D if gold[k] == "other" and not out[k])
        diff = sum(1 for k in D if out[k] != base[k])
        print(f"  {name:<24}{miss:>7}{blur:>8}{blur/FPS:>8.1f}"
              f"{cov/of:>9.3f}{diff:>9}")
    print("  广播之后每条轨迹只有一个标签，所以暴露事件数 = 整轨漏的条数，"
          "\n  每次暴露的长度 = 那条轨迹的长度。这就是聚合的全部风险。")

    print(f"\n  === 99 条别人的手 vs 169 条自己的手，整条概率轨迹长什么样 ===")
    print(f"  {'':<12}{'轨迹':>5}{'长度中位':>9}{'other帧占比':>11}"
          f"{'中位logit':>10}{'最长错run':>10}{'标签跳变':>9}")
    for want in ("other", "owner"):
        ks = [k for k in D if gold[k] == want]
        rows = []
        for k in ks:
            v = D[k]
            l1, l2, raw, ema = stats(v)
            wrong, cur, best = (gold[k] == "other"), 0, 0
            for x in l2:
                if x == wrong:
                    cur += 1
                    best = max(best, cur)
                else:
                    cur = 0
            rows.append((len(v), sum(1 for x in l2 if not x) / len(v),
                         statistics.median(raw), best,
                         sum(1 for x, y in zip(l2, l2[1:]) if x != y)))
        med = lambda i: statistics.median([r[i] for r in rows])
        print(f"  {want:<12}{len(ks):>5}{med(0):>9.0f}{med(1):>11.2f}"
              f"{med(2):>10.2f}{med(3):>10.0f}{med(4):>9.0f}")
    print("  『最长错run』= 整条里连续判错的最长帧数；『标签跳变』= 逐帧标签"
          "改变次数。\n  两者都接近 0 就说明轨迹本身已经很稳，多数票没什么可修的。")

    if a.out:
        with open(a.out, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["rec", "tid", "human_ownership", "n_frames",
                        "frac_other_p2", "mean_logit_raw", "median_logit_raw",
                        "mean_logit_ema", "longest_wrong_run",
                        "label_transitions"] + [n.split()[0] for n, _ in ARMS])
            for k, v in sorted(D.items()):
                l1, l2, raw, ema = stats(v)
                wrong, cur, best = (gold[k] == "other"), 0, 0
                for x in l2:
                    cur = cur + 1 if x == wrong else 0
                    best = max(best, cur)
                w.writerow([k[0], k[1], gold[k], len(v),
                            round(sum(1 for x in l2 if not x) / len(v), 3),
                            round(statistics.mean(raw), 3),
                            round(statistics.median(raw), 3),
                            round(statistics.mean(ema), 3), best,
                            sum(1 for x, y in zip(l2, l2[1:]) if x != y)]
                           + [int(out[k]) for _, out in ARMS])
        print(f"\n  每条轨迹的形态 -> {a.out}")


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

    def b64(img, w, q=84, box=None):
        # THE BOX GOES ON AFTER THE RESIZE. Two pixels of line on a 1400 px
        # crop is a quarter of a pixel once the crop is 185 px wide, and
        # cv2.resize samples rather than averages on the way down, so the
        # line does not thin -- it disappears. Measured on the sheets this
        # file has already produced: 34 of 1050 crops lost their box.
        s = w / float(max(1, img.shape[1]))
        out = cv2.resize(img, (w, max(1, int(round(img.shape[0] * s)))))
        if box is not None:
            cv2.rectangle(out,
                          (int(round(box[0] * s)), int(round(box[1] * s))),
                          (int(round(box[2] * s)), int(round(box[3] * s))),
                          (60, 255, 60), 2)
        ok, buf = cv2.imencode(".jpg", out,
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
                shots.append({"f": f,
                              "img": b64(img[cy0:cy1, cx0:cx1], a.crop_w,
                                         box=(x0 - cx0, y0 - cy0,
                                              x1 - cx0, y1 - cy0))})
                if f in ends:
                    wides.append(b64(img, a.wide_w, 80,
                                     box=(x0, y0, x1, y1)))
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
    ap.add_argument("--consolidate", action="store_true",
                    help="track-level ownership consolidation: estimators "
                         "of the one latent state, and the track morphology")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.labels:
        if a.consolidate:
            consolidate(a)
        elif a.arms:
            arms(a)
        else:
            report(a)
    elif a.out:
        build(a)
    else:
        ap.error("--out (build) or --labels (report)")


if __name__ == "__main__":
    main()
