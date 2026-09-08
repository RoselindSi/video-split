"""Does a person agree with the stored convention of cutting repetitions?

A THIRD OF EVERY BOUNDARY IN THIS CORPUS SITS BETWEEN TWO REPETITIONS OF THE
SAME ACTION. Over 471 recordings and 25433 adjacent segment pairs, 8137 pairs
carry an identical label and another 395 differ only by an ordinal -- `Relocate
second white slipper` followed by `Relocate third white slipper`. The stored
annotation cuts there, consistently, 8532 times. That is a convention, and it
is not the same thing as a fact.

WHAT IS NOT KNOWN IS WHETHER ANYONE AGREES WITH IT. 44.4% of stored boundaries
that have been audited were judged not to be boundaries, and the audited
events barely intersect the ones with adjacent labels -- four events out of
sixty-three, one of the eight known repetition cuts. So the question `is the
repetition convention part of the 44.4%, or is it the reliable part` has no
answer with any power behind it, and a rule written from the convention would
be codifying something never checked, at a third of all boundaries.

THE SAMPLE IS STRATIFIED AND THE STRATUM IS HIDDEN. Repetition candidates and
ordinary candidates are drawn separately, interleaved at random, and the sheet
shows neither the labels nor which pile a candidate came from -- the labels
would give the stratum away instantly, since a repetition pair reads the same
on both sides. What the judge sees is the picture before and the picture
after, which is the evidence the question is actually about.

READ AS A DIFFERENCE, NOT AS A RATE. Neither pile is a random sample of the
corpus, so neither number is `the` boundary precision. The comparison between
them is what carries: if repetition cuts are rejected far more often than
ordinary ones, the convention is the problem; if the two agree, the convention
is as good as everything else and the rule can simply adopt it.
"""
from __future__ import annotations

import argparse
import base64
import collections
import csv
import json
import os
import random
import re

# Words that make `Relocate second slipper` and `Relocate third slipper` the
# same action twice rather than two different actions.
ORDINAL = re.compile(
    r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|"
    r"1st|2nd|3rd|\d+th|again|another|repeat|next)\b", re.I)

# Where to sample frames around the candidate, in seconds. Dense near the
# boundary because that is where the evidence is, and out to three seconds
# because a repetition is only visible as a repetition at that scale.
OFFSETS = (-3.0, -2.0, -1.0, -0.4, 0.4, 1.0, 2.0, 3.0)

SHEET = """<meta charset=utf-8><title>boundary judgement</title><style>
body{font:13px/1.5 system-ui;margin:0;background:#111;color:#ddd}
#bar{position:sticky;top:0;background:#181818;padding:9px 14px;z-index:9;
  border-bottom:1px solid #333;display:flex;gap:16px;align-items:center;
  flex-wrap:wrap}
button{font:13px system-ui;padding:5px 10px;cursor:pointer}
b{color:#ffd33d}
.c{padding:12px 14px;border-bottom:1px solid #262626}
.c.cur{background:#1d2430;outline:2px solid #4a8}
.c.yes{border-left:5px solid #2a6}
.c.no{border-left:5px solid #d33}
.c.unsure{border-left:5px solid #666}
.strip{display:flex;gap:4px;align-items:flex-start}
.strip figure{margin:0;flex:1}
.strip img{width:100%;border-radius:3px;display:block}
.strip figcaption{font-size:10px;color:#888;text-align:center;margin-top:2px}
.split{width:3px;background:#ffd33d;border-radius:2px;align-self:stretch;
  margin:0 4px}
.meta{color:#9ab;font-size:12px;margin-bottom:6px}
.num{color:#8ab4c8;font-family:ui-monospace,monospace;font-size:11px}
</style>
<div id=bar>
 <span id=prog></span>
 <span><b>1</b> 是边界 &nbsp; <b>2</b> 不是边界 &nbsp; <b>3</b> 说不准
  &nbsp; <b>&uarr;&darr;</b> move &nbsp; <b>u</b> undo</span>
 <button onclick="dl()">download CSV</button>
 <span class=num>黄线左边是候选时刻之前，右边是之后。问的是：这里该不该切一刀。</span>
</div>
<div id=list></div>
<script>
const D = __PAYLOAD__;
const KEY = "rep:" + D.tag;
const V = ["yes","no","unsure"];
let lab = {}, cur = 0, hist = [];
try { lab = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch(e) { lab={}; }
const list = document.getElementById("list");
D.cands.forEach((c, i) => {
  const d = document.createElement("div");
  d.className = "c"; d.id = "c" + i;
  let strip = "";
  c.frames.forEach((f, j) => {
    if (j === c.split) strip += '<div class=split></div>';
    strip += '<figure><img src="' + f.img + '"><figcaption>' +
             f.dt + 's</figcaption></figure>';
  });
  d.innerHTML = '<div class=meta>#' + (i+1) + ' &nbsp; <span class=num>' +
    c.rec + '  t=' + c.t.toFixed(1) + 's</span> <span id=v' + i +
    '></span></div><div class=strip>' + strip + '</div>';
  d.onclick = () => { cur = i; draw(); };
  list.appendChild(d);
});
function draw(){
  D.cands.forEach((c,i)=>{
    const v = lab[c.key];
    document.getElementById("c"+i).className =
      "c " + (v || "") + (i===cur ? " cur" : "");
    document.getElementById("v"+i).innerHTML = v ?
      '<b style="color:' + (v==="yes"?"#5c5":v==="no"?"#d55":"#999") + '">' +
      (v==="yes"?"是边界":v==="no"?"不是":"说不准") + '</b>' : '';
  });
  const n = Object.keys(lab).length;
  document.getElementById("prog").innerHTML =
    "<b>" + D.tag + "</b> &nbsp; " + n + "/" + D.cands.length + " 已判";
  localStorage.setItem(KEY, JSON.stringify(lab));
  const el = document.getElementById("c"+cur);
  if(el) el.scrollIntoView({block:"nearest"});
}
document.onkeydown = ev => {
  if(ev.key>="1" && ev.key<="3"){
    const c = D.cands[cur]; if(!c) return;
    hist.push([c.key, lab[c.key]]);
    lab[c.key] = V[+ev.key-1];
    cur = Math.min(cur+1, D.cands.length-1);
    draw();
  }
  else if(ev.key==="ArrowDown"){ cur=Math.min(cur+1,D.cands.length-1); draw(); }
  else if(ev.key==="ArrowUp"){ cur=Math.max(cur-1,0); draw(); }
  else if(ev.key==="u"){ const h=hist.pop(); if(h){ if(h[1]===undefined)
      delete lab[h[0]]; else lab[h[0]]=h[1]; draw(); } }
  else return;
  ev.preventDefault();
};
draw();
function dl(){
  let s = "key,recording,t,verdict\\n";
  for(const c of D.cands) if(c.key in lab)
    s += c.key + "," + c.rec + "," + c.t.toFixed(2) + "," + lab[c.key] + "\\n";
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([s], {type:"text/csv"}));
  a.download = "repeat_" + D.tag + ".csv";
  a.click();
}
</script>
"""


def strip_ordinal(s):
    return re.sub(r"\s+", " ", ORDINAL.sub("", s)).strip(" ,").lower()


def adjacent_pairs(index_csv):
    """-> [(recording, t, kind, prev_label, next_label)] over one index file."""
    by = collections.defaultdict(list)
    for r in csv.DictReader(open(index_csv, encoding="utf-8-sig")):
        by[r["recording_id"]].append(r)
    out = []
    for rec, v in by.items():
        v.sort(key=lambda r: float(r["start_s"]))
        for a, b in zip(v, v[1:]):
            la, lb = a["label_en"].strip(), b["label_en"].strip()
            if la.lower() == lb.lower():
                kind = "repeat_same"
            elif (ORDINAL.search(la) or ORDINAL.search(lb)) and \
                    strip_ordinal(la) == strip_ordinal(lb):
                kind = "repeat_ordinal"
            else:
                kind = "other"
            # The cut is where the previous segment ends. Where the next one
            # starts can differ -- the stored segments do not always tile.
            out.append((rec, float(a["end_s"]), kind, la, lb))
    return out


def sample(pairs, n_each, per_rec, seed):
    """Stratified, capped per recording, interleaved.

    CAPPING MATTERS MORE THAN IT LOOKS. One recording contributes eleven
    consecutive `Lift and replace two wall-leaning tools` segments; without a
    cap a single scene would supply a fifth of the repetition arm and the
    comparison would be between two recordings rather than two conventions."""
    rng = random.Random(seed)
    picked = []
    for want_kind, tag in ((lambda k: k.startswith("repeat"), "repeat"),
                           (lambda k: k == "other", "control")):
        pool = collections.defaultdict(list)
        for rec, t, kind, la, lb in pairs:
            if want_kind(kind):
                pool[rec].append((rec, t, kind, la, lb))
        recs = sorted(pool)
        rng.shuffle(recs)
        got = []
        for rec in recs:
            rng.shuffle(pool[rec])
            got += [(tag,) + x for x in pool[rec][:per_rec]]
            if len(got) >= n_each:
                break
        picked += got[:n_each]
    rng.shuffle(picked)                   # the judge must not see the pattern
    return picked


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", action="append", required=True,
                    help="a corpus part directory (repeat for both)")
    ap.add_argument("--n", type=int, default=60, help="candidates per arm")
    ap.add_argument("--per_rec", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--width", type=int, default=300)
    ap.add_argument("--out", required=True)
    ap.add_argument("--truth", help="write the hidden stratum here, for the "
                                    "report to join after judging")
    a = ap.parse_args()

    import cv2

    pairs, where = [], {}
    for root in a.root:
        idx = os.path.join(root, "global_segment_index.csv")
        got = adjacent_pairs(idx)
        pairs += got
        for rec, *_ in got:
            where[rec] = os.path.join(root, "recordings", rec, "mid.mp4")
    kinds = collections.Counter(k for _, _, k, _, _ in pairs)
    print(f"  {len(pairs)} 个相邻边界   {dict(kinds)}")

    picked = sample(pairs, a.n, a.per_rec, a.seed)
    print(f"  抽了 {len(picked)}   "
          f"{collections.Counter(t for t, *_ in picked)}")

    cands, truth = [], []
    for i, (tag, rec, t, kind, la, lb) in enumerate(picked):
        path = where.get(rec)
        if not path or not os.path.exists(path):
            print(f"  !! {rec} 没有视频，跳过")
            continue
        cap = cv2.VideoCapture(path)
        frames = []
        for dt in OFFSETS:
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, (t + dt) * 1000.0))
            ok, img = cap.read()
            if not ok:
                continue
            h = int(round(img.shape[0] * a.width / img.shape[1]))
            ok2, buf = cv2.imencode(".jpg", cv2.resize(img, (a.width, h)),
                                    [int(cv2.IMWRITE_JPEG_QUALITY), 78])
            if ok2:
                frames.append({"dt": f"{dt:+.1f}",
                               "img": "data:image/jpeg;base64,"
                                      + base64.b64encode(buf).decode()})
        cap.release()
        if len(frames) < len(OFFSETS):
            print(f"  !! {rec} t={t} 只取到 {len(frames)} 帧，跳过")
            continue
        key = f"c{i:03d}"
        cands.append({"key": key, "rec": rec, "t": t, "frames": frames,
                      "split": len([o for o in OFFSETS if o < 0])})
        # The labels and the stratum go in the side file, never in the sheet.
        truth.append({"key": key, "recording": rec, "t": round(t, 2),
                      "arm": tag, "kind": kind,
                      "prev_label": la, "next_label": lb})
        if (i + 1) % 20 == 0:
            print(f"    {i + 1}/{len(picked)}", flush=True)

    with open(a.out, "w", encoding="utf-8") as f:
        f.write(SHEET.replace("__PAYLOAD__",
                              json.dumps({"tag": "boundary", "cands": cands})))
    print(f"\n  {len(cands)} 条 -> {a.out} "
          f"({os.path.getsize(a.out) / 1e6:.1f} MB)")
    if a.truth:
        with open(a.truth, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(truth[0].keys()))
            w.writeheader()
            w.writerows(truth)
        print(f"  分层与标签（判完再看）-> {a.truth}")
    print("  1 是边界   2 不是边界   3 说不准。页面上没有标签、没有分层，"
          "两臂随机交错。")


if __name__ == "__main__":
    main()
