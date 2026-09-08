"""Where the corpus contradicts itself about repetition, side by side.

THE SAME LABEL IS CUT INTO TWO-SECOND PIECES IN ONE PLACE AND LEFT AS A
SEVENTY-SECOND BLOCK IN ANOTHER. `fold and press paper sheet` has a median
duration of 2.0s over 31 instances and a longest instance of 73.3s;
`tissue fold-unfold cycle` runs 2.0s median over 78 instances and 43.5s at
its longest. One convention says a repetition is a segment, the other says a
bout of repetitions is a segment, and the corpus uses both. 153 label
families are split this way and they cover 4628 segments, 17.9% of everything.

THIS IS THE ONE QUESTION THAT CANNOT BE ANSWERED BY SAMPLING BOUNDARIES.
Asking a person whether a given instant is a boundary measures whether they
accept the convention that was applied there; it cannot reveal that the
opposite convention was applied elsewhere to the same action. The
contradiction is a property of the pair, so the pair is what gets shown.

THE JUDGEMENT IS PER FAMILY, NOT PER SEGMENT. One decision repairs every
instance of that action, which is why this is two orders of magnitude cheaper
than relabelling: the last time labels were corrected it was eight events, and
that moved AUROC by +0.025 with 72% of the gain landing on events that were
not touched.

NOTHING HERE COMES FROM A MODEL. The families are found by comparing the
annotation against itself -- same label, incompatible durations -- so
adjudicating them cannot bias the label set toward any model, which is the
trap recorded against `only fix what the model disagrees with`.

MERGING IS MECHANICAL AND SPLITTING IS NOT. If the verdict is `one segment per
bout`, adjacent identical labels can be joined without anyone watching
anything, and the corpus loses 8137 of its 25904 segments. If the verdict is
`one segment per repetition`, the long blocks have to be cut by hand and the
work is unbounded. The asymmetry is worth knowing before answering, so it is
printed rather than hidden.
"""
from __future__ import annotations

import argparse
import base64
import collections
import csv
import json
import os
import re
import statistics as st

ORDINAL = re.compile(
    r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|"
    r"1st|2nd|3rd|\d+th|again|another|repeat|next)\b", re.I)

# A family is contradictory when its longest instance is at least this many
# times its median. Three is not a tuned number: at 3x a single instance holds
# at least three of whatever the median instance holds, which is the smallest
# ratio that cannot be explained by one slow repetition.
RATIO = 3.0
MIN_N = 6

SHEET = """<meta charset=utf-8><title>repetition conflicts</title><style>
body{font:13px/1.5 system-ui;margin:0;background:#111;color:#ddd}
#bar{position:sticky;top:0;background:#181818;padding:9px 14px;z-index:9;
  border-bottom:1px solid #333;display:flex;gap:14px;align-items:center;
  flex-wrap:wrap}
button{font:13px system-ui;padding:5px 10px;cursor:pointer}
b{color:#ffd33d}
.f{padding:12px 14px;border-bottom:1px solid #262626}
.f.cur{background:#1d2430;outline:2px solid #4a8}
.f.split{border-left:5px solid #2a6}
.f.merge{border-left:5px solid #38d}
.f.depends{border-left:5px solid #d83}
.f.unsure{border-left:5px solid #666}
.lab{font-size:14px;color:#ffd33d;margin-bottom:2px}
.num{color:#8ab4c8;font-family:ui-monospace,monospace;font-size:11px}
.arm{margin:6px 0}
.arm h4{margin:0 0 3px;font-size:12px;font-weight:600;color:#bcd}
.strip{display:flex;gap:3px;align-items:flex-start}
.strip figure{margin:0;flex:1}
.strip img{width:100%;border-radius:2px;display:block}
.strip figcaption{font-size:10px;color:#888;text-align:center}
.cutmark{width:3px;background:#ffd33d;border-radius:2px;align-self:stretch}
</style>
<div id=bar>
 <span id=prog></span>
 <span><b>1</b> 切开对（每次重复一段） &nbsp; <b>2</b> 合并对（一轮重复一段）
  &nbsp; <b>3</b> 看情况（需要判据） &nbsp; <b>4</b> 说不准
  &nbsp; <b>&uarr;&darr;</b> move &nbsp; <b>u</b> undo</span>
 <button onclick="dl()">download CSV</button>
 <span class=num id=impact></span>
</div>
<div id=list></div>
<script>
const D = __PAYLOAD__;
const KEY = "conf:" + D.tag;
const V = ["split","merge","depends","unsure"];
const CN = {split:"切开对", merge:"合并对", depends:"看情况", unsure:"说不准"};
let lab = {}, cur = 0, hist = [];
try { lab = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch(e) { lab={}; }
const list = document.getElementById("list");
function strip(fr, marks){
  let s = "";
  fr.forEach((f, j) => {
    if (marks && j > 0) s += '<div class=cutmark></div>';
    s += '<figure><img src="' + f.img + '"><figcaption>' + f.cap +
         '</figcaption></figure>';
  });
  return '<div class=strip>' + s + '</div>';
}
D.fams.forEach((c, i) => {
  const d = document.createElement("div");
  d.className = "f"; d.id = "f" + i;
  d.innerHTML = '<div class=lab>' + c.label + ' <span id=v' + i +
    '></span></div><span class=num>族内 ' + c.n + ' 条，中位 ' + c.med +
    's，覆盖 ' + c.n + ' 段</span>' +
    '<div class=arm><h4>切开的例子 &mdash; ' + c.cut_rec + '，连续 ' +
    c.cut_n + ' 段，每段约 ' + c.cut_dur + 's（黄线=标注的切点）</h4>' +
    strip(c.cut, true) + '</div>' +
    '<div class=arm><h4>没切的例子 &mdash; ' + c.long_rec + '，1 段 ' +
    c.long_dur + 's，同一个标签</h4>' + strip(c.long, false) + '</div>';
  d.onclick = () => { cur = i; draw(); };
  list.appendChild(d);
});
function draw(){
  D.fams.forEach((c,i)=>{
    const v = lab[c.key];
    document.getElementById("f"+i).className =
      "f " + (v || "") + (i===cur ? " cur" : "");
    document.getElementById("v"+i).innerHTML =
      v ? '<span class=num style="color:#8f8">[' + CN[v] + ']</span>' : '';
  });
  const n = Object.keys(lab).length;
  document.getElementById("prog").innerHTML =
    "<b>" + D.tag + "</b> &nbsp; " + n + "/" + D.fams.length + " 已判";
  let segs = 0;
  for (const c of D.fams) if (lab[c.key]) segs += c.n;
  document.getElementById("impact").innerHTML =
    "已判决覆盖 " + segs + " 段标注";
  localStorage.setItem(KEY, JSON.stringify(lab));
  const el = document.getElementById("f"+cur);
  if(el) el.scrollIntoView({block:"nearest"});
}
document.onkeydown = ev => {
  if(ev.key>="1" && ev.key<="4"){
    const c = D.fams[cur]; if(!c) return;
    hist.push([c.key, lab[c.key]]);
    lab[c.key] = V[+ev.key-1];
    cur = Math.min(cur+1, D.fams.length-1);
    draw();
  }
  else if(ev.key==="ArrowDown"){ cur=Math.min(cur+1,D.fams.length-1); draw(); }
  else if(ev.key==="ArrowUp"){ cur=Math.max(cur-1,0); draw(); }
  else if(ev.key==="u"){ const h=hist.pop(); if(h){ if(h[1]===undefined)
      delete lab[h[0]]; else lab[h[0]]=h[1]; draw(); } }
  else return;
  ev.preventDefault();
};
draw();
function dl(){
  let s = "family,verdict,n_segments,median_s,longest_s\\n";
  for(const c of D.fams) if(lab[c.key])
    s += '"' + c.label.replace(/"/g,"'") + '",' + lab[c.key] + "," + c.n +
         "," + c.med + "," + c.long_dur + "\\n";
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([s], {type:"text/csv"}));
  a.download = "repeat_conflict.csv";
  a.click();
}
</script>
"""


def normalise(s):
    return re.sub(r"\s+", " ", ORDINAL.sub("", s.lower())).strip(" ,")


def load(roots):
    """-> (rows, {recording: video path})"""
    rows, where = [], {}
    for root in roots:
        idx = os.path.join(root, "global_segment_index.csv")
        for r in csv.DictReader(open(idx, encoding="utf-8-sig")):
            r["dur"] = float(r["end_s"]) - float(r["start_s"])
            r["start"] = float(r["start_s"])
            rows.append(r)
            where.setdefault(r["recording_id"],
                             os.path.join(root, "recordings",
                                          r["recording_id"], "mid.mp4"))
    return rows, where


def families(rows):
    """Contradictory families, worst first. -> [dict]"""
    by_rec = collections.defaultdict(list)
    for r in rows:
        by_rec[r["recording_id"]].append(r)
    for v in by_rec.values():
        v.sort(key=lambda r: r["start"])

    fam = collections.defaultdict(list)
    for r in rows:
        fam[normalise(r["label_en"])].append(r)

    out = []
    for label, v in fam.items():
        if len(v) < MIN_N:
            continue
        med = st.median(x["dur"] for x in v)
        longest = max(v, key=lambda x: x["dur"])
        if med <= 0 or longest["dur"] / med < RATIO:
            continue
        # The cut exemplar is the longest RUN of consecutive segments sharing
        # this label -- that is the convention at its most explicit.
        best_run = None
        for rec, seq in by_rec.items():
            i = 0
            while i < len(seq):
                j = i
                while (j + 1 < len(seq)
                       and normalise(seq[j + 1]["label_en"]) == label
                       and normalise(seq[i]["label_en"]) == label):
                    j += 1
                if normalise(seq[i]["label_en"]) == label and j > i:
                    if best_run is None or (j - i) > (best_run[2] - best_run[1]):
                        best_run = (rec, i, j)
                i = j + 1
        if best_run is None:
            continue                      # never cut anywhere: no contradiction
        rec, i, j = best_run
        run = by_rec[rec][i:j + 1]
        out.append({"label": label, "n": len(v), "med": round(med, 1),
                    "long": longest, "run_rec": rec, "run": run})
    out.sort(key=lambda d: -d["n"])
    return out


def frames_at(cap, times, width, caps, cv2):
    got = []
    for t, cap_txt in zip(times, caps):
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t * 1000.0))
        ok, img = cap.read()
        if not ok:
            return None
        h = int(round(img.shape[0] * width / img.shape[1]))
        ok2, buf = cv2.imencode(".jpg", cv2.resize(img, (width, h)),
                                [int(cv2.IMWRITE_JPEG_QUALITY), 76])
        if not ok2:
            return None
        got.append({"cap": cap_txt,
                    "img": "data:image/jpeg;base64,"
                           + base64.b64encode(buf).decode()})
    return got


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", action="append", required=True)
    ap.add_argument("--limit", type=int, default=60,
                    help="families to render, biggest corpus impact first")
    ap.add_argument("--shots", type=int, default=5)
    ap.add_argument("--width", type=int, default=240)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import cv2

    rows, where = load(a.root)
    fams = families(rows)
    covered = sum(f["n"] for f in fams)
    print(f"  {len(rows)} 段   {len(fams)} 个矛盾族   覆盖 {covered} 段 "
          f"= {covered / len(rows):.1%}")

    out = []
    for f in fams[:a.limit]:
        vid_run = where.get(f["run_rec"])
        vid_long = where.get(f["long"]["recording_id"])
        if not (vid_run and vid_long
                and os.path.exists(vid_run) and os.path.exists(vid_long)):
            continue
        run = f["run"][:a.shots]
        cap = cv2.VideoCapture(vid_run)
        cut = frames_at(cap, [s["start"] + s["dur"] / 2 for s in run],
                        a.width, [f"{s['dur']:.1f}s" for s in run], cv2)
        cap.release()
        L = f["long"]
        cap = cv2.VideoCapture(vid_long)
        ts = [L["start"] + L["dur"] * k / (a.shots - 1.0)
              for k in range(a.shots)]
        lg = frames_at(cap, ts, a.width,
                       [f"+{t - L['start']:.0f}s" for t in ts], cv2)
        cap.release()
        if not cut or not lg:
            print(f"  !! {f['label'][:40]} 取帧失败，跳过")
            continue
        out.append({"key": f"f{len(out):03d}", "label": f["label"],
                    "n": f["n"], "med": f["med"],
                    "cut": cut, "cut_rec": f["run_rec"], "cut_n": len(f["run"]),
                    "cut_dur": round(st.median(s["dur"] for s in f["run"]), 1),
                    "long": lg, "long_rec": L["recording_id"],
                    "long_dur": round(L["dur"], 1)})
        if len(out) % 10 == 0:
            print(f"    {len(out)}/{min(a.limit, len(fams))}", flush=True)

    with open(a.out, "w", encoding="utf-8") as fh:
        fh.write(SHEET.replace("__PAYLOAD__",
                               json.dumps({"tag": "repeat", "fams": out})))
    print(f"\n  {len(out)} 个族 -> {a.out} "
          f"({os.path.getsize(a.out) / 1e6:.1f} MB)")
    print(f"  这 {len(out)} 个族覆盖 {sum(d['n'] for d in out)} 段标注")
    print("  1 切开对   2 合并对   3 看情况   4 说不准")
    print("  合并是机械可执行的（相邻同标签直接接上），切开不是 —— 长段要人工重切。")


if __name__ == "__main__":
    main()
