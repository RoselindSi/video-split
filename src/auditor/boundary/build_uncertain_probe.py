"""Is `uncertain` reachable? Nine clips, three buttons, and one question about the option not taken.

WHY A PROBE AND NOT A THRESHOLD. reviewer_2 used `uncertain` zero times in 64
segments. That is a fact about the round, not a verdict: the value may be
unnecessary, or it may be unreachable from the manual and the UI, and those
need opposite fixes. Usage rate cannot separate them, so it is not an
acceptance criterion here. What separates them is asking directly.

THE DIAGNOSTIC IS THE SECOND QUESTION. Every case asks for a verdict, and
then asks whether `uncertain` was considered and why it was not chosen:

  found it, judged it unnecessary    -> the contract is what needs changing
  wanted it, picked something else   -> the wording, the option or the
                                        validation is what needs changing
  did not see it                     -> the UI is what needs changing

THE NINE ARE THREE OF EACH, and the reader is not told which is which.
Three where the evidence is plainly there, three where v1's own annotator
wrote that they could not see the action, and the three transport cases the
E0 return disagreed on. Without the plainly-judgeable ones a reader can infer
that everything on the sheet is meant to be hard; without the invisible ones
there is nothing that SHOULD be uncertain, and a zero would prove nothing.

The three transport cases are not blind -- reviewer_2 has labelled them
already. They are here to locate a known disagreement, not to validate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess

SEED = 20260926
PAD = 3.0          # seconds of run-up before the span
MAX_CLIP = 40.0    # a probe case is a look, not a full annotation

# kind is for the analysis key only and never reaches the reader.
CASES = [
    ("graph_review_0013", 4.6, 28.7, "clear_interaction"),
    ("graph_review_0009", 7.9, 29.4, "clear_interaction"),
    ("graph_review_0001", 17.9, 45.0, "clear_interaction"),
    ("graph_review_0015", 20.0, 50.0, "invisible"),
    ("mosaic_graph_0027", 0.0, 17.2, "invisible"),
    ("graph_review_0011", 22.0, 24.8, "invisible"),
    ("mosaic_graph_0002", 57.1, 97.4, "transport"),
    ("mosaic_graph_0022", 113.3, 120.0, "transport"),
    ("mosaic_graph_0026", 67.3, 81.2, "transport"),
]
MANIFESTS = [
    "results/auditor/interaction_graph_mosaic_complement_v1/manifest.json",
    "results/auditor/interaction_graph_timeline_demo/manifest.json",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifests", nargs="+", default=MANIFESTS)
    a = ap.parse_args()

    src = {}
    for p in a.manifests:
        for v in json.load(open(p, encoding="utf-8"))["videos"]:
            src[v["video_id"]] = v["video"]

    out = os.path.abspath(a.out)
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(os.path.join(out, "clips"))

    rows = list(CASES)
    random.Random(SEED).shuffle(rows)
    public, key = [], []
    for i, (vid, t0, t1, kind) in enumerate(rows, 1):
        if vid not in src:
            raise SystemExit("清单里没有 %s" % vid)
        s0 = max(0.0, t0 - PAD)
        s1 = min(t1 + PAD, s0 + MAX_CLIP)
        name = "case%02d.mp4" % i
        subprocess.run(
            ["ffmpeg", "-v", "error", "-ss", "%.2f" % s0, "-i", src[vid],
             "-t", "%.2f" % (s1 - s0), "-c:v", "libx264", "-preset", "veryfast",
             "-crf", "23", "-movflags", "+faststart", "-an",
             os.path.join(out, "clips", name), "-y"], check=True)
        public.append({"id": "case%02d" % i, "clip": "clips/" + name,
                       "seconds": round(s1 - s0, 1)})
        key.append({"id": "case%02d" % i, "video_id": vid,
                    "span": [t0, t1], "clip_span": [round(s0, 1), round(s1, 1)],
                    "kind": kind})

    page = PAGE.replace("__CASES__", json.dumps(public, ensure_ascii=False))
    open(os.path.join(out, "打开判断小测.html"), "w", encoding="utf-8").write(page)
    json.dump({"seed": SEED, "cases": key},
              open(out.rstrip("/") + "_key.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    open(os.path.join(out, "README_先看这里.txt"), "w", encoding="utf-8").write(
        "九段短片，每段问两个问题。用 Chrome 或 Edge 打开「打开判断小测.html」。\n\n"
        "1. 这一段是交互节点、特殊状态，还是不确定？\n"
        "2. 你有没有考虑过「不确定」？如果考虑了但最后没选，是为什么？\n\n"
        "第二个问题才是这次真正要问的。选了什么都可以，没有标准答案。\n"
        "如果你觉得某一段根本不需要「不确定」，就直接这么写。\n\n"
        "填完点「导出结果」，把导出的 JSON 发回。\n")
    h = hashlib.sha256(open(os.path.join(out, "打开判断小测.html"), "rb").read())
    print("-> %s（%d 段）\n   页面 sha256 %s\n   对照表 %s_key.json（不要发给标注者）"
          % (out, len(public), h.hexdigest()[:16], out.rstrip("/")))


PAGE = """<!doctype html>
<html lang="zh-CN">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>判断小测</title>
<style>
:root{color-scheme:dark;--bg:#111513;--panel:#191e1b;--line:#46504a;--text:#f3f5f3;--muted:#b5bdb8;--accent:#72cdaa;--accent2:#2f7558}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:17px/1.6 system-ui,sans-serif}
main{max-width:820px;margin:auto;padding:18px}
h1{font-size:20px}
.rule{border-left:4px solid var(--accent);background:#1d2722;padding:12px 15px;margin-bottom:18px}
.case{border:1px solid var(--line);background:var(--panel);padding:16px;margin-bottom:20px}
.case h2{font-size:17px;margin:0 0 10px}
video{width:100%;max-height:52vh;background:#000;display:block}
.choices{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin:12px 0}
.choices button{min-height:52px;background:#252c28;border:1px solid #68736d;border-radius:4px;color:var(--text);font:inherit;cursor:pointer}
.choices button.on{background:var(--accent2);border-color:var(--accent);font-weight:700}
label{display:block;font-weight:700;margin:12px 0 6px}
textarea{width:100%;min-height:64px;padding:10px;background:#252c28;color:var(--text);border:1px solid #68736d;border-radius:4px;font:inherit;resize:vertical}
.muted{color:var(--muted);font-weight:400}
footer{position:sticky;bottom:0;background:var(--bg);border-top:1px solid var(--line);padding:14px 0;display:flex;gap:10px;align-items:center}
footer button{min-height:48px;padding:10px 16px;background:var(--accent2);border:1px solid var(--accent);border-radius:4px;color:var(--text);font:inherit;cursor:pointer}
footer button:disabled{opacity:.45;cursor:default}
@media(max-width:600px){.choices{grid-template-columns:1fr}}
</style>
<main>
<h1>判断小测</h1>
<section class="rule">
<b>九段短片，每段两个问题。</b>
<div class="muted">第一个问：这一段是<b>交互节点</b>（能确定物体、看得到动作、看得到结果）、<b>特殊状态</b>（等待跟随巡视位移遮挡）、还是<b>不确定</b>？</div>
<div class="muted">第二个问才是重点：你有没有考虑过「不确定」？考虑了但最后没选，是为什么？</div>
<div class="muted">没有标准答案。觉得某段根本用不上「不确定」，就直接这么写。</div>
</section>
<div id="list"></div>
<footer><button id="export" disabled>导出结果</button><span class="muted" id="progress"></span></footer>
</main>
<script>
const CASES=__CASES__;
const KEY='uncertain-probe-v21';
let ans={};try{ans=JSON.parse(localStorage.getItem(KEY))||{}}catch{}
const TYPES=[['interaction_node','交互节点'],['special_state','特殊状态'],['uncertain','不确定']];
const $=s=>document.querySelector(s);
function save(){try{localStorage.setItem(KEY,JSON.stringify(ans))}catch{}
 const done=CASES.filter(c=>ans[c.id]?.type&&(ans[c.id]?.why_not_uncertain||'').trim()).length;
 $('#progress').textContent='已填 '+done+' / '+CASES.length;$('#export').disabled=done!==CASES.length}
function render(){const root=$('#list');root.innerHTML='';
 CASES.forEach((c,i)=>{const a=ans[c.id]||(ans[c.id]={});
  const el=document.createElement('section');el.className='case';
  el.innerHTML='<h2>第 '+(i+1)+' / '+CASES.length+' 段 <span class="muted">'+c.seconds+' 秒</span></h2>'
   +'<video controls preload="metadata" src="'+c.clip+'"></video>'
   +'<div class="choices"></div>'
   +'<label>为什么这么判？<span class="muted">写你看到了什么</span></label><textarea class="why"></textarea>'
   +'<label>你有没有考虑过「不确定」？考虑了但最后没选，是为什么？</label><textarea class="wn"></textarea>';
  const box=el.querySelector('.choices');
  TYPES.forEach(([v,label])=>{const b=document.createElement('button');b.textContent=label;
   if(a.type===v)b.className='on';
   b.onclick=()=>{a.type=v;save();render();
    document.querySelectorAll('.case')[i].scrollIntoView({block:'nearest'})};
   box.appendChild(b)});
  const why=el.querySelector('.why'),wn=el.querySelector('.wn');
  why.value=a.why||'';wn.value=a.why_not_uncertain||'';
  why.oninput=e=>{a.why=e.target.value;save()};
  wn.oninput=e=>{a.why_not_uncertain=e.target.value;save()};
  root.appendChild(el)});
 save()}
$('#export').onclick=()=>{const out={schema_version:'uncertain_probe_v21',
  answers:CASES.map(c=>({id:c.id,...ans[c.id]}))};
 const a=document.createElement('a');
 a.href=URL.createObjectURL(new Blob([JSON.stringify(out,null,2)],{type:'application/json'}));
 a.download='uncertain_probe_answers.json';a.click()};
render();
</script>
</html>
"""


if __name__ == "__main__":
    main()
