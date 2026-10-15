"""The E1a audit as a web page: one clip, Qwen's five claims, six human verdicts.

THE AUDITOR NEVER SEES THE v1 DESCRIPTION. Showing another annotator's words
next to the model's would turn "is this claim supported by the footage" into
"does the model agree with v1", which is a weaker question and a different
one. The v1 text stays in the key file and is joined back afterwards.

THE TWO SAMPLES ARE SEPARATE TABS AND ARE NEVER SUMMED. 80 representative,
drawn stratified under a frozen seed, is the one that estimates a rate. 20
challenge, chosen to cover the visual conditions E0 exposed, is enriched by
construction; pooling them would report a number for a population that does
not exist. The page shows two progress counters for the same reason.

THE VERDICT VOCABULARIES ARE CLOSED, and `result_status` carries the point of
the whole experiment:

  correct         claims a result, and the footage shows it
  unsupported     claims a result the footage does not show -- the
                  "selected a suitable motorcycle" failure
  missing         says null, but a result was in fact visible
  not_observable  says null, and nothing visible concluded -- correct

`not_observable` is a right answer, so it stays out of the precision
denominator. Otherwise a model could buy precision by answering null more
often, and only `missing` would push back.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import zipfile

CLIP_DIR = "/workspace/e1a_clips"


def clip_name(seg_id):
    return seg_id.replace("#", "_s") + ".mp4"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--clips", default=CLIP_DIR)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    s = json.load(open(a.sample, encoding="utf-8"))
    pred = {}
    for line in open(a.predictions, encoding="utf-8"):
        d = json.loads(line)
        pred[d["segment_id"]] = d
    pop = {r["segment_id"]: r for r in s["population"]}
    challenge = {c["segment_id"]: c["challenge"] for c in s["challenge"]}

    out = os.path.abspath(a.out)
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(os.path.join(out, "clips"))

    items, key = [], []
    for arm, ids in (("representative", s["representative"]),
                     ("challenge", [c["segment_id"] for c in s["challenge"]])):
        for i, sid in enumerate(ids, 1):
            d = pred.get(sid)
            if d is None or not d.get("parsed"):
                print("  跳过（无解析结果）%s" % sid)
                continue
            src = os.path.join(a.clips, clip_name(sid))
            if not os.path.isfile(src):
                print("  跳过（无片段）%s" % sid)
                continue
            name = "%s_%03d.mp4" % (arm[:4], i)
            shutil.copy2(src, os.path.join(out, "clips", name))
            p = d["parsed"]
            items.append({
                "id": "%s_%03d" % (arm[:4], i), "arm": arm,
                "clip": "clips/" + name,
                "seconds": round(d["duration_s"], 1),
                "object": p.get("object"),
                "observed_action": p.get("observed_action"),
                "observed_result": p.get("observed_result"),
                "uncertain": bool(p.get("uncertain")),
                "uncertainty_reason": p.get("uncertainty_reason"),
            })
            key.append({"id": "%s_%03d" % (arm[:4], i), "arm": arm,
                        "segment_id": sid, "video_id": pop[sid]["video_id"],
                        "batch": pop[sid]["batch"],
                        "start_s": pop[sid]["start_s"],
                        "end_s": pop[sid]["end_s"],
                        "duration_s": pop[sid]["duration_s"],
                        "dur_tertile": pop[sid].get("dur_tertile"),
                        "challenge": challenge.get(sid, ""),
                        "v1_description": pop[sid]["v1_description"]})

    page = PAGE.replace("__ITEMS__", json.dumps(items, ensure_ascii=False))
    open(os.path.join(out, "打开核验页面.html"), "w", encoding="utf-8").write(page)
    json.dump({"frozen_sample": os.path.basename(a.sample), "items": key},
              open(out.rstrip("/") + "_key.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    open(os.path.join(out, "README_先看这里.txt"), "w", encoding="utf-8").write(
        "E1a 人工核验：100 段，每段看片子、判模型说得对不对。\n\n"
        "用 Chrome 或 Edge 打开「打开核验页面.html」。\n\n"
        "只判一件事：模型写的东西，这个片段的画面**支不支持**。\n"
        "不是判它写得好不好听，也不是判它和别人写的一不一样。\n\n"
        "最关键的一栏是「看到的结果」：\n"
        "  模型说有结果、画面里也看得见   -> 属实\n"
        "  模型说有结果、画面里看不见     -> 没有依据   （这一栏是本次实验的重点）\n"
        "  模型写了 null、但其实看得见   -> 漏了\n"
        "  模型写了 null、确实看不见     -> 本来就没有（这是判对）\n\n"
        "两组分开：代表性 80 段、困难 20 段，进度也分开显示。\n"
        "填完点「导出结果」，把导出的 JSON 发回。\n")
    with zipfile.ZipFile(out + ".zip", "w", zipfile.ZIP_STORED) as z:
        for root, _, files in os.walk(out):
            for f in sorted(files):
                p = os.path.join(root, f)
                z.write(p, os.path.join(os.path.basename(out),
                                        os.path.relpath(p, out)))
    n = {k: sum(1 for x in items if x["arm"] == k)
         for k in ("representative", "challenge")}
    print("-> %s  代表性 %d / 困难 %d" % (out, n["representative"], n["challenge"]))
    print("   zip %.0f MB  页面 sha256 %s"
          % (os.path.getsize(out + ".zip") / 1048576,
             hashlib.sha256(open(os.path.join(out, "打开核验页面.html"),
                                 "rb").read()).hexdigest()[:16]))


PAGE = """<!doctype html>
<html lang="zh-CN">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>E1a 核验</title>
<style>
:root{color-scheme:dark;--bg:#111513;--panel:#191e1b;--line:#46504a;--text:#f3f5f3;--muted:#b5bdb8;--accent:#72cdaa;--accent2:#2f7558;--warn:#efbd69;--danger:#ef9187}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:16px/1.6 system-ui,sans-serif}
header{position:sticky;top:0;z-index:10;display:flex;gap:14px;align-items:center;padding:10px 18px;background:var(--panel);border-bottom:1px solid var(--line);flex-wrap:wrap}
header h1{font-size:17px;margin:0}
.tab{min-height:40px;padding:6px 14px;background:#252c28;border:1px solid #68736d;border-radius:4px;color:var(--text);font:inherit;cursor:pointer}
.tab.on{background:var(--accent2);border-color:var(--accent);font-weight:700}
main{max-width:940px;margin:auto;padding:18px}
.rule{border-left:4px solid var(--accent);background:#1d2722;padding:12px 15px;margin-bottom:18px}
.rule b{display:block;margin-bottom:4px}
.item{border:1px solid var(--line);background:var(--panel);padding:16px;margin-bottom:22px}
.item h2{font-size:16px;margin:0 0 10px}
video{width:100%;max-height:56vh;background:#000;display:block}
.claims{margin:12px 0;border-left:3px solid var(--warn);background:#24221b;padding:10px 12px}
.claims div{display:grid;grid-template-columns:120px 1fr;gap:8px;padding:2px 0}
.claims span:first-child{color:var(--muted)}
.null{color:var(--muted);font-style:italic}
fieldset{border:1px solid #3a423d;margin:12px 0 0;padding:10px 12px}
legend{color:var(--muted);font-size:14px;padding:0 6px}
.opts{display:flex;gap:8px;flex-wrap:wrap}
.opts button{min-height:44px;padding:8px 12px;background:#252c28;border:1px solid #68736d;border-radius:4px;color:var(--text);font:inherit;cursor:pointer}
.opts button.on{background:var(--accent2);border-color:var(--accent);font-weight:700}
.opts button.bad.on{background:#7a3b34;border-color:var(--danger)}
textarea{width:100%;min-height:46px;margin-top:8px;padding:8px;background:#252c28;color:var(--text);border:1px solid #68736d;border-radius:4px;font:inherit;resize:vertical}
label.small{display:block;color:var(--muted);font-size:14px;margin-top:8px}
footer{position:sticky;bottom:0;background:var(--bg);border-top:1px solid var(--line);padding:12px 0;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
footer button{min-height:46px;padding:10px 18px;background:var(--accent2);border:1px solid var(--accent);border-radius:4px;color:var(--text);font:inherit;cursor:pointer}
footer button:disabled{opacity:.45;cursor:default}
.muted{color:var(--muted)}
@media(max-width:600px){.claims div{grid-template-columns:1fr}}
</style>
<header><h1>E1a 核验</h1>
<button class="tab on" data-arm="representative">代表性 <span id="p-representative"></span></button>
<button class="tab" data-arm="challenge">困难 <span id="p-challenge"></span></button>
</header>
<main>
<section class="rule"><b>只判一件事：模型写的东西，这个片段的画面支不支持。</b>
<span class="muted">不是判它写得好不好听，也不是判它和别人写的一不一样。黄框里是模型说的，先看完片子再看它。</span></section>
<div id="list"></div>
<footer><button id="export" disabled>导出结果</button><span class="muted" id="progress"></span></footer>
</main>
<script>
const ITEMS=__ITEMS__;
const KEY='e1a-audit-v1';
let ans={};try{ans=JSON.parse(localStorage.getItem(KEY))||{}}catch{}
let arm='representative';
const $=s=>document.querySelector(s);
const FIELDS=[
 ["object_status","物体判得对吗",[["correct","对"],["partial","部分对"],["incorrect","错",1],["unobservable","画面里看不清"]]],
 ["action_status","动作判得对吗",[["correct","对"],["partial","部分对"],["incorrect","错",1],["unobservable","画面里看不清"]]],
 ["result_status","「看到的结果」这一栏",[["correct","属实"],["unsupported","没有依据",1],["missing","漏了（其实看得见）",1],["not_observable","本来就没有（写 null 是对的）"]]],
 ["uncertainty_status","它该不该说不确定",[["appropriate","处理得当"],["over_abstain","过度弃权",1],["under_abstain","该弃权没弃权",1],["not_applicable","不适用"]]],
 ["unsupported_inference","有没有出现臆测（目的/结果）",[["no","没有"],["yes","有",1]]],
 ["overall_usable","这份语义够不够给下一步用",[["yes","够用"],["no","不够",1]]],
];
const TEXTS=[["corrected_object","物体应该是"],["corrected_action","动作应该是"],
             ["corrected_result","结果应该是（没有就留空）"],["review_note","备注"]];
function done(it){const a=ans[it.id]||{};return FIELDS.every(f=>a[f[0]])}
function save(){try{localStorage.setItem(KEY,JSON.stringify(ans))}catch{}
 for(const k of ['representative','challenge']){
  const xs=ITEMS.filter(i=>i.arm===k);
  $('#p-'+k).textContent=xs.filter(done).length+'/'+xs.length}
 const all=ITEMS.filter(done).length;
 $('#progress').textContent='已填 '+all+' / '+ITEMS.length+(all===ITEMS.length?'':'（两组都填完才能导出）');
 $('#export').disabled=all!==ITEMS.length}
function esc(x){return String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
function claim(v){return (v===null||v===undefined||v==='')?'<span class="null">null</span>':esc(v)}
function render(){const root=$('#list');root.innerHTML='';
 const xs=ITEMS.filter(i=>i.arm===arm);
 xs.forEach((it,idx)=>{const a=ans[it.id]||(ans[it.id]={});
  const el=document.createElement('section');el.className='item';
  el.innerHTML='<h2>'+(idx+1)+' / '+xs.length+' <span class="muted">'+it.seconds+' 秒</span></h2>'
   +'<video controls preload="none" src="'+it.clip+'"></video>'
   +'<div class="claims">'
   +'<div><span>物体</span><span>'+claim(it.object)+'</span></div>'
   +'<div><span>看到的动作</span><span>'+claim(it.observed_action)+'</span></div>'
   +'<div><span>看到的结果</span><span>'+claim(it.observed_result)+'</span></div>'
   +'<div><span>不确定</span><span>'+(it.uncertain?'true'+(it.uncertainty_reason?'：'+esc(it.uncertainty_reason):''):'false')+'</span></div>'
   +'</div>';
  FIELDS.forEach(([k,label,opts])=>{
   const fs=document.createElement('fieldset');
   fs.innerHTML='<legend>'+label+'</legend><div class="opts"></div>';
   const box=fs.querySelector('.opts');
   opts.forEach(([v,text,bad])=>{const b=document.createElement('button');
    b.textContent=text;b.className=(bad?'bad ':'')+(a[k]===v?'on':'');
    b.onclick=()=>{a[k]=v;save();
      [...box.children].forEach((c,ci)=>c.className=(opts[ci][2]?'bad ':'')+(opts[ci][0]===v?'on':''))};
    box.appendChild(b)});
   el.appendChild(fs)});
  TEXTS.forEach(([k,label])=>{const l=document.createElement('label');l.className='small';l.textContent=label;
   const t=document.createElement('textarea');t.value=a[k]||'';
   t.oninput=e=>{a[k]=e.target.value;save()};
   el.appendChild(l);el.appendChild(t)});
  root.appendChild(el)});
 save()}
document.querySelectorAll('.tab').forEach(b=>b.onclick=()=>{
 arm=b.dataset.arm;
 document.querySelectorAll('.tab').forEach(x=>x.classList.toggle('on',x===b));
 render();scrollTo({top:0})});
$('#export').onclick=()=>{
 const out={schema_version:'e1a_audit_v1',
  answers:ITEMS.map(i=>({id:i.id,arm:i.arm,...ans[i.id]}))};
 const a=document.createElement('a');
 a.href=URL.createObjectURL(new Blob([JSON.stringify(out,null,2)],{type:'application/json'}));
 a.download='e1a_audit_answers.json';a.click()};
render();
</script>
</html>
"""


if __name__ == "__main__":
    main()
