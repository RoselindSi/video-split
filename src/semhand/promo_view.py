"""A page for answering the promoted-track sheet, built from the package itself.

WHY A PAGE AND NOT THE CSV. The sheet asks three questions per track that only
make sense against moving pictures: whose hand it was, whose it became, and
whether those are the same physical hand. Answering that from a spreadsheet
means alt-tabbing between thirty files and two tables, and the slot order --
B1, B2, then A1, A2, A3 across the crossing -- is the one thing that must not
get shuffled while doing it.

IT STAYS BLIND. The package deliberately carries no belief values, no birth
score, no verdict and no reason the track was selected; those are in the key
file and are joined after the answers exist. This renderer reads only
`frames.csv`, `tracks.csv` and the media, so there is nothing for it to leak
even by accident.

IT IS LOCAL AND STAYS LOCAL. The frames show colleagues' hands and a live
workplace. The page references the extracted files by relative path and is
opened from disk; nothing is uploaded anywhere.

ANSWERS ARE EXPORTED AS THE SAME TWO CSVs the sheet expects, so the join back
to the key needs no new code. A draft is kept in the browser so a half-finished
pass survives a closed tab, and it is only a convenience -- the export is the
artefact.
"""
from __future__ import annotations

import argparse
import csv
import json
import os

SLOTS = ["B1", "B2", "A1", "A2", "A3"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkg", required=True, help="解压后的 promopkg 目录")
    ap.add_argument("--out", default=None, help="默认写到 <pkg>/view.html")
    a = ap.parse_args()

    frames = list(csv.DictReader(open(os.path.join(a.pkg, "frames.csv"),
                                      encoding="utf-8")))
    tracks = list(csv.DictReader(open(os.path.join(a.pkg, "tracks.csv"),
                                      encoding="utf-8")))
    by_track = {}
    for f in frames:
        by_track.setdefault((f["rec"], f["tid"]), []).append(f)

    data = []
    for t in tracks:
        fs = sorted(by_track.get((t["rec"], t["tid"]), []),
                    key=lambda f: (SLOTS.index(f["slot"]) if f["slot"] in SLOTS
                                   else 99))
        data.append({
            "rec": t["rec"], "tid": t["tid"],
            "n_before": t["n_before"], "n_after": t["n_after"],
            "clip": "clips/" + t["clip"],
            "clip_frames": t["clip_frames"], "box_drawn": t["box_drawn"],
            "frames": [{"stem": f["stem"], "slot": f["slot"],
                        "phase": f["phase"], "frame": f["frame"], "w": f["w"],
                        "src": "context/%s.jpg" % f["stem"]} for f in fs],
        })

    out = a.out or os.path.join(a.pkg, "view.html")
    open(out, "w", encoding="utf-8").write(PAGE.replace("__DATA__",
                                                        json.dumps(data, ensure_ascii=False)))
    print("-> %s  （%d 条轨迹 / %d 帧）" % (out, len(data), len(frames)))


PAGE = r"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>提升轨迹判读</title>
<style>
:root{--bg:#fbfbfa;--fg:#1a1a18;--mut:#6b6b66;--line:#e3e3df;--card:#fff;
      --b:#3b6ea5;--a:#a5623b;--ok:#2f7d4f}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --bg:#17171a;--fg:#e8e8e4;--mut:#9a9a94;--line:#2e2e33;--card:#1e1e22;
  --b:#7aa8d8;--a:#d89a7a;--ok:#6fc08f}}
:root[data-theme="dark"]{--bg:#17171a;--fg:#e8e8e4;--mut:#9a9a94;--line:#2e2e33;
  --card:#1e1e22;--b:#7aa8d8;--a:#d89a7a;--ok:#6fc08f}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.6 -apple-system,
  "PingFang SC","Helvetica Neue",sans-serif;padding:0 16px 80px}
.wrap{max-width:1100px;margin:0 auto}
h1{font-size:20px;margin:28px 0 6px}
.lede{color:var(--mut);font-size:14px;max-width:70ch;margin:0 0 4px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;
  padding:16px;margin:18px 0}
.hd{display:flex;flex-wrap:wrap;gap:12px;align-items:baseline;
  border-bottom:1px solid var(--line);padding-bottom:10px;margin-bottom:14px}
.hd b{font-size:16px}
.hd span{color:var(--mut);font-size:13px}
.row{display:flex;gap:18px;flex-wrap:wrap}
.clipbox{flex:0 0 320px;max-width:100%}
video{width:100%;border-radius:6px;border:1px solid var(--line);background:#000}
.shots{display:flex;gap:8px;flex:1 1 380px;min-width:0;overflow-x:auto;
  padding-bottom:4px}
.shot{flex:0 0 150px}
.shot img{width:100%;border-radius:5px;border:1px solid var(--line);
  cursor:zoom-in;display:block;background:#000}
.tag{font-size:12px;font-weight:600;margin:6px 0 2px}
.tag.before{color:var(--b)} .tag.after{color:var(--a)}
.meta{font-size:11px;color:var(--mut)}
select{width:100%;margin-top:4px;padding:3px;font-size:12px;
  background:var(--card);color:var(--fg);border:1px solid var(--line);
  border-radius:4px}
select.done{border-color:var(--ok)}
.qs{margin-top:16px;padding-top:14px;border-top:1px solid var(--line);
  display:grid;gap:12px;grid-template-columns:repeat(auto-fit,minmax(200px,1fr))}
.q label{display:block;font-size:13px;font-weight:600;margin-bottom:3px}
.q .hint{font-size:11px;color:var(--mut);margin-bottom:4px}
.q select,.q input{width:100%;padding:5px;font-size:13px;background:var(--card);
  color:var(--fg);border:1px solid var(--line);border-radius:5px}
.bar{position:fixed;left:0;right:0;bottom:0;background:var(--card);
  border-top:1px solid var(--line);padding:10px 16px;display:flex;gap:12px;
  align-items:center;justify-content:center;flex-wrap:wrap}
button{padding:7px 16px;font-size:14px;border-radius:6px;cursor:pointer;
  border:1px solid var(--line);background:var(--card);color:var(--fg)}
button.primary{background:var(--ok);border-color:var(--ok);color:#fff}
#prog{color:var(--mut);font-size:13px}
#lb{position:fixed;inset:0;background:rgba(0,0,0,.9);display:none;
  align-items:center;justify-content:center;z-index:9;cursor:zoom-out}
#lb img{max-width:96vw;max-height:94vh}
@media(max-width:700px){.clipbox{flex:1 1 100%}}
</style></head><body><div class="wrap">
<h1>提升轨迹判读 · 6 条</h1>
<p class="lede">每条轨迹都是先被判成「别人的手」，之后置信度越过 0.70 改判成
「佩戴者自己的手」。五张定点帧按时序排列：<b style="color:var(--b)">B1 B2</b>
在翻转之前，<b style="color:var(--a)">A1 A2 A3</b> 在翻转之后。绿框画在该轨迹
出现的每一帧；没有框的帧是这条轨迹当时不存在。</p>
<p class="lede">包里没有模型分数、判定或挑中理由。第三问单独回答：前后都是
「别人的」也不等于同一只手，可能是从一个人的手串到了另一个人的手。</p>
<div id="app"></div></div>
<div class="bar"><span id="prog"></span>
<button onclick="dl('frames')">导出 frames.csv</button>
<button class="primary" onclick="dl('tracks')">导出 tracks.csv</button>
<button onclick="if(confirm('清空所有已填答案？'))clr()">清空</button></div>
<div id="lb" onclick="this.style.display='none'"><img id="lbi" alt=""></div>
<script>
const D=__DATA__;
const KEY='promopkg_v1';
let S={};
try{S=JSON.parse(localStorage.getItem(KEY)||'{}')}catch(e){S={}}
function save(){try{localStorage.setItem(KEY,JSON.stringify(S))}catch(e){}}
function clr(){S={};save();location.reload()}
const LAB=[['','—'],['owner','owner 自己的'],['other','other 别人的'],
           ['nothand','nothand 不是手'],['uncertain','uncertain 看不清']];
const OWN=[['','—'],['owner','owner 佩戴者自己的'],['other','other 别人的'],
           ['nothand','nothand 不是手'],['uncertain','uncertain 看不清']];
const SAME=[['','—'],['yes','yes 同一只'],['no','no 不是同一只'],
            ['uncertain','uncertain 看不清']];
const opts=(a,v)=>a.map(([k,t])=>
  `<option value="${k}"${k===v?' selected':''}>${t}</option>`).join('');
function set(k,v){S[k]=v;save();prog();}
function prog(){
  const nf=D.reduce((s,t)=>s+t.frames.length,0);
  const df=D.reduce((s,t)=>s+t.frames.filter(f=>S['f:'+f.stem]).length,0);
  const dt=D.filter(t=>['ownership_before','ownership_after',
    'same_physical_hand'].every(c=>S[`t:${t.rec}:${t.tid}:${c}`])).length;
  document.getElementById('prog').textContent=
    `帧 ${df}/${nf} · 轨迹三问 ${dt}/${D.length}`;
  document.querySelectorAll('select[data-k]').forEach(s=>
    s.classList.toggle('done',!!s.value));
}
document.getElementById('app').innerHTML=D.map(t=>{
  const id=`${t.rec}:${t.tid}`;
  const shots=t.frames.map(f=>`<div class="shot">
    <div class="tag ${f.phase}">${f.slot}</div>
    <img src="${f.src}" alt="${f.slot}" loading="lazy"
         onclick="lb(this.src)">
    <div class="meta">帧 ${f.frame} · 框宽 ${f.w}px</div>
    <select data-k="f:${f.stem}" onchange="set(this.dataset.k,this.value)">
      ${opts(LAB,S['f:'+f.stem]||'')}</select></div>`).join('');
  const q=(c,lab,hint,arr)=>`<div class="q"><label>${lab}</label>
    <div class="hint">${hint}</div>
    <select data-k="t:${id}:${c}" onchange="set(this.dataset.k,this.value)">
      ${opts(arr,S[`t:${id}:${c}`]||'')}</select></div>`;
  return `<div class="card"><div class="hd">
    <b>${t.rec} · tid ${t.tid}</b>
    <span>翻转前 ${t.n_before} 帧 → 翻转后 ${t.n_after} 帧</span>
    <span>小片 ${t.clip_frames} 帧，其中 ${t.box_drawn} 帧有框</span></div>
    <div class="row"><div class="clipbox">
      <video src="${t.clip}" controls loop muted playsinline preload="metadata"></video>
      <div class="meta" style="margin-top:4px">慢放；没有框的帧是轨迹不存在</div>
    </div><div class="shots">${shots}</div></div>
    <div class="qs">
      ${q('ownership_before','B 段那只手是谁的','B1 B2，翻转之前',OWN)}
      ${q('ownership_after','A 段那只手是谁的','A1 A2 A3，翻转之后',OWN)}
      ${q('same_physical_hand','前后是不是同一只物理的手',
          '前后都填 other 也不等于同一只',SAME)}
    </div>
    <div class="q" style="margin-top:10px"><label>凭什么判的</label>
      <div class="hint">视觉连续性 / 左右手 / 手套袖子前臂 / 遮挡重现</div>
      <input data-k2="t:${id}:evidence" value="${(S[`t:${id}:evidence`]||'')
        .replace(/"/g,'&quot;')}"
        oninput="set(this.dataset.k2,this.value)"></div></div>`;
}).join('');
function lb(s){const b=document.getElementById('lb');
  document.getElementById('lbi').src=s;b.style.display='flex';}
function csv(rows){return rows.map(r=>r.map(x=>{
  x=(x==null?'':String(x));
  return /[",\n]/.test(x)?'"'+x.replace(/"/g,'""')+'"':x;}).join(',')).join('\n');}
function dl(which){
  let rows;
  if(which==='frames'){
    rows=[['stem','rec','tid','slot','phase','frame','w','label']];
    D.forEach(t=>t.frames.forEach(f=>rows.push(
      [f.stem,t.rec,t.tid,f.slot,f.phase,f.frame,f.w,S['f:'+f.stem]||''])));
  }else{
    rows=[['rec','tid','n_before','n_after','ownership_before',
           'ownership_after','same_physical_hand','evidence']];
    D.forEach(t=>{const id=`${t.rec}:${t.tid}`;rows.push(
      [t.rec,t.tid,t.n_before,t.n_after,S[`t:${id}:ownership_before`]||'',
       S[`t:${id}:ownership_after`]||'',S[`t:${id}:same_physical_hand`]||'',
       S[`t:${id}:evidence`]||'']);});
  }
  const b=new Blob(['﻿'+csv(rows)],{type:'text/csv;charset=utf-8'});
  const a=document.createElement('a');
  a.href=URL.createObjectURL(b);a.download=which+'_filled.csv';a.click();
}
prog();
</script></body></html>
"""


if __name__ == "__main__":
    main()
