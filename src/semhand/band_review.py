"""A blind sheet for the 60-100px band, where the teacher has never been checked.

WHY THIS BAND AND WHY NOW. Other people's hands are a median 87px wide, so the
60-100px band is where the whole hand-filter either works or does not, and it
is the band the first harvest deliberately avoided: the 150px floor came from
the probe scoring 3 of 55 at 23-28px, and that was extrapolated across
everything below 150 without measuring what lies between. Measured against the
508 human labels already in the manifest, the teacher's AUC in this band is
0.920 -- the best of any band. But AUC only says the ranking is right; it says
nothing about whether the 0.5 cut falls in the right place, and the whole
harvest depends on that cut.

TWO GROUPS, ANSWERING TWO DIFFERENT QUESTIONS:

    A  40 boxes the teacher called not-a-hand. How many really are? This is
       the precision of the verdict that removes things from the pipeline.
    B  40 boxes the teacher called a hand, sampled at random from all 6,273
       of them. How many are actually not hands? This is the rate the harvest
       depends on, and random is the only sampling that estimates it --
       taking the lowest-scoring ones would answer a different question and
       overstate the problem.

THE TWO GROUPS ARE SHUFFLED TOGETHER and no score is shown, so the sheet
cannot be answered by reading the model instead of the picture. The key stays
out of the package.

THE FULL FRAME ALREADY CARRIES THE BOX, drawn green by the harvest, because a
60-100px crop on its own is often unreadable -- which is exactly the property
that makes this band hard and worth checking.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil


PAGE = r"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>小框判读 · 60-100px</title>
<style>
:root{--bg:#fbfbfa;--fg:#1a1a18;--mut:#6b6b66;--line:#e3e3df;--card:#fff;--ok:#2f7d4f}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --bg:#17171a;--fg:#e8e8e4;--mut:#9a9a94;--line:#2e2e33;--card:#1e1e22;--ok:#6fc08f}}
:root[data-theme="dark"]{--bg:#17171a;--fg:#e8e8e4;--mut:#9a9a94;--line:#2e2e33;
  --card:#1e1e22;--ok:#6fc08f}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);padding:0 16px 80px;
  font:15px/1.6 -apple-system,"PingFang SC","Helvetica Neue",sans-serif}
.wrap{max-width:1100px;margin:0 auto}
h1{font-size:19px;margin:24px 0 6px}
.lede{color:var(--mut);font-size:14px;max-width:70ch}
/* 网格而不是 flex：图片宽度是固定的，不会收缩，用 flex 时会溢出容器
   压到答案栏底下（第一版就是这样）。网格按列分配，溢出不了。 */
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;
  padding:12px;margin:14px 0;display:grid;gap:14px;align-items:start;
  grid-template-columns:2.2em minmax(0,1fr) 160px 190px}
@media(max-width:860px){.card{grid-template-columns:2.2em minmax(0,1fr) 120px;}
  .card .ans{grid-column:2/-1}}
.card .n{font-weight:600}
.card figure{margin:0;min-width:0}
.card img{border:1px solid var(--line);border-radius:6px;cursor:zoom-in;
  display:block;background:#000;width:100%;height:auto}
figcaption{font-size:11px;color:var(--mut);margin-top:3px}
.ans{flex:0 0 190px}
select{width:100%;padding:6px;font-size:14px;background:var(--card);color:var(--fg);
  border:1px solid var(--line);border-radius:6px}
select.done{border-color:var(--ok)}
input.note{width:100%;margin-top:6px;padding:5px;font-size:12px;background:var(--card);
  color:var(--fg);border:1px solid var(--line);border-radius:5px}
.meta{font-size:11px;color:var(--mut);margin-top:6px}
.bar{position:fixed;left:0;right:0;bottom:0;background:var(--card);
  border-top:1px solid var(--line);padding:10px 16px;display:flex;gap:12px;
  align-items:center;justify-content:center;flex-wrap:wrap}
button{padding:7px 16px;font-size:14px;border-radius:6px;cursor:pointer;
  border:1px solid var(--line);background:var(--card);color:var(--fg)}
button.primary{background:var(--ok);border-color:var(--ok);color:#fff}
#lb{position:fixed;inset:0;background:rgba(0,0,0,.92);display:none;
  align-items:center;justify-content:center;z-index:9;cursor:zoom-out}
#lb img{max-width:96vw;max-height:94vh}
</style></head><body><div class="wrap">
<h1>小框判读 · 60–100px · __N__ 个</h1>
<p class="lede">绿框里的东西<b>是不是一只手</b>。是谁的手不重要，看不清就选「看不清」，
不要猜。整帧给的是位置，裁剪图给的是细节——这一带的框本来就小，两张要一起看。</p>
<p class="lede">这些框都在 60–100px 之间，而别人的手中位就是 87px。模型在这一带的判定
从来没有被人工核对过，这张表就是来核对它的。</p>
<div id="app"></div></div>
<div class="bar"><span id="prog"></span>
<button class="primary" onclick="dl()">导出 CSV</button>
<button onclick="if(confirm('清空所有答案？'))clr()">清空</button></div>
<div id="lb" onclick="this.style.display='none'"><img id="lbi" alt=""></div>
<script>
const D=__ITEMS__, KEY="bandreview_"+__HASH__;
let S={}; try{S=JSON.parse(localStorage.getItem(KEY)||"{}")}catch(e){S={}}
const $=s=>document.querySelector(s);
function save(){try{localStorage.setItem(KEY,JSON.stringify(S))}catch(e){}}
function clr(){S={};save();location.reload()}
const OPT=[["",""],["hand","手"],["nothand","不是手"],["unsure","看不清"]];
function prog(){
  const n=D.filter(d=>S["v:"+d.stem]).length;
  $("#prog").textContent=n+" / "+D.length;
  document.querySelectorAll("select").forEach(s=>s.classList.toggle("done",!!s.value));
}
function set(k,v){S[k]=v;save();prog()}
$("#app").innerHTML=D.map((d,i)=>`<div class="card">
  <div class="n">${i+1}</div>
  <figure class="full"><img src="img/${d.stem}_full.jpg" loading="lazy"
      onclick="lb(this.src)"><figcaption>整帧（绿框=要判的目标）</figcaption></figure>
  <figure class="crop"><img src="img/${d.stem}_crop.jpg" loading="lazy"
      onclick="lb(this.src)"><figcaption>裁剪</figcaption></figure>
  <div class="ans">
    <select data-k="v:${d.stem}" onchange="set(this.dataset.k,this.value)">
      ${OPT.map(([v,t])=>`<option value="${v}"${S["v:"+d.stem]===v?" selected":""}>${t||"—"}</option>`).join("")}
    </select>
    <input class="note" placeholder="备注（可空）" value="${(S["n:"+d.stem]||"").replace(/"/g,"&quot;")}"
      oninput="set('n:${d.stem}',this.value)">
    <div class="meta">框宽 ${Math.round(d.w_px)}px</div>
  </div></div>`).join("");
function lb(s){$("#lbi").src=s;$("#lb").style.display="flex"}
function dl(){
  const rows=[["stem","label","note","w_px"]];
  D.forEach(d=>rows.push([d.stem,S["v:"+d.stem]||"",S["n:"+d.stem]||"",d.w_px]));
  const csv=rows.map(r=>r.map(x=>{x=String(x??"");
    return /[",\n]/.test(x)?'"'+x.replace(/"/g,'""')+'"':x}).join(",")).join("\n");
  const b=new Blob(["﻿"+csv],{type:"text/csv;charset=utf-8"});
  const a=document.createElement("a");a.href=URL.createObjectURL(b);
  a.download="band_review_filled.csv";a.click();
}
prog();
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", required=True, help="抽好的 items.json（盲，无分数）")
    ap.add_argument("--views", required=True, help="neghar views 目录")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    items = json.load(open(a.items, encoding="utf-8"))
    img = os.path.join(a.out, "img")
    os.makedirs(img, exist_ok=True)
    missing = []
    for it in items:
        for kind in ("full", "crop"):
            src = os.path.join(a.views, "%s_%s.jpg" % (it["stem"], kind))
            if not os.path.exists(src):
                missing.append(src)
                continue
            shutil.copy2(src, os.path.join(img, os.path.basename(src)))
    if missing:
        raise SystemExit("缺图 %d 张，第一张 %s" % (len(missing), missing[0]))

    # 包里不带分数，也不带组别：组别本身就会泄漏答案
    public = [{"stem": it["stem"], "w_px": round(float(it["w_px"]), 1)}
              for it in items]
    h = hashlib.sha256(json.dumps(public, sort_keys=True).encode()).hexdigest()[:12]
    page = (PAGE.replace("__ITEMS__", json.dumps(public, ensure_ascii=False))
                .replace("__N__", str(len(public)))
                .replace("__HASH__", json.dumps(h)))
    open(os.path.join(a.out, "打开判读页面.html"), "w", encoding="utf-8").write(page)
    open(os.path.join(a.out, "README.txt"), "w", encoding="utf-8").write(
        "小框判读 · 60-100px · %d 个\n\n"
        "1. 整个文件夹解压出来，用 Chrome 或 Edge 打开「打开判读页面.html」。\n"
        "2. 只判一件事：绿框里是不是一只手。是谁的手不重要。\n"
        "3. 看不清就选「看不清」，不要猜——猜出来的答案没法用。\n"
        "4. 填完点「导出 CSV」，把导出的文件发回。\n\n"
        "答案不在这个包里。这一带（60-100px）模型的判定从没被人工核对过，\n"
        "而别人的手中位就是 87px，正落在这里。\n" % len(public))
    print("-> %s（%d 条，%d 张图）" % (a.out, len(public), 2 * len(public)))
    print("   组别与教师分数都不在包内")


if __name__ == "__main__":
    main()
