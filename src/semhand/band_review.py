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


PAGE = r"""<meta charset=utf-8><title>小框判读 60-100px</title><style>
body{font:13px/1.5 system-ui;margin:0;background:#111;color:#ddd}
#bar{position:sticky;top:0;background:#181818;padding:9px 14px;z-index:9;
  border-bottom:1px solid #333;display:flex;gap:14px;align-items:center;
  flex-wrap:wrap}
button{font:13px system-ui;padding:5px 10px;cursor:pointer}
b{color:#ffd33d}
.t{padding:9px 14px;border-bottom:1px solid #262626;display:flex;gap:12px;
  align-items:flex-start}
.t.cur{background:#1d2430;outline:2px solid #4a8}
.t.hand{border-left:5px solid #2a6}
.t.nothand{border-left:5px solid #d33}
.t.unsure{border-left:5px solid #777}
.meta{min-width:120px;color:#9ab;font-size:12px}
.num{color:#8ab4c8;font-family:ui-monospace,monospace;font-size:11px}
.shots{display:flex;gap:8px;flex:1;align-items:flex-start;min-width:0}
.shots figure{margin:0}
.shots img{display:block;border-radius:3px;cursor:zoom-in}
.wide img{width:min(58vw,640px)}
.crop img{width:170px}
.shots figcaption{font-size:10px;color:#888;text-align:center}
.note{background:#151515;color:#ccc;border:1px solid #333;border-radius:3px;
  padding:4px;font:12px system-ui;width:150px;margin-top:5px}
.key{font-size:11px;color:#999}
#lb{position:fixed;inset:0;background:rgba(0,0,0,.93);display:none;
  align-items:center;justify-content:center;z-index:20;cursor:zoom-out}
#lb img{max-width:96vw;max-height:94vh}
</style>
<div id=bar><span id=prog></span>
<span><b>1</b> 是手 &nbsp; <b>2</b> 不是手 &nbsp; <b>3</b> 看不清
 &nbsp; <b>&uarr;&darr;</b> 换 &nbsp; <b>u</b> undo</span>
<button onclick="dl()">导出 CSV</button>
<span class=key>只问一件事：绿框里是不是一只手。不问是谁的。看不清就按 3，不要猜。
这些框都在 60-100px，而别人的手中位就是 87px。</span></div>
<div id=list></div>
<div id=lb onclick="this.style.display='none'"><img id=lbi alt=""></div>
<script>
const D = __ITEMS__;
const KEY = "bandreview:" + __HASH__;
const V = ["hand", "nothand", "unsure"];
const CN = {hand:"是手", nothand:"不是手", unsure:"看不清"};
const CL = {hand:"#5c5", nothand:"#d55", unsure:"#999"};
let lab = {}, note = {}, cur = 0, hist = [];
try { const o = JSON.parse(localStorage.getItem(KEY) || "{}");
      lab = o.lab || {}; note = o.note || {}; } catch(e) { lab={}; note={}; }
const list = document.getElementById("list");
D.forEach((d, i) => {
  const e = document.createElement("div");
  e.className = "t"; e.id = "t" + i;
  e.innerHTML = '<div class=meta>' + (i+1) + ' / ' + D.length +
    ' <span id=v' + i + '></span><br><span class=num>框宽 ' +
    Math.round(d.w_px) + 'px</span>' +
    '<br><input class=note id=n' + i + ' placeholder="备注（可空）">' +
    '</div><div class=shots>' +
    '<figure class=wide><img src="img/' + d.stem + '_full.jpg" loading=lazy>' +
    '<figcaption>整帧（绿框=要判的目标）</figcaption></figure>' +
    '<figure class=crop><img src="img/' + d.stem + '_crop.jpg" loading=lazy>' +
    '<figcaption>裁剪</figcaption></figure></div>';
  e.onclick = ev => { if(ev.target.tagName!=="INPUT"){ cur = i; draw(); } };
  list.appendChild(e);
});
// 图片点一下放大；放大不改变当前条，所以放在冒泡之前拦掉
list.querySelectorAll("img").forEach(im => im.onclick = ev => {
  ev.stopPropagation();
  document.getElementById("lbi").src = im.src;
  document.getElementById("lb").style.display = "flex";
});
D.forEach((d,i) => {
  const n = document.getElementById("n"+i);
  n.value = note[d.stem] || "";
  n.oninput = () => { note[d.stem] = n.value; persist(); };
});
function persist(){
  try{ localStorage.setItem(KEY, JSON.stringify({lab:lab, note:note})); }catch(e){}
}
function draw(){
  D.forEach((d,i) => {
    const v = lab[d.stem];
    document.getElementById("t"+i).className = "t " + (v||"") + (i===cur?" cur":"");
    document.getElementById("v"+i).innerHTML = v ?
      '<b style="color:'+CL[v]+'">'+CN[v]+'</b>' : '';
  });
  const n = D.filter(d => lab[d.stem]).length;
  document.getElementById("prog").innerHTML = "<b>" + n + "/" + D.length + "</b> 已判";
  persist();
  const el = document.getElementById("t"+cur);
  if(el) el.scrollIntoView({block:"nearest"});
}
document.onkeydown = ev => {
  // 在备注框里打字时不能吃掉按键，否则写个 "1" 就跳条了
  const t = ev.target.tagName;
  if(t === "INPUT" || t === "TEXTAREA") return;
  if(document.getElementById("lb").style.display === "flex" && ev.key === "Escape"){
    document.getElementById("lb").style.display = "none"; ev.preventDefault(); return;
  }
  if(ev.key >= "1" && ev.key <= "3"){
    const d = D[cur]; if(!d) return;
    hist.push([d.stem, lab[d.stem]]);
    lab[d.stem] = V[+ev.key - 1];
    cur = Math.min(cur + 1, D.length - 1); draw();
  }
  else if(ev.key === "ArrowDown"){ cur = Math.min(cur+1, D.length-1); draw(); }
  else if(ev.key === "ArrowUp"){ cur = Math.max(cur-1, 0); draw(); }
  else if(ev.key === "u"){ const h = hist.pop(); if(h){
    if(h[1] === undefined) delete lab[h[0]]; else lab[h[0]] = h[1]; draw(); } }
  else return;
  ev.preventDefault();
};
draw();
function dl(){
  let s = "stem,label,note,w_px\n";
  for(const d of D){
    const nt = (note[d.stem]||"").replace(/"/g,'""');
    s += d.stem + "," + (lab[d.stem]||"") + "," +
         (/[",]/.test(nt) ? '"'+nt+'"' : nt) + "," + d.w_px + "\n";
  }
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob(["\ufeff"+s],{type:"text/csv;charset=utf-8"}));
  a.download = "band_review_filled.csv"; a.click();
}
</script>
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
        "2. 键盘操作：1=是手  2=不是手  3=看不清  上下箭头=换一条  u=撤销。\n"
        "   按 1/2/3 会自动跳到下一条，不用点鼠标。\n"
        "3. 只判一件事：绿框里是不是一只手。是谁的手不重要。\n"
        "4. 看不清就按 3，不要猜——猜出来的答案没法用。\n"
        "5. 填完点「导出 CSV」，把导出的文件发回。\n\n"
        "答案不在这个包里。这一带（60-100px）模型的判定从没被人工核对过，\n"
        "而别人的手中位就是 87px，正落在这里。\n" % len(public))
    print("-> %s（%d 条，%d 张图）" % (a.out, len(public), 2 * len(public)))
    print("   组别与教师分数都不在包内")


if __name__ == "__main__":
    main()
