"""Build the R1 annotation page: indicate A and B on sampled frames.

WHAT THE PAGE IS FOR. The gold has to answer one question -- is each of the
wearer's own hands held by one continuous track -- and no existing gold can,
because they are all keyed on rec:tid, the tracker's own output. So the
annotator indicates, every `stride` frames, where hand A is and where hand B
is. Visible intervals and anchors both fall out of that single pass, so anchor
density equals the stride by construction rather than by the annotator
remembering to place them.

THE CANDIDATE BOXES CARRY NO LABELS, AND THAT IS THE WHOLE DESIGN. They come
from the pipeline's own detections, but tid, own and side are stripped before
they reach the page. What this gold measures is whether two pieces are the same
hand; showing the pipeline's grouping would be marking its own exam. Drawing is
always available, because a detector miss must not become a hole in the gold --
that miss is one of the defects being measured.

THE DELIVERED VIDEO CANNOT BE USED, WHICH IS WHY THIS TAKES FRAMES. `video/` in
a pipeline run is a debug render: it already has the verdicts drawn on it, with
labels like "self left own 1.00 det 0.64" and a header counting self and other,
and it is a two-panel 1920x3108 composite whose coordinates do not match the
boxes. Clean cam3 frames at 1920x1520 do match, one to one, by absolute frame
number. Verified by drawing frame 494's boxes on frame 494's image and looking
at it -- the first attempt drew frame 400's boxes and landed on shelving, which
is the kind of error that would otherwise have shipped.

UNCERTAIN IS A FIRST-CLASS ANSWER. A frame where the annotator cannot tell is
dropped from the denominator and counted, because its count is the boundary of
what the result can claim. Folding it into "not visible" would hide that.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os

PAGE = """<meta charset="utf-8">
<title>%(title)s</title>
<style>
:root{--bg:#fbfbfa;--fg:#16181d;--dim:#5b6270;--line:#d9dce3;--a:#2f5d8c;--b:#a4453f;
      --ok:#4a7a52;--warn:#8a4b2a}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --bg:#16181d;--fg:#e9ecf1;--dim:#9aa1ad;--line:#333842;--a:#7aa6d0;--b:#d98b86;
  --ok:#8fc098;--warn:#d9a06a;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#16181d;--fg:#e9ecf1;--dim:#9aa1ad;--line:#333842;
  --a:#7aa6d0;--b:#d98b86;--ok:#8fc098;--warn:#d9a06a;color-scheme:dark}
body{background:var(--bg);color:var(--fg);font:14px/1.5 -apple-system,system-ui,
     "Helvetica Neue",Arial,sans-serif;margin:0;padding:16px;max-width:1200px}
h1{font-size:17px;margin:0 0 4px}
.hint{font-size:12px;color:var(--dim);margin:0 0 10px}
.wrap{position:relative;display:inline-block;border:1px solid var(--line);
      max-width:100%%;line-height:0}
#img{max-width:100%%;height:auto;display:block}
#ov{position:absolute;inset:0;width:100%%;height:100%%}
.row{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin:10px 0}
button{font:inherit;padding:6px 12px;border:1px solid var(--line);border-radius:6px;
       background:transparent;color:var(--fg);cursor:pointer}
button:hover{border-color:var(--fg)}
button.on{background:var(--fg);color:var(--bg)}
.A{border-color:var(--a);color:var(--a)} .B{border-color:var(--b);color:var(--b)}
.pill{font-size:12px;padding:3px 9px;border-radius:999px;border:1px solid var(--line);
      color:var(--dim)}
table{border-collapse:collapse;font-size:12px;margin-top:8px}
td,th{border:1px solid var(--line);padding:3px 8px;text-align:left}
textarea{width:100%%;height:150px;font:12px ui-monospace,Menlo,monospace;
         background:transparent;color:var(--fg);border:1px solid var(--line);
         border-radius:6px;padding:8px}
</style>

<h1>%(rec)s — 指认自己的两只手</h1>
<p class="hint">每 %(stride)d 帧停一次。对这一帧分别指认 <b>A</b> 和 <b>B</b>：点一个候选框，
或按 <b>D</b> 自己画，或点「不在画面」/「不确定」。<b>A/B 是匿名的，不要求判断左右</b>，
只要在这条录像里一直跟住同一只手。候选框不带任何标签，这是故意的。<br>
标完 A/B 后按 <b>O</b> 进「其他框」模式，把剩下的候选框逐个点成
<b>别人的手</b> / <b>不是手</b>（前臂、倒影、物体）/ <b>不确定</b>（再点一次清除）。
这两类的代价不同：并错别人的手是隐私泄露，并错自己的前臂只是轨迹脏。</p>

<div class="row">
  <button id="pick" hidden>选择 frames 文件夹</button>
  <span class="pill" id="imgstat">未加载图片</span>
  <span class="pill" id="prog">0 / %(nsamp)d</span>
</div>

<div class="wrap"><img id="img" alt=""><svg id="ov" viewBox="0 0 %(W)d %(H)d"
  preserveAspectRatio="none"></svg></div>

<div class="row">
  <button id="prev">← 上一帧</button><button id="next">下一帧 →</button>
  <span class="pill" id="fr">—</span>
  <button id="draw">D 画框</button><button id="selO">O 其他框</button>
  <span class="pill" id="oleft"></span>
</div>
<div class="row">
  <span class="pill">正在指认</span>
  <button class="A" id="selA">A</button><button class="B" id="selB">B</button>
  <button id="absent">不在画面</button><button id="unsure">不确定</button>
  <button id="clear">清除这一帧</button>
</div>
<table id="tbl"></table>

<div class="row" style="margin-top:14px">
  <button id="exp">导出 JSON</button><button id="copy">复制</button>
  <span class="pill" id="msg"></span>
</div>
<textarea id="out" placeholder="点导出"></textarea>

<script>
const REC=%(recj)s, STRIDE=%(stride)d, W=%(W)d, H=%(H)d;
const SAMP=%(samp)s;      // 抽样帧号
const CAND=%(cand)s;      // {frame: [[x0,y0,x1,y1],...]}  无标签
const IMGS=%(imgs)s;      // {frame: base64 jpeg} 或 null
const KEY="handgold-"+REC;
// who 取 "A" / "B" / "O"。O 是「其他框」模式：点一个候选框就在
// 别人的手 -> 不是手 -> 不确定 -> 清除 之间循环。
// 为什么需要它：v1 只标本人的两只手，凡不是这两只的一律落进同一个桶，
// 而那个桶里混着两种代价完全不同的东西 —— 别人的手（并错了 = 隐私泄露）
// 和前臂/倒影（并错了 = 轨迹脏，零泄露）。安全判据只该约束前者。
const OKIND=["other_hand","not_hand","unsure"];
const OLABEL={other_hand:"别人的手", not_hand:"不是手", unsure:"不确定"};
let st={}, cur=0, who="A", drawing=false, files=new Map(), dragging=null;
try{const s=localStorage.getItem(KEY); if(s) st=JSON.parse(s);}catch(e){}
const $=s=>document.querySelector(s);
const save=()=>{try{localStorage.setItem(KEY,JSON.stringify(st));}catch(e){}};
const rec=()=>{const f=SAMP[cur]; st[f]=st[f]||{}; return st[f];};

function render(){
  const f=SAMP[cur];
  $("#fr").textContent="帧 "+f+"（第 "+(cur+1)+" / "+SAMP.length+"）";
  const done=SAMP.filter(x=>st[x]&&st[x].A&&st[x].B).length;
  $("#prog").textContent=done+" / "+SAMP.length;
  const nm=String(f).padStart(6,"0")+".jpg";
  if(IMGS&&IMGS[f]){ $("#img").src="data:image/jpeg;base64,"+IMGS[f];
                     $("#imgstat").textContent="图已内嵌"; }
  else { const fo=files.get(nm);
    if(fo){ $("#img").src=URL.createObjectURL(fo); $("#imgstat").textContent=nm; }
    else { $("#img").removeAttribute("src"); $("#imgstat").textContent="缺 "+nm; } }
  const ov=$("#ov"); ov.innerHTML="";
  const r=rec();
  (CAND[f]||[]).forEach((b,i)=>{
    const el=document.createElementNS("http://www.w3.org/2000/svg","rect");
    el.setAttribute("x",b[0]); el.setAttribute("y",b[1]);
    el.setAttribute("width",b[2]-b[0]); el.setAttribute("height",b[3]-b[1]);
    let col="#9aa1ad", wdt=3;
    const isA=r.A&&r.A.box&&same(r.A.box,b), isB=r.B&&r.B.box&&same(r.B.box,b);
    if(isA){col=getCSS("--a");wdt=6;}
    if(isB){col=getCSS("--b");wdt=6;}
    const ok=(r.other||{})[i];
    if(!isA&&!isB&&ok){ col=ok==="other_hand"?getCSS("--warn"):"#6b7280"; wdt=5; }
    el.setAttribute("fill","none"); el.setAttribute("stroke",col);
    if(!isA&&!isB&&ok==="unsure") el.setAttribute("stroke-dasharray","8 6");
    el.setAttribute("stroke-width",wdt); el.style.cursor="pointer";
    el.onclick=()=>{ if(who==="O") cycleOther(i); else assign(b,false); };
    ov.appendChild(el);
    if(!isA&&!isB&&ok){
      const t=document.createElementNS("http://www.w3.org/2000/svg","text");
      t.setAttribute("x",b[0]+6); t.setAttribute("y",b[1]-10);
      t.setAttribute("fill",col); t.setAttribute("font-size","52");
      // 加描边光晕：这些框常落在杂乱背景上（车间、货架），纯色字读不出来。
      // 52 是在 1920 宽的坐标系里的值，页面缩到约 480px 宽时约合 13px。
      t.setAttribute("stroke","#000"); t.setAttribute("stroke-width","6");
      t.setAttribute("paint-order","stroke"); t.setAttribute("font-weight","700");
      t.textContent=OLABEL[ok]; ov.appendChild(t);
    }
  });
  ["A","B"].forEach(k=>{
    const v=r[k];
    if(v&&v.box&&!(CAND[f]||[]).some(b=>same(b,v.box))){
      const el=document.createElementNS("http://www.w3.org/2000/svg","rect");
      el.setAttribute("x",v.box[0]); el.setAttribute("y",v.box[1]);
      el.setAttribute("width",v.box[2]-v.box[0]);
      el.setAttribute("height",v.box[3]-v.box[1]);
      el.setAttribute("fill","none");
      el.setAttribute("stroke",getCSS(k==="A"?"--a":"--b"));
      el.setAttribute("stroke-width",6);
      el.setAttribute("stroke-dasharray","12 8");
      ov.appendChild(el);
    }
  });
  $("#selA").classList.toggle("on",who==="A");
  $("#selB").classList.toggle("on",who==="B");
  $("#selO").classList.toggle("on",who==="O");
  $("#draw").classList.toggle("on",drawing);
  const left=othersLeft(f);
  $("#oleft").textContent = left===0 ? "其他框：判完了"
    : "其他框：还剩 "+left+" 个要判";
  $("#oleft").style.color = left===0 ? "" : getCSS("--warn");
  $("#tbl").innerHTML="<tr><th>手</th><th>这一帧</th></tr>"+
    ["A","B"].map(k=>{const v=r[k];
      return "<tr><td>"+k+"</td><td>"+(!v?"未标":
        v.state==="absent"?"不在画面":v.state==="unsure"?"不确定":
        ("框 "+v.box.map(Math.round).join(",")+(v.drawn?"（手画）":"")))+"</td></tr>";
    }).join("");
}
const getCSS=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const same=(a,b)=>a&&b&&a.every((v,i)=>Math.abs(v-b[i])<0.5);
function assign(box,drawn){ if(who==="O")return;
  // 指认为手时，清掉这个框在「其他框」里的判定 —— 否则同一个框会既是手
  // 又是 others，在安全判据里同时进正例和负例，把那个事件变成必败。
  // 第三批标到第 3 条就出现了一次（124431 帧 680，IoU=1.0）。
  const r=rec(); const f=SAMP[cur];
  (CAND[f]||[]).forEach((b,i)=>{ if(same(b,box)&&r.other&&r.other[i]) delete r.other[i]; });
  r[who]={state:"box",box:box.slice(),drawn:!!drawn};
  save(); who=who==="A"?"B":"A"; render(); }
function mark(state){ if(who==="O")return;
  rec()[who]={state:state}; save(); who=who==="A"?"B":"A"; render(); }
function cycleOther(i){ const r=rec();
  // 反方向也要堵：已经指认成 A/B 的框不接受「其他框」判定
  const b=(CAND[SAMP[cur]]||[])[i];
  if(b&&((r.A&&r.A.box&&same(r.A.box,b))||(r.B&&r.B.box&&same(r.B.box,b)))){
    $("#msg").textContent="这个框已经指认为手，先清除再判"; return; }
  r.other=r.other||{};
  const k=OKIND.indexOf(r.other[i]);
  if(k+1>=OKIND.length) delete r.other[i]; else r.other[i]=OKIND[k+1];
  save(); render(); }
// 这一帧还有几个候选框既不是 A/B 也没被判过。导出时用它决定这一帧算不算
// 「判全了」—— 不记这个就分不清「这帧没有别的框」和「还没判到这帧」，
// 而那正是分母会被悄悄改小的地方。
function othersLeft(f){ const r=st[f]||{}; const o=r.other||{}; let n=0;
  (CAND[f]||[]).forEach((b,i)=>{
    if(r.A&&r.A.box&&same(r.A.box,b))return;
    if(r.B&&r.B.box&&same(r.B.box,b))return;
    if(!o[i]) n++; });
  return n; }

$("#pick").onclick=()=>{const i=document.createElement("input");
  i.type="file"; i.webkitdirectory=true; i.multiple=true;
  i.onchange=()=>{for(const f of i.files) files.set(f.name,f);
    $("#imgstat").textContent="已载入 "+files.size+" 张"; render();};
  i.click();};
$("#prev").onclick=()=>{cur=Math.max(0,cur-1);render();};
$("#next").onclick=()=>{cur=Math.min(SAMP.length-1,cur+1);render();};
$("#selA").onclick=()=>{who="A";render();};
$("#selB").onclick=()=>{who="B";render();};
$("#selO").onclick=()=>{who=who==="O"?"A":"O";render();};
$("#absent").onclick=()=>mark("absent");
$("#unsure").onclick=()=>mark("unsure");
$("#clear").onclick=()=>{delete st[SAMP[cur]];save();render();};
$("#draw").onclick=()=>{drawing=!drawing;render();};
document.onkeydown=e=>{
  if(e.target.tagName==="TEXTAREA")return;
  if(e.key==="ArrowLeft")$("#prev").click();
  if(e.key==="ArrowRight")$("#next").click();
  if(e.key==="a"||e.key==="A")$("#selA").click();
  if(e.key==="b"||e.key==="B")$("#selB").click();
  if(e.key==="d"||e.key==="D")$("#draw").click();
  if(e.key==="n"||e.key==="N")$("#absent").click();
  if(e.key==="u"||e.key==="U")$("#unsure").click();
  if(e.key==="o"||e.key==="O")$("#selO").click();
};
const ov=$("#ov");
ov.addEventListener("pointerdown",e=>{ if(!drawing)return;
  const p=pt(e); dragging={x0:p.x,y0:p.y,x1:p.x,y1:p.y}; ov.setPointerCapture(e.pointerId);});
ov.addEventListener("pointermove",e=>{ if(!dragging)return;
  const p=pt(e); dragging.x1=p.x; dragging.y1=p.y; preview();});
ov.addEventListener("pointerup",()=>{ if(!dragging)return;
  const d=dragging; dragging=null;
  const b=[Math.min(d.x0,d.x1),Math.min(d.y0,d.y1),
           Math.max(d.x0,d.x1),Math.max(d.y0,d.y1)];
  if(b[2]-b[0]>4&&b[3]-b[1]>4){drawing=false;assign(b,true);} else render();});
function pt(e){const r=ov.getBoundingClientRect();
  return {x:(e.clientX-r.left)/r.width*W, y:(e.clientY-r.top)/r.height*H};}
function preview(){ render();
  if(!dragging)return; const el=document.createElementNS("http://www.w3.org/2000/svg","rect");
  el.setAttribute("x",Math.min(dragging.x0,dragging.x1));
  el.setAttribute("y",Math.min(dragging.y0,dragging.y1));
  el.setAttribute("width",Math.abs(dragging.x1-dragging.x0));
  el.setAttribute("height",Math.abs(dragging.y1-dragging.y0));
  el.setAttribute("fill","none"); el.setAttribute("stroke",getCSS("--warn"));
  el.setAttribute("stroke-width",4); ov.appendChild(el);}

function build(){
  const hands=["A","B"].map(k=>{
    const anchors=[], vis=[];
    SAMP.forEach(f=>{const v=(st[f]||{})[k];
      if(v&&v.state==="box"){anchors.push({frame:f,box:v.box.map(x=>Math.round(x)),
        drawn:!!v.drawn}); vis.push(f);}});
    const iv=[]; let s=null,p=null;
    vis.forEach(f=>{ if(s===null){s=f;p=f;} else if(f-p<=STRIDE){p=f;}
      else {iv.push([s,p]); s=f; p=f;} });
    if(s!==null) iv.push([s,p]);
    return {hand_key:k, side_note:"", intervals:iv, anchors:anchors};
  });
  const unc=[]; SAMP.forEach(f=>["A","B"].forEach(k=>{
    const v=(st[f]||{})[k]; if(v&&v.state==="unsure") unc.push({frame:f,hand_key:k});}));
  const others=[], complete=[];
  SAMP.forEach(f=>{
    const o=((st[f]||{}).other)||{};
    (CAND[f]||[]).forEach((b,i)=>{ if(o[i])
      others.push({frame:f, box:b.map(x=>Math.round(x)), kind:o[i]}); });
    if(othersLeft(f)===0) complete.push(f);
  });
  return {schema_version:"hand_gold_v2", recording:REC, stride:STRIDE,
          annotator:"", frame_size:[W,H], hands:hands, uncertain:unc,
          sampled_frames:SAMP, others:others,
          others_complete_frames:complete};
}
$("#exp").onclick=()=>{ $("#out").value=JSON.stringify(build(),null,1);
  const n=SAMP.filter(f=>st[f]&&st[f].A&&st[f].B).length;
  const oc=SAMP.filter(f=>othersLeft(f)===0).length;
  $("#msg").textContent="手 "+n+"/"+SAMP.length+"，其他框判全的 "+oc+"/"+SAMP.length+
    ((n<SAMP.length||oc<SAMP.length)?"（未标完）":"");};
$("#copy").onclick=async()=>{ if(!$("#out").value)$("#exp").click();
  try{await navigator.clipboard.writeText($("#out").value);
    $("#msg").textContent="已复制";}
  catch(e){$("#out").select();$("#msg").textContent="按 Cmd/Ctrl+C";}};
if(!IMGS) $("#pick").hidden=false;
render();
</script>
"""


def sampled(lo, hi, stride):
    """抽样网格 —— 固定起点与步长，**不看有没有检测**。

    两件事以前是错的，都会让结果偏好看，所以写在这里而不是注释在调用处。

    网格起点必须是跑的那个窗口的起点，不是 detect2 的首帧。按首帧对齐时，
    153159（首帧 415）和 015550（412）排出 415/425/… 的网格，和 400 对齐的
    帧号一个都对不上，页面抽样帧数变成 0。另外八条首帧恰好是 400，于是**靠
    巧合**对齐了 —— 包括已标完的三条。靠巧合成立的网格不是网格，而且它让
    不同录像的 gold 不可合并。

    网格也不能过滤「这一帧有没有框」。原来写的是 `if f in set(fs)`，于是
    检测器一个框都没出的帧根本不会出现在页面上 —— 而那正是「手在画面里但
    没被检到」的帧，是这份 gold 要测的缺陷本身。153159 上这一条静默扔掉了
    37 帧里的 16 帧。有图就进网格，候选为空时标注者自己画。
    """
    return list(range(lo, hi, stride))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detect2", required=True, help="该录像的 detect2 csv")
    ap.add_argument("--recording", required=True)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--grid-start", type=int, default=400,
                    help="抽样网格起点 —— 跑的那个窗口的起点，不是 detect2 首帧")
    ap.add_argument("--grid-end", type=int, default=800, help="不含")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1520)
    ap.add_argument("--embed", help="frame->base64 jpeg 的 json，内嵌进页面，"
                                    "这样 artifact 自带图、不用选文件夹")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    boxes = collections.defaultdict(list)
    for r in csv.DictReader(open(a.detect2, encoding="utf-8-sig")):
        try:
            f = int(r["frame"])
            b = [round(float(r["x0"])), round(float(r["y0"])),
                 round(float(r["x1"])), round(float(r["y1"]))]
        except (KeyError, ValueError):
            continue
        # tid / own / side 在这里被丢掉，页面永远拿不到它们
        boxes[f].append(b)

    grid = sampled(a.grid_start, a.grid_end, a.stride)
    embed = {}
    if a.embed:
        embed = {int(k): v for k, v in
                 json.load(open(a.embed, encoding="utf-8")).items()}
        # 只留真有图的抽样帧：缺图的帧标不了，留着只会让进度条骗人。
        # 但缺了多少必须报出来 —— 帧库是按模型相关判据抽的子集，缺的帧几乎
        # 都是检测器没出框的帧，所以在剩下的帧上算出的覆盖率是上界不是估计。
        samp = [f for f in grid if f in embed]
    else:
        samp = list(grid)
    cand = {f: boxes.get(f, []) for f in samp}
    html = PAGE % {"title": "%s 手级标注" % a.recording,
                   "rec": a.recording, "recj": json.dumps(a.recording),
                   "stride": a.stride, "nsamp": len(samp),
                   "W": a.width, "H": a.height,
                   "samp": json.dumps(samp),
                   "cand": json.dumps(cand, separators=(",", ":")),
                   "imgs": (json.dumps({str(f): embed[f] for f in samp},
                                       separators=(",", ":"))
                            if embed else "null")}
    open(a.out, "w", encoding="utf-8").write(html)
    n_box = sum(len(v) for v in cand.values())
    n_empty = sum(1 for v in cand.values() if not v)
    print("-> %s" % a.out)
    print("  网格 %d 帧（%d-%d 步长 %d）；有图 %d，缺图 %d -> present_rate %.2f"
          % (len(grid), a.grid_start, a.grid_end - a.stride, a.stride,
             len(samp), len(grid) - len(samp), len(samp) / max(1, len(grid))))
    if len(samp) < len(grid):
        print("    缺图的帧：%s" % [f for f in grid if f not in embed])
        print("    这些帧绝大多数是检测器没出框的帧 —— 所以覆盖率是上界")
    print("  候选框 %d 个，已剥掉 tid / own / side；零候选的帧 %d 个（要自己画）"
          % (n_box, n_empty))
    print("  每帧候选数：%s"
          % dict(sorted(collections.Counter(len(v) for v in cand.values()).items())))
    print("  页面要配 frames 目录：%06d.jpg 命名、%dx%d" % (0, a.width, a.height))


if __name__ == "__main__":
    main()
