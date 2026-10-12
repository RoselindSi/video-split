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

PAGE = """<title>%(title)s</title>
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
只要在这条录像里一直跟住同一只手。候选框不带任何标签，这是故意的。</p>

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
  <button id="draw">D 画框</button>
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
    if(r.A&&r.A.box&&same(r.A.box,b)){col=getCSS("--a");wdt=6;}
    if(r.B&&r.B.box&&same(r.B.box,b)){col=getCSS("--b");wdt=6;}
    el.setAttribute("fill","none"); el.setAttribute("stroke",col);
    el.setAttribute("stroke-width",wdt); el.style.cursor="pointer";
    el.onclick=()=>{assign(b,false);};
    ov.appendChild(el);
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
  $("#draw").classList.toggle("on",drawing);
  $("#tbl").innerHTML="<tr><th>手</th><th>这一帧</th></tr>"+
    ["A","B"].map(k=>{const v=r[k];
      return "<tr><td>"+k+"</td><td>"+(!v?"未标":
        v.state==="absent"?"不在画面":v.state==="unsure"?"不确定":
        ("框 "+v.box.map(Math.round).join(",")+(v.drawn?"（手画）":"")))+"</td></tr>";
    }).join("");
}
const getCSS=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const same=(a,b)=>a&&b&&a.every((v,i)=>Math.abs(v-b[i])<0.5);
function assign(box,drawn){ rec()[who]={state:"box",box:box.slice(),drawn:!!drawn};
  save(); who=who==="A"?"B":"A"; render(); }
function mark(state){ rec()[who]={state:state}; save(); who=who==="A"?"B":"A"; render(); }

$("#pick").onclick=()=>{const i=document.createElement("input");
  i.type="file"; i.webkitdirectory=true; i.multiple=true;
  i.onchange=()=>{for(const f of i.files) files.set(f.name,f);
    $("#imgstat").textContent="已载入 "+files.size+" 张"; render();};
  i.click();};
$("#prev").onclick=()=>{cur=Math.max(0,cur-1);render();};
$("#next").onclick=()=>{cur=Math.min(SAMP.length-1,cur+1);render();};
$("#selA").onclick=()=>{who="A";render();};
$("#selB").onclick=()=>{who="B";render();};
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
  return {schema_version:"hand_gold_v1", recording:REC, stride:STRIDE,
          annotator:"", frame_size:[W,H], hands:hands, uncertain:unc,
          sampled_frames:SAMP};
}
$("#exp").onclick=()=>{ $("#out").value=JSON.stringify(build(),null,1);
  const n=SAMP.filter(f=>st[f]&&st[f].A&&st[f].B).length;
  $("#msg").textContent="已标 "+n+" / "+SAMP.length+" 帧"+
    (n<SAMP.length?"（未标完）":"");};
$("#copy").onclick=async()=>{ if(!$("#out").value)$("#exp").click();
  try{await navigator.clipboard.writeText($("#out").value);
    $("#msg").textContent="已复制";}
  catch(e){$("#out").select();$("#msg").textContent="按 Cmd/Ctrl+C";}};
if(!IMGS) $("#pick").hidden=false;
render();
</script>
"""


def sampled(frames, stride):
    fs = sorted(frames)
    if not fs:
        return []
    lo, hi = fs[0], fs[-1]
    return [f for f in range(lo, hi + 1, stride) if f in set(fs)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detect2", required=True, help="该录像的 detect2 csv")
    ap.add_argument("--recording", required=True)
    ap.add_argument("--stride", type=int, default=10)
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

    samp = sampled(boxes, a.stride)
    embed = {}
    if a.embed:
        embed = {int(k): v for k, v in
                 json.load(open(a.embed, encoding="utf-8")).items()}
        # 只留真有图的抽样帧：缺图的帧标不了，留着只会让进度条骗人
        samp = [f for f in samp if f in embed]
    cand = {f: boxes[f] for f in samp}
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
    print("-> %s" % a.out)
    print("  抽样帧 %d 个（步长 %d，覆盖 %d-%d）"
          % (len(samp), a.stride, samp[0] if samp else 0, samp[-1] if samp else 0))
    print("  候选框 %d 个，已剥掉 tid / own / side" % n_box)
    print("  每帧候选数：%s"
          % dict(sorted(collections.Counter(len(v) for v in cand.values()).items())))
    print("  页面要配 frames 目录：%06d.jpg 命名、%dx%d" % (0, a.width, a.height))


if __name__ == "__main__":
    main()
