"""Judge the cuts the generator itself calls a task change.

THE OLD POSITIVES CANNOT ANSWER THIS. Every boundary probe this project holds
came from candidates we proposed -- stored labels and a naive change detector
-- so `rolan does not cut at our boundaries` and `rolan cuts at boundaries we
never proposed` produce the same number. The only way to tell them apart is to
take ITS candidates and have a person judge them.

ITS OWN RELATION PASS PICKS THEM, NOT A FILTER OF MINE. The timeline labels
every gap `same_task`, `different_task` or `uncertain`, and two thirds come
back `same_task` -- cuts it does not claim are task boundaries. Scoring those
against a task-boundary audit measured the wrong set and inflated the
rate-matched baseline with them. The 159 it calls `different_task` are the
claims, and they are what gets judged.

BLIND TO ITS OWN LABELS. The episode names either side of a cut are the
model's account of what changed, and showing them would turn the question from
`did a task end here` into `does this description sound right`. Only the
video appears.

THE CRITERION IS STATED, NOT LEFT TO TASTE. It was derived from gold with no
exceptions over 25 cases: a cut belongs between two runs of the same action
only if a DISENGAGEMENT happened -- the object left the hand and was picked up
again, or both hands went idle and re-engaged. Direction reversals, regrasps
and position changes inside a continuous action are not boundaries. Putting it
in the header is what makes two annotators mean the same thing.

ONE VIDEO PER PAGE, NOT ONE PER CANDIDATE. Forty ten-second clips concatenated
is one seekable file of a few megabytes; forty files is forty things to keep
next to the html.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os
import random
import subprocess

PAD_S = 5.0
IDENT = [("boundary", "是任务边界"), ("not_boundary", "不是（同一任务内部）"),
         ("unclear", "说不准"), ("unusable", "这段看不清/没内容")]

SHEET = """<meta charset=utf-8><title>cut audit __TAG__</title><style>
body{font:13px/1.5 system-ui;margin:0;background:#111;color:#ddd}
#bar{position:sticky;top:0;background:#181818;padding:8px 14px;z-index:9;
  border-bottom:1px solid #333;display:flex;gap:12px;align-items:center;
  flex-wrap:wrap}
button{font:13px system-ui;padding:5px 10px;cursor:pointer}
b{color:#ffd33d}
#stage{position:relative;margin:10px 14px;width:__W__px}
video{width:__W__px;border-radius:4px;display:block;background:#000}
#mark{height:26px;position:relative;background:#1b1b1b;border-radius:3px;
  margin-top:4px;cursor:pointer}
#at{position:absolute;top:0;bottom:0;width:3px;background:#f55;left:50%}
#atlab{position:absolute;top:3px;left:50%;margin-left:7px;color:#f77;
  font-size:11px;white-space:nowrap}
#now{position:absolute;top:0;bottom:0;width:2px;background:#ffd33d}
/* THE MOMENT IS IN TIME, NOT IN SPACE. A line down the middle of the frame
   would be read as pointing AT something. This is a band that appears while
   playback is crossing the instant, plus a signed countdown, so the question
   `which moment am I judging` never needs the strip underneath. */
#flash{position:absolute;left:0;right:0;top:0;height:34px;display:none;
  background:linear-gradient(rgba(255,60,60,.85),rgba(255,60,60,0));
  color:#fff;font:13px/22px system-ui;text-align:center;border-radius:4px 4px 0 0}
#cd{position:absolute;right:8px;top:8px;background:rgba(0,0,0,.6);
  color:#ffd33d;font:12px/1.6 ui-monospace,monospace;padding:1px 7px;
  border-radius:3px}
#list{margin:0 14px 40px;font-size:12px}
.row{padding:3px 6px;border-bottom:1px solid #222;cursor:pointer;
  display:flex;gap:10px}
.row.cur{background:#1d2430;outline:1px solid #4a8}
.row.done{border-left:4px solid #2a6}
.v{color:#ffd33d}
.key{font-size:11px;color:#999;max-width:900px}
</style>
<div id=bar>
 <span id=prog></span>
 <button onclick="again()">重放 (r)</button>
 <span class=key><b>1</b> 是任务边界 &nbsp; <b>2</b> 不是（同一任务内部）
 &nbsp; <b>3</b> 说不准 &nbsp; <b>4</b> 看不清 &nbsp; <b>r</b> 重放
 &nbsp; <b>&uarr;&darr;</b> 换一个 &nbsp; <b>u</b> 撤销<br>
 每段 10 秒，<b>候选时刻在正中（第 5 秒）</b>：播到那一刻画面顶部会闪红条，
 右上角的计数从 −5.0s 走到 +5.0s，0 就是被问的那一刻；下面的红线是它在进度条上的位置，
 进度条可以点着跳。<b>问的是：在那一刻，前一个任务结束、下一个开始了吗。</b><br>
 判据：两段同类动作之间只有发生了
 <b>disengagement</b> 才算边界——对象脱手后重新拿取，或双手 idle／离开工作区
 后重新接触。连续动作内部的方向反转、换握、位置转移都不是边界。</span>
 <button onclick="dl()">download CSV</button>
</div>
<div id=stage><video id=v src="__VIDEO__" playsinline></video>
<div id=flash>候选时刻</div><div id=cd></div>
<div id=mark><div id=at></div><div id=atlab>候选时刻</div>
<div id=now></div></div></div>
<div id=list></div>
<script>
const D = __PAYLOAD__;
const ID = __IDENT__;
const KEY = "cutaudit:" + D.tag;
let st = {}, cur = 0, hist = [];
try { st = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch(e) { st={}; }
const v = document.getElementById("v"), list = document.getElementById("list");
D.items.forEach((it, i) => {
  const d = document.createElement("div");
  d.className = "row"; d.id = "r" + i;
  d.innerHTML = '<span style="width:150px">' + it.rec + '</span>' +
    '<span style="width:90px">t=' + it.t.toFixed(1) + 's</span>' +
    '<span id=v' + i + ' class=v></span>';
  d.onclick = () => { cur = i; play(); };
  list.appendChild(d);
});
function play(){
  const it = D.items[cur];
  v.currentTime = it.off;
  v.play();
  draw();
}
function again(){ play(); }
function tick(){
  const it = D.items[cur];
  if (it){
    if (v.currentTime > it.off + it.len) v.pause();
    const rel = v.currentTime - (it.off + it.len/2);   // signed, 0 = the cut
    const f = Math.max(0, Math.min(1, (v.currentTime - it.off) / it.len));
    document.getElementById("now").style.left = (f*100) + "%";
    document.getElementById("cd").textContent =
      (rel >= 0 ? "+" : "") + rel.toFixed(1) + "s";
    document.getElementById("flash").style.display =
      Math.abs(rel) < 0.35 ? "block" : "none";
  }
  requestAnimationFrame(tick);
}
tick();
document.getElementById("mark").onclick = e => {
  const it = D.items[cur]; if (!it) return;
  const r = e.currentTarget.getBoundingClientRect();
  v.currentTime = it.off + (e.clientX - r.left) / r.width * it.len;
};
function draw(){
  D.items.forEach((it,i)=>{
    const s = st[it.key];
    document.getElementById("r"+i).className =
      "row" + (s ? " done" : "") + (i===cur ? " cur" : "");
    document.getElementById("v"+i).textContent =
      s ? (ID.find(x=>x[0]===s)||["",s])[1] : "";
  });
  const n = D.items.filter(it=>st[it.key]).length;
  document.getElementById("prog").innerHTML =
    "<b>" + D.tag + "</b> &nbsp; " + n + "/" + D.items.length + " 已判";
  localStorage.setItem(KEY, JSON.stringify(st));
  const el = document.getElementById("r"+cur);
  if (el) el.scrollIntoView({block:"nearest"});
}
document.onkeydown = ev => {
  const n = +ev.key;
  if (n >= 1 && n <= ID.length){
    const it = D.items[cur]; if (!it) return;
    hist.push([it.key, st[it.key]]);
    st[it.key] = ID[n-1][0];
    cur = Math.min(cur+1, D.items.length-1);
    play();
  }
  else if (ev.key === "r"){ again(); }
  else if (ev.key === "u"){
    const h = hist.pop();
    if (h){ if (h[1] === undefined) delete st[h[0]]; else st[h[0]] = h[1]; draw(); }
  }
  else if (ev.key === "ArrowDown"){ cur = Math.min(cur+1, D.items.length-1); play(); }
  else if (ev.key === "ArrowUp"){ cur = Math.max(cur-1, 0); play(); }
  else return;
  ev.preventDefault();
};
v.addEventListener("loadedmetadata", () => { play(); });
draw();
function dl(){
  let s = "key,recording_id,t_s,verdict\\n";
  for (const it of D.items) if (st[it.key])
    s += it.key + "," + it.rec + "," + it.t.toFixed(1) + "," + st[it.key] + "\\n";
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([s],{type:"text/csv"}));
  a.download = "cut_audit_" + D.tag + ".csv"; a.click();
}
</script>
"""


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="/workspace/rolan61")
    ap.add_argument("--corpus", default="/shared/datasets/datasets")
    ap.add_argument("--relation", default="different_task")
    ap.add_argument("--pad", type=float, default=PAD_S)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--crf", type=int, default=30)
    ap.add_argument("--page", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()

    src = {}
    for p in ("01", "02"):
        for d in glob.glob(f"{a.corpus}/human_ego_recording_segmentation_10fps"
                           f"_r01_part_{p}/recordings/recording_*"):
            v = os.path.join(d, "mid.mp4")
            if os.path.exists(v):
                src[os.path.basename(d)] = v

    items = []
    for f in sorted(glob.glob(f"{a.runs}/*/run/global_timeline.json")):
        rid = os.path.basename(os.path.dirname(os.path.dirname(f)))
        if rid not in src:
            continue
        d = json.load(open(f))
        dur = float(d.get("recording_duration_s") or 0)
        for r in d.get("relations", []):
            if r.get("relation") != a.relation:
                continue
            t = r.get("boundary_s")
            if t is None:
                continue
            t = float(t)
            if t < a.pad or t > dur - a.pad:
                continue          # no room for the context either side
            items.append({"rec": rid, "t": t})
    # ORDER CARRIES NO SIGNAL. Grouped by recording, a judge learns the scene
    # and starts answering from it; in time order they learn the rhythm of one
    # recording's episodes.
    random.Random(a.seed).shuffle(items)
    print(f"  {len(items)} 个 {a.relation} 候选，来自 "
          f"{len({i['rec'] for i in items})} 段录像")
    os.makedirs(a.outdir, exist_ok=True)

    pages = [items[i:i + a.page] for i in range(0, len(items), a.page)]
    for pi, page in enumerate(pages, 1):
        tag = f"p{pi}"
        parts, cur, meta = [], 0.0, []
        tmp = os.path.join(a.outdir, f"_tmp_{tag}")
        os.makedirs(tmp, exist_ok=True)
        for k, it in enumerate(page):
            clip = os.path.join(tmp, f"{k:03d}.mp4")
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error",
                 "-ss", f"{it['t'] - a.pad:.3f}", "-i", src[it["rec"]],
                 "-t", f"{2 * a.pad:.3f}",
                 # The left eye again: side by side, the model once called the
                 # operator's green shirt a green towel, and a person reading
                 # a doubled image has the same problem.
                 "-vf", f"crop=iw/2:ih:0:0,scale={a.width}:-2",
                 "-c:v", "libx264", "-preset", "veryfast", "-crf", str(a.crf),
                 "-pix_fmt", "yuv420p", "-an", clip], check=True)
            parts.append(clip)
            meta.append({"key": f"{it['rec']}:{it['t']:.1f}", "rec": it["rec"],
                         "t": round(it["t"], 1), "off": round(cur, 2),
                         "len": round(2 * a.pad, 2)})
            cur += 2 * a.pad
        lst = os.path.join(tmp, "list.txt")
        with open(lst, "w") as fh:
            for p in parts:
                fh.write(f"file '{os.path.abspath(p)}'\n")
        mp4 = os.path.join(a.outdir, f"cut_{tag}.mp4")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat",
                        "-safe", "0", "-i", lst, "-c", "copy",
                        "-movflags", "+faststart", mp4], check=True)
        for p in parts:
            os.remove(p)
        os.remove(lst)
        os.rmdir(tmp)
        import cv2
        cap = cv2.VideoCapture(mp4)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or a.width)
        cap.release()
        html = (SHEET.replace("__PAYLOAD__",
                              json.dumps({"tag": tag, "items": meta},
                                         ensure_ascii=False))
                .replace("__IDENT__", json.dumps([[k, v] for k, v in IDENT],
                                                 ensure_ascii=False))
                .replace("__VIDEO__", os.path.basename(mp4))
                .replace("__TAG__", tag).replace("__W__", str(w)))
        path = os.path.join(a.outdir, f"cut_{tag}.html")
        open(path, "w", encoding="utf-8").write(html)
        print(f"  {tag}: {len(page)} 个 -> {path} "
              f"(mp4 {os.path.getsize(mp4) / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
