"""The finished video against the raw one, with no model output shown.

EVERY MEASUREMENT SO FAR HAS BEEN CONDITIONED ON A PROPOSAL. The hand numbers
are over tracks the tracker admitted; the face numbers are over candidates the
detector emitted; the 1.000 face recall counts faces that were proposed. None
of them can see a hand or a face the system never put forward, because such a
thing appears in no denominator any of those tools can build.

So this shows a person two pictures -- the source frame and what the pipeline
produced from it -- and asks what is wrong with the second one. No boxes, no
scores, no tracks, nothing the models believe. If a colleague's face is
plainly visible in the output, that is a miss whether or not the detector had
an opinion about it, and this is the only instrument in the project that can
say so.

THE FLAGS ARE THE FOUR WAYS THE OUTPUT CAN BE WRONG.

    1  a face is visible that should have been covered
    2  another person's hand is visible that should have been covered
    3  the wearer's own hand is covered when it should not be
    4  something that is neither a face nor a hand is covered

They are independent and a frame can carry several, so they toggle rather than
replace one another. `0` clears the frame to `looks right`.

FRAMES ARE SAMPLED, EVENTS ARE WHAT COUNT. A false cover that lasts half a
second is one mistake, not fifteen; nine of the fourteen face false positives
in the last audit came from a single continuous event. Consecutive flagged
samples of the same kind are therefore grouped into events downstream, and the
sampling stride is recorded so a rate per minute can be computed rather than
a rate per sampled frame.
"""
from __future__ import annotations

import argparse
import base64
import json
import os

FLAGS = (("face_miss", "1", "漏了脸"),
         ("other_miss", "2", "漏了别人的手"),
         ("owner_blur", "3", "误糊自己的手"),
         ("junk_blur", "4", "误糊非脸非手"))

SHEET = """<meta charset=utf-8><title>end-to-end audit</title><style>
body{font:13px/1.5 system-ui;margin:0;background:#111;color:#ddd}
#bar{position:sticky;top:0;background:#181818;padding:9px 14px;z-index:9;
  border-bottom:1px solid #333;display:flex;gap:16px;align-items:center;
  flex-wrap:wrap}
button{font:13px system-ui;padding:5px 10px;cursor:pointer}
b{color:#ffd33d}
.pair{padding:10px 14px;border-bottom:1px solid #262626}
.pair.cur{background:#1d2430;outline:2px solid #4a8}
.pair.flagged{border-left:5px solid #d33}
.pair.clean{border-left:5px solid #2a6}
.imgs{display:flex;gap:10px}
.imgs figure{margin:0}
.imgs img{border-radius:3px;display:block;width:100%}
.imgs figcaption{font-size:11px;color:#999;margin-top:3px;text-align:center}
.meta{color:#9ab;font-size:12px;margin-bottom:5px}
.tag{background:#733;color:#fff;padding:1px 7px;border-radius:3px;
  font-size:11px;margin-left:6px}
</style>
<div id=bar>
 <span id=prog></span>
 <span><b>1</b> 漏了脸 &nbsp; <b>2</b> 漏了别人的手 &nbsp;
   <b>3</b> 误糊自己的手 &nbsp; <b>4</b> 误糊非脸非手 &nbsp;
   <b>0</b> 没问题 &nbsp; <b>&uarr;&darr;</b> move &nbsp; <b>u</b> undo</span>
 <button onclick="dl()">download CSV</button>
</div>
<div id=list></div>
<script>
const D = __PAYLOAD__;
const KEY = "e2e:" + D.tag;
const FL = __FLAGS__;
let lab = {}, cur = 0, hist = [];
try { lab = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch(e) { lab={}; }
const list = document.getElementById("list");
D.pairs.forEach((p, i) => {
  const d = document.createElement("div");
  d.className = "pair"; d.id = "p" + i;
  d.innerHTML = '<div class=meta>f' + p.frame + ' <span id=t' + i +
    '></span></div><div class=imgs>' +
    '<figure><img src="' + p.raw + '"><figcaption>原始</figcaption></figure>' +
    '<figure><img src="' + p.out + '"><figcaption>管线输出</figcaption>' +
    '</figure></div>';
  d.onclick = () => { cur = i; draw(); };
  list.appendChild(d);
});
function draw(){
  D.pairs.forEach((p,i)=>{
    const v = lab[p.frame] || [];
    const e = document.getElementById("p"+i);
    e.className = "pair" + (i===cur ? " cur" : "") +
      (v.length ? " flagged" : (p.frame in lab ? " clean" : ""));
    document.getElementById("t"+i).innerHTML =
      v.map(k => '<span class=tag>' + FL[k] + '</span>').join('');
  });
  const n = Object.keys(lab).length;
  const bad = Object.values(lab).filter(v=>v.length).length;
  document.getElementById("prog").innerHTML =
    "<b>" + D.tag + "</b> &nbsp; " + n + "/" + D.pairs.length +
    " 看过, " + bad + " 有问题";
  localStorage.setItem(KEY, JSON.stringify(lab));
  const e = document.getElementById("p"+cur);
  if(e) e.scrollIntoView({block:"nearest"});
}
function toggle(k){
  const p = D.pairs[cur]; if(!p) return;
  const v = (lab[p.frame] || []).slice();
  hist.push([p.frame, lab[p.frame]]);
  const i = v.indexOf(k);
  if(i >= 0) v.splice(i,1); else v.push(k);
  lab[p.frame] = v;
  draw();
}
document.onkeydown = e => {
  if(e.key>="1" && e.key<="4") toggle(["face_miss","other_miss",
    "owner_blur","junk_blur"][+e.key-1]);
  else if(e.key==="0"){ const p=D.pairs[cur];
    hist.push([p.frame, lab[p.frame]]); lab[p.frame]=[];
    cur=Math.min(cur+1,D.pairs.length-1); draw(); }
  else if(e.key==="ArrowDown"){ cur=Math.min(cur+1,D.pairs.length-1); draw(); }
  else if(e.key==="ArrowUp"){ cur=Math.max(cur-1,0); draw(); }
  else if(e.key==="u"){ const h=hist.pop(); if(h){ if(h[1]===undefined)
      delete lab[h[0]]; else lab[h[0]]=h[1]; draw(); } }
  else return;
  e.preventDefault();
};
draw();
function dl(){
  let s = "recording,frame,stride,face_miss,other_miss,owner_blur,junk_blur\\n";
  for(const p of D.pairs){
    if(!(p.frame in lab)) continue;
    const v = lab[p.frame];
    s += D.tag + "," + p.frame + "," + D.stride + "," +
         ["face_miss","other_miss","owner_blur","junk_blur"]
           .map(k => v.includes(k) ? 1 : 0).join(",") + "\\n";
  }
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([s], {type:"text/csv"}));
  a.download = "e2e_" + D.tag + ".csv";
  a.click();
}
</script>
"""


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--databag", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--sample", type=int, default=5,
                    help="frames between audited pairs. The pipeline still "
                         "runs on every frame; this is how often a person is "
                         "asked to look.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=620)
    ap.add_argument("--clf_ctx", default="/workspace/own_ctx_best.pt")
    ap.add_argument("--geom", default="/workspace/geom_inv2.json")
    ap.add_argument("--face_model",
                    default="/workspace/models/yolov8n-face-lindevs.onnx")
    a = ap.parse_args()

    import cv2
    from src.rig import demo_video

    tag = os.path.basename(a.databag.rstrip("/")).replace("databag-26_", "R")
    cal = os.path.join(a.databag, "calibration.yaml")
    vids = {k: os.path.join(a.databag, f"{k}.mp4")
            for k in ("cam12", "cam34", "cam56")}
    from src.rig.calibration import RigCalibration
    from src.rig import own_ctx, geom_prior
    from ultralytics import YOLO

    rig = RigCalibration(cal)
    ctx_model, ctx_device, ctx_arm = own_ctx.load_model(a.clf_ctx)
    if ctx_model is None:
        raise SystemExit(f"no ownership checkpoint at {a.clf_ctx}")
    geom = geom_prior.load_model(a.geom)

    pairs = []

    def keep(k, clean, out):
        """Called by the renderer for every frame; stores every `sample`th."""
        if k % a.sample:
            return
        h = int(round(clean.shape[0] * a.width / clean.shape[1]))

        def b64(img):
            ok, buf = cv2.imencode(
                ".jpg", cv2.resize(img, (a.width, h)),
                [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            return ("data:image/jpeg;base64,"
                    + base64.b64encode(buf).decode()) if ok else ""
        pairs.append({"frame": a.start + k, "raw": b64(clean),
                      "out": b64(out)})

    demo_video.run(rig, vids, None, a.start, a.n, 1,
                   YOLO("/shared/models/HaWoR/weights/external/detector.pt"),
                   None, None, 10, 14.0, 12.0,
                   face_model=a.face_model, geom=geom, geom_w=0.5,
                   max_owner=2, ctx=(ctx_model, ctx_device),
                   frame_hook=keep)

    html = (SHEET.replace("__PAYLOAD__",
                          json.dumps({"tag": tag, "stride": a.sample,
                                      "pairs": pairs}))
            .replace("__FLAGS__",
                     json.dumps({k: z for k, _, z in FLAGS})))
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n  {len(pairs)} 对 -> {a.out} "
          f"({os.path.getsize(a.out) / 1e6:.1f} MB)")
    print("  左边是原始画面，右边是管线输出。没有框、没有分数、没有轨迹 —— "
          "看的是\n  输出本身哪里不对，这样才测得到系统根本没提出的东西。")


if __name__ == "__main__":
    main()
