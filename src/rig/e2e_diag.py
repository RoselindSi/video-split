"""The frames a person called wrong, with what the pipeline believed shown.

THE AUDIT HID THE MODEL OUTPUT AND THIS DELIBERATELY DOES NOT. Those are two
different jobs. Finding a miss requires not being shown what the system thinks,
or the auditor checks its work instead of looking at the picture. Attributing a
miss requires exactly that information: a face nobody covered is a different
defect depending on whether the detector proposed it and was overruled, or
never proposed it at all, and no amount of staring at the output distinguishes
those.

FOUR PLACES A COVER CAN BE LOST, AND THEY HAVE DIFFERENT FIXES.

    detector   nothing was proposed near the region -- a recall problem, and
               no threshold or policy downstream can reach it
    admission  a detection exists in `raw_dets` but got no track id, so it
               never reached ownership
    ownership  it was tracked and called the wearer's
    mask       it was called foreign and the cut or the veto produced no
               pixels there

The frame-level trace already had a taxonomy like this, but it was anchored on
frames where SUPPRESSION WENT EMPTY -- a frame where the system covered the
wrong thing confidently looks fine to it. This one is anchored on frames a
person named, which is the only anchor that can see a miss the system was
never uncertain about.

WHAT IT SHOWS PER FRAME. The source, the delivered output, and the source with
every proposal drawn: hand detections with their track admission and P(owner),
face detections after the hand veto. The judgement is then one click.
"""
from __future__ import annotations

import argparse
import base64
import csv
import glob
import json
import os

FLAGS = ("face_miss", "other_miss", "owner_blur", "junk_blur")
CRITICAL = ("face_miss", "other_miss")

SHEET = """<meta charset=utf-8><title>miss attribution</title><style>
body{font:13px/1.5 system-ui;margin:0;background:#111;color:#ddd}
#bar{position:sticky;top:0;background:#181818;padding:9px 14px;z-index:9;
  border-bottom:1px solid #333;display:flex;gap:16px;align-items:center;
  flex-wrap:wrap}
button{font:13px system-ui;padding:5px 10px;cursor:pointer}
b{color:#ffd33d}
.ev{padding:10px 14px;border-bottom:1px solid #262626}
.ev.cur{background:#1d2430;outline:2px solid #4a8}
.ev.detector{border-left:5px solid #d33}
.ev.admission{border-left:5px solid #d83}
.ev.ownership{border-left:5px solid #38d}
.ev.mask{border-left:5px solid #8a3}
.ev.unsure{border-left:5px solid #666}
.imgs{display:flex;gap:8px}
.imgs figure{margin:0;flex:1}
.imgs img{border-radius:3px;display:block;width:100%}
.imgs figcaption{font-size:11px;color:#999;margin-top:3px;text-align:center}
.meta{color:#9ab;font-size:12px;margin-bottom:5px}
.tag{background:#733;color:#fff;padding:1px 7px;border-radius:3px;
  font-size:11px;margin-left:6px}
.num{color:#8ab4c8;font-family:ui-monospace,monospace;font-size:11px}
</style>
<div id=bar>
 <span id=prog></span>
 <span><b>1</b> 检测器没提出 &nbsp; <b>2</b> 提出了但没建轨迹 &nbsp;
   <b>3</b> 归属判错 &nbsp; <b>4</b> 掩码没盖住 &nbsp; <b>5</b> 说不准
   &nbsp; <b>&uarr;&darr;</b> move &nbsp; <b>u</b> undo</span>
 <button onclick="dl()">download CSV</button>
</div>
<div id=list></div>
<script>
const D = __PAYLOAD__;
const KEY = "diag:" + D.tag;
const CLS = ["detector","admission","ownership","mask","unsure"];
const CN = {detector:"检测器没提出", admission:"没建轨迹",
            ownership:"归属判错", mask:"掩码没盖住", unsure:"说不准"};
let lab = {}, cur = 0, hist = [];
try { lab = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch(e) { lab={}; }
const list = document.getElementById("list");
D.events.forEach((e, i) => {
  const d = document.createElement("div");
  d.className = "ev"; d.id = "e" + i;
  d.innerHTML = '<div class=meta>' + e.rec + ' f' + e.frame +
    e.kinds.map(k => '<span class=tag>' + k + '</span>').join('') +
    ' <span id=v' + i + '></span><br><span class=num>' + e.info +
    '</span></div><div class=imgs>' +
    '<figure><img src="' + e.raw + '"><figcaption>原始</figcaption></figure>' +
    '<figure><img src="' + e.out + '"><figcaption>输出</figcaption></figure>' +
    '<figure><img src="' + e.dbg + '"><figcaption>原始+全部候选</figcaption>' +
    '</figure></div>';
  d.onclick = () => { cur = i; draw(); };
  list.appendChild(d);
});
function draw(){
  D.events.forEach((e,i)=>{
    const v = lab[e.key];
    document.getElementById("e"+i).className =
      "ev " + (v || "") + (i===cur ? " cur" : "");
    document.getElementById("v"+i).innerHTML =
      v ? '<span class=tag style="background:#356">' + CN[v] + '</span>' : '';
  });
  const n = Object.keys(lab).length;
  document.getElementById("prog").innerHTML =
    "<b>" + D.tag + "</b> &nbsp; " + n + "/" + D.events.length + " 已归因";
  localStorage.setItem(KEY, JSON.stringify(lab));
  const el = document.getElementById("e"+cur);
  if(el) el.scrollIntoView({block:"nearest"});
}
document.onkeydown = ev => {
  if(ev.key>="1" && ev.key<="5"){
    const e = D.events[cur]; if(!e) return;
    hist.push([e.key, lab[e.key]]);
    lab[e.key] = CLS[+ev.key-1];
    cur = Math.min(cur+1, D.events.length-1);
    draw();
  }
  else if(ev.key==="ArrowDown"){ cur=Math.min(cur+1,D.events.length-1); draw(); }
  else if(ev.key==="ArrowUp"){ cur=Math.max(cur-1,0); draw(); }
  else if(ev.key==="u"){ const h=hist.pop(); if(h){ if(h[1]===undefined)
      delete lab[h[0]]; else lab[h[0]]=h[1]; draw(); } }
  else return;
  ev.preventDefault();
};
draw();
function dl(){
  let s = "recording,frame,kinds,cause\\n";
  for(const e of D.events) if(lab[e.key])
    s += e.rec + "," + e.frame + "," + e.kinds.join("|") + "," +
         lab[e.key] + "\\n";
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([s], {type:"text/csv"}));
  a.download = "diag_" + D.tag + ".csv";
  a.click();
}
</script>
"""


def load_flags(paths, kinds):
    """-> {recording: {frame: [kind]}} for frames a person flagged."""
    by = {}
    for p in paths:
        for r in csv.DictReader(open(p, encoding="utf-8-sig")):
            hit = [k for k in kinds if int(r.get(k, 0))]
            if hit:
                by.setdefault(r["recording"], {})[int(r["frame"])] = hit
    return by


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", action="append", required=True)
    ap.add_argument("--clips", default="/workspace/e2e_main2.txt",
                    help="databag:start[:n] per line, to find the sources")
    ap.add_argument("--kinds", default=",".join(CRITICAL),
                    help="which flags to attribute; default the two "
                         "privacy-critical ones")
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=430)
    ap.add_argument("--clf_ctx", default="/workspace/own_ctx_best.pt")
    ap.add_argument("--geom", default="/workspace/geom_inv2.json")
    ap.add_argument("--face_model",
                    default="/workspace/models/yolov8n-face-lindevs.onnx")
    a = ap.parse_args()

    import cv2
    from ultralytics import YOLO
    from src.rig import demo_video, own_ctx, geom_prior
    from src.rig.calibration import RigCalibration

    paths = []
    for c in a.csv:
        paths += sorted(glob.glob(c)) or [c]
    kinds = [k.strip() for k in a.kinds.split(",") if k.strip()]
    want = load_flags([p for p in paths if os.path.exists(p)], kinds)
    if not want:
        raise SystemExit("no flagged frames in those csv files")
    n_frames = sum(len(v) for v in want.values())
    print(f"  {n_frames} 个被标记的帧，{len(want)} 段录像   类别 {kinds}")

    src_of = {}
    for line in open(a.clips):
        line = line.strip()
        if not line:
            continue
        # `databag:start` or `databag:start:n`; the databag path never
        # contains a colon, so splitting from the right is unambiguous.
        d = line.rsplit(":", 2)[0] if line.count(":") >= 2 else \
            line.rsplit(":", 1)[0]
        tag = os.path.basename(d.rstrip("/")).replace("databag-26_", "R")
        src_of.setdefault(tag, []).append(d)

    ctx_model, ctx_device, _ = own_ctx.load_model(a.clf_ctx)
    if ctx_model is None:
        raise SystemExit(f"no ownership checkpoint at {a.clf_ctx}")
    geom = geom_prior.load_model(a.geom)
    model = YOLO("/shared/models/HaWoR/weights/external/detector.pt")

    events = []
    for rec, frames in sorted(want.items()):
        cand = src_of.get(rec)
        if not cand:
            print(f"  !! {rec} 找不到源 databag，跳过 {len(frames)} 帧")
            continue
        databag = cand[0]
        lo, hi = min(frames), max(frames)
        rig = RigCalibration(os.path.join(databag, "calibration.yaml"))
        vids = {k: os.path.join(databag, f"{k}.mp4")
                for k in ("cam12", "cam34", "cam56")}
        print(f"  {rec}: {len(frames)} 帧, 重跑 {lo}-{hi}")

        got = {}

        def keep(k, clean, out, info=None):
            f = info["frame"] if info else None
            if f not in frames:
                return
            h = int(round(clean.shape[0] * a.width / clean.shape[1]))

            def b64(img):
                ok, buf = cv2.imencode(
                    ".jpg", cv2.resize(img, (a.width, h)),
                    [int(cv2.IMWRITE_JPEG_QUALITY), 82])
                return ("data:image/jpeg;base64,"
                        + base64.b64encode(buf).decode()) if ok else ""
            dbg = clean.copy()
            # Every hand the detector proposed, whether or not it was
            # admitted: an unadmitted box is the whole point of the
            # `admission` category and would be invisible if only `dets`
            # were drawn.
            adm = {tuple(d["box"]) for d in info["dets"]}
            for d in info["raw_dets"]:
                t = tuple(d["box"])
                x0, y0, x1, y1 = t
                inn = t in adm
                col = (60, 220, 255) if inn else (110, 110, 110)
                cv2.rectangle(dbg, (x0, y0), (x1, y1), col, 3 if inn else 2)
                cv2.putText(dbg, f"{d['conf']:.2f}", (x0, max(14, y0 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
            for i, (d, own, p) in enumerate(zip(info["dets"], info["own"],
                                                info["p_owner"])):
                x0, y0, x1, y1 = d["box"]
                cv2.putText(dbg, f"{'OWN' if own else 'OTH'} {p:.2f}",
                            (x0, min(dbg.shape[0] - 4, y1 + 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (90, 255, 140) if own else (90, 140, 255), 2)
            for x0, y0, x1, y1 in info["faces"]:
                cv2.rectangle(dbg, (x0, y0), (x1, y1), (255, 120, 255), 3)
            got[f] = {
                "key": f"{rec}:{f}", "rec": rec, "frame": f,
                "kinds": frames[f], "raw": b64(clean), "out": b64(out),
                "dbg": b64(dbg),
                "info": (f"raw_det {len(info['raw_dets'])}  "
                         f"admitted {len(info['dets'])}  "
                         f"own {sum(info['own'])}  "
                         f"faces {len(info['faces'])}  "
                         f"oth_px {info['oth_px']}  "
                         f"veto_px {info['veto_px']}")}

        demo_video.run(rig, vids, None, lo, hi - lo + 1, 1, model, None, None,
                       10, 14.0, 12.0, verbose=False,
                       face_model=a.face_model, geom=geom, geom_w=0.5,
                       max_owner=2, ctx=(ctx_model, ctx_device),
                       frame_hook=keep)
        events += [got[f] for f in sorted(got)]
        missing = sorted(set(frames) - set(got))
        if missing:
            print(f"    !! {len(missing)} 帧没重现（解码漂移？）: "
                  f"{missing[:5]}")

    tag = "critical" if set(kinds) == set(CRITICAL) else "_".join(kinds)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(SHEET.replace("__PAYLOAD__",
                              json.dumps({"tag": tag, "events": events})))
    print(f"\n  {len(events)} 个事件 -> {a.out} "
          f"({os.path.getsize(a.out) / 1e6:.1f} MB)")
    print("  第三张图是原始画面加上全部候选：青色=进了管线的手，灰色=检测器"
          "提出但没建轨迹，\n  洋红=脸。手下方标 OWN/OTH 和 P(owner)。")
    print("  1 检测器没提出   2 提出了但没建轨迹   3 归属判错   "
          "4 掩码没盖住   5 说不准")


if __name__ == "__main__":
    main()
