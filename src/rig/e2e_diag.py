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
face detections, and separately the faces the hand veto discarded. That last
distinction is not cosmetic -- a vetoed face and a face the detector never
found are identical in the delivered picture and identical in a panel that
only draws survivors, and they are repaired at opposite ends of the pipeline.
The judgement is then one click.
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
FALSE_COVER = ("owner_blur", "junk_blur")

# A MISS AND A FALSE COVER DO NOT SHARE A TAXONOMY, and forcing them to share
# one is how the over-blur half of this audit stayed unexplained. `the
# detector never proposed` is the commonest cause of an uncovered face and is
# meaningless for a mosaic that should not be there: something DID propose it,
# and the question is what. The two sets below are chosen so every class names
# a different repair.
CAUSES = {
    "miss": (["detector", "admission", "ownership", "mask", "veto", "unsure"],
             {"detector": "检测器没提出", "admission": "没建轨迹",
              "ownership": "归属判错", "mask": "掩码没盖住",
              "veto": "被手部否决", "unsure": "说不准"},
             "<b>1</b> 检测器没提出 &nbsp; <b>2</b> 提出了但没建轨迹 &nbsp;"
             "<b>3</b> 归属判错 &nbsp; <b>4</b> 掩码没盖住 &nbsp;"
             "<b>5</b> 被手部否决<span style=\"color:#a76fd0\">（紫框 VETO）"
             "</span> &nbsp; <b>6</b> 说不准"),
    "cover": (["face_fp", "hand_oth", "residue", "spill", "unsure"],
              {"face_fp": "人脸误检", "hand_oth": "手判成别人的",
               "residue": "残留（框已经没了）", "spill": "掩码溢出",
               "unsure": "说不准"},
              "<b>1</b> 人脸误检<span style=\"color:#ff78ff\">（洋红框扣在"
              "非脸上）</span> &nbsp; <b>2</b> 手判成别人的"
              "<span style=\"color:#28dcff\">（蓝框扣在自己手上）</span>"
              " &nbsp; <b>3</b> 残留<span style=\"color:#ccc\">（白圈里没有"
              "任何框）</span> &nbsp; <b>4</b> 掩码溢出<span "
              "style=\"color:#ccc\">（框对了但白圈糊出去太多）</span>"
              " &nbsp; <b>5</b> 说不准"),
}

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
.ev.veto{border-left:5px solid #a76fd0}
.ev.unsure{border-left:5px solid #666}
.ev.face_fp{border-left:5px solid #d33}
.ev.hand_oth{border-left:5px solid #38d}
.ev.residue{border-left:5px solid #d83}
.ev.spill{border-left:5px solid #8a3}
.imgs{display:flex;gap:8px}
.imgs figure{margin:0;flex:1}
.imgs img{border-radius:3px;display:block;width:100%}
.imgs figcaption{font-size:11px;color:#999;margin-top:3px;text-align:center}
.meta{color:#9ab;font-size:12px;margin-bottom:5px}
.tag{background:#733;color:#fff;padding:1px 7px;border-radius:3px;
  font-size:11px;margin-left:6px}
.num{color:#8ab4c8;font-family:ui-monospace,monospace;font-size:11px}
.key{font-size:11px;color:#aaa}
.key i{display:inline-block;width:10px;height:10px;border-radius:2px;
  margin:0 4px 0 10px;vertical-align:-1px}
</style>
<div id=bar>
 <span id=prog></span>
 <span>__KEYS__
   &nbsp; <b>&uarr;&darr;</b> move &nbsp; <b>u</b> undo</span>
 <button onclick="dl()">download CSV</button>
 <span class=key>第三张图：
   <i style="background:#5aff5a"></i>OWN 判成自己的手
   <i style="background:#28dcff"></i>OTH 判成别人的手
   <i style="background:#828282"></i>没建轨迹
   <i style="background:#ff78ff"></i>脸
   <i style="background:#a03c78"></i>脸被手部否决
   <i style="background:#fff"></i>白圈=输出里真的被糊掉的区域</span>
</div>
<div id=list></div>
<script>
const D = __PAYLOAD__;
const KEY = "diag:" + D.tag;
const CLS = __CLS__;
const CN = __CN__;
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
  if(ev.key>="1" && ev.key<=String(CLS.length)){
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


def load_flags(paths, kinds, per_event=False):
    """-> {recording: {frame: [kind]}} for frames a person flagged.

    `per_event` keeps ONE FRAME PER RUN of consecutive flagged samples.
    A mosaic that sits on a bench for two seconds is one mistake with one
    cause, and attributing all twelve of its sampled frames costs twelve
    reruns and twelve judgements to learn the same thing once. The event is
    the unit everywhere else in this audit; this makes it the unit here too.
    The middle frame is taken rather than the first, because the first is
    where a cause is most likely to be transitional."""
    by, stride = {}, {}
    for p in paths:
        for r in csv.DictReader(open(p, encoding="utf-8-sig")):
            hit = [k for k in kinds if int(r.get(k, 0))]
            if hit:
                rec = r["recording"]
                by.setdefault(rec, {})[int(r["frame"])] = hit
                stride[rec] = int(r.get("stride", 5) or 5)
    if not per_event:
        return by
    out = {}
    for rec, frames in by.items():
        step, run, keep = stride.get(rec, 5), [], {}
        for f in sorted(frames):
            # A gap wider than the sampling stride means an unflagged sample
            # came between: two events, not one.
            if run and f - run[-1] > step:
                mid = run[len(run) // 2]
                keep[mid] = frames[mid]
                run = []
            run.append(f)
        if run:
            mid = run[len(run) // 2]
            keep[mid] = frames[mid]
        out[rec] = keep
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", action="append", required=True)
    ap.add_argument("--clips", default="/workspace/e2e_main2.txt",
                    help="databag:start[:n] per line, to find the sources")
    ap.add_argument("--per_event", action="store_true",
                    help="attribute one frame per run of consecutive "
                         "flagged samples instead of every frame")
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
    import numpy as np
    from ultralytics import YOLO
    from src.rig import demo_video, own_ctx, geom_prior
    from src.rig.calibration import RigCalibration

    paths = []
    for c in a.csv:
        paths += sorted(glob.glob(c)) or [c]
    kinds = [k.strip() for k in a.kinds.split(",") if k.strip()]
    want = load_flags([p for p in paths if os.path.exists(p)], kinds,
                      per_event=a.per_event)
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
            # EVERYTHING IS DRAWN AT FULL RESOLUTION AND THEN SQUEEZED TO
            # `a.width`. A 0.55 font on a frame four times wider than the
            # panel arrives about three pixels tall, which is why the verdict
            # was unreadable. Scale the annotation by the same factor the
            # picture is about to shrink by, so it lands at a fixed size.
            k = max(1.0, clean.shape[1] / float(a.width))
            fs, th = 0.6 * k, max(2, int(round(2 * k)))
            bt = max(2, int(round(2.5 * k)))

            # OWNERSHIP IN THE COLOUR, NOT ONLY IN THE TEXT. Admitted-versus-
            # not survives the change -- an unadmitted box has no ownership
            # verdict to show and stays grey -- so this three-way encoding
            # carries strictly more than the two-way one it replaces, and
            # carries it at a glance rather than at reading distance.
            OWN_C, OTH_C, GREY = (90, 255, 90), (255, 220, 40), (130, 130, 130)
            verdict = {tuple(d["box"]): (o, p) for d, o, p in
                       zip(info["dets"], info["own"], info["p_owner"])}
            for d in info["raw_dets"]:
                t = tuple(d["box"])
                x0, y0, x1, y1 = t
                v = verdict.get(t)
                col = GREY if v is None else (OWN_C if v[0] else OTH_C)
                cv2.rectangle(dbg, (x0, y0), (x1, y1), col,
                              bt if v is not None else max(2, bt // 2))
                # The confidence is what there is to say about a box with no
                # verdict; where there is a verdict, the verdict is the point.
                txt = (f"{d['conf']:.2f}" if v is None
                       else f"{'OWN' if v[0] else 'OTH'} {v[1]:.2f}")
                cv2.putText(dbg, txt, (x0, max(int(fs * 24), y0 - int(6 * k))),
                            cv2.FONT_HERSHEY_SIMPLEX, fs, col, th)
            for x0, y0, x1, y1 in info["faces"]:
                cv2.rectangle(dbg, (x0, y0), (x1, y1), (255, 120, 255), bt)
            # A face the detector DID propose and the hand veto threw away.
            # Drawn separately because the alternative -- not drawing it --
            # makes an uncovered face read as `detector never proposed` and
            # sends the fix to the wrong stage.
            for x0, y0, x1, y1 in info.get("faces_vetoed", []):
                cv2.rectangle(dbg, (x0, y0), (x1, y1), (120, 60, 160), bt)
                cv2.putText(dbg, "VETO",
                            (x0, max(int(fs * 24), y0 - int(6 * k))),
                            cv2.FONT_HERSHEY_SIMPLEX, fs, (150, 80, 200), th)
            # WHAT WAS ACTUALLY COVERED, taken from the pictures rather than
            # from the pipeline's own account of itself. `sup` differs from
            # `clean` exactly where pixels were destroyed, so the outline is
            # the delivered mosaic by definition -- no mask has to be
            # threaded through the hook, and a cover produced by a hold or a
            # coasting track shows up even though no box explains it. For a
            # false cover that coincidence is the whole judgement: a white
            # ring with a magenta box in it is a face false positive, and a
            # white ring with nothing in it is residue.
            diff = np.any(clean != out, axis=2).astype(np.uint8)
            inside = ""
            if diff.any():
                cont, _ = cv2.findContours(diff, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(dbg, cont, -1, (255, 255, 255),
                                 max(2, bt // 2))
                # WHAT SITS INSIDE THE COVERED REGION IS A FACT, NOT A
                # JUDGEMENT, so counting it here saves the auditor from doing
                # it by eye 165 times. It is reported and nothing is inferred
                # from it: `a magenta box is inside the ring` says the face
                # detector produced this mosaic, and whether the thing under
                # it is a face is the part only a person can settle.
                covered = [cv2.boundingRect(c) for c in cont]

                def hits(box):
                    x0, y0, x1, y1 = [int(v) for v in box[:4]]
                    for cx, cy, cw, ch in covered:
                        if (x0 < cx + cw and cx < x1
                                and y0 < cy + ch and cy < y1):
                            return True
                    return False
                n_face = sum(1 for b in info["faces"] if hits(b))
                oth = [d["box"] for d, o in zip(info["dets"], info["own"])
                       if not o]
                own = [d["box"] for d, o in zip(info["dets"], info["own"]) if o]
                inside = (f"  白圈内 洋红{n_face} 蓝{sum(1 for b in oth if hits(b))}"
                          f" 绿{sum(1 for b in own if hits(b))}")
            got[f] = {
                "key": f"{rec}:{f}", "rec": rec, "frame": f,
                "kinds": frames[f], "raw": b64(clean), "out": b64(out),
                "dbg": b64(dbg),
                "info": (f"raw_det {len(info['raw_dets'])}  "
                         f"admitted {len(info['dets'])}  "
                         f"own {sum(info['own'])}  "
                         f"faces {len(info['faces'])}"
                         + (f"+{len(info['faces_vetoed'])}veto"
                            if info.get("faces_vetoed") else "") + "  "
                         f"oth_px {info['oth_px']}  "
                         f"veto_px {info['veto_px']}" + inside)}

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
    # WHICH QUESTION IS BEING ASKED FOLLOWS FROM WHICH FLAGS WERE SELECTED.
    # Mixing a miss and a false cover in one sheet would put both taxonomies
    # on screen and let a frame be attributed with a class that cannot apply
    # to it, so the two are simply not allowed together.
    if set(kinds) <= set(FALSE_COVER):
        cls, cn, keys = CAUSES["cover"]
    elif set(kinds) <= set(CRITICAL):
        cls, cn, keys = CAUSES["miss"]
    else:
        raise SystemExit("--kinds 不能把漏检和误糊混在一起：它们的归因类别"
                         "不同，混在一张表里会让一帧被标上不适用的原因")
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(SHEET.replace("__PAYLOAD__",
                              json.dumps({"tag": tag, "events": events}))
                .replace("__CLS__", json.dumps(cls))
                .replace("__CN__", json.dumps(cn, ensure_ascii=False))
                .replace("__KEYS__", keys))
    print(f"\n  {len(events)} 个事件 -> {a.out} "
          f"({os.path.getsize(a.out) / 1e6:.1f} MB)")
    print("  第三张图是原始画面加上全部候选。手框的颜色就是归属判断："
          "绿=判成自己的手，\n  蓝=判成别人的手，灰=检测器提出但没建轨迹（没有"
          "归属可言）。洋红=脸，\n  紫色 VETO=脸检测器提出了但被手部否决丢掉。"
          "框上的数字是平滑后的 P(owner)。\n  白色轮廓=输出里真的被糊掉的区域，"
          "由原始帧和输出帧逐像素相减得到。")
    print("  1 检测器没提出   2 提出了但没建轨迹   3 归属判错   "
          "4 掩码没盖住   5 被手部否决   6 说不准")


if __name__ == "__main__":
    main()
