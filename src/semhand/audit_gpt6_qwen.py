"""Blind human audit of the two box-level teachers: Qwen (numbered prompt) and GPT-6.

WHAT IS BEING SETTLED. On 9,701 images labelled by both, the two agree on
91.5% of matched boxes, but the disagreements run one way: Qwen calls owner
where GPT-6 calls other 957 times, the reverse 109 times. And GPT-6 drew 7,100
"other" boxes where the shared hand detector produced no candidate at all --
either hands the detector misses (a privacy hole no classifier can close) or
boxes GPT-6 invented. Neither set has any human truth.

STRATA (a box is one item). Boxes are matched per image at IoU >= 0.3.
    A  Qwen owner  / GPT-6 other          disagreement, the privacy direction
    B  Qwen other  / GPT-6 owner          disagreement, the other direction
    C  GPT-6 other, no detector candidate detector miss or invention
    D  GPT-6 owner, no detector candidate
    E  both owner
    F  both other
Matched items show the detector's box (the hand the pipeline would act on);
GPT-6-only items show GPT-6's box.

BLIND. The page draws one neutral box and never shows a model, a label or a
stratum; items are shuffled across strata with a fixed seed. Context: the
full image with the box, a zoom on the box, and the same camera's other two
sampled moments from that window.

WRITTEN BEFORE ANY ANSWER EXISTS -- WHAT IS REPORTED.
    per stratum   share of human owner / other / not_hand / unsure
    A, B          which model is right when they disagree
    C, D          share of GPT-6-only boxes that are real hands, and whose
    E, F          accuracy when they agree
    population    Qwen's and GPT-6's owner precision and other precision on
                  matched boxes, weighting A/B/E/F by their full counts;
                  "unsure" answers are left out of every denominator and
                  counted beside it
"""
from __future__ import annotations

import argparse
import base64
import collections
import io
import json
import os
import random

Q = "/shared/ownership_labels/boxes_qwen_has_other_v1"
G = "/shared/ownership_labels/boxes_gpt6_qwen50958_v1"
PLAN = {"A": 70, "B": 40, "C": 70, "D": 30, "E": 45, "F": 45}
SEED = 20260917
IOU = 0.3


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def strata():
    """-> ({stratum: [item]}, {stratum: full count})"""
    qb = collections.defaultdict(list)
    for line in open(Q + "/candidates.jsonl"):
        r = json.loads(line)
        qb[(r["window_id"], r["camera"], int(r["frame_idx"]))].append(r)
    out = collections.defaultdict(list)
    for line in open(G + "/labels_completed.jsonl"):
        r = json.loads(line)
        key = (r["window_id"], r["camera"], int(r["cache_frame_idx"]))
        if key not in qb or not r.get("labels"):
            continue
        qs, gs = qb[key], r["labels"].get("hands") or []
        pairs = sorted(((iou(q["box_xyxy"], g["box_xyxy"]), i, j)
                        for i, q in enumerate(qs) for j, g in enumerate(gs)), reverse=True)
        uq, ug = set(), set()
        base = {"image": os.path.join(G, r["image"]), "window_id": r["window_id"],
                "camera": r["camera"], "frame_idx": int(r["cache_frame_idx"]),
                "recording": r["recording"], "device": r["device"]}
        for v, i, j in pairs:
            if v < IOU:
                break
            if i in uq or j in ug:
                continue
            uq.add(i)
            ug.add(j)
            ql, gl = qs[i]["qwen_label"], gs[j]["label"]
            s = {("owner", "other"): "A", ("other", "owner"): "B",
                 ("owner", "owner"): "E", ("other", "other"): "F"}.get((ql, gl))
            if s:
                out[s].append(dict(base, box=qs[i]["box_xyxy"], qwen=ql, gpt6=gl,
                                   gpt6_box=gs[j]["box_xyxy"], gpt6_reason=gs[j].get("reason", "")))
        for j, g in enumerate(gs):
            if j not in ug and g["label"] in ("other", "owner"):
                s = "C" if g["label"] == "other" else "D"
                out[s].append(dict(base, box=g["box_xyxy"], qwen=None, gpt6=g["label"],
                                   gpt6_reason=g.get("reason", "")))
    return out, {s: len(v) for s, v in out.items()}


def context_frames(item):
    """The same camera's other sampled frames in this window, from the Qwen sample dir."""
    d = os.path.join(Q, "samples", item["window_id"], item["camera"])
    frames = []
    if os.path.isdir(d):
        for name in sorted(os.listdir(d)):
            if name.startswith("frame_") and name.endswith(".jpg"):
                idx = int(name[6:-4])
                if idx != item["frame_idx"]:
                    frames.append(os.path.join(d, name))
    return frames[:2]


def b64(img, quality):
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def build(a):
    from PIL import Image, ImageDraw
    pools, full = strata()
    print("各类总数", full)
    rng = random.Random(SEED)
    items = []
    for s, n in PLAN.items():
        pool = pools.get(s, [])
        # at most two items per recording, so one workstation cannot fill a stratum
        rng.shuffle(pool)
        per_rec = collections.Counter()
        for it in pool:
            if len([x for x in items if x["stratum"] == s]) >= n:
                break
            if per_rec[it["recording"]] >= 2:
                continue
            per_rec[it["recording"]] += 1
            items.append(dict(it, stratum=s))
    rng.shuffle(items)
    cards = []
    for k, it in enumerate(items):
        im = Image.open(it["image"]).convert("RGB")
        x0, y0, x1, y1 = it["box"]
        side = max(x1 - x0, y1 - y0) * 1.8
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        crop = im.crop((int(cx - side / 2), int(cy - side / 2), int(cx + side / 2),
                        int(cy + side / 2))).resize((240, 240), Image.BICUBIC)
        full_img = im.copy()
        ImageDraw.Draw(full_img).rectangle([x0, y0, x1, y1], outline=(255, 230, 0), width=4)
        ctx = [b64(Image.open(p).convert("RGB").resize((240, 192)), 70) for p in context_frames(it)]
        cards.append({"i": k, "full": b64(full_img, 78), "crop": b64(crop, 85), "ctx": ctx})
        it["i"] = k
    key = [{k: v for k, v in it.items() if k not in ("image",)} for it in items]
    os.makedirs(a.out, exist_ok=True)
    json.dump({"seed": SEED, "plan": PLAN, "full_counts": full, "items": key},
              open(os.path.join(a.out, "audit_key.json"), "w"), ensure_ascii=False, indent=1)
    html = PAGE.replace("__CARDS__", json.dumps(cards)).replace("__N__", str(len(cards)))
    open(os.path.join(a.out, "audit_gpt6_qwen.html"), "w", encoding="utf-8").write(html)
    print(f"-> {a.out}/audit_gpt6_qwen.html  {len(cards)} 条  {dict(collections.Counter(i['stratum'] for i in items))}")


def score(a):
    key = json.load(open(a.key))
    ans = json.load(open(a.answers))
    got = {int(k): v for k, v in (ans.get("answers") or ans).items()}
    by = collections.defaultdict(collections.Counter)
    for it in key["items"]:
        v = got.get(it["i"])
        if v:
            by[it["stratum"]][v] += 1
    names = {"A": "Qwen owner / GPT-6 other", "B": "Qwen other / GPT-6 owner",
             "C": "只有 GPT-6，other", "D": "只有 GPT-6，owner", "E": "两边都 owner", "F": "两边都 other"}
    print(f"已答 {sum(sum(c.values()) for c in by.values())} / {len(key['items'])}")
    print(f"  {'类别':<26}{'人判 owner':>10}{'other':>8}{'不是手':>8}{'看不出':>8}")
    for s in "ABCDEF":
        c = by[s]
        print(f"  {s} {names[s]:<24}{c['owner']:>10}{c['other']:>8}{c['not_hand']:>8}{c['unsure']:>8}")
    # population precision on matched boxes, weighted by full counts
    full = key["full_counts"]
    def rate(s, lab):
        c = by[s]
        n = c["owner"] + c["other"] + c["not_hand"]
        return c[lab] / n if n else float("nan")
    q_owner = (full["A"] * rate("A", "owner") + full["E"] * rate("E", "owner")) / (full["A"] + full["E"])
    g_owner = (full["B"] * rate("B", "owner") + full["E"] * rate("E", "owner")) / (full["B"] + full["E"])
    q_other = (full["B"] * rate("B", "other") + full["F"] * rate("F", "other")) / (full["B"] + full["F"])
    g_other = (full["A"] * rate("A", "other") + full["F"] * rate("F", "other")) / (full["A"] + full["F"])
    print("\n按全量加权（两边都画到的框）：")
    print(f"  Qwen  说 owner 的准确率 {q_owner:.1%}   说 other 的准确率 {q_other:.1%}")
    print(f"  GPT-6 说 owner 的准确率 {g_owner:.1%}   说 other 的准确率 {g_other:.1%}")
    c = by["C"]
    n = sum(c[x] for x in ("owner", "other", "not_hand"))
    if n:
        print(f"  检测器没检出、GPT-6 说 other 的框：真是别人的手 {c['other'] / n:.1%}，"
              f"是佩戴者的手 {c['owner'] / n:.1%}，不是手 {c['not_hand'] / n:.1%}"
              f"  → 全量估计漏检的别人的手约 {full['C'] * c['other'] / n:.0f} 个")


def q1prep(a):
    """Write the audited boxes where `semhand.qwen` reads items, one row each."""
    import csv
    key = json.load(open(a.key))
    d = os.path.join(a.root, "fresh", "audit")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "index.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=("rec", "frame", "tid", "databag", "image",
                                           "x0", "y0", "x1", "y1", "status"))
        w.writeheader()
        for it in key["items"]:
            img = os.path.join(Q, "samples", it["window_id"], it["camera"], f"frame_{it['frame_idx']:03d}.jpg")
            assert os.path.exists(img), img
            x0, y0, x1, y1 = it["box"]
            w.writerow({"rec": "audit", "frame": it["i"], "tid": 0, "databag": it["recording"],
                        "image": img, "x0": x0, "y0": y0, "x1": x1, "y1": y1, "status": "ok"})
    print(f"-> {d}/index.csv  {len(key['items'])} 条")


def q1score(a):
    import glob
    key = json.load(open(a.key))
    ans = json.load(open(a.answers))["answers"]
    q1 = {}
    for f in glob.glob(os.path.join(a.root, "qwen", "Q1_*.jsonl")):
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                q1[int(r["id"].split("|")[1])] = r["p"]
    names = {"A": "Qwen编号 owner / GPT-6 other", "B": "Qwen编号 other / GPT-6 owner",
             "C": "只有 GPT-6，other", "D": "只有 GPT-6，owner", "E": "两边都 owner", "F": "两边都 other"}
    by = collections.defaultdict(collections.Counter)
    for it in key["items"]:
        h = ans.get(str(it["i"]))
        if h not in ("owner", "other") or it["i"] not in q1:
            continue
        mine = "owner" if q1[it["i"]] >= 0.5 else "other"
        by[it["stratum"]][(h, mine)] += 1
    print(f"Q1 已判 {len(q1)} / {len(key['items'])}（只统计人判为 owner/other 的框）")
    print(f"  {'类别':<30}{'对':>5}{'错':>5}   {'人=别人 Q1=自己':>14}{'人=自己 Q1=别人':>14}   对照:编号Qwen对 / GPT-6对")
    tot = collections.Counter()
    for s in "ABCDEF":
        c = by[s]
        right = c[("owner", "owner")] + c[("other", "other")]
        wrong = c[("other", "owner")] + c[("owner", "other")]
        tot["right"] += right
        tot["wrong"] += wrong
        ref = {"A": ("owner", "other"), "B": ("other", "owner"), "E": ("owner", "owner"), "F": ("other", "other"),
               "C": (None, "other"), "D": (None, "owner")}[s]
        qn = sum(v for (h, _), v in c.items() if h == ref[0]) if ref[0] else None
        gp = sum(v for (h, _), v in c.items() if h == ref[1])
        n = right + wrong
        print(f"  {s} {names[s]:<28}{right:>5}{wrong:>5}   {c[('other', 'owner')]:>14}{c[('owner', 'other')]:>14}   "
              f"{'-' if qn is None else qn}/{n}  {gp}/{n}")
    print(f"  合计 对 {tot['right']} 错 {tot['wrong']}")


PAGE = r"""<meta charset="utf-8"><title>手部归属抽检</title>
<style>
body{font-family:-apple-system,system-ui,sans-serif;margin:0;background:#15171a;color:#e8e8e8}
.top{display:flex;gap:16px;align-items:center;padding:10px 16px;background:#202328;position:sticky;top:0}
.bar{flex:1;height:8px;background:#333;border-radius:4px;overflow:hidden}.bar i{display:block;height:100%;background:#4caf50}
.main{display:flex;gap:16px;padding:16px;flex-wrap:wrap;justify-content:center}
.full{max-width:100%;border-radius:6px}.side{display:flex;flex-direction:column;gap:10px}
.side img{border-radius:6px}.ctx{display:flex;gap:8px}
.btns{display:flex;gap:10px;justify-content:center;padding:6px 16px 20px;flex-wrap:wrap}
button{font-size:17px;padding:12px 18px;border-radius:8px;border:0;cursor:pointer;background:#2d3138;color:#eee}
button.on{outline:3px solid #ffd400}.b1{background:#1f6f43}.b2{background:#8a2b2b}.b3{background:#555}.b4{background:#6b5a1e}
small{color:#999}
</style>
<div class="top"><b>手部归属抽检</b><span id="pos"></span><div class="bar"><i id="prog"></i></div>
<button onclick="go(-1)">← 上一条</button><button onclick="go(1)">下一条 →</button>
<button onclick="exportAns()">导出结果</button></div>
<div class="main"><img id="full" class="full" width="640" height="512">
<div class="side"><div><small>黄框放大</small><br><img id="crop" width="240" height="240"></div>
<div><small>同一窗口、同一相机的另外两个时刻</small><div class="ctx" id="ctx"></div></div></div></div>
<div class="btns">
<button class="b1" onclick="ans('owner')">1 佩戴者的手</button>
<button class="b2" onclick="ans('other')">2 别人的手</button>
<button class="b3" onclick="ans('not_hand')">3 不是手</button>
<button class="b4" onclick="ans('unsure')">4 看不出</button></div>
<p style="text-align:center"><small>只判黄框里的那只手是谁的。按 1/2/3/4 作答，←/→ 翻页。进度自动保存在本浏览器里。</small></p>
<script>
const C=__CARDS__, N=__N__, KEY="audit_gpt6_qwen_v1";
let A={}; try{A=JSON.parse(localStorage.getItem(KEY)||"{}")}catch(e){}
let k=0; for(;k<N&&A[k];k++); if(k>=N)k=0;
function show(){const c=C[k];document.getElementById("full").src="data:image/jpeg;base64,"+c.full;
document.getElementById("crop").src="data:image/jpeg;base64,"+c.crop;
document.getElementById("ctx").innerHTML=c.ctx.map(x=>'<img width="240" height="192" src="data:image/jpeg;base64,'+x+'">').join("");
const done=Object.keys(A).length;document.getElementById("pos").textContent=`第 ${k+1} / ${N} 条，已答 ${done}`;
document.getElementById("prog").style.width=(100*done/N)+"%";
document.querySelectorAll(".btns button").forEach(b=>b.classList.toggle("on",b.getAttribute("onclick").includes("'"+A[k]+"'")));}
function go(d){k=Math.max(0,Math.min(N-1,k+d));show();}
function ans(v){A[k]=v;try{localStorage.setItem(KEY,JSON.stringify(A))}catch(e){}
if(k<N-1)k++;show(); if(Object.keys(A).length===N)alert("全部答完，点「导出结果」保存");}
function exportAns(){const blob=new Blob([JSON.stringify({answers:A,n:N,exported:new Date().toISOString()})],{type:"application/json"});
const a=document.createElement("a");a.href=URL.createObjectURL(blob);a.download="audit_gpt6_qwen_answers.json";a.click();}
document.addEventListener("keydown",e=>{if(e.key==="1")ans("owner");if(e.key==="2")ans("other");if(e.key==="3")ans("not_hand");
if(e.key==="4")ans("unsure");if(e.key==="ArrowLeft")go(-1);if(e.key==="ArrowRight")go(1);});
show();
</script>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("build", "score", "q1prep", "q1score"), required=True)
    ap.add_argument("--root", default="/workspace/audit_q1")
    ap.add_argument("--out", default="/workspace/audit_gpt6_qwen")
    ap.add_argument("--key", default="/workspace/audit_gpt6_qwen/audit_key.json")
    ap.add_argument("--answers")
    a = ap.parse_args()
    {"build": build, "score": score, "q1prep": q1prep, "q1score": q1score}[a.mode](a)


if __name__ == "__main__":
    main()
