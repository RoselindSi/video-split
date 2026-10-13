#!/usr/bin/env python3
"""Build a blind, self-contained review packet for canonical hand tracks."""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import re
import shutil


REVIEW_FIELDS = ["rec", "canonical_tid", "label", "note", "n_views"]


def _safe(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def track_items(index_path, views_dir, out_dir, max_views=5):
    grouped = collections.defaultdict(list)
    with open(index_path, encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            canonical_tid = row.get("canonical_tid") or row.get("tid")
            if canonical_tid:
                grouped[(row["rec"], canonical_tid)].append(row)

    image_dir = os.path.join(out_dir, "images")
    os.makedirs(image_dir, exist_ok=True)
    items = []
    for (rec, canonical_tid), rows in sorted(grouped.items()):
        rows.sort(key=lambda row: int(row["frame"]))
        if len(rows) > max_views:
            indices = [round(i * (len(rows) - 1) / (max_views - 1))
                       for i in range(max_views)] if max_views > 1 else [len(rows) // 2]
            rows = [rows[index] for index in dict.fromkeys(indices)]
        views = []
        prefix = f"{_safe(rec)}__t{_safe(canonical_tid)}"
        for index, row in enumerate(rows):
            full_source = (row.get("full_path") or
                           os.path.join(views_dir, row["stem"] + "_full.jpg"))
            crop_source = (row.get("crop_path") or
                           os.path.join(views_dir, row["stem"] + "_crop.jpg"))
            full_name = f"{prefix}__v{index}_full.jpg"
            crop_name = f"{prefix}__v{index}_crop.jpg"
            shutil.copy2(full_source, os.path.join(image_dir, full_name))
            shutil.copy2(crop_source, os.path.join(image_dir, crop_name))
            views.append({"frame": row["frame"],
                          "full": "images/" + full_name,
                          "crop": "images/" + crop_name})
        items.append({"rec": rec, "canonical_tid": canonical_tid,
                      "n_views": len(views), "views": views})
    return items


def page(items):
    payload = json.dumps(items, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Canonical Track 多视图盲审</title>
<style>
:root {{ color-scheme: dark; font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
* {{ box-sizing: border-box; }}
body {{ margin:0; background:#121514; color:#f5f6f5; }}
header {{ position:sticky; top:0; z-index:2; display:flex; align-items:center; gap:18px; padding:14px 20px; background:#181d1b; border-bottom:1px solid #46504c; }}
h1 {{ margin:0; font-size:20px; }} #progress {{ color:#bac5c0; font-variant-numeric:tabular-nums; }}
main {{ max-width:1500px; margin:0 auto; padding:18px; }}
.meta {{ display:flex; justify-content:space-between; gap:16px; margin-bottom:12px; color:#d3d9d6; }}
.views {{ display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:8px; }}
.view {{ min-width:0; }} .frame {{ margin:0 0 5px; color:#b9c3bf; text-align:center; font-size:14px; }}
.full,.crop {{ display:block; width:100%; object-fit:contain; background:#090b0a; border:1px solid #46504c; }}
.full {{ aspect-ratio:16/10; }} .crop {{ aspect-ratio:1; margin-top:5px; }}
.controls {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; margin-top:18px; }}
button {{ min-height:58px; border:1px solid #68746f; border-radius:6px; background:#252c29; color:white; font-size:18px; cursor:pointer; }}
button:hover,button:focus-visible {{ border-color:#a6d9c4; outline:none; }} button.selected {{ box-shadow:inset 0 0 0 2px #76d5ad; border-color:#76d5ad; }}
.hand {{ background:#174d38; }} .nothand {{ background:#6a2c2c; }} .mixed {{ background:#65421e; }} .unsure {{ background:#4b4656; }}
textarea {{ width:100%; min-height:70px; margin-top:14px; padding:10px; resize:vertical; border:1px solid #56615d; border-radius:6px; background:#1d2220; color:white; font-size:16px; }}
.nav {{ display:flex; justify-content:space-between; margin-top:12px; }} .nav button,#download {{ min-height:44px; padding:0 18px; font-size:16px; }}
@media (max-width:900px) {{ .views {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} .controls {{ grid-template-columns:1fr 1fr; }} header {{ flex-wrap:wrap; }} }}
@media (max-width:520px) {{ .views {{ grid-template-columns:1fr; }} .controls {{ grid-template-columns:1fr; }} }}
</style></head>
<body><header><h1>Canonical Track 多视图盲审</h1><span id="progress"></span><button id="download">导出 CSV</button></header>
<main><div class="meta"><span id="position"></span><span id="identity"></span></div><div id="views" class="views"></div>
<div class="controls"><button class="hand" data-label="hand">1　全是真手</button><button class="nothand" data-label="nothand">2　全不是手</button><button class="mixed" data-label="mixed">3　混合 / 串轨</button><button class="unsure" data-label="unsure">4　看不清</button></div>
<textarea id="note" placeholder="备注，可不填"></textarea><div class="nav"><button id="prev">← 上一条</button><button id="next">下一条 →</button></div></main>
<script>
const items={payload}; const key="handness-canonical-track-review"; const saved=JSON.parse(localStorage.getItem(key)||"{{}}");
let index=Math.min(Number(localStorage.getItem(key+"-index")||0),Math.max(0,items.length-1));
function id(x){{return x.rec+"|"+x.canonical_tid}} function current(){{return items[index]}} function state(){{return saved[id(current())]||{{label:"",note:""}}}}
function persist(){{localStorage.setItem(key,JSON.stringify(saved));localStorage.setItem(key+"-index",index)}}
function render(){{const item=current(),value=state();document.querySelector("#position").textContent=`第 ${{index+1}} / ${{items.length}} 条`;document.querySelector("#identity").textContent=`${{item.rec}}　canonical track ${{item.canonical_tid}}`;
document.querySelector("#views").innerHTML=item.views.map(v=>`<div class="view"><p class="frame">frame ${{v.frame}}</p><img class="full" src="${{v.full}}" alt="完整上下文"><img class="crop" src="${{v.crop}}" alt="候选框裁剪"></div>`).join("");
document.querySelector("#progress").textContent=`已标 ${{Object.values(saved).filter(x=>x.label).length}} / ${{items.length}}`;document.querySelector("#note").value=value.note||"";document.querySelectorAll("[data-label]").forEach(b=>b.classList.toggle("selected",b.dataset.label===value.label));}}
function choose(label){{saved[id(current())]={{...state(),label}};persist();if(index<items.length-1)index++;render()}}
document.querySelectorAll("[data-label]").forEach(b=>b.onclick=()=>choose(b.dataset.label));document.querySelector("#note").oninput=e=>{{saved[id(current())]={{...state(),note:e.target.value}};persist()}};
document.querySelector("#prev").onclick=()=>{{index=Math.max(0,index-1);persist();render()}};document.querySelector("#next").onclick=()=>{{index=Math.min(items.length-1,index+1);persist();render()}};
document.onkeydown=e=>{{if(e.target.tagName==="TEXTAREA")return;if(["1","2","3","4"].includes(e.key))choose(["hand","nothand","mixed","unsure"][Number(e.key)-1]);if(e.key==="ArrowLeft")document.querySelector("#prev").click();if(e.key==="ArrowRight")document.querySelector("#next").click();}};
function cell(v){{return `"${{String(v??"").replaceAll('"','""')}}"`}} document.querySelector("#download").onclick=()=>{{const lines=["rec,canonical_tid,label,note,n_views"];items.forEach(item=>{{const value=saved[id(item)]||{{}};lines.push([item.rec,item.canonical_tid,value.label||"",value.note||"",item.n_views].map(cell).join(","))}});const blob=new Blob(["\ufeff"+lines.join("\\n")],{{type:"text/csv;charset=utf-8"}});const a=document.createElement("a");a.href=URL.createObjectURL(blob);a.download="handness_track_review.csv";a.click();URL.revokeObjectURL(a.href)}};render();
</script></body></html>"""


def build(index_path, views_dir, out_dir, max_views=5):
    os.makedirs(out_dir, exist_ok=True)
    items = track_items(index_path, views_dir, out_dir, max_views)
    with open(os.path.join(out_dir, "review.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows({"rec": item["rec"], "canonical_tid": item["canonical_tid"],
                          "label": "", "note": "", "n_views": item["n_views"]}
                         for item in items)
    with open(os.path.join(out_dir, "打开标注页面.html"), "w", encoding="utf-8") as handle:
        handle.write(page(items))
    with open(os.path.join(out_dir, "README 使用说明.txt"), "w", encoding="utf-8") as handle:
        handle.write("每页是一条 canonical track 的多个时间点。\n"
                     "1=全是真手；2=全不是手；3=混合/串轨；4=看不清。\n"
                     "混合轨迹会 fail-open 保留，并单独统计为跟踪错误。\n"
                     "进度自动保存在浏览器；完成后点击导出 CSV。\n")
    return len(items)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True)
    parser.add_argument("--views", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-views", type=int, default=5)
    args = parser.parse_args()
    print(f"{build(args.index, args.views, args.out, args.max_views)} tracks -> {args.out}")


if __name__ == "__main__":
    main()
