#!/usr/bin/env python3
"""Build a self-contained blind review packet for Handness V2 labels."""
from __future__ import annotations

import argparse
import csv
import html
import json
import os
import shutil


REVIEW_FIELDS = ["item_id", "rec", "frame", "label", "note"]


def review_rows(manifest, status="required"):
    rows = list(csv.DictReader(open(manifest, encoding="utf-8")))
    return [row for row in rows if row["review_status"] == status]


def copy_view(source, destination):
    if os.path.exists(destination):
        return
    shutil.copy2(source, destination)


def page(items):
    payload = json.dumps(items, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Handness V2 盲审</title>
<style>
:root {{ color-scheme: dark; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: #121514; color: #f4f5f4; }}
header {{ position: sticky; top: 0; z-index: 2; display: flex; align-items: center; gap: 18px; padding: 14px 20px; background: #181d1b; border-bottom: 1px solid #46504c; }}
h1 {{ margin: 0; font-size: 20px; }}
#progress {{ color: #b9c4bf; font-variant-numeric: tabular-nums; }}
main {{ max-width: 1320px; margin: 0 auto; padding: 18px; }}
.meta {{ display: flex; justify-content: space-between; gap: 16px; margin-bottom: 12px; color: #c9d0cd; }}
.views {{ display: grid; grid-template-columns: minmax(0, 1fr) 320px; gap: 14px; align-items: start; }}
.views img {{ display: block; width: 100%; max-height: 68vh; object-fit: contain; background: #090b0a; border: 1px solid #46504c; border-radius: 6px; }}
.controls {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; margin-top: 16px; }}
button {{ min-height: 58px; border: 1px solid #68746f; border-radius: 6px; background: #252c29; color: white; font-size: 18px; cursor: pointer; }}
button:hover, button:focus-visible {{ border-color: #9fd7c0; outline: none; }}
button.selected {{ border-color: #76d5ad; box-shadow: inset 0 0 0 2px #76d5ad; }}
.hand {{ background: #174d38; }} .nothand {{ background: #6a2c2c; }} .unsure {{ background: #66531f; }}
.nav {{ display: flex; justify-content: space-between; margin-top: 12px; }}
.nav button, #download {{ min-height: 44px; padding: 0 18px; font-size: 16px; }}
textarea {{ width: 100%; min-height: 72px; margin-top: 14px; padding: 10px; resize: vertical; border: 1px solid #56615d; border-radius: 6px; background: #1d2220; color: white; font-size: 16px; }}
@media (max-width: 760px) {{ .views {{ grid-template-columns: 1fr; }} .views img:last-child {{ max-height: 34vh; }} header {{ flex-wrap: wrap; }} }}
</style>
</head>
<body>
<header><h1>Handness V2 盲审</h1><span id="progress"></span><button id="download">导出 CSV</button></header>
<main>
  <div class="meta"><span id="position"></span><span id="identity"></span></div>
  <div class="views"><img id="full" alt="完整上下文"><img id="crop" alt="候选框裁剪"></div>
  <div class="controls">
    <button class="hand" data-label="hand">1　是手</button>
    <button class="nothand" data-label="nothand">2　不是手</button>
    <button class="unsure" data-label="unsure">3　看不清</button>
  </div>
  <textarea id="note" placeholder="备注，可不填"></textarea>
  <div class="nav"><button id="prev">← 上一条</button><button id="next">下一条 →</button></div>
</main>
<script>
const items = {payload};
const key = "handness-v2-review";
const saved = JSON.parse(localStorage.getItem(key) || "{{}}");
let index = Math.min(Number(localStorage.getItem(key + "-index") || 0), Math.max(0, items.length - 1));
function current() {{ return items[index]; }}
function state() {{ return saved[current().item_id] || {{label:"", note:""}}; }}
function persist() {{ localStorage.setItem(key, JSON.stringify(saved)); localStorage.setItem(key + "-index", index); }}
function render() {{
  const item = current(), value = state();
  document.querySelector("#full").src = item.full;
  document.querySelector("#crop").src = item.crop;
  document.querySelector("#position").textContent = `第 ${{index + 1}} / ${{items.length}} 条`;
  document.querySelector("#identity").textContent = `${{item.rec}}　frame ${{item.frame}}`;
  const done = Object.values(saved).filter(x => x.label).length;
  document.querySelector("#progress").textContent = `已标 ${{done}} / ${{items.length}}`;
  document.querySelector("#note").value = value.note || "";
  document.querySelectorAll("[data-label]").forEach(button => button.classList.toggle("selected", button.dataset.label === value.label));
}}
function choose(label) {{ saved[current().item_id] = {{...state(), label}}; persist(); if (index < items.length - 1) index++; render(); }}
document.querySelectorAll("[data-label]").forEach(button => button.onclick = () => choose(button.dataset.label));
document.querySelector("#note").oninput = event => {{ saved[current().item_id] = {{...state(), note:event.target.value}}; persist(); }};
document.querySelector("#prev").onclick = () => {{ index = Math.max(0, index - 1); persist(); render(); }};
document.querySelector("#next").onclick = () => {{ index = Math.min(items.length - 1, index + 1); persist(); render(); }};
document.onkeydown = event => {{ if (event.target.tagName === "TEXTAREA") return; if (event.key === "1") choose("hand"); if (event.key === "2") choose("nothand"); if (event.key === "3") choose("unsure"); if (event.key === "ArrowLeft") document.querySelector("#prev").click(); if (event.key === "ArrowRight") document.querySelector("#next").click(); }};
function csvCell(value) {{ return `"${{String(value ?? "").replaceAll('"','""')}}"`; }}
document.querySelector("#download").onclick = () => {{
  const lines = ["item_id,rec,frame,label,note"];
  items.forEach(item => {{ const value = saved[item.item_id] || {{}}; lines.push([item.item_id,item.rec,item.frame,value.label||"",value.note||""].map(csvCell).join(",")); }});
  const blob = new Blob(["\ufeff" + lines.join("\\n")], {{type:"text/csv;charset=utf-8"}});
  const anchor = document.createElement("a"); anchor.href = URL.createObjectURL(blob); anchor.download = "handness_v2_review.csv"; anchor.click(); URL.revokeObjectURL(anchor.href);
}};
render();
</script>
</body>
</html>"""


def build(manifest, out_dir, status="required"):
    rows = review_rows(manifest, status)
    image_dir = os.path.join(out_dir, "images")
    os.makedirs(image_dir, exist_ok=True)
    items = []
    review = []
    for row in rows:
        item_id = row["item_id"]
        full_name = item_id + "_full.jpg"
        crop_name = item_id + "_crop.jpg"
        copy_view(row["full_path"], os.path.join(image_dir, full_name))
        copy_view(row["crop_path"], os.path.join(image_dir, crop_name))
        items.append({"item_id": item_id, "rec": row["rec"],
                      "frame": row["frame"],
                      "full": "images/" + full_name,
                      "crop": "images/" + crop_name})
        review.append({"item_id": item_id, "rec": row["rec"],
                       "frame": row["frame"], "label": "", "note": ""})
    with open(os.path.join(out_dir, "review.csv"), "w", newline="",
              encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(review)
    with open(os.path.join(out_dir, "打开标注页面.html"), "w",
              encoding="utf-8") as handle:
        handle.write(page(items))
    with open(os.path.join(out_dir, "README 使用说明.txt"), "w",
              encoding="utf-8") as handle:
        handle.write("双击 打开标注页面.html。\n1=是手，2=不是手，3=看不清。\n"
                     "进度自动保存在浏览器中；完成后点击导出 CSV。\n")
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--status", default="required")
    args = parser.parse_args()
    print(f"{build(args.manifest, args.out, args.status)} items -> {args.out}")


if __name__ == "__main__":
    main()
