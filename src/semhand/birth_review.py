"""Build an offline, track-grouped reviewer for a birth-probe package."""

from __future__ import annotations

import argparse
import csv
import json
from collections import OrderedDict
from pathlib import Path


LABELS = {"owner", "other", "nothand", "unsure"}


PAGE = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>手部归属盲判</title>
<style>
:root {
  color-scheme: light;
  --ink: #18201d;
  --muted: #5c6862;
  --line: #cfd7d2;
  --paper: #ffffff;
  --ground: #eef1ee;
  --accent: #087f66;
  --owner: #176b4d;
  --other: #315f8d;
  --nothand: #9a4d1f;
  --unsure: #545d62;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--ground);
  color: var(--ink);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", sans-serif;
  font-size: 18px;
}
button { font: inherit; letter-spacing: 0; }
.topbar {
  position: sticky;
  top: 0;
  z-index: 20;
  display: grid;
  grid-template-columns: auto minmax(180px, 1fr) auto;
  gap: 18px;
  align-items: center;
  padding: 12px 20px;
  border-bottom: 1px solid var(--line);
  background: rgba(255, 255, 255, .97);
}
.brand { font-size: 22px; font-weight: 750; white-space: nowrap; }
.progress-wrap { min-width: 0; }
.progress-copy { display: flex; justify-content: space-between; gap: 12px; margin-bottom: 6px; color: var(--muted); font-size: 15px; }
.progress-track { height: 10px; overflow: hidden; background: #dfe5e1; border-radius: 5px; }
.progress-fill { width: 0; height: 100%; background: var(--accent); transition: width .2s ease; }
.top-actions { display: flex; gap: 8px; }
.icon-btn, .export-btn {
  min-height: 44px;
  border: 1px solid #aab5af;
  border-radius: 6px;
  background: var(--paper);
  color: var(--ink);
  cursor: pointer;
}
.icon-btn { width: 48px; font-size: 25px; }
.export-btn { padding: 0 16px; font-weight: 700; background: #e7f4ef; border-color: #79a99a; }
.export-btn:disabled { cursor: not-allowed; color: #8a938f; background: #edf0ee; border-color: #cbd2ce; }
main { max-width: 1580px; margin: 0 auto; padding: 20px; }
.question-row {
  display: flex;
  justify-content: space-between;
  gap: 20px;
  align-items: flex-end;
  margin-bottom: 14px;
}
h1 { margin: 0 0 5px; font-size: clamp(24px, 3vw, 34px); line-height: 1.2; letter-spacing: 0; }
.meta { margin: 0; color: var(--muted); font-size: 16px; }
.autosave { color: var(--accent); font-size: 15px; white-space: nowrap; }
.labels {
  display: grid;
  grid-template-columns: repeat(4, minmax(150px, 1fr));
  gap: 10px;
  margin-bottom: 18px;
}
.label-btn {
  min-height: 70px;
  padding: 10px 12px;
  border: 3px solid transparent;
  border-radius: 7px;
  color: white;
  cursor: pointer;
  font-size: 20px;
  font-weight: 750;
  line-height: 1.15;
}
.label-btn small { display: block; margin-top: 5px; color: rgba(255,255,255,.86); font-size: 13px; font-weight: 650; }
.label-btn[data-label="owner"] { background: var(--owner); }
.label-btn[data-label="other"] { background: var(--other); }
.label-btn[data-label="nothand"] { background: var(--nothand); }
.label-btn[data-label="unsure"] { background: var(--unsure); }
.label-btn.selected { border-color: #ffcd39; box-shadow: 0 0 0 2px #5a4300; }
.frames {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
  gap: 10px;
}
.frame {
  margin: 0;
  overflow: hidden;
  border: 1px solid #aeb8b2;
  border-radius: 6px;
  background: var(--paper);
}
.frame button { display: block; width: 100%; padding: 0; border: 0; background: #101210; cursor: zoom-in; }
.frame img { display: block; width: 100%; aspect-ratio: 16 / 9; object-fit: contain; }
.frame figcaption { padding: 8px 10px; color: #46514b; font-size: 15px; text-align: center; }
.track-nav {
  display: grid;
  grid-template-columns: repeat(9, minmax(42px, 1fr));
  gap: 7px;
  margin-top: 18px;
  padding-top: 15px;
  border-top: 1px solid var(--line);
}
.track-dot {
  min-height: 44px;
  border: 1px solid #aeb8b2;
  border-radius: 5px;
  background: var(--paper);
  color: var(--ink);
  cursor: pointer;
  font-weight: 700;
}
.track-dot.done { color: white; background: var(--accent); border-color: var(--accent); }
.track-dot.current { outline: 3px solid #ffcd39; outline-offset: 1px; }
dialog { width: min(96vw, 1500px); max-width: none; padding: 10px; border: 0; border-radius: 6px; background: #111; }
dialog::backdrop { background: rgba(0, 0, 0, .82); }
dialog img { display: block; width: 100%; max-height: 88vh; object-fit: contain; }
.close-btn { position: absolute; top: 18px; right: 18px; width: 48px; height: 48px; border: 0; border-radius: 50%; background: white; color: #111; font-size: 28px; cursor: pointer; }
@media (max-width: 850px) {
  .topbar { grid-template-columns: 1fr auto; }
  .progress-wrap { grid-column: 1 / -1; grid-row: 2; }
  .labels { grid-template-columns: repeat(2, minmax(135px, 1fr)); }
  .question-row { align-items: flex-start; flex-direction: column; gap: 5px; }
  .track-nav { grid-template-columns: repeat(5, minmax(42px, 1fr)); }
}
@media (max-width: 480px) {
  body { font-size: 16px; }
  main { padding: 14px; }
  .topbar { padding: 10px 12px; gap: 10px; }
  .brand { font-size: 18px; }
  .export-btn { padding: 0 10px; }
  .labels { grid-template-columns: 1fr 1fr; }
  .label-btn { min-height: 76px; font-size: 18px; }
  .frames { grid-template-columns: 1fr; }
}
</style>
</head>
<body>
<header class="topbar">
  <div class="brand">手部归属盲判</div>
  <div class="progress-wrap">
    <div class="progress-copy"><span id="position"></span><span id="doneCopy"></span></div>
    <div class="progress-track" aria-hidden="true"><div class="progress-fill" id="progressFill"></div></div>
  </div>
  <div class="top-actions">
    <button class="icon-btn" id="prevBtn" type="button" title="上一条" aria-label="上一条">&#8592;</button>
    <button class="icon-btn" id="nextBtn" type="button" title="下一条" aria-label="下一条">&#8594;</button>
    <button class="export-btn" id="exportBtn" type="button">导出 CSV</button>
  </div>
</header>
<main>
  <section class="question-row" aria-labelledby="question">
    <div>
      <h1 id="question">绿框里的是谁的手？</h1>
      <p class="meta" id="meta"></p>
    </div>
    <div class="autosave" id="saveState">草稿自动保存</div>
  </section>
  <section class="labels" aria-label="选择答案">
    <button class="label-btn" type="button" data-label="owner">佩戴者自己的手<small>owner</small></button>
    <button class="label-btn" type="button" data-label="other">别人的手<small>other</small></button>
    <button class="label-btn" type="button" data-label="nothand">不是手<small>nothand</small></button>
    <button class="label-btn" type="button" data-label="unsure">看不清<small>unsure</small></button>
  </section>
  <section class="frames" id="frames" aria-label="同一条轨迹的所有画面"></section>
  <nav class="track-nav" id="trackNav" aria-label="轨迹进度"></nav>
</main>
<dialog id="zoomDialog">
  <button class="close-btn" id="closeZoom" type="button" aria-label="关闭">&#215;</button>
  <img id="zoomImage" alt="放大的标注画面">
</dialog>
<script>
const DATA = __PAYLOAD__;
const STORAGE_KEY = "birth_review_" + DATA.packageId;
const VALID_LABELS = new Set(["owner", "other", "nothand", "unsure"]);
let answers = {};
try {
  const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
  for (const [key, value] of Object.entries(saved)) {
    if (VALID_LABELS.has(value)) answers[key] = value;
  }
} catch (_) {}
for (const group of DATA.groups) {
  if (!answers[group.id] && group.initialLabel) answers[group.id] = group.initialLabel;
}
let index = Math.max(0, DATA.groups.findIndex(group => !answers[group.id]));
if (index < 0) index = 0;

const position = document.getElementById("position");
const doneCopy = document.getElementById("doneCopy");
const progressFill = document.getElementById("progressFill");
const meta = document.getElementById("meta");
const frames = document.getElementById("frames");
const trackNav = document.getElementById("trackNav");
const exportBtn = document.getElementById("exportBtn");
const zoomDialog = document.getElementById("zoomDialog");
const zoomImage = document.getElementById("zoomImage");

function saveDraft() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(answers));
  const state = document.getElementById("saveState");
  state.textContent = "已保存";
  window.setTimeout(() => { state.textContent = "草稿自动保存"; }, 900);
}

function go(step) {
  index = Math.max(0, Math.min(DATA.groups.length - 1, index + step));
  render();
}

function choose(label) {
  const group = DATA.groups[index];
  answers[group.id] = label;
  saveDraft();
  render();
  if (index < DATA.groups.length - 1) {
    window.setTimeout(() => { index += 1; render(); }, 260);
  }
}

function openZoom(src) {
  zoomImage.src = src;
  zoomDialog.showModal();
}

function render() {
  const group = DATA.groups[index];
  const done = DATA.groups.filter(item => answers[item.id]).length;
  const frameDone = DATA.groups.reduce((sum, item) => sum + (answers[item.id] ? item.rows.length : 0), 0);
  position.textContent = `第 ${index + 1} / ${DATA.groups.length} 条轨迹`;
  doneCopy.textContent = `已完成 ${done} 条，${frameDone} / ${DATA.rows.length} 框`;
  progressFill.style.width = `${100 * done / DATA.groups.length}%`;
  meta.textContent = `${group.rec} · tid ${group.tid} · ${group.rows.length} 帧一起判断`;
  document.querySelectorAll(".label-btn").forEach(button => {
    button.classList.toggle("selected", button.dataset.label === answers[group.id]);
  });
  frames.replaceChildren(...group.rows.map(row => {
    const figure = document.createElement("figure");
    figure.className = "frame";
    const button = document.createElement("button");
    button.type = "button";
    button.title = "点击放大";
    const image = document.createElement("img");
    image.src = row.image;
    image.alt = `帧 ${row.frame}，绿框为待判断区域`;
    button.appendChild(image);
    button.addEventListener("click", () => openZoom(row.image));
    const caption = document.createElement("figcaption");
    caption.textContent = `帧 ${row.frame}`;
    figure.append(button, caption);
    return figure;
  }));
  trackNav.replaceChildren(...DATA.groups.map((item, itemIndex) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "track-dot";
    if (answers[item.id]) button.classList.add("done");
    if (itemIndex === index) button.classList.add("current");
    button.textContent = String(itemIndex + 1).padStart(2, "0");
    button.title = `${item.rec} · tid ${item.tid}`;
    button.addEventListener("click", () => { index = itemIndex; render(); });
    return button;
  }));
  document.getElementById("prevBtn").disabled = index === 0;
  document.getElementById("nextBtn").disabled = index === DATA.groups.length - 1;
  exportBtn.disabled = done !== DATA.groups.length;
  exportBtn.textContent = done === DATA.groups.length ? "导出 CSV" : `还差 ${DATA.groups.length - done} 条`;
}

function csvCell(value) {
  const text = String(value ?? "");
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function exportCsv() {
  if (DATA.groups.some(group => !answers[group.id])) return;
  const groupForRow = new Map();
  for (const group of DATA.groups) {
    for (const row of group.rows) groupForRow.set(row.rowIndex, group.id);
  }
  const lines = [DATA.fields.map(csvCell).join(",")];
  DATA.rows.forEach((row, rowIndex) => {
    const output = {...row, label: answers[groupForRow.get(rowIndex)]};
    lines.push(DATA.fields.map(field => csvCell(output[field])).join(","));
  });
  const blob = new Blob(["\ufeff" + lines.join("\r\n") + "\r\n"], {type: "text/csv;charset=utf-8"});
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = "hands_labeled.csv";
  link.click();
  URL.revokeObjectURL(link.href);
}

document.querySelectorAll(".label-btn").forEach(button => button.addEventListener("click", () => choose(button.dataset.label)));
document.getElementById("prevBtn").addEventListener("click", () => go(-1));
document.getElementById("nextBtn").addEventListener("click", () => go(1));
exportBtn.addEventListener("click", exportCsv);
document.getElementById("closeZoom").addEventListener("click", () => zoomDialog.close());
zoomDialog.addEventListener("click", event => { if (event.target === zoomDialog) zoomDialog.close(); });
document.addEventListener("keydown", event => {
  if (zoomDialog.open) return;
  if (event.key === "ArrowLeft") go(-1);
  if (event.key === "ArrowRight") go(1);
  if (event.key === "1") choose("owner");
  if (event.key === "2") choose("other");
  if (event.key === "3") choose("nothand");
  if (event.key === "4") choose("unsure");
});
render();
</script>
</body>
</html>
"""


def build(package: Path, out: Path) -> tuple[int, int]:
    csv_path = package / "hands.csv"
    context = package / "context"
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise SystemExit(f"No CSV header in {csv_path}")
        fields = list(reader.fieldnames)
        rows = list(reader)

    required = {"stem", "rec", "tid", "frame", "label"}
    missing = required.difference(fields)
    if missing:
        raise SystemExit(f"Missing columns: {', '.join(sorted(missing))}")

    grouped: OrderedDict[tuple[str, str], list[dict[str, str]]] = OrderedDict()
    for row_index, row in enumerate(rows):
        image_name = f"{row['stem']}.jpg"
        image_path = context / image_name
        if not image_path.is_file():
            raise SystemExit(f"Missing image: {image_path}")
        row["_row_index"] = row_index
        row["_image"] = f"context/{image_name}"
        grouped.setdefault((row["rec"], row["tid"]), []).append(row)

    groups = []
    for (rec, tid), group_rows in grouped.items():
        present = {row.get("label", "").strip() for row in group_rows}
        present.discard("")
        initial = next(iter(present)) if len(present) == 1 and present <= LABELS else ""
        groups.append(
            {
                "id": f"{rec}__t{tid}",
                "rec": rec,
                "tid": tid,
                "initialLabel": initial,
                "rows": [
                    {
                        "rowIndex": row["_row_index"],
                        "frame": row["frame"],
                        "image": row["_image"],
                    }
                    for row in group_rows
                ],
            }
        )

    clean_rows = [{field: row.get(field, "") for field in fields} for row in rows]
    payload = {
        "packageId": package.name.replace(" ", "_") + "_v1",
        "fields": fields,
        "rows": clean_rows,
        "groups": groups,
    }
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    out.write_text(PAGE.replace("__PAYLOAD__", encoded), encoding="utf-8")
    return len(groups), len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    package = args.package.expanduser().resolve()
    out = args.out.expanduser().resolve() if args.out else package / "打开标注页面.html"
    groups, rows = build(package, out)
    print(f"Wrote {out} ({groups} tracks, {rows} frames)")


if __name__ == "__main__":
    main()
