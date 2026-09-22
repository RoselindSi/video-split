"""Annotate a whole video as reusable event-node segments and export its graph.

Each video is partitioned exhaustively.  A segment points to a recording-local
event node (A, B, ...), carries an optional note, and may explicitly say that
the same event repeats inside the segment.  Reusing A after B records B -> A;
assigning A to consecutive segments records A -> A.  A one-segment repeated
action can also record A -> A with ``repeats_within_segment``.

Run on the machine that stores the videos and reach it through an SSH tunnel:

    python -m src.auditor.boundary.interaction_graph_timeline \
      --manifest results/auditor/interaction_graph_timeline_demo/manifest.json \
      --out-dir results/auditor/interaction_graph_timeline_demo/annotations \
      --annotator reviewer_1 --port 18772

The server refuses non-loopback binds.  The videos may contain identifiable
people and this focused tool intentionally has no public-network authentication.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import html
import io
import json
import math
import re
import urllib.parse
from collections import Counter, defaultdict
from pathlib import Path

from src.auditor.boundary.final_timeline_audit import (
    AuditServer,
    Handler as VideoHandler,
    atomic_json,
    file_sha256,
    safe_name,
)


SCHEMA = "interaction_graph_timeline_v1"
MANIFEST_SCHEMA = "interaction_graph_timeline_manifest_v1"
VIDEO_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
EVENT_ID_RE = re.compile(r"[A-Z]{1,4}")
MAX_BODY = 8 << 20
EPSILON = 0.05


def _finite_number(value: object, field: str) -> float:
    try:
        number = round(float(value), 3)
    except (TypeError, ValueError):
        raise ValueError(f"{field} 不是有效时间") from None
    if not math.isfinite(number):
        raise ValueError(f"{field} 不是有限数字")
    return number


def load_manifest(path: str | Path) -> tuple[dict, dict[str, dict], str]:
    manifest_path = Path(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError(f"unexpected manifest schema: {manifest.get('schema_version')!r}")
    rows = manifest.get("videos")
    if not isinstance(rows, list) or not rows:
        raise ValueError("manifest videos must be a non-empty list")
    videos: dict[str, dict] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("manifest video entry is not an object")
        video_id = str(row.get("video_id") or "")
        if not VIDEO_ID_RE.fullmatch(video_id) or video_id in videos:
            raise ValueError(f"invalid or duplicate video_id: {video_id!r}")
        video = Path(str(row.get("video") or ""))
        if not video.is_file():
            raise FileNotFoundError(f"video is missing: {video}")
        duration = _finite_number(row.get("duration_s"), "duration_s")
        if duration <= 0:
            raise ValueError(f"video has no positive duration: {video_id}")
        expected_hash = str(row.get("video_sha256") or "")
        if expected_hash and file_sha256(video) != expected_hash:
            raise ValueError(f"video changed after manifest preparation: {video_id}")
        videos[video_id] = {
            **row,
            "video_id": video_id,
            "video": str(video),
            "duration_s": duration,
            "source_pool": str(row.get("source_pool") or "未分类")[:120],
            "source_ref": str(row.get("source_ref") or "")[:1000],
        }
    declared = manifest.get("n_videos")
    if declared is not None and int(declared) != len(videos):
        raise ValueError(f"manifest declares {declared} videos but contains {len(videos)}")
    return manifest, videos, file_sha256(manifest_path)


def blank_state(video_id: str, duration_s: float) -> dict:
    return {
        "schema_version": SCHEMA,
        "video_id": video_id,
        "events": [],
        "segments": [{
            "start_s": 0.0,
            "end_s": duration_s,
            "event_id": "",
            "repeats_within_segment": False,
            "note": "",
        }],
        "confirmed_full_video": False,
    }


def validate_submission(value: object, video_id: str, duration_s: float,
                        final: bool = False) -> dict:
    if not isinstance(value, dict) or value.get("video_id") != video_id:
        raise ValueError("video_id 与页面不一致")

    raw_events = value.get("events")
    raw_segments = value.get("segments")
    if not isinstance(raw_events, list) or not isinstance(raw_segments, list):
        raise ValueError("events 和 segments 必须是列表")
    if len(raw_events) > 1000 or not 1 <= len(raw_segments) <= 5000:
        raise ValueError("事件或区间数量异常")

    events = []
    event_ids = set()
    for index, event in enumerate(raw_events):
        if not isinstance(event, dict):
            raise ValueError(f"事件 {index + 1} 格式错误")
        event_id = str(event.get("event_id") or "").strip().upper()
        if not EVENT_ID_RE.fullmatch(event_id) or event_id in event_ids:
            raise ValueError(f"事件编号无效或重复：{event_id or '(空)'}")
        description = str(event.get("description") or "").strip()[:500]
        events.append({"event_id": event_id, "description": description})
        event_ids.add(event_id)

    segments = []
    for index, segment in enumerate(raw_segments):
        if not isinstance(segment, dict):
            raise ValueError(f"第 {index + 1} 段格式错误")
        start = _finite_number(segment.get("start_s"), "start_s")
        end = _finite_number(segment.get("end_s"), "end_s")
        if not 0 <= start < end <= duration_s + EPSILON:
            raise ValueError(f"第 {index + 1} 段时间超出视频或长度为零")
        event_id = str(segment.get("event_id") or "").strip().upper()
        if event_id and event_id not in event_ids:
            raise ValueError(f"第 {index + 1} 段引用了不存在的事件 {event_id}")
        repeats = segment.get("repeats_within_segment")
        if not isinstance(repeats, bool):
            raise ValueError(f"第 {index + 1} 段的重复标记必须为真或假")
        segments.append({
            "start_s": start,
            "end_s": end,
            "event_id": event_id,
            "repeats_within_segment": repeats,
            "note": str(segment.get("note") or "").strip()[:1000],
        })

    segments.sort(key=lambda item: (item["start_s"], item["end_s"]))
    if abs(segments[0]["start_s"]) > EPSILON:
        raise ValueError("第一段必须从视频开头开始")
    if abs(segments[-1]["end_s"] - duration_s) > EPSILON:
        raise ValueError("最后一段必须到视频结尾")
    segments[0]["start_s"] = 0.0
    segments[-1]["end_s"] = duration_s
    for index in range(1, len(segments)):
        previous = segments[index - 1]
        current = segments[index]
        if abs(previous["end_s"] - current["start_s"]) > EPSILON:
            raise ValueError(f"第 {index} 段和第 {index + 1} 段之间有空白或重叠")
        current["start_s"] = previous["end_s"]

    confirmed = value.get("confirmed_full_video") is True
    used = {segment["event_id"] for segment in segments if segment["event_id"]}
    descriptions = {event["event_id"]: event["description"] for event in events}
    if final:
        missing = [index + 1 for index, segment in enumerate(segments)
                   if not segment["event_id"]]
        if missing:
            raise ValueError(f"这些区间还没有选择事件：{', '.join(map(str, missing[:12]))}")
        empty = [event_id for event_id in sorted(used) if not descriptions[event_id]]
        if empty:
            raise ValueError(f"请填写事件描述：{', '.join(empty)}")
        unused = sorted(event_ids - used)
        if unused:
            raise ValueError(f"请删除未使用的事件：{', '.join(unused)}")
        if not confirmed:
            raise ValueError("请确认已经从头到尾看完视频")

    return {
        "schema_version": SCHEMA,
        "video_id": video_id,
        "events": events,
        "segments": segments,
        "confirmed_full_video": confirmed,
    }


def build_graph(annotation: dict) -> dict:
    descriptions = {event["event_id"]: event["description"]
                    for event in annotation["events"]}
    durations: Counter[str] = Counter()
    segment_indices: defaultdict[str, list[int]] = defaultdict(list)
    occurrences = []

    for index, segment in enumerate(annotation["segments"], 1):
        event_id = segment["event_id"]
        if not event_id:
            continue
        durations[event_id] += segment["end_s"] - segment["start_s"]
        segment_indices[event_id].append(index)
        if segment["repeats_within_segment"]:
            occurrences.append({
                "source": event_id,
                "target": event_id,
                "kind": "internal_repeat",
                "at_s": segment["start_s"],
                "from_segment": index,
                "to_segment": index,
            })

    assigned = [(index, segment) for index, segment
                in enumerate(annotation["segments"], 1) if segment["event_id"]]
    for (left_index, left), (right_index, right) in zip(assigned, assigned[1:]):
        occurrences.append({
            "source": left["event_id"],
            "target": right["event_id"],
            "kind": "self_transition" if left["event_id"] == right["event_id"]
                    else "transition",
            "at_s": right["start_s"],
            "from_segment": left_index,
            "to_segment": right_index,
        })

    aggregate: dict[tuple[str, str], dict] = {}
    for occurrence in occurrences:
        key = (occurrence["source"], occurrence["target"])
        item = aggregate.setdefault(key, {
            "source": key[0], "target": key[1], "count": 0,
            "kinds": Counter(), "times_s": [],
        })
        item["count"] += 1
        item["kinds"][occurrence["kind"]] += 1
        item["times_s"].append(occurrence["at_s"])

    nodes = [{
        "event_id": event_id,
        "description": descriptions.get(event_id, ""),
        "segment_count": len(segment_indices[event_id]),
        "total_duration_s": round(durations[event_id], 3),
        "segment_indices": segment_indices[event_id],
    } for event_id in sorted(segment_indices)]
    edges = []
    for key in sorted(aggregate):
        item = aggregate[key]
        edges.append({
            "source": item["source"],
            "target": item["target"],
            "count": item["count"],
            "kinds": dict(sorted(item["kinds"].items())),
            "times_s": item["times_s"],
        })
    return {
        "nodes": nodes,
        "edges": edges,
        "edge_occurrences": occurrences,
        "event_sequence": [segment["event_id"] for segment in annotation["segments"]
                           if segment["event_id"]],
    }


def page(video: dict, initial: dict | None, token: str) -> bytes:
    payload = {
        "video_id": video["video_id"],
        "duration_s": video["duration_s"],
        "source_pool": video["source_pool"],
        "initial": initial or blank_state(video["video_id"], video["duration_s"]),
        "token": token,
    }
    template = Path(__file__).with_name("interaction_graph_timeline_page.html")
    script_data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    source = template.read_text(encoding="utf-8").replace("__PAYLOAD__", script_data)
    return source.encode("utf-8")


class Store:
    def __init__(self, manifest_path: str | Path, out_dir: str | Path,
                 annotator: str):
        self.manifest, self.videos, self.manifest_sha256 = load_manifest(manifest_path)
        self.out_dir = Path(out_dir)
        self.annotator = safe_name(annotator)
        self.root = self.out_dir / self.annotator
        self.root.mkdir(parents=True, exist_ok=True)
        import secrets
        self.token = secrets.token_urlsafe(32)

    def paths(self, video_id: str) -> tuple[Path, list[Path]]:
        draft = self.root / f"{video_id}.draft.json"
        finals = sorted(self.root.glob(f"{video_id}.final.*.json"))
        return draft, finals

    def state(self, video_id: str) -> dict:
        draft, finals = self.paths(video_id)
        choices = ([draft] if draft.exists() else []) + finals[-1:]
        if not choices:
            video = self.videos[video_id]
            return blank_state(video_id, video["duration_s"])
        selected = max(choices, key=lambda item: item.stat().st_mtime_ns)
        state = json.loads(selected.read_text(encoding="utf-8"))
        if state.get("manifest_sha256") != self.manifest_sha256:
            raise ValueError(f"saved annotation belongs to another manifest: {video_id}")
        return state

    def status(self, video_id: str) -> str:
        draft, finals = self.paths(video_id)
        if finals:
            return "final"
        return "draft" if draft.exists() else "not_started"

    def save(self, video_id: str, raw: object, final: bool) -> dict:
        video = self.videos[video_id]
        value = validate_submission(raw, video_id, video["duration_s"], final=final)
        value.update({
            "manifest_sha256": self.manifest_sha256,
            "annotator": self.annotator,
            "saved_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "status": "final" if final else "draft",
            "graph": build_graph(value),
        })
        draft, _ = self.paths(video_id)
        if final:
            stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            target = self.root / f"{video_id}.final.{stamp}.json"
        else:
            target = draft
        atomic_json(target, value)
        return value

    def latest_finals(self) -> list[dict]:
        rows = []
        for video_id in self.videos:
            _, finals = self.paths(video_id)
            if finals:
                rows.append(json.loads(finals[-1].read_text(encoding="utf-8")))
        return rows

    def export_json(self) -> bytes:
        value = {
            "schema_version": "interaction_graph_timeline_export_v1",
            "manifest_sha256": self.manifest_sha256,
            "annotator": self.annotator,
            "videos": self.latest_finals(),
        }
        return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()

    def export_csv(self) -> bytes:
        stream = io.StringIO(newline="")
        fields = ["video_id", "segment_index", "start_s", "end_s", "event_id",
                  "event_description", "repeats_within_segment", "note",
                  "annotator", "saved_at"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for annotation in self.latest_finals():
            descriptions = {event["event_id"]: event["description"]
                            for event in annotation["events"]}
            for index, segment in enumerate(annotation["segments"], 1):
                writer.writerow({
                    "video_id": annotation["video_id"],
                    "segment_index": index,
                    **segment,
                    "event_description": descriptions.get(segment["event_id"], ""),
                    "annotator": annotation["annotator"],
                    "saved_at": annotation["saved_at"],
                })
        return stream.getvalue().encode("utf-8-sig")


class Handler(VideoHandler):
    store: Store

    def log_message(self, fmt: str, *args: object) -> None:
        if self.command == "POST":
            super().log_message(fmt, *args)

    def _json(self, status: int, value: dict) -> None:
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _html(self, body: bytes) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _download(self, body: bytes, content_type: str, filename: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _index(self) -> bytes:
        status_labels = {"not_started": "未开始", "draft": "草稿", "final": "已提交"}
        rows = []
        counts = Counter()
        total_seconds = 0.0
        for video_id, video in self.store.videos.items():
            status = self.store.status(video_id)
            counts[status] += 1
            total_seconds += video["duration_s"]
            state = self.store.state(video_id)
            assigned = sum(bool(item["event_id"]) for item in state["segments"])
            rows.append(
                f'<tr><td><a href="/r/{urllib.parse.quote(video_id)}">{html.escape(video_id)}</a></td>'
                f'<td>{html.escape(video["source_pool"])}</td>'
                f'<td>{video["duration_s"] / 60:.1f} 分钟</td>'
                f'<td>{assigned}/{len(state["segments"])} 段</td>'
                f'<td class="{status}">{status_labels[status]}</td></tr>')
        body = f"""<!doctype html><html lang=zh-CN><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>动作图标注</title>
<style>:root{{color-scheme:dark;--bg:#111513;--panel:#191e1b;--line:#3f4843;--text:#f2f4f2;--muted:#aab2ad;--accent:#65c29b;--warn:#e7b75f}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:16px/1.5 system-ui,sans-serif;letter-spacing:0}}header{{padding:18px 24px;background:var(--panel);border-bottom:1px solid var(--line);display:flex;gap:16px;align-items:center;flex-wrap:wrap}}h1{{font-size:22px;margin:0}}main{{max-width:1160px;margin:auto;padding:22px}}.summary{{color:var(--muted);margin:0 0 18px}}.actions{{margin-left:auto;display:flex;gap:8px}}a{{color:#9ed8c0}}button{{font:inherit;color:var(--text);background:#252c28;border:1px solid #64706a;border-radius:4px;padding:9px 12px;cursor:pointer}}table{{width:100%;border-collapse:collapse}}th,td{{padding:12px 10px;border-bottom:1px solid #303733;text-align:left}}th{{color:var(--muted);font-size:14px}}.final{{color:#7ed5a8}}.draft{{color:var(--warn)}}.not_started{{color:var(--muted)}}@media(max-width:700px){{main{{padding:14px}}th:nth-child(2),td:nth-child(2){{display:none}}}}</style>
<header><h1>动作图标注</h1><div class=actions><button onclick="location.href='/api/export.csv'">导出 CSV</button><button onclick="location.href='/api/export'">导出 JSON</button></div></header>
<main><p class=summary>已提交 {counts['final']} / {len(rows)}，草稿 {counts['draft']}，共 {total_seconds / 60:.1f} 分钟。</p>
<table><thead><tr><th>视频</th><th>视频池</th><th>长度</th><th>区间</th><th>状态</th></tr></thead><tbody>{''.join(rows)}</tbody></table></main></html>"""
        return body.encode("utf-8")

    def do_GET(self) -> None:
        path = urllib.parse.unquote(urllib.parse.urlparse(self.path).path)
        if path == "/":
            return self._html(self._index())
        if path == "/api/export":
            return self._download(self.store.export_json(),
                                  "application/json; charset=utf-8",
                                  "interaction_graph_annotations.json")
        if path == "/api/export.csv":
            return self._download(self.store.export_csv(),
                                  "text/csv; charset=utf-8",
                                  "interaction_graph_segments.csv")
        for prefix in ("/r/", "/video/", "/api/state/"):
            if path.startswith(prefix):
                video_id = path.removeprefix(prefix)
                if video_id not in self.store.videos:
                    return self.send_error(404)
                if prefix == "/r/":
                    return self._html(page(self.store.videos[video_id],
                                           self.store.state(video_id),
                                           self.store.token))
                if prefix == "/video/":
                    return self._video(Path(self.store.videos[video_id]["video"]))
                return self._json(200, self.store.state(video_id))
        return self.send_error(404)

    def do_POST(self) -> None:
        path = urllib.parse.unquote(urllib.parse.urlparse(self.path).path)
        match = re.fullmatch(r"/api/(draft|final)/([A-Za-z0-9][A-Za-z0-9_.-]{0,127})", path)
        if not match or match.group(2) not in self.store.videos:
            return self._json(404, {"error": "找不到视频或保存地址"})
        if self.headers.get("X-Review-Token") != self.store.token:
            return self._json(403, {"error": "页面令牌失效，请刷新页面"})
        try:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > MAX_BODY:
                raise ValueError("提交内容大小异常")
            raw = json.loads(self.rfile.read(length))
            value = self.store.save(match.group(2), raw, final=match.group(1) == "final")
            return self._json(200, {"ok": True, "status": value["status"],
                                    "graph": value["graph"]})
        except (ValueError, json.JSONDecodeError, UnicodeError) as exc:
            return self._json(400, {"error": str(exc)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--annotator", required=True)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18772)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.bind not in {"127.0.0.1", "::1", "localhost"}:
        raise SystemExit("refusing a non-loopback bind; use an SSH tunnel")
    store = Store(args.manifest, args.out_dir, args.annotator)
    print(f"manifest {store.manifest_sha256}: {len(store.videos)} videos, "
          f"{sum(v['duration_s'] for v in store.videos.values()) / 60:.1f} minutes")
    if args.dry_run:
        return
    Handler.store = store
    server = AuditServer((args.bind, args.port), Handler)
    print(f"open http://{args.bind}:{args.port}/  (annotator {store.annotator})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
