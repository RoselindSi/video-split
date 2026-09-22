"""Label an exhaustive final-test timeline without exposing prior answers.

The server reads only the frozen split manifest and the selected ``mid.mp4``
files. It never opens ``segments.json``, candidate manifests, model outputs, or
old labels. Drafts are atomically overwritten; final submissions accumulate as
timestamped snapshots so corrections have an audit trail.

Run on the machine that mounts the corpus, then reach it through an SSH tunnel:

    python -m src.auditor.boundary.final_timeline_audit \
        --manifest /workspace/final_test_v1.json \
        --out-dir /workspace/final_test_v1_annotations \
        --annotator roselind --port 8765

    ssh -L 8765:127.0.0.1:8765 <host>

The server intentionally refuses a non-loopback bind. These videos contain
identifiable people and this focused tool does not implement authentication.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
import http.server
import json
import os
import re
import socketserver
import tempfile
import urllib.parse
from pathlib import Path


SCHEMA = "episode_boundary_final_timeline_v1"
MANIFEST_SCHEMA = "episode_boundary_final_test_split_v1"
RID_RE = re.compile(r"recording_[0-9]{6}")
RELATIONS = {
    "new_action",
    "same_action_new_instance",
    "cannot_determine",
    "initial_action_start",
    "terminal_action_end",
}
BOUNDARY_RELATIONS = {"new_action", "same_action_new_instance"}
INTERNAL_RELATIONS = BOUNDARY_RELATIONS | {"cannot_determine"}
MAX_BODY = 2 << 20


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: str | Path) -> tuple[dict, dict[str, dict], str]:
    manifest_path = Path(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError(f"unexpected manifest schema: {manifest.get('schema_version')!r}")
    if manifest.get("frozen") is not True:
        raise ValueError("final-test manifest is not frozen")
    rows = manifest.get("selected") or []
    expected = int(manifest.get("n_selected_recordings") or 0)
    if len(rows) != expected or expected <= 0:
        raise ValueError(f"manifest selected count is {len(rows)}, declared {expected}")
    by_id = {}
    for row in rows:
        rid = str(row.get("recording_id") or "")
        if not RID_RE.fullmatch(rid):
            raise ValueError(f"invalid recording ID in manifest: {rid!r}")
        if rid in by_id:
            raise ValueError(f"duplicate recording ID in manifest: {rid}")
        video = Path(str(row.get("video") or ""))
        if not video.is_file():
            raise FileNotFoundError(f"selected video is missing: {video}")
        duration = float(row.get("duration_s") or 0.0)
        if duration <= 0:
            raise ValueError(f"selected recording has no duration: {rid}")
        by_id[rid] = dict(row, duration_s=duration, video=str(video))
    return manifest, by_id, file_sha256(manifest_path)


def validate_submission(value: object, recording_id: str,
                        duration_s: float) -> dict:
    if not isinstance(value, dict):
        raise ValueError("submission must be a JSON object")
    if value.get("recording_id") != recording_id:
        raise ValueError("recording_id does not match the URL")
    no_boundary = value.get("no_internal_boundary")
    if not isinstance(no_boundary, bool):
        raise ValueError("no_internal_boundary must be true or false")
    raw_events = value.get("events")
    if not isinstance(raw_events, list):
        raise ValueError("events must be a list")

    events = []
    for index, event in enumerate(raw_events):
        if not isinstance(event, dict):
            raise ValueError(f"event {index} is not an object")
        relation = str(event.get("instance_relation") or "")
        if relation not in RELATIONS:
            raise ValueError(f"event {index} has invalid relation {relation!r}")
        try:
            start = round(float(event["start_s"]), 3)
            end = round(float(event.get("end_s", start)), 3)
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"event {index} has invalid time") from None
        if not (0 <= start <= end <= duration_s + 0.05):
            raise ValueError(
                f"event {index} interval [{start}, {end}] is outside "
                f"[0, {duration_s:.3f}]")
        note = str(event.get("note") or "").strip()[:1000]
        events.append({"start_s": start, "end_s": end,
                       "instance_relation": relation, "note": note})

    events.sort(key=lambda event: (event["start_s"], event["end_s"],
                                   event["instance_relation"]))
    event_keys = [(event["start_s"], event["end_s"],
                   event["instance_relation"]) for event in events]
    if len(event_keys) != len(set(event_keys)):
        raise ValueError("duplicate events are not allowed")
    if no_boundary and any(e["instance_relation"] in INTERNAL_RELATIONS
                           for e in events):
        raise ValueError("no_internal_boundary conflicts with an internal event")
    if not no_boundary and not events:
        raise ValueError("add an event or mark no_internal_boundary")
    return {
        "schema_version": SCHEMA,
        "recording_id": recording_id,
        "no_internal_boundary": no_boundary,
        "events": events,
    }


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise


def safe_name(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("._")
    if not clean:
        raise ValueError("annotator name is empty after sanitizing")
    return clean[:64]


def page(recording: dict, initial: dict | None) -> bytes:
    rid = recording["recording_id"]
    payload = {
        "recording_id": rid,
        "duration_s": recording["duration_s"],
        "initial": initial or {"recording_id": rid,
                               "no_internal_boundary": False, "events": []},
    }
    script_data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    source = PAGE.replace("__RID__", html.escape(rid)).replace(
        "__PAYLOAD__", script_data)
    return source.encode("utf-8")


PAGE = r"""<!doctype html><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__RID__ timeline</title>
<style>
:root{color-scheme:dark;--bg:#111;--panel:#191919;--line:#333;--text:#e8e8e8;
  --muted:#999;--new:#ee6a5f;--repeat:#e6a23c;--unclear:#999;
  --endpoint:#62a7d8}
*{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--text);
  font:14px/1.45 system-ui} header{height:48px;display:flex;align-items:center;
  gap:12px;padding:0 16px;background:var(--panel);border-bottom:1px solid var(--line)}
h1{font-size:15px;margin:0} a{color:#8cc8ef;text-decoration:none}
#save{margin-left:auto;color:var(--muted);font-size:12px}.wrap{max-width:1180px;
  margin:auto;padding:14px 16px 40px}video{display:block;width:100%;max-height:62vh;
  background:#000}.toolbar{display:flex;align-items:center;gap:8px;flex-wrap:wrap;
  padding:10px 0}.group{display:flex;border:1px solid var(--line);border-radius:6px;
  overflow:hidden}.group button{border:0;border-right:1px solid var(--line);
  border-radius:0}.group button:last-child{border-right:0}button{min-height:34px;
  padding:6px 10px;background:#242424;color:var(--text);border:1px solid var(--line);
  border-radius:6px;cursor:pointer}button:hover{background:#303030}button.active{background:#444}
button.new{border-left:4px solid var(--new)}button.repeat{border-left:4px solid var(--repeat)}
button.unclear{border-left:4px solid var(--unclear)}button.endpoint{border-left:4px solid var(--endpoint)}
#clock{font:13px ui-monospace,monospace;min-width:128px}.rail{height:30px;background:#202020;
  border:1px solid var(--line);position:relative;overflow:hidden}.mark{position:absolute;top:0;
  width:4px;height:100%;border:0;border-radius:0;padding:0;min-height:0}
.mark.new_action{background:var(--new)}.mark.same_action_new_instance{background:var(--repeat)}
.mark.cannot_determine{background:var(--unclear)}.mark.initial_action_start,
.mark.terminal_action_end{background:var(--endpoint)}
.section{margin-top:16px;border-top:1px solid var(--line);padding-top:12px}
.row{display:grid;grid-template-columns:120px 100px 100px minmax(180px,1fr) 76px;
  gap:8px;align-items:center;padding:7px 0;border-bottom:1px solid #252525}
select,input{width:100%;min-height:32px;background:#202020;color:var(--text);
  border:1px solid var(--line);border-radius:4px;padding:5px 7px}
.empty{color:var(--muted);padding:18px 0}.check{display:flex;gap:7px;align-items:center;
  margin-left:auto}.check input{width:auto;min-height:0}#submit{background:#235d3c;border-color:#347b52}
@media(max-width:760px){.row{grid-template-columns:1fr 1fr}.row input:nth-child(4){grid-column:1/3}}
</style>
<header><a href="/">All recordings</a><h1>__RID__</h1><span id="save">loading</span></header>
<main class="wrap">
  <video id="video" controls preload="metadata" src="/video/__RID__"></video>
  <div class="toolbar">
    <span id="clock">0.0 / 0.0 s</span>
    <div class="group" id="speed"><button data-v="1" class="active">1x</button><button data-v="2">2x</button><button data-v="4">4x</button></div>
    <button class="new" data-add="new_action">New action</button>
    <button class="repeat" data-add="same_action_new_instance">Same action, new instance</button>
    <button class="unclear" data-add="cannot_determine">Cannot determine</button>
    <button class="endpoint" data-add="initial_action_start">Initial start</button>
    <button class="endpoint" data-add="terminal_action_end">Terminal end</button>
    <label class="check"><input id="none" type="checkbox">No internal boundary</label>
    <button id="submit">Submit final</button>
  </div>
  <div class="rail" id="rail"></div>
  <section class="section"><div id="events"></div></section>
</main>
<script>
const P=__PAYLOAD__, video=document.getElementById('video'), out=document.getElementById('events');
let state=P.initial, timer=null, dirty=false;
const names={new_action:'New action',same_action_new_instance:'Same action, new instance',
  cannot_determine:'Cannot determine',initial_action_start:'Initial start',terminal_action_end:'Terminal end'};
function t(v){return Math.max(0,Math.min(P.duration_s,Math.round(v*10)/10))}
function payload(){return {recording_id:P.recording_id,no_internal_boundary:document.getElementById('none').checked,events:state.events}}
function schedule(){dirty=true;document.getElementById('save').textContent='unsaved';clearTimeout(timer);timer=setTimeout(save,350)}
async function save(){if(!dirty)return;const r=await fetch('/api/draft/'+P.recording_id,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload())});
  const d=await r.json();if(!r.ok){document.getElementById('save').textContent=d.error||'save failed';return}dirty=false;document.getElementById('save').textContent='saved'}
function add(rel){const at=t(video.currentTime);state.events.push({start_s:at,end_s:at,instance_relation:rel,note:''});document.getElementById('none').checked=false;draw();schedule()}
function draw(){state.events.sort((a,b)=>a.start_s-b.start_s);out.innerHTML='';const rail=document.getElementById('rail');rail.innerHTML='';
  state.events.forEach((e,i)=>{const row=document.createElement('div');row.className='row';
    row.innerHTML='<select data-k="instance_relation"></select><input type="number" step="0.1" min="0" max="'+P.duration_s+'" data-k="start_s"><input type="number" step="0.1" min="0" max="'+P.duration_s+'" data-k="end_s"><input maxlength="1000" placeholder="Note" data-k="note"><button data-del>Delete</button>';
    const s=row.querySelector('select');Object.keys(names).forEach(k=>{const o=document.createElement('option');o.value=k;o.textContent=names[k];s.appendChild(o)});
    row.querySelectorAll('[data-k]').forEach(el=>{const k=el.dataset.k;el.value=e[k];el.onchange=()=>{e[k]=(k==='start_s'||k==='end_s')?t(+el.value):el.value;draw();schedule()}});
    row.querySelector('[data-del]').onclick=()=>{state.events.splice(i,1);draw();schedule()};row.ondblclick=()=>{video.currentTime=e.start_s};out.appendChild(row);
    const m=document.createElement('button');m.className='mark '+e.instance_relation;m.style.left=(100*e.start_s/P.duration_s)+'%';m.title=names[e.instance_relation]+' @ '+e.start_s.toFixed(1)+'s';m.onclick=()=>{video.currentTime=e.start_s};rail.appendChild(m)});
  if(!state.events.length)out.innerHTML='<div class="empty">No marked events</div>'}
video.ontimeupdate=()=>{document.getElementById('clock').textContent=video.currentTime.toFixed(1)+' / '+(video.duration||P.duration_s).toFixed(1)+' s'};
document.querySelectorAll('[data-add]').forEach(b=>b.onclick=()=>add(b.dataset.add));
document.querySelectorAll('#speed button').forEach(b=>b.onclick=()=>{video.playbackRate=+b.dataset.v;document.querySelectorAll('#speed button').forEach(x=>x.classList.toggle('active',x===b))});
document.getElementById('none').checked=!!state.no_internal_boundary;document.getElementById('none').onchange=()=>{state.no_internal_boundary=document.getElementById('none').checked;schedule()};
document.getElementById('submit').onclick=async()=>{await save();const r=await fetch('/api/final/'+P.recording_id,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload())});const d=await r.json();document.getElementById('save').textContent=r.ok?'final submitted':(d.error||'submit failed')};
draw();document.getElementById('save').textContent='loaded';
</script>"""


class AuditServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


class Handler(http.server.BaseHTTPRequestHandler):
    recordings: dict[str, dict] = {}
    out_dir = Path(".")
    annotator = ""
    manifest_sha256 = ""

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

    def _paths(self, rid: str) -> tuple[Path, list[Path]]:
        base = self.out_dir / self.annotator
        draft = base / f"{rid}.draft.json"
        finals = sorted(base.glob(f"{rid}.final.*.json"))
        return draft, finals

    def _state(self, rid: str) -> dict | None:
        draft, finals = self._paths(rid)
        choices = ([draft] if draft.exists() else []) + finals[-1:]
        if not choices:
            return None
        path = max(choices, key=lambda item: item.stat().st_mtime_ns)
        return json.loads(path.read_text(encoding="utf-8"))

    def _index(self) -> bytes:
        rows = []
        complete = 0
        for rid, recording in self.recordings.items():
            draft, finals = self._paths(rid)
            if finals:
                status, cls = "Final", "final"
                complete += 1
            elif draft.exists():
                status, cls = "Draft", "draft"
            else:
                status, cls = "Not started", "none"
            minutes = recording["duration_s"] / 60
            rows.append(f'<tr><td><a href="/r/{rid}">{rid}</a></td>'
                        f'<td>{minutes:.1f} min</td><td class="{cls}">{status}</td></tr>')
        total_min = sum(r["duration_s"] for r in self.recordings.values()) / 60
        source = f"""<!doctype html><meta charset=utf-8><title>Final timeline audit</title>
<style>body{{font:14px/1.5 system-ui;margin:0;background:#111;color:#ddd}}header{{padding:14px 18px;background:#191919;border-bottom:1px solid #333}}h1{{font-size:16px;margin:0}}p,table{{margin:16px 18px}}table{{border-collapse:collapse;min-width:520px}}td,th{{padding:8px 12px;border-bottom:1px solid #292929;text-align:left}}a{{color:#8cc8ef;text-decoration:none}}.final{{color:#72c58b}}.draft{{color:#e6a23c}}.none{{color:#888}}</style>
<header><h1>Final timeline audit</h1></header><p>{complete}/{len(rows)} final, {total_min:.1f} minutes total</p><table><tr><th>Recording</th><th>Duration</th><th>Status</th></tr>{''.join(rows)}</table>"""
        return source.encode("utf-8")

    def do_GET(self) -> None:
        path = urllib.parse.urlparse(self.path).path
        if path == "/":
            return self._html(self._index())
        if path.startswith("/r/"):
            rid = path.removeprefix("/r/")
            if rid not in self.recordings:
                return self.send_error(404)
            return self._html(page(self.recordings[rid], self._state(rid)))
        if path.startswith("/video/"):
            rid = path.removeprefix("/video/")
            if rid not in self.recordings:
                return self.send_error(404)
            return self._video(Path(self.recordings[rid]["video"]))
        if path.startswith("/api/state/"):
            rid = path.removeprefix("/api/state/")
            if rid not in self.recordings:
                return self._json(404, {"error": "unknown recording"})
            return self._json(200, self._state(rid) or {})
        return self.send_error(404)

    def _video(self, path: Path) -> None:
        size = path.stat().st_size
        match = re.fullmatch(r"bytes=(\d*)-(\d*)",
                             (self.headers.get("Range") or "").strip())
        if match:
            lo, hi = match.groups()
            if lo:
                start = int(lo)
                end = min(int(hi) if hi else size - 1, size - 1)
            else:
                length = min(int(hi or 0), size)
                start, end = size - length, size - 1
            if start < 0 or start > end or start >= size:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            status = 206
        else:
            start, end, status = 0, size - 1, 200
        self.send_response(status)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with path.open("rb") as handle:
            handle.seek(start)
            remaining = end - start + 1
            while remaining:
                block = handle.read(min(1024 * 1024, remaining))
                if not block:
                    break
                try:
                    self.wfile.write(block)
                except (BrokenPipeError, ConnectionResetError):
                    break
                remaining -= len(block)

    def do_POST(self) -> None:
        path = urllib.parse.urlparse(self.path).path
        match = re.fullmatch(r"/api/(draft|final)/(recording_[0-9]{6})", path)
        if not match or match.group(2) not in self.recordings:
            return self._json(404, {"error": "unknown endpoint or recording"})
        try:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > MAX_BODY:
                raise ValueError("invalid request size")
            raw = json.loads(self.rfile.read(length))
            rid = match.group(2)
            value = validate_submission(raw, rid,
                                        self.recordings[rid]["duration_s"])
            value.update({"manifest_sha256": self.manifest_sha256,
                          "annotator": self.annotator,
                          "saved_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                          "status": match.group(1)})
            draft, _ = self._paths(rid)
            if match.group(1) == "draft":
                target = draft
            else:
                stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                target = draft.parent / f"{rid}.final.{stamp}.json"
            atomic_json(target, value)
            return self._json(200, {"ok": True, "path": str(target)})
        except (ValueError, json.JSONDecodeError, UnicodeError) as exc:
            return self._json(400, {"error": str(exc)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--annotator", required=True)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.bind not in {"127.0.0.1", "::1", "localhost"}:
        raise SystemExit("refusing a non-loopback bind; use an SSH tunnel")
    _manifest, recordings, manifest_hash = load_manifest(args.manifest)
    annotator = safe_name(args.annotator)
    print(f"manifest {manifest_hash}: {len(recordings)} recordings, "
          f"{sum(r['duration_s'] for r in recordings.values()) / 60:.1f} minutes")
    print("label-blind: no segments, candidates, model outputs, or old labels loaded")
    if args.dry_run:
        return
    Handler.recordings = recordings
    Handler.out_dir = Path(args.out_dir)
    Handler.annotator = annotator
    Handler.manifest_sha256 = manifest_hash
    Handler.out_dir.mkdir(parents=True, exist_ok=True)
    server = AuditServer((args.bind, args.port), Handler)
    print(f"open http://{args.bind}:{args.port}/  (annotator {annotator})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
