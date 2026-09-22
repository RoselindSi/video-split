"""Loopback-only, alias-blind v2 development annotation server."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import html
import io
import json
import math
import re
import secrets
import threading
from pathlib import Path
from urllib.parse import urlparse

from src.auditor.boundary.final_timeline_audit import AuditServer, Handler, atomic_json, file_sha256, safe_name

SCHEMA = "interaction_relation_review_v2"
FIELDS = {
    "relation": {"CONTINUE", "SAME_ACTION_NEW_INSTANCE", "NEW_ACTION", "UNOBSERVABLE"},
    "boundary_scope": {"internal", "initial", "terminal"},
    "visibility": {"visible", "partial", "not_visible"},
    "context_sufficient": {"yes", "no", "uncertain"},
    "transition_shape": {"sharp", "gradual", "overlap", "no_transition", "uncertain"},
    "continuous_contact": {"yes", "no", "uncertain"},
    "release_observed": {"yes", "no", "uncertain"},
}


def validate(value, review_id, final):
    if not isinstance(value, dict) or value.get("review_id") != review_id:
        raise ValueError("Review ID does not match")
    if not isinstance(value.get("revision"), int) or isinstance(value["revision"], bool):
        raise ValueError("Missing revision")
    result = {"review_id": review_id}
    for field, allowed in FIELDS.items():
        v = value.get(field, "")
        if not isinstance(v, str) or (v and v not in allowed):
            raise ValueError(f"Invalid {field}")
        result[field] = v
    note = value.get("why_one_line", "")
    if not isinstance(note, str) or len(note) > 2000:
        raise ValueError("Invalid note")
    result["why_one_line"] = note.strip()
    context = value.get("context_half_s", 6)
    if context not in (6, 15):
        raise ValueError("Invalid context window")
    result["context_half_s"] = context
    result["human_confirmed"] = value.get("human_confirmed") is True
    if final:
        for field in FIELDS:
            if field != "relation" and not result[field]:
                raise ValueError(f"Choose {field.replace('_', ' ')}")
        if result["boundary_scope"] == "internal" and not result["relation"]:
            raise ValueError("Choose the relation")
        if result["boundary_scope"] != "internal" and result["relation"]:
            raise ValueError("One-sided events must leave relation empty")
        if not result["why_one_line"]:
            raise ValueError("Add a short reason")
        if not result["human_confirmed"]:
            raise ValueError("Confirm that you reviewed this event")
    return result


class ReviewStore:
    def __init__(self, manifest_path, out, annotator):
        self.manifest = json.loads(Path(manifest_path).read_text())
        if self.manifest.get("schema_version") != "interaction_review_clips_v2":
            raise ValueError("Not a v2 clip manifest")
        rows = self.manifest["clips"]
        self.clips = {r["review_id"]: r for r in rows}
        if len(self.clips) != len(rows) or len(rows) != self.manifest["n_events"]:
            raise ValueError("Duplicate clips or count mismatch")
        for rid, row in self.clips.items():
            if not re.fullmatch(r"review_[0-9]{4}", rid) or not Path(row["video"]).is_file():
                raise ValueError("Invalid alias or missing clip")
            duration, offset = row["duration_s"], row["candidate_offset_s"]
            if not all(math.isfinite(x) for x in (duration, offset)) or not 0 <= offset <= duration:
                raise ValueError("Invalid clip timing")
        self.ids = sorted(self.clips)
        self.annotator = safe_name(annotator)
        self.out = Path(out) / self.annotator
        self.out.mkdir(parents=True, exist_ok=True)
        self.manifest_hash = file_sha256(manifest_path)
        self.token = secrets.token_urlsafe(32)
        self.lock = threading.RLock()

    def state(self, rid):
        p = self.out / f"{rid}.latest.json"
        if not p.exists():
            return {"review_id": rid, "revision": 0, "status": "not_started"}
        value = json.loads(p.read_text())
        if value.get("clips_manifest_sha256") != self.manifest_hash:
            raise ValueError("Saved annotations belong to a different clip manifest")
        return value

    def progress(self):
        states = {rid: self.state(rid)["status"] for rid in self.ids}
        return {"schema_version": SCHEMA, "total": len(states),
                "final": sum(s == "final" for s in states.values()), "states": states}

    def save(self, rid, raw, final):
        clean = validate(raw, rid, final)
        with self.lock:
            current = self.state(rid)
            if raw["revision"] != current["revision"]:
                raise RuntimeError("This event changed in another tab; reload before saving")
            clean.update(schema_version=SCHEMA, revision=current["revision"] + 1,
                         status="final" if final else "draft", annotator=self.annotator,
                         clips_manifest_sha256=self.manifest_hash,
                         packet_sha256=self.manifest["packet_sha256"],
                         contract_sha256=self.manifest["contract_sha256"],
                         saved_at=dt.datetime.now(dt.timezone.utc).isoformat())
            if final:
                atomic_json(self.out / f"{rid}.final.{clean['revision']:05d}.json", clean)
            atomic_json(self.out / f"{rid}.latest.json", clean)
            atomic_json(self.out / "progress.json", self.progress())
            return clean

    def export(self):
        with self.lock:
            rows = [self.state(rid) for rid in self.ids if self.state(rid)["status"] == "final"]
        stream = io.StringIO(newline="")
        columns = ["review_id", *FIELDS, "why_one_line", "context_half_s", "human_confirmed",
                   "annotator", "schema_version", "contract_sha256", "packet_sha256",
                   "clips_manifest_sha256", "revision", "status", "saved_at"]
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
        return stream.getvalue().encode()


class ReviewHandler(Handler):
    store: ReviewStore

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            rows = []
            progress = self.store.progress()
            for i, rid in enumerate(self.store.ids, 1):
                state = progress["states"][rid]
                rows.append(f'<tr data-state="{state}"><td>{i:03d}</td><td><a href="/r/{rid}">{rid}</a></td>'
                            f'<td class="{state}">{html.escape(state.replace("_", " "))}</td></tr>')
            source = INDEX.replace("__ROWS__", "".join(rows)).replace("__COUNT__", str(progress["final"])).replace("__TOTAL__", str(progress["total"]))
            return self._html(source.encode())
        if path == "/api/progress":
            return self._json(200, self.store.progress())
        if path == "/api/export":
            body = self.store.export()
            self.send_response(200)
            self.send_header("Content-Type", "text/csv; charset=utf-8")
            self.send_header("Content-Disposition", 'attachment; filename="review_completed.csv"')
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        match = re.fullmatch(r"/(r|clip|api/state)/(review_[0-9]{4})", path)
        if not match or match[2] not in self.store.clips:
            return self.send_error(404)
        rid = match[2]
        if match[1] == "clip":
            return self._video(Path(self.store.clips[rid]["video"]))
        state = self.store.state(rid)
        if match[1] == "api/state":
            return self._json(200, state)
        i = self.store.ids.index(rid)
        clip = self.store.clips[rid]
        payload = {"review_id": rid, "index": i + 1, "total": len(self.store.ids),
                   "previous": self.store.ids[i - 1] if i else None,
                   "next": self.store.ids[i + 1] if i + 1 < len(self.store.ids) else None,
                   "candidate_offset_s": clip["candidate_offset_s"], "duration_s": clip["duration_s"],
                   "state": state, "token": self.store.token}
        template = Path(__file__).with_name("interaction_review_page.html").read_text()
        source = template.replace("__PAYLOAD__", json.dumps(payload).replace("</", "<\\/"))
        return self._html(source.encode())

    def do_POST(self):
        path = urlparse(self.path).path
        match = re.fullmatch(r"/api/(draft|final)/(review_[0-9]{4})", path)
        if not match or match[2] not in self.store.clips:
            return self._json(404, {"error": "Unknown event"})
        if self.headers.get("X-Review-Token") != self.store.token:
            return self._json(403, {"error": "Invalid session token; reload the page"})
        origin = self.headers.get("Origin")
        if origin and urlparse(origin).netloc != self.headers.get("Host"):
            return self._json(403, {"error": "Cross-origin submission rejected"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 16384:
                raise ValueError("Invalid request size")
            raw = json.loads(self.rfile.read(length))
            value = self.store.save(match[2], raw, match[1] == "final")
            return self._json(200, {"ok": True, "revision": value["revision"], "status": value["status"]})
        except RuntimeError as exc:
            return self._json(409, {"error": str(exc)})
        except (ValueError, UnicodeError) as exc:
            return self._json(400, {"error": str(exc)})


INDEX = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Development review | v2</title><style>
body{margin:0;background:#151717;color:#eceeed;font:15px/1.5 system-ui}header{padding:18px 24px;border-bottom:1px solid #424646;display:flex;gap:20px;flex-wrap:wrap}h1{font-size:18px;margin:0}a{color:#8bd3c5;text-decoration:none}.wrap{max-width:900px;margin:auto;padding:24px}table{border-collapse:collapse;width:100%;margin-top:20px}th,td{text-align:left;padding:12px;border-bottom:1px solid #383b3b}.final{color:#91d8a8}.draft{color:#ebba67}select{padding:8px;color:#eee;background:#282c2b;border:1px solid #555;border-radius:4px}.toolbar{display:flex;gap:20px;align-items:center;flex-wrap:wrap}th{color:#aaa;font-weight:500}
</style><header><h1>Development review / v2</h1><span>__COUNT__ / __TOTAL__ final</span></header>
<main class="wrap"><div class="toolbar"><label>Status <select id="filter"><option value="">All</option><option value="not_started">Not started</option><option value="draft">Draft</option><option value="final">Final</option></select></label><a href="/api/export">Export completed</a></div>
<table><thead><tr><th>#</th><th>Event</th><th>Status</th></tr></thead><tbody>__ROWS__</tbody></table></main>
<script>document.querySelector('#filter').onchange=e=>document.querySelectorAll('tr[data-state]').forEach(r=>r.hidden=!!e.target.value&&r.dataset.state!==e.target.value);</script></html>"""


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--annotator", default="roselind")
    p.add_argument("--port", type=int, default=8766)
    a = p.parse_args()
    store = ReviewStore(a.manifest, a.out, a.annotator)
    handler = type("BoundReviewHandler", (ReviewHandler,), {"store": store})
    server = AuditServer(("127.0.0.1", a.port), handler)
    print(f"review_ready http://127.0.0.1:{a.port}/ {len(store.ids)} events", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
