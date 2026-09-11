"""Serve the zone pages and keep every annotator's work in one directory.

FOUR PEOPLE WITH FOUR ZIP FILES IS HOW ANNOTATIONS GO MISSING. The pages
already hold their state in the browser, which survives a reload and nothing
else: a cleared cache, a different laptop, or an annotator who forgets to press
download and the afternoon is gone. Serving the same directory over http gives
every page somewhere to post to, so the work lands centrally as it is drawn
rather than at the end if someone remembers.

DRAFTS OVERWRITE, FINALS ARE KEPT SEPARATELY. A draft is the current state of
one person on one recording and only the latest is interesting; a final is a
claim that they are done, so it is written under its own name and never
clobbered by a later autosave. Both carry the annotator, so two people on the
same recording produce two files and can be compared -- which is the only way
to find out whether `the wearer's side` means the same thing to two people.

THE NAME IS SANITISED BECAUSE IT ARRIVES FROM A BROWSER. Whatever a person
types goes into a filename, and a filename assembled from network input is a
path traversal waiting to happen. Only word characters survive, and the file
is written inside the submissions directory by construction, never by joining
a supplied path.

LOOPBACK BY DEFAULT. This writes files on behalf of anyone who can reach it and
has no authentication worth the name, so it binds to localhost unless told
otherwise; `--bind 0.0.0.0` is for a trusted lab network, with `--token` if
even that is more exposure than it deserves.
"""
from __future__ import annotations

import argparse
import datetime
import functools
import http.server
import json
import os
import re
import socketserver
import threading

SAFE = re.compile(r"[^\w.\-]+", re.UNICODE)
MAX_BODY = 8 << 20          # a page of 64-point curves is tens of kilobytes


def slug(s: str, fallback: str) -> str:
    s = SAFE.sub("_", (s or "").strip())[:60].strip("._-")
    return s or fallback


class Handler(http.server.SimpleHTTPRequestHandler):
    token = ""
    subdir = ""
    lock = threading.Lock()

    def log_message(self, fmt, *args):        # one line per POST, not per GET
        if self.command == "POST":
            print(f"  {self.address_string()}  {fmt % args}", flush=True)

    # RANGE REQUESTS, OR THE TIMELINE IS DEAD. SimpleHTTPRequestHandler answers
    # every GET with the whole file and no Accept-Ranges, so a browser reports
    # the clip as unseekable: currentTime assignments are ignored, the scrub bar
    # does nothing, and the keyframe workflow -- find the frame where the
    # boundary stops being right -- is impossible. It works off the disk and
    # breaks the moment the same page is served, which is the worst way for it
    # to break, so the server answers ranges itself.
    def end_headers(self):
        if self.command in ("GET", "HEAD"):
            self.send_header("Accept-Ranges", "bytes")
        super().end_headers()

    def do_GET(self):
        rng = (self.headers.get("Range") or "").strip()
        if not rng:
            return super().do_GET()
        path = self.translate_path(self.path)
        m = re.match(r"bytes=(\d*)-(\d*)$", rng)
        if os.path.isdir(path) or not m:
            return super().do_GET()
        try:
            f = open(path, "rb")
        except OSError:
            return self.send_error(404)
        with f:
            size = os.fstat(f.fileno()).st_size
            lo, hi = m.group(1), m.group(2)
            if lo == "":                       # bytes=-N: the last N bytes
                start, end = max(0, size - int(hi or 0)), size - 1
            else:
                start = int(lo)
                end = min(int(hi), size - 1) if hi else size - 1
            if start >= size or start > end:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self.send_response(206)
            self.send_header("Content-Type", self.guess_type(path))
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Content-Length", str(end - start + 1))
            self.end_headers()
            f.seek(start)
            left = end - start + 1
            try:
                while left > 0:
                    chunk = f.read(min(1 << 16, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass          # a seek away mid-download is normal, not an error

    def _json(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path.rstrip("/").rsplit("/", 1)[-1] != "submit":
            return self._json(404, {"error": "no such endpoint"})
        if self.token and self.headers.get("X-Token", "") != self.token:
            return self._json(403, {"error": "bad token"})
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return self._json(400, {"error": "bad length"})
        if n <= 0 or n > MAX_BODY:
            return self._json(413, {"error": "body too large"})
        try:
            msg = json.loads(self.rfile.read(n).decode("utf-8"))
        except Exception as e:
            return self._json(400, {"error": f"bad json: {e}"})

        rec = slug(str(msg.get("rec", "")), "unknown")
        who = slug(str(msg.get("annotator", "")), "anon")
        final = msg.get("status") == "final"
        data = msg.get("data")
        if not isinstance(data, dict):
            return self._json(400, {"error": "no data object"})
        stamp = datetime.datetime.now().isoformat(timespec="seconds")
        data = dict(data, annotator=who, recording=data.get("recording", rec),
                    submitted_at=stamp, status="final" if final else "draft")

        name = (f"{rec}__{who}__final_{stamp.replace(':', '')}.json" if final
                else f"{rec}__{who}__draft.json")
        path = os.path.join(self.subdir, name)
        tmp = path + ".part"
        with self.lock:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=1)
            os.replace(tmp, path)           # never a half-written file on disk
            kf = sum(len(e.get("keyframes", []))
                     for e in (data.get("eyes") or {}).values())
            with open(os.path.join(self.subdir, "index.csv"), "a",
                      encoding="utf-8") as f:
                if f.tell() == 0:
                    f.write("at,recording,annotator,status,keyframes,file\n")
                f.write(f"{stamp},{rec},{who},"
                        f"{'final' if final else 'draft'},{kf},{name}\n")
        print(f"  {'FINAL' if final else 'draft'}  {rec}  {who}  "
              f"{kf} 关键帧 -> {name}", flush=True)
        return self._json(200, {"ok": True, "file": name, "keyframes": kf})


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True,
                    help="the zone_video --outdir (html + mp4 live here)")
    ap.add_argument("--port", type=int, default=8777)
    ap.add_argument("--bind", default="127.0.0.1",
                    help="0.0.0.0 to let other machines reach it")
    ap.add_argument("--token", default="",
                    help="require this in an X-Token header")
    # THE DEFAULT IS WHERE GIT CAN SEE IT. Landing next to the mp4s put the
    # annotations in the one directory that is deliberately not tracked --
    # the clips are too big for the repo and never change -- so the work would
    # sit outside version control until someone moved it by hand. `annotations`
    # under the directory you run from is the clone you are about to push.
    ap.add_argument("--subs", default="annotations",
                    help="where annotations land, relative to where you run "
                         "this; default ./annotations")
    a = ap.parse_args()

    root = os.path.abspath(a.dir)
    subs = os.path.abspath(a.subs or os.path.join(root, "submissions"))
    os.makedirs(subs, exist_ok=True)
    Handler.token, Handler.subdir = a.token, subs

    class Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    pages = sorted(f for f in os.listdir(root) if f.endswith(".html"))
    with Server((a.bind, a.port),
                functools.partial(Handler, directory=root)) as httpd:
        host = "localhost" if a.bind in ("127.0.0.1", "") else a.bind
        print(f"serving {root}  ->  http://{host}:{a.port}/")
        print(f"submissions -> {subs}")
        if a.bind == "127.0.0.1":
            print("（只有本机能开。要给别人用：--bind 0.0.0.0）")
        for p in pages:
            print(f"  http://{host}:{a.port}/{p}")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
