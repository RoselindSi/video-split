"""Serve the zone pages behind a login and keep everyone's work in one place.

FOUR PEOPLE WITH FOUR ZIP FILES IS HOW ANNOTATIONS GO MISSING. The pages hold
their state in the browser, which survives a reload and nothing else: a cleared
cache, a different laptop, or an annotator who forgets to press download and
the afternoon is gone. Serving the same directory over http gives every page
somewhere to post to, so the work lands centrally as it is drawn rather than at
the end if someone remembers.

WHO YOU ARE COMES FROM THE LOGIN, NOT FROM THE PAGE. A typed name makes a
second annotator out of a typo and anonymous work out of an empty box. With
accounts the server knows who is posting and writes that name, ignoring
whatever the body claims; the page's name field is filled and locked. It is
also the whole difference between `two people labelled this recording` and
`one person labelled it twice`, which is the measurement the batch exists for.

A LIST PAGE, BECAUSE TWELVE URLS IN A CHAT MESSAGE IS NOT A WORK QUEUE. The
index shows every recording with what this person has done to it and what
others have done, so picking the next one up is a click.

DRAFTS OVERWRITE, FINALS ACCUMULATE. A draft is the current state of one person
on one recording and only the latest is interesting; a final is a claim to be
done, so it is written under its own timestamped name and never clobbered by a
later autosave.

ONE FOLDER PER DATABAG, NAMED AFTER IT. A flat pile of json is fine until you
are holding a label a year later and want the footage it came from; nesting
each recording's work under its databag name makes that a `cd` instead of a
lookup. The databag is read out of the page this server built, never out of
the POST body -- a client-supplied path is a directory traversal, and the
answer is already on disk locally.

NOT INSIDE THE RAW CORPUS, THOUGH. The databags live in a shared incoming
dataset that other things read and re-sync; derived labels written in there
are indistinguishable from source data a year later and can be wiped by
whatever refreshes it. The tree mirrors the names, so `cd` still works.

A CLIP CAN BE REPORTED INSTEAD OF ANNOTATED. Black frames, a camera that was
covered, nothing happening -- these have all happened here, and without a way
to say so the annotator either draws a meaningless boundary or silently skips,
and both look the same afterwards as `not done yet`.

REFUSES TO BE EXPOSED WITHOUT ACCOUNTS. Binding anywhere but loopback publishes
workplace video of identifiable people and lets whoever finds the port write
files, so that combination needs --users or an explicit --no-auth. Basic auth
puts the password on the wire in base64, which is plaintext, so a real
deployment also wants --certfile or TLS terminated in front of it.
"""
from __future__ import annotations

import argparse
import base64
import binascii
import datetime
import functools
import getpass
import hashlib
import hmac
import html
import http.server
import json
import os
import re
import socketserver
import ssl
import sys
import threading
import time

SAFE = re.compile(r"[^\w.\-]+", re.UNICODE)
MAX_BODY = 8 << 20          # a page of 64-point curves is tens of kilobytes
ITERS = 240_000


def slug(s: str, fallback: str) -> str:
    s = SAFE.sub("_", (s or "").strip())[:60].strip("._-")
    return s or fallback


# PASSWORDS ARE NEVER STORED, ONLY VERIFIED. The file holds a per-user salt and
# a PBKDF2 digest, so it can sit next to the code without being a list of
# passwords, and a copy of it does not hand anyone an account.
def hash_pw(pw: str, salt: bytes = b"") -> str:
    salt = salt or os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", pw.encode(), salt, ITERS)
    return f"pbkdf2_sha256${ITERS}${salt.hex()}${dk.hex()}"


def check_pw(pw: str, stored: str) -> bool:
    try:
        algo, iters, salt, dk = stored.split("$")
        if algo != "pbkdf2_sha256":
            return False
        got = hashlib.pbkdf2_hmac("sha256", pw.encode(),
                                  bytes.fromhex(salt), int(iters))
    except (ValueError, binascii.Error):
        return False
    return hmac.compare_digest(got.hex(), dk)      # constant time, never ==


def load_users(path):
    users = {}
    if not path or not os.path.exists(path):
        return users
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, _, digest = line.partition(":")
        if name.strip() and digest.strip():
            users[name.strip()] = digest.strip()
    return users


PAGE_CSS = """<style>
body{font:14px/1.65 system-ui;margin:0;background:#111;color:#ddd}
#top{display:flex;align-items:baseline;gap:14px;padding:14px 18px;
  border-bottom:1px solid #282828;background:#181818}
h1{font-size:16px;margin:0}
.who{color:#888;font-size:12px;margin-left:auto}
p.sub{margin:14px 18px 10px;color:#999;font-size:12.5px}
table{border-collapse:collapse;margin:0 18px 44px;min-width:620px}
td,th{padding:8px 14px;border-bottom:1px solid #242424;text-align:left}
th{color:#7a7a7a;font-weight:normal;font-size:12px}
tr:hover td{background:#171d24}
a{color:#7cf;text-decoration:none}
a:hover{text-decoration:underline}
.done{color:#3f6}.draft{color:#ffd33d}.none{color:#666}.bad{color:#e77}
.n{color:#888;font-size:12px}
</style>"""


class Handler(http.server.SimpleHTTPRequestHandler):
    token = ""
    subdir = ""
    users = {}
    bags = {}                 # recording -> the databag directory's own name
    local_user = "anon"       # stands in for a login when there is none
    realm = "zone annotation"
    lock = threading.Lock()

    def bagdir(self, rec):
        """Where this recording's labels go: <subs>/<databag name>/.

        `bags` was read from the pages on this disk, so the name is ours, not
        something a request supplied -- and it is a bare basename joined onto
        the annotations root, never a path from anywhere else.
        """
        d = os.path.join(self.subdir, slug(self.bags.get(rec, rec), rec))
        os.makedirs(d, exist_ok=True)
        return d

    def log_message(self, fmt, *args):        # one line per POST, not per GET
        if self.command == "POST":
            print(f"  {self.address_string()}  {fmt % args}", flush=True)

    # ---------------------------------------------------------------- auth

    def whoami(self):
        if not self.users:
            return self.local_user
        got = (self.headers.get("Authorization") or "").split(None, 1)
        if len(got) == 2 and got[0].lower() == "basic":
            try:
                name, _, pw = base64.b64decode(got[1]).decode().partition(":")
            except (binascii.Error, UnicodeDecodeError, ValueError):
                name = pw = ""
            if name in self.users and check_pw(pw, self.users[name]):
                return name
        return None

    def deny(self):
        time.sleep(0.5)                  # a brute force should not be free
        self.send_response(401)
        self.send_header("WWW-Authenticate", f'Basic realm="{self.realm}"')
        self.send_header("Content-Length", "0")
        self.end_headers()

    # ------------------------------------------------------------- reading

    def written(self):
        for dirpath, _dirs, names in os.walk(self.subdir):
            for g in sorted(names):
                yield dirpath, g

    def states(self):
        """(recording, annotator) -> its strongest state, from the filenames.

        Filenames are what the POST handler wrote, so this needs no open() to
        answer the question the index asks; only the annotator's own rows are
        read for counts.
        """
        out = {}
        # An empty final ranks under a problem report, same rule as the
        # collector -- only finals are opened, and they are a few KB.
        rank = {"draft": 1, "final_empty": 2, "unusable": 3, "final": 4}
        for dirpath, g in self.written():
            if not g.endswith(".json"):
                continue
            stem = g[:-5]
            for state in ("final", "draft", "unusable"):
                mark = f"__{state}"
                if mark not in stem:
                    continue
                rec, _, rest = stem.partition("__")
                who = rest.split("__")[0]
                key = (rec, who)
                if state == "final":
                    try:
                        d = json.load(open(os.path.join(dirpath, g),
                                           encoding="utf-8"))
                        n = sum(len(e.get("keyframes", []))
                                for e in (d.get("eyes") or {}).values())
                    except Exception:
                        n = 1
                    state = "final" if n else "final_empty"
                if rank[state] >= rank.get(out.get(key, ""), 0):
                    out[key] = state
                break
        # an unusable report outranks an empty final, so anything still
        # `final_empty` here had no report beside it and is shown as submitted
        return {k: ("final" if v == "final_empty" else v)
                for k, v in out.items()}

    def count_kf(self, rec, who):
        for suffix in ("final", "draft"):
            for dirpath, g in sorted(self.written(), reverse=True):
                if g.startswith(f"{rec}__{who}__{suffix}") and g.endswith(".json"):
                    try:
                        d = json.load(open(os.path.join(dirpath, g),
                                           encoding="utf-8"))
                    except Exception:
                        return None
                    return sum(len(e.get("keyframes", []))
                               for e in (d.get("eyes") or {}).values())
        return None

    def index(self, who):
        me = slug(who, "anon")
        st = self.states()
        recs = [f[5:-5] for f in sorted(os.listdir(self.directory))
                if f.startswith("zone_") and f.endswith(".html")]
        label = {"final": ("已提交", "done"), "draft": ("有草稿", "draft"),
                 "unusable": ("报告了问题", "bad")}
        rows = []
        for rec in recs:
            mine = st.get((rec, me), "")
            others = sorted({w for (r, w) in st if r == rec and w != me})
            text, cls = label.get(mine, ("未开始", "none"))
            n = self.count_kf(rec, me) if mine in ("final", "draft") else None
            rows.append(
                f"<tr><td><a href='zone_{html.escape(rec)}.html'>"
                f"{html.escape(rec)}</a></td>"
                f"<td class={cls}>{text}</td>"
                f"<td class=n>{'' if n is None else str(n) + ' 个关键帧'}</td>"
                f"<td class=n>{html.escape('、'.join(others)) or '—'}</td></tr>")
        done = sum(1 for rec in recs if st.get((rec, me)) == "final")
        page = (
            "<meta charset=utf-8><title>区域标注</title>" + PAGE_CSS +
            "<div id=top><h1>佩戴者区域标注</h1>"
            f"<span class=who>{html.escape(who)}</span></div>"
            f"<p class=sub>{len(recs)} 段录像，你已提交 {done} 段。"
            "点录像号开始标；标到一半关掉也不会丢，回来接着标。"
            "同一段被两个人标过是好事，不要跳过别人标过的。</p>"
            "<table><tr><th>录像</th><th>你的进度</th><th></th>"
            "<th>还有谁标过</th></tr>" + "".join(rows) + "</table>")
        out = page.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")   # progress must be live
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    # THE PAGE IS TOLD WHO IS LOOKING AT IT, ahead of its own script, so the
    # name is filled and locked before it renders instead of being typed and
    # having to match yesterday's spelling.
    def serve_html(self, who):
        path = self.translate_path(self.path.split("?", 1)[0])
        try:
            raw = open(path, "rb").read()
        except OSError:
            return self.send_error(404)
        # Locked to the login when there are accounts; on a single-person
        # local run there is no login to read, and making someone type their
        # own name before the submit button works is friction that sends them
        # to the download button instead.
        var = "__WHO__" if self.users else "__WHO_DEFAULT__"
        raw = (f"<script>window.{var}={json.dumps(who)};</script>"
               ).encode() + raw
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_HEAD(self):
        if self.whoami() is None:
            return self.deny()
        return super().do_HEAD()

    def do_GET(self):
        who = self.whoami()
        if who is None:
            return self.deny()
        bare = self.path.split("?", 1)[0]
        if bare.rstrip("/") in ("", "/index.html"):
            return self.index(who)
        if bare.endswith(".html"):
            return self.serve_html(who)
        # RANGE REQUESTS, OR THE TIMELINE IS DEAD. SimpleHTTPRequestHandler
        # answers every GET with the whole file and no Accept-Ranges, so a
        # browser reports the clip as unseekable: currentTime assignments are
        # ignored and the scrub bar does nothing, which kills the entire
        # workflow -- find the frame where the boundary stops being right. It
        # works off the disk and breaks the moment the page is served, the
        # worst way for it to break, so ranges are answered here.
        rng = (self.headers.get("Range") or "").strip()
        if not rng:
            return super().do_GET()
        path = self.translate_path(bare)
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
                pass       # a seek away mid-download is normal, not an error

    # ------------------------------------------------------------- writing

    def _json(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        who = self.whoami()
        if who is None:
            return self.deny()
        if self.path.split("?", 1)[0].rstrip("/").rsplit("/", 1)[-1] != "submit":
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
        # THE BODY DOES NOT GET TO SAY WHO IT IS once there are accounts: a
        # posted name would let anyone file work under a colleague's, and a
        # typo would quietly invent an annotator.
        who = (slug(who, "anon") if self.users
               else slug(str(msg.get("annotator", "")), "anon"))
        status = msg.get("status")
        if status not in ("final", "draft", "unusable"):
            status = "draft"
        data = msg.get("data")
        if not isinstance(data, dict):
            return self._json(400, {"error": "no data object"})
        stamp = datetime.datetime.now().isoformat(timespec="seconds")
        data = dict(data, annotator=who, recording=data.get("recording", rec),
                    submitted_at=stamp, status=status)
        if status == "unusable":
            data["problem"] = str(msg.get("problem", ""))[:500]

        name = (f"{rec}__{who}__draft.json" if status == "draft" else
                f"{rec}__{who}__{status}_{stamp.replace(':', '')}.json")
        path = os.path.join(self.bagdir(rec), name)
        tmp = path + ".part"
        kf = sum(len(e.get("keyframes", []))
                 for e in (data.get("eyes") or {}).values())
        with self.lock:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=1)
            os.replace(tmp, path)          # never a half-written file on disk
            with open(os.path.join(self.subdir, "log.csv"), "a",
                      encoding="utf-8") as f:
                if f.tell() == 0:
                    f.write("at,recording,annotator,status,keyframes,file\n")
                f.write(f"{stamp},{rec},{who},{status},{kf},"
                        f"{os.path.relpath(path, self.subdir)}\n")
        if status != "draft":
            print(f"  {status.upper():9s} {rec}  {who}  {kf} 关键帧 -> {name}",
                  flush=True)
        # The folder is the half worth showing: `where did my file go` is
        # answered by the databag name, not by a filename the page already knew.
        return self._json(200, {"ok": True, "keyframes": kf,
                                "file": os.path.relpath(path, self.subdir)})


# THE PAGES ALREADY KNOW WHERE THEY CAME FROM. zone_video bakes the source
# path into each page's metadata, so the databag name is recoverable from the
# directory being served -- no manifest to keep in sync, and nothing about the
# layout depends on a request.
def read_bags(root):
    bags = {}
    for f in sorted(os.listdir(root)):
        if not (f.startswith("zone_") and f.endswith(".html")):
            continue
        rec = f[5:-5]
        try:
            head = open(os.path.join(root, f), encoding="utf-8").read(8000)
        except OSError:
            continue
        m = re.search(r'"source"\s*:\s*"([^"]*)"', head)
        src = m.group(1) if m else ""
        bag = os.path.basename(os.path.dirname(src)) if src else ""
        bags[rec] = bag or rec
    return bags


def adduser(path, name):
    name = name.strip()
    if not name or ":" in name:
        sys.exit("用户名不能为空，也不能带冒号")
    pw = getpass.getpass(f"给 {name} 设密码: ")
    if len(pw) < 8:
        sys.exit("密码太短（至少 8 位）")
    if pw != getpass.getpass("再输一遍: "):
        sys.exit("两次不一致")
    users = load_users(path)
    verb = "改了密码" if name in users else "加了用户"
    users[name] = hash_pw(pw)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# zone annotation accounts. 存的是 PBKDF2 摘要，不是密码。\n")
        for k, v in sorted(users.items()):
            f.write(f"{k}:{v}\n")
    os.chmod(path, 0o600)
    print(f"{verb}: {name} -> {path}（现在共 {len(users)} 个账号）")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", help="the zone_video --outdir (html + mp4)")
    ap.add_argument("--port", type=int, default=8777)
    ap.add_argument("--bind", default="127.0.0.1",
                    help="0.0.0.0 to let other machines reach it")
    ap.add_argument("--users", default="zone_users.txt",
                    help="accounts file; default ./zone_users.txt")
    ap.add_argument("--adduser", metavar="NAME",
                    help="create or re-password an account, then exit")
    ap.add_argument("--no-auth", action="store_true",
                    help="serve with no accounts at all (say so deliberately)")
    ap.add_argument("--certfile", help="PEM cert, to serve https directly")
    ap.add_argument("--keyfile")
    ap.add_argument("--token", default="",
                    help="additionally require this in an X-Token header")
    # THE DEFAULT IS WHERE GIT CAN SEE IT. Landing next to the mp4s put the
    # annotations in the one directory deliberately left untracked -- the clips
    # are too big for the repo and never change -- so the work sat outside
    # version control until someone moved it by hand.
    ap.add_argument("--subs", default="annotations",
                    help="where annotations land, relative to where you run "
                         "this; default ./annotations")
    a = ap.parse_args()

    if a.adduser:
        return adduser(a.users, a.adduser)
    if not a.dir:
        ap.error("需要 --dir")

    users = load_users(a.users)
    loopback = a.bind in ("127.0.0.1", "localhost", "::1", "")
    if not loopback and not users and not a.no_auth:
        ap.error(f"要对外开就得先建账号：--adduser NAME（会写进 {a.users}）。"
                 "真要不设密码开放，显式加 --no-auth。")

    root = os.path.abspath(a.dir)
    subs = os.path.abspath(a.subs)
    os.makedirs(subs, exist_ok=True)
    Handler.token, Handler.subdir, Handler.users = a.token, subs, users
    Handler.bags = read_bags(root)
    Handler.local_user = slug(getpass.getuser(), "anon")

    class Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    scheme = "https" if a.certfile else "http"
    with Server((a.bind, a.port),
                functools.partial(Handler, directory=root)) as httpd:
        if a.certfile:
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ctx.load_cert_chain(a.certfile, a.keyfile or a.certfile)
            httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
        host = "localhost" if loopback else a.bind
        pages = sum(1 for f in os.listdir(root)
                    if f.startswith("zone_") and f.endswith(".html"))
        say = functools.partial(print, flush=True)   # nohup buffers otherwise
        say(f"{pages} 段录像  {root}")
        say(f"标注结果 -> {subs}/<视频包名>/")
        named = sum(1 for r, b in Handler.bags.items() if b != r)
        say(f"  按原视频包分目录，{named}/{len(Handler.bags)} 段能对到包名")
        say(f"网址: {scheme}://{host}:{a.port}/")
        if users:
            say(f"账号 {len(users)} 个: {', '.join(sorted(users))}")
        else:
            say("没有账号，谁都能进（--adduser NAME 建一个）")
        if loopback:
            say("只有本机能开。要给别人用：--bind 0.0.0.0")
        elif scheme == "http":
            say("!! http 上的 Basic 认证密码等于明文。"
                "对外请加 --certfile，或者前面挡一层 https。")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
