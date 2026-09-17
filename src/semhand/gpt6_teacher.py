"""GPT-6 as a second, independent teacher: one boxed hand per request.

WHY PER HAND, NOT GPT-6'S OWN BOXES. The student is trained on the shared
detector's boxes, so a teacher has to say something about exactly those boxes.
GPT-6 drawing its own boxes on the monocular image would need a camera-to-
panorama box match for every hand; asking it about the box we already have
needs none, and is the same unit Q1 answers. The view is Q1's: the frame with
one green box, and a zoom on the box.

THE PROMPT IS NEW, SO IT IS CHECKED BEFORE IT LABELS ANYTHING. GPT-6 was
audited in its own box-drawing mode, not in this one. `--root
/workspace/audit_q1` runs it on the 300 boxes the user judged blind, and
`--mode score` compares it with those answers and with Q1. The pool run
starts only after that.

CREDENTIALS ONLY FROM THE ENVIRONMENT. GPT6_API_KEY and GPT6_BASE_URL are read
from the process environment, never from an argument or a file, never logged;
any error text has the key replaced before it is written. GPT6_MODEL defaults
to gpt-6-astra. Frames are sent inline as base64 and nothing is uploaded.

A REPLY FROM ANOTHER MODEL IS NOT A GPT-6 LABEL. The relay has returned a
different model before; the returned model name is checked and a mismatch is
recorded as a failure, not as an answer.

Resumable by id; each technical failure is retried at most MAX_TRIES times,
and 429 responses lower the request rate. A null answer ("cannot tell") is a
valid result and is not retried.
"""
from __future__ import annotations

import argparse
import base64
import collections
import concurrent.futures as cf
import io
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request

from src.semhand.qwen import load_items, views

MAX_TRIES = 5
PROMPT = (
    "This image comes from a camera worn on the head of a factory worker (the camera wearer). "
    "One hand is outlined by a green box; the second image is a zoomed crop around that box. "
    "Other people's hands may also be visible. A person has at most two hands.\n"
    "Is the outlined hand one of the camera wearer's own hands? Judge from the whole scene -- "
    "where the arm comes from, whose body it connects to, how it moves with the camera -- not "
    "from position, size, glove colour or counting alone. If the image does not let you tell, "
    "answer null.\n"
    'Reply with JSON only: {"wearer": true | false | null, "reason": "<visible evidence, max 20 words>"}')


def env():
    key, url = os.environ.get("GPT6_API_KEY"), os.environ.get("GPT6_BASE_URL")
    if not key or not url:
        raise SystemExit("需要在环境变量里设置 GPT6_API_KEY 和 GPT6_BASE_URL（不要写进文件）")
    return key, url.rstrip("/"), os.environ.get("GPT6_MODEL", "gpt-6-astra")


def jpeg_b64(img, quality=90):
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


class Rate:
    """A shared minimum spacing between request starts; 429 widens it."""

    def __init__(self, per_s):
        self.gap, self.lock, self.next = 1.0 / per_s, threading.Lock(), 0.0

    def wait(self):
        with self.lock:
            now = time.monotonic()
            t = max(now, self.next)
            self.next = t + self.gap
        time.sleep(max(0.0, t - now))

    def slow(self, retry_after=None):
        with self.lock:
            self.gap = min(self.gap * 1.5, 5.0)
            self.next = time.monotonic() + (retry_after or 5.0)


def ask(item, key, url, model, rate):
    frame, crop = views(item)
    body = json.dumps({
        "model": model, "reasoning": {"effort": "low"}, "store": False, "stream": False,
        "instructions": "Answer the user's question about the image. Return only JSON.",
        "input": [{"role": "user", "content": [
            {"type": "input_text", "text": PROMPT},
            {"type": "input_image", "image_url": "data:image/jpeg;base64," + jpeg_b64(frame), "detail": "high"},
            {"type": "input_image", "image_url": "data:image/jpeg;base64," + jpeg_b64(crop), "detail": "high"}]}],
    }).encode()
    rate.wait()
    req = urllib.request.Request(url + "/responses", body,
                                 {"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=180) as resp:
        final = json.loads(resp.read())
    got = str(final.get("model", ""))
    if not got.startswith("gpt-6"):
        raise ValueError(f"returned model {got!r}, not gpt-6")
    if final.get("status") not in (None, "completed"):
        raise ValueError(f"response status {final.get('status')}")
    text = "".join(c.get("text", "") for it in final.get("output", []) for c in it.get("content", []) or []
                   if c.get("type") == "output_text")
    m = re.search(r"\{.*\}", text, re.S)
    d = json.loads(m.group(0)) if m else None
    if not isinstance(d, dict) or d.get("wearer") not in (True, False, None):
        raise ValueError(f"unparseable answer {text[:120]!r}")
    return {"wearer": d["wearer"], "reason": str(d.get("reason", ""))[:200], "model": got,
            "usage": final.get("usage"), "sec": round(time.time() - t, 2)}


def run(a):
    key, url, model = env()
    if a.frame_wh:
        import src.semhand.qwen as qw
        qw.FRAME_WH = tuple(int(v) for v in a.frame_wh.lower().split("x"))
    items = {k: v for k, v in load_items(a.root).items() if v["kind"] == "fresh"}
    out = os.path.join(a.root, "gpt6", "answers.jsonl")
    fail = os.path.join(a.root, "gpt6", "failures.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    done = set()
    if os.path.exists(out):
        done = {json.loads(l)["id"] for l in open(out) if l.strip()}
    todo = sorted(k for k in items if k not in done)[:a.limit or None]
    print(f"{len(items)} 只手，已答 {len(done)}，本次 {len(todo)}；并发 {a.concurrency}，速率上限 {a.rate}/s", flush=True)
    rate, lock = Rate(a.rate), threading.Lock()
    stats = collections.Counter()
    t0 = time.time()

    def work(k):
        err = None
        for attempt in range(1, MAX_TRIES + 1):
            try:
                r = ask(items[k], key, url, model, rate)
                with lock, open(out, "a") as fh:
                    fh.write(json.dumps(dict(r, id=k, attempts=attempt)) + "\n")
                    stats["ok"] += 1
                    stats[f"wearer_{r['wearer']}"] += 1
                return
            except urllib.error.HTTPError as e:
                err = f"HTTP {e.code}"
                if e.code == 429:
                    ra = e.headers.get("Retry-After")
                    rate.slow(float(ra) if ra and ra.replace(".", "").isdigit() else None)
                elif e.code in (401, 402, 403):
                    raise SystemExit(f"鉴权或额度错误 {e.code}，停止")
                else:
                    time.sleep(2 ** attempt)
            except Exception as e:                       # network, parse, wrong model
                err = f"{type(e).__name__}: {e}"
                time.sleep(2 ** attempt)
        with lock, open(fail, "a") as fh:
            fh.write(json.dumps({"id": k, "error": err.replace(key, "[REDACTED]")[:300]}) + "\n")
            stats["failed"] += 1

    with cf.ThreadPoolExecutor(a.concurrency) as pool:
        futs = [pool.submit(work, k) for k in todo]
        for n, f in enumerate(cf.as_completed(futs), 1):
            f.result()
            if n % 200 == 0 or n == len(todo):
                el = time.time() - t0
                print(f"  {n}/{len(todo)}  {n / el:.2f}/s  {dict(stats)}", flush=True)
    print("ALL_DONE", flush=True)


def score(a):
    """GPT-6 per-hand vs the human audit answers, beside Q1, per stratum."""
    import glob
    key = json.load(open(a.key))
    human = json.load(open(a.answers))["answers"]
    g6 = {}
    for line in open(os.path.join(a.root, "gpt6", "answers.jsonl")):
        r = json.loads(line)
        g6[int(r["id"].split("|")[1])] = r["wearer"]
    q1 = {}
    for f in glob.glob(os.path.join(a.root, "qwen", "Q1_*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            q1[int(r["id"].split("|")[1])] = r["p"] >= 0.5
    tab = collections.defaultdict(collections.Counter)
    for it in key["items"]:
        h = human.get(str(it["i"]))
        if h not in ("owner", "other") or it["i"] not in g6:
            continue
        s = it["stratum"]
        for name, v in (("GPT-6 per-hand", g6[it["i"]]), ("Q1", q1.get(it["i"]))):
            if v is None:
                tab[(name, s)]["null"] += 1
            elif (v and h == "owner") or (not v and h == "other"):
                tab[(name, s)]["right"] += 1
            elif v:
                tab[(name, s)]["other_called_wearer"] += 1
            else:
                tab[(name, s)]["wearer_called_other"] += 1
    print(f"GPT-6 已答 {len(g6)} / {len(key['items'])}（null {sum(1 for v in g6.values() if v is None)}）")
    print(f"  {'':<16}{'类':>3}{'对':>6}{'别人→自己':>10}{'自己→别人':>10}{'null':>6}")
    for name in ("GPT-6 per-hand", "Q1"):
        tot = collections.Counter()
        for s in "ABCDEF":
            c = tab[(name, s)]
            tot.update(c)
            print(f"  {name:<16}{s:>3}{c['right']:>6}{c['other_called_wearer']:>10}{c['wearer_called_other']:>10}{c['null']:>6}")
        print(f"  {name:<16}{'合计':>3}{tot['right']:>6}{tot['other_called_wearer']:>10}{tot['wearer_called_other']:>10}{tot['null']:>6}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("run", "score"), default="run")
    ap.add_argument("--root", required=True)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--rate", type=float, default=2.0, help="max request starts per second")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--frame_wh", default=None)
    ap.add_argument("--key", default="/workspace/audit_gpt6_qwen/audit_key.json")
    ap.add_argument("--answers", default="/workspace/audit_gpt6_qwen/answers.json")
    a = ap.parse_args()
    (run if a.mode == "run" else score)(a)


if __name__ == "__main__":
    main()
