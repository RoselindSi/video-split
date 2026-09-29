"""Ask the hand-ness teacher about the rows it was never asked about.

WHY THE COMPARISON NEEDED THIS. The teacher was run over the pool of candidate
negatives, so on the current manifest it has a score for 98.7% of the rows a
person called `nothand` and 2.8% of the rows a person called `hand`. Putting
student and teacher side by side was therefore restricted to 85 rows, 75 of
whose negatives came from a single recording -- one recording is not a
population, and the positives in it are the hardest hands in the set rather
than typical ones. Nothing about that is fixed by a better statistic; the
missing scores have to be produced.

THEY ARE CHEAP TO PRODUCE, which is the other half of the reason. Hand-ness is
a single forward pass -- the assistant turn is opened at `{"hand":` and P(hand)
is the softmax over the next-token logits for ` true` against ` false`, so
nothing is generated. That is the inexpensive half of what the owner gate
does, and against a vLLM server with a few requests in flight a few hundred
rows take about a minute. The reason they were missing was never cost.

SAME PROMPT, SAME IMAGES, SAME PREFIX as src.semhand.handness, imported rather
than restated, so the scores that come back are comparable to the ones already
in the manifest instead of being a second teacher that happens to agree.

The output is a jsonl keyed by item_id; it is not merged into the manifest
here. A filled score is evidence for a comparison, not a label, and writing it
into the manifest would make it indistinguishable from the rows the negative
pool was actually built from.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:18997")
    ap.add_argument("--model", default="qwen-bf16")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--only-missing", dest="only_missing", action="store_true",
                    help="只补 teacher_p 为空的行")
    ap.add_argument("--labelled-only", dest="labelled_only", action="store_true",
                    help="只打有人工标签的行（对比只用得上这些）")
    ap.add_argument("--conc", type=int, default=8)
    ap.add_argument("--top", type=int, default=64)
    a = ap.parse_args()

    from src.semhand.handness import QUESTION, PREFIX
    from src.semhand.vllm_probe import data_url, two_way, run_phase
    import urllib.request

    rows = [r for r in csv.DictReader(open(a.manifest, encoding="utf-8"))
            if r["split"] == a.split]
    if a.labelled_only:
        rows = [r for r in rows if r.get("human_label") in ("hand", "nothand")]
    if a.only_missing:
        rows = [r for r in rows if not r.get("teacher_p")]
    rows = [r for r in rows
            if os.path.exists(r["full_path"]) and os.path.exists(r["crop_path"])]
    print("要打 %d 行（split=%s）" % (len(rows), a.split))
    if not rows:
        raise SystemExit("没有可打的行")

    def ask(r):
        body = {
            "model": a.model,
            "messages": [
                {"role": "user", "content": [
                    {"type": "image_url",
                     "image_url": {"url": data_url(r["full_path"])}},
                    {"type": "image_url",
                     "image_url": {"url": data_url(r["crop_path"])}},
                    {"type": "text", "text": QUESTION},
                ]},
                {"role": "assistant", "content": PREFIX},
            ],
            "max_tokens": 1, "temperature": 0.0,
            "logprobs": True, "top_logprobs": a.top,
            "add_generation_prompt": False, "continue_final_message": True,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        req = urllib.request.Request(
            a.url.rstrip("/") + "/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as resp:
            return json.loads(resp.read())

    t0 = time.time()
    responses = run_phase(rows, ask, a.conc)
    dt = time.time() - t0

    written, missing = 0, 0
    with open(a.out, "w", encoding="utf-8") as fh:
        for r, resp in zip(rows, responses):
            p, seen = two_way(resp)
            if p is None:
                missing += 1
                continue
            fh.write(json.dumps({
                "item_id": r["item_id"], "rec": r["rec"],
                "human_label": r.get("human_label", ""),
                "p": p, "had_teacher_p": bool(r.get("teacher_p")),
            }) + "\n")
            written += 1
    print("%.1fs，%.1f 行/分钟 -> %s（%d 行）"
          % (dt, 60 * len(rows) / dt, a.out, written))
    if missing:
        print("true/false 没同时进 top-%d 的：%d 行" % (a.top, missing))


if __name__ == "__main__":
    main()
