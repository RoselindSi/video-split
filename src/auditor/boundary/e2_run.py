"""E2: the same boundary, the same clip, five ways of being told what came before.

THE PROMPT IS IMPORTED FROM E1b, NOT RETYPED. M0 -- video with no history --
was already run as E1b-video, and it is only a valid zero rung if the
question put to the model is byte-identical. So the history block is
prepended to `e1b_run.VIDEO_PROMPT` and nothing else changes; if that prompt
is ever edited, M0 has to be re-run with it.

WHAT VARIES ACROSS CONDITIONS IS ONLY THE HISTORY BLOCK. Same clip, same fps,
same temperature, same output schema. Prompt length is recorded per call,
because a condition that helps only by being longer is a finding about length
and the plan asks for the length to be on the record.
"""
from __future__ import annotations

import argparse
import json
import os

from src.auditor.boundary.e1b_run import SYSTEM, VIDEO_PROMPT
from src.auditor.boundary.e2_build import render

MODEL = "/workspace/tr1/ckpts/Qwen3-VL-8B-Instruct"
FPS = 1.0
CONDS = ["M1raw", "M1canon", "M2raw", "M2canon", "M3", "M2rawd", "M3raw",
         "M3raw_noloop", "M3raw_shuffled", "M3raw_other",
         "M2lab_perm", "M2lab_opaque", "M2lab_first"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefixes", required=True)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--clips", required=True)
    ap.add_argument("--cond", required=True, choices=CONDS)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--fps", type=float, default=FPS)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    rows = json.load(open(a.prefixes, encoding="utf-8"))["rows"]
    gold = {c["cand_id"]: c for c in
            json.load(open(a.candidates, encoding="utf-8"))["candidates"]}
    if a.limit:
        rows = rows[:a.limit]

    done = set()
    out_path = a.out if a.nshard == 1 else "%s.%d" % (a.out, a.shard)
    for p in (a.out, out_path):
        if os.path.exists(p):
            for line in open(p, encoding="utf-8"):
                try:
                    done.add(json.loads(line)["cand_id"])
                except Exception:
                    pass
    todo = [r for i, r in enumerate(rows)
            if i % a.nshard == a.shard and r["cand_id"] not in done]
    print("%s 分片 %d：待跑 %d（总体 %d，已完成 %d）"
          % (a.cond, a.shard, len(todo), len(rows), len(done)))
    if not todo:
        return

    from src.auditor.vision_backends import build_backend
    be = build_backend("qwen", a.model)
    with open(out_path, "a", encoding="utf-8") as fh:
        for i, r in enumerate(todo):
            g = gold.get(r["cand_id"])
            clip = os.path.join(a.clips, r["cand_id"].replace("@", "_t") + ".mp4")
            if g is None or not os.path.exists(clip):
                print("  跳过 %s" % r["cand_id"])
                continue
            hist = render(r, a.cond)
            prompt = hist + "\n\n" + VIDEO_PROMPT if hist else VIDEO_PROMPT
            raw = be.generate(SYSTEM, prompt, video=clip, fps=a.fps,
                              temperature=0.0, max_new_tokens=160)
            rec = {"cand_id": r["cand_id"], "cond": a.cond,
                   "video_id": r["video_id"], "boundary_s": r["boundary_s"],
                   "gold_state": g["gold_state"], "gold_task": g["gold_task"],
                   "window_clean": g["window_clean"],
                   "n_frames_1fps": g["n_frames_1fps"],
                   "n_history": r["n_history"], "n_nodes_seen": r["n_nodes_seen"],
                   "prompt_chars": len(prompt), "raw": raw}
            try:
                s = raw[raw.index("{"):raw.rindex("}") + 1]
                rec["parsed"] = json.loads(s)
            except Exception as e:
                rec["parse_error"] = str(e)[:120]
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            if i % 40 == 0:
                print("  %d/%d" % (i, len(todo)), flush=True)
    print("-> %s" % out_path)


if __name__ == "__main__":
    main()
