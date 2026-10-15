"""E1b: given the boundary, can Qwen say whether the state changed?

TWO CONDITIONS, AND THE ONE THE PLAN ASKED FOR IS NOT BUILDABLE.

  video     the footage around the boundary, the state change at its midpoint
  selftext  E1a's own per-segment descriptions of the two sides, no footage

The design called for a third, `oracle-text`: human semantic descriptions
instead of video, to see whether relation reasoning works once perception is
removed. It cannot be built from what exists. v1's descriptions hang on the
NODE and are reused across its occurrences, so for every one of the 133
SAME_STATE candidates the two sides carry a byte-identical string and for all
174 DIFFERENT_STATE candidates they do not. A rule reading "same string ->
SAME_STATE" scores 100.0%. That arm would not measure reasoning; it would
measure string equality, and would be read as evidence that perfect
perception makes the task easy.

E1a's descriptions are written per OCCURRENCE, so the same node described
twice gets two independently written strings. The same string rule scores
65.8% on them -- above the 56.7% majority baseline, so the text carries real
signal, but far from the answer. That makes `selftext` a condition with
something to beat, and the 65.8% is reported beside it every time.

WHAT THE THIRD COMPARISON BUYS. If video clearly beats selftext, splitting
perception from reasoning loses information -- and every history
representation in E2 (M1 flat text, M2 node memory, M3 graph) is text. That
would be a result about E2's whole design, found before E2 runs.

TASK RELATION IS ASKED BUT NOT SCORED AS ACCURACY. 302 of 307 candidates are
SAME_TASK. Any accuracy figure here reports the base rate; per-class recall
with its n printed beside it is the only honest form, and E3 needs an
enriched set before the question can really be asked.
"""
from __future__ import annotations

import argparse
import json
import os

MODEL = "/workspace/tr1/ckpts/Qwen3-VL-8B-Instruct"
FPS = 1.0

SYSTEM = ("你是视频动作的观察者。只根据给出的材料判断，不猜测看不到的意图。"
          "只输出合法 JSON。")

VIDEO_PROMPT = """这段视频来自佩戴在头部的相机。**正中间那一刻**，人工标注认为这里发生了一次变化。

前半段和后半段各是一个动作状态。请判断这两个状态之间的关系。

只输出一个 JSON 对象：

{{"state_relation": "SAME_STATE" | "DIFFERENT_STATE" | "UNCERTAIN",
  "task_relation": "SAME_TASK" | "RETURN_TASK" | "DIFFERENT_TASK" | "UNCERTAIN",
  "why": "<十几个字，说你看到了什么>"}}

取值的意思：
- SAME_STATE：前后是**同一种**交互状态。它中断之后又重新开始了同一件事（例如洗完一个杯子，再拿起下一个杯子洗）。
- DIFFERENT_STATE：前后是**不同的**交互状态，动作或被作用的物体换了。
- SAME_TASK：前后属于同一个直接任务目标。
- RETURN_TASK：后半段回到了**更早出现过、中间离开过**的任务。只凭这段视频看不出来就不要选它。
- DIFFERENT_TASK：换了另一个任务目标。
- UNCERTAIN：看不清或判不了。

注意：同一个状态内部的来回反复不是变化；只有状态本身换了才是 DIFFERENT_STATE。
JSON 之外不要有任何文字。"""

TEXT_PROMPT = """下面是同一段工作视频里**相邻两个时间段**的描述，由观察者分别写出。你看不到视频。

前一段：
  物体：{a_obj}
  动作：{a_act}
  结果：{a_res}

后一段：
  物体：{b_obj}
  动作：{b_act}
  结果：{b_res}

请判断这两个状态之间的关系。只输出一个 JSON 对象：

{{"state_relation": "SAME_STATE" | "DIFFERENT_STATE" | "UNCERTAIN",
  "task_relation": "SAME_TASK" | "RETURN_TASK" | "DIFFERENT_TASK" | "UNCERTAIN",
  "why": "<十几个字>"}}

取值的意思：
- SAME_STATE：两段是**同一种**交互状态，只是分成了两次（例如洗完一个杯子，再洗下一个）。
- DIFFERENT_STATE：动作或被作用的物体换了。
- SAME_TASK / RETURN_TASK / DIFFERENT_TASK：是否属于同一个直接任务目标；判不了就写 UNCERTAIN。

两段的描述是分别写的，**用词不同不代表状态不同，用词相同也不代表状态相同**。按内容判断。
JSON 之外不要有任何文字。"""


def nz(v):
    return "（未写）" if v is None or str(v).strip().lower() in ("", "null", "none") else str(v)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--arm", choices=("video", "selftext"), required=True)
    ap.add_argument("--clips", help="video 臂需要")
    ap.add_argument("--e1a", help="selftext 臂需要：E1a 的 jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--fps", type=float, default=FPS)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    cands = json.load(open(a.candidates, encoding="utf-8"))["candidates"]
    if a.limit:
        cands = cands[:a.limit]

    seg = {}
    if a.arm == "selftext":
        for line in open(a.e1a, encoding="utf-8"):
            d = json.loads(line)
            if d.get("parsed"):
                seg[d["segment_id"]] = d["parsed"]

    done = set()
    out_path = a.out if a.nshard == 1 else "%s.%d" % (a.out, a.shard)
    for p in (a.out, out_path):
        if os.path.exists(p):
            for line in open(p, encoding="utf-8"):
                try:
                    done.add(json.loads(line)["cand_id"])
                except Exception:
                    pass
    todo = [r for i, r in enumerate(cands)
            if i % a.nshard == a.shard and r["cand_id"] not in done]
    print("%s 臂：本分片待跑 %d（总体 %d，已完成 %d）"
          % (a.arm, len(todo), len(cands), len(done)))
    if not todo:
        return

    from src.auditor.vision_backends import build_backend
    be = build_backend("qwen", a.model)
    with open(out_path, "a", encoding="utf-8") as fh:
        for i, r in enumerate(todo):
            if a.arm == "video":
                clip = os.path.join(a.clips, r["cand_id"].replace("@", "_t") + ".mp4")
                if not os.path.exists(clip):
                    print("  缺片段 %s" % r["cand_id"])
                    continue
                raw = be.generate(SYSTEM, VIDEO_PROMPT, video=clip, fps=a.fps,
                                  temperature=0.0, max_new_tokens=160)
            else:
                ka = "%s#%02d" % (r["video_id"], r["prev_seg_index"])
                kb = "%s#%02d" % (r["video_id"], r["next_seg_index"])
                pa, pb = seg.get(ka), seg.get(kb)
                if not pa or not pb:
                    print("  缺 E1a 描述 %s" % r["cand_id"])
                    continue
                prompt = TEXT_PROMPT.format(
                    a_obj=nz(pa.get("object")), a_act=nz(pa.get("observed_action")),
                    a_res=nz(pa.get("observed_result")),
                    b_obj=nz(pb.get("object")), b_act=nz(pb.get("observed_action")),
                    b_res=nz(pb.get("observed_result")))
                raw = be.generate(SYSTEM, prompt, temperature=0.0,
                                  max_new_tokens=160)
            rec = {"cand_id": r["cand_id"], "arm": a.arm,
                   "video_id": r["video_id"], "boundary_s": r["boundary_s"],
                   "gold_state": r["gold_state"], "gold_task": r["gold_task"],
                   "window_clean": r["window_clean"],
                   "n_frames_1fps": r["n_frames_1fps"], "raw": raw}
            try:
                s = raw[raw.index("{"):raw.rindex("}") + 1]
                rec["parsed"] = json.loads(s)
            except Exception as e:
                rec["parse_error"] = str(e)[:120]
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            if i % 25 == 0:
                print("  %d/%d" % (i, len(todo)), flush=True)
    print("-> %s" % out_path)


if __name__ == "__main__":
    main()
