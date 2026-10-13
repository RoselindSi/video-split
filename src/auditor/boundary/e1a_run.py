"""E1a: can Qwen describe one segment's visual semantics without inventing a result?

THE SEGMENT IS CUT INTO ITS OWN FILE, not passed as a time range. "Show the
model this segment and nothing else" is then a property of the input rather
than a sentence in the prompt that a long-context model may quietly ignore.
No neighbouring segment, no node label, no cluster, no v1 description, no
graph, and nothing after the segment ends.

THE PROMPT IS FROZEN BEFORE THE RUN AND NOT TOUCHED AFTER. Tuning it against
these 378 outputs would turn a diagnostic into a fit. If it needs development
that happens on dev clips outside the 100 audited ids.

WHAT THE RESULT FIELD IS REALLY FOR. E0 found an annotator writing "selected a
suitable motorcycle" where the video showed a person walking past scooters
holding a phone. The same failure in a model is the one that matters most
here, because every later experiment reads this field as evidence. So the
prompt says plainly that `observed_result` is null when nothing visible
concludes, and the audit sheet scores an invented result as `unsupported`
rather than as a wrong string.

ALL 378 ARE RUN AND CACHED even though only 100 are audited, because
inference is cheap and re-running to widen the audit later is not.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess

MODEL = "/workspace/tr1/ckpts/Qwen3-VL-8B-Instruct"
FPS = 1.0
MANIFESTS = [
    "results/auditor/interaction_graph_mosaic_complement_v1/manifest.json",
    "results/auditor/interaction_graph_timeline_demo/manifest.json",
]

# CHINESE, BECAUSE THE BASELINE IS. The five-fold run this diagnostic has to
# be comparable with uses a Chinese system prompt and Chinese rules, and its
# rule 5 already says "不猜测看不到的意图" -- the same thing this experiment
# measures. Asking in English would fold a language difference into the
# account of Qwen's visual semantics.
SYSTEM = ("你是视频动作的观察者。只报告画面里看得见的东西，"
          "不猜测人物的意图，也不猜测片段结束之后会发生什么。只输出合法 JSON。")

PROMPT = """这是一段第一视角工作视频的一个完整片段，来自佩戴在头部的相机。你看到的就是这整个片段，它前面和后面的内容你都看不到。

只描述这个片段里看得见的东西。

只输出一个 JSON 对象，恰好包含这些键：

{"object": "<手作用的主要物体，几个字；如果没有任何物体被作用就写 null>",
 "observed_action": "<手对它做了什么，几个字>",
 "observed_result": "<这个动作在物体或场景上留下的、你看得见的变化；如果在这个片段里没有任何事情收尾，就写 null>",
 "uncertain": true | false,
 "uncertainty_reason": "<几个字；uncertain 为 false 时写 null>"}

规则：
1. observed_result 必须是你在这个片段里**看见**发生的变化。片段在事情收尾之前就结束、或者结果在画外、被遮挡，就写 null。不要写这个动作大概是为了什么，也不要写接下来多半会发生什么。
2. 把东西从一处搬到另一处，只有当你**看见**它被放下、交出、或者在目的地被使用时，才算有结果；否则 observed_result 写 null。
3. uncertain 只用于「看不出物体是什么」或「看不出手在对它做什么」。对**目的**不确定不是写 true 的理由。
4. 用简短、可观察的中文动作描述，不猜测看不到的意图。
5. JSON 之外不要有任何文字。"""


def load_paths(manifests):
    out = {}
    for p in manifests:
        for v in json.load(open(p, encoding="utf-8"))["videos"]:
            out[v["video_id"]] = v["video"]
    return out


def clip_path(root, seg_id):
    return os.path.join(root, seg_id.replace("#", "_s") + ".mp4")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", default="results/auditor/interaction_graph_e1a/"
                                        "sample_frozen.json")
    ap.add_argument("--clips", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--fps", type=float, default=FPS)
    ap.add_argument("--manifests", nargs="+", default=MANIFESTS)
    ap.add_argument("--prep", action="store_true", help="只切片段，不跑模型")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="冒烟用")
    a = ap.parse_args()

    pop = json.load(open(a.sample, encoding="utf-8"))["population"]
    if a.limit:
        pop = pop[:a.limit]

    if a.prep:
        src = load_paths(a.manifests)
        os.makedirs(a.clips, exist_ok=True)
        made = skipped = 0
        for r in pop:
            p = clip_path(a.clips, r["segment_id"])
            if os.path.exists(p) and os.path.getsize(p) > 1024:
                skipped += 1
                continue
            # Re-encoded rather than stream-copied: a keyframe-aligned copy
            # would silently widen the window by up to a GOP, which is exactly
            # the "nothing outside the segment" guarantee this file exists for.
            subprocess.run(
                ["ffmpeg", "-v", "error", "-ss", "%.2f" % r["start_s"],
                 "-i", src[r["video_id"]], "-t", "%.2f" % r["duration_s"],
                 "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                 "-movflags", "+faststart", "-an", p, "-y"], check=True)
            made += 1
        print("切片 %d 个（已存在 %d）-> %s" % (made, skipped, a.clips))
        return

    done = set()
    if os.path.exists(a.out):
        for line in open(a.out, encoding="utf-8"):
            try:
                done.add(json.loads(line)["segment_id"])
            except Exception:
                pass
    todo = [r for i, r in enumerate(pop)
            if i % a.nshard == a.shard and r["segment_id"] not in done]
    print("本分片待跑 %d（总体 %d，已完成 %d）" % (len(todo), len(pop), len(done)))
    if not todo:
        return

    from src.auditor.vision_backends import build_backend
    be = build_backend("qwen", a.model)
    out_path = a.out if a.nshard == 1 else "%s.%d" % (a.out, a.shard)
    with open(out_path, "a", encoding="utf-8") as fh:
        for i, r in enumerate(todo):
            p = clip_path(a.clips, r["segment_id"])
            if not os.path.exists(p):
                print("  缺片段 %s" % r["segment_id"])
                continue
            raw = be.generate(SYSTEM, PROMPT, video=p, fps=a.fps,
                              temperature=0.0, max_new_tokens=256)
            rec = {"segment_id": r["segment_id"], "video_id": r["video_id"],
                   "batch": r["batch"], "start_s": r["start_s"],
                   "end_s": r["end_s"], "duration_s": r["duration_s"],
                   "raw": raw}
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
