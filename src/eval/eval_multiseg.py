"""Held-out evaluation for multi-segment procedure segmentation (approach B).

Runs any full checkpoint (base Time-R1, seg_stage1_merged, ...) on the val
split and reports:
  - Seg-F1 @ IoU 0.3 / 0.5 / 0.7  (matched pair with IoU>=t counts as TP;
    unmatched pred = FP, unmatched GT = FN)
  - mean IoU over matched pairs
  - boundary compound score (Time-R1 iou_v2 style), averaged per GT
  - name similarity on matched pairs (sentence-transformers, optional)
  - coverage / overlap / count accuracy (sequence quality)
  - format validity rate

Run inside the INFERENCE venv (vllm) from the time-r1 repo root, single GPU:

    source /workspace/tr1/env.sh
    cd /workspace/tr1/time-r1
    CUDA_VISIBLE_DEVICES=0 python eval_multiseg.py \
        --model_base /workspace/tr1/ckpts/seg_stage1_merged \
        --data /workspace/tr1/data_handtask/train_multiseg_val.json \
        --out logs/eval_stage1_val.jsonl
"""

import argparse
import json
import os
import statistics
from types import SimpleNamespace

from transformers import AutoProcessor

from src.vllm_inference.vllm_infer import vllmWrapper
from src.vllm_inference.utils import monkey_patch
from src.utils import process_vision_info_v3
from src.seg_prompt import QUESTION_TEMPLATE_SEG
from src.seg_rewards import parse_segments, greedy_match, _as_segs, _iou

monkey_patch()


def build_seg_inputs(processor, video, total_pixels):
    ele = {"min_pixels": 16 * 28 * 28, "total_pixels": total_pixels}
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are a video analysis expert."}]},
        {"role": "user", "content": [
            {"type": "video", "video": video, **ele},
            {"type": "text", "text": QUESTION_TEMPLATE_SEG},
        ]},
    ]
    _, video_inputs, utils = process_vision_info_v3(messages, return_video_kwargs=True)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    fps = utils["fps"]
    return {
        "raw_prompt_ids": [processor.tokenizer.encode(text, add_special_tokens=False)],
        "multi_modal_data": [{"video": video_inputs}],
        "mm_processor_kwargs": [({"fps": fps} if fps is not None else {})],
    }


def eval_one(preds, gts, dur):
    """Per-video metrics. preds/gts: [(name,s,e)]."""
    pairs = greedy_match(preds, gts)
    ious = [iou for _, _, iou in pairs]

    m = {"n_pred": len(preds), "n_gt": len(gts), "n_match": len(pairs)}
    for t in (0.3, 0.5, 0.7):
        tp = sum(1 for _, _, iou in pairs if iou >= t)
        fp = len(preds) - tp
        fn = len(gts) - tp
        m[f"f1@{t}"] = 2 * tp / max(2 * tp + fp + fn, 1e-6)
    m["mean_iou_matched"] = statistics.mean(ious) if ious else 0.0

    # boundary compound (iou_v2 style), averaged over |GT|
    comp = 0.0
    for pi, gj, iou in pairs:
        _, ps, pe = preds[pi]
        _, gs, ge = gts[gj]
        align = (1 - abs(gs / dur - ps / dur)) * (1 - abs(ge / dur - pe / dur)) if dur else 1.0
        comp += iou * max(0.0, align)
    m["boundary_score"] = comp / max(len(gts), 1)

    # sequence quality against GT window
    if preds and gts:
        t0 = min(g[1] for g in gts)
        t1 = max(g[2] for g in gts)
        span = max(t1 - t0, 1e-6)
        covered, cursor = 0.0, t0
        for _, s, e in sorted(preds, key=lambda x: x[1]):
            s, e = max(s, cursor), min(max(e, cursor), t1)
            if e > s:
                covered += e - s
                cursor = e
        m["coverage"] = min(covered / span, 1.0)
        ps = sorted([(s, e) for _, s, e in preds])
        ov = sum(max(0.0, ps[i][1] - ps[i + 1][0]) for i in range(len(ps) - 1))
        m["overlap"] = min(ov / (sum(e - s for _, s, e in preds) or 1e-6), 1.0)
        m["count_acc"] = max(0.0, 1 - abs(len(preds) - len(gts)) / len(gts))
    else:
        m["coverage"] = m["count_acc"] = 0.0
        m["overlap"] = 1.0
    return m, pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_base", required=True)
    ap.add_argument("--data", default="/workspace/tr1/data_handtask/train_multiseg_val.json")
    ap.add_argument("--out", default="logs/eval_multiseg.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--total_pixels", type=int, default=3584 * 28 * 28)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--pipeline_parallel_size", type=int, default=1)
    ap.add_argument("--no_name_sim", action="store_true", help="skip sentence-transformers")
    args = ap.parse_args()

    rows = json.load(open(args.data))
    if args.limit:
        rows = rows[: args.limit]

    sim_fn = None
    if not args.no_name_sim:
        try:
            from src.seg_rewards import _default_sim_fn
            sim_fn = _default_sim_fn()
        except Exception as e:
            print(f"[warn] name similarity disabled ({type(e).__name__}); "
                  f"install sentence-transformers or pass --no_name_sim")

    processor = AutoProcessor.from_pretrained(args.model_base, use_fast=True)
    processor.tokenizer.padding_side = "left"
    model = vllmWrapper(SimpleNamespace(
        model_base=args.model_base,
        pipeline_parallel_size=args.pipeline_parallel_size,
        total_pixels=args.total_pixels,
        max_new_tokens=args.max_new_tokens,
    ))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fout = open(args.out, "w")
    agg = []
    for r in rows:
        gts = _as_segs(r["solution"])
        inputs = build_seg_inputs(processor, r["video"], args.total_pixels)
        out = model.generate(inputs, max_new_tokens=args.max_new_tokens)[0]
        preds = parse_segments(out)
        m, pairs = eval_one(preds, gts, r.get("duration"))

        m["format_ok"] = 1.0 if preds else 0.0
        if sim_fn and pairs:
            gt_names = [g[0] for g in gts]
            sims = [sim_fn(preds[pi][0], gt_names)[gj] for pi, gj, _ in pairs]
            m["name_sim_matched"] = statistics.mean(sims)
        agg.append(m)

        vid = os.path.basename(r["video"])
        print(f"{vid}: pred={m['n_pred']} gt={m['n_gt']} "
              f"F1@0.5={m['f1@0.5']:.2f} mIoU={m['mean_iou_matched']:.2f} "
              f"cov={m['coverage']:.2f} cnt={m['count_acc']:.2f}")
        fout.write(json.dumps({"video": r["video"], **m, "raw": out}) + "\n")
        fout.flush()

    print("\n==== AGGREGATE (n=%d videos) ====" % len(agg))
    keys = ["f1@0.3", "f1@0.5", "f1@0.7", "mean_iou_matched", "boundary_score",
            "coverage", "overlap", "count_acc", "format_ok"]
    if any("name_sim_matched" in m for m in agg):
        keys.append("name_sim_matched")
    for k in keys:
        vals = [m[k] for m in agg if k in m]
        print(f"{k:20s} {statistics.mean(vals):.3f}")


if __name__ == "__main__":
    main()
