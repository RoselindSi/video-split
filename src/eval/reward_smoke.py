"""Zero-training reward smoke test (approach B).

Uses the existing Time-R1 checkpoint (NO training) to generate a few
multi-segment completions with the seg prompt, then scores them with our
reward functions. Validates the whole path: data -> seg prompt -> generation
-> parse -> reward. Run inside the time-r1 repo root with a single GPU:

    CUDA_VISIBLE_DEVICES=0 python reward_smoke.py --limit 3
"""

import argparse
import json
import os
from types import SimpleNamespace

from transformers import AutoProcessor

from src.vllm_inference.vllm_infer import vllmWrapper
from src.vllm_inference.utils import monkey_patch
from src.utils import process_vision_info_v3
from src.seg_prompt import QUESTION_TEMPLATE_SEG
from src.seg_rewards import (
    iou_seg_reward,
    name_seg_reward,
    seq_reward,
    format_seg_reward,
    parse_segments,
)

monkey_patch()


def build_seg(processor, video, total_pixels):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/workspace/tr1/data_handtask/train_multiseg.json")
    ap.add_argument("--model_base", default="/workspace/tr1/ckpts/Time-R1-7B")
    ap.add_argument("--limit", type=int, default=3)
    ap.add_argument("--total_pixels", type=int, default=3584 * 28 * 28)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--pipeline_parallel_size", type=int, default=1)
    args = ap.parse_args()

    rows = json.load(open(args.data))[: args.limit]
    processor = AutoProcessor.from_pretrained(args.model_base, use_fast=True)
    processor.tokenizer.padding_side = "left"
    model = vllmWrapper(SimpleNamespace(
        model_base=args.model_base,
        pipeline_parallel_size=args.pipeline_parallel_size,
        total_pixels=args.total_pixels,
        max_new_tokens=args.max_new_tokens,
    ))

    for r in rows:
        inputs = build_seg(processor, r["video"], args.total_pixels)
        out = model.generate(inputs, max_new_tokens=args.max_new_tokens)[0]
        sol = [r["solution"]]
        dur = [r["duration"]]
        rew = {
            "format": round(format_seg_reward([out])[0], 3),
            "iou": round(iou_seg_reward([out], sol, durations=dur)[0], 3),
            "seq": round(seq_reward([out], sol)[0], 3),
        }
        try:
            rew["name"] = round(name_seg_reward([out], sol)[0], 3)
        except Exception as e:
            rew["name"] = "skip(%s)" % type(e).__name__

        print("=" * 72)
        print("VIDEO:", os.path.basename(r["video"]),
              "| GT segments:", len(r["solution"]),
              "| parsed pred segments:", len(parse_segments(out)))
        print("--- model output (first 900 chars) ---")
        print(out[:900])
        print("--- rewards ---", rew)

    print("\nSMOKE TEST DONE")


if __name__ == "__main__":
    main()
