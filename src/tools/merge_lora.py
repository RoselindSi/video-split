"""Merge a trained LoRA adapter into its base model and save a full checkpoint.

Needed between curriculum stages: Time-R1's main.py loads the model with
`from_pretrained(path)` expecting a FULL model, but each GRPO stage saves only
the LoRA adapter. So after stage N finishes:

    python merge_lora.py \
        --base  /workspace/tr1/ckpts/Time-R1-7B \
        --adapter /workspace/tr1/time-r1/logs/seg_stage1_iou_format \
        --out   /workspace/tr1/ckpts/seg_stage1_merged

Then stage N+1 uses --model_name_or_path /workspace/tr1/ckpts/seg_stage1_merged
and trains a FRESH LoRA on top of the merged weights.

Run inside venv_train (transformers 4.51.1). Pass the directory that contains
adapter_config.json (the final save dir, or a checkpoint-XX subdir).
"""

import argparse
import os

import torch
from peft import PeftModel
from transformers import AutoProcessor, AutoModelForImageTextToText


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="base model dir (full model)")
    ap.add_argument("--adapter", required=True, help="dir containing adapter_config.json")
    ap.add_argument("--out", required=True, help="output dir for merged full model")
    args = ap.parse_args()

    if not os.path.exists(os.path.join(args.adapter, "adapter_config.json")):
        raise SystemExit(
            f"no adapter_config.json in {args.adapter} — pass the dir that has it "
            "(final output_dir or a checkpoint-XX subdir)"
        )

    print(f"loading base: {args.base}")
    model = AutoModelForImageTextToText.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map="cpu"
    )
    print(f"loading adapter: {args.adapter}")
    model = PeftModel.from_pretrained(model, args.adapter)
    print("merging...")
    model = model.merge_and_unload()

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out, safe_serialization=True)
    # ship the processor/tokenizer alongside so the dir is self-contained
    processor = AutoProcessor.from_pretrained(args.base)
    processor.save_pretrained(args.out)
    print(f"merged model saved to {args.out}")


if __name__ == "__main__":
    main()
