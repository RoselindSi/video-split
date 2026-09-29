"""Compare the exact token sequences vLLM and transformers build for the same prompt.

WHY TOKENS AND NOT MORE SCORES. Three explanations survive for vLLM and
transformers disagreeing on 30% of views with the same weights -- the chat
template, the image patchification, and how the assistant prefix is joined --
and no amount of further scoring separates them. All three change the token
sequence, and each changes it in a different, recognisable place:

    template        the opening tokens, and where the image placeholders sit
    patchification  how many vision tokens one image expands into
    prefix          the last few tokens, where `{"wearer":` is appended

So the sequences are put side by side once. vLLM's `/tokenize` returns what it
will actually feed; `Qwen.inputs` is what the current scorer feeds. If the
tails differ the prefix is the answer; if the vision token counts differ the
processor settings are; if the heads differ the template is. If they are
identical, all three are cleared and the difference is in the forward pass
itself, which is a different investigation.

THE SCORE COMES FROM THE LAST POSITION, so a single extra or missing token at
the tail is not a rounding difference -- it means the two runs are reading the
distribution after different text, and every downstream number is
incomparable.
"""
from __future__ import annotations

import argparse
import base64
import collections
import csv
import json
import os
import urllib.request

PREFIX = '{"wearer":'


def data_url(path):
    raw = open(path, "rb").read()
    ext = os.path.splitext(path)[1].lower()
    mime = "image/png" if ext == ".png" else "image/jpeg"
    return "data:%s;base64,%s" % (mime, base64.b64encode(raw).decode())


def vllm_tokens(url, model, full, crop, intro, question):
    body = {
        "model": model,
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": intro},
                {"type": "image_url", "image_url": {"url": data_url(full)}},
                {"type": "image_url", "image_url": {"url": data_url(crop)}},
                {"type": "text", "text": question},
            ]},
            {"role": "assistant", "content": PREFIX},
        ],
        "add_generation_prompt": False,
        "continue_final_message": True,
        "add_special_tokens": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        url.rstrip("/") + "/tokenize",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:18997")
    ap.add_argument("--model", default="qwen-bf16")
    ap.add_argument("--views", required=True)
    ap.add_argument("--hf", default="/workspace/models/Qwen3.8-27B")
    ap.add_argument("--id", default="",
                    help="比哪一条视图；留空取第一条。判定分歧关在一条录像里时，"
                         "要比的是那条录像的视图，不是恰好排在最前的那条")
    a = ap.parse_args()

    from src.semhand.owner_gate import CONTEXT_INTRO, CONTEXT_QUESTION
    from src.semhand.qwen import ANSWER_PREFIX

    items = list(csv.DictReader(open(os.path.join(a.views, "manifest.csv"),
                                     encoding="utf-8")))
    if a.id:
        hit = [i for i in items if i["id"] == a.id]
        if not hit:
            raise SystemExit("manifest 里没有 %s" % a.id)
        it = hit[0]
    else:
        it = items[0]
    full, crop = it.get("context") or it["full"], it["crop"]
    print("用这一条比：%s" % it["id"])
    print("  整帧 %s" % full)
    print("  裁剪 %s\n" % crop)
    from PIL import Image as _I
    for nm, pth in (("整帧", full), ("裁剪", crop)):
        with _I.open(pth) as _im:
            print("  %s 尺寸 %dx%d" % (nm, _im.width, _im.height))
    print()

    v = vllm_tokens(a.url, a.model, full, crop, CONTEXT_INTRO, CONTEXT_QUESTION)
    vt = v.get("tokens") or v.get("token_ids") or []
    print("vLLM /tokenize -> %d 个 token（字段 %s）"
          % (v.get("count", len(vt)), ",".join(sorted(v))))

    # transformers 侧：只用处理器复现 Qwen.inputs 的那三步，不加载模型。
    # 加载模型要 GPU，而这里比的是 token 序列，和权重无关。
    from PIL import Image
    from transformers import AutoProcessor
    from qwen_vl_utils import process_vision_info
    proc = AutoProcessor.from_pretrained(a.hf)
    ps = int(proc.image_processor.patch_size)
    conv = [{"role": "user", "content": [
        {"type": "text", "text": CONTEXT_INTRO},
        {"type": "image", "image": Image.open(full).convert("RGB")},
        {"type": "image", "image": Image.open(crop).convert("RGB")},
        {"type": "text", "text": CONTEXT_QUESTION},
    ]}]
    text = proc.apply_chat_template([conv][0], tokenize=False,
                                    add_generation_prompt=True,
                                    enable_thinking=False) + ANSWER_PREFIX
    images, _ = process_vision_info([conv], image_patch_size=ps)
    x = proc(text=[text], images=images, padding=True, return_tensors="pt")
    ht = x["input_ids"][0].tolist()
    q = type("Q", (), {"proc": proc})()
    print("transformers   -> %d 个 token" % len(ht))
    if "image_grid_thw" in x:
        print("  image_grid_thw = %s" % x["image_grid_thw"].tolist())

    print("\n=== 三处对照 ===")
    print("总长度      vLLM %d   transformers %d   差 %+d"
          % (len(vt), len(ht), len(vt) - len(ht)))

    # patch 化：视觉占位 token 出现多少次
    cv, ch = collections.Counter(vt), collections.Counter(ht)
    vis = [t for t, n in ch.items() if n > 50]        # 图像占位符必然高频
    for t in sorted(vis, key=lambda t: -ch[t])[:3]:
        print("  高频 token %-8d vLLM %6d 次   transformers %6d 次%s"
              % (t, cv.get(t, 0), ch[t],
                 "   ← 图像 patch 数不同" if cv.get(t, 0) != ch[t] else ""))

    print("\n开头 20 个")
    print("  vLLM         %s" % vt[:20])
    print("  transformers %s" % ht[:20])
    print("  一致" if vt[:20] == ht[:20] else "  ← 不一致：chat template 不同")
    # 分叉点之前一样、之后不一样，解码出来直接看是哪几个字
    i = 0
    while i < min(len(vt), len(ht)) and vt[i] == ht[i]:
        i += 1
    print("\n第 %d 个 token 开始分叉，两边各解码 60 个 token：" % i)
    tk0 = q.proc.tokenizer
    print("  vLLM         %r" % tk0.decode(vt[max(0, i - 4):i + 60]))
    print("  transformers %r" % tk0.decode(ht[max(0, i - 4):i + 60]))

    print("\n结尾 16 个（前缀就接在这里，打分读的是最后一个位置）")
    print("  vLLM         %s" % vt[-16:])
    print("  transformers %s" % ht[-16:])
    print("  一致" if vt[-16:] == ht[-16:] else "  ← 不一致：助手前缀的拼法不同")

    if vt and ht:
        tk = q.proc.tokenizer if hasattr(q.proc, "tokenizer") else None
        if tk is not None:
            print("\n结尾解码")
            print("  vLLM         %r" % tk.decode(vt[-16:]))
            print("  transformers %r" % tk.decode(ht[-16:]))


if __name__ == "__main__":
    main()
