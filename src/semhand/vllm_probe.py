"""Can a vLLM server stand in for the local forward pass the owner gate scores with?

WHAT THE GATE ACTUALLY NEEDS is not generated text. `Qwen.score` appends the
assistant prefix `{"wearer":`, runs one forward pass, takes the logits at the
final position and softmaxes over exactly two token ids -- " true" and
" false". The returned `p` is that two-way probability, and every threshold in
the gate is calibrated against it.

SO THERE ARE TWO THINGS TO VERIFY, and a server that does the first but not
the second is useless here:

    prefill    the server has to continue a partially written assistant turn
               rather than start a fresh one, or the distribution being read
               is the one after `{"wearer":` has yet to be written
    logprobs   both " true" and " false" have to appear in the returned
               top-k at that position. If only the winner comes back, the
               two-way renormalisation cannot be reproduced and `p` collapses
               to 1.0 or 0.0 -- which would look like a working probe while
               silently destroying every threshold downstream.

THE COMPARISON IS AGAINST THE SCORES ALREADY ON DISK, view by view, because
the question is not whether the server answers but whether it answers the same.
FP8 and bf16 differ numerically; what matters is whether that difference ever
crosses a decision boundary.
"""
from __future__ import annotations

import argparse
import base64
import collections
import glob
import io
import json
import math
import os
import statistics
import time
import urllib.request

PREFIX = '{"wearer":'


def data_url(path, reencode=False, max_side=1280):
    """原始字节直传，不做 PIL 往返。

    第一版为了传输方便把图 resize 到 1280 再按 q88 重新编码 JPEG。那在比较里
    多引入了一个变量，而且偏偏是最可疑的一个——这个门判的是 60-100px 的小框，
    重压缩正好抹掉那一带的细节。默认直传原始字节；`--reencode` 留着是为了
    单独测量重编码本身值多少。
    """
    if not reencode:
        raw = open(path, "rb").read()
        ext = os.path.splitext(path)[1].lower()
        mime = "image/png" if ext == ".png" else "image/jpeg"
        return "data:%s;base64,%s" % (mime, base64.b64encode(raw).decode())
    from PIL import Image
    im = Image.open(path).convert("RGB")
    if max(im.size) > max_side:
        sc = max_side / float(max(im.size))
        im = im.resize((int(im.width * sc), int(im.height * sc)))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def ask(url, model, full, crop, intro, question, top=64, timeout=180,
        reencode=False, description=None):
    # 带不带描述必须和基线一致。owner_gate 打分是两遍的：先生成一段中性描述，
    # 再把 `Description: ...` 插在两张图和问题之间重走一遍。只送图和问题得到的
    # 是另一个提示下的分布，和基线不可比——两边引擎都没错，是提示不同。
    content = [
        {"type": "text", "text": intro},
        {"type": "image_url", "image_url": {"url": data_url(full, reencode)}},
        {"type": "image_url", "image_url": {"url": data_url(crop, reencode)}},
    ]
    if description:
        content.append({"type": "text", "text": "Description: %s" % description})
    content.append({"type": "text", "text": question})
    body = {
        "model": model,
        "messages": [
            {"role": "user", "content": content},
            # 助手前缀：让它接着写，而不是重开一轮
            {"role": "assistant", "content": PREFIX},
        ],
        "max_tokens": 1, "temperature": 0.0,
        "logprobs": True, "top_logprobs": top,
        "add_generation_prompt": False, "continue_final_message": True,
        # 不传这个，模板会按 reasoning_effort 的默认值 'xhigh' 插一段
        # system 进去（42 个 token），而打分那条路没有它。同一个模型、
        # 同样的图，前面多一段「请深思熟虑」，最后一个位置读到的就是
        # 另一个分布——之前 30% 的判定分歧全出在这里。
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def two_way(resp):
    """-> (p_true, 看到的候选) 只在 true/false 两个 token 上重新归一。"""
    lp = resp["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    seen = {e["token"]: e["logprob"] for e in lp}
    t = next((v for k, v in seen.items() if k.strip() == "true"), None)
    f = next((v for k, v in seen.items() if k.strip() == "false"), None)
    if t is None or f is None:
        return None, seen
    m = max(t, f)
    et, ef = math.exp(t - m), math.exp(f - m)
    return et / (et + ef), seen


def describe(url, model, full, crop, prompt, timeout=300, reencode=False,
             max_tokens=160):
    """第一遍：生成中性描述。生产路径的贵的那一半。

    打分那一遍只取 1 个 token 的 logits，这一遍要真生成约 160 个。所以比较
    两个实现的速度时，只测打分等于只测便宜的一半——`--with-description` 复用
    基线存好的描述，那是在借 transformers 已经算过的结果。
    """
    body = {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": data_url(full, reencode)}},
            {"type": "image_url", "image_url": {"url": data_url(crop, reencode)}},
        ]}],
        "max_tokens": max_tokens, "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


def format_desc(raw):
    """把生成的 JSON 排成 owner_gate 喂给第二遍的那一行字。

    解析和排版都用生产那份代码，不在这里重写一遍——否则测的是我复刻得像不像，
    不是两个实现快慢。
    """
    from src.semhand.qwen import parse_desc, sem_text
    parsed = parse_desc(raw)
    if parsed is None:
        return sem_text({"raw": raw, "parsed": None}), False
    return ("%s hand; motion path: %s; interaction: %s; object: %s; "
            "occlusion: %s; ambiguous: %s"
            % (parsed.get("hand_side", "unclear"), parsed.get("motion_path", ""),
               parsed.get("interaction", ""), parsed.get("object", ""),
               parsed.get("occlusion", ""),
               parsed.get("ambiguous_regions", ""))), True


def run_phase(items, fn, conc):
    """并发跑一阶段。vLLM 的本事是连续批处理，单路测不出来。"""
    if conc <= 1:
        return [fn(x) for x in items]
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(conc) as ex:
        return list(ex.map(fn, items))


def gap(p, cap=30.0):
    """概率还原成 logit 差，因为 p 是两路 softmax 出来的。

    比较两个引擎时 p 会骗人：0.9999 和 0.99 看着都是「是」，但一个的 logit 差
    是 9 另一个是 4.6，差了一倍。分歧到底是数值噪声把一个贴着 0.5 的判定推过
    界，还是两边真的各自很确信、方向还相反——只有在 logit 上看得出来。p 到
    了 1.0 就饱和了，所以夹一下再取对数，免得算出 inf 把整张表毁掉。
    """
    p = min(max(p, 1e-13), 1 - 1e-13)
    return max(-cap, min(cap, math.log(p / (1 - p))))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:18997")
    ap.add_argument("--model", default="qwen-fp8")
    ap.add_argument("--views", required=True)
    ap.add_argument("--baseline", required=True, help="bf16 打分 jsonl 的 glob")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--top", type=int, default=64)
    ap.add_argument("--reencode", action="store_true",
                    help="传图前用 PIL 重新编码（第一版的行为，单独测它值多少）")
    ap.add_argument("--tag", default="")
    ap.add_argument("--with-description", dest="with_description",
                    action="store_true",
                    help="把基线记录里存好的描述原文插回提示，和基线同一条路径")
    ap.add_argument("--describe", action="store_true",
                    help="自己跑第一遍生成描述（生产路径）。和 --with-description "
                         "的区别是后者借的是 transformers 已经算好的描述")
    ap.add_argument("--conc", type=int, default=1,
                    help="并发请求数；vLLM 的连续批处理单路测不出来")
    ap.add_argument("--dump", default="",
                    help="每条视图写一行 jsonl：两边的 logit 差和裁剪框尺寸")
    a = ap.parse_args()

    from src.semhand.owner_gate import CONTEXT_INTRO, CONTEXT_QUESTION
    import csv

    base, desc = {}, {}
    for path in glob.glob(a.baseline):
        for line in open(path, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                base[r["id"]] = float(r["p"])
                desc[r["id"]] = r.get("description") or ""
    n_desc = sum(1 for v in desc.values() if v)
    print("基线里 %d/%d 条带描述 ← 基线走的是两遍路径" % (n_desc, len(desc))
          if n_desc else "基线不带描述")
    if n_desc and not (a.with_description or a.describe):
        print("  本次不带描述发送：比的是「提示不同」，不是「引擎不同」")

    items = list(csv.DictReader(open(os.path.join(a.views, "manifest.csv"),
                                     encoding="utf-8")))
    items = [i for i in items if i["id"] in base][:a.n]
    print("拿 %d 个视图对照（bf16 基线共 %d 条）" % (len(items), len(base)))

    dump = open(a.dump, "w", encoding="utf-8") if a.dump else None

    def ctx(it):
        return it.get("context") or it["full"]

    # 第一遍
    t_desc, n_parsed = 0.0, 0
    if a.describe:
        from src.semhand.owner_gate import CONTEXT_DESCRIBE
        t0 = time.time()
        raws = run_phase(items, lambda it: describe(
            a.url, a.model, ctx(it), it["crop"], CONTEXT_DESCRIBE,
            reencode=a.reencode), a.conc)
        t_desc = time.time() - t0
        own = {}
        for it, raw in zip(items, raws):
            txt, ok = format_desc(raw)
            own[it["id"]] = txt
            n_parsed += int(ok)
        print("  第一遍 %.1fs（%.1f 条/分钟），解析成功 %d/%d"
              % (t_desc, 60 * len(items) / t_desc, n_parsed, len(items)))
        desc_used = own
    elif a.with_description:
        desc_used = desc
    else:
        desc_used = {}

    # 第二遍
    missing, t0 = 0, time.time()
    resps = run_phase(items, lambda it: ask(
        a.url, a.model, ctx(it), it["crop"], CONTEXT_INTRO, CONTEXT_QUESTION,
        top=a.top, reencode=a.reencode,
        description=desc_used.get(it["id"])), a.conc)
    t_score = time.time() - t0
    print("  第二遍 %.1fs（%.1f 条/分钟）" % (t_score, 60 * len(items) / t_score))

    rows = []
    for i, (it, resp) in enumerate(zip(items, resps)):
        p, seen = two_way(resp)
        if p is None:
            missing += 1
            if missing <= 2:
                print("  两个 token 没同时出现，top-%d 里是：%s"
                      % (a.top, list(seen)[:8]))
            continue
        rows.append((it["id"], p, base[it["id"]]))
        if dump is not None:
            try:
                from PIL import Image
                with Image.open(it["crop"]) as im:
                    cw, chh = im.size
            except Exception:
                cw = chh = 0
            dump.write(json.dumps({
                "id": it["id"], "rec": it["rec"], "tid": it["tid"],
                "frame": it["frame"],
                "p_vllm": p, "p_base": base[it["id"]],
                "gap_vllm": gap(p), "gap_base": gap(base[it["id"]]),
                "crop_w": cw, "crop_h": chh,
                "desc_src": ("own" if a.describe
                             else "baseline" if a.with_description else "none"),
                "desc_used": desc_used.get(it["id"], ""),
                "desc_base": desc.get(it["id"], ""),
            }) + "\n")
        if i == 0:
            print("  首条：%s p=%.4f  基线 p=%.4f" % (a.model, p, base[it["id"]]))
    if dump is not None:
        dump.close()
        print("  每条明细 -> %s" % a.dump)
    dt = t_desc + t_score
    print("\n端到端 %.1fs（第一遍 %.1f + 第二遍 %.1f），%d 条 -> %.1f 条/分钟"
          "（并发 %d）" % (dt, t_desc, t_score, len(items),
                          60 * len(items) / dt, a.conc))
    if missing:
        print("两个 token 没同时进 top-%d 的：%d 条 ← 这种情况无法复现两路归一"
              % (a.top, missing))
    if not rows:
        raise SystemExit("没有可比的条目")

    d = [abs(p - b) for _, p, b in rows]
    d.sort()
    agree = sum(1 for _, p, b in rows if (p >= 0.5) == (b >= 0.5))
    print("\n=== %s vs 基线 ===" % (a.tag or "vLLM"))
    print("  可比 %d 条" % len(rows))
    print("  |Δp| 中位 %.4f  90%% 分位 %.4f  最大 %.4f"
          % (d[len(d) // 2], d[int(0.9 * len(d))], d[-1]))
    print("  硬判定一致 %d/%d = %.1f%%" % (agree, len(rows), 100 * agree / len(rows)))
    big = [(k, p, b) for k, p, b in rows if abs(p - b) > 0.2]
    if big:
        print("  |Δp|>0.2 的 %d 条，前几个：" % len(big))
        for k, p, b in big[:5]:
            print("    %-34s FP8 %.3f  bf16 %.3f" % (k[:34], p, b))


if __name__ == "__main__":
    main()
