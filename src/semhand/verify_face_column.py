#!/usr/bin/env python3
"""把 batch3 那九条录像的「有脸率」查干净。

为什么需要这个：batch3 的选片和关系表都用 detect2 的 `n_faces` 当有脸信号，
**没有任何大小过滤**。R26_0831_134109 报 42%（400 帧里 331 帧），人工抠帧看过，
一张脸都没有 —— 331 次检测全部落在 biggest_face 0.0038-0.0057（≈115px），
是固定浅色物体。一行错值在 n=9 上把 Spearman 从 0.953 拉到 0.783。

九条里六条的 detect csv 随 scratchpad 一起没了，所以这个脚本要在服务器上跑。

**两步都要做，不许只跑第一步。** 大小过滤是机械的，而我上一次栽就栽在
「相信了一个没对着图像核过的数值」。所以第二步必须把检测器声称有脸的帧抠出来
真的看，大小过滤只是帮你把要看的帧挑出来。

用法:
    python src/semhand/verify_face_column.py --det-dir <放 *.detect2.csv 的目录> \\
        --out results/semhand/hand_gold/batch3_face_verified.json \\
        [--frames-dir <每条录像一个子目录的帧图>] [--dump-dir <抠帧写到哪>]
"""
import argparse
import collections
import csv
import json
import math
import os

# 面积占比阈值 —— **这个值已被推翻，留在这里只是为了让脚本能跑并打印大小分布。**
#
# 原本取 0.010：下界是 R26_0831_134109 已证伪的纯 FP 簇（0.0038-0.0057），
# 上界是「R26_0826_102657 的 0.0198-0.0821 是真脸」。后面那句是断言，没看过图。
# 同一天抠帧看了 102657 的 450/470 两帧：没有脸，实验台加佩戴者自己戴手套的双手。
# 于是已验证的误检横跨 105-545px，而已验证的真脸**一个都没有**。
# 不存在把真假分开的大小阈值 —— 至少在已能查证的数据上不存在。
#
# 要重定阈值，先去有确认 other_hand 的录像（113022/110805/133140/115154/132741/113216）
# 上抠帧，拿到第一个真脸样本再说。在那之前不要把 filtered_pct 当成「真脸率」。
THR = 0.010
W, H = 1920, 1520
GRID = range(400, 800, 10)   # 标注用的抽样格，other_hand 的计数就来自这 40 帧


def per_frame(det_csv):
    """-> {frame: (n_faces, 该帧最大脸的面积占比)}

    `biggest_face` = (w*h)/(W*H)，面积占比。**不要用 `face_px_frac`** —— 那一列
    从 mask 算，mask 为 None 时整列是 0，看上去像「所有脸都是零大小」。
    一行一个框，同一帧多行，脸的字段在行间是重复的，所以取 max 而不是 sum。
    """
    out = {}
    with open(det_csv, encoding='utf-8-sig') as fh:
        for r in csv.DictReader(fh):
            f = int(r['frame'])
            n = int(float(r.get('n_faces') or 0))
            b = float(r.get('biggest_face') or 0.0)
            cur = out.get(f, (0, 0.0))
            out[f] = (max(cur[0], n), max(cur[1], b))
    return out


def side_px(frac):
    return math.sqrt(frac * W * H) if frac > 0 else 0.0


def rates(d):
    if not d:
        return None
    raw = sum(1 for n, _ in d.values() if n > 0)
    flt = sum(1 for _, b in d.values() if b >= THR)
    return {"frames": len(d),
            "raw_pct": round(100 * raw / len(d), 1),
            "filtered_pct": round(100 * flt / len(d), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--det-dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--gold-dir', default='results/semhand/hand_gold/batch3')
    ap.add_argument('--frames-dir', default=None,
                    help='每条录像一个子目录，帧文件名含帧号。给了就抠图供人工看。')
    ap.add_argument('--dump-dir', default=None)
    ap.add_argument('--dump-per-rec', type=int, default=4)
    a = ap.parse_args()

    gold = {}
    for p in sorted(os.listdir(a.gold_dir)):
        if not p.endswith('.json'):
            continue
        g = json.load(open(os.path.join(a.gold_dir, p)))
        gold[g['recording']] = sum(1 for o in g.get('others', [])
                                   if o['kind'] == 'other_hand')

    out = {}
    for fn in sorted(os.listdir(a.det_dir)):
        if not fn.endswith('.detect2.csv'):
            continue
        rec = fn.split('.')[0]
        d = per_frame(os.path.join(a.det_dir, fn))
        grid = {f: v for f, v in d.items() if f in GRID}
        claimed = sorted(f for f, (n, _) in grid.items() if n > 0)
        sizes = sorted({round(b, 4) for _, b in grid.values() if b > 0})
        out[rec] = {"all_window": rates(d), "grid": rates(grid),
                    "other_hand": gold.get(rec),
                    "claimed_face_frames_in_grid": claimed,
                    "sizes_in_grid": sizes,
                    "size_span_px": [round(side_px(min(sizes)), 0),
                                     round(side_px(max(sizes)), 0)] if sizes else None,
                    # 大小几乎不变 = 固定物体的签名。134109 是 0.0038-0.0057，17 帧。
                    # 这是**线索不是判据**，仍然要看帧。
                    # 至少 3 帧才判：单个观测恒为「扁平」，第一版就在 170216 上
                    # 这么误报过（格子里只有一帧有脸，465-465）。
                    "size_is_flat": len(claimed) >= 3 and bool(sizes)
                                    and (max(sizes) / min(sizes) < 1.6),
                    "visually_checked": False}

    print(f"{'录像':24s} {'原口径':>7s} {'过滤后':>7s} {'脸大小(px)':>14s} {'大小扁平':>8s} {'other_hand':>11s}")
    print('-' * 86)
    for rec in sorted(out):
        o = out[rec]
        g = o['grid'] or {}
        sp = o['size_span_px']
        print(f"{rec:24s} {g.get('raw_pct', 0):6.0f}% {g.get('filtered_pct', 0):6.0f}% "
              f"{(f'{sp[0]:.0f}-{sp[1]:.0f}' if sp else '-'):>14s} "
              f"{('是 ←可疑' if o['size_is_flat'] else '否'):>8s} "
              f"{(o['other_hand'] if o['other_hand'] is not None else '待标'):>11}")

    if a.frames_dir and a.dump_dir:
        os.makedirs(a.dump_dir, exist_ok=True)
        import shutil
        n = 0
        for rec, o in out.items():
            src = os.path.join(a.frames_dir, rec)
            if not os.path.isdir(src):
                print(f"  [抠帧] {rec}: 没有 {src}，跳过")
                continue
            by_frame = {}
            for fn in os.listdir(src):
                digits = ''.join(c for c in os.path.splitext(fn)[0] if c.isdigit())
                if digits:
                    by_frame.setdefault(int(digits.lstrip('0') or '0'), fn)
            for f in o['claimed_face_frames_in_grid'][:a.dump_per_rec]:
                fn = by_frame.get(f)
                if not fn:
                    continue
                dst = os.path.join(a.dump_dir, f"{rec}_f{f}{os.path.splitext(fn)[1]}")
                shutil.copy(os.path.join(src, fn), dst)
                n += 1
        print(f"\n抠出 {n} 张到 {a.dump_dir} —— **去看**，然后把 visually_checked 改成 true。")
    else:
        print("\n没给 --frames-dir/--dump-dir：只做了大小过滤。"
              "\n在把任何一条标成已核实之前，必须看帧 —— 大小扁平只是线索。")

    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    json.dump({"threshold_biggest_face": THR, "unit": "面积占比(w*h)/(W*H)",
               "frozen": "2026-10-15", "recordings": out},
              open(a.out, 'w'), ensure_ascii=False, indent=1)
    print(f"写出 {a.out}")


if __name__ == '__main__':
    main()
