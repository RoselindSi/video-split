"""按人脸预筛录像，找「别人的手可能出现」的那些 —— 不用手部流水线的任何输出。

为什么只能用人脸。危险负例的筛选规则一旦用上手检测、轨迹或 ownership 分数，
筛出来的样本就不独立于被测对象。这个项目为同类做法付过代价（按切换行为选片的
proxy 与真值相关 r = -0.46）。人脸模型与手部流水线无关，是现成且独立的信号。

曾经提过的另外三个信号都被否掉了：
  手部空间接近 / 时间共存  —— 需要手检测，是被测对象本身
  轨迹碎片                —— **直接选在 tracker 的失败上**，而碎片化正是召回的来源
人工分类标签（label.json 的 environment / skill_group）在 25 条上测过：
最好的格子只有 1-2 条录像，n 太小，给不出可用规则。

实测（25 条已标录像，真危险 = 含「别人的手」而非前臂/倒影）：

    随机基线            16%
    人脸帧数 >= 100     精度 75%，召回 75%     <- 4.7 倍富集
    零人脸却含真危险    只有 1 条

**筛出来的是富集集不是代表性样本。** 两者分开用：
    随机集  —— 测发生率（batch2 那 15 条就是）
    富集集  —— 测安全约束（危险负例太稀，随机抽永远测不准）
**不可合并**，合并后任何发生率都是被筛选规则决定的。
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys


def sample_frames(n_total, k, start=400):
    """在 start 之后均匀取 k 个帧号 —— 不是取开头，开头常是架设镜头。"""
    if n_total <= start + k:
        return list(range(start, max(start + 1, n_total)))
    step = (n_total - start) // k
    return [start + i * step for i in range(k)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="raw_databag 根目录")
    ap.add_argument("--n", type=int, default=200, help="随机抽多少条来筛")
    ap.add_argument("--frames", type=int, default=20, help="每条取多少帧验脸")
    ap.add_argument("--exclude", help="已用过的录像名，每行一个")
    ap.add_argument("--seed", type=int, default=20261014)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    used = set()
    if a.exclude and os.path.exists(a.exclude):
        used = {ln.strip() for ln in open(a.exclude) if ln.strip()}

    pool = []
    for h in sorted(os.listdir(a.root)):
        d = os.path.join(a.root, h)
        if not os.path.isdir(d):
            continue
        for b in sorted(os.listdir(d)):
            if not b.startswith("databag-26_"):
                continue
            rec = "R" + b[len("databag-"):]
            if rec in used:
                continue
            p = os.path.join(d, b)
            if os.path.exists(os.path.join(p, "cam34.mp4")):
                pool.append((rec, p))
    print("候选池 %d 条（排除已用 %d 条）" % (len(pool), len(used)))
    rng = random.Random(a.seed)
    pick = rng.sample(pool, min(a.n, len(pool)))

    # 必须用流水线自己那个人脸检测器（`src/rig/face_mask`，YOLOv8n-face ONNX），
    # 因为筛选规则是在它产出的 `n_faces` 上验证的 —— 换一个人脸模型，
    # 「有脸帧 >= T」这个阈值就和验证时的口径对不上了。
    import cv2
    sys.path.insert(0, "/workspace/tr1/vs")
    from src.rig import face_mask
    det = face_mask.load_detector()
    rows = []
    for i, (rec, bag) in enumerate(sorted(pick), 1):
        v = os.path.join(bag, "cam34.mp4")
        cap = cv2.VideoCapture(v)
        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if n_total < 800:
            cap.release()
            rows.append({"rec": rec, "bag": bag, "too_short": True,
                         "n_frames": n_total, "face_frames": 0, "faces": 0})
            continue
        want = sample_frames(n_total, a.frames)
        ff = nf = read_ok = 0
        for f in want:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, img = cap.read()
            if not ok:
                continue          # 共享存储读超时会走到这里
            read_ok += 1
            img = img[:, :img.shape[1] // 2]          # cam3 = 左半
            k = len(face_mask.detect_faces(det, cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) or [])
            nf += k
            ff += k > 0
        cap.release()
        # `read_ok` 必须单独记：`len(want)` 是**想取**多少帧，读失败的帧会被
        # 静默跳过。只记前者的话，读超时的录像看起来就是「脸少」而不是「没读到」，
        # 于是分母悄悄变小 —— 第一版就是这样，我拿 `sampled` 去查读失败，
        # 查了个恒等于 20 的数，等于没查。
        rows.append({"rec": rec, "bag": bag, "too_short": False,
                     "n_frames": n_total, "face_frames": ff, "faces": nf,
                     "wanted": len(want), "read_ok": read_ok})
        if i % 20 == 0:
            print("  %d / %d" % (i, len(pick)))
            sys.stdout.flush()

    ok = [r for r in rows if not r["too_short"]]
    ok.sort(key=lambda r: -r["face_frames"])
    json.dump(rows, open(a.out, "w", encoding="utf-8"), ensure_ascii=False,
              indent=1)
    print("\n  可用 %d 条（片长 >= 800 帧），太短排除 %d 条"
          % (len(ok), len(rows) - len(ok)))
    print("  有脸帧数分布：")
    for q, nm in [(0, "最多"), (len(ok) // 4, "75 分位"), (len(ok) // 2, "中位"),
                  (3 * len(ok) // 4, "25 分位"), (len(ok) - 1, "最少")]:
        if 0 <= q < len(ok):
            print("    %-7s %d / %d 帧" % (nm, ok[q]["face_frames"],
                                            ok[q]["read_ok"]))
    bad = [r for r in ok if r["read_ok"] < r["wanted"]]
    if bad:
        print("  ！%d 条有帧读失败（共享存储超时），它们的有脸占比偏低、会被少选"
              % len(bad))
    # 阈值用**占比**不是绝对帧数：验证时是 800 帧里 100 帧有脸 = 12.5%，
    # 这里只取 20 帧，等价阈值是 2-3 帧。换算错会把入选数差一个量级。
    hit = [r for r in ok if r["face_frames"] / max(1, r["read_ok"]) >= 0.125]
    print("  有脸占比 >= 12.5%%（与验证口径一致）：%d 条 —— 富集集候选" % len(hit))
    print("\n-> %s" % a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
