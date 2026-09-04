"""Is the context window even able to tell two nearby hands apart?

THE QUESTION THIS ANSWERS WITHOUT TRAINING ANYTHING. The context branch is
handed a window round a hand and asked whose the hand is. Nothing in that
window says WHICH hand is being asked about: the crop is centred on it, and
the drawn box that used to mark it was removed because it also drew the
geometry features into the pixels. For one hand alone in its window that is
harmless. For two hands overlapping it is a structural failure -- the same
picture is presented twice with opposite correct answers, and no parameter
setting can make one network give two answers to one input.

TWO MEASUREMENTS, BOTH CHEAP. How many hands are actually inside each
window, grouped by whether the models got that track right; and, for pairs of
hands in the SAME FRAME with OPPOSITE ownership, how similar the context
encoder's embeddings of them are. A high cosine on an opposite-label pair is
the ambiguity made numeric: the branch cannot distinguish inputs it must
answer differently about.

WHAT IT CANNOT SETTLE. A high cosine is consistent with two mechanisms -- the
window not identifying its subject, and the window being downsampled until the
forearms are unreadable. Both produce embeddings that collapse together. The
2x2 over {target marker} x {context resolution} is what separates them; this
only says whether the collapse is happening at all, and on which tracks.

FRAME COVERAGE IS PARTIAL AND THAT IS REPORTED. Crops are stored every third
frame of each track, and tracks start on different frames, so two tracks
coincide on a stored frame about a third of the time. Counts below are over
the frames that do coincide, not over all frames.
"""
from __future__ import annotations

import argparse
import os

import numpy as np


def window(box, scale):
    """-> (x0, y0, x1, y1) in frame fractions, the context window."""
    cx, cy, bw, bh = box
    half = max(bw, bh) * scale / 2.0
    return (cx - half, cy - half, cx + half, cy + half)


def embeddings(rows, clf_ctx, batch=32):
    """-> (N, 512) context-trunk features, one per row.

    The trunk's output, not the head's: the question is whether the branch
    can tell the inputs apart, which is a property of its representation and
    not of what the head does with it."""
    import cv2
    import torch
    from src.rig import own_ctx

    model, device, _ = own_ctx.load_model(clf_ctx)
    if model is None:
        raise SystemExit(f"no checkpoint at {clf_ctx}")
    if getattr(model, "ctx", None) is None:
        raise SystemExit("this checkpoint has no context branch")
    ds = own_ctx.Pairs([], strip=False)
    out = []
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        cs = []
        for r in chunk:
            img = cv2.imread(r["_ctx"])
            hand = cv2.imread(r["_crop"])
            c = (own_ctx.context_crop(img, r["box"], own_ctx.CTX_SCALE)
                 if img is not None else None)
            if c is None or c.size == 0:
                c = hand
            cs.append(ds._prep(c, own_ctx.CTX_PX))
        with torch.no_grad():
            e = model.ctx(torch.stack(cs).to(device))
        out.append(e.cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkg", action="append", required=True)
    ap.add_argument("--clf", required=True)
    ap.add_argument("--clf_ctx", required=True)
    ap.add_argument("--ctx_scale", type=float, default=2.5)
    a = ap.parse_args()

    from src.rig.track_pool import load_pkg, frame_scores
    from src.rig.ctx_probe import census

    rows = []
    for p in a.pkg:
        rows += load_pkg(p)[1]
    if not rows:
        raise SystemExit("no labelled tracks with crops")
    for r in rows:
        r["box"] = tuple(float(r["raw"][c]) for c in
                         ("box_cx", "box_cy", "box_w", "box_h"))
        r["frame"] = int(r["raw"]["frame"])
    print(f"  {len(rows)} 帧样本 over "
          f"{len({(r['tag'], r['tid']) for r in rows})} 条已标 track")

    hand = frame_scores(rows, clf=a.clf)
    ctx = frame_scores(rows, clf_ctx=a.clf_ctx)
    q = census(rows, hand, ctx)
    where = {}
    for name, ks in q.items():
        for k in ks:
            where[k] = name
    print(f"  四象限: " + "  ".join(f"{k} {len(v)}" for k, v in q.items()))

    # --- how many hands sit inside each window ---
    by_frame = {}
    for i, r in enumerate(rows):
        by_frame.setdefault((r["tag"], r["frame"]), []).append(i)
    n_in, pairs = [], []
    for i, r in enumerate(rows):
        w = window(r["box"], a.ctx_scale)
        same = by_frame[(r["tag"], r["frame"])]
        cnt = 0
        for j in same:
            o = rows[j]
            if (w[0] <= o["box"][0] <= w[2]) and (w[1] <= o["box"][1] <= w[3]):
                cnt += 1
        n_in.append(cnt)
        if len(same) > 1:
            for j in same:
                if j <= i:
                    continue
                pairs.append((i, j))
    n_in = np.array(n_in)
    coincide = sum(1 for v in by_frame.values() if len(v) > 1)
    print(f"\n  同一存帧上出现 >1 只手的帧 {coincide}/{len(by_frame)} "
          f"({coincide / len(by_frame):.1%})  —— 抽样是每条 track 每 3 帧，"
          f"相位不同，所以这是下界")

    print(f"\n=== ③a 窗口里有几只手（按象限）===")
    print(f"  {'象限':<18} {'track':>6} {'窗口内手数中位':>14} {'>=2 只的帧占比':>14}")
    for name in ("both_right", "context_rescues", "context_hurts",
                 "both_wrong"):
        ks = q[name]
        if not ks:
            print(f"  {name:<18} {0:>6}")
            continue
        idx = [i for i, r in enumerate(rows)
               if (r["tag"], r["tid"]) in set(ks)]
        v = n_in[idx]
        print(f"  {name:<18} {len(ks):>6} {np.median(v):>14.1f} "
              f"{np.mean(v >= 2):>14.1%}")

    # --- same-frame, opposite-ownership pairs ---
    E = embeddings(rows, a.clf_ctx)
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    opp = [(i, j) for i, j in pairs if rows[i]["y"] != rows[j]["y"]]
    same = [(i, j) for i, j in pairs if rows[i]["y"] == rows[j]["y"]]
    print(f"\n=== ③b 同帧、归属相反的一对，context 嵌入有多像 ===")
    print(f"  同帧成对 {len(pairs)}   其中归属相反 {len(opp)}   相同 {len(same)}")

    def wov(i, j):
        """Overlap of the two context WINDOWS, not the distance between the
        two hands.

        Centre distance was the first cut and it measured the wrong thing:
        the pairs it called `opposite ownership` were mostly hands at
        opposite ends of the frame, whose windows do not intersect at all, so
        a low cosine between them says they look different rather than that
        the encoder can tell them apart. Ambiguity only exists where the two
        windows SHOW THE SAME THING, and that is an overlap question. It also
        scales correctly: a near hand has a large window, so two near hands
        can overlap heavily while their centres sit far apart in frame
        fractions."""
        a1 = window(rows[i]["box"], a.ctx_scale)
        b1 = window(rows[j]["box"], a.ctx_scale)
        ix = max(0.0, min(a1[2], b1[2]) - max(a1[0], b1[0]))
        iy = max(0.0, min(a1[3], b1[3]) - max(a1[1], b1[1]))
        inter = ix * iy
        if inter <= 0:
            return 0.0
        ua = (a1[2] - a1[0]) * (a1[3] - a1[1])
        ub = (b1[2] - b1[0]) * (b1[3] - b1[1])
        return inter / (ua + ub - inter)

    def show(v, name):
        if not v:
            print(f"  {name:<28} 无")
            return
        c = np.array([float(E[i] @ E[j]) for i, j in v])
        d = np.array([wov(i, j) for i, j in v])
        print(f"  {name:<28} n={len(v):<5} cosine 中位 {np.median(c):.3f}  "
              f"p90 {np.percentile(c, 90):.3f}  >0.9 的比例 "
              f"{np.mean(c > 0.9):.1%}   窗口重叠中位 {np.median(d):.3f}")
        return c, d

    show(opp, "归属相反")
    show(same, "归属相同")
    for t in (0.10, 0.30, 0.50):
        show([(i, j) for i, j in opp if wov(i, j) >= t],
             f"归属相反且窗口重叠>={t:.2f}")
    for t in (0.10, 0.30, 0.50):
        show([(i, j) for i, j in same if wov(i, j) >= t],
             f"归属相同且窗口重叠>={t:.2f}")

    wrong_k = set(q["both_wrong"] + q["context_hurts"])
    bad = [(i, j) for i, j in opp
           if (rows[i]["tag"], rows[i]["tid"]) in wrong_k
           or (rows[j]["tag"], rows[j]["tid"]) in wrong_k]
    show(bad, "归属相反且含判错的 track")

    print("\n  cosine 高说明 context 编码器把两个必须给出相反答案的输入编成了"
          "几乎同一个向量。\n  它和两种机制都相容——窗口没标出问的是哪只手，"
          "或者压到 128px 之后前臂读不出来。\n  分开它们要靠 "
          "{有无 target 通道} x {128 或 256} 的 2x2，这里只说塌缩有没有发生。")


if __name__ == "__main__":
    main()
