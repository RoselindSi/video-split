"""Which input is carrying the verdict, on the tracks context saves and loses.

CONTEXT IS NOT UNIFORMLY BETTER AND THE AVERAGE HIDES THAT. On thirty-four
labelled tracks the context model beats the hand-only one by a wide margin,
but the wins and the losses are different phenomena and pooling them into one
f1 conceals both. One track is `other` and the hand-only model answers
P(owner)=1.00 on every frame while context answers 0.00-0.43 throughout: that
is context supplying evidence the crop does not contain. Another is `other`
and CONTEXT answers 0.84, 0.96, 0.99, 1.00 across most of its frames: that is
context confidently wrong, held steady, and any temporal aggregation would
make it more stable rather than less.

SO THE FIRST OUTPUT IS A FOUR-WAY CENSUS BY TRACK, not a score. Both right,
both wrong, context rescues, context hurts. Averaging over frames would let a
long track drown three short ones, and the question here is about kinds of
failure rather than about how many frames each kind covers.

THE SECOND OUTPUT ASKS WHAT THE MODEL IS EATING. Two ways, because they answer
different halves. The geometry features are fourteen numbers and can simply be
printed and compared between the rescue group and the hurt group -- if a cue
points one way in one group and the other way in the other, that cue is the
mechanism. The two image branches cannot be read that way, so they are probed
by ablation: the forward pass is repeated with one input replaced by a neutral
value, and how far the score moves is how much that input was carrying.

ABLATION IS ATTRIBUTION, NOT CAUSATION. Replacing an input with a neutral
value puts the network off its training distribution, so a large swing means
`this input mattered here` and not `this input would be enough alone`. The
neutral geometry vector is the MEAN OF THE ROWS BEING PROBED, which is a
property of this sample rather than of the training set -- stated because it
makes the ablation relative to these tracks, not to what the model learned.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from src.rig.track_pool import load_pkg


def _grouped(rows, scores):
    """-> {(tag, tid): {'y', 'idx', 'p'}}"""
    g = {}
    for i, (r, p) in enumerate(zip(rows, scores)):
        k = (r["tag"], r["tid"])
        d = g.setdefault(k, {"y": r["y"], "idx": [], "p": []})
        d["idx"].append(i)
        d["p"].append(float(p))
    return g


def census(rows, hand, ctx):
    """Four ways a pair of models can land on one track. -> dict of lists"""
    gh, gc = _grouped(rows, hand), _grouped(rows, ctx)
    out = {"both_right": [], "context_rescues": [],
           "context_hurts": [], "both_wrong": []}
    for k in sorted(gh):
        y = gh[k]["y"]
        # A track's verdict is its median frame: one outlier frame should not
        # decide which quadrant a track belongs in.
        h_ok = (float(np.median(gh[k]["p"])) >= 0.5) == y
        c_ok = (float(np.median(gc[k]["p"])) >= 0.5) == y
        key = ("both_right" if h_ok and c_ok else
               "context_hurts" if h_ok and not c_ok else
               "context_rescues" if c_ok else "both_wrong")
        out[key].append(k)
    return out


def probe(rows, clf_ctx, batch=32):
    """-> dict of arrays: full score and one per ablated input.

    `hand_off`, `ctx_off` and `geom_off` are the same forward pass with that
    one input replaced. The distance from `full` is what that input was
    worth on that row."""
    import cv2
    import torch
    from src.rig import own_ctx

    model, device, arm = own_ctx.load_model(clf_ctx)
    if model is None:
        raise SystemExit(f"no checkpoint at {clf_ctx}")
    ds = own_ctx.Pairs([], strip=False)

    G = np.array([[float(r["raw"][k]) for k in own_ctx.GEOM] for r in rows],
                 np.float32)
    gmean = torch.from_numpy(G.mean(0))
    grey = np.full((64, 64, 3), 128, np.uint8)
    neutral_h = ds._prep(grey, own_ctx.HAND_PX)
    neutral_c = ds._prep(grey, own_ctx.CTX_PX)

    keys = ("full", "hand_off", "ctx_off", "geom_off")
    got = {k: [] for k in keys}
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        hs, cs, gs = [], [], []
        for r in chunk:
            hand = cv2.imread(r["_crop"])
            ci = cv2.imread(r["_ctx"])
            box = tuple(float(r["raw"][c]) for c in
                        ("box_cx", "box_cy", "box_w", "box_h"))
            c = (own_ctx.context_crop(ci, box, own_ctx.CTX_SCALE)
                 if ci is not None else None)
            if c is None or c.size == 0:
                c = hand
            hs.append(ds._prep(hand, own_ctx.HAND_PX))
            cs.append(ds._prep(c, own_ctx.CTX_PX))
            gs.append(torch.tensor(
                [float(r["raw"][k]) for k in own_ctx.GEOM],
                dtype=torch.float32))
        H = torch.stack(hs).to(device)
        C = torch.stack(cs).to(device)
        Gt = torch.stack(gs).to(device)
        nh = neutral_h.unsqueeze(0).repeat(len(chunk), 1, 1, 1).to(device)
        nc = neutral_c.unsqueeze(0).repeat(len(chunk), 1, 1, 1).to(device)
        ng = gmean.unsqueeze(0).repeat(len(chunk), 1).to(device)
        with torch.no_grad():
            for name, args in (("full", (H, C, Gt)),
                               ("hand_off", (nh, C, Gt)),
                               ("ctx_off", (H, nc, Gt)),
                               ("geom_off", (H, C, ng))):
                p = torch.softmax(model(*args), 1)[:, 1]
                got[name].append(p.cpu().numpy())
    return {k: np.concatenate(v) for k, v in got.items()}, arm


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkg", action="append", required=True)
    ap.add_argument("--clf", required=True)
    ap.add_argument("--clf_ctx", required=True)
    ap.add_argument("--dump", help="write the per-track table here as csv")
    a = ap.parse_args()

    from src.rig.track_pool import frame_scores
    from src.rig import own_ctx

    rows = []
    for p in a.pkg:
        rows += load_pkg(p)[1]
    if not rows:
        raise SystemExit("no labelled tracks with crops")
    print(f"  {len(rows)} crops over "
          f"{len({(r['tag'], r['tid']) for r in rows})} labelled tracks")

    hand = frame_scores(rows, clf=a.clf)
    ab, arm = probe(rows, a.clf_ctx)
    ctx = ab["full"]

    q = census(rows, hand, ctx)
    print(f"\n=== 四象限（按 track，逐帧中位数定判决）===")
    print(f"  {'':<16} {'ctx 对':>10} {'ctx 错':>10}")
    print(f"  {'hand 对':<16} {len(q['both_right']):>10} "
          f"{len(q['context_hurts']):>10}   <- context hurts")
    print(f"  {'hand 错':<16} {len(q['context_rescues']):>10} "
          f"{len(q['both_wrong']):>10}")
    for k in ("context_rescues", "context_hurts", "both_wrong"):
        if q[k]:
            print(f"    {k}: " + ", ".join(f"{t} #{i}" for t, i in q[k]))

    gh, gc = _grouped(rows, hand), _grouped(rows, ctx)
    gab = {name: _grouped(rows, ab[name]) for name in ab}

    print(f"\n=== 输入消融（每条 track 的均值，值是 P(owner)）===")
    print(f"  {'track':<22} {'真值':>5} {'hand':>6} {'full':>6} "
          f"{'-hand':>7} {'-ctx':>7} {'-geom':>7}")
    order = (q["context_rescues"] + q["context_hurts"]
             + q["both_wrong"] + q["both_right"])
    tag_of = {}
    for k in order:
        for lab in ("context_rescues", "context_hurts", "both_wrong",
                    "both_right"):
            if k in q[lab]:
                tag_of[k] = lab
                break
    for k in order:
        y = "owner" if gh[k]["y"] == 1 else "other"
        print(f"  {k[0] + ' #' + str(k[1]):<22} {y:>5} "
              f"{np.mean(gh[k]['p']):>6.2f} "
              f"{np.mean(gab['full'][k]['p']):>6.2f} "
              f"{np.mean(gab['hand_off'][k]['p']):>7.2f} "
              f"{np.mean(gab['ctx_off'][k]['p']):>7.2f} "
              f"{np.mean(gab['geom_off'][k]['p']):>7.2f}"
              + ("   <- " + tag_of[k] if tag_of[k] != "both_right" else ""))
    print("\n  `-X` 是把那一路输入换成中性值后的分数。离 `full` 越远，说明这一"
          "路在这条\n  track 上承担得越多。中性几何向量取的是被探测这批行的均值，"
          "所以它是相对\n  这批 track 的归因，不是相对训练集的。")

    print(f"\n=== 几何特征：rescue 组 对 hurts 组（每条 track 先取均值）===")
    G = {}
    for k in gh:
        idx = gh[k]["idx"]
        G[k] = np.array([[float(rows[i]["raw"][c]) for c in own_ctx.GEOM]
                         for i in idx], np.float32).mean(0)
    res, hur = q["context_rescues"], q["context_hurts"]
    if res and hur:
        print(f"  {'cue':<14} {'rescue n=' + str(len(res)):>14} "
              f"{'hurts n=' + str(len(hur)):>14} {'差':>8}")
        for j, c in enumerate(own_ctx.GEOM):
            r = float(np.mean([G[k][j] for k in res]))
            h = float(np.mean([G[k][j] for k in hur]))
            star = "  <-" if abs(r - h) > 0.15 else ""
            print(f"  {c:<14} {r:>14.3f} {h:>14.3f} {r - h:>8.3f}{star}")
        print("\n  两组样本都很小，这张表是找方向不是做检验。一个 cue 在两组里"
              "符号相反，\n  才值得追；差值小的不要读。")
    else:
        print("  一组为空，无法比较")

    if a.dump:
        import csv as _csv
        with open(a.dump, "w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["recording", "tid", "label", "quadrant", "n_frames",
                        "hand_mean", "ctx_full", "ctx_hand_off",
                        "ctx_ctx_off", "ctx_geom_off"] + list(own_ctx.GEOM))
            for k in order:
                w.writerow([k[0], k[1],
                            "owner" if gh[k]["y"] == 1 else "other",
                            tag_of[k], len(gh[k]["idx"]),
                            round(float(np.mean(gh[k]["p"])), 4),
                            round(float(np.mean(gab["full"][k]["p"])), 4),
                            round(float(np.mean(gab["hand_off"][k]["p"])), 4),
                            round(float(np.mean(gab["ctx_off"][k]["p"])), 4),
                            round(float(np.mean(gab["geom_off"][k]["p"])), 4)]
                           + [round(float(x), 4) for x in G[k]])
        print(f"\n  per-track table -> {a.dump}")


if __name__ == "__main__":
    main()
