"""Filter attribution audit (P0 deliverable #2) -- pure CV, no Qwen.

Question: how many frames NEAR a GT boundary does each P0 sub-filter delete?
The hypothesis is that blur filtering (th_blur) systematically removes the
motion-blurred frames that occur exactly at task transitions, capping the
boundary head. This audit quantifies it WITHOUT running the ViT: it re-decodes
candidate frames, scores black/blur/static, and for each policy measures whether
a surviving frame lands near each GT boundary.

Policies:
    P0 = black + blur + static      P1 = black + static (drop blur)
    P2 = black only                 P3 = no filtering

Reports per policy: boundary survival recall @0.2/0.5/1.0s, mean & p95
nearest-kept-frame distance; plus P(blur|near-boundary) vs P(blur|non-boundary).

Usage (server):
    python -m src.boundary.filter_audit --data /workspace/tr1/data_recseg/recseg_val.json
"""
import argparse, json
import numpy as np


def gray(f):
    return f.astype(np.float32).mean(axis=2)


def lap_var(g):
    l = (-4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1]
         + g[1:-1, :-2] + g[1:-1, 2:])
    return float(l.var())


def gt_boundaries(sol):
    ts = set()
    for s in sol:
        ts.add(round(float(s[1]), 2)); ts.add(round(float(s[2]), 2))
    b = sorted(ts)
    return b[1:-1] if len(b) > 2 else b


POLICIES = {
    "P0_black_blur_static": lambda blk, blr, stat: blk or blr or stat,
    "P1_black_static":      lambda blk, blr, stat: blk or stat,
    "P2_black_only":        lambda blk, blr, stat: blk,
    "P3_none":              lambda blk, blr, stat: False,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--th_black", type=float, default=20.0)
    ap.add_argument("--th_blur", type=float, default=100.0)
    ap.add_argument("--th_static", type=float, default=2.0)
    ap.add_argument("--near_s", type=float, default=0.5, help="near-boundary window")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    from decord import VideoReader
    rows = json.load(open(a.data))
    if a.limit:
        rows = rows[:a.limit]

    surv = {p: {0.2: [], 0.5: [], 1.0: []} for p in POLICIES}
    dist = {p: [] for p in POLICIES}
    blur_near = blur_far = n_near = n_far = 0

    for r in rows:
        vr = VideoReader(r["video"]); n = len(vr); vfps = vr.get_avg_fps()
        step = max(1, int(round(vfps / a.fps)))
        idxs = list(range(0, n, step))
        frames = vr.get_batch(idxs).asnumpy()
        times = np.array([i / vfps for i in idxs])
        gts = np.array(gt_boundaries(r["solution"]))
        if len(gts) == 0:
            continue

        prev_g = None
        black = np.zeros(len(idxs), bool); blur = np.zeros(len(idxs), bool)
        stat = np.zeros(len(idxs), bool)
        for j, f in enumerate(frames):
            g = gray(f)
            black[j] = float(g.mean()) < a.th_black
            blur[j] = lap_var(g) < a.th_blur
            if prev_g is not None:
                stat[j] = float(np.abs(g - prev_g).mean()) < a.th_static
            prev_g = g

        # blur-vs-boundary conditional
        near_b = np.min(np.abs(times[:, None] - gts[None, :]), axis=1) <= a.near_s
        blur_near += int(blur[near_b].sum()); n_near += int(near_b.sum())
        blur_far += int(blur[~near_b].sum()); n_far += int((~near_b).sum())

        for p, rule in POLICIES.items():
            drop = np.array([rule(black[j], blur[j], stat[j]) for j in range(len(idxs))])
            kept_t = times[~drop]
            if len(kept_t) == 0:
                kept_t = np.array([times[0]])
            nd = np.min(np.abs(gts[:, None] - kept_t[None, :]), axis=1)  # per GT boundary
            dist[p].extend(nd.tolist())
            for tol in (0.2, 0.5, 1.0):
                surv[p][tol].append(float((nd <= tol).mean()))

    print("==== FILTER ATTRIBUTION AUDIT (val, pure CV) ====")
    print(f"{'policy':22s} {'surv@.2':>8s} {'surv@.5':>8s} {'surv@1':>8s} "
          f"{'meanD':>7s} {'p95D':>7s}")
    for p in POLICIES:
        s = surv[p]
        print(f"{p:22s} {np.mean(s[0.2]):8.3f} {np.mean(s[0.5]):8.3f} "
              f"{np.mean(s[1.0]):8.3f} {np.mean(dist[p]):7.2f} "
              f"{np.percentile(dist[p],95):7.2f}")
    print(f"\nP(blur | near-boundary ±{a.near_s}s) = {blur_near/max(n_near,1):.3f}")
    print(f"P(blur | non-boundary)          = {blur_far/max(n_far,1):.3f}")
    print("-> if near >> far AND P1 survival >> P0, blur filter is capping boundaries")


if __name__ == "__main__":
    main()
