"""Model-free frame analysis for the adaptive preprocessing pipeline.

Pure-CV (numpy only) per-frame stats -> classify each sampled frame as
black / blur / static / normal, and mark motion-peak KEYFRAMES. Drives:
  (a) invalid/static filtering  -> throughput (drop useless frames),
  (b) keyframe high-res budget   -> naming/visual focus (pixels where it matters).

This is the CALIBRATION pass: it prints the distributions of intensity / blur /
motion across the corpus so thresholds are set FROM DATA, not guessed. Default
thresholds give a first-pass label breakdown + per-video keyframe counts.

Usage (server, decord available):
    python -m src.preprocess.analyze_frames \
        --data /workspace/tr1/data_handtask/train_multiseg_val.json --base_fps 2
"""
import argparse, json, os, statistics
import numpy as np


def gray(f):                      # f: H,W,3 uint8 -> H,W float32
    return f.astype(np.float32).mean(axis=2)


def lap_var(g):                   # variance of discrete Laplacian; low => blurry
    l = (-4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1]
         + g[1:-1, :-2] + g[1:-1, 2:])
    return float(l.var())


def pct(xs, ps=(1, 5, 25, 50, 75, 95, 99)):
    a = np.array(xs)
    return {p: round(float(np.percentile(a, p)), 1) for p in ps}


def analyze_video(path, base_fps):
    from decord import VideoReader
    vr = VideoReader(path)
    n = len(vr); vfps = vr.get_avg_fps()
    step = max(1, int(round(vfps / base_fps)))
    idxs = list(range(0, n, step))
    frames = vr.get_batch(idxs).asnumpy()          # T,H,W,3
    grays = [gray(f) for f in frames]
    inten = [float(g.mean()) for g in grays]
    blur = [lap_var(g) for g in grays]
    motion = [0.0] + [float(np.abs(grays[i] - grays[i - 1]).mean())
                      for i in range(1, len(grays))]
    ts = [idxs[i] / vfps for i in range(len(idxs))]
    return {"inten": inten, "blur": blur, "motion": motion, "ts": ts,
            "hw": frames.shape[1:3]}


def classify(v, th_black, th_blur, th_static, th_key):
    labels = []
    for i in range(len(v["inten"])):
        if v["inten"][i] < th_black:
            labels.append("black")
        elif v["blur"][i] < th_blur:
            labels.append("blur")
        elif i > 0 and v["motion"][i] < th_static:
            labels.append("static")
        else:
            labels.append("normal")
    # keyframes: motion local maxima above th_key among non-invalid frames
    m = v["motion"]; keys = []
    for i in range(len(m)):
        if labels[i] in ("black", "blur"):
            continue
        if (m[i] >= th_key
                and (i == 0 or m[i] >= m[i - 1])
                and (i == len(m) - 1 or m[i] >= m[i + 1])):
            keys.append(i)
    return labels, keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--base_fps", type=float, default=2.0)
    ap.add_argument("--th_black", type=float, default=20.0)   # mean intensity 0-255
    ap.add_argument("--th_blur", type=float, default=100.0)   # laplacian variance
    ap.add_argument("--th_static", type=float, default=2.0)   # inter-frame MAD
    ap.add_argument("--th_key", type=float, default=8.0)      # motion peak threshold
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    rows = json.load(open(a.data))
    if a.limit:
        rows = rows[:a.limit]
    all_i, all_b, all_m = [], [], []
    label_tot = {}; nkeys = []; nframes = []
    for r in rows:
        v = analyze_video(r["video"], a.base_fps)
        all_i += v["inten"]; all_b += v["blur"]; all_m += v["motion"][1:]
        labels, keys = classify(v, a.th_black, a.th_blur, a.th_static, a.th_key)
        for l in labels:
            label_tot[l] = label_tot.get(l, 0) + 1
        nkeys.append(len(keys)); nframes.append(len(labels))
        print(f"{os.path.basename(r['video']):24s} "
              f"frames {len(labels):3d} keys {len(keys):2d} "
              f"hw {v['hw']} "
              f"{ {l: labels.count(l) for l in set(labels)} }")

    N = sum(nframes)
    print(f"\n==== DISTRIBUTIONS (n_frames={N}, base_fps={a.base_fps}) ====")
    print("intensity pct:", pct(all_i))
    print("blur(lapvar) pct:", pct(all_b))
    print("motion pct:", pct(all_m))
    print("\n==== FIRST-PASS LABELS (default thresholds) ====")
    for l, c in sorted(label_tot.items(), key=lambda x: -x[1]):
        print(f"  {l:8s} {c:5d}  {c/max(N,1):.1%}")
    print(f"\nkeyframes/video: mean {statistics.mean(nkeys):.1f} "
          f"(of {statistics.mean(nframes):.1f} frames/video)")
    print("-> drop black+blur+static for throughput; give keyframes high-res.")


if __name__ == "__main__":
    main()
