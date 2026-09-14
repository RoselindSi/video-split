"""How much does the wearer zone move, differ between eyes, and differ between tries?

THREE NUMBERS, ONE UNIT. Everything is the fraction of the frame whose
own-side label differs between two masks, because that is what a blur mask
gets wrong: temporal drift is a frame against the recording's best single
static zone, stereo is cam3 against cam4 on the same grid, and repeatability
is one annotation against another of the same clip. Stated in the same unit,
the question `is this difference real` becomes a comparison.

REPEATABILITY IS THE YARDSTICK FOR THE OTHER TWO. A cam3-cam4 difference of
five percent means nothing until you know two tries by the same person differ
by twelve; the `_retest` clips exist to supply that number, and until it is
known neither the stereo nor the drift figure licenses a decision.

MASKS ARE REBUILT AT EVERY FRAME, NOT READ AT KEYFRAMES. The export carries a
polygon only where a keyframe was set; between them the page showed an
interpolated curve closed against the border. This ports that closure, infers
the side flag (it is not exported) by matching the exported polygon, and
reports how far the rebuild is from the export -- the floor below which a
difference is measurement, not annotation.

ONE KEYFRAME IS NOT EVIDENCE OF A STILL ZONE. A clip with a single keyframe
scores zero drift by construction, whether the zone held or nobody checked, so
those clips are counted apart rather than averaged in.
"""
import argparse, glob, json, os, re, sys
import numpy as np
from PIL import Image, ImageDraw

ANN = PAGES = None
W, H = 1920, 1520
S = 4                                   # raster at 1/4: 480 x 380
P = 2 * (W + H)


def snap(p, d):
    ts = []
    if abs(d[0]) > 1e-9: ts += [(0 - p[0]) / d[0], (W - p[0]) / d[0]]
    if abs(d[1]) > 1e-9: ts += [(0 - p[1]) / d[1], (H - p[1]) / d[1]]
    good = [(p[0] + d[0] * t, p[1] + d[1] * t) for t in ts if t > 0]
    good = [q for q in good if -1 <= q[0] <= W + 1 and -1 <= q[1] <= H + 1]
    if not good:
        return (min(max(p[0], 0), W), min(max(p[1], 0), H))
    return min(good, key=lambda q: (q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2)


def perimT(p):
    e = 1e-6
    cl = lambda x, m: min(max(x, 0), m)
    if p[1] <= e: return cl(p[0], W)
    if p[0] >= W - e: return W + cl(p[1], H)
    if p[1] >= H - e: return W + H + (W - cl(p[0], W))
    return 2 * W + H + (H - cl(p[1], H))


CORNERS = [((W, 0), W), ((W, H), W + H), ((0, H), 2 * W + H), ((0, 0), 2 * W + 2 * H)]


def corners_between(tA, tB, d):
    span = ((tB - tA) % P) if d > 0 else ((tA - tB) % P)
    out = []
    for pt, tc in CORNERS:
        dd = ((tc - tA) % P) if d > 0 else ((tA - tc) % P)
        if 1e-9 < dd < span - 1e-9:
            out.append((pt, dd))
    return [o[0] for o in sorted(out, key=lambda o: o[1])]


def own_poly(c, flip):
    a = snap(c[0], (c[0][0] - c[1][0], c[0][1] - c[1][1]))
    b = snap(c[-1], (c[-1][0] - c[-2][0], c[-1][1] - c[-2][1]))
    mid = [a] + [tuple(p) for p in c[1:-1]] + [b]
    return mid + corners_between(perimT(b), perimT(a), -1 if flip else 1)


def raster(poly):
    im = Image.new("L", (W // S, H // S), 0)
    ImageDraw.Draw(im).polygon([(x / S, y / S) for x, y in poly], fill=1)
    return np.asarray(im, bool)


def area(poly):
    x = np.array([p[0] for p in poly]); y = np.array([p[1] for p in poly])
    return abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2


def infer_flip(k):
    """flip is not exported; pick the side whose polygon matches the export."""
    want = raster(k["own_polygon"])
    d0 = (raster(own_poly(k["curve"], 0)) ^ want).mean()
    d1 = (raster(own_poly(k["curve"], 1)) ^ want).mean()
    return (0, d0) if d0 <= d1 else (1, d1)


def meta_for(rec):
    for d in PAGES:
        p = os.path.join(d, f"zone_{rec}.html")
        if os.path.exists(p):
            s = open(p, encoding="utf-8").read(9000)
            return json.loads(re.search(r"const M = (\{.*?\});", s, re.S).group(1))
    raise FileNotFoundError(f"没有 zone_{rec}.html 在 {PAGES}")


def latest_final(rec, who):
    fs = sorted(glob.glob(f"{ANN}/*/{rec}__{who}__final_*.json"))
    return json.load(open(fs[-1], encoding="utf-8")) if fs else None


class Eye:
    def __init__(self, kfs):
        self.k = sorted(kfs, key=lambda k: k["frame"])
        self.fit = []
        for k in self.k:
            f, err = infer_flip(k)
            k["_flip"] = f
            self.fit.append(err)

    def mask(self, frame):
        ks = self.k
        lo = [k for k in ks if k["frame"] <= frame]
        hi = [k for k in ks if k["frame"] > frame]
        if not lo: A = B = hi[0]; s = 0
        elif not hi: A = B = lo[-1]; s = 0
        else:
            A, B = lo[-1], hi[0]
            s = (frame - A["frame"]) / (B["frame"] - A["frame"])
        c = [(a[0] + (b[0] - a[0]) * s, a[1] + (b[1] - a[1]) * s)
             for a, b in zip(A["curve"], B["curve"])]
        return raster(own_poly(c, A["_flip"]))


def compare(a, b, rec):
    m = meta_for(rec)
    fr = list(range(m["start"], m["start"] + int(round(m["dur"] * m["fps"])),
                    int(m["fps"])))
    out = {}
    for eye in ("cam3", "cam4"):
        ka, kb = a["eyes"][eye]["keyframes"], b["eyes"][eye]["keyframes"]
        if not ka or not kb:
            continue
        A, B = Eye(ka), Eye(kb)
        dd = np.array([(A.mask(f) ^ B.mask(f)).mean() for f in fr])
        out[eye] = (dd.mean(), dd.max(), len(A.k), len(B.k))
    return out


def main():
    global ANN, PAGES
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ann", required=True, help="annotations/ root")
    ap.add_argument("--pages", action="append", required=True,
                    help="directory of zone_*.html (repeatable)")
    ap.add_argument("--who", default="roselindsi")
    a = ap.parse_args()
    ANN, PAGES = a.ann, [os.path.expanduser(p) for p in a.pages]
    who = a.who

    recs = sorted({os.path.basename(p).split("__")[0]
                   for p in glob.glob(f"{ANN}/*/*__{who}__final_*.json")
                   if "_retest__" not in os.path.basename(p)})
    rows = []
    for rec in recs:
        d = latest_final(rec, who)
        m = meta_for(rec)
        n = int(round(m["dur"] * m["fps"]))
        start = m["start"]
        frames = list(range(start, start + n, int(m["fps"])))   # 1 Hz
        e3 = Eye(d["eyes"]["cam3"]["keyframes"])
        e4 = Eye(d["eyes"]["cam4"]["keyframes"])
        M3 = np.stack([e3.mask(f) for f in frames])
        own = M3.mean(axis=(1, 2))
        maj = M3.mean(0) >= 0.5                     # best single static zone
        dis_static = (M3 ^ maj).mean(axis=(1, 2))
        dis_first = (M3 ^ M3[0]).mean(axis=(1, 2))
        src4 = [k.get("src") for k in e4.k]
        drawn4 = any(s == "drawn" for s in src4)
        stereo = best = None
        if drawn4:
            M4 = np.stack([e4.mask(f) for f in frames])
            stereo = (M3 ^ M4).mean()
            # effective disparity: horizontal shift of cam3's zone that best
            # matches cam4's, at the raster's 4 px (full-res) step
            errs = {}
            for sh in range(-50, 51):
                A = np.roll(M3, sh, axis=2)
                errs[sh] = (A ^ M4).mean()
            sh = min(errs, key=errs.get)
            best = (sh * S, errs[sh])
        rows.append(dict(rec=rec, secs=n / m["fps"], k3=len(e3.k), k4=len(e4.k),
                         src4="/".join(sorted(set(src4))),
                         own_mean=own.mean(), own_min=own.min(), own_max=own.max(),
                         static_mean=dis_static.mean(), static_max=dis_static.max(),
                         first_max=dis_first.max(), fit=max(e3.fit + e4.fit),
                         stereo=stereo, best=best,
                         flips3={k["_flip"] for k in e3.k},
                         flips4={k["_flip"] for k in e4.k}))

    print(f"{'rec':<14}{'秒':>4}{'kf3':>4}{'kf4':>4} {'cam4来源':<13}"
          f"{'自己侧占画面':>14}  {'单一静态区域错':>14}  {'只画第一帧错':>10}  {'双目':>18}")
    for r in rows:
        st = ("—" if r["stereo"] is None else
              f"{r['stereo']:.1%} 移{r['best'][0]:+d}px→{r['best'][1]:.1%}")
        print(f"{r['rec']:<14}{r['secs']:>4.0f}{r['k3']:>4}{r['k4']:>4} {r['src4']:<13}"
              f"{r['own_mean']:>6.0%} [{r['own_min']:.0%}-{r['own_max']:.0%}]"
              f"  均{r['static_mean']:>5.1%} 最{r['static_max']:>5.1%}"
              f"  {r['first_max']:>9.1%}  {st:>18}")
    print("\nflip 反推误差最大:", f"{max(r['fit'] for r in rows):.3%}",
          " (导出多边形 vs 由曲线重建的多边形；接近 0 说明重建和页面一致)")
    print("flip 一致:", all(r["flips3"] == r["flips4"] for r in rows))

    one = [r["rec"] for r in rows if r["k3"] == 1]
    print(f"只有 1 个关键帧（漂移按构造为 0，不计入）: {len(one)} 段 {' '.join(one)}")
    multi = [r for r in rows if r["k3"] > 1]
    if multi:
        print(f"≥2 关键帧 {len(multi)} 段: 静态区域错 均 "
              f"{np.mean([r['static_mean'] for r in multi]):.1%}，逐段最坏的中位 "
              f"{np.median([r['static_max'] for r in multi]):.1%}")

    print("\n重复标注（同一片段两次）:")
    pairs = []
    for p in sorted(glob.glob(f"{ANN}/*/*_retest__{who}__final_*.json")):
        base = os.path.basename(p).split("__")[0][:-len("_retest")]
        if latest_final(base, who) is not None:
            pairs.append((base, latest_final(base, who),
                          latest_final(base + "_retest", who), "retest"))
    for p in sorted(glob.glob(f"{ANN}/*/*__anon__final_*.json")):
        base = os.path.basename(p).split("__")[0]
        if latest_final(base, who) is not None:
            pairs.append((base, json.load(open(p, encoding="utf-8")),
                          latest_final(base, who), "离线会话"))
    seen = set()
    for base, x, y, kind in pairs:
        if (base, kind) in seen:
            continue
        seen.add((base, kind))
        for eye, (mn, mx, na, nb) in compare(x, y, base).items():
            print(f"  {base:<14}{kind:<6}{eye}: 分歧 均 {mn:.1%} 最坏 {mx:.1%}  "
                  f"({na} vs {nb} 关键帧)")
    if not pairs:
        print("  （没有）")

if __name__ == "__main__":
    main()
