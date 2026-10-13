"""Draw the regime figure from the measured points, so it cannot drift from them.

The previous version of this figure was hand-written SVG with the coordinates
typed in, and it placed Ego-Exo4D at a single point. Two uniform draws from one
of its families returned hit rates of 10.5% and 33.8%, so that point was an
artefact of which draw I happened to have measured. Generating the figure from
`paper_regime_points.json` means the picture and the numbers cannot disagree,
and re-running the measurement redraws it.

WHAT THE FIGURE NOW SHOWS THAT THE HAND-DRAWN ONE DENIED. At scenario
resolution the upper-right region is not empty. First Aid - CPR sits in it at
77.3% hit and 3.16% branching, and three more scenarios clear a 50% hit rate.
The honest claim is therefore not "no data has both" but that the region is
sparsely occupied, that no occupant reaches the frozen per-window yield, and
that the tendency is moderate -- Spearman rho = -0.535 over the 16 keystep
scenarios, printed on the figure rather than described as a law.

Point area encodes decision count, because a scenario measured on 86 decisions
and one measured on 2888 should not read as equally firm.
"""
from __future__ import annotations

import argparse
import json

W, H = 820, 580
L, R, T, B = 110, 700, 110, 450          # 绘图区
XMAX, YMAX = 100.0, 8.0                   # 命中率 %, |S|>=2 %
HIT_GATE, BR_GATE = 50.0, 3.0             # 期望区间的两条线

STYLE = {
    "keystep": ("#a4453f", 0.55),
    "contract": ("#2f5d8c", 0.80),
    "human": ("#6a7a3a", 0.80),
}


def esc(s):
    """XML-escape label text. Scenario names carry raw & (Cucumber & Tomato
    Salad), which makes the file not well-formed and renders as nothing."""
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


def sx(v):
    return L + (R - L) * min(v, XMAX) / XMAX


def sy(v):
    return B - (B - T) * min(v, YMAX) / YMAX


def spearman(xs, ys):
    def rank(v):
        s = sorted(range(len(v)), key=lambda i: v[i])
        r = [0] * len(v)
        for k, i in enumerate(s):
            r[i] = k + 1
        return r
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--points", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label-min-hit", dest="lmh", type=float, default=45.0,
                    help="命中率高于此值的点标名字，其余不标以免糊成一团")
    a = ap.parse_args()

    pts = [p for p in json.load(open(a.points, encoding="utf-8"))["points"]
           if p.get("hit") is not None]
    ks = [p for p in pts if p["grain"] == "keystep"]
    rho = spearman([p["hit"] for p in ks], [p["p_S_ge_2"] for p in ks])
    nmax = max(p["n_decisions"] for p in pts)

    o = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" width="%d" '
         'height="%d" font-family="Helvetica, Arial, sans-serif">' % (W, H, W, H),
         '<rect width="%d" height="%d" fill="#ffffff"/>' % (W, H),
         '<text x="40" y="34" font-size="16" font-weight="700" fill="#16181d">'
         'A transition prior needs recurrence and branching at once.</text>',
         '<text x="40" y="54" font-size="12" fill="#5b6270">'
         '每个点一个 scenario，同一个 classify、同一套严格前缀。点面积 = 决策数。</text>',
         '<text x="40" y="74" font-size="12" fill="#5b6270">'
         '倾向是中等强度，Spearman rho = %+.3f（%d 个 keystep scenario）。</text>'
         % (rho, len(ks)),
         '<text x="40" y="92" font-size="11" fill="#8a4b2a">'
         '注意：y 轴依赖窗口单位（这里 180s）。几乎每个 scenario 的 |S|&#8805;2 '
         '都随窗口单调上升，缩到 60s / 90s 后无点越过两条门槛；x 轴不依赖窗口。</text>']

    # 期望区间
    o.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" fill="#eaf3ea"/>'
             % (sx(HIT_GATE), T, R - sx(HIT_GATE), sy(BR_GATE) - T))
    o.append('<text x="%.1f" y="%.1f" font-size="11" font-weight="700" '
             'fill="#4a7a52" text-anchor="end">期望区间：命中 &#8805;50%% 且 '
             '|S|&#8805;2 &#8805;3%%</text>' % (R - 8, T + 18))

    # 轴
    o.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="#16181d" '
             'stroke-width="1.4"/>' % (L, B, R, B))
    o.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="#16181d" '
             'stroke-width="1.4"/>' % (L, B, L, T))
    for v in range(0, 101, 20):
        o.append('<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" stroke="#16181d"/>'
                 % (sx(v), B, sx(v), B + 5))
        o.append('<text x="%.1f" y="%d" font-size="11" fill="#5b6270" '
                 'text-anchor="middle">%d</text>' % (sx(v), B + 20, v))
    for v in range(0, 9):
        o.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="#16181d"/>'
                 % (L - 5, sy(v), L, sy(v)))
        o.append('<text x="%d" y="%.1f" font-size="11" fill="#5b6270" '
                 'text-anchor="end">%d</text>' % (L - 9, sy(v) + 4, v))
    o.append('<text x="%.1f" y="%d" font-size="13" font-weight="600" '
             'fill="#16181d" text-anchor="middle">状态复现：S 非空时 gold 真在可达集里'
             '的比例（%%）</text>' % ((L + R) / 2, B + 44))
    o.append('<text x="%d" y="%d" font-size="11" fill="#5b6270" '
             'text-anchor="middle">不是 P(n_t = n_{t-1}) —— 那是原地不动，不是回到做过的状态'
             '</text>' % ((L + R) / 2, B + 62))
    o.append('<text x="30" y="%d" font-size="13" font-weight="600" fill="#16181d" '
             'text-anchor="middle" transform="rotate(-90 30 %d)">分叉机会：'
             '|S|&#8805;2 占决策的比例（%%）</text>' % ((T + B) / 2, (T + B) / 2))

    o.append('<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" stroke="#aeb4c0" '
             'stroke-dasharray="4 4"/>' % (sx(HIT_GATE), T, sx(HIT_GATE), B))
    o.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="#aeb4c0" '
             'stroke-dasharray="4 4"/>' % (L, sy(BR_GATE), R, sy(BR_GATE)))

    for p in sorted(pts, key=lambda q: -q["n_decisions"]):
        col, op = STYLE.get(p["grain"], ("#6b7280", 0.7))
        r = 4 + 9 * (p["n_decisions"] / nmax) ** 0.5
        x, y = sx(100 * p["hit"]), sy(100 * p["p_S_ge_2"])
        o.append('<circle cx="%.1f" cy="%.1f" r="%.1f" fill="%s" fill-opacity="%.2f" '
                 'stroke="%s" stroke-width="1"/>' % (x, y, r, col, op, col))
        if 100 * p["hit"] >= a.lmh or p["grain"] != "keystep":
            anchor = "end" if x > (L + R) / 2 else "start"
            dx = -(r + 5) if anchor == "end" else (r + 5)
            o.append('<text x="%.1f" y="%.1f" font-size="11" fill="%s" '
                     'text-anchor="%s">%s</text>'
                     % (x + dx, y + 4, col, anchor, esc(p["label"])))

    lg = [("keystep", "Ego-Exo4D scenario（keystep 粒度）"),
          ("contract", "内部 release（人工合同粒度）"),
          ("human", "人工粗标批次")]
    for i, (k, t) in enumerate(lg):
        col, op = STYLE[k]
        yy = T + 230 + 20 * i
        o.append('<circle cx="%d" cy="%d" r="6" fill="%s" fill-opacity="%.2f" '
                 'stroke="%s"/>' % (L + 14, yy, col, op, col))
        o.append('<text x="%d" y="%d" font-size="11" fill="#5b6270">%s</text>'
                 % (L + 26, yy + 4, esc(t)))

    o.append('<text x="40" y="%d" font-size="10" fill="#8a8f9a">'
             '数据 results/auditor/b2_egoexo/paper_regime_points.json，'
             '由 paper_regime_scatter.py 测量、paper_regime_figure.py 生成。'
             '窗口 180s，scenario 少于 10 个窗口的不画。</text>' % (H - 12))
    o.append('</svg>')
    open(a.out, "w", encoding="utf-8").write("\n".join(o) + "\n")
    print("-> %s" % a.out)
    print("  点 %d 个（keystep %d / 其他 %d），rho = %+.3f"
          % (len(pts), len(ks), len(pts) - len(ks), rho))
    inq = [p for p in pts if 100 * p["hit"] >= HIT_GATE
           and 100 * p["p_S_ge_2"] >= BR_GATE]
    print("  落在期望区间的点：%s"
          % ([(p["label"], round(p["per_window"], 2)) for p in inq] or "无"))


if __name__ == "__main__":
    main()
