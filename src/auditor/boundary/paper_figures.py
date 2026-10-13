"""Generate paper figures 2-6 from the stored results, one claim per figure.

WHY GENERATED AND NOT DRAWN. The first version of figure 1 was hand-written
SVG with the coordinates typed in, and it put a whole dataset at one point --
a point that turned out to be one of two draws differing by 3x. Generating
from the result files means the picture cannot drift from the numbers, and it
catches things prose never would: the first generated run produced an invalid
SVG because a scenario name contains a raw ampersand.

FIGURE 4 IS THE EXCEPTION AND IT IS LABELLED AS ONE. E2's raw run output was
never stored; its numbers survive only in the prose of
`docs/interaction_graph_stage_closed.md`. Rather than hardcode them here,
where they would look like every other measured value, they live in
`results/auditor/paper_fig4_e2_from_doc.json` with `reproducible: false` and
a note saying why. The figure prints that caveat on itself, so a reader cannot
mistake a transcription for a measurement.

WHAT EACH FIGURE IS FOR (see docs/paper_figures_and_framing.md):
  2  regime breakdown      the gain is entirely in the deterministic regime,
                           and vision already answers the branching decisions
  3  prior ablation        the signal is successor-set membership, not degree
                           or frequency; and where the set truly constrains it
                           is right 97.5% of the time
  4  representation        serialising history into language keeps the length
                           cost and loses the structure; the positive effect
                           is content-independent
  5  online modes          the bottleneck is identity maintenance, and a hard
                           constraint is the step that amplifies it
  6  grain                 finer grain manufactures a transport primitive;
                           no real state is ever visited three times
"""
from __future__ import annotations

import argparse
import json
import os

W, H = 820, 520
INK, DIM, GRID = "#16181d", "#5b6270", "#aeb4c0"
C_VIS, C_CACHE, C_SUPP = "#8a8f9a", "#2f5d8c", "#a4453f"
C_WARN, C_OK = "#8a4b2a", "#4a7a52"


def esc(s):
    """Scenario and state names carry raw & and <, which make the file
    not well-formed and render as nothing at all."""
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


def head(title, sub, w=W, h=H):
    return ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" '
            'width="%d" height="%d" font-family="Helvetica, Arial, sans-serif">'
            % (w, h, w, h),
            '<rect width="%d" height="%d" fill="#ffffff"/>' % (w, h),
            '<text x="40" y="34" font-size="16" font-weight="700" fill="%s">%s'
            '</text>' % (INK, esc(title)),
            '<text x="40" y="54" font-size="12" fill="%s">%s</text>'
            % (DIM, esc(sub))]


def foot(o, note, h=H):
    o.append('<text x="40" y="%d" font-size="10" fill="#8a8f9a">%s</text>'
             % (h - 12, esc(note)))
    o.append('</svg>')
    return o


def axis_y(o, x, top, bot, lo, hi, step, fmt="%g", label=None):
    o.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" '
             'stroke-width="1.3"/>' % (x, bot, x, top, INK))
    n = int(round((hi - lo) / step))
    for i in range(n + 1):
        v = lo + i * step
        yy = bot - (bot - top) * (v - lo) / (hi - lo)
        o.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s"/>'
                 % (x - 5, yy, x, yy, INK))
        o.append('<text x="%.1f" y="%.1f" font-size="10" fill="%s" '
                 'text-anchor="end">%s</text>'
                 % (x - 8, yy + 3, DIM, fmt % v))
    if label:
        o.append('<text x="%.1f" y="%.1f" font-size="12" font-weight="600" '
                 'fill="%s" text-anchor="middle" transform="rotate(-90 %.1f %.1f)">'
                 '%s</text>' % (x - 46, (top + bot) / 2, INK, x - 46,
                                (top + bot) / 2, esc(label)))


def legend(o, items, x, y):
    for i, (col, t) in enumerate(items):
        o.append('<rect x="%d" y="%d" width="11" height="11" fill="%s"/>'
                 % (x, y + 16 * i - 9, col))
        o.append('<text x="%d" y="%d" font-size="11" fill="%s">%s</text>'
                 % (x + 17, y + 16 * i, DIM, esc(t)))


# ------------------------------------------------------------------ fig 2
def fig2(src, out):
    d = json.load(open(src, encoding="utf-8"))
    A, R, N = d["arms"], d["regimes"], d["n_decisions"]
    order = ["no-support", "deterministic", "branching", "novel"]
    L, T, B = 110, 110, 360
    gw = 560 / len(order)
    o = head("The gain is entirely in the deterministic regime.",
             "ladder n=%d 决策 / %d 录像。branching 那一格视觉单独已经满分，"
             "结构没有余量可占。" % (N, d["n_videos"]))
    axis_y(o, L, T, B, 0.5, 1.0, 0.1, "%.1f", "top-1 准确率")
    o.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="%s" '
             'stroke-width="1.3"/>' % (L, B, L + 560, B, INK))
    arms = [("visual", C_VIS), ("cache", C_CACHE), ("support", C_SUPP)]
    for gi, reg in enumerate(order):
        x0 = L + gi * gw
        for ai, (arm, col) in enumerate(arms):
            v = A[arm][reg]
            bw = gw * 0.22
            x = x0 + gw * 0.12 + ai * bw
            hh = (B - T) * (v - 0.5) / 0.5
            o.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" '
                     'fill="%s"/>' % (x, B - hh, bw - 3, hh, col))
            o.append('<text x="%.1f" y="%.1f" font-size="9" fill="%s" '
                     'text-anchor="middle">%.3f</text>'
                     % (x + bw / 2 - 1.5, B - hh - 4, DIM, v))
        o.append('<text x="%.1f" y="%d" font-size="12" font-weight="600" '
                 'fill="%s" text-anchor="middle">%s</text>'
                 % (x0 + gw / 2, B + 18, INK, reg))
        o.append('<text x="%.1f" y="%d" font-size="10" fill="%s" '
                 'text-anchor="middle">n=%d（%.1f%%）</text>'
                 % (x0 + gw / 2, B + 34, DIM, R[reg], 100 * R[reg] / N))
    legend(o, arms_legend(), L + 14, T + 16)
    # 增益归属
    o.append('<text x="%d" y="%d" font-size="13" font-weight="700" fill="%s">'
             'support 比 visual 多答对 %.0f 题，来自哪里</text>'
             % (L, B + 70, INK, (A["support"]["overall"] - A["visual"]["overall"]) * N))
    for i, reg in enumerate(order):
        dv = (A["support"][reg] - A["visual"][reg]) * R[reg]
        col = C_SUPP if abs(dv) > 0.5 else DIM
        o.append('<text x="%d" y="%d" font-size="12" fill="%s">%-14s %+4.0f</text>'
                 % (L + 10, B + 92 + 18 * i, col, esc(reg), dv))
    o.append('<text x="%d" y="%d" font-size="11" fill="%s">branching 这一格'
             'visual 已经 %.3f —— 11 题全对，没有空间给结构</text>'
             % (L + 220, B + 110, C_WARN, A["visual"]["branching"]))
    o.append('<text x="%d" y="%d" font-size="11" fill="%s">'
             'support − cache = %+.4f（CI 跨零）→ 完整可达集相对单条目缓存无增益</text>'
             % (L + 220, B + 128, C_WARN,
                A["support"]["overall"] - A["cache"]["overall"]))
    foot(o, "数据 %s。self/recency/frequency 三臂的 λ 被 CV 选成 0，与 visual 逐位相同，"
            "因此不画 —— 它们不是三个独立结果。" % os.path.basename(src))
    write(out, o)


def arms_legend():
    return [(C_VIS, "visual（无先验）"), (C_CACHE, "cache（单条目后继）"),
            (C_SUPP, "support（完整可达集）")]


# ------------------------------------------------------------------ fig 3
def fig3(src, out):
    d = json.load(open(src, encoding="utf-8"))
    pops = [("all", "全部决策", d["all"]), ("nonvacuous", "只取约束非空的",
                                          d.get("nonvacuous"))]
    pops = [p for p in pops if p[2]]
    conds = ["U", "F", "Rt", "Rs"]
    name = {"U": "均匀", "F": "节点频率", "Rt": "真可达集", "Rs": "度匹配打乱"}
    col = {"U": C_VIS, "F": "#9a7f3a", "Rt": C_SUPP, "Rs": C_CACHE}
    L, T, B = 110, 120, 360
    gw = 540 / len(pops)
    o = head("The signal is successor-set membership, not degree or frequency.",
             "纯先验排序，无视觉模型、无语言模型。n=%d 决策 / %d 录像。"
             "top-1 是随机打破平局下的期望值。"
             % (d["n_decisions"], d["n_videos"]))
    axis_y(o, L, T, B, 0, 100, 20, "%d", "top-1（%）")
    o.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="%s" '
             'stroke-width="1.3"/>' % (L, B, L + 540, B, INK))
    for gi, (key, lab, s) in enumerate(pops):
        x0 = L + gi * gw
        bw = gw * 0.17
        for ci, c in enumerate(conds):
            v = 100 * s["top1"][c]
            x = x0 + gw * 0.10 + ci * bw
            hh = (B - T) * v / 100
            o.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" '
                     'fill="%s"/>' % (x, B - hh, bw - 4, hh, col[c]))
            o.append('<text x="%.1f" y="%.1f" font-size="10" fill="%s" '
                     'text-anchor="middle">%.1f</text>'
                     % (x + bw / 2 - 2, B - hh - 5, DIM, v))
        o.append('<text x="%.1f" y="%d" font-size="12" font-weight="600" '
                 'fill="%s" text-anchor="middle">%s（n=%d）</text>'
                 % (x0 + gw / 2, B + 20, INK, esc(lab), s["n"]))
        ct = s["contrasts"].get("Rt-Rs")
        if ct:
            o.append('<text x="%.1f" y="%d" font-size="11" fill="%s" '
                     'text-anchor="middle">Rt − Rs = %+.1f%% '
                     'CI [%+.1f, %+.1f]</text>'
                     % (x0 + gw / 2, B + 38, C_SUPP, 100 * ct["delta"],
                        100 * ct["ci"][0], 100 * ct["ci"][1]))
    legend(o, [(col[c], name[c]) for c in conds], L + 14, T + 14)
    rsu = d["all"]["contrasts"].get("Rs-U")
    fu = d["all"]["contrasts"].get("F-U")
    y = B + 76
    o.append('<text x="%d" y="%d" font-size="12" fill="%s">度匹配打乱之后一点不剩：'
             'Rs − U = %+.1f%% CI [%+.1f, %+.1f]，跨零</text>'
             % (L, y, INK, 100 * rsu["delta"], 100 * rsu["ci"][0],
                100 * rsu["ci"][1]))
    o.append('<text x="%d" y="%d" font-size="12" fill="%s">频率先验低于均匀：'
             'F − U = %+.1f%% —— 转移频次的贡献不成立</text>'
             % (L, y + 20, INK, 100 * fu["delta"]))
    if d.get("nonvacuous"):
        o.append('<text x="%d" y="%d" font-size="12" font-weight="600" fill="%s">'
                 '约束真的收窄候选时，可达集命中 %.1f%% —— 稀缺但不空洞</text>'
                 % (L, y + 42, C_OK, 100 * d["nonvacuous"]["top1"]["Rt"]))
    foot(o, "数据 %s。注意：均匀先验 %.1f%% 不是视觉基线 —— 视觉基线是 ladder 的 "
            "0.730（n=322），这里没有视觉模型。"
            % (os.path.basename(src), 100 * d["all"]["top1"]["U"]))
    write(out, o)


# ------------------------------------------------------------------ fig 4
def fig4(src, out):
    d = json.load(open(src, encoding="utf-8"))
    L, T = 260, 120
    o = head("Serialising history into language keeps the cost and loses the structure.",
             "E2 表示消融：长度与结构解耦之后的效应，外加两个内容无关对照。")
    x0 = L + 180
    sc = 8.0
    o.append('<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" stroke="%s" '
             'stroke-width="1.3"/>' % (x0, T - 10, x0, T + 150, INK))
    o.append('<text x="%.1f" y="%d" font-size="10" fill="%s" '
             'text-anchor="middle">0</text>' % (x0, T - 16, DIM))
    for i, e in enumerate(d["effects"]):
        y = T + 14 + 34 * i
        v = e["delta_points"]
        col = C_SUPP if v < 0 else C_OK
        wd = abs(v) * sc
        x = x0 if v > 0 else x0 - wd
        o.append('<rect x="%.1f" y="%.1f" width="%.1f" height="16" fill="%s" '
                 'fill-opacity="0.85"/>' % (x, y - 12, wd, col))
        o.append('<text x="%.1f" y="%.1f" font-size="12" fill="%s" '
                 'text-anchor="end">%s</text>' % (x0 - 190, y, INK, esc(e["name"])))
        tag = "" if e["significant"] is None else (
            "" if e["significant"] else "（不显著）")
        ci = e.get("ci")
        txt = "%+.1f 点 %s" % (v, tag)
        if ci:
            txt = "%+.1f [%+.1f, %+.1f] %s" % (v, ci[0], ci[1], tag)
        o.append('<text x="%.1f" y="%.1f" font-size="11" fill="%s">%s</text>'
                 % (x + wd + 6 if v > 0 else x0 + 6, y, DIM, esc(txt)))
        if ci:
            a, b = x0 + ci[0] * sc, x0 + ci[1] * sc
            o.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" '
                     'stroke-width="1"/>' % (a, y - 4, b, y - 4, col))
    cc = d["confound_corrected"]
    y = T + 180
    o.append('<text x="%d" y="%d" font-size="13" font-weight="700" fill="%s">'
             '一处混淆的更正</text>' % (70, y, INK))
    bci = cc.get("before_ci"); aci = cc.get("after_ci")
    o.append('<text x="%d" y="%d" font-size="12" fill="%s">%s 最初 %+.1f%s → '
             '加入对照臂后 %+.1f%s%s</text>'
             % (70, y + 22, INK, esc(cc["arm"]), cc["before"],
                (" [%+.1f, %+.1f]" % tuple(bci)) if bci else "",
                cc["after"], (" [%+.1f, %+.1f]" % tuple(aci)) if aci else "",
                "（跨零）" if cc["after_crosses_zero"] else ""))
    o.append('<text x="%d" y="%d" font-size="11" fill="%s">%s</text>'
             % (70, y + 42, C_WARN, esc(cc["cause"])))
    o.append('<text x="%d" y="%d" font-size="12" font-weight="600" fill="%s">%s'
             '</text>' % (70, y + 74, C_OK, esc(d["claim"])))
    note = ("数据 %s" % os.path.basename(src))
    if d.get("reproducible"):
        note += ("，n=%s / %s 条视频，从服务器逐条 jsonl 重新评分（零模型调用）"
                 % (d.get("n"), d.get("n_videos")))
    else:
        note += " —— **转录自文档，reproducible=false**，其余五张图均为测量产物"
    foot(o, note)
    write(out, o)


# ------------------------------------------------------------------ fig 5
def fig5(src, out):
    d = json.load(open(src, encoding="utf-8"))
    M = d["modes"]
    order = [("visual", "visual"), ("oracle", "oracle 前驱"),
             ("prev", "预测前驱"), ("full", "全自维护")]
    order = [(k, l) for k, l in order if k in M]
    L, T, B = 110, 120, 340
    gw = 560 / len(order)
    o = head("The bottleneck is identity maintenance, not transition prediction.",
             "Experiment D，n=%d 决策 / %d 录像。两条线必须成对读。"
             % (d["n_decisions"], d["n_videos"]))
    axis_y(o, L, T, B, 0.5, 1.0, 0.1, "%.1f", "比例")
    o.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="%s" '
             'stroke-width="1.3"/>' % (L, B, L + 560, B, INK))
    series = [("identity_top1", C_SUPP, "identity top-1（整体准确率）"),
              ("diff_state_separated", C_CACHE, "diff-state separated（状态分离度）")]
    for key, col, _ in series:
        pts = []
        for gi, (k, _l) in enumerate(order):
            v = M[k][key]
            x = L + gi * gw + gw / 2
            y = B - (B - T) * (v - 0.5) / 0.5
            pts.append((x, y, v))
        o.append('<polyline fill="none" stroke="%s" stroke-width="2" '
                 'points="%s"/>' % (col, " ".join("%.1f,%.1f" % (p[0], p[1])
                                                  for p in pts)))
        for x, y, v in pts:
            o.append('<circle cx="%.1f" cy="%.1f" r="5" fill="%s"/>'
                     % (x, y, col))
            o.append('<text x="%.1f" y="%.1f" font-size="10" fill="%s" '
                     'text-anchor="middle">%.3f</text>' % (x, y - 10, col, v))
    for gi, (k, lab) in enumerate(order):
        x = L + gi * gw + gw / 2
        o.append('<text x="%.1f" y="%d" font-size="12" font-weight="600" '
                 'fill="%s" text-anchor="middle">%s</text>'
                 % (x, B + 20, INK, esc(lab)))
        po = M[k].get("prior_override")
        if po is not None:
            col = C_WARN if po >= 10 else DIM
            o.append('<text x="%.1f" y="%d" font-size="11" fill="%s" '
                     'text-anchor="middle">先验推翻视觉 %d 次</text>'
                     % (x, B + 38, col, po))
    legend(o, [(c, t) for _k, c, t in series], L + 14, T + 14)
    y = B + 72
    o.append('<text x="%d" y="%d" font-size="12" font-weight="600" fill="%s">'
             '① 先验作为分离器有效，崩的是喂给它的身份估计</text>' % (L, y, INK))
    o.append('<text x="%d" y="%d" font-size="11" fill="%s">预测前驱的状态分离度 '
             '%.3f 是四种模式里最高的，而整体准确率掉回 %.3f（visual %.3f）</text>'
             % (L + 14, y + 18, DIM, M["prev"]["diff_state_separated"],
                M["prev"]["identity_top1"], M["visual"]["identity_top1"]))
    o.append('<text x="%d" y="%d" font-size="12" font-weight="600" fill="%s">'
             '② 把先验做成硬约束，是身份误差转成预测误差的那一步</text>'
             % (L, y + 42, INK))
    o.append('<text x="%d" y="%d" font-size="11" fill="%s">prior_override 从 oracle '
             '的 %d 跳到预测前驱的 %d —— 一旦前驱靠估计，它推翻视觉 %d 次而多半推翻错'
             '</text>' % (L + 14, y + 60, DIM, M["oracle"]["prior_override"],
                         M["prev"]["prior_override"], M["prev"]["prior_override"]))
    foot(o, "数据 %s。oracle 前驱 %+.3f、预测前驱 %+.3f、全自维护 %+.3f（相对 visual）。"
         % (os.path.basename(src),
            M["oracle"]["identity_top1"] - M["visual"]["identity_top1"],
            M["prev"]["identity_top1"] - M["visual"]["identity_top1"],
            M["full"]["identity_top1"] - M["visual"]["identity_top1"]))
    write(out, o)


# ------------------------------------------------------------------ fig 6
def fig6(src, out, keystep_per_window=0.0):
    d = json.load(open(src, encoding="utf-8"))
    steps = d["steps"]
    L, T, B = 110, 120, 300
    o = head("Finer grain manufactures a transport primitive, not branching.",
             "B2 第三轮，修车族 %d 个窗口，动作级粒度 + 强制 id 复用。"
             % d["n_final"])
    axis_y(o, L, T, B, 0, 0.6, 0.1, "%.1f", "G-branch / 窗口")
    o.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="%s" '
             'stroke-width="1.3"/>' % (L, B, L + 300, B, INK))
    show = [("keystep 粒度", keystep_per_window, C_VIS)] + [
        (s["label"].split("+ ")[-1] if "+" in s["label"] else s["label"],
         s["per_window"], C_SUPP if s["per_window"] > 0 else C_OK)
        for s in (steps[0], steps[-1])]
    bw = 70
    for i, (lab, v, col) in enumerate(show):
        x = L + 24 + i * (bw + 26)
        hh = (B - T) * min(v, 0.6) / 0.6
        o.append('<rect x="%.1f" y="%.1f" width="%d" height="%.1f" fill="%s"/>'
                 % (x, B - hh, bw, hh, col))
        o.append('<text x="%.1f" y="%.1f" font-size="11" fill="%s" '
                 'text-anchor="middle">%.2f</text>' % (x + bw / 2, B - hh - 6,
                                                       DIM, v))
        o.append('<text x="%.1f" y="%d" font-size="11" fill="%s" '
                 'text-anchor="middle">%s</text>'
                 % (x + bw / 2, B + 18, INK, esc(lab)))
    o.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="%s" '
             'stroke-dasharray="4 3"/>' % (L, B - (B - T) * 0.5 / 0.6,
                                           L + 300, B - (B - T) * 0.5 / 0.6,
                                           GRID))
    o.append('<text x="%d" y="%.1f" font-size="10" fill="%s">冻结判据宽端 0.5</text>'
             % (L + 302, B - (B - T) * 0.5 / 0.6 + 3, DIM))

    # 右panel：访问次数
    x0 = L + 400
    o.append('<text x="%d" y="%d" font-size="13" font-weight="700" fill="%s">'
             'G-branch 的前提：某状态被访问 &#8805;3 次</text>' % (x0, T - 14, INK))
    rd, ad = d["real_visit_dist"], d["acq_visit_dist"]
    kmax = max([int(k) for k in list(rd) + list(ad)] or [4])
    bw2 = 44
    for i in range(kmax + 1):
        for j, (dist, col, lab) in enumerate(((rd, C_SUPP, "真实状态"),
                                              (ad, C_VIS, "取得类"))):
            n = dist.get(str(i), dist.get(i, 0))
            x = x0 + i * (bw2 + 8) + j * (bw2 / 2)
            hh = 12 * n
            o.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" '
                     'fill="%s"/>' % (x, B - hh, bw2 / 2 - 2, hh, col))
            if n:
                o.append('<text x="%.1f" y="%.1f" font-size="9" fill="%s" '
                         'text-anchor="middle">%d</text>'
                         % (x + bw2 / 4, B - hh - 4, DIM, n))
        o.append('<text x="%.1f" y="%d" font-size="11" fill="%s" '
                 'text-anchor="middle">%d 次</text>'
                 % (x0 + i * (bw2 + 8) + bw2 / 2, B + 18, INK, i))
    o.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="%s" '
             'stroke-width="1.3"/>' % (x0, B, x0 + (kmax + 1) * (bw2 + 8), B, INK))
    o.append('<text x="%d" y="%d" font-size="10" fill="%s">纵向 = 录像条数</text>'
             % (x0, B + 36, DIM))
    legend(o, [(C_SUPP, "真实状态的最大访问次数"),
               (C_VIS, "取得类状态的最大访问次数")], x0, T + 8)
    need = sum(v for k, v in rd.items() if int(k) >= 3)
    y = B + 72
    o.append('<text x="%d" y="%d" font-size="12" font-weight="600" fill="%s">'
             '真实状态达到 3 次的录像：%d / %d —— 为 0 则 G-branch 不可能由真实状态产生'
             '</text>' % (L, y, C_WARN, need, d["n_final"]))
    hubs = steps[0].get("hubs") or {}
    if hubs:
        tot = sum(hubs.values())
        top = max(hubs, key=hubs.get)
        o.append('<text x="%d" y="%d" font-size="12" fill="%s">如实计的 %d 个 '
                 'G-branch 里，前驱是「%s」的占 %d 个（%.0f%%）</text>'
                 % (L, y + 20, INK, tot, esc(top), hubs[top],
                    100 * hubs[top] / tot))
    o.append('<text x="%d" y="%d" font-size="11" fill="%s">产出率可以靠把过渡动作'
             '切得更碎无限刷高，所以它不能作为「域里有 branching」的证据</text>'
             % (L, y + 42, C_WARN))
    foot(o, "数据 %s。折叠规则：状态 = 作用于什么 + 怎么作用；只为取得手段而存在的"
            "移动不建状态。" % os.path.basename(src))
    write(out, o)


def write(path, o):
    open(path, "w", encoding="utf-8").write("\n".join(o) + "\n")
    import xml.dom.minidom as md
    md.parse(path)                      # 不合法就在这里炸，而不是在论文里空白
    print("  -> %s" % path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ladder", default="results/auditor/interaction_graph_ladder.json")
    ap.add_argument("--g1b", default="results/auditor/interaction_graph_g1b_summary.json")
    ap.add_argument("--e2", default="results/auditor/paper_fig4_e2_from_doc.json")
    ap.add_argument("--expd", default="results/auditor/interaction_graph_expD_online.json")
    ap.add_argument("--fold", default="results/auditor/b2_egoexo/b2_bike_fold_r1.json")
    ap.add_argument("--bike-keystep-per-window", dest="bkw", type=float,
                    default=0.0, help="修车族 keystep 粒度的每窗口产出，图 6 的对照柱")
    ap.add_argument("--outdir", default="docs")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    print("生成论文图 2-6：")
    fig2(a.ladder, os.path.join(a.outdir, "paper_fig2_regime_breakdown.svg"))
    fig3(a.g1b, os.path.join(a.outdir, "paper_fig3_prior_ablation.svg"))
    fig4(a.e2, os.path.join(a.outdir, "paper_fig4_representation.svg"))
    fig5(a.expd, os.path.join(a.outdir, "paper_fig5_online_modes.svg"))
    fig6(a.fold, os.path.join(a.outdir, "paper_fig6_grain.svg"), a.bkw)


if __name__ == "__main__":
    main()
