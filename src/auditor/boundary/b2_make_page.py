"""Rebuild the coarse page for a batch by swapping its embedded manifest.

The page carries a batch as two lines, a CASES array and a BATCH/MHASH line,
and nothing else is batch-specific. So a rebuild is a substitution, and the
rest of the page -- speed buttons, segment arrows, draggable boundaries, merge,
the branch-condition prompt, the live G-branch counter -- stays byte-identical
across batches. That is the point rather than a convenience: the instrument
must not drift between batches except where a protocol change is declared, and
a diff of two generated pages should show only the declared lines.

The localStorage key is "b2-"+BATCH+"-"+ANNOTATOR, so a new batch id keeps new
drafts from colliding with an earlier batch's. Checked by refusing to write a
page that still mentions the old batch id anywhere.

TWO DECLARED INSERTS, BOTH FORCED BY BATCH 1'S RESULT.

`--grain` adds the action-level rule. At batch 1's activity grain -- three
states per three minutes -- no predecessor was ever revisited, so the
measurement had a floor at zero whatever the domain contained.

`--emphasise-reuse` promotes the existing "reuse the code when you return to
something" hint into the measured variable. The internal release's successor
set holds the truth 95.6% of the time and Ego-Exo4D's 11%, and part of that
gap is that `step_unique_id` indexes which step rather than which state, so
returning to the same physical state draws a fresh id. Whether that gap is the
dataset or the instrument is exactly what a human pass with enforced reuse
separates.

`--order-seed` shuffles presentation order. It matters only when re-labelling
windows the annotator has seen, where the risk is reproducing the earlier
segmentation from memory. That anchoring biases toward the old coarse cuts and
so against finding branching, which is the conservative direction -- a
re-label that does find branching is therefore safe to believe, while one that
does not leaves anchoring unexcluded and needs a fresh arm.
"""
from __future__ import annotations

import argparse
import json
import random
import re

GRAIN = (
    '    <p class="hint" style="color:var(--fg)"><b>状态定在动作级，不是活动级。</b>'
    '写「拿盐罐往锅里放盐」，不要写「调味」。自检：如果一个状态在这条录像里'
    '只可能出现一次，它就定得太粗了。</p>\n'
)
REUSE = (
    '    <p class="hint" style="color:var(--fg)"><b>回到做过的事，必须选回原来那个代号。</b>'
    '新建一个同义的代号，会让这次分叉在分析里直接消失 —— 这一批要测的就是这件事。'
    '同一个容器、同一个动作，哪怕隔了很久，也选回原代号。</p>\n'
)
ANCHOR = ('    <p class="hint">回到之前做过的事，<b>一定要选回原来那个代号</b>。'
          '不确定就选「看不清」。</p>\n')


def build(src, pm, annotator, grain=False, reuse=False, order_seed=None):
    """-> page text. Raises on any check that would let a batch leak."""
    m = re.search(r'const BATCH="([^"]+)"', src)
    if not m:
        raise SystemExit("页面里找不到 const BATCH，格式变了")
    old_batch = m.group(1)

    cases = [{"video_id": v["video_id"], "duration_s": v["duration_s"]}
             for v in pm["videos"]]
    if order_seed is not None:
        random.Random(order_seed).shuffle(cases)

    new_cases = "const CASES=%s;" % json.dumps(cases, ensure_ascii=False,
                                               separators=(",", ":"))
    new_batch = ('const BATCH="%s", ANNOTATOR="%s", MHASH="%s";'
                 % (pm["batch_id"], annotator, pm["manifest_sha256"][:12]))

    out, n_c, n_b, n_ins = [], 0, 0, 0
    for line in src.splitlines(keepends=True):
        if line.startswith("const CASES=["):
            out.append(new_cases + "\n"); n_c += 1; continue
        if line.startswith("const BATCH="):
            out.append(new_batch + "\n"); n_b += 1; continue
        if line == ANCHOR:
            out.append(line)
            if grain:
                out.append(GRAIN); n_ins += 1
            if reuse:
                out.append(REUSE); n_ins += 1
            continue
        out.append(line)
    if (n_c, n_b) != (1, 1):
        raise SystemExit("CASES 替换 %d 次 / BATCH 替换 %d 次，都应该是 1"
                         % (n_c, n_b))
    txt = "".join(out)
    if grain and GRAIN not in txt:
        raise SystemExit("粒度说明没插进去：锚点文案变了")
    if reuse and REUSE not in txt:
        raise SystemExit("id 复用说明没插进去：锚点文案变了")
    if old_batch in txt:
        raise SystemExit("旧 batch id %s 还残留，草稿会和上一批串" % old_batch)
    for c in cases[:3]:
        if c["video_id"] not in txt:
            raise SystemExit("新 manifest 没写进去：缺 %s" % c["video_id"])
    return txt, old_batch, cases, n_ins


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--page", required=True, help="上一批的页面 html")
    ap.add_argument("--packet", required=True, help="packet_manifest.json")
    ap.add_argument("--annotator", default="r1")
    ap.add_argument("--title", help="改 <title>，不改则沿用上一批的")
    ap.add_argument("--grain", action="store_true", help="插入动作级粒度规则")
    ap.add_argument("--emphasise-reuse", dest="reuse", action="store_true",
                    help="把 id 复用提成被测变量")
    ap.add_argument("--order-seed", dest="seed", type=int,
                    help="打乱呈现顺序；重标看过的窗口时用")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    src = open(a.page, encoding="utf-8").read()
    pm = json.load(open(a.packet, encoding="utf-8"))
    txt, old, cases, n_ins = build(src, pm, a.annotator, a.grain, a.reuse,
                                   a.seed)
    if a.title:
        txt, n = re.subn(r"<title>[^<]*</title>", "<title>%s</title>" % a.title,
                         txt, count=1)
        if n != 1:
            raise SystemExit("<title> 没替换成功")
    open(a.out, "w", encoding="utf-8").write(txt)

    print("页面 -> %s" % a.out)
    print("  batch %s / 标注者 %s / MHASH %s"
          % (pm["batch_id"], a.annotator, pm["manifest_sha256"][:12]))
    print("  视频 %d 个，时长合计 %.0f 分钟"
          % (len(cases), sum(c["duration_s"] for c in cases) / 60))
    print("  localStorage key = b2-%s-%s（与上一批 %s 不冲突）"
          % (pm["batch_id"], a.annotator, old))
    print("  声明的插入 %d 条（粒度=%s，id 复用=%s）；呈现顺序种子 %s"
          % (n_ins, a.grain, a.reuse, a.seed))
    print("  模板 %d 行；只改 2 行、插 %d 行" % (len(src.splitlines()), n_ins))


if __name__ == "__main__":
    main()
