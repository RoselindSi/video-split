"""Reward functions for multi-segment procedure segmentation (GRPO, approach B).

Plug into Time-R1's reward_funcs_registry. Each function follows Time-R1's
calling convention: called as ``fn(prompts=..., completions=..., **reward_kwargs)``
where reward_kwargs are the extra dataset columns. So every function accepts
``**kwargs`` and ignores what it does not use.

Ground-truth conventions (must match src/data/convert_multiseg.py):
    solution[i]  : list of [name, start, end]  (the GT segments for example i)
    durations[i] : float video duration in seconds (in kwargs)

All functions return list[float], one reward per completion.
"""

import json
import re

_SEG_RE = re.compile(
    # separator between the two times may be "to" or a dash (-, en/em dash),
    # since models improvise; SFT targets use "to".
    r"<seg>\s*<name>(.*?)</name>\s*<span>\s*(\d+\.?\d*)\s*(?:to|[-–—])\s*(\d+\.?\d*)\s*</span>\s*</seg>",
    re.S | re.I,
)
_STRUCT_RE = re.compile(r"<think>.*?</think>\s*<segments>.*?</segments>", re.S)


def parse_segments(text):
    """Completion text -> list of (name, start, end). Robust to extra prose."""
    out = []
    for m in _SEG_RE.finditer(text):
        name = m.group(1).strip()
        s, e = float(m.group(2)), float(m.group(3))
        out.append((name, s, e))
    return out


def _as_segs(sol):
    """Normalize a GT solution entry (JSON string OR list) -> [(name,s,e),...].

    We store solution as a JSON string in the HF dataset to survive Arrow's
    mixed-type-list limitation, so reward funcs must decode it here.
    """
    if isinstance(sol, str):
        sol = json.loads(sol)
    return [(x[0], float(x[1]), float(x[2])) for x in sol]


def _iou(a_s, a_e, b_s, b_e):
    inter = max(0.0, min(a_e, b_e) - max(a_s, b_s))
    union = max(a_e, b_e) - min(a_s, b_s)
    return inter / union if union > 0 else 0.0


def greedy_match(preds, gts, iou_thresh=0.1):
    """Greedy IoU matching. preds/gts: list of (name, s, e).

    Returns list of (pred_idx, gt_idx, iou), each pred/gt used at most once.
    """
    cand = []
    for pi, (_, ps, pe) in enumerate(preds):
        for gj, (_, gs, ge) in enumerate(gts):
            iou = _iou(ps, pe, gs, ge)
            if iou > iou_thresh:
                cand.append((iou, pi, gj))
    cand.sort(reverse=True)
    used_p, used_g, pairs = set(), set(), []
    for iou, pi, gj in cand:
        if pi in used_p or gj in used_g:
            continue
        used_p.add(pi)
        used_g.add(gj)
        pairs.append((pi, gj, iou))
    return pairs


# ---------------------------------------------------------------- format reward
def format_seg_reward(completions, **kwargs):
    """1.0 = full structure + chronological + valid spans; 0.5 = parses but
    violates chrono/validity; 0.0 = unparseable."""
    out = []
    for c in completions:
        c = c.strip()
        if not _STRUCT_RE.fullmatch(c):
            out.append(0.0)
            continue
        segs = parse_segments(c)
        if not segs:
            out.append(0.0)
            continue
        starts = [s for _, s, _ in segs]
        chrono = all(starts[i] <= starts[i + 1] for i in range(len(starts) - 1))
        valid = all(e > s for _, s, e in segs)
        out.append(1.0 if (chrono and valid) else 0.5)
    return out


# --------------------------------------------------------------- boundary reward
def iou_seg_reward(completions, solution, **kwargs):
    """Mean over |GT| of the Time-R1 iou_v2 compound score on matched pairs.
    Compound = IoU * (1-|Δstart_norm|) * (1-|Δend_norm|). Missed GT -> 0."""
    durations = kwargs.get("durations")
    rewards = []
    for i, (content, sol) in enumerate(zip(completions, solution)):
        dur = durations[i] if durations else None
        gts = _as_segs(sol)
        preds = parse_segments(content)
        if not preds or not gts:
            rewards.append(0.0)
            continue
        pairs = greedy_match(preds, gts)
        total = 0.0
        for pi, gj, iou in pairs:
            _, ps, pe = preds[pi]
            _, gs, ge = gts[gj]
            if dur and dur > 0:
                align = (1 - abs(gs / dur - ps / dur)) * (1 - abs(ge / dur - pe / dur))
            else:
                align = 1.0
            total += iou * max(0.0, align)
        # divide by max(|GT|, |pred|) not just |GT|: penalizes over-segmentation
        # too (extra unmatched preds dilute the score), not only missed GT.
        # Recall-only (/|GT|) let GRPO reward-hack by spamming segments.
        rewards.append(total / max(len(gts), len(preds)))
    return rewards


# ------------------------------------------------------------------ name reward
_SIM_FN = None  # module-level singleton: load the embedding model once, not per step


def _default_sim_fn():
    """Lazy, cached sentence-transformers similarity. fn(a:str, bs:list)->list."""
    global _SIM_FN
    if _SIM_FN is not None:
        return _SIM_FN
    from sentence_transformers import SentenceTransformer, util  # lazy

    model = SentenceTransformer("all-MiniLM-L6-v2")

    def sim(a, bs):
        ea = model.encode(a, convert_to_tensor=True)
        eb = model.encode(bs, convert_to_tensor=True)
        return [float(x) for x in util.cos_sim(ea, eb)[0]]

    _SIM_FN = sim
    return _SIM_FN


def name_seg_reward(completions, solution, sim_fn=None, generic_w=0.5, **kwargs):
    """Mean over |GT| of naming similarity on matched pairs, minus a
    genericity penalty (name that also matches OTHER GT names in the same
    video -> a catch-all, down-weighted). sim_fn injectable for testing."""
    if sim_fn is None:
        sim_fn = _default_sim_fn()
    rewards = []
    for content, sol in zip(completions, solution):
        gts = _as_segs(sol)
        preds = parse_segments(content)
        if not preds or not gts:
            rewards.append(0.0)
            continue
        gt_names = [g[0] for g in gts]
        pairs = greedy_match(preds, gts)
        total = 0.0
        for pi, gj, _ in pairs:
            sims = sim_fn(preds[pi][0], gt_names)  # sim to every GT name
            correct = sims[gj]
            others = [sims[k] for k in range(len(gts)) if k != gj]
            generic = max(others) if others else 0.0
            total += max(0.0, correct - generic_w * generic)
        rewards.append(total / len(gts))
    return rewards


# -------------------------------------------------------------- sequence reward
def seq_reward(completions, solution, w_cov=0.4, w_ov=0.3, w_cnt=0.3, **kwargs):
    """Sequence-level coherence: coverage of the [first_gt_start, last_gt_end]
    window (this also implements task2 head/tail trimming), non-overlap, and
    segment-count match. CoTR/Time-R1 have no equivalent; this is ours."""
    rewards = []
    for content, sol in zip(completions, solution):
        gts = _as_segs(sol)
        preds = parse_segments(content)
        if not preds or not gts:
            rewards.append(0.0)
            continue
        t0 = min(g[1] for g in gts)
        t1 = max(g[2] for g in gts)
        span = max(t1 - t0, 1e-6)

        # coverage: union of predicted spans clipped to [t0, t1]
        covered, cursor = 0.0, t0
        for _, s, e in sorted([(n, s, e) for n, s, e in preds], key=lambda x: x[1]):
            s, e = max(s, cursor), min(max(e, cursor), t1)
            if e > s:
                covered += e - s
                cursor = e
        coverage = min(covered / span, 1.0)

        # overlap ratio between predicted segments
        ps = sorted([(s, e) for _, s, e in preds])
        ov = sum(max(0.0, ps[i][1] - ps[i + 1][0]) for i in range(len(ps) - 1))
        tot = sum(e - s for _, s, e in preds) or 1e-6
        overlap = min(ov / tot, 1.0)

        count = max(0.0, 1 - abs(len(preds) - len(gts)) / len(gts))
        rewards.append(w_cov * coverage + w_ov * (1 - overlap) + w_cnt * count)
    return rewards


# Registry fragment to merge into Time-R1's main.py reward_funcs_registry.
SEG_REWARD_FUNCS = {
    "iou_seg": iou_seg_reward,
    "name_seg": name_seg_reward,
    "seq": seq_reward,
    "format_seg": format_seg_reward,
}
