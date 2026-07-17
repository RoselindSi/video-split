"""B6 (text-level pass, before any video clips) -- classify the 56.8% missed
GT boundaries and the false peaks from the B2 run using the segment NAMES
already saved in the logits file (train_head_multi.py --save_logits, with
"segments" included). No re-decoding of video needed for this pass.

Two things this answers cheaply, before spending time pulling frames:
  1. Are missed boundaries disproportionately the ones where the segment name
     DOESN'T change across the cut (same repeated action, e.g. cycle N vs
     cycle N+1 of "fold tissue")? Those are plausibly harder because there's
     no real "state change", vs boundaries where the action itself changes.
  2. Where do false peaks land -- inside which named segments, and how far
     into the segment (start/mid/end)? Peaks clustered near segment
     start/end but just outside tolerance point at LOCALIZATION error
     (right idea, wrong second); peaks in the middle of a segment point at
     spurious motion (hand adjustment, camera shake) being mistaken for a
     boundary.

This narrows down WHICH boundary-type hypotheses (from the B6 plan: semantic
change / same-action-internal / hand in-out / camera shake / occlusion /
pause-then-continue / GT ambiguity / fine-grained same-object stage) are worth
pulling actual video clips for, instead of eyeballing 50 clips blind.

Usage (server, after a train_head_multi.py --save_logits run):
    python -m src.boundary.fp_fn_text_audit --logits /tmp/b2_logits.pt \
        --thr 0.45 --min_gap 1.0 --tol 0.5
"""
import argparse, statistics
from collections import Counter


def peaks_threshold(prob, times, thr, min_gap):
    cand = [i for i in range(len(prob))
            if prob[i] >= thr
            and (i == 0 or prob[i] >= prob[i - 1])
            and (i == len(prob) - 1 or prob[i] >= prob[i + 1])]
    cand.sort(key=lambda i: -prob[i]); kept = []
    for i in cand:
        if all(abs(times[i] - times[j]) >= min_gap for j in kept):
            kept.append(i)
    return sorted(times[i] for i in kept)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logits", required=True)
    ap.add_argument("--thr", type=float, default=0.45)
    ap.add_argument("--min_gap", type=float, default=1.0)
    ap.add_argument("--tol", type=float, default=0.5)
    ap.add_argument("--examples", type=int, default=12)
    a = ap.parse_args()

    import torch
    data = torch.load(a.logits, weights_only=False)
    if not data or "segments" not in data[0]:
        raise SystemExit("logits file has no 'segments' field -- re-run "
                          "train_head_multi.py --save_logits with the updated "
                          "dump_logits() that stores x['segments'].")

    boundary_stats = Counter()          # (same_or_diff, matched_or_missed) -> n
    missed_examples = {"same": [], "diff": []}
    missed_video_cap = Counter()        # recording_id -> # examples already taken
    false_peak_examples = []
    false_peak_video_cap = Counter()
    false_peak_positions = []           # frac into containing segment, or None
    missed_by_video = Counter()         # for the outlier-video sanity check
    false_peak_by_video = Counter()
    PER_VIDEO_CAP = 2

    for v in data:
        segs = sorted(v["segments"], key=lambda s: s[1])
        starts = {round(s[1], 2): s[0] for s in segs}
        ends = {round(s[2], 2): s[0] for s in segs}
        gts, preds = v["gt"], peaks_threshold(v["prob"], v["times"], a.thr, a.min_gap)

        used = set()
        for p in preds:
            best, bj = a.tol + 1, -1
            for j, g in enumerate(gts):
                if j not in used and abs(p - g) < best:
                    best, bj = abs(p - g), j
            if bj >= 0 and best <= a.tol:
                used.add(bj)

        for j, g in enumerate(gts):
            before = ends.get(round(g, 2), "?")
            after = starts.get(round(g, 2), "?")
            kind = "same" if before == after else "diff"
            status = "matched" if j in used else "missed"
            boundary_stats[(kind, status)] += 1
            if status == "missed":
                missed_by_video[v["recording_id"]] += 1
                if (len(missed_examples[kind]) < a.examples
                        and missed_video_cap[v["recording_id"]] < PER_VIDEO_CAP):
                    missed_examples[kind].append(
                        (v["recording_id"], round(g, 2), before, after))
                    missed_video_cap[v["recording_id"]] += 1

        for p in preds:
            d = min((abs(p - g) for g in gts), default=999)
            if d > a.tol:
                false_peak_by_video[v["recording_id"]] += 1
                containing = next((s for s in segs if s[1] <= p <= s[2]), None)
                if containing:
                    frac = (p - containing[1]) / max(containing[2] - containing[1], 1e-6)
                    false_peak_positions.append(frac)
                if (len(false_peak_examples) < a.examples
                        and false_peak_video_cap[v["recording_id"]] < PER_VIDEO_CAP):
                    if containing:
                        false_peak_examples.append(
                            (v["recording_id"], round(p, 2), containing[0], round(frac, 2)))
                    else:
                        false_peak_examples.append((v["recording_id"], round(p, 2), "<gap>", None))
                    false_peak_video_cap[v["recording_id"]] += 1

    def rate(kind):
        m, x = boundary_stats[(kind, "matched")], boundary_stats[(kind, "missed")]
        tot = m + x
        return x, tot, x / max(tot, 1)

    print(f"=== missed-boundary rate by same-name vs diff-name across the cut ===")
    for kind in ("same", "diff"):
        x, tot, r = rate(kind)
        print(f"  {kind}-name boundaries: missed {x}/{tot} = {r:.1%}")
    print("(if 'same' misses far more than 'diff', repeated-cycle/no-state-change "
          "boundaries are the hard case -> pull video clips for THOSE first, "
          "hand-object contact signal likely more useful there than more visual "
          "features. If rates are close, the model is failing broadly, not just "
          "on ambiguous same-action cuts.)")

    print(f"\n=== false-peak position within their containing segment "
          f"(0=segment start, 1=segment end; n={len(false_peak_positions)}) ===")
    if false_peak_positions:
        near_edge = sum(1 for f in false_peak_positions if f < 0.15 or f > 0.85)
        print(f"  mean={statistics.mean(false_peak_positions):.2f}  "
              f"median={statistics.median(false_peak_positions):.2f}  "
              f"near an edge (<0.15 or >0.85): {near_edge}/{len(false_peak_positions)} "
              f"= {near_edge/len(false_peak_positions):.1%}")
        print("(high near-edge fraction = localization error, model senses the "
              "right region but times it wrong -> B4/temporal head, not audit. "
              "Low near-edge fraction = spurious mid-segment activations "
              "(motion/occlusion/camera) -> B6 video audit / B8 hand signal.)")

    total_missed = sum(missed_by_video.values())
    total_fp = sum(false_peak_by_video.values())
    print(f"\n=== outlier-video check (is one video dominating the failure counts?) ===")
    print(f"missed boundaries, top 5 videos:")
    for rid, c in missed_by_video.most_common(5):
        print(f"  {rid}: {c}/{total_missed} = {c/max(total_missed,1):.1%}")
    print(f"false peaks, top 5 videos:")
    for rid, c in false_peak_by_video.most_common(5):
        print(f"  {rid}: {c}/{total_fp} = {c/max(total_fp,1):.1%}")

    print(f"\n=== example missed boundaries, same-name (repeated action) ===")
    for rid, t, before, after in missed_examples["same"]:
        print(f"  {rid} @ {t}s: '{before}' -> '{after}'")
    print(f"\n=== example missed boundaries, diff-name (real action change) ===")
    for rid, t, before, after in missed_examples["diff"]:
        print(f"  {rid} @ {t}s: '{before}' -> '{after}'")
    print(f"\n=== example false peaks (recording, time, containing segment, "
          f"frac into segment) ===")
    for rid, t, name, frac in false_peak_examples:
        print(f"  {rid} @ {t}s  in '{name}'  frac={frac}")


if __name__ == "__main__":
    main()
