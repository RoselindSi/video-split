"""B-final media step -- for a balanced sample of classified events from
boundary_error_audit.py's predictions.jsonl, render the three artifacts a
human audit actually needs (not just numbers):

  {event_id}.mp4              : ~3s before/after the event center
  {event_id}_contact_sheet.png: 9 frames across that window (local action
                                 change, same layout family as the other
                                 contact-sheet scripts in this repo)
  {event_id}_score_plot.png   : the boundary-probability curve in that
                                 window, with GT boundaries (green dashed)
                                 and predicted peaks (red dashed) marked

Samples across ALL of boundary_error_audit.py's categories (exact/early/late/
missed-weak_signal/missed-signal_present_not_top for GT boundaries;
duplicate/false_near_edge/false_mid_segment for unmatched predicted peaks),
capped per category AND per recording so no single video dominates.

Requires ffmpeg on PATH for the .mp4 clips; if unavailable, clips are
skipped with a warning but contact sheets + score plots (matplotlib + decord
only) still render.

Usage (server, after boundary_error_audit.py has written predictions.jsonl):
    python -m src.boundary.render_audit_media \
        --predictions /workspace/tr1/results/boundary/error_audit/predictions.jsonl \
        --logits /workspace/tr1/results/boundary/b2_logits.pt \
        --data /workspace/tr1/data_recseg/recseg_val.json \
        --out_dir /workspace/tr1/results/boundary/error_audit \
        --n_per_category 8 --window_s 3.0
"""
import argparse, csv, json, os, random, shutil, subprocess
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from decord import VideoReader
from PIL import Image, ImageDraw


def make_contact_sheet(vr, vfps, center, window_s, gt_in_window, pred_in_window, thumb_w=160):
    n = len(vr)
    offsets = [-window_s + i * (2 * window_s / 8) for i in range(9)]
    pts = [max(0, min(n - 1, int((center + o) * vfps))) for o in offsets]
    imgs = []
    for i in pts:
        im = Image.fromarray(vr[i].asnumpy())
        w, h = im.size
        im = im.resize((thumb_w, int(h * thumb_w / w)))
        imgs.append(im)
    H = max(im.height for im in imgs)
    sheet = Image.new("RGB", (thumb_w * 9, H + 50), "white")
    d = ImageDraw.Draw(sheet)
    for i, (im, o) in enumerate(zip(imgs, offsets)):
        sheet.paste(im, (i * thumb_w, 30))
        d.text((i * thumb_w + 4, 4), f"{o:+.1f}s", fill="black")
    d.text((4, H + 34), f"GT in window: {[round(g,2) for g in gt_in_window]}  "
                        f"pred in window: {[round(p,2) for p in pred_in_window]}", fill="darkred")
    return sheet


def make_score_plot(times, prob, center, window_s, gt_in_window, pred_in_window, out_path, title):
    lo, hi = center - window_s, center + window_s
    idx = [i for i, t in enumerate(times) if lo <= t <= hi]
    if not idx:
        return False
    t = [times[i] for i in idx]; p = [prob[i] for i in idx]
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(t, p, color="steelblue", lw=1.5)
    for g in gt_in_window:
        ax.axvline(g, color="green", linestyle="--", alpha=0.8, label="GT")
    for pk in pred_in_window:
        ax.axvline(pk, color="red", linestyle=":", alpha=0.8, label="pred")
    ax.axvline(center, color="gray", linestyle="-", alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for hd, lb in zip(handles, labels):
        seen.setdefault(lb, hd)
    if seen:
        ax.legend(seen.values(), seen.keys(), loc="upper right", fontsize=8)
    ax.set_xlabel("time (s)"); ax.set_ylabel("boundary probability"); ax.set_title(title, fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return True


def make_clip(video_path, center, window_s, out_path):
    if shutil.which("ffmpeg") is None:
        return False
    start = max(0.0, center - window_s)
    duration = 2 * window_s
    cmd = ["ffmpeg", "-y", "-ss", f"{start:.2f}", "-i", video_path, "-t", f"{duration:.2f}",
           "-vf", "scale=480:-2", "-c:v", "libx264", "-preset", "fast", "-an", out_path]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True, help="output of boundary_error_audit.py")
    ap.add_argument("--logits", required=True)
    ap.add_argument("--data", required=True, help="recseg json, for video paths")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_per_category", type=int, default=8)
    ap.add_argument("--per_video_cap", type=int, default=2)
    ap.add_argument("--window_s", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from src.eval.run_manifest import print_manifest_if_exists, write_manifest
    print_manifest_if_exists(a.predictions); print_manifest_if_exists(a.logits)

    preds = [json.loads(l) for l in open(a.predictions)]
    logits_data = {v.get("recording_id", ""): v for v in torch.load(a.logits, weights_only=False)}
    video_path = {r["recording_id"]: r["video"] for r in json.load(open(a.data))}

    media_dir = os.path.join(a.out_dir, "media")
    os.makedirs(media_dir, exist_ok=True)

    # ---------------- build flat event list ----------------
    cands = defaultdict(list)  # category -> [(recording_id, center, gt_list, pred_list, extra)]
    for rec in preds:
        rid = rec["recording_id"]
        gts_all = [g["gt_time"] for g in rec["gt_boundaries"]]
        preds_all = [p["pred_time"] for p in rec["predicted_peaks"]] + \
                    [g["matched_pred_time"] for g in rec["gt_boundaries"] if "matched_pred_time" in g]
        for g in rec["gt_boundaries"]:
            if g["status"] == "missed":
                cat = f"missed_{g['signal']}"
            else:
                cat = g["status"]  # exact / early / late
            cands[cat].append((rid, g["gt_time"], gts_all, preds_all, g))
        for p in rec["predicted_peaks"]:
            cands[p["status"]].append((rid, p["pred_time"], gts_all, preds_all, p))

    print("candidate pool per category:", {k: len(v) for k, v in cands.items()})

    rng = random.Random(a.seed)
    selected = []
    for cat, items in cands.items():
        rng.shuffle(items)
        cap = Counter(); picked = 0
        for rid, center, gts_all, preds_all, extra in items:
            if picked >= a.n_per_category:
                break
            if cap[rid] >= a.per_video_cap:
                continue
            cap[rid] += 1; picked += 1
            selected.append((cat, rid, center, gts_all, preds_all, extra))

    print(f"total events selected: {len(selected)}")

    rows = []
    for cat, rid, center, gts_all, preds_all, extra in selected:
        event_id = f"{rid}_{cat}_t{center:.1f}"
        vp = video_path.get(rid)
        ld = logits_data.get(rid)
        gt_in_window = [g for g in gts_all if abs(g - center) <= a.window_s]
        pred_in_window = [p for p in preds_all if abs(p - center) <= a.window_s]

        clip_path = os.path.join(media_dir, f"{event_id}.mp4")
        contact_path = os.path.join(media_dir, f"{event_id}_contact_sheet.png")
        plot_path = os.path.join(media_dir, f"{event_id}_score_plot.png")

        clip_ok = make_clip(vp, center, a.window_s, clip_path) if vp else False
        contact_ok = False
        if vp:
            vr = VideoReader(vp, num_threads=1)
            sheet = make_contact_sheet(vr, vr.get_avg_fps(), center, a.window_s, gt_in_window, pred_in_window)
            sheet.save(contact_path)
            contact_ok = True
            del vr
        plot_ok = False
        if ld:
            plot_ok = make_score_plot(ld["times"], ld["prob"], center, a.window_s, gt_in_window,
                                      pred_in_window, plot_path, f"{rid} [{cat}] t={center:.1f}s")

        rows.append({"event_id": event_id, "recording_id": rid, "category": cat,
                    "center_time": round(center, 2),
                    "offset": round(extra.get("offset", 0), 3) if "offset" in extra else "",
                    "clip_path": clip_path if clip_ok else "",
                    "contact_sheet_path": contact_path if contact_ok else "",
                    "score_plot_path": plot_path if plot_ok else "",
                    "primary_error_type": "", "secondary_error_type": "",
                    "visual_evidence_present": "", "notes": ""})
        print(f"{event_id}: clip={'ok' if clip_ok else 'SKIPPED (no ffmpeg?)'} "
              f"contact_sheet={'ok' if contact_ok else 'skip'} score_plot={'ok' if plot_ok else 'skip'}")

    csv_path = os.path.join(a.out_dir, "audit_sample.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        w.writeheader()
        for row in rows:
            w.writerow(row)

    print(f"\nwrote {len(rows)} sampled events -> {csv_path}")
    print(f"media -> {media_dir}/")
    write_manifest(csv_path, input_paths=[a.predictions, a.logits, a.data],
                   extra={"n_events": len(rows), "window_s": a.window_s,
                          "n_per_category": a.n_per_category, "per_video_cap": a.per_video_cap})


if __name__ == "__main__":
    main()
