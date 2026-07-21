"""B-final media step (v2) -- for a balanced sample of classified events from
boundary_error_audit.py's predictions.jsonl, render the three artifacts a
human audit actually needs, WITH the segment-name context that v1 was
missing (a category+timestamp alone can't tell you if this is a
"rinse mug -> place mug" transition or a "pick up -> put down" one):

  {event_id}.mp4              : ~3s before/after the event center, with the
                                 segment-label transition burned into the
                                 video via ffmpeg drawtext (if ffmpeg on PATH)
  {event_id}_contact_sheet.png: 9 frames across that window, header shows
                                 Previous/Next (or containing) segment labels
                                 + GT/pred times so you don't have to cross-
                                 reference the CSV while looking at the image
  {event_id}_score_plot.png   : probability curve with GT (green dashed) /
                                 pred (red dashed) marked, title includes the
                                 same label + time context

CSV columns (audit_sample.csv) now include gt_time, pred_time, pred_score,
offset, and segment labels -- for GT-centered events (exact/early/late/
missed_*): prev_segment_label/next_segment_label (the two segments either
side of the GT cut). For prediction-centered false positives (duplicate/
false_near_edge/false_mid_segment/false_gap): containing_segment_label,
nearest_previous_segment_label, nearest_next_segment_label,
nearest_gt_boundary_time, distance_to_nearest_gt -- so you can tell a
false_mid_segment peak apart as "inside 'wipe mug', 1.8s from the nearest GT"
vs "actually sitting near an unlabeled sub-action".

Usage (server, after boundary_error_audit.py's v2 -- run it again first if
predictions.jsonl predates the segment-label fields):
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

GT_CENTERED = {"exact", "early", "late"}  # + "missed_*" handled by prefix


def is_gt_centered(cat):
    return cat in GT_CENTERED or cat.startswith("missed_")


def label_pair_text(row):
    """Short 'A -> B' style label string for titles/captions, from whichever
    fields this row's category populated. If no segment starts exactly at a
    GT-centered event (a real annotation gap, not a bug -- segments in this
    dataset aren't always contiguous), falls back to the NEAREST next
    segment + how many seconds after, so this never dead-ends at '?'."""
    if row["prev_segment_label"] or row["next_segment_label"] or row["nearest_next_label"]:
        a = row["prev_segment_label"] or "?"
        if row["next_segment_label"]:
            b = row["next_segment_label"]
        elif row["nearest_next_label"]:
            b = f"[gap {row['nearest_next_gap_s']}s] {row['nearest_next_label']}"
        else:
            b = "?"
        return f"{a}  ->  {b}"
    if row["containing_segment_label"]:
        a = row["nearest_previous_segment_label"] or "?"
        c = row["containing_segment_label"]
        b = row["nearest_next_segment_label"] or "?"
        return f"{a} | [{c}] | {b}"
    return ""


def ffmpeg_escape(text):
    return text.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


def make_contact_sheet(vr, vfps, center, window_s, gt_in_window, pred_in_window, row, thumb_w=160):
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
    sheet = Image.new("RGB", (thumb_w * 9, H + 70), "white")
    d = ImageDraw.Draw(sheet)
    for i, (im, o) in enumerate(zip(imgs, offsets)):
        sheet.paste(im, (i * thumb_w, 30))
        d.text((i * thumb_w + 4, 4), f"{o:+.1f}s", fill="black")
    d.text((4, H + 34), label_pair_text(row), fill="darkblue")
    gt_str = f"GT={row['gt_time']}" if row["gt_time"] != "" else "GT=n/a"
    pred_str = f"pred={row['pred_time']}" if row["pred_time"] != "" else "pred=n/a"
    d.text((4, H + 54), f"{row['category']}  {gt_str}  {pred_str}  "
                       f"offset={row['offset']}  score={row['pred_score']}", fill="darkred")
    return sheet


def make_score_plot(times, prob, center, window_s, gt_in_window, pred_in_window, out_path, row):
    lo, hi = center - window_s, center + window_s
    idx = [i for i, t in enumerate(times) if lo <= t <= hi]
    if not idx:
        return False
    t = [times[i] for i in idx]; p = [prob[i] for i in idx]
    fig, ax = plt.subplots(figsize=(6, 3.3))
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
    ax.set_xlabel("time (s)"); ax.set_ylabel("boundary probability")
    title1 = f"{row['recording_id']} [{row['category']}]"
    title2 = label_pair_text(row)
    ax.set_title(f"{title1}\n{title2}" if title2 else title1, fontsize=8)
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return True


def _run_ffmpeg(cmd):
    """Returns (ok, stderr_tail) -- never silently swallows the real error."""
    try:
        r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if r.returncode == 0:
            return True, ""
        return False, r.stderr.decode(errors="replace")[-500:]
    except FileNotFoundError as e:
        return False, str(e)


def make_clip(video_path, center, window_s, out_path, caption, ffmpeg_bin="ffmpeg"):
    if ffmpeg_bin == "ffmpeg" and shutil.which("ffmpeg") is None:
        print("    [make_clip] ffmpeg not found on PATH")
        return False
    start = max(0.0, center - window_s)
    duration = 2 * window_s
    base_cmd = [ffmpeg_bin, "-y", "-ss", f"{start:.2f}", "-i", video_path, "-t", f"{duration:.2f}"]
    tail = ["-c:v", "libx264", "-preset", "fast", "-an", out_path]

    if caption:
        # drawtext needs a font -- fails hard on minimal servers with no
        # fontconfig/font files installed. Try it, but fall back to a plain
        # (uncaptioned) clip rather than skipping the clip entirely.
        vf_captioned = (f"scale=480:-2,drawtext=text='{ffmpeg_escape(caption)}':"
                       f"x=8:y=8:fontsize=14:fontcolor=yellow:box=1:boxcolor=black@0.5")
        ok, err = _run_ffmpeg(base_cmd + ["-vf", vf_captioned] + tail)
        if ok:
            return True
        print(f"    [make_clip] drawtext failed (likely missing font), "
              f"falling back to uncaptioned clip. ffmpeg stderr tail:\n{err}")

    ok, err = _run_ffmpeg(base_cmd + ["-vf", "scale=480:-2"] + tail)
    if not ok:
        print(f"    [make_clip] ffmpeg failed:\n{err}")
    return ok


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
    ap.add_argument("--ffmpeg_bin", default="ffmpeg",
                     help="explicit path to an ffmpeg binary with drawtext "
                          "support (e.g. /usr/bin/ffmpeg), to bypass PATH "
                          "resolution finding a different ffmpeg first "
                          "(common when a static/minimal build shadows a "
                          "full apt-installed one earlier in PATH)")
    a = ap.parse_args()

    from src.eval.run_manifest import print_manifest_if_exists, write_manifest
    print_manifest_if_exists(a.predictions); print_manifest_if_exists(a.logits)

    preds = [json.loads(l) for l in open(a.predictions)]
    logits_data = {v.get("recording_id", ""): v for v in torch.load(a.logits, weights_only=False)}
    video_path = {r["recording_id"]: r["video"] for r in json.load(open(a.data))}

    media_dir = os.path.join(a.out_dir, "media")
    os.makedirs(media_dir, exist_ok=True)

    if preds and preds[0]["gt_boundaries"] and "prev_segment_label" not in preds[0]["gt_boundaries"][0]:
        print("WARNING: predictions.jsonl looks like it predates the segment-label "
              "fields -- re-run boundary_error_audit.py first.")

    # ---------------- build flat event list ----------------
    cands = defaultdict(list)  # category -> [(recording_id, center, gts_all, preds_all, extra)]
    for rec in preds:
        rid = rec["recording_id"]
        gts_all = [g["gt_time"] for g in rec["gt_boundaries"]]
        preds_all = [p["pred_time"] for p in rec["predicted_peaks"]] + \
                    [g["matched_pred_time"] for g in rec["gt_boundaries"] if "matched_pred_time" in g]
        for g in rec["gt_boundaries"]:
            cat = f"missed_{g['signal']}" if g["status"] == "missed" else g["status"]
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

    header = ["event_id", "recording_id", "category", "gt_time", "pred_time", "pred_score", "offset",
             "prev_segment_label", "next_segment_label", "next_label_is_gap",
             "nearest_next_label", "nearest_next_gap_s",
             "containing_segment_label", "nearest_previous_segment_label", "nearest_next_segment_label",
             "nearest_gt_boundary_time", "distance_to_nearest_gt",
             "clip_path", "contact_sheet_path", "score_plot_path",
             "primary_error_type", "secondary_error_type", "visual_evidence_present", "notes"]
    rows = []
    for cat, rid, center, gts_all, preds_all, extra in selected:
        event_id = f"{rid}_{cat}_t{center:.1f}"
        vp = video_path.get(rid)
        ld = logits_data.get(rid)
        gt_in_window = [g for g in gts_all if abs(g - center) <= a.window_s]
        pred_in_window = [p for p in preds_all if abs(p - center) <= a.window_s]

        row = {h: "" for h in header}
        row.update({"event_id": event_id, "recording_id": rid, "category": cat})
        if is_gt_centered(cat):
            row["gt_time"] = round(center, 3)
            row["pred_time"] = extra.get("matched_pred_time", "")
            row["pred_score"] = extra.get("pred_score", extra.get("gt_prob_at_time", ""))
            row["offset"] = extra.get("offset", "")
            row["prev_segment_label"] = extra.get("prev_segment_label", "") or ""
            row["next_segment_label"] = extra.get("next_segment_label", "") or ""
            row["next_label_is_gap"] = extra.get("next_label_is_gap", "")
            row["nearest_next_label"] = extra.get("nearest_next_label", "") or ""
            row["nearest_next_gap_s"] = extra.get("nearest_next_gap_s", "")
        else:
            row["pred_time"] = round(center, 3)
            row["pred_score"] = extra.get("pred_score", "")
            row["gt_time"] = extra.get("nearest_gt_boundary_time", extra.get("near_gt_time", ""))
            row["distance_to_nearest_gt"] = extra.get("distance_to_nearest_gt", "")
            row["nearest_gt_boundary_time"] = extra.get("nearest_gt_boundary_time", extra.get("near_gt_time", ""))
            row["containing_segment_label"] = extra.get("containing_segment_label", "") or ""
            row["nearest_previous_segment_label"] = extra.get("nearest_previous_segment_label", "") or ""
            row["nearest_next_segment_label"] = extra.get("nearest_next_segment_label", "") or ""

        caption = f"{cat}"
        lp = label_pair_text(row)
        if lp:
            caption += f" | {lp}"
        if row["gt_time"] != "":
            caption += f" | GT={row['gt_time']}"
        if row["pred_time"] != "":
            caption += f" | Pred={row['pred_time']}"

        clip_path = os.path.join(media_dir, f"{event_id}.mp4")
        contact_path = os.path.join(media_dir, f"{event_id}_contact_sheet.png")
        plot_path = os.path.join(media_dir, f"{event_id}_score_plot.png")

        clip_ok = make_clip(vp, center, a.window_s, clip_path, caption, a.ffmpeg_bin) if vp else False
        contact_ok = False
        if vp:
            vr = VideoReader(vp, num_threads=1)
            sheet = make_contact_sheet(vr, vr.get_avg_fps(), center, a.window_s, gt_in_window, pred_in_window, row)
            sheet.save(contact_path)
            contact_ok = True
            del vr
        plot_ok = False
        if ld:
            plot_ok = make_score_plot(ld["times"], ld["prob"], center, a.window_s, gt_in_window,
                                      pred_in_window, plot_path, row)

        row["clip_path"] = clip_path if clip_ok else ""
        row["contact_sheet_path"] = contact_path if contact_ok else ""
        row["score_plot_path"] = plot_path if plot_ok else ""
        rows.append(row)
        print(f"{event_id}: {lp or '(no label context)'}  "
              f"clip={'ok' if clip_ok else 'FAILED (see error above)'} "
              f"contact_sheet={'ok' if contact_ok else 'skip'} score_plot={'ok' if plot_ok else 'skip'}")

    csv_path = os.path.join(a.out_dir, "audit_sample.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
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
