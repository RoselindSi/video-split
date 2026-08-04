"""C3-0: read-only audit of source video resolution, before any C3-lite code
is written. Answers three questions that gate the whole "should we switch to
1080p" decision, none of which need a model or a GPU:

  1. What resolution are the videos we ACTUALLY extracted features from, and
     what does the model actually see after preprocessing? The first run
     answered this and overturned the premise: the sources are 1280x480, not
     720p, and reach the model at 1260x448 -- a 0.984x linear scale, i.e. the
     pixel budget is already almost entirely spent. Switching to 1920x1080
     would give 1008x560: MORE vertical resolution but LESS horizontal. It is
     not a uniform upgrade and has to be compared per axis.
  1b. What else is sitting in the same recording directory? The real files
     turned out to be .../recordings/recording_XXXXXX/mid.mp4 -- a name that
     implies siblings (other views, other resolutions). A higher-resolution
     original is far likelier to be next to the file we already use than to be
     found by guessing a root and a filename token, which is what the first
     run did, finding nothing.
  1c. Is the frame TWO VIEWS packed into one image? A 1280x480 frame at an 8:3
     aspect is exactly what a 640x480 stereo pair packed side-by-side looks
     like. If it is, the hand's effective resolution is half what the frame
     size suggests and half of every ViT's tokens re-encode the same scene
     from a shifted viewpoint -- which would make splitting the frame a bigger
     and far cheaper win than any resolution change.
  2. Do higher-resolution counterparts exist for the recordings that matter
     (the clean-145 dev recordings), and for how many of them? A source that
     covers 12 of 46 dev recordings cannot support a paired 720p-vs-1080p
     ablation, however good those 12 are.
  3. Is a "1080p" file genuinely native, or a software upscale of the same
     720p master? An upscale adds no real detail, so re-extracting from one
     spends the entire extraction budget for nothing. The decisive test is
     downscale-PSNR: scale the candidate back to the current file's resolution
     and measure PSNR against it. Undoing an upscale reconstructs the original
     almost exactly (>40 dB); an independent master does not. Checked against a
     fixture with known ground truth this separated the two cases by 21 dB
     (21.95 vs 43.37), whereas the cheaper bits-per-pixel ratio did not
     separate them at all (0.69 vs 0.64) and is therefore reported for context
     only, deciding nothing.

Why this runs BEFORE any C3 model code: the global feature branch provably
gains nothing from 1080p. extract_features_recseg.py passes max_pixels =
768*28*28 = 602112 to the processor, and Qwen's smart_resize maps BOTH
1280x720 and 1920x1080 to exactly 1008x560 -- the identical model input
tensor. A hand occupying 10% of frame width arrives as 101 px in both cases.
So the entire value of 1080p lies in the LOCAL crop branch, which crops from
native pixels before any resize, and the only thing worth auditing is whether
native pixels exist to crop from.

Usage (server, read-only -- probes files, writes nothing but its own report):
    python -m src.boundary.c3_source_audit \
        --data /workspace/tr1/data_recseg/recseg_train.json \
        --data /workspace/tr1/data_recseg/recseg_val.json \
        --data /workspace/tr1/data_recseg_part2/recseg_train.json \
        --data /workspace/tr1/data_recseg_part2/recseg_val.json \
        --recordings_from /workspace/tr1/results/hal/slow_latent_c2/events_v2_final.csv \
        --search_root /storage \
        --out /workspace/tr1/results/hal/c3/source_audit.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
from collections import Counter

FACTOR = 28
DEFAULT_MAX_PIXELS = 768 * 28 * 28


def smart_resize(h, w, factor=FACTOR, max_pixels=DEFAULT_MAX_PIXELS):
    """Qwen-VL's preprocessing resize, replicated so the audit can state what
    the model actually sees rather than what the file contains."""
    h_bar = round(h / factor) * factor
    w_bar = round(w / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((h * w) / max_pixels)
        h_bar = math.floor(h / beta / factor) * factor
        w_bar = math.floor(w / beta / factor) * factor
    return w_bar, h_bar


def probe(path, ffprobe="ffprobe"):
    """ffprobe -> dict, or {"error": ...}. Never raises: a missing file or a
    missing ffprobe must degrade the report, not abort a 400-file audit."""
    if not os.path.exists(path):
        return {"error": "missing"}
    try:
        out = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,codec_name,avg_frame_rate,bit_rate,nb_frames",
             "-show_entries", "format=duration,bit_rate,size",
             "-of", "json", path],
            capture_output=True, text=True, timeout=60)
        if out.returncode != 0:
            return {"error": (out.stderr or "ffprobe failed").strip()[:200]}
        j = json.loads(out.stdout)
        st = (j.get("streams") or [{}])[0]
        fm = j.get("format") or {}
        def _i(v):
            try:
                return int(v)
            except (TypeError, ValueError):
                return None
        def _f(v):
            try:
                return float(v)
            except (TypeError, ValueError):
                return None
        fps = None
        afr = st.get("avg_frame_rate") or ""
        if "/" in afr:
            n, d = afr.split("/", 1)
            fps = (float(n) / float(d)) if _f(d) else None
        w, h = _i(st.get("width")), _i(st.get("height"))
        size = _i(fm.get("size"))
        dur = _f(fm.get("duration"))
        br = _i(st.get("bit_rate")) or _i(fm.get("bit_rate"))
        # bits per pixel per frame. Intuitively an upscale should show a much
        # lower bpp than its source, but measured against a known-ground-truth
        # fixture this did NOT hold well enough to use (0.69 for a genuine
        # 1080p render vs 0.64 for a pure upscale): encoder rate control
        # confounds it. Kept as context; downscale_psnr() is the real test.
        bpp = (br / (w * h * fps)) if (br and w and h and fps) else None
        return {"width": w, "height": h, "codec": st.get("codec_name"),
                "fps": fps, "duration_s": dur, "bit_rate": br,
                "size_bytes": size, "bits_per_pixel": bpp}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"[:200]}


def downscale_psnr(candidate, current, w, h, duration=None, t0=5.0, dur=2.0, ffmpeg="ffmpeg"):
    """Downscale `candidate` to the current file's WxH and measure PSNR against
    `current` over a short window. This is the decisive native-vs-upscale test,
    and it exists because the cheaper bits-per-pixel heuristic did not
    discriminate when checked against a fixture with known ground truth
    (0.69 for a genuine 1080p render vs 0.64 for a pure upscale -- unusable).

    If the "1080p" file was produced BY upscaling this very 720p file,
    downscaling it undoes that operation and returns almost exactly the
    original: PSNR is limited only by resampling and re-encode loss, and lands
    high (typically >40 dB). Two genuinely independent encodes of the same
    scene at different resolutions do not reconstruct each other that closely.

    Returns (psnr_db, None) or (None, reason). Requires the two files to be
    time-aligned; a large negative result may mean misalignment rather than an
    independent master, so a LOW value is weak evidence while a HIGH value is
    strong evidence of an upscale."""
    if not (w and h):
        return None, "unknown current resolution"
    # Seek adaptively: a fixed 5s offset silently yields ZERO frames (and a
    # meaningless "no psnr line") on any clip shorter than that, which is a
    # failure that looks like a parsing problem rather than a seek problem.
    if duration:
        t0 = min(t0, max(0.0, duration * 0.25))
        dur = min(dur, max(0.5, duration - t0))
    try:
        out = subprocess.run(
            # -v info (not error): the psnr filter prints its summary line at
            # INFO level, so -v error silently discards the only output we want.
            [ffmpeg, "-v", "info", "-ss", str(t0), "-t", str(dur), "-i", candidate,
             "-ss", str(t0), "-t", str(dur), "-i", current,
             "-lavfi", f"[0:v]scale={w}:{h},setpts=PTS-STARTPTS[a];"
                       f"[1:v]setpts=PTS-STARTPTS[b];[a][b]psnr",
             "-f", "null", "-"],
            capture_output=True, text=True, timeout=180)
        for line in (out.stderr or "").splitlines():
            if "average:" in line and "psnr" in line.lower():
                for tok in line.split():
                    if tok.startswith("average:"):
                        v = tok.split(":", 1)[1]
                        return (None, "inf (identical)") if v == "inf" else (float(v), None)
        if "frame=    0" in (out.stderr or "") or "frame=0" in (out.stderr or ""):
            return None, "0 frames decoded (seek past end, or unreadable stream)"
        return None, ((out.stderr or "no psnr line").strip().splitlines() or ["?"])[-1][:120]
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"[:120]


VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".avi", ".webm")


def scan_siblings(video, exts=VIDEO_EXTS):
    """Every other video file sitting in the same directory as `video`.

    The first audit run searched a token ('1080') under a guessed root and
    found nothing, while the real data turned out to live somewhere else
    entirely under a per-recording directory whose file was named 'mid.mp4'.
    A name like that implies siblings (other views, other resolutions), and a
    higher-resolution original is far more likely to be sitting next to the
    file we already use than to be found by guessing a root and a filename
    token. This looks where we already know the data is."""
    d = os.path.dirname(video)
    if not os.path.isdir(d):
        return []
    out = []
    for fn in sorted(os.listdir(d)):
        p = os.path.join(d, fn)
        if p != video and os.path.splitext(fn)[1].lower() in exts and os.path.isfile(p):
            out.append(p)
    return out


def _frame_array(path, t0=5.0, duration=None, ffmpeg="ffmpeg"):
    """Decode a single frame to a numpy array (H,W,3), or None."""
    import tempfile
    import numpy as np
    if duration:
        t0 = min(t0, max(0.0, duration * 0.25))
    with tempfile.TemporaryDirectory() as td:
        png = os.path.join(td, "f.png")
        try:
            r = subprocess.run([ffmpeg, "-v", "error", "-ss", str(t0), "-i", path,
                                "-frames:v", "1", "-y", png],
                               capture_output=True, text=True, timeout=120)
            if r.returncode != 0 or not os.path.exists(png):
                return None
            from PIL import Image
            return np.asarray(Image.open(png).convert("RGB"), dtype=float)
        except Exception:
            return None


def stereo_check(path, t0=5.0, duration=None, ffmpeg="ffmpeg"):
    """Is this frame two views packed into one image (side-by-side or stacked)?

    A 1280x480 frame at an 8:3 aspect is exactly what a 640x480 stereo pair
    packed side-by-side looks like, and that would matter a great deal: the
    hand's effective resolution would be half of what the frame size suggests,
    and half of every ViT's tokens would be spent re-encoding the same scene
    from a slightly shifted viewpoint.

    Two views of one scene correlate strongly at some small horizontal
    disparity, while the two halves of an ordinary wide photograph do not.
    So: split, then take the maximum normalized cross-correlation over a range
    of horizontal shifts. Reported for both the side-by-side and the
    top/bottom split, since either packing is possible."""
    import numpy as np
    a = _frame_array(path, t0, duration, ffmpeg)
    if a is None:
        return {"error": "could not decode a frame"}
    g = a.mean(axis=2)
    h, w = g.shape

    def ncc(x, y):
        x = x - x.mean(); y = y - y.mean()
        d = float(np.sqrt((x * x).sum() * (y * y).sum()))
        return float((x * y).sum() / d) if d > 0 else 0.0

    def best_shift_ncc(l, r, max_shift):
        best, arg = -1.0, 0
        for s in range(0, max_shift + 1, max(1, max_shift // 24)):
            v = ncc(l[:, s:], r[:, :r.shape[1] - s]) if s else ncc(l, r)
            if v > best:
                best, arg = v, s
        return best, arg

    out = {"width": w, "height": h, "aspect": w / h if h else None}
    lw = w // 2
    sb, sb_shift = best_shift_ncc(g[:, :lw], g[:, lw:lw * 2], max_shift=max(4, lw // 8))
    out["side_by_side_ncc"] = sb
    out["side_by_side_best_shift_px"] = sb_shift
    hh = h // 2
    tb = ncc(g[:hh, :], g[hh:hh * 2, :])
    out["top_bottom_ncc"] = tb
    # Which split to trust is decided by the frame geometry, not by taking the
    # max of the two. A horizontally-packed fixture with known ground truth
    # scored 0.930 on the TOP/BOTTOM test purely because its content repeats
    # vertically, so an OR over both tests false-positives on anything with a
    # horizon, a work surface, or repeating texture. A frame twice as wide as
    # it is tall can only plausibly be side-by-side packing, so that is the
    # test that decides; top/bottom is still reported, as context.
    out["decided_by"] = "side_by_side" if out["aspect"] and out["aspect"] >= 2.0 else "either"
    out["likely_packed_stereo"] = bool(
        sb >= 0.80 if out["decided_by"] == "side_by_side" else (sb >= 0.80 or tb >= 0.80))
    return out


def find_candidates(search_root, token="1080", exts=VIDEO_EXTS):
    """Walk search_root for video files whose NAME contains `token`. Returns
    the list of paths plus a directory histogram, so an unmatched result can
    be diagnosed (wrong root vs wrong naming) instead of just coming back
    empty."""
    hits, dirs = [], Counter()
    if not os.path.isdir(search_root):
        return hits, dirs, f"search_root {search_root!r} is not a directory"
    for dirpath, dirnames, filenames in os.walk(search_root, onerror=lambda e: None):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for fn in filenames:
            if token in fn and os.path.splitext(fn)[1].lower() in exts:
                hits.append(os.path.join(dirpath, fn))
                dirs[dirpath] += 1
    return hits, dirs, None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", action="append", required=True,
                    help="recseg json(s) giving recording_id -> video path")
    ap.add_argument("--recordings_from",
                    help="CSV with a recording_id column (e.g. a slow_latent_c2 "
                         "--dump_events file) to restrict the coverage report to "
                         "the recordings that actually carry dev events")
    ap.add_argument("--search_root", default="/storage",
                    help="where to look for higher-resolution counterparts")
    ap.add_argument("--token", default="1080", help="filename token identifying them")
    ap.add_argument("--max_pixels", type=int, default=DEFAULT_MAX_PIXELS)
    ap.add_argument("--ffprobe", default="ffprobe")
    ap.add_argument("--ffmpeg", default="ffmpeg")
    ap.add_argument("--psnr_limit", type=int, default=6,
                    help="run the decisive downscale-PSNR test on this many pairs "
                         "(a few seconds each; a handful is enough to tell an "
                         "upscaled batch from a genuine one)")
    ap.add_argument("--probe_limit", type=int, default=40,
                    help="probe at most this many of each group (ffprobe is ~50ms "
                         "per file but there may be thousands)")
    ap.add_argument("--out")
    a = ap.parse_args()

    video_path = {}
    for path in a.data:
        with open(path, encoding="utf-8") as f:
            for r in json.load(f):
                video_path[r["recording_id"]] = r["video"]
    print(f"recordings in --data: {len(video_path)}")

    dev_recs = None
    if a.recordings_from:
        dev_recs = set()
        with open(a.recordings_from, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("recording_id"):
                    dev_recs.add(row["recording_id"])
        print(f"dev recordings from --recordings_from: {len(dev_recs)}")
        missing = sorted(dev_recs - set(video_path))
        if missing:
            print(f"  !! {len(missing)} dev recordings have no video path in --data: "
                  f"{missing[:5]}")

    # ---- 1. what are we actually using -----------------------------------
    targets = sorted(dev_recs & set(video_path)) if dev_recs else sorted(video_path)
    probe_ids = targets[:a.probe_limit]
    print(f"\n=== 1. resolution of the videos we extract features from "
          f"(probing {len(probe_ids)} of {len(targets)}) ===")
    res_hist, current = Counter(), {}
    for rid in probe_ids:
        info = probe(video_path[rid], a.ffprobe)
        current[rid] = info
        if "error" in info:
            res_hist[f"ERROR:{info['error'][:40]}"] += 1
        else:
            res_hist[f"{info['width']}x{info['height']}"] += 1
    for k, v in res_hist.most_common():
        print(f"  {k:<28} {v:>4}")
    ok = [i for i in current.values() if "error" not in i]
    if ok:
        w, h = ok[0]["width"], ok[0]["height"]
        ow, oh = smart_resize(h, w, max_pixels=a.max_pixels)
        print(f"\n  a {w}x{h} frame reaches the model as {ow}x{oh} "
              f"(max_pixels={a.max_pixels})")
        ow2, oh2 = smart_resize(1080, 1920, max_pixels=a.max_pixels)
        print(f"  a 1920x1080 frame reaches the model as {ow2}x{oh2}")
        if (ow, oh) == (ow2, oh2):
            print("  -> IDENTICAL model input. The GLOBAL branch gains nothing "
                  "from 1080p; only a native-resolution LOCAL crop can use it.")
        else:
            print(f"  -> current linear scale {ow / w:.3f}x, 1080p would be "
                  f"{ow2 / 1920:.3f}x. The pixel budget is already nearly spent "
                  f"on the current frame, so 'switch to 1080p' is not automatically "
                  f"an increase in what the model sees -- compare {ow}x{oh} against "
                  f"{ow2}x{oh2} directly, per axis.")
        if h and (w / h) > 2.2:
            print(f"  !! aspect ratio {w / h:.2f} is far wider than a normal camera "
                  f"frame -- section 1c checks whether this is two views packed "
                  f"into one image")

    # ---- 1b. what else is sitting next to the file we use ----------------
    print("\n=== 1b. sibling video files in the same recording directory ===")
    sib_hist, siblings = Counter(), {}
    for rid in probe_ids:
        sibs = scan_siblings(video_path[rid])
        siblings[rid] = sibs
        sib_hist[tuple(os.path.basename(s) for s in sibs)] += 1
    for names, n in sib_hist.most_common(6):
        print(f"  {n:>4} recordings have siblings: {list(names) or '(none)'}")
    probed_sibs = {}
    for rid in probe_ids[:4]:
        for s in siblings.get(rid, []):
            info = probe(s, a.ffprobe)
            probed_sibs[s] = info
            cur = current.get(rid, {})
            bigger = (info.get("width") and cur.get("width")
                      and info["width"] * info["height"] > cur["width"] * cur["height"])
            print(f"    {os.path.basename(s):<24} "
                  f"{info.get('width')}x{info.get('height')} "
                  f"{info.get('codec')} "
                  f"{'<-- HIGHER RESOLUTION than the file we use' if bigger else ''}")
    if any(probe(s, a.ffprobe).get("width", 0) or 0 for rid in probe_ids[:1]
           for s in siblings.get(rid, [])[:1]):
        print("  -> if a sibling is higher-resolution, THAT is the source to test, "
              "not a file found by guessing a root and a filename token")

    # ---- 1c. is the frame two views packed into one ----------------------
    print("\n=== 1c. packed-stereo check (are the two halves the same scene?) ===")
    stereo = {}
    for rid in probe_ids[:3]:
        st = stereo_check(video_path[rid], duration=current.get(rid, {}).get("duration_s"),
                          ffmpeg=a.ffmpeg)
        stereo[rid] = st
        if "error" in st:
            print(f"  {rid}: {st['error']}")
            continue
        print(f"  {rid}: {st['width']}x{st['height']} aspect {st['aspect']:.2f}  "
              f"side-by-side NCC {st['side_by_side_ncc']:+.3f} "
              f"(best shift {st['side_by_side_best_shift_px']}px)  "
              f"top/bottom NCC {st['top_bottom_ncc']:+.3f}  "
              f"[decided by {st['decided_by']}; top/bottom is inflated by any "
              f"vertically repeating content and does not decide a wide frame]")
    packed = [s for s in stereo.values() if s.get("likely_packed_stereo")]
    if packed:
        print("  !! LIKELY TWO VIEWS PACKED INTO ONE FRAME. If so the hand's real "
              "resolution is half the frame width suggests, and half of every "
              "ViT's tokens re-encode the same scene from a shifted viewpoint. "
              "Splitting the frame and using ONE view would be a bigger and far "
              "cheaper win than any resolution change. Confirm by eye on one "
              "extracted frame before acting.")
    elif stereo and not any("error" in s for s in stereo.values()):
        print("  -> halves are not duplicates: this is a genuinely wide single "
              "view, and the unusual aspect ratio is the camera's, not a packing "
              "artifact.")

    # ---- 2. do higher-resolution sources exist ---------------------------
    print(f"\n=== 2. candidates under {a.search_root} with {a.token!r} in the name ===")
    hits, dirs, err = find_candidates(a.search_root, a.token)
    if err:
        print(f"  !! {err}")
    print(f"  found {len(hits)} files in {len(dirs)} directories")
    for d, n in dirs.most_common(10):
        print(f"    {n:>5}  {d}")
    for p in hits[:5]:
        print(f"    e.g. {p}")

    # match by recording_id appearing in the path -- reported, not assumed:
    # if this yields nothing the naming convention differs and the printed
    # examples above are what to write the real rule against.
    matched, unmatched = {}, []
    for rid in targets:
        m = [p for p in hits if rid in p]
        if m:
            matched[rid] = sorted(m)[0]
        else:
            unmatched.append(rid)
    print(f"\n  matched by recording_id in path: {len(matched)}/{len(targets)} "
          f"target recordings")
    if unmatched:
        print(f"  unmatched examples: {unmatched[:5]}")
        print(f"  their current paths: {[video_path[r] for r in unmatched[:2]]}")
    if len(matched) < len(targets):
        print("  -> if this coverage is low, compare the example paths above "
              "against the current paths and adjust the match rule; a partial "
              "match set cannot support a paired 720p-vs-1080p ablation.")

    # ---- 3. native or upscaled ------------------------------------------
    print(f"\n=== 3. native vs upscaled (probing up to {a.probe_limit} matches) ===")
    pairs = []
    for rid in sorted(matched)[:a.probe_limit]:
        hi = probe(matched[rid], a.ffprobe)
        lo = current.get(rid) or probe(video_path[rid], a.ffprobe)
        if "error" in hi or "error" in lo:
            continue
        pairs.append({"recording_id": rid, "current": lo, "candidate": hi,
                      "candidate_path": matched[rid]})
    if not pairs:
        print("  no probeable matched pairs")
    else:
        print(f"{'recording':<22} {'current':>11} {'cand':>11} {'cur bpp':>9} "
              f"{'cand bpp':>9} {'ratio':>7}")
        for p in pairs[:15]:
            c, d = p["current"], p["candidate"]
            cb, db = c.get("bits_per_pixel"), d.get("bits_per_pixel")
            ratio = f"{db / cb:.2f}" if (cb and db) else "n/a"
            print(f"{p['recording_id']:<22} {c['width']}x{c['height']:<6} "
                  f"{d['width']}x{d['height']:<6} "
                  f"{(f'{cb:.4f}' if cb else 'n/a'):>9} "
                  f"{(f'{db:.4f}' if db else 'n/a'):>9} {ratio:>7}")
        print("  (bits-per-pixel is shown for context only -- checked against a "
              "fixture with known ground truth it did NOT separate a genuine "
              "1080p render from a pure upscale, so it decides nothing)")

        print(f"\n  downscale-PSNR test on up to {a.psnr_limit} pairs "
              f"(high dB => the candidate is an upscale of the current file):")
        for p in pairs[:a.psnr_limit]:
            db, why = downscale_psnr(p["candidate_path"], video_path[p["recording_id"]],
                                     p["current"]["width"], p["current"]["height"],
                                     duration=p["current"].get("duration_s"),
                                     ffmpeg=a.ffmpeg)
            p["downscale_psnr_db"] = db
            p["downscale_psnr_note"] = why
            if db is None:
                print(f"    {p['recording_id']:<22} n/a  ({why})")
            else:
                verdict = ("UPSCALE of the current file" if db >= 40 else
                           "likely independent master" if db < 30 else
                           "ambiguous")
                print(f"    {p['recording_id']:<22} {db:>6.2f} dB   {verdict}")
        dbs = [p["downscale_psnr_db"] for p in pairs[:a.psnr_limit]
               if p.get("downscale_psnr_db") is not None]
        if dbs:
            med = sorted(dbs)[len(dbs) // 2]
            print(f"\n  median PSNR {med:.2f} dB")
            if med >= 40:
                print("  -> the 1080p files reconstruct the current 720p files almost "
                      "exactly. They are upscales and carry NO additional real detail: "
                      "re-extracting from them would cost the full extraction budget "
                      "for nothing. Do not switch.")
            elif med < 30:
                print("  -> the 1080p files do not reconstruct the current files, so "
                      "they are plausibly independent higher-resolution masters and "
                      "DO carry extra detail for a native-resolution local crop. "
                      "Confirm on one clip by eye before committing to re-extraction "
                      "(a low value can also mean the two files are not time-aligned).")
            else:
                print("  -> ambiguous. Check time alignment first, then compare one "
                      "pair of frames by eye at 1:1 zoom on a hand.")

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump({
                "max_pixels": a.max_pixels,
                "model_input_for_720p": smart_resize(720, 1280, max_pixels=a.max_pixels),
                "model_input_for_1080p": smart_resize(1080, 1920, max_pixels=a.max_pixels),
                "n_recordings_in_data": len(video_path),
                "n_target_recordings": len(targets),
                "current_resolution_histogram": dict(res_hist),
                "current_probe": current,
                "siblings": {k: v for k, v in siblings.items()},
                "probed_siblings": probed_sibs,
                "stereo_check": stereo,
                "n_candidates_found": len(hits),
                "candidate_dirs": dict(dirs.most_common(20)),
                "candidate_examples": hits[:20],
                "matched": matched, "unmatched": unmatched,
                "pairs": pairs,
            }, f, ensure_ascii=False, indent=2)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
