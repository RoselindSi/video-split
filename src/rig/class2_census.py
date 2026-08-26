"""How often does anyone ELSE's hand enter the workspace?

This is a prevalence measurement, and it exists because the segmentation head
splits into two very unequal problems. Telling hands from the conveyor is easy
and RGB does it almost for free. Telling the WEARER'S hand from a
COLLEAGUE'S is the actual task, and it can only be learned from frames that
contain a colleague. One recording sampled 35 ways contained exactly one such
frame. If that rate holds, a uniformly sampled 300-frame annotation budget
buys about nine examples of the thing being learned, and the head will
correctly conclude that class 2 does not exist.

So the number is measured before the budget is spent, not after.

NO CALIBRATION IS NEEDED AND THAT IS THE POINT. Only 15 of 40 databags carry
real calibration, so anything routed through the wide render can see barely a
third of the corpus -- and prevalence measured on a third that was selected by
whether someone calibrated it is not the corpus prevalence. Instead each
sampled instant is shown as the three modules' left eyes laid side by side,
which is the full 176 degree field in raw pixels. Every databag qualifies.

SAMPLING IS STRATIFIED RANDOM, NOT EVENLY SPACED. The work here is a
repeating assembly cycle of a few seconds. Evenly spaced samples can land on
the same phase of that cycle every time and report a confident number about
one moment of the task. The recording is cut into equal blocks and one random
frame is drawn from each, which spreads coverage without locking to a period.
The seed comes from the recording's own name, so the same recording always
yields the same frames and a rerun is a rerun rather than a new sample.

AND IT IS BLIND. No detector, no depth rule, no score orders these frames.
That matters more here than anywhere else in the pipeline: a mined sample can
train a head, but only a blind sample can say what the deployment rate is, and
mixing the two is how a dev set comes to disagree with reality. Mining comes
later and writes to a different file.

WHAT COMES BACK is a contact sheet per recording plus a CSV with one row per
sampled instant, for a human to fill in. The tool cannot label these itself --
that is the whole reason the question is open.
"""
from __future__ import annotations

import csv
import os
import zlib
from pathlib import Path

import numpy as np

# One row of the sheet is one instant, shown across all three modules.
VIEWS = ("cam12", "cam34", "cam56")

# Six instants per sheet. A second person is a large feature, but the sheet is
# downscaled to be looked at, and more rows means smaller tiles; six keeps each
# view readable at a glance.
ROWS_PER_SHEET = 6
TILE = 400

# Refuse to write onto a filesystem this full. The shared machine's root has
# been filled once already, and a full disk shows up in other people's
# programs as a write error rather than as a disk error.
MIN_FREE_GB = 20.0

# The taxonomy a reviewer fills in. `other_near` is the class the head needs;
# `other_far` is a person across the line whose hands never reach the
# workspace, which is a different and much easier case, and lumping the two
# together would inflate exactly the number this tool exists to measure.
LABELS = ("owner_only", "other_near", "other_far", "none", "transit")


def find_recordings(root):
    """-> [(recording_id, {view: path})], sorted. A databag is a directory
    holding cam*.mp4; nothing about the layout above that is assumed."""
    root = Path(root)
    found = {}
    for p in sorted(root.rglob("*.mp4")):
        stem = p.stem.lower()
        for v in VIEWS:
            if v in stem:
                found.setdefault(str(p.parent), {})[v] = str(p)
                break
    out = []
    for d, views in sorted(found.items()):
        out.append((Path(d).name or d, views, d))
    return out


def frame_count(views):
    """Frames in the longest module of a recording. Metadata only, no decode."""
    import cv2
    n = 0
    for v in VIEWS:
        if v in views:
            cap = cv2.VideoCapture(views[v])
            n = max(n, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            cap.release()
    return n


def allocate(lengths, total):
    """Instants per recording, proportional to length. -> [int], summing to
    `total` (or to the number of frames, if that is smaller).

    THE CORPUS IS NOT A LIST OF EQUAL RECORDINGS. These 40 databags span three
    orders of magnitude, from a few seconds to over an hour, and about a dozen
    of them together are under half a percent of the footage. Giving each
    recording the same number of instants would spend a third of the budget on
    material that is a rounding error in deployment, and would report a
    prevalence for a corpus nobody has.

    The question is what fraction of the FRAMES a colleague appears in, so the
    frames are what gets sampled: a recording holding 18% of the footage draws
    18% of the instants and one holding 0.03% draws none. Largest-remainder
    apportionment, so the parts sum to the whole instead of drifting by the
    rounding."""
    tot = float(sum(lengths))
    if tot <= 0:
        return [0] * len(lengths)
    exact = [total * n / tot for n in lengths]
    base = [int(np.floor(e)) for e in exact]
    left = total - sum(base)
    for i in np.argsort([-(e - b) for e, b in zip(exact, base)])[:max(left, 0)]:
        base[int(i)] += 1
    return [min(b, n) for b, n in zip(base, lengths)]


def sample_frames(n_frames, k, rec_id):
    """Stratified random: one frame from each of k equal blocks.

    Seeded from the recording's own name so the draw is reproducible without
    a global seed that would shift every recording when one is added."""
    seed = zlib.crc32(rec_id.encode()) & 0xFFFFFFFF
    rng = np.random.default_rng(seed)
    edges = np.linspace(0, n_frames, k + 1).astype(int)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        if b > a:
            out.append(int(rng.integers(a, b)))
    return out


def _grab(path, idxs):
    """Frames at `idxs` from one video, seeking once per frame. -> {idx: img}"""
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return {}
    out = {}
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, img = cap.read()
        if ok:
            out[i] = img[:, :img.shape[1] // 2]      # left eye of the pair
    cap.release()
    return out


def build_sheets(rec_id, views, out_dir, k, tile=TILE, rows=ROWS_PER_SHEET):
    """-> (list of sheet paths, list of manifest rows)."""
    import cv2
    n = 0
    for v in VIEWS:
        if v in views:
            cap = cv2.VideoCapture(views[v])
            n = max(n, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            cap.release()
    if n <= 0:
        return [], []
    idxs = sample_frames(n, k, rec_id)
    grabbed = {v: _grab(views[v], idxs) for v in VIEWS if v in views}

    paths, manifest = [], []
    for s in range((len(idxs) + rows - 1) // rows):
        block = idxs[s * rows:(s + 1) * rows]
        sheet = np.zeros((len(block) * tile, len(VIEWS) * tile, 3), np.uint8)
        for r, i in enumerate(block):
            for c, v in enumerate(VIEWS):
                img = grabbed.get(v, {}).get(i)
                if img is None:
                    continue
                t = cv2.resize(img, (tile, tile))
                cv2.putText(t, f"{v} f{i} {i/fps:.0f}s", (8, 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                sheet[r * tile:(r + 1) * tile, c * tile:(c + 1) * tile] = t
            manifest.append({"recording": rec_id, "sheet": s, "row": r,
                             "frame": i, "t_sec": round(i / fps, 1),
                             "label": "", "n_other_hands": "", "note": ""})
        p = os.path.join(out_dir, f"{rec_id}_sheet{s}.jpg")
        cv2.imwrite(p, sheet, [cv2.IMWRITE_JPEG_QUALITY, 86])
        paths.append(p)
    return paths, manifest


def _check_space(out_dir):
    import shutil
    os.makedirs(out_dir, exist_ok=True)
    free = shutil.disk_usage(out_dir).free / 1e9
    if free < MIN_FREE_GB:
        raise SystemExit(
            f"refusing to write to {out_dir}: {free:.1f} GB free, "
            f"under the {MIN_FREE_GB:.0f} GB floor.\n"
            f"  Large outputs belong on the SAN, not the shared root "
            f"filesystem -- a full root\n  surfaces in other people's "
            f"programs as a write error, not a disk error.")
    return free


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("survey", "sheets"))
    ap.add_argument("--root", required=True,
                    help="directory searched recursively for cam*.mp4")
    ap.add_argument("--out", help="output directory. On the dev container the "
                    "SAN is /workspace (also /storage), 64T; / is the shared "
                    "host disk and is not for outputs.")
    ap.add_argument("--instants", type=int, default=180,
                    help="total blind instants across the whole corpus, "
                         "apportioned by recording length")
    ap.add_argument("--equal", action="store_true",
                    help="give every recording the same number of instants "
                         "instead. Answers a different question -- the rate "
                         "in a typical recording rather than in the footage "
                         "-- and here spends a third of the budget on "
                         "databags that are 0.4%% of it.")
    a = ap.parse_args()

    recs = find_recordings(a.root)
    if not recs:
        raise SystemExit(f"no cam*.mp4 found under {a.root}")

    full = [r for r in recs if len(r[1]) == len(VIEWS)]
    if not full:
        raise SystemExit("no recording has all three modules")
    lengths = [frame_count(v) for _, v, _ in full]
    share = [n / max(sum(lengths), 1) for n in lengths]

    if a.mode == "survey":
        print(f"{len(recs)} recordings under {a.root}")
        print(f"  with all three modules  {len(full)}")
        print(f"  partial                 {len(recs) - len(full)}")
        print(f"  total footage           {sum(lengths)/30/3600:.1f} h "
              f"at 30 fps\n")
        alloc = allocate(lengths, a.instants)
        print(f"    {'minutes':>8}{'share':>8}{'instants':>9}  recording")
        for (rid, _, _), n, s, k in sorted(
                zip(full, lengths, share, alloc), key=lambda r: -r[1]):
            print(f"    {n/30/60:8.1f}{s:7.1%}{k:9d}  {rid}")
        zero = sum(1 for k in alloc if k == 0)
        print(f"\n  {a.instants} instants apportioned by length; {zero} "
              f"recordings draw none because\n  together they are "
              f"{sum(s for s, k in zip(share, alloc) if k == 0):.1%} of the "
              f"footage. Equal-per-recording would\n  have spent a third of "
              f"the budget on them.")
        return

    if not a.out:
        ap.error("--out is required for sheets, and it belongs on the SAN")
    free = _check_space(a.out)
    if a.equal:
        per = max(a.instants // len(full), 1)
        alloc = [per] * len(full)
        print(f"EQUAL allocation: {len(full)} recordings x {per} instants. "
              f"This measures the\n  rate in a typical recording, NOT in the "
              f"footage.\n")
    else:
        alloc = allocate(lengths, a.instants)
    todo = [(r, k) for r, k in zip(full, alloc) if k > 0]

    print(f"{sum(k for _, k in todo)} instants over "
          f"{len(todo)} recordings, stratified random, blind\n"
          f"  -> {a.out}  ({free:.0f} GB free)\n")
    manifest, sheets = [], []
    for i, ((rid, views, d), k) in enumerate(todo, 1):
        p, m = build_sheets(rid, views, a.out, k)
        sheets += p
        manifest += m
        print(f"  [{i}/{len(todo)}] {rid}: {len(m)} instants, "
              f"{len(p)} sheet(s)")

    csv_path = os.path.join(a.out, "class2_census.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
        w.writeheader()
        w.writerows(manifest)
    total = sum(os.path.getsize(p) for p in sheets) + os.path.getsize(csv_path)
    assert os.path.getsize(csv_path) > 0, "wrote an empty manifest"

    print(f"\n  {len(manifest)} instants, {len(sheets)} sheets, "
          f"{total/1e6:.1f} MB total")
    print(f"  manifest {csv_path}")
    print(f"\n  label column takes one of: {', '.join(LABELS)}")
    print("    other_near  a colleague's hand or arm reaching the workspace "
          "-- the class\n                the head has to learn and the only "
          "one that counts\n    other_far   a person visible across the line "
          "whose hands never arrive")
    print("\n  Blind and unmined on purpose: a mined sample can train a head, "
          "only a blind\n  one can say what the deployment rate is.")


if __name__ == "__main__":
    main()
