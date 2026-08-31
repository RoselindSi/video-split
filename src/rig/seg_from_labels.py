"""Turn ownership labels into segmentation masks, without drawing one.

The ownership rule works and has been verified -- 1028 labels over 13
recordings, zero errors -- but it works in ONE coordinate system. It reads the
height at which the forearm's ray leaves the frame, and that is only "towards
the wearer's body" because the wide render's up axis is the fan's rotation
axis. In a raw module view cam1 is mounted rolled and the wearer's arm enters
from the lower LEFT; the same rule there would be nonsense. Even inside the
wide frame it leans on a single number, 0.55 of the height, to cover an arm
reaching in from the far left at mid height.

A segmentation model has no such dependency: it reads appearance and context
and does not need to be told which way is down.

WHAT MAKES THIS POSSIBLE NOW AND NOT BEFORE. The three-class head was
abandoned because there were no `other_arm` labels -- all 83 candidates the
colour detector produced were furniture. That is fixed, and not by drawing
anything:

    detector          box and 21 keypoints
    GrabCut           box + keypoints -> that hand's pixels
    the 1028 labels   whose hand each one is

Together those are a full three-class ground truth. The masks come from the
same GrabCut the pipeline already uses, so the model is being taught the
boundaries the pipeline can actually produce, and the CLASS comes from a
person.

DETECTIONS ARE MATCHED BY BOX, NOT BY INDEX. Re-running the detector on the
same frame is deterministic today, but an ultralytics upgrade that reorders
its output would silently attach every label to the wrong hand and nothing
would fail. The label's stored box centre is matched against the fresh
detections and a poor match is dropped rather than guessed.
"""
from __future__ import annotations

import csv
import os
import re

import numpy as np

# `own_label` writes stems as f"{tag}f{frame:06d}_h{j}", and every tag so far
# ends in an underscore. Anchoring on the frame number rather than splitting on
# the first "f" means a tag that happens to contain one does not silently
# become a different recording.
STEM_RE = re.compile(r"^(.*?)f(\d{6})_h(\d+)$")

# A label is attached to a fresh detection only if their box centres are this
# close, as a fraction of the frame width. Loose enough for a re-render to
# differ by a pixel, tight enough that two hands cannot be swapped.
MATCH_TOL_FRAC = 0.03


def load_labels(pkgs, require_label=True):
    """-> {(tag, frame): [rows]}. The tag identifies the recording.

    `require_label=False` is for `recover`, which only needs to know which
    frame a package showed -- and must work on the server copies, which were
    never labelled."""
    out = {}
    missing = []
    for p in pkgs:
        q = os.path.join(p, "hands.csv")
        if not os.path.exists(q):
            # Silence here cost a build that quietly used 4 labels instead of
            # 272: two --pkg paths did not exist on the server and the run
            # looked like a success.
            missing.append(p)
            continue
        for r in csv.DictReader(open(q, encoding="utf-8-sig")):
            if require_label and r.get("label") not in ("owner", "other"):
                continue
            m = STEM_RE.match(r["stem"])
            if not m:
                raise SystemExit(
                    f"unrecognised stem {r['stem']!r} in {q}. The recording a "
                    f"label belongs to is\n  read from the stem; guessing it "
                    f"would attach real labels to another\n  recording's "
                    f"pixels and nothing would fail.")
            out.setdefault((m.group(1), int(r["frame"])), []).append(r)
    if missing:
        print(f"  !! {len(missing)} --pkg path(s) have no hands.csv:")
        for m_ in missing:
            print(f"     {m_}")
    return out


def match(dets, rows, shape, tol=MATCH_TOL_FRAC):
    """-> [(det, row)]. Nearest box centre, refused past `tol`."""
    H, W = shape[:2]
    if not dets or not rows:
        return []
    dc = np.array([[(d["box"][0] + d["box"][2]) / 2 / W,
                    (d["box"][1] + d["box"][3]) / 2 / H] for d in dets])
    rc = np.array([[float(r["box_cx"]), float(r["box_cy"])] for r in rows])
    dist = np.linalg.norm(dc[:, None] - rc[None], axis=-1)
    pairs = []
    for _ in range(min(len(dets), len(rows))):
        i, j = np.unravel_index(np.argmin(dist), dist.shape)
        if not np.isfinite(dist[i, j]) or dist[i, j] > tol:
            break
        pairs.append((dets[i], rows[j]))
        dist[i, :] = np.inf
        dist[:, j] = np.inf
    return pairs


# Only the middle module is rendered for recovery. It is the one that spans
# the centre of the wide frame, so the centre third of a stored context image
# is very nearly what it alone produces -- and it costs ONE video open instead
# of three. On this mount an open is a 30-second timeout, which is the entire
# cost of the operation; the rendering itself is free by comparison.
RECOVER_MODULE = 1
RECOVER_BAND = (0.34, 0.66)


def _band(img, band=RECOVER_BAND):
    W = img.shape[1]
    return img[:, int(W * band[0]):int(W * band[1])]


def _score(a, b):
    """Normalised correlation of two frames, on a heavy downscale.

    Downscaled hard on purpose. The stored context image has a yellow box and
    a magenta ray drawn on it that the fresh render does not, and at 64 pixels
    wide those few strokes cannot outvote the scene."""
    import cv2
    ga = cv2.cvtColor(cv2.resize(a, (64, 48)), cv2.COLOR_BGR2GRAY).astype(float)
    gb = cv2.cvtColor(cv2.resize(b, (64, 48)), cv2.COLOR_BGR2GRAY).astype(float)
    ga -= ga.mean()
    gb -= gb.mean()
    den = np.linalg.norm(ga) * np.linalg.norm(gb)
    return float((ga * gb).sum() / den) if den > 0 else 0.0


# A package is tied to a databag only if the best match beats the runner-up by
# this much. Two recordings of the same bench from the same rig correlate
# highly with each other; without a margin the winner would often be a
# coin toss dressed as evidence.
RECOVER_MARGIN = 0.15

# A true match is not a good score, it is very nearly identity: the same
# recording at the same frame index rendered the same way. The first run
# measured 0.986 to 1.000 for the six it identified and at most 0.62 for the
# eight it did not, with nothing in between. That gap is the useful signal --
# a low best score does not mean two candidates were confusable, it means the
# real recording was never among them. Reporting those two cases as one word
# sent the reader looking for a threshold to tune instead of for the missing
# directory.
RECOVER_IDENTITY = 0.90


def _open_middle(databag):
    """-> (rig, vcam, cap, module) with ONE module's video open.

    The middle module is preferred because it spans the centre of the wide
    frame, but one databag's cam34.mp4 could not be decoded at all and that
    silently removed the whole recording from the candidate list -- if it had
    been the true source of a package, the package would have been reported
    unidentifiable with no hint that a file was to blame. Falling back to
    another module costs nothing and keeps the candidate in the running."""
    import cv2
    from src.rig.calibration import RigCalibration
    from src.rig.geometry import VirtualWideCamera
    rig = RigCalibration(os.path.join(databag, "calibration.yaml"))
    order = [RECOVER_MODULE] + [i for i in range(len(rig.modules))
                                if i != RECOVER_MODULE]
    for i in order:
        m = rig.modules[i]
        key = f"cam{m.left.name[-1]}{m.right.name[-1]}"
        q = os.path.join(databag, f"{key}.mp4")
        if not os.path.exists(q):
            continue
        cap = cv2.VideoCapture(q)
        if cap.isOpened():
            return rig, VirtualWideCamera.from_rig(rig), cap, m
        cap.release()
    return None


def _render_at(state, frame, width, map_cache):
    import cv2
    from src.rig.render_wide import split_halves, render
    rig, vcam, cap, m = state
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
    ok, img = cap.read()
    if not ok:
        return None
    l, r = split_halves(img)
    sources = {m.left.name: l, m.right.name: r}
    try:
        rgb, _, _, _ = render(rig, vcam, sources, 0.6, map_cache=map_cache)
    except TypeError:
        rgb, _, _, _ = render(rig, vcam, sources, 0.6)
    H, W = rgb.shape[:2]
    return cv2.resize(rgb, (width, int(width * H / W)))


def recover(pkgs, candidates, margin=RECOVER_MARGIN, verbose=True):
    """Work out which databag each package came from, from its own pixels.

    Ten packages were labelled before `own_label` started recording their
    source. The mapping was in a log and a shell history that no longer exist,
    but the package stores the rendered frame it showed the labeller, so
    re-rendering that frame number from each candidate identifies the
    recording directly. -> {tag: databag}

    THE LOOP IS DATABAG-OUTER ON PURPOSE. Package-outer is the natural way to
    write this and costs an open per (package, candidate) pair -- 195 opens
    where 13 suffice. On a read-only network mount an open is thirty seconds
    and everything else is free, so the loop order IS the runtime."""
    import cv2
    labels = load_labels(pkgs, require_label=False)
    by_tag = {}
    for (t, f) in labels:
        by_tag.setdefault(t, []).append(f)
    ctx_dir = {}
    for p in pkgs:
        for r in load_labels([p], require_label=False):
            ctx_dir[r[0]] = p

    refs, frames = {}, {}
    for t in sorted(by_tag):
        frame = sorted(by_tag[t])[len(by_tag[t]) // 2]
        stem = next(r["stem"] for r in labels[(t, frame)])
        ref = cv2.imread(os.path.join(ctx_dir[t], "context", stem + ".jpg"))
        if ref is None:
            if verbose:
                print(f"  {t:6s} no context image; cannot recover")
            continue
        refs[t], frames[t] = ref, frame
    if not refs:
        return {}

    scores = {t: [] for t in refs}
    for n, d in enumerate(candidates, 1):
        name = os.path.basename(d)
        if verbose:
            print(f"  [{n}/{len(candidates)}] {name}", flush=True)
        try:
            state = _open_middle(d)
        except Exception as e:
            if verbose:
                print(f"      {type(e).__name__}: {e}")
            continue
        if state is None:
            continue
        mc = {}
        try:
            for t, ref in refs.items():
                img = _render_at(state, frames[t], ref.shape[1], mc)
                if img is None:
                    continue
                h = min(img.shape[0], ref.shape[0])
                scores[t].append((_score(_band(img[:h]), _band(ref[:h])), d))
        finally:
            state[2].release()

    out = {}
    for t in sorted(refs):
        sc = sorted(scores[t], reverse=True)
        if not sc:
            if verbose:
                print(f"  {t:6s} nothing rendered")
            continue
        gap = sc[0][0] - (sc[1][0] if len(sc) > 1 else -1.0)
        if sc[0][0] < RECOVER_IDENTITY:
            verdict = "NOT IN CANDIDATES"
        elif gap < margin:
            verdict = "AMBIGUOUS"
        else:
            verdict = "OK"
        if verbose:
            print(f"  {t:6s} frame {frames[t]:6d}  "
                  f"{os.path.basename(sc[0][1]):28s} r={sc[0][0]:+.3f}"
                  f"  gap {gap:+.3f}  {verdict}")
            for v, d in sc[1:3]:
                print(f"         {'':6s}  {os.path.basename(d):28s} r={v:+.3f}")
        if verdict == "OK":
            out[t] = sc[0][1]
    n_miss = sum(1 for t in refs if t not in out)
    if verbose and n_miss:
        print(f"\n  {n_miss} unidentified. NOT IN CANDIDATES means the best "
              f"score is nowhere near\n  identity -- widen the candidate list "
              f"rather than lowering the margin. A true\n  match scores above "
              f"{RECOVER_IDENTITY:.2f}; the highest a wrong one has reached "
              f"is 0.62.")
    return out


def build(rig, videos, tag, labels, out_dir, model, split, with_depth=True,
          verbose=True):
    """Re-render each labelled frame, build its 3-class mask. -> [manifest rows]"""
    import time
    import cv2
    from src.rig.geometry import VirtualWideCamera
    from src.rig.render_wide import split_halves, render
    from src.rig.hand_detect import detect, masks_from
    from src.rig.seg_dataset import encode_range

    want = sorted(f for (t, f) in labels if t == tag)
    if not want:
        return []
    # The videos are opened ONCE and seeked, not reopened per frame. On this
    # read-only mount an open is a thirty-second ffmpeg timeout, so the
    # obvious `read_frame(path, n)` per frame would cost 224 frames x 3 videos
    # x 30s -- five hours of waiting for a few minutes of work. Frames are
    # visited in ascending order so most seeks are short.
    # Keyed by camera-name pair, not by the module. `Module` is a frozen
    # dataclass holding numpy arrays, so it is unhashable -- and the
    # TypeError only appears once a real video opens, well past any smoke
    # test that never touches the filesystem.
    caps = []
    for m in rig.modules:
        key = f"cam{m.left.name[-1]}{m.right.name[-1]}"
        if key not in videos or not os.path.exists(videos[key]):
            continue
        c = cv2.VideoCapture(videos[key])
        if c.isOpened():
            caps.append((m, c))
        else:
            c.release()
    if not caps:
        raise SystemExit(f"no readable video for {tag} in {videos}")
    vcam = VirtualWideCamera.from_rig(rig)
    for sub in ("images", "masks", "range", "overlays"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
    open(os.path.join(out_dir, "SPLIT"), "w").write(split)

    mc, rect, rows_out = {}, {}, []
    n_unmatched = 0
    t0 = time.time()
    for n, frame in enumerate(want):
        sources = {}
        for m, c in caps:
            c.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
            ok, img = c.read()
            if not ok:
                sources = {}
                break
            l, r = split_halves(img)
            sources[m.left.name], sources[m.right.name] = l, r
        if not sources:
            continue
        try:
            rgb, _, _, _ = render(rig, vcam, sources, 0.6, map_cache=mc)
        except TypeError:
            rgb, _, _, _ = render(rig, vcam, sources, 0.6)
        dets = detect(model, rgb)
        pairs = match(dets, labels[(tag, frame)], rgb.shape)
        n_unmatched += len(labels[(tag, frame)]) - len(pairs)
        if not pairs:
            continue

        own = [d for d, r in pairs if r["label"] == "owner"]
        oth = [d for d, r in pairs if r["label"] == "other"]
        mask = np.zeros(rgb.shape[:2], np.uint8)
        # class 2 first, then class 1 over it: where a colleague's hand and
        # the wearer's overlap, the wearer wins -- the same precedence the
        # suppressor uses, so the model is taught the pipeline's own tie-break.
        if oth:
            mask[masks_from(rgb, oth)] = 2
        if own:
            mask[masks_from(rgb, own)] = 1

        # `seg_head.load_manifest` reconstructs this name from the manifest
        # as f"{recording}_f{frame:06d}.png", and `recording` is the tag with
        # its trailing underscore stripped. Writing it any other way builds a
        # complete dataset that the trainer then reports as having no masks
        # at all.
        stem = f"{tag}f{frame:06d}.png"
        cv2.imwrite(os.path.join(out_dir, "images", stem), rgb)
        cv2.imwrite(os.path.join(out_dir, "masks", stem), mask)
        ov = rgb.copy()
        ov[mask == 1] = (0.45 * ov[mask == 1]
                         + 0.55 * np.array([0, 230, 0])).astype(np.uint8)
        ov[mask == 2] = (0.45 * ov[mask == 2]
                         + 0.55 * np.array([0, 90, 255])).astype(np.uint8)
        cv2.imwrite(os.path.join(out_dir, "overlays",
                                 stem.replace(".png", ".jpg")), ov,
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
        has_depth = 0
        if with_depth:
            from src.rig.wide_depth import wide_depth
            from src.rig.depth import rectify_maps
            for m in rig.modules:
                if m.name not in rect:
                    rect[m.name] = rectify_maps(rig, m)
            wd = wide_depth(rig, vcam, sources, rect_cache=rect)
            cv2.imwrite(os.path.join(out_dir, "range", stem),
                        encode_range(wd.range_m))
            has_depth = 1
        rows_out.append({
            "recording": tag.rstrip("_"), "frame": frame, "split": split,
            "has_depth": has_depth, "census_label": "",
            "label_source": "ownership_labels",
            "n_owner": len(own), "n_other": len(oth),
            "owner_px": int((mask == 1).sum()),
            "other_px": int((mask == 2).sum())})
        if verbose and ((n + 1) % 20 == 0 or n + 1 == len(want)):
            el = time.time() - t0
            print(f"    [{n+1}/{len(want)}] {len(rows_out)} frames, "
                  f"{el:.0f}s, {el/(n+1)*(len(want)-n-1):.0f}s left",
                  flush=True)
    for _, c in caps:
        c.release()
    if n_unmatched and verbose:
        print(f"    !! {n_unmatched} labels could not be matched to a fresh "
              f"detection and were dropped")
    return rows_out


def _self_test():
    import tempfile
    ok = 0

    def chk(name, cond):
        nonlocal ok
        print(f"  {'ok  ' if cond else 'FAIL'} {name}")
        ok += bool(cond)

    d = tempfile.mkdtemp()
    with open(os.path.join(d, "hands.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, ["stem", "frame", "box_cx", "box_cy", "label"])
        w.writeheader()
        w.writerows([
            {"stem": "D7_f004000_h0", "frame": 4000, "box_cx": 0.20,
             "box_cy": 0.60, "label": "owner"},
            {"stem": "D7_f004000_h1", "frame": 4000, "box_cx": 0.80,
             "box_cy": 0.30, "label": "other"},
            {"stem": "D7_f004500_h0", "frame": 4500, "box_cx": 0.50,
             "box_cy": 0.50, "label": ""},
        ])
    lab = load_labels([d])
    chk("unlabelled hands are not loaded", set(lab) == {("D7_", 4000)})
    chk("both labels on one frame are kept", len(lab[("D7_", 4000)]) == 2)

    shape = (100, 200)

    def det(cx, cy):
        return {"box": (cx * 200 - 10, cy * 100 - 10,
                        cx * 200 + 10, cy * 100 + 10)}
    rows = lab[("D7_", 4000)]
    # Deliberately reversed: index-matching would swap owner and other here.
    pairs = match([det(0.80, 0.30), det(0.20, 0.60)], rows, shape)
    chk("matched by box, not by index",
        len(pairs) == 2
        and all(abs((p[0]["box"][0] + p[0]["box"][2]) / 2 / 200
                    - float(p[1]["box_cx"])) < 1e-6 for p in pairs))

    chk("a detection far from every label is refused",
        len(match([det(0.20, 0.60), det(0.05, 0.05)], rows, shape)) == 1)
    chk("no detections -> no pairs", match([], rows, shape) == [])
    chk("a label with no detection is dropped, not guessed",
        len(match([det(0.20, 0.60)], rows, shape)) == 1)

    # Two labels closer together than the tolerance must not both bind to one
    # detection -- that would put the same pixels in two classes.
    close = [dict(rows[0]), dict(rows[1])]
    close[1]["box_cx"], close[1]["box_cy"] = "0.205", "0.605"
    got = match([det(0.20, 0.60)], close, shape)
    chk("one detection binds at most one label", len(got) == 1)

    with open(os.path.join(d, "hands.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, ["stem", "frame", "box_cx", "box_cy", "label"])
        w.writeheader()
        w.writerow({"stem": "weird", "frame": 1, "box_cx": 0, "box_cy": 0,
                    "label": "owner"})
    try:
        load_labels([d])
        chk("an unparseable stem is refused", False)
    except SystemExit:
        chk("an unparseable stem is refused", True)

    chk("a missing package is skipped, not crashed",
        load_labels(["/nonexistent"]) == {})
    print(f"\n  {ok}/9")
    return ok == 9


def main():
    import argparse
    import sys
    if "--self_test" in sys.argv:
        raise SystemExit(0 if _self_test() else 1)
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkg", action="append",
                    help="a labelled package directory; repeatable")
    ap.add_argument("--map", action="append",
                    metavar="TAG=DATABAG_DIR",
                    help="which recording each package's tag came from, e.g. "
                         "D3_=/shared/.../databag-26_0822_183131")
    ap.add_argument("--out")
    ap.add_argument("--split", choices=("train", "eval"))
    ap.add_argument("--no_depth", action="store_true")
    ap.add_argument("--weights",
                    default="/shared/models/HaWoR/weights/external/detector.pt")
    ap.add_argument("--recover", action="append", metavar="DATABAG_DIR",
                    help="candidate recording for --mode recover; repeatable")
    ap.add_argument("--mode", default="build", choices=("build", "recover"))
    ap.add_argument("--self_test", action="store_true")
    a = ap.parse_args()

    # `recover` writes nothing and needs no detector, so it must not be made
    # to pass a disk-space check on an output directory it will never create,
    # nor wait for ultralytics to import.
    if a.mode == "recover":
        if not a.pkg or not a.recover:
            raise SystemExit("recover needs --pkg and --recover")
        got = recover(a.pkg, a.recover)
        print(f"\n  {len(got)} of the packages identified. Paste as --map:")
        for t, d in sorted(got.items()):
            print(f"    --map {t}={d} \\")
        print("\n  An AMBIGUOUS row is not a near miss to be nudged: two "
              "recordings of the\n  same bench correlate highly, so a small "
              "gap means the pixels do not\n  distinguish them and the "
              "mapping has to come from the log instead.")
        return

    from ultralytics import YOLO
    from src.rig.calibration import RigCalibration
    from src.rig.class2_census import _check_space

    for need in ("pkg", "map", "out", "split"):
        if not getattr(a, need):
            raise SystemExit(f"--{need} is required")
    _check_space(a.out)
    labels = load_labels(a.pkg)
    if not labels:
        raise SystemExit("no labelled hands in those packages")
    tags = sorted({t for t, _ in labels})
    mapping = dict(s.split("=", 1) for s in a.map)
    missing = [t for t in tags if t not in mapping]
    if missing:
        raise SystemExit(
            f"no --map for tag(s) {missing}. Every tag must be tied to the "
            f"databag it came\n  from -- rendering the wrong recording's "
            f"frames would attach real labels to\n  the wrong pixels and "
            f"nothing would fail.")

    model = YOLO(a.weights)
    n_lab = sum(len(v) for v in labels.values())
    print(f"{n_lab} labelled hands over {len(labels)} frames, "
          f"{len(tags)} recordings -> {a.out} [{a.split}]\n")
    allrows = []
    for t in tags:
        d = mapping[t]
        print(f"  {t} <- {os.path.basename(d)}")
        rig = RigCalibration(os.path.join(d, "calibration.yaml"))
        vids = {k: os.path.join(d, f"{k}.mp4")
                for k in ("cam12", "cam34", "cam56")}
        allrows += build(rig, vids, t, labels, a.out, model, a.split,
                         with_depth=not a.no_depth)
    if not allrows:
        raise SystemExit("nothing built")
    with open(os.path.join(a.out, "manifest.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(allrows[0].keys()))
        w.writeheader()
        w.writerows(allrows)
    n_o = sum(r["n_other"] for r in allrows)
    px_o = sum(r["other_px"] for r in allrows)
    px_w = sum(r["owner_px"] for r in allrows)
    print(f"\n  {len(allrows)} frames, {sum(r['n_owner'] for r in allrows)} "
          f"owner hands, {n_o} other hands")
    print(f"  pixels: owner {px_w/1e6:.1f}M, other {px_o/1e6:.2f}M "
          f"({px_o/max(px_w+px_o,1):.1%} of labelled hand pixels)")
    print(f"  {sum(r['has_depth'] for r in allrows)}/{len(allrows)} with "
          f"depth")
    print("\n  The masks are GrabCut's, the CLASSES are a person's. A model "
          "trained here\n  learns ownership from appearance and context, "
          "with no dependence on which\n  way the wide render calls down.")
    if n_o < 20:
        print(f"\n  !! only {n_o} `other` hands. The class is real now but it "
              f"is thin;\n     a score on it will have a wide interval and "
              f"should be quoted with one.")


if __name__ == "__main__":
    main()
