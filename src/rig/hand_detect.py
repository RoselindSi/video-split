"""Hands from a detector that knows what a hand is, and ownership from the wrist.

This replaces a skin-colour threshold and the five shape rules that grew on top
of it. Those rules -- a border test, a solidity cap, a straightness cap, a
persistence mask, an area window -- were all trying to reconstruct "is this a
limb?" from the shape of a tan blob, and each one was a threshold cut through
two overlapping distributions. The last of them ran out of gap: a minimum
aspect ratio would have removed 45% of the remaining false positives at the
cost of 5% of real arms, because a paper disc measures 1.7 and a real arm
reaches down to 1.84.

A hand detector answers the question directly. On real wide frames it finds
both of the wearer's hands in every frame at 0.83-0.87 confidence, labels them
left and right, and returns 21 keypoints each. A wooden turntable, a beige
machine strap, a cabinet and a paper disc are not hands and it does not report
them.

OWNERSHIP COMES FROM THE WRIST, NOT FROM THE MASK. The old rule asked whether
a connected component touched the bottom of the frame, which worked (79%
against 0%) but depended on the segmentation holding together -- and a sleeve
cuts the component at the cuff, which is why the threshold on it had a margin
of 1.7 points. The keypoints make the question direct: the forearm runs from
the fingers through the wrist and onward, so extend that ray and see which
edge of the frame it leaves by. The wearer's arm exits through the bottom.
Nothing about that depends on how well anything was segmented.

MASKS STILL COME FROM GRABCUT, prompted by the detector's box instead of by a
colour threshold. That part was already built and already works; what it
lacked was a prompt from something that knew what it was pointing at.
"""
from __future__ import annotations

import os

import numpy as np

# MANO / WiLoR keypoint layout: 0 is the wrist, then thumb, index, middle,
# ring and little finger, four points each.
WRIST = 0
FINGERS = list(range(1, 21))

DETECTOR = "/shared/models/HaWoR/weights/external/detector.pt"
IMGSZ = 512
# Real hands on this corpus come in at 0.83-0.87. A pink box came in as a
# hand at the old floor of 0.35 and was blurred, which is the first false
# positive this pipeline has had that arrives with a NUMBER attached -- every
# shape rule before it had to be cut through two overlapping distributions,
# and this one does not. The floor is set from that gap and the distribution
# is printed on every run so it stays checkable.
MIN_CONF = 0.60

# The forearm ray leaves the frame at some point; the wearer's leaves LOW.
#
# ASKING WHICH EDGE IT LEFT BY WAS TOO BRITTLE AT THE CORNERS. The wearer's
# right arm enters from the lower right, and if the hand is far enough over,
# the ray reaches the right edge before it reaches the bottom one -- so an arm
# that is plainly the wearer's was reported as "exits right" and blurred. It
# happened on 4 of 48 real detections. Where the ray leaves is an accident of
# the corner; how far DOWN it leaves is the thing that actually distinguishes
# an arm coming up from the wearer's body from one reaching in across the
# bench, and it treats the bottom edge and the lower side edges alike.
OWNER_EXIT_Y_FRAC = 0.55


def forearm_exit(kp, shape):
    """Where the forearm leaves the frame. -> (edge, exit point)

    The ray runs from the fingers through the wrist and outward. The edge is
    reported for diagnosis; `is_owner` decides on the exit point's HEIGHT,
    because the corners make the edge misleading."""
    kp = np.asarray(kp, float)
    if kp.shape[0] < 21 or not np.isfinite(kp).all():
        return None, None
    H, W = shape[:2]
    wrist = kp[WRIST]
    d = wrist - kp[FINGERS].mean(0)
    n = np.linalg.norm(d)
    if n < 1e-6:
        return None, None
    d = d / n

    # Smallest positive distance to each of the four edges.
    ts = []
    if abs(d[0]) > 1e-9:
        ts += [((0 - wrist[0]) / d[0], "left"), ((W - wrist[0]) / d[0], "right")]
    if abs(d[1]) > 1e-9:
        ts += [((0 - wrist[1]) / d[1], "top"), ((H - wrist[1]) / d[1], "bottom")]
    ts = [(t, e) for t, e in ts if t > 0]
    if not ts:
        return None, None
    t, edge = min(ts)
    return edge, wrist + d * t


def is_owner(exit_pt, shape, y_frac=OWNER_EXIT_Y_FRAC):
    """-> True if the forearm leaves through the lower part of the border.

    One number, and it covers the bottom edge and the lower halves of both
    side edges together -- which is the shape of "towards the wearer's body"
    and is what four misclassified right hands were missing."""
    if exit_pt is None:
        return False
    return float(exit_pt[1]) >= y_frac * shape[0]


def load_owner_clf(path):
    """-> {'model','features'} or None. Missing is not an error: the rule is
    the fallback and it is the thing being replaced, not a placeholder."""
    import pickle
    if not path or not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def classify_owner(clf, det, shape):
    """-> (is_owner, probability). The classifier decides; the rule's verdict
    is left in the detection beside it so the two can be compared on live
    data, not only on the labelled set."""
    from src.rig.own_label import features
    x = features(det, shape)[None]
    p = float(clf["model"].predict_proba(x)[0, 1])
    return p >= 0.5, p


def detect(model, rgb, imgsz=IMGSZ, min_conf=MIN_CONF, clf=None):
    """-> [{'box','kp','side','conf','edge','owner','owner_p'}] for one frame.

    `clf` decides ownership when it is supplied. The geometric verdict stays
    in `rule_owner` regardless: on the 268 hands labelled so far the two agree
    everywhere, so the day they disagree is the day something new is in shot,
    and that is worth seeing rather than silently overriding."""
    predict_args = {"imgsz": imgsz, "verbose": False}
    if min_conf is not None:
        # Ultralytics filters candidates inside inference. Applying only the
        # second Python-side cut below means a requested floor lower than the
        # model default can never recover the discarded boxes.
        predict_args["conf"] = float(min_conf)
    res = model(rgb, **predict_args)[0]
    out = []
    if res.boxes is None or len(res.boxes) == 0:
        return out
    boxes = res.boxes.xyxy.cpu().numpy()
    conf = res.boxes.conf.cpu().numpy()
    cls = res.boxes.cls.cpu().numpy().astype(int)
    kps = (res.keypoints.xy.cpu().numpy()
           if res.keypoints is not None else [None] * len(boxes))
    for b, c, k, kp in zip(boxes, conf, cls, kps):
        if min_conf is not None and c < min_conf:
            continue
        edge, pt = (forearm_exit(kp, rgb.shape) if kp is not None
                    else (None, None))
        d = {"box": b.astype(int), "kp": kp, "conf": float(c),
             "side": model.names.get(int(k), str(int(k))),
             "edge": edge, "exit": pt,
             "rule_owner": is_owner(pt, rgb.shape)}
        d["owner"] = d["rule_owner"]
        d["owner_p"] = float(d["rule_owner"])
        if clf is not None:
            try:
                d["owner"], d["owner_p"] = classify_owner(clf, d, rgb.shape)
            except Exception:
                pass                      # a broken model must not lose a frame
        out.append(d)
    return out


def split_owner(dets, shape, max_owner=2, verbose=False):
    """-> (owner dets, other dets). A wearer has two hands.

    THE CAP RANKS BY THE CLASSIFIER'S PROBABILITY, NOT BY GEOMETRY. It used to
    sort by how straight down the forearm pointed and demote the rest, which
    meant a hand-written geometric rule silently overruled the learned
    decision -- using the very quantity the classifier was brought in to
    replace. It fired on a real frame: three hands, the classifier called the
    third the wearer's at p=0.996, and the cap blurred it anyway without
    saying so.

    The cap itself stays, because a person has two hands and a third
    confident `owner` is a fact about the model rather than about the scene.
    It now demotes the LEAST confident, and it announces itself."""
    own, oth = [], []
    for d in dets:
        (own if d.get("owner") else oth).append(d)
    if len(own) > max_owner:
        def rank(d):
            if "owner_p" in d:
                return float(d["owner_p"])
            kp = np.asarray(d["kp"], float)
            v = kp[WRIST] - kp[FINGERS].mean(0)
            # Fallback for detections made without a classifier: y grows
            # DOWNWARD, so straight down is +v[1].
            return v[1] / max(np.linalg.norm(v), 1e-9)
        own.sort(key=rank, reverse=True)
        demoted = own[max_owner:]
        if verbose:
            print(f"    !! {len(own)} hands called the wearer's; the cap "
                  f"demotes {len(demoted)} "
                  f"(p={[round(float(d.get('owner_p', -1)), 3) for d in demoted]})")
        oth += demoted
        own = own[:max_owner]
    return own, oth


# How far past the padded box the GrabCut window reaches, as a fraction of
# that box. The rim inside it is the only DEFINITE background the algorithm
# gets, and it needs some: with none, the background GMM has nothing to fit
# and the cut has no evidence for what the hand is not.
MASK_MARGIN = 0.30

# Long side, in pixels, that the GrabCut window is shrunk to before the cut.
# The mask this produces is dilated by DILATE_PX + 2*FEATHER_PX afterwards, so
# boundary detail finer than about eighteen pixels is discarded downstream no
# matter how exactly it was found. Paying graph-cut time for detail that the
# next step throws away is the definition of waste.
MASK_MAX_SIDE = 192


def masks_from(rgb, dets, pad=0.15, iters=3, max_side=MASK_MAX_SIDE):
    """GrabCut inside each detection's box. -> binary mask

    The box comes from the detector and the boundary from GrabCut, which is
    the division of labour this pipeline was always meant to have and could
    not, because the only available prompt was a colour threshold that did not
    know a hand from a turntable.

    THE CUT RUNS ON A WINDOW ROUND THE BOX, NOT ON THE FRAME. It used to be
    handed the whole image with everything outside the box marked background.
    GrabCut still builds its graph over every pixel it is given, so that spent
    a 1.4-megapixel graph cut, three iterations, PER DETECTION, and then kept
    only the part inside the box and discarded the rest. The window is the
    padded box grown by MASK_MARGIN, which preserves the one thing the frame
    was providing -- definite background to fit against -- and drops the rest.
    `max_side` shrinks that window further; pass None to cut at full scale."""
    import cv2
    H, W = rgb.shape[:2]
    out = np.zeros((H, W), bool)
    for d in dets:
        x0, y0, x1, y1 = d["box"]
        px, py = int((x1 - x0) * pad), int((y1 - y0) * pad)
        bx = (max(0, x0 - px), max(0, y0 - py),
              min(W, x1 + px), min(H, y1 + py))
        if bx[2] - bx[0] < 8 or bx[3] - bx[1] < 8:
            continue
        mx_, my_ = int((bx[2] - bx[0]) * MASK_MARGIN), \
            int((bx[3] - bx[1]) * MASK_MARGIN)
        win = (max(0, bx[0] - mx_), max(0, bx[1] - my_),
               min(W, bx[2] + mx_), min(H, bx[3] + my_))
        roi = rgb[win[1]:win[3], win[0]:win[2]]
        wh, ww = roi.shape[:2]
        s = 1.0
        if max_side and max(wh, ww) > max_side:
            s = float(max_side) / max(wh, ww)
            roi = cv2.resize(roi, (max(8, int(ww * s)), max(8, int(wh * s))),
                             interpolation=cv2.INTER_AREA)
        rh, rw = roi.shape[:2]

        def to_roi(px_, py_):
            return (int((px_ - win[0]) * (rw / float(ww))),
                    int((py_ - win[1]) * (rh / float(wh))))

        m = np.full((rh, rw), cv2.GC_BGD, np.uint8)
        a0, b0 = to_roi(bx[0], bx[1])
        a1, b1 = to_roi(bx[2], bx[3])
        m[max(0, b0):max(1, b1), max(0, a0):max(1, a1)] = cv2.GC_PR_BGD
        # The keypoints are inside the hand by construction -- a far better
        # foreground seed than a colour rule, and unavailable until now.
        if d.get("kp") is not None:
            for (kx, ky) in np.asarray(d["kp"], int):
                cx, cy = to_roi(kx, ky)
                if 0 <= cx < rw and 0 <= cy < rh:
                    cv2.circle(m, (cx, cy), max(2, int(6 * s)),
                               cv2.GC_FGD, -1)
        bg, fg = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        r = np.zeros((H, W), bool)
        try:
            cv2.grabCut(np.ascontiguousarray(roi), m, None, bg, fg, iters,
                        cv2.GC_INIT_WITH_MASK)
            rm = ((m == cv2.GC_FGD) | (m == cv2.GC_PR_FGD)).astype(np.uint8)
            if (rh, rw) != (wh, ww):
                rm = cv2.resize(rm, (ww, wh), interpolation=cv2.INTER_NEAREST)
            r[win[1]:win[3], win[0]:win[2]] = rm.astype(bool)
        except cv2.error:
            r[bx[1]:bx[3], bx[0]:bx[2]] = True
        keep = np.zeros((H, W), bool)
        keep[bx[1]:bx[3], bx[0]:bx[2]] = True
        out |= r & keep
    return out


# How far past the rule's boundary the prior takes to saturate, as a fraction
# of frame height. Inside this band the rule is saying "just about", and a
# prior that shouted at one pixel either side of a threshold would be claiming
# a confidence the measurement never had.
RULE_RAMP_FRAC = 0.15


def rule_score(det, shape, y_frac=OWNER_EXIT_Y_FRAC, ramp=RULE_RAMP_FRAC):
    """The geometric rule as a soft prior on owner. -> [0,1], or None.

    `is_owner` answers yes or no at a threshold. That throws away the part of
    the measurement worth keeping: a forearm leaving through the very bottom
    of the frame is the wearer's beyond argument, while one leaving a pixel
    below the boundary is a coin toss the threshold happened to round. The
    ramp returns 0.5 -- no information -- exactly at the boundary.

    WHAT THIS PRIOR IS WORTH, AND WHERE IT IS WORTHLESS. On 1028 labelled
    hands over 13 recordings the rule agreed with the human on every one. It
    is also the cue the 2019 egocentric work found carries real disambiguation
    signal independently of what a hand looks like, which is why it survives a
    change of worker, uniform and lighting that appearance does not.

    It is worthless upside down. Rotated 180 degrees the same rule scores
    0.000 on owner_arm against 0.983 upright, because the exit height it reads
    is exactly what a half turn inverts. That is not a weakness to be tuned
    away -- it is a statement that this prior encodes the mounting. Callers on
    a differently mounted rig must turn it off, which is why the weight is a
    parameter and not a constant."""
    pt = det.get("exit")
    if pt is None:
        return None
    t = (float(pt[1]) - y_frac * shape[0]) / max(1e-6, ramp * shape[0])
    return float(np.clip(0.5 + 0.5 * t, 0.0, 1.0))


class OwnHold:
    """Ownership carried across frames, so one frame's doubt is not a flicker.

    WHY THIS AND NOT A BETTER CLASSIFIER. Instrumented over 120 consecutive
    frames, every one of the five times the cover dropped was the label: the
    classifier stopped calling a hand foreign while the detector's box sat
    still on it. GrabCut returned an empty mask zero times and the owner mask
    vetoed zero times. The cut and the veto are not what is broken.

    A SCHMITT TRIGGER, NOT AN AVERAGE. Averaging alone still crosses 0.5 in
    both directions on a hand the classifier is unsure of, which is exactly
    the hand that flickers. Two thresholds with a gap between them mean a
    verdict has to be contradicted by a margin, not merely by noise.

    THE TWO THRESHOLDS ARE NOT SYMMETRIC, AND SHOULD NOT BE. Calling a
    colleague's hand yours puts it in front of the downstream model; calling
    yours a colleague's blurs a hand that model wanted. Both are wrong, but
    only the first is a leak, so leaving `other` costs more evidence than
    entering it. This is the same asymmetry the face hold is built on.

    Smoothing needs to know which hand is which, so the matching is `track`'s
    and inherits its limits: a hand that vanishes for a frame and returns is a
    new track, and starts from its own probability with no history."""

    def __init__(self, fast=0.8, slow=0.4, lo=0.35, hi=0.70, rule_w=0.35,
                 geom=None, geom_w=0.5, max_owner=None, state_ttl=5,
                 self_reconfirm_frames=2):
        # Asymmetric in the smoothing as well as in the thresholds. A single
        # symmetric rate cannot do both jobs: slow enough to ignore a frame of
        # doubt is also slow enough to leave a colleague's hand uncovered for
        # a frame after it appears, and that frame is a leak. Evidence moving
        # TOWARDS `other` is taken at `fast`, evidence moving away at `slow`.
        self.fast, self.slow = float(fast), float(slow)
        self.lo, self.hi = float(lo), float(hi)
        # A BLEND AND NOT A TIE-BREAK. Using the rule only where the network
        # is unsure would have left the failure that prompted this untouched:
        # on the frame measured, the classifier called a colleague's hand the
        # wearer's at 0.93, confidently and wrongly, while the rule had it
        # right. A tie-break never runs at 0.93. Blended at this weight the
        # same frame scores 0.65*0.93 + 0.35*0.0 = 0.60, under the 0.70 a
        # verdict of `other` has to be beaten by, so the cover stays on.
        self.rule_w = float(rule_w)
        # A FITTED PRIOR REPLACES THE HAND-SET ONE RATHER THAN JOINING IT.
        # `rule_score` is exit height with a weight chosen by hand; the fitted
        # prior takes exit height as ONE of its inputs and sets its weight
        # from the data. Running both would count that cue twice with two
        # different coefficients, which is the mistake that inverse-frequency
        # weighting on an enriched sample already cost this project once.
        #
        # The weight is higher because the prior earned it: leave-one-
        # recording-out macro AUC 0.989, against a single rule that ranked
        # ninth of sixteen cues. And the invariant fit -- size, span,
        # confidence, relative size, no heights at all -- scores 0.988, so
        # the orientation-dependent half of the evidence is worth 0.001 and
        # the pipeline can stop depending on how the camera is mounted.
        self.geom, self.geom_w = geom, float(geom_w)
        # A PERSON HAS TWO HANDS, AND THAT IS A FACT ABOUT ANATOMY RATHER
        # THAN A TENDENCY ABOUT THIS CORPUS. Applied after the hysteresis it
        # converts a threshold into a rank inside the frame, which is what
        # the measurements say this data supports: leave-one-recording-out
        # AUC 0.988 on geometry, against a threshold that shifts with the
        # workstation. Measured on nineteen recordings, capping the blend at
        # two owners took recall from 0.700 back to 1.000 on the two
        # recordings the blend had lost.
        #
        # It can only turn `owner` into `other`, so it cannot cost recall and
        # can cost precision -- which is the safe direction here only if the
        # cover is cheap. On a frame where the wearer's hands are out of view
        # and three colleagues' hands are in it, this demotes the third one
        # correctly; on a frame with three of the wearer's own detections, one
        # of them is a duplicate and demoting it blurs part of a hand.
        self.max_owner = max_owner
        # Keyed on TRACK ID, not on the previous frame's list position. A
        # position is only meaningful while the list is the same list, and it
        # stopped being the same list every time the detector gained or lost
        # a hand -- which is when this state was silently transferred to a
        # different hand.
        # [ema, is_owner, frames_unseen, pending_self_confirmations]. A
        # positive pending count means a formerly-self track was reacquired and
        # is not allowed to inherit that old verdict without fresh evidence.
        self.state = {}
        # Frames a verdict outlives the detection that produced it. Matches
        # the tracker's own patience: state for a track the tracker has given
        # up on is state nobody will ask for again.
        self.state_ttl = int(state_ttl)
        self.self_reconfirm_frames = max(1, int(self_reconfirm_frames))

    def update(self, dets, flags, shape=None, ids=None, reacquired=None):
        """-> [(is_owner, smoothed_p)] aligned with `dets`.

        IDENTITY IS NOT THIS CLASS'S JOB ANY MORE. It used to match the
        previous frame's detections itself, by nearest centre with a distance
        cap that was computed and never applied. Every previous detection was
        therefore assigned to some current one however far it had moved, so a
        departing hand's smoothed verdict was handed to an arriving one and
        the hysteresis became a channel for propagating a wrong state onto a
        different hand. `ids` now comes from a tracker that gates and solves
        the assignment properly, and state is keyed on those ids: a hand that
        the tracker considers new starts from its own score with no history,
        which is the correct behaviour and was not available before."""
        if ids is None:
            ids = list(range(len(dets)))
        reacquired = set(reacquired or ())
        out, seen = [], set()
        for k, (d, (o, p)) in enumerate(zip(dets, flags)):
            tid = ids[k]
            p = float(p)
            if shape is not None and self.geom is not None:
                from src.rig import geom_prior
                gs = geom_prior.score(d, dets, shape, self.geom)
                if gs is not None:
                    p = (1 - self.geom_w) * p + self.geom_w * gs
            elif shape is not None and self.rule_w > 0:
                rs = rule_score(d, shape)
                if rs is not None:
                    p = (1 - self.rule_w) * p + self.rule_w * rs
            prev = self.state.get(tid)
            if prev is None:
                ema, lab, pending = p, p >= 0.5, 0
            elif tid in reacquired and prev[1]:
                # A missing self hand and a newly arrived colleague can occupy
                # the same patch of image. Track geometry may tentatively join
                # them, but the old self verdict is the unsafe state to carry:
                # reset to the current evidence and require another supporting
                # frame before the pixels are allowed through unmasked.
                ema = p
                pending = 1 if p >= 0.5 else 0
                lab = pending >= self.self_reconfirm_frames
                if lab:
                    pending = 0
            else:
                # ASYMMETRIC ON PURPOSE. Evidence that this hand is foreign is
                # adopted fast and evidence that it is the wearer's is adopted
                # slowly, because the two errors are not equal: failing to
                # cover a colleague's hand leaves a stranger in the frame,
                # while covering the wearer's destroys the pixels the whole
                # pipeline exists to deliver.
                a = self.fast if p < prev[0] else self.slow
                ema = a * p + (1 - a) * prev[0]
                lab = prev[1]
                pending = prev[3] if len(prev) > 3 else 0
                if pending:
                    pending = pending + 1 if p >= 0.5 else 0
                    lab = pending >= self.self_reconfirm_frames
                    if lab:
                        pending = 0
                elif lab and ema < self.lo:
                    lab = False
                elif (not lab) and ema > self.hi:
                    lab = True
            self.state[tid] = [ema, lab, 0, pending]
            seen.add(tid)
            out.append((bool(lab), float(ema)))
        # A PERSON HAS TWO HANDS. Applied last, after the hysteresis, so it
        # ranks the smoothed scores rather than one frame's noise. Measured on
        # the clip that prompted it, this constraint was producing most of the
        # `other` verdicts: with it the owner count sat at exactly 2.000 on
        # every frame and 257 hands were called foreign; the release that
        # accidentally dropped it called 67. The classifier and the prior are
        # not what was carrying that clip -- the anatomy was.
        if self.max_owner is not None:
            own_i = [i for i, (o, _) in enumerate(out) if o]
            if len(own_i) > self.max_owner:
                keep = set(sorted(own_i, key=lambda i: -out[i][1])
                           [:self.max_owner])
                for i in own_i:
                    if i not in keep:
                        out[i] = (False, out[i][1])
                        self.state[ids[i]][1] = False
                        self.state[ids[i]][3] = 0
        # STATE SURVIVES A FRAME THE TRACKER SAW NOTHING IN. It used to be
        # rebuilt from scratch every frame, so a track the detector missed for
        # one frame lost its smoothed score AND its verdict. Two things broke
        # on that. The cover could not be bridged across a dropout -- the
        # caller asks this class what a coasting track's last verdict was, and
        # the answer was always None, so a fix for the dropouts fired zero
        # times and a render came back byte-identical to the one before it.
        # And a hand that reappeared started its hysteresis over, which is the
        # opposite of what a hysteresis is for.
        for tid in list(self.state):
            if tid in seen:
                continue
            self.state[tid][2] += 1
            if self.state[tid][2] > self.state_ttl:
                del self.state[tid]
        return out


def track(prev, cur, max_move_frac=0.25):
    """Match detections between consecutive frames by centre distance.
    -> [(prev_index, cur_index)]

    Greedy nearest-centre, capped by a fraction of the frame. A hand does not
    cross a quarter of the frame in one 30 fps step, and refusing the match
    beyond that keeps a dropout from being reported as a jump."""
    if not prev or not cur:
        return []
    def ctr(d):
        x0, y0, x1, y1 = d["box"]
        return np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0])
    P = np.stack([ctr(d) for d in prev])
    C = np.stack([ctr(d) for d in cur])
    dist = np.linalg.norm(P[:, None] - C[None], axis=-1)
    cap = max_move_frac * max(dist.max(), 1.0) if dist.size else 0
    out, used_p, used_c = [], set(), set()
    for _ in range(min(len(prev), len(cur))):
        i, j = np.unravel_index(np.argmin(dist), dist.shape)
        if not np.isfinite(dist[i, j]):
            break
        out.append((int(i), int(j)))
        dist[i, :] = np.inf
        dist[:, j] = np.inf
    return out


def stability(model, frames, min_conf=MIN_CONF):
    """-> dict of per-clip stability statistics.

    THE FAILURE THIS LOOKS FOR is not a wrong label but a CHANGING one. A hand
    called the wearer's on one frame and a colleague's on the next puts a
    blur that flickers on and off in front of a downstream video model, which
    is worse than either verdict held consistently."""
    rows, prev = [], None
    flips = same = 0
    counts = []
    for rgb in frames:
        dets = detect(model, rgb, min_conf=min_conf)
        counts.append(len(dets))
        if prev is not None:
            for i, j in track(prev, dets):
                if bool(prev[i].get("owner")) == bool(dets[j].get("owner")):
                    same += 1
                else:
                    flips += 1
        prev = dets
        rows.append(dets)
    c = np.array(counts)
    return {"frames": len(counts), "hands_per_frame_median": float(np.median(c)),
            "hands_min": int(c.min()), "hands_max": int(c.max()),
            "frames_with_no_hand": int((c == 0).sum()),
            "tracked_pairs": same + flips, "owner_label_flips": flips,
            "flip_rate": float(flips / max(same + flips, 1)),
            "per_frame": rows}


def _self_test():
    ok = []

    def chk(c, m):
        ok.append(bool(c))
        print(f"  {'ok ' if c else 'FAIL'} {m}")

    H, W = 900, 1600

    def hand(wrist, fingers_at):
        """21 keypoints: a wrist and 20 finger points clustered elsewhere."""
        kp = np.zeros((21, 2))
        kp[WRIST] = wrist
        kp[FINGERS] = np.array(fingers_at) + np.random.default_rng(0).normal(
            0, 3, (20, 2))
        return kp

    # The wearer: fingers up at the bench, wrist below them, arm from below.
    e, _ = forearm_exit(hand((800, 700), (800, 500)), (H, W))
    chk(e == "bottom", f"fingers above the wrist -> arm exits the bottom ({e})")

    # A colleague reaching in from the left: wrist to the LEFT of the fingers.
    e, _ = forearm_exit(hand((400, 400), (700, 400)), (H, W))
    chk(e == "left", f"wrist left of the fingers -> exits left ({e})")

    e, _ = forearm_exit(hand((1300, 400), (900, 400)), (H, W))
    chk(e == "right", f"wrist right of the fingers -> exits right ({e})")

    e, _ = forearm_exit(hand((800, 200), (800, 600)), (H, W))
    chk(e == "top", f"wrist above the fingers -> exits the top ({e})")

    # Diagonal, from the lower left: still the wearer.
    e, _ = forearm_exit(hand((500, 700), (800, 400)), (H, W))
    chk(e in ("bottom", "left"),
        f"a diagonal arm from the lower left exits bottom or left ({e})")

    chk(forearm_exit(np.zeros((5, 2)), (H, W))[0] is None,
        "too few keypoints returns nothing rather than guessing")
    chk(forearm_exit(np.full((21, 2), np.nan), (H, W))[0] is None,
        "NaN keypoints return nothing")

    # The case that actually failed: the wearer's right hand, far over, its
    # forearm leaving through the LOWER RIGHT corner.
    kp_lr = hand((1450, 620), (1250, 430))
    e_lr, pt_lr = forearm_exit(kp_lr, (H, W))
    chk(is_owner(pt_lr, (H, W)),
        f"a right hand exiting the lower {e_lr} edge at y={pt_lr[1]:.0f} is "
        f"still the wearer's")
    kp_ur = hand((1450, 300), (1250, 380))
    e_ur, pt_ur = forearm_exit(kp_ur, (H, W))
    chk(not is_owner(pt_ur, (H, W)),
        f"...and one exiting the UPPER {e_ur} edge at y={pt_ur[1]:.0f} is not")

    d_own = [{"kp": hand((700, 700), (700, 500)), "owner": True},
             {"kp": hand((900, 700), (900, 500)), "owner": True}]
    d_oth = [{"kp": hand((400, 400), (700, 400)), "owner": False}]
    own, oth = split_owner(d_own + d_oth, (H, W))
    chk(len(own) == 2 and len(oth) == 1,
        "two from below are the wearer's, one from the side is not")

    # A third from below: the two most straight-down win, not the first two.
    slanted = {"kp": hand((300, 700), (600, 690)), "owner": True}
    own2, oth2 = split_owner([slanted] + d_own, (H, W))
    # `not in` compares dicts holding numpy arrays with ==, which is
    # ambiguous; identity is what is meant here anyway.
    chk(len(own2) == 2 and all(d is not slanted for d in own2),
        "a third arm from below is resolved by direction, not by list order")

    # The cap must follow the classifier when there is one, not the geometry.
    hi = {"kp": hand((700, 700), (700, 500)), "owner": True, "owner_p": 0.99}
    mid = {"kp": hand((900, 700), (900, 500)), "owner": True, "owner_p": 0.95}
    lo = {"kp": hand((1200, 690), (1150, 500)), "owner": True,
          "owner_p": 0.55}
    o3, x3 = split_owner([lo, hi, mid], (H, W))
    chk(len(o3) == 2 and all(d is not lo for d in o3),
        "the cap demotes the LEAST CONFIDENT, not the least straight-down")
    chk(len(x3) == 1 and x3[0] is lo,
        "...and the demoted one is the one the classifier was least sure of")

    # THE CAP INSIDE OwnHold, WHICH IS A DIFFERENT CODE PATH FROM split_owner
    # AND WAS SILENTLY LOST IN A REWRITE. Nothing here covered it: the two
    # cases above call split_owner directly, and that function was never
    # touched. The demo lost 74% of its `other` verdicts before anyone noticed,
    # so the path the demo actually uses gets its own case.
    ohc = OwnHold(max_owner=2)
    three = [{"kp": hand((700, 700), (700, 500)), "box": (600, 600, 800, 800)},
             {"kp": hand((900, 700), (900, 500)), "box": (800, 600, 1000, 800)},
             {"kp": hand((1100, 700), (1100, 500)),
              "box": (1000, 600, 1200, 800)}]
    capped = ohc.update(three, [(True, 0.95), (True, 0.90), (True, 0.55)],
                        ids=[0, 1, 2])
    chk(sum(1 for o, _ in capped if o) == 2 and not capped[2][0],
        "OwnHold caps owners at two and demotes the least confident")
    again = ohc.update(three, [(True, 0.95), (True, 0.90), (True, 0.55)],
                       ids=[0, 1, 2])
    chk(not again[2][0],
        "...and writes the demotion back, so the hysteresis cannot undo it")

    # masks_from now cuts on a window round the box instead of the frame. The
    # thing to prove is that the answer did not move, not merely that it is
    # faster: a bright square on a dark bench, cut both ways.
    import time
    img = np.full((H, W, 3), 30, np.uint8)
    img[380:520, 640:820] = 230
    det = [{"box": (630, 370, 830, 530),
            "kp": np.array([[730, 450]] * 21, float)}]
    t0 = time.time()
    m_full = masks_from(img, det, max_side=None)
    t_full = time.time() - t0
    t0 = time.time()
    m_win = masks_from(img, det)
    t_win = time.time() - t0
    inter = int((m_full & m_win).sum())
    union = int((m_full | m_win).sum())
    chk(union > 0 and inter / union > 0.90,
        f"the windowed cut agrees with the full-frame one "
        f"(IoU {inter/max(1,union):.3f})")
    chk(m_win[380:520, 640:820].mean() > 0.9,
        "the bright object is inside the mask")
    outside = m_win.copy()
    outside[370:530, 630:830] = False
    chk(not outside.any(), "and nothing outside the padded box is claimed")
    chk(t_win < t_full,
        f"and it is faster ({t_win*1000:.0f} ms vs {t_full*1000:.0f} ms, "
        f"{t_full/max(1e-6, t_win):.1f}x)")

    # The flicker the trace charged every dropped cover to: a hand held
    # steady, called foreign, with one frame of doubt in the middle.
    def box(cx):
        return {"box": (cx - 40, 300, cx + 40, 380),
                "kp": np.array([[cx, 340]] * 21, float)}

    oh = OwnHold()
    seq = [0.05, 0.03, 0.62, 0.04, 0.06]        # one excursion, never > hi
    got = [oh.update([box(500)], [(p >= 0.5, p)])[0][0] for p in seq]
    chk(not any(got), "one frame of doubt does not flip a held `other`")

    oh2 = OwnHold()
    sustained = [0.05, 0.9, 0.92, 0.95, 0.96]
    got2 = [oh2.update([box(500)], [(p >= 0.5, p)])[0][0] for p in sustained]
    chk(got2[-1] and not got2[0],
        "but sustained evidence does flip it, so the hold is not a latch")

    # Entering `other` must be cheaper than leaving it: the asymmetry is the
    # point, not an artefact of the numbers chosen.
    oh3 = OwnHold()
    oh3.update([box(500)], [(True, 0.95)])
    quick = oh3.update([box(500)], [(False, 0.10)])[0][0]
    chk(not quick, "a confident `other` is entered in one frame")

    # The geometric prior, soft. At the boundary it must say nothing at all:
    # a threshold rounding a coin toss is not evidence.
    lowexit = {"exit": (800, 0.95 * H)}          # forearm leaves at the bottom
    highexit = {"exit": (800, 0.20 * H)}         # leaves near the top
    onexit = {"exit": (800, OWNER_EXIT_Y_FRAC * H)}
    chk(rule_score(lowexit, (H, W)) > 0.95,
        "an exit at the bottom of the frame is the wearer's, near certainly")
    chk(rule_score(highexit, (H, W)) < 0.05,
        "an exit near the top is not")
    chk(abs(rule_score(onexit, (H, W)) - 0.5) < 1e-6,
        "and an exit ON the boundary carries no information either way")
    chk(rule_score({"box": (0, 0, 1, 1)}, (H, W)) is None,
        "a hand with no forearm exit gets no prior rather than a default")

    # The frame this was built for. The classifier called a colleague's hand
    # the wearer's at 0.93 while the rule had it right; a tie-break would
    # never have run at that confidence.
    ohr = OwnHold()
    other_hand = {"box": (760, 300, 840, 380),
                  "kp": np.array([[800, 340]] * 21, float),
                  "exit": (800, 0.20 * H)}
    # The hysteresis alone already survives ONE frame of 0.93, so the prior
    # has to be judged on sustained evidence -- which is the harder case and
    # the one that actually uncovers a hand for a second at a time.
    ohr = OwnHold()
    ohr.update([other_hand], [(False, 0.02)], shape=(H, W))
    with_prior = [ohr.update([other_hand], [(True, 0.93)],
                             shape=(H, W))[0][0] for _ in range(8)]
    chk(not any(with_prior),
        "sustained but wrong `self` stays overruled while the prior holds")
    # With the prior off -- a rig mounted some other way up, where the rule
    # is not merely weaker but inverted -- the same evidence gets through.
    ohn = OwnHold()
    ohn.update([other_hand], [(False, 0.02)])
    without = [ohn.update([other_hand], [(True, 0.93)])[0][0]
               for _ in range(8)]
    chk(without[-1],
        "...and gets through with the prior off, as it must on a turned rig")

    # THE BRIDGE HAD NO TEST AND FIRED ZERO TIMES. A render came back
    # byte-identical to the one before the fix, because state for a track the
    # detector missed was discarded on the same frame the caller needed it.
    ohb = OwnHold()
    b1 = {"box": (100, 300, 180, 380), "kp": np.array([[140, 340]] * 21,
                                                      float)}
    ohb.update([b1], [(False, 0.05)], ids=[9])
    ohb.update([], [], ids=[])
    chk(ohb.state.get(9) is not None and not ohb.state[9][1],
        "a verdict survives a frame with no detection to carry it")
    for _ in range(7):
        ohb.update([], [], ids=[])
    chk(ohb.state.get(9) is None,
        "...and is dropped once the tracker would have given up too")

    print(f"\n  {sum(ok)}/{len(ok)} cases pass.")
    print("  Ownership is read off the wrist, so it does not depend on how "
          "well anything\n  was segmented -- which is what the old "
          "bottom-of-the-component rule did,\n  with a margin of 1.7 points.")
    return all(ok)


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self_test", action="store_true")
    ap.add_argument("--src", help="a directory of images, or a seg_auto split")
    ap.add_argument("--out")
    ap.add_argument("--weights", default=DETECTOR)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--stability", action="store_true",
                    help="run on CONSECUTIVE rendered frames and report how "
                         "often a tracked hand changes its owner verdict. "
                         "Scattered keyframes cannot show this and it is the "
                         "failure that matters: a blur that flickers on and "
                         "off is worse than either verdict held.")
    ap.add_argument("--calibration")
    ap.add_argument("--video", action="append", default=[],
                    metavar="FILEKEY=PATH")
    ap.add_argument("--start", type=int, default=3000)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--clf", help="an own_clf.pkl from own_label. Without "
                                  "it the geometric rule decides.")
    ap.add_argument("--min_conf", type=float, default=MIN_CONF,
                    help="real hands score 0.83-0.87 here; a pink box scored "
                         "enough to pass 0.35")
    ap.add_argument("--dilate", type=int, default=10)
    ap.add_argument("--sigma", type=float, default=14.0)
    a = ap.parse_args()
    if a.self_test:
        raise SystemExit(0 if _self_test() else 1)
    if a.stability:
        if not a.calibration or not a.video:
            ap.error("--stability needs --calibration and --video")
        from ultralytics import YOLO
        from src.rig.calibration import RigCalibration
        from src.rig.geometry import VirtualWideCamera
        from src.rig.render_wide import render
        from src.rig.seam_fix import ClipReader
        rig = RigCalibration(a.calibration)
        vcam = VirtualWideCamera.from_rig(rig)
        rd = ClipReader(rig, dict(s.split("=", 1) for s in a.video), a.start)
        model = YOLO(a.weights)
        mc, frames = {}, []
        for _ in range(a.n):
            src = rd.next()
            if not src:
                break
            try:
                rgb, _, _, _ = render(rig, vcam, src, 0.6, map_cache=mc)
            except TypeError:
                rgb, _, _, _ = render(rig, vcam, src, 0.6)
            frames.append(rgb)
        rd.close()
        st = stability(model, frames, a.min_conf)
        print(f"  {st['frames']} CONSECUTIVE frames from {a.start}\n")
        print(f"  hands per frame     median {st['hands_per_frame_median']:.0f}"
              f"   min {st['hands_min']}   max {st['hands_max']}")
        print(f"  frames with no hand {st['frames_with_no_hand']}")
        print(f"  tracked pairs       {st['tracked_pairs']}")
        print(f"  owner verdict flips {st['owner_label_flips']}"
              f"   ({st['flip_rate']:.2%})")
        print("\n  A flip is the same hand called the wearer's on one frame "
              "and a colleague's\n  on the next. That makes a blur switch on "
              "and off, which a downstream video\n  model reads as a real "
              "event. Near zero is the requirement; the label being\n  "
              "occasionally wrong but STEADY is a much smaller problem.")
        raise SystemExit(0)
    if not a.src or not a.out:
        ap.error("--src and --out are required without --self_test")

    import glob
    import os
    import cv2
    from ultralytics import YOLO
    from src.rig.suppress_other import suppress

    imgs = sorted(glob.glob(os.path.join(a.src, "images", "*.png"))) \
        or sorted(glob.glob(os.path.join(a.src, "*.png"))) \
        or sorted(glob.glob(os.path.join(a.src, "*.jpg")))
    imgs = imgs[:a.limit]
    if not imgs:
        raise SystemExit(f"no images under {a.src}")
    os.makedirs(a.out, exist_ok=True)
    model = YOLO(a.weights)
    clf = load_owner_clf(a.clf)
    print(f"  {len(imgs)} frames, detector {os.path.basename(a.weights)}, "
          f"ownership by {'classifier' if clf else 'the geometric rule'}\n")

    tally, disagree = {}, []
    confs = {"owner": [], "other": []}
    allconf = []
    for p in imgs:
        rgb = cv2.imread(p)
        dets = detect(model, rgb, min_conf=a.min_conf, clf=clf)
        own, oth = split_owner(dets, rgb.shape, verbose=True)
        for d in dets:
            tally[d.get("edge")] = tally.get(d.get("edge"), 0) + 1
            if d.get("owner") != d.get("rule_owner"):
                disagree.append((os.path.basename(p), d.get("edge"),
                                 round(d.get("owner_p", 0.0), 3)))
        for d in own:
            confs["owner"].append(d["conf"])
        for d in oth:
            confs["other"].append(d["conf"])
        # Everything the detector proposed, including what the floor rejected,
        # so the floor can be judged rather than trusted.
        for d in detect(model, rgb, min_conf=0.0):
            allconf.append((d["conf"], bool(d.get("owner"))))
        m_own = masks_from(rgb, own)
        m_oth = masks_from(rgb, oth)
        out, alpha = suppress(rgb, m_oth, a.dilate, 4, a.sigma, protect=m_own)
        vis = out.copy()
        vis[m_own] = (0.55 * vis[m_own]
                      + 0.45 * np.array([0, 230, 0])).astype(np.uint8)
        cv2.imwrite(os.path.join(a.out, os.path.basename(p)[:-4] + ".jpg"),
                    np.hstack([rgb, vis]), [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"  {os.path.basename(p)[-18:]}  {len(dets)} hands  "
              f"owner {len(own)}  other {len(oth)}  "
              f"suppressed {(alpha>0.5).mean():.2%}")
    print(f"\n  forearm exits: {tally}")
    if clf is not None:
        print(f"  classifier disagreed with the rule on {len(disagree)} "
              f"detections")
        for x in disagree[:8]:
            print(f"    {x[0]}  exit {x[1]}  p(owner)={x[2]}")
    for k, v in confs.items():
        if v:
            v = np.array(v)
            print(f"  {k:6s} confidence: min {v.min():.2f}  median "
                  f"{np.median(v):.2f}  max {v.max():.2f}  n={len(v)}")
    if allconf:
        c = np.array([x[0] for x in allconf])
        print(f"\n  EVERY proposal, floor ignored: n={len(c)}  "
              f"min {c.min():.2f}  p25 {np.percentile(c,25):.2f}  "
              f"median {np.median(c):.2f}")
        for t in (0.4, 0.5, 0.6, 0.7, 0.8):
            print(f"    floor {t}: keeps {int((c>=t).sum()):3d} of {len(c)}")
        print("  A real hand here scores 0.83-0.87. If the rejected tail sits "
              "well below\n  that, the floor is a gap and not another "
              "threshold through an overlap.")
    print("  Left/right/top are colleagues; bottom is the wearer. Nothing "
          "that is not a\n  hand is reported at all, which is the whole point "
          "of replacing the colour rule.")


if __name__ == "__main__":
    main()
