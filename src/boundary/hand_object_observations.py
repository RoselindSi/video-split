"""The observation schema layers 2 and 3 are allowed to depend on.

One record per hand per frame, in a vocabulary no detector owns. 100DOH,
EgoHOS and an open-vocabulary detector disagree about almost everything --
whether an object is a box or a mask, whether contact is a class or a
distance, whether hands carry an identity -- and relation/reset/continuity
logic written against any one of them would have to be rewritten to try
another. The pilot exists precisely to find out whether the first choice
works, so binding to it would defeat the experiment.

WHAT THE SCHEMA REFUSES TO DO IS FILL IN GAPS. `hand_visible` and
`object_visible` are separate flags and neither implies the other: a detector
that sees a hand and no object is a different observation from one that sees
nothing, and the second must never be recorded as "hand touching nothing".
That distinction is the one the layer-3 fixtures caught the logic getting
wrong, and it is cheaper to enforce here once than in every consumer.

CONTACT ARRIVES AS A CLASS FROM SOME DETECTORS AND AS GEOMETRY FROM OTHERS.
Both are kept. `contact_state` is the detector's own verdict where it has one,
`contact_evidence` carries the geometry that a proxy would have used, and
`contact_source` records which. A pilot that concludes "contact is observable"
has to be able to say whether that was the detector's judgement or a distance
threshold, because only the first survives a change of backend.

OBJECT IDENTITY IS NOT THE DETECTOR'S. Object track ids are assigned by the
extractor across frames, never taken from a per-frame class label: two bowls
carry the same label and are not the same instance, and instance continuity
is the whole discriminating quantity in layer 2. `object_label` is recorded
when a backend supplies one, and is documentation, not identity.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np

# our vocabulary. 100DOH's five classes map onto it; a geometric proxy
# produces only the first three.
CONTACT = "contact"          # touching a manipulable object
NEAR = "near"                # close, not touching
FREE = "free"                # hand present, touching nothing
SELF = "self_contact"        # hand touching the other hand or the body
OTHER_PERSON = "other_person"
UNKNOWN = "unknown"          # hand seen, contact undetermined
CONTACT_STATES = (CONTACT, NEAR, FREE, SELF, OTHER_PERSON, UNKNOWN)

# 100DOH's contactstate head, in its own order
DOH_CONTACT = {0: FREE, 1: SELF, 2: OTHER_PERSON, 3: CONTACT, 4: CONTACT}
# classes 3 and 4 are portable and stationary objects. Both are contact for
# our purpose -- a boundary is no less real for the object being a countertop
# -- but the distinction is kept in `object_kind` because "released the pan"
# and "let go of the counter" are not equally likely to end an action.
DOH_OBJECT_KIND = {3: "portable", 4: "stationary"}


@dataclass
class HandObservation:
    """One hand, one frame."""
    rel_t: float                       # seconds from the candidate
    abs_t: float
    hand_track_id: int | None = None
    hand_visible: bool = False
    hand_box: tuple | None = None      # (x0, y0, x1, y1) in eye-frame pixels
    hand_score: float = float("nan")
    handedness: str | None = None
    object_visible: bool = False
    object_box: tuple | None = None
    object_score: float = float("nan")
    object_track_id: int | None = None
    object_label: str | None = None    # documentation only, never identity
    object_kind: str | None = None     # portable / stationary, when known
    contact_state: str = UNKNOWN
    contact_source: str = "none"       # "detector" | "geometry" | "none"
    contact_evidence: dict = field(default_factory=dict)   # iou, gap
    detector_confidence: float = float("nan")
    hand_mask_rle: dict | None = None
    object_mask_rle: dict | None = None

    def __post_init__(self):
        if self.contact_state not in CONTACT_STATES:
            raise ValueError(f"contact_state {self.contact_state!r} not in "
                             f"{CONTACT_STATES}")
        # A hand that was not seen cannot be touching anything, and an object
        # box without a visible object is a stale box from a previous frame
        # that a consumer would read as an observation.
        if not self.hand_visible:
            if self.contact_state not in (UNKNOWN,):
                raise ValueError("hand_visible=False requires contact_state="
                                 f"{UNKNOWN!r}, got {self.contact_state!r}")
            if self.object_box is not None or self.object_visible:
                raise ValueError("hand_visible=False cannot carry an object")
        if self.contact_state in (CONTACT, NEAR) and not self.object_visible:
            raise ValueError(f"contact_state={self.contact_state!r} requires "
                             f"object_visible=True -- contact with an object "
                             f"that was never detected is not an observation")

    def to_relation(self):
        """The record layers 1-3 consume, or None when nothing was observed.

        None and FREE are different and stay different: None means the hand
        was not seen, FREE means it was seen holding nothing. reset_events
        counts a release from the second and refuses to infer one from the
        first."""
        if not self.hand_visible:
            return None
        return {"object": self.object_track_id,
                "state": self.contact_state,
                "label": self.object_label,
                "iou": self.contact_evidence.get("iou", float("nan")),
                "gap": self.contact_evidence.get("gap", float("nan")),
                "rel_xy": self.rel_xy()}

    def rel_xy(self):
        """Hand centre in the object's own frame, so a cup and a pan compare."""
        if not (self.hand_box and self.object_box):
            return (float("nan"), float("nan"))
        hb, ob = self.hand_box, self.object_box
        hc = ((hb[0] + hb[2]) / 2.0, (hb[1] + hb[3]) / 2.0)
        w, h = max(ob[2] - ob[0], 1e-6), max(ob[3] - ob[1], 1e-6)
        return ((hc[0] - ob[0]) / w, (hc[1] - ob[1]) / h)

    def as_row(self):
        d = asdict(self)
        for k in ("hand_box", "object_box"):
            b = d.pop(k)
            d[k] = "" if b is None else " ".join(f"{v:.1f}" for v in b)
        ev = d.pop("contact_evidence") or {}
        d["contact_iou"] = ev.get("iou", "")
        d["contact_gap"] = ev.get("gap", "")
        d.pop("hand_mask_rle"), d.pop("object_mask_rle")
        return d


@dataclass
class RawInteraction:
    """What a backend returns for one detected hand, before tracking.

    Deliberately minimal. A backend supplies boxes, its own contact verdict if
    it has one, and confidences; it never supplies a track id, because
    identity across frames is the extractor's job and a per-frame detector has
    no basis for it."""
    hand_box: tuple
    hand_score: float = float("nan")
    handedness: str | None = None
    object_box: tuple | None = None
    object_score: float = float("nan")
    object_label: str | None = None
    object_kind: str | None = None
    contact_state: str | None = None    # None -> the extractor uses geometry
    mask: dict | None = None


def frame_summary(obs):
    """Per-frame counts, for the QA report and nothing else."""
    return {"n_hands": sum(1 for o in obs if o.hand_visible),
            "n_objects": len({o.object_track_id for o in obs
                              if o.object_visible and o.object_track_id is not None}),
            "n_contact": sum(1 for o in obs if o.contact_state == CONTACT)}


def series_for_hand(frames, hand_track_id):
    """The per-frame relation series for one hand, aligned to `frames`.

    Missing frames become None rather than being dropped, because layers 2
    and 3 index by frame and a compacted series would silently close a gap --
    the same way an interpolated stretch once scored as the most stable
    trajectory in the local-crop work."""
    out = []
    for fr in frames:
        hit = [o for o in fr if o.hand_track_id == hand_track_id]
        out.append(hit[0].to_relation() if hit else None)
    return out


def dominant_hand(frames):
    """The hand track seen in most frames. Layers 2 and 3 are single-hand;
    running them on whichever hand happens to be first in each frame would
    interleave two hands into one trajectory."""
    c = {}
    for fr in frames:
        for o in fr:
            if o.hand_visible and o.hand_track_id is not None:
                c[o.hand_track_id] = c.get(o.hand_track_id, 0) + 1
    return max(c, key=c.get) if c else None


def colour_signature(img, box, bins=8):
    """Coarse RGB histogram of a box, for appearance-assisted tracking.

    Not a re-identification embedding and not meant as one. It exists to break
    ties that IoU alone cannot -- two same-class objects side by side -- and to
    survive a short occlusion. A signature this crude will fail on two objects
    of the same colour, which is why the pilot reports track continuity for a
    human to look at instead of assuming it."""
    if box is None:
        return None
    h, w = img.shape[:2]
    x0, y0, x1, y1 = (int(max(0, min(w - 1, box[0]))), int(max(0, min(h - 1, box[1]))),
                      int(max(1, min(w, box[2]))), int(max(1, min(h, box[3]))))
    if x1 <= x0 or y1 <= y0:
        return None
    crop = img[y0:y1, x0:x1].reshape(-1, 3)
    hist = np.concatenate([np.histogram(crop[:, c], bins=bins, range=(0, 256))[0]
                           for c in range(3)]).astype(float)
    n = hist.sum()
    return hist / n if n > 0 else None


def signature_similarity(a, b):
    if a is None or b is None:
        return 0.0
    d = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / d) if d > 0 else 0.0
