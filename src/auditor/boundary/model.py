"""Boundary v1: one encoder, several masked heads. No action is decided here.

    morphology      POINT / INTERVAL / NO_TRANSITION / UNOBSERVABLE
    relation        EXACT / EARLY / LATE / DUPLICATE / NO_VALID
    offset          seconds from candidate to the corrected boundary
    width           duration of the transition
    observability   hand visibility, interaction visibility
    nuisance        camera-dominant

EVERY HEAD IS MASKED INDEPENDENTLY. An `annotation_convention` event has no
morphology target and may still have a relation target; a `camera` event has
no morphology target and does have a nuisance one. Masks are per head and per
event, not a single "clean subset" -- the old pipeline's CLEAN_BINARY filter
threw away the whole event when any part of it was unsupervised, which is how
gradual, camera, offscreen and annotation vanished from training and then
reappeared at inference as 14 of 16 wrong auto-keeps.

RELATION, OFFSET AND WIDTH ARE BUILT BUT NOT EXPECTED TO TRAIN. On the current
gold there are 6 EARLY and 4 LATE across 8 recordings, no DUPLICATE at all,
and 83 of 97 offset targets are essentially zero. The heads exist because the
architecture is the deliverable and the labels arrive later; the evaluation
refuses to report a class with too few events rather than printing a number
that a single fold could flip. Reading a relation metric off this data would
be the same error as the earlier operating points.

WIDTH HAS NO DIRECT SUPERVISION YET and is trained only through the morphology
prior in the ontology (POINT is narrow, INTERVAL is wide). It is reported, not
used by the policy.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.auditor.common.temporal_encoder import TemporalEncoder

MORPHOLOGY = ["POINT_TRANSITION", "INTERVAL_TRANSITION", "NO_TRANSITION",
              "UNOBSERVABLE"]
RELATION = ["EXACT", "EARLY", "LATE", "DUPLICATE", "NO_VALID"]
VISIBILITY = ["clear", "partial", "insufficient"]


class BoundaryModel(nn.Module):
    def __init__(self, in_dim, hidden=128, dropout=0.3):
        super().__init__()
        self.enc = TemporalEncoder(in_dim, hidden=hidden, dropout=dropout)
        d = self.enc.out_dim
        head = lambda k: nn.Sequential(nn.Dropout(dropout), nn.Linear(d, k))
        self.morphology = head(len(MORPHOLOGY))
        self.relation = head(len(RELATION))
        self.offset = head(1)
        self.log_width = head(1)
        self.hand_visibility = head(len(VISIBILITY))
        self.interaction_visibility = head(len(VISIBILITY))
        self.camera_dominant = head(1)

    def forward(self, x, mask):
        z = self.enc(x, mask)
        return {"morphology": self.morphology(z),
                "relation": self.relation(z),
                "offset": self.offset(z).squeeze(-1),
                # width is a duration, so it is predicted in log space and
                # exponentiated -- a linear head can and does emit negative
                # seconds, which is not a width
                "log_width": self.log_width(z).squeeze(-1),
                "hand_visibility": self.hand_visibility(z),
                "interaction_visibility": self.interaction_visibility(z),
                "camera_dominant": self.camera_dominant(z).squeeze(-1)}


def build_input(pg, pl, valid_g, valid_l):
    """[B,T,4*d+2]: projected global and local levels, their first
    differences, and the two validity masks as explicit channels.

    THE MASK IS AN INPUT, not only a gate. No masked convolution makes a
    missing frame neutral -- renormalising by the valid tap count over-corrects
    by about as much as zero-filling under-corrects (+2.2% against -2.3% on a
    constant input), and renormalising by the valid weight sum is unstable.
    Since the artefact cannot be removed, the model is given the information
    needed to identify it, so "no measurement here" and "nothing changed here"
    are separable rather than confounded. That is the same distinction
    UNOBSERVABLE exists to express, and hiding it from the input while asking a
    head to predict it would be incoherent.

    The differences are what a step and a ramp differ in -- levels alone let a
    head answer from the endpoints, which is the summary feature this model
    exists to replace. The first position's difference is zero by
    construction and its mask entry follows the level, so no head is asked to
    read a difference across the window edge."""
    dg = torch.zeros_like(pg)
    dl = torch.zeros_like(pl)
    dg[:, 1:] = pg[:, 1:] - pg[:, :-1]
    dl[:, 1:] = pl[:, 1:] - pl[:, :-1]
    # a difference is only a measurement where BOTH frames were measured
    vg = valid_g.clone()
    vg[:, 1:] &= valid_g[:, :-1]
    vl = valid_l.clone()
    vl[:, 1:] &= valid_l[:, :-1]
    x = torch.cat([pg * valid_g.unsqueeze(-1),
                   pl * valid_l.unsqueeze(-1),
                   dg * vg.unsqueeze(-1),
                   dl * vl.unsqueeze(-1),
                   valid_g.unsqueeze(-1).to(pg.dtype),
                   valid_l.unsqueeze(-1).to(pg.dtype)], -1)
    return x, valid_g
