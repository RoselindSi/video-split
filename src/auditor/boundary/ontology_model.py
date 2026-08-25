"""Candidate encoder + an energy whose ontology cannot be learned backwards.

Level C of the frozen ablation: morphology stops being a post-hoc veto and
enters the score that ranks candidates against each other. The detector still
proposes; this decides.

WHY THAT MIGHT MATTER, stated as the thing being tested rather than assumed.
86% of missed boundaries are `signal_present_not_top` -- a peak exists at the
right place and is ranked below others. A veto applied after ranking cannot
promote anything; it can only remove. So a veto never fixes that failure by
construction, and moving the same evidence into the ranking is the smallest
change that could.

THE ONE ARCHITECTURAL COMMITMENT. The ontology coefficients pass through
softplus, so they cannot go negative:

    E_B(i) = logit(p_detector) + r_i
             + eta_point * log P(POINT)
             - eta_no_transition * log P(NO_TRANSITION)

    eta = softplus(raw) >= 0

A free coefficient can learn that NO_TRANSITION supports a boundary, if that
happens to fit the training set, and the resulting model is indistinguishable
from a working one until it meets data where the ontology matters. The sign is
not a parameter. `ontology_constitution.check_energy_signs` asserts it after
every step.

WHAT IS DELIBERATELY ABSENT. Reset, release, disengagement and continuity all
belong in this energy and none has supervision -- adding a head named after an
ontology term without data behind it puts the name in the model and nothing
else. INTERVAL and UNOBSERVABLE are predicted and may not license an action:
25 and 17 training events. The constitution refuses both.

AND WHAT MAY NOT REACH IT. `instance_relation`, the stored action names, the
audit outcome and `is_true_boundary` are answers. A scorer reading any of them
scores its own target; `check_features` raises on the whole list.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.auditor.boundary.model import MORPHOLOGY
from src.auditor.boundary.ontology_constitution import Constitution
from src.auditor.common.temporal_encoder import TemporalEncoder

# The features the energy is allowed to read. Checked against the
# constitution's forbidden list at construction, not by inspection later.
RUNTIME_FEATURES = ("detector_score", "temporal_visual_features",
                    "morphology_evidence", "visibility_mask")


class OntologyScorer(nn.Module):
    """z_i -> (morphology logits, reranker r_i), and E_B from those + detector.

    THE ENCODER IS THE ONE ALREADY IN USE. Same TemporalEncoder, same +/-6s
    window, same 25-frame grid, same projection. Level C changes what the
    outputs feed into and nothing about how the candidate is represented --
    otherwise a gain could belong to either, and the ablation would answer
    neither question."""

    def __init__(self, in_dim, hidden=128, dropout=0.3, constitution=None):
        super().__init__()
        self.C = constitution or Constitution()
        self.C.check_features(RUNTIME_FEATURES)
        self.C.check_level("C_ontology_rerank")

        self.enc = TemporalEncoder(in_dim, hidden=hidden, dropout=dropout)
        d = self.enc.out_dim
        self.morphology = nn.Sequential(nn.Dropout(dropout),
                                        nn.Linear(d, len(MORPHOLOGY)))
        # THE RERANKER IS NOT A SECOND DETECTOR. It sees the same window and
        # answers a different question: given that something changed here, is
        # this change more like an episode boundary than the other candidates
        # in this recording. Which is why its loss is pairwise and within
        # recording, not a per-candidate binary.
        self.rerank = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, 1))

        # Raw parameters; the coefficients are softplus of these and therefore
        # never negative. Initialised small so the first steps are dominated
        # by the detector and reranker rather than by an untrained morphology.
        self.raw_eta_point = nn.Parameter(torch.tensor(-2.0))
        self.raw_eta_no_transition = nn.Parameter(torch.tensor(-2.0))

    # -- the coefficients, always non-negative -----------------------------
    def etas(self):
        return {"eta_point": F.softplus(self.raw_eta_point),
                "eta_no_transition": F.softplus(self.raw_eta_no_transition)}

    def check_signs(self):
        self.C.check_energy_signs({k: float(v) for k, v in self.etas().items()})

    def forward(self, x, mask, detector_logit):
        """x [B,T,D], mask [B,T], detector_logit [B] -> dict."""
        z = self.enc(x, mask)
        m_logits = self.morphology(z)
        r = self.rerank(z).squeeze(-1)
        logp = F.log_softmax(m_logits, -1)
        i_pt = MORPHOLOGY.index("POINT_TRANSITION")
        i_nt = MORPHOLOGY.index("NO_TRANSITION")
        e = self.etas()
        # The sign is written here, once, and cannot be relearned: POINT adds,
        # NO_TRANSITION subtracts, and both coefficients are non-negative.
        energy = (detector_logit
                  + r
                  + e["eta_point"] * logp[:, i_pt]
                  - e["eta_no_transition"] * logp[:, i_nt])
        return {"energy": energy, "rerank": r, "morphology": m_logits,
                "log_morphology": logp,
                "eta_point": e["eta_point"],
                "eta_no_transition": e["eta_no_transition"]}


def detector_logit(p, eps=1e-4):
    """p in (0,1) -> logit, clamped. The detector's score is a proposal
    strength, not a boundary probability, and the name in the constitution
    says so -- but it is the natural scale to add evidence onto."""
    p = torch.as_tensor(p, dtype=torch.float32).clamp(eps, 1 - eps)
    return torch.log(p / (1 - p))


def pairwise_ranking_loss(energy, is_positive, recording_id, margin=1.0,
                          time_s=None, max_gap_s=None):
    """E(true) > E(false_mid) + margin, WITHIN a recording.

    THE PAIRING IS THE POINT. A positive from one kitchen against a negative
    from another can be separated by recognising the kitchen, and the failure
    being attacked -- a real boundary ranked below an internal motion in the
    SAME recording -- is invisible to that. Cross-recording pairs would train
    the shortcut and score well on it.

    `max_gap_s` narrows further to temporal neighbours. A boundary at 205s
    against an internal motion at 197s is the discrimination that matters; the
    same boundary against something 400s away shares almost nothing and is an
    easier problem the model does not need help with."""
    by = {}
    for i, r in enumerate(recording_id):
        by.setdefault(r, []).append(i)
    losses = []
    for idx in by.values():
        pos = [i for i in idx if is_positive[i]]
        neg = [i for i in idx if not is_positive[i]]
        for p in pos:
            for n in neg:
                if (max_gap_s is not None and time_s is not None
                        and abs(float(time_s[p]) - float(time_s[n]))
                        > max_gap_s):
                    continue
                losses.append(F.relu(margin - energy[p] + energy[n]))
    if not losses:
        return energy.sum() * 0.0, 0
    return torch.stack(losses).mean(), len(losses)
