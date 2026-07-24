"""HAL-inspired cheap temporal features, computed directly from the cached
frozen-ViT per-frame features (extract_features_recseg.py output) -- no
model training, no VLM call.

Background: the boundary model's dominant real error mode is motion-semantic
confusion -- it fires on ANY strong visual change (repetitive wiping,
direction reversal, regrasp) rather than specifically a SEMANTIC action
change (see boundary-diagnosis-findings memory; confirmed again by the
visual auditor: hard-slice(2) tracks exactly this). The mentor's "HAL"
proposal is NOT "action state changes slowly" as a global prior (this
dataset has many sub-2s real actions, a global slow-latent assumption would
suppress them) -- it's an EVENT-GATED relative-consistency idea: within one
ongoing action, high-level state should be more stable than raw visual
motion; at a REAL transition, it can jump quickly. These features are a
cheap, non-learned approximation of that distinction, meant to be validated
as a diagnostic (do they actually separate real boundaries from motion-only
false ones on the 72 audited events?) BEFORE any classifier or contrastive
loss is built on top of them.

Given a per-frame feature sequence (feats [N,D], times [N]) for one
recording, and a candidate time t:

  short_change(t)   : 1 - cosine(mean(feats in [t-0.75s, t)),
                               mean(feats in [t, t+0.75s)))
                      Raw representation change right at t, over a window
                      short enough to catch fast (~1-2s) real actions.

  context_change(t) : same, but over a wider ±3s window. A real action-level
                      transition should look different at BOTH scales; a
                      brief motion blip (direction reversal, regrasp) often
                      shows up at the short scale but washes out at the
                      wider one.

  change_persistence(t) : does the post-t state STAY different from the
                      pre-t state a few seconds later, or does it revert
                      (i.e. a transient blip, not a real transition)?
                      1 - cosine(mean(feats in [t-3s,t)),
                                  mean(feats in [t+3s, t+6s))).

  left/right_internal_variance(t) : mean squared deviation from the window
                      mean, computed separately just before and just after
                      t (0.75s each). High internal variance with only a
                      MODEST short_change is the motion-hard-negative
                      signature: lots of movement, but the representation
                      isn't actually leaving its own neighborhood.

None of these five requires the trained boundary head at all -- they are a
property of the raw frozen features alone. (boundary_score / candidate_rank,
which DO need the trained head's saved probability sequence, are a natural
follow-on once these are validated -- not implemented here, to keep this
first diagnostic dependency-free.)

Usage: see hal_diagnostic.py for a worked CLI over the 72-event gold set.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def load_feature_caches(paths):
    """Load one or more extract_features_recseg.py .pt caches and index by
    recording_id. Later paths win on a recording_id collision (so a val
    cache can be listed after a train cache without erroring)."""
    by_id = {}
    for p in paths:
        cache = torch.load(p, weights_only=False)
        for rec in cache:
            rid = rec.get("recording_id") or rec.get("video")
            by_id[rid] = rec
    return by_id


def _pre_mask(times, t, half):
    return (times >= t - half) & (times < t)


def _post_mask(times, t, half):
    return (times >= t) & (times < t + half)


def _mean_feat(feats, mask):
    return feats[mask].mean(dim=0) if mask.any() else None


def _cos_dist(a, b):
    if a is None or b is None:
        return None
    return float(1.0 - F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


def _internal_variance(feats, mask):
    if mask.sum() < 2:
        return None
    sub = feats[mask]
    mu = sub.mean(dim=0, keepdim=True)
    return float(((sub - mu) ** 2).mean().item())


def hal_features_at(feats, times, t, short_half=0.75, context_half=3.0, variance_half=None):
    """feats: [N,D] tensor, times: [N] tensor (seconds, same recording), t:
    candidate time in seconds. Returns a dict of the 5 features above; any
    value is None if the relevant window had zero (or, for the variance
    features, fewer than 2) frames -- e.g. t near the start/end of the
    recording, or a window narrower than the frame spacing.

    variance_half: window half-width used ONLY for left/right_internal_
    variance, independent of short_half. Variance needs >=2 frames to be
    non-degenerate; at ~0.5s frame spacing (2fps features), short_half=0.75s
    gives ~1.5 frames on average -- almost always insufficient (confirmed on
    the server: only 2/72 events had a computable value at short_half=0.75).
    Defaults to max(short_half, 1.5s) so it's usable out of the box without
    silently widening the short_change window itself."""
    if not isinstance(times, torch.Tensor):
        times = torch.as_tensor(times)
    if variance_half is None:
        variance_half = max(short_half, 1.5)

    pre_s, post_s = _pre_mask(times, t, short_half), _post_mask(times, t, short_half)
    pre_c, post_c = _pre_mask(times, t, context_half), _post_mask(times, t, context_half)
    pre_v, post_v = _pre_mask(times, t, variance_half), _post_mask(times, t, variance_half)
    far_post = (times >= t + context_half) & (times < t + 2 * context_half)

    mu_pre_s, mu_post_s = _mean_feat(feats, pre_s), _mean_feat(feats, post_s)
    mu_pre_c = _mean_feat(feats, pre_c)
    mu_far_post = _mean_feat(feats, far_post)

    return {
        "short_change": _cos_dist(mu_pre_s, mu_post_s),
        "context_change": _cos_dist(mu_pre_c, _mean_feat(feats, post_c)),
        "change_persistence": _cos_dist(mu_pre_c, mu_far_post),
        "left_internal_variance": _internal_variance(feats, pre_v),
        "right_internal_variance": _internal_variance(feats, post_v),
    }


def hal_features_for_recording(rec, times_list, **kw):
    """Compute hal_features_at for every time in `times_list` against one
    recording's cached (feats, times). Returns a list of dicts, same order."""
    feats, times = rec["feats"], rec["times"]
    return [hal_features_at(feats, times, t, **kw) for t in times_list]
