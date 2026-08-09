"""The mask test the design principle implies: missing is not static.

If invalid grid positions are zero-filled and only re-masked at the pool, a
kernel near the boundary of the valid region sees a zero tap and its output
magnitude drops -- the encoder learns "the signal got smaller here", which is
indistinguishable from "the motion stopped here". That is precisely the
morphology answer the mask was supposed to withhold.

So the check is a claim about BIAS: on a constant input, invalidating the left
edge must not systematically change the activation magnitude near that edge.
Zero-filling can only ever lose taps, so it shrinks the edge on every seed;
partial convolution renormalises by the valid tap count and the residual is
symmetric. Exact equality is NOT the property -- renormalising by a count
cannot cancel taps with unequal weights, and asserting it would assert
something no masked convolution provides.

Run:
    python -m tests.test_masked_encoder
"""
from __future__ import annotations

import torch

from src.auditor.common.temporal_encoder import TemporalEncoder, PartialConv1d

T, C, HID = 25, 16, 32
N_INVALID = 4
# two dilated kernels of width 3 reach 1 + 2 = 3 positions either side
REACH = 3


def _encoder():
    torch.manual_seed(0)
    enc = TemporalEncoder(C, hidden=HID, dropout=0.0)
    enc.eval()
    return enc


def edge_magnitude_bias(encoder_factory, n_seeds=40):
    """Mean relative change in activation magnitude at the edge of the valid
    region, on a CONSTANT input, over many random initialisations.

    Constant input means nothing is happening. Any systematic magnitude change
    where the window runs out of frames is a transition manufactured from a
    missing measurement."""
    rel = []
    for s in range(n_seeds):
        torch.manual_seed(s)
        enc = encoder_factory()
        enc.eval()
        x = torch.ones(1, T, C)
        full = torch.ones(1, T, dtype=torch.bool)
        clipped = full.clone()
        clipped[:, :N_INVALID] = False
        with torch.no_grad():
            h_full, _ = enc.forward_features(x, full, pre_norm=True)
            h_clip, _ = enc.forward_features(x * clipped.unsqueeze(-1), clipped,
                                             pre_norm=True)
        edge = slice(N_INVALID, N_INVALID + REACH)
        a = h_full[:, edge].abs().mean().item()
        b = h_clip[:, edge].abs().mean().item()
        if a > 1e-8:
            rel.append((b - a) / a)
    t = torch.tensor(rel)
    return t.mean().item(), t.std().item() / max(len(rel) ** 0.5, 1)


def test_missing_frames_do_not_systematically_shrink_the_edge():
    """The principle is that missing is not static, and it is a claim about
    BIAS, not about exact equality.

    Renormalising by the count of valid taps cannot cancel taps whose weights
    differ, so no masked convolution reproduces the full-support value exactly
    on a constant input -- asserting that would be asserting something partial
    convolution does not provide, and an earlier draft of this test did. What
    it does provide is the absence of a systematic direction: over random
    initialisations the edge magnitude should be as likely to rise as to fall.
    Zero-filling has no such property; it can only lose taps, so it shrinks the
    edge every time."""
    m, se = edge_magnitude_bias(lambda: TemporalEncoder(C, hidden=HID,
                                                        dropout=0.0))
    assert abs(m) < 4 * max(se, 1e-6) or abs(m) < 0.02, (
        f"edge magnitude changes by {m:+.3%} (se {se:.3%}) when frames are "
        f"missing. That is a systematic direction, so the encoder is reading "
        f"'no measurement' as a change in the signal.")
    print(f"  edge magnitude bias {m:+.3%} (se {se:.3%})  OK")


def test_the_mask_propagates_to_positions_with_no_support():
    enc = _encoder()
    x = torch.ones(1, T, C)
    clipped = torch.ones(1, T, dtype=torch.bool)
    clipped[:, :N_INVALID] = False
    with torch.no_grad():
        _, m_clip = enc.forward_features(x * clipped.unsqueeze(-1), clipped)
    assert m_clip[0, :N_INVALID - REACH].sum() == 0, (
        "positions with no valid tap anywhere under the receptive field must "
        "stay invalid rather than being filled from nothing")
    print("  positions beyond the receptive field of any measurement stay "
          "invalid  OK")


def test_partial_conv_renormalises_a_partly_valid_kernel():
    """A kernel with one tap missing must not simply lose that tap's share."""
    torch.manual_seed(1)
    conv = PartialConv1d(1, kernel=3, dilation=1)
    with torch.no_grad():
        conv.conv.weight.fill_(1.0)
        conv.bias.fill_(0.0)
    h = torch.ones(1, 1, 5)
    m_full = torch.ones(1, 1, 5)
    m_hole = m_full.clone()
    m_hole[0, 0, 0] = 0.0
    with torch.no_grad():
        y_full, _ = conv(h, m_full)
        y_hole, _ = conv(h * m_hole, m_hole)
    # position 2 has all three taps valid in both cases
    assert abs(y_full[0, 0, 2] - y_hole[0, 0, 2]) < 1e-6
    # position 1 loses one of three taps; without renormalisation it would read
    # 2.0 against 3.0, which is the "signal got smaller" artefact
    assert abs(y_hole[0, 0, 1].item() - 3.0) < 1e-5, (
        f"a kernel with 2 of 3 valid taps produced {y_hole[0, 0, 1].item():.4f} "
        f"instead of the renormalised 3.0")
    print("  2-of-3 valid taps renormalise to the full-support magnitude  OK")


def test_all_invalid_stays_invalid():
    enc = _encoder()
    x = torch.randn(1, T, C)
    none = torch.zeros(1, T, dtype=torch.bool)
    with torch.no_grad():
        h, m = enc.forward_features(x, none)
    assert m.sum() == 0, "an entirely unmeasured window must stay unmeasured"
    out = enc(x, none)
    assert torch.isfinite(out).all(), (
        "an entirely unmeasured window must still produce finite values "
        "rather than a division by zero; the event is UNOBSERVABLE, not a nan")
    print("  a fully invalid window stays invalid and finite  OK")


if __name__ == "__main__":
    test_missing_frames_do_not_systematically_shrink_the_edge()
    test_the_mask_propagates_to_positions_with_no_support()
    test_partial_conv_renormalises_a_partly_valid_kernel()
    test_all_invalid_stays_invalid()
    print("all mask tests passed")
