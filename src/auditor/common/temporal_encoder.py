"""A small dilated TCN with PARTIAL convolutions over the event window.

355 trainable events over 193 recordings. A transformer here would have more
parameters than examples and no evidence behind it, and this project already
has one instance of a 263-feature block over ~300 samples reading AUROC 1.000
in sample -- capacity mistaken for performance. Two dilated convolutions with
dropout is the smallest thing that can express "where in the window does the
change sit and how wide is it", which is the question the pre/post summary
features could not represent at all.

MASKING IS DONE INSIDE THE CONVOLUTION, NOT ONLY AT THE POOL. Zero-filling the
invalid positions and re-masking afterwards is not enough: a kernel sitting
near the edge of a recording sees (valid, valid, 0) and the missing tap still
changes the output magnitude, so "no measurement here" gets encoded as "the
signal got smaller here" -- which is a morphology claim. Each layer therefore
computes

    numerator   = W * (x . m)        bias-free
    denominator = count of valid taps under the kernel
    y           = numerator * (K / denominator) + b

so a kernel with two of three taps valid is rescaled to the same magnitude as
one with three, and a position with no valid tap under it stays invalid and
propagates that forward.

WHAT THIS DOES NOT ACHIEVE, measured rather than assumed. Renormalising by the
COUNT of valid taps cannot cancel taps whose weights differ, so it does not
make the edge neutral -- it over-corrects. On a constant input over 400 random
initialisations the pre-LayerNorm edge magnitude moves +2.17% (se 0.16%) here
against -2.31% (se 0.15%) for plain zero-fill: an equal and opposite artefact,
not a removed one. Renormalising by the valid WEIGHT sum is exact in principle
and unusable in practice, since that sum crosses zero on 12% of positions and
the bias blows up to +155%. After the LayerNorm both variants fall below 0.3%,
so the norm absorbs most of what is left either way.

The conclusion drawn from that measurement is that missing frames cannot be
made neutral, so they are made VISIBLE instead: the validity mask is an
explicit input channel (see model.build_input). A head can then separate "no
measurement" from "no change", which is the distinction the normalisation was
trying and failing to enforce implicitly, and it is the same distinction
UNOBSERVABLE exists to express.

POOLING IS MASKED MEAN PLUS THE CENTRE. Mean alone discards where the change
is; the centre tap alone discards how wide it is. The morphology head needs
both, and neither is meaningful over positions that carry no measurement.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PartialConv1d(nn.Module):
    """Conv1d that renormalises by the valid tap count and propagates a mask."""

    def __init__(self, ch, kernel=3, dilation=1):
        super().__init__()
        self.pad = dilation * (kernel // 2)
        self.dilation = dilation
        self.kernel = kernel
        self.conv = nn.Conv1d(ch, ch, kernel, padding=self.pad,
                              dilation=dilation, bias=False)
        self.bias = nn.Parameter(torch.zeros(ch))
        self.register_buffer("ones", torch.ones(1, 1, kernel))

    def forward(self, h, m):
        """h [B,C,T], m [B,1,T] in {0,1} -> (y [B,C,T], m_out [B,1,T])"""
        num = self.conv(h * m)
        with torch.no_grad():
            den = F.conv1d(m, self.ones, padding=self.pad,
                           dilation=self.dilation)
            m_out = (den > 0).to(m.dtype)
        scale = self.kernel / den.clamp(min=1.0)
        y = (num * scale + self.bias.view(1, -1, 1)) * m_out
        return y, m_out


class TemporalEncoder(nn.Module):
    def __init__(self, in_dim, hidden=128, dropout=0.3, kernel=3,
                 dilations=(1, 2)):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(),
                                  nn.Dropout(dropout))
        self.convs = nn.ModuleList(
            [PartialConv1d(hidden, kernel, d) for d in dilations])
        self.acts = nn.ModuleList(
            [nn.Sequential(nn.GELU(), nn.Dropout(dropout)) for _ in dilations])
        self.norm = nn.LayerNorm(hidden)
        self.out_dim = hidden * 2

    def forward_features(self, x, mask, pre_norm=False):
        """(h [B,T,hidden], m [B,T] float). Exposed so the mask behaviour can
        be tested per position rather than only through the pooled vector.

        pre_norm returns the activations BEFORE the LayerNorm. The zero-fill
        artefact is a magnitude effect and LayerNorm rescales each position,
        so it is largely absorbed by the time the features leave this method --
        which means a test run on the post-norm output cannot tell a masked
        convolution from a naive one. The mechanism has to be checked where it
        lives."""
        m = mask.unsqueeze(-1).to(x.dtype)
        h = self.proj(x * m) * m
        h = h.transpose(1, 2)
        mm = m.transpose(1, 2)
        for conv, act in zip(self.convs, self.acts):
            y, mm_new = conv(h, mm)
            # residual only where the input position was itself measured;
            # adding h into a position it never occupied would reintroduce the
            # zero-fill the partial convolution just removed
            h = (act(y) + h * mm) * mm_new
            mm = mm_new
        h = h.transpose(1, 2)
        return (h if pre_norm else self.norm(h)), mm.squeeze(1)

    def forward(self, x, mask):
        h, m = self.forward_features(x, mask)
        m = m.unsqueeze(-1)
        n = m.sum(1).clamp(min=1.0)
        pooled = (h * m).sum(1) / n
        centre = h[:, h.shape[1] // 2, :]
        return torch.cat([pooled, centre], -1)


def n_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
