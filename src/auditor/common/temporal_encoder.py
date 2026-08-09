"""A small dilated TCN over the event window. Deliberately small.

355 trainable events over 193 recordings. A transformer here would have more
parameters than examples and no evidence behind it, and this project already
has one instance of a 263-feature block over ~300 samples reading AUROC 1.000
in sample -- capacity mistaken for performance. Two dilated convolutions with
dropout is the smallest thing that can express "where in the window does the
change sit and how wide is it", which is the question the summary features
could not represent.

THE MASK IS CARRIED THROUGH THE CONVOLUTIONS. Invalid grid positions are
zero-filled by the loader, and a convolution over zeros produces a confident
zero rather than an abstention -- so the mask multiplies the input, and the
pooling divides by the count of valid positions instead of the window length.
Without that, an event with half its window off the end of the recording would
read as "the second half is perfectly still", which is a morphology claim the
data does not support.

POOLING IS MASKED MEAN PLUS THE CENTRE. Mean alone discards where the change
is; the centre tap alone discards how wide it is. Both are cheap and the two
together are what the morphology head needs.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TemporalEncoder(nn.Module):
    def __init__(self, in_dim, hidden=128, dropout=0.3, kernel=3,
                 dilations=(1, 2)):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(),
                                  nn.Dropout(dropout))
        layers = []
        for d in dilations:
            layers.append(nn.Sequential(
                nn.Conv1d(hidden, hidden, kernel, padding=d * (kernel // 2),
                          dilation=d),
                nn.GELU(), nn.Dropout(dropout)))
        self.blocks = nn.ModuleList(layers)
        self.norm = nn.LayerNorm(hidden)
        self.out_dim = hidden * 2

    def forward(self, x, mask):
        """x [B,T,C], mask [B,T] bool -> [B, 2*hidden]"""
        m = mask.unsqueeze(-1).to(x.dtype)
        h = self.proj(x * m) * m
        h = h.transpose(1, 2)
        for b in self.blocks:
            # re-masked after every block: padding lets a convolution write
            # into positions that have no measurement behind them
            h = (h + b(h)) * m.transpose(1, 2)
        h = h.transpose(1, 2)
        h = self.norm(h)
        n = m.sum(1).clamp(min=1.0)
        pooled = (h * m).sum(1) / n
        centre = h[:, h.shape[1] // 2, :]
        return torch.cat([pooled, centre], -1)


def n_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
