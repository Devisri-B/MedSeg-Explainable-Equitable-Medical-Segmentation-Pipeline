"""View a multi-class nucleus model as a "nucleus vs background" detector.

An external dataset such as MoNuSeg only says whether a pixel is a nucleus. To audit a
PanNuke model there, we fold its five nucleus-class scores into one foreground score
with logsumexp, a smooth "any nucleus class" score:

    fg_logit = log( exp(l1) + exp(l2) + exp(l3) + exp(l4) + exp(l5) )

The wrapper returns two logits (background, foreground), so every explanation method and
the deletion test run unchanged with target class 1. The encoder and decoder are exposed
under their usual names so the CAM layers and LRP find them.
"""
from __future__ import annotations

import torch


class BinaryForeground(torch.nn.Module):
    def __init__(self, base: torch.nn.Module):
        super().__init__()
        self.base = base

    # Exposed as properties, not attributes: registering the same module under two names
    # makes LRP hook it twice and breaks the backward pass.
    @property
    def encoder(self) -> torch.nn.Module:
        return self.base.encoder

    @property
    def decoder(self) -> torch.nn.Module:
        return self.base.decoder

    @property
    def segmentation_head(self) -> torch.nn.Module:
        return self.base.segmentation_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.base(x)
        fg = torch.logsumexp(logits[:, 1:], dim=1)
        return torch.stack([logits[:, 0], fg], dim=1)
