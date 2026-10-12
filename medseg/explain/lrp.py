"""Layer-wise Relevance Propagation (LRP) for a segmentation network.

What LRP does, in plain words
-----------------------------
Gradient methods ask "how much would the score change if this pixel changed?".
LRP asks a different question: "how much of the final score was *contributed* by this
pixel?". It starts with the score, then pushes that score backwards layer by layer,
splitting it among the inputs of each layer in proportion to how much each input
contributed. A pixel's final share is its relevance. Relevance is (approximately)
conserved: the shares add back up to the score that was explained.

We use the composite "epsilon-plus-flat" recipe:
    * epsilon rule (a small stabiliser, eps) on fully connected layers
    * z+ rule (only positive contributions count) on convolutions
    * flat rule on the very first convolution (spreads relevance evenly over the
      receptive field, which gives smooth, readable pixel maps)
Batch-norm layers are first merged into the convolutions before them and residual
additions are given their own relevance rule, so both are explained correctly. This
bookkeeping is done by the `zennit` library (Anders et al., 2021); the explained
quantity and everything after it is ours.

The explained quantity matches every other map in this package: the target-class logit
at the pixels of a region. The relevance seed is that logit map itself, so the total
relevance at the output equals the explained score.
"""
from __future__ import annotations

import copy
from typing import Optional, Tuple

import numpy as np
import torch

from medseg.explain.attribution import _minmax


class LRPAttribution:
    """Pixel relevance for one image, by LRP (epsilon / z+ / flat composite)."""

    RULES = ("eps_plus_flat", "eps_plus", "eps_gamma_box")

    def __init__(self, model: torch.nn.Module, rule: str = "eps_plus_flat"):
        """`rule` picks the recipe:
        eps_plus_flat  epsilon (dense), z+ (conv), flat (first conv). The default.
        eps_plus       the same without the flat rule on the first layer.
        eps_gamma_box  epsilon (dense), gamma=0.25 (conv, favours positive evidence but keeps
                       some negative), z-box on the first layer (respects the valid pixel range).
        """
        from zennit.composites import EpsilonGammaBox, EpsilonPlus, EpsilonPlusFlat
        from zennit.torchvision import ResNetCanonizer

        if rule not in self.RULES:
            raise ValueError(f"rule must be one of {self.RULES}")
        # A private copy: the canonizers rewrite parts of the network in place.
        self.model = copy.deepcopy(model).eval()
        canon = [ResNetCanonizer()]
        if rule == "eps_plus_flat":
            self.composite = EpsilonPlusFlat(canonizers=canon)
        elif rule == "eps_plus":
            self.composite = EpsilonPlus(canonizers=canon)
        else:
            from medseg.data.dataset import IMAGENET_MEAN, IMAGENET_STD  # pixel range after normalisation

            low = ((0.0 - IMAGENET_MEAN) / IMAGENET_STD).reshape(1, 3, 1, 1)
            high = ((1.0 - IMAGENET_MEAN) / IMAGENET_STD).reshape(1, 3, 1, 1)
            self.composite = EpsilonGammaBox(low=low, high=high, canonizers=canon)

    def __call__(
        self, image: torch.Tensor, target_class: int, region: torch.Tensor
    ) -> Tuple[Optional[np.ndarray], float, float]:
        """Return (relevance map in [0,1] or None, total relevance, explained score).

        `region` is a (H, W) bool mask. Relevance is the positive part of the pixel sum
        over colour channels. The two floats let callers check conservation.
        """
        sel = region.bool().unsqueeze(0)
        if int(sel.sum()) == 0:
            return None, 0.0, 0.0
        x = image.detach().clone().requires_grad_(True)
        with self.composite.context(self.model) as modified, torch.enable_grad():
            logits = modified(x)
            seed = torch.zeros_like(logits)
            seed[:, target_class] = logits[:, target_class].detach() * sel
            logits.backward(gradient=seed)
        rel = (x.grad * x).detach()                      # per-pixel, per-channel relevance
        pix = rel.sum(dim=1)[0]                          # (H, W)
        score = float(seed.sum())
        total = float(pix.sum())
        return _minmax(torch.relu(pix)).cpu().numpy(), total, score
