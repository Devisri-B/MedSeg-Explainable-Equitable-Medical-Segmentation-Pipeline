"""Segmentation losses.

CombinedLoss = Cross-Entropy + soft multi-class Dice. Cross-entropy gives stable
pixel-wise gradients; Dice directly optimises overlap and is far more robust to
the severe class imbalance in histopathology (background dominates the image, the
"Dead" class is rare). Background is excluded from the Dice term by default so the
model is rewarded for the clinically meaningful foreground classes.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, include_background: bool = False, eps: float = 1.0):
        super().__init__()
        self.include_background = include_background
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        target_1h = F.one_hot(target.clamp(0, num_classes - 1), num_classes)
        target_1h = target_1h.permute(0, 3, 1, 2).float()
        if not self.include_background:
            probs = probs[:, 1:]
            target_1h = target_1h[:, 1:]
        dims = (0, 2, 3)
        intersection = (probs * target_1h).sum(dims)
        cardinality = probs.sum(dims) + target_1h.sum(dims)
        dice = (2.0 * intersection + self.eps) / (cardinality + self.eps)
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    def __init__(
        self,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        class_weights: Optional[torch.Tensor] = None,
        include_background_in_dice: bool = False,
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.dice = DiceLoss(include_background=include_background_in_dice)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.ce_weight * self.ce(logits, target) + self.dice_weight * self.dice(logits, target)


@torch.no_grad()
def estimate_class_weights(loader, num_classes: int, max_batches: int = 50) -> torch.Tensor:
    """Inverse-frequency class weights (median-frequency balancing) from a loader."""
    counts = torch.zeros(num_classes, dtype=torch.double)
    for i, batch in enumerate(loader):
        m = batch["mask"]
        counts += torch.bincount(m.reshape(-1), minlength=num_classes).double()
        if i + 1 >= max_batches:
            break
    freq = counts / counts.sum().clamp(min=1)
    median = freq[freq > 0].median()
    weights = torch.where(freq > 0, median / freq, torch.zeros_like(freq))
    return weights.float()
