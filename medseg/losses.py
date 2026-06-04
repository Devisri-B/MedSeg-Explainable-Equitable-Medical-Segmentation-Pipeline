"""Segmentation losses.

CombinedLoss = Cross-Entropy + a region-overlap term (Dice, Tversky, or
Focal-Tversky). Cross-entropy gives stable pixel-wise gradients; the overlap term
directly optimises segmentation quality and is far more robust to the severe class
imbalance in histopathology (background dominates the image; the "Dead" class is
extremely rare).

For rare, hard structures, **Focal-Tversky** (Abraham & Khan, 2019) is the strong
choice: Tversky lets us penalise false-negatives (missed pixels) more than
false-positives, and the focal exponent suppresses the loss of already-easy
classes so optimisation concentrates on the hard ones.
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


class TverskyLoss(nn.Module):
    """Multi-class Tversky loss.

    TI = TP / (TP + alpha*FP + beta*FN). alpha weights false-positives, beta
    weights false-negatives. Set beta > alpha to punish missed (FN) pixels harder
    — the right bias for rare structures the model tends to under-segment.
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.7,
                 include_background: bool = False, eps: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.include_background = include_background
        self.eps = eps

    def tversky_index(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        target_1h = F.one_hot(target.clamp(0, num_classes - 1), num_classes)
        target_1h = target_1h.permute(0, 3, 1, 2).float()
        if not self.include_background:
            probs = probs[:, 1:]
            target_1h = target_1h[:, 1:]
        dims = (0, 2, 3)
        tp = (probs * target_1h).sum(dims)
        fp = (probs * (1 - target_1h)).sum(dims)
        fn = ((1 - probs) * target_1h).sum(dims)
        return (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 1.0 - self.tversky_index(logits, target).mean()


class FocalTverskyLoss(TverskyLoss):
    """Focal-Tversky loss: mean_c (1 - TI_c) ** gamma, gamma in [1, 3].

    gamma > 1 down-weights classes that already have a high Tversky index (easy),
    forcing the optimiser to focus on low-TI (hard, rare) classes such as 'Dead'.
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, gamma: float = 1.3333,
                 include_background: bool = False, eps: float = 1.0):
        super().__init__(alpha, beta, include_background, eps)
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ti = self.tversky_index(logits, target)
        return ((1.0 - ti) ** self.gamma).mean()


def build_seg_loss(name: str, include_background: bool, tversky_alpha: float,
                   tversky_beta: float, focal_gamma: float) -> nn.Module:
    name = (name or "dice").lower()
    if name == "dice":
        return DiceLoss(include_background=include_background)
    if name == "tversky":
        return TverskyLoss(tversky_alpha, tversky_beta, include_background)
    if name in ("focal_tversky", "focaltversky", "ftl"):
        return FocalTverskyLoss(tversky_alpha, tversky_beta, focal_gamma, include_background)
    raise ValueError(f"Unknown seg_loss {name!r} (use dice | tversky | focal_tversky)")


class CombinedLoss(nn.Module):
    def __init__(
        self,
        ce_weight: float = 1.0,
        seg_weight: float = 1.0,
        class_weights: Optional[torch.Tensor] = None,
        seg_loss: str = "dice",
        include_background_in_dice: bool = False,
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7,
        focal_gamma: float = 1.3333,
        dice_weight: Optional[float] = None,   # backward-compatible alias for seg_weight
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.seg_weight = dice_weight if dice_weight is not None else seg_weight
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.seg = build_seg_loss(
            seg_loss, include_background_in_dice, tversky_alpha, tversky_beta, focal_gamma
        )
        self.seg_loss_name = (seg_loss or "dice").lower()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.ce_weight * self.ce(logits, target) + self.seg_weight * self.seg(logits, target)


@torch.no_grad()
def estimate_class_weights(
    loader,
    num_classes: int,
    scheme: str = "median",
    clip: float = 0.0,
    max_batches: int = 50,
) -> torch.Tensor:
    """Class weights from pixel frequencies (median-frequency balancing).

    scheme:
      "median" -> median_freq / freq           (strong; can be huge for rare classes)
      "sqrt"   -> sqrt(median_freq / freq)      (gentler; recommended with rare classes)
      "none"   -> all ones
    clip > 0 caps the maximum weight so one ultra-rare class can't dominate / destabilise.
    """
    counts = torch.zeros(num_classes, dtype=torch.double)
    for i, batch in enumerate(loader):
        m = batch["mask"]
        counts += torch.bincount(m.reshape(-1), minlength=num_classes).double()
        if i + 1 >= max_batches:
            break
    freq = counts / counts.sum().clamp(min=1)
    median = freq[freq > 0].median()
    weights = torch.where(freq > 0, median / freq, torch.zeros_like(freq))
    scheme = (scheme or "median").lower()
    if scheme == "sqrt":
        weights = weights.sqrt()
    elif scheme == "none":
        weights = torch.where(freq > 0, torch.ones_like(freq), torch.zeros_like(freq))
    if clip and clip > 0:
        weights = weights.clamp(max=clip)
    return weights.float()
