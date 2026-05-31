"""Predictive uncertainty for segmentation.

Two complementary signals a reviewer can trust:
  * **Softmax entropy** (aleatoric-ish): how flat is the per-pixel class distribution.
  * **MC-Dropout** (epistemic): variance across stochastic forward passes; the
    mutual-information term highlights where the *model itself* is unsure.

All maps are normalised to [0, 1] (divided by log(num_classes)) for easy overlay.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def predictive_entropy(logits: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    probs = F.softmax(logits, dim=1)
    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
    if normalize:
        entropy = entropy / float(np.log(logits.shape[1]))
    return entropy


@torch.no_grad()
def confidence(logits: torch.Tensor) -> torch.Tensor:
    """Max softmax probability per pixel (a simple confidence proxy)."""
    return F.softmax(logits, dim=1).max(dim=1).values


def _enable_dropout(model: nn.Module) -> int:
    n = 0
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            m.train()
            n += 1
    return n


@torch.no_grad()
def mc_dropout_uncertainty(model: nn.Module, image: torch.Tensor, passes: int = 20) -> dict:
    """Monte-Carlo dropout. If the model has no dropout, falls back to a single pass."""
    was_training = model.training
    model.eval()
    n_dropout = _enable_dropout(model)
    effective = passes if n_dropout > 0 else 1

    probs = torch.stack([F.softmax(model(image), dim=1) for _ in range(effective)], dim=0)
    mean = probs.mean(dim=0)
    norm = float(np.log(mean.shape[1]))
    pred_entropy = -(mean * torch.log(mean + 1e-8)).sum(dim=1) / norm
    expected_entropy = (-(probs * torch.log(probs + 1e-8)).sum(dim=2)).mean(dim=0) / norm
    mutual_info = (pred_entropy - expected_entropy).clamp(min=0)

    if was_training:
        model.train()
    return {
        "mean_prob": mean,
        "entropy": pred_entropy,
        "mutual_information": mutual_info,   # epistemic uncertainty
        "passes_effective": effective,
        "has_dropout": n_dropout > 0,
    }
