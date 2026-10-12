"""Quantitative quality measures for attribution maps ("how good is this explanation?").

A heatmap that *looks* sensible is not evidence that it is faithful to the model.
This module turns four questions into numbers, so they can be compared across methods
and across patient/tissue/stain subgroups:

1. Localization   Does the explanation put its weight on the structure being explained?
2. Faithfulness   If we hide the pixels the explanation calls important, does the model's
                  confidence really fall faster than if we hide random pixels?
3. Stability      If the stain colours shift a little, does the explanation stay the same?
4. Concentration  How spread out is the explanation? (A map spread evenly over the whole
                  image says little.)

Everything here works on plain NumPy arrays (maps) and a PyTorch model (deletion test).
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

EPS = 1e-8


# --------------------------------------------------------------------------- basics
def dice_binary(pred: np.ndarray, gt: np.ndarray) -> float:
    """Dice overlap of two boolean masks. Returns 0.0 if both are empty-vs-nonempty."""
    p, g = pred.astype(bool), gt.astype(bool)
    denom = p.sum() + g.sum()
    if denom == 0:
        return float("nan")  # class absent from both: undefined, caller drops it
    return float(2.0 * (p & g).sum() / denom)


def downsample(m: np.ndarray, factor: int = 4) -> np.ndarray:
    """Average-pool a 2D map by `factor` (256x256 -> 64x64). Removes pixel-level noise."""
    h, w = m.shape
    h2, w2 = h - h % factor, w - w % factor
    return m[:h2, :w2].reshape(h2 // factor, factor, w2 // factor, factor).mean(axis=(1, 3))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation of two maps (flattened). NaN if either map is constant."""
    a, b = a.ravel(), b.ravel()
    if a.std() < EPS or b.std() < EPS:
        return float("nan")
    return float(spearmanr(a, b).statistic)


# ------------------------------------------------------------------ 1. localization
def mass_fraction(m: np.ndarray, mask: np.ndarray) -> float:
    """Share of the map's total weight that falls inside `mask` (0..1)."""
    total = float(m.sum())
    if total < EPS:
        return float("nan")
    return float(m[mask.astype(bool)].sum() / total)


def enrichment(m: np.ndarray, mask: np.ndarray) -> float:
    """mass_fraction divided by the mask's area fraction.

    1.0 means the map is spread like the mask is (no information); 2.0 means twice as
    much weight sits inside the mask as chance would put there. Dividing by area makes
    large and small structures comparable.
    """
    area = float(mask.mean())
    if area < EPS:
        return float("nan")
    mf = mass_fraction(m, mask)
    return mf / area if mf == mf else float("nan")


# ----------------------------------------------------------------- 4. concentration
def concentration(m: np.ndarray) -> float:
    """Normalised Shannon entropy of the map seen as a probability distribution.

    0 = all weight on one pixel (very concentrated); 1 = perfectly even spread.
    """
    p = m.ravel().astype(np.float64)
    s = p.sum()
    if s < EPS:
        return float("nan")
    p = p / s
    nz = p[p > 0]
    return float(-(nz * np.log(nz)).sum() / np.log(p.size))


# ------------------------------------------------------------------ 2. faithfulness
def _order_desc(m: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """Pixel indices from most to least important; ties broken at random."""
    noisy = m.ravel() + rng.rand(m.size) * 1e-6
    return np.argsort(-noisy)


def _region_random_order(region: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """Random order that visits the explained region first, then everything else.

    This is a stricter baseline than plain random: it already 'knows' where the
    explained structure is, so beating it means the explanation ranks pixels
    *inside* the structure better than chance.
    """
    flat = region.ravel()
    inside = np.flatnonzero(flat)
    outside = np.flatnonzero(~flat)
    rng.shuffle(inside)
    rng.shuffle(outside)
    return np.concatenate([inside, outside])


@torch.no_grad()
def _class_confidence(logits: torch.Tensor, region: torch.Tensor, target_class: int) -> np.ndarray:
    """Mean softmax probability of `target_class` over `region`, for each batch item."""
    prob = F.softmax(logits, dim=1)[:, target_class]  # (B, H, W)
    sel = region.unsqueeze(0).expand_as(prob).to(prob.dtype)
    return ((prob * sel).sum(dim=(1, 2)) / sel.sum(dim=(1, 2)).clamp(min=1)).cpu().numpy()


@torch.no_grad()
def deletion_curves(
    model: torch.nn.Module,
    image: torch.Tensor,
    region: np.ndarray,
    target_class: int,
    orders: Dict[str, np.ndarray],
    fractions: Sequence[float] = (0.05, 0.15, 0.30),
    fill: float = 0.0,
    batch_size: int = 16,
) -> Dict[str, np.ndarray]:
    """Deletion test. For each ordering, hide the top-p fraction of pixels and re-run the model.

    image    (1, 3, H, W) normalised tensor. Hidden pixels are set to `fill` (0 in
             normalised space = the ImageNet mean colour).
    region   (H, W) bool, the pixels the model assigned to `target_class`.
    orders   {name: pixel order, most important first (flat indices)}.

    Returns {name: array of normalised confidence at p = 0 and each fraction}, where the
    confidence is the mean probability of the target class over the *original* region
    divided by its value on the untouched image. Lower = the deletion hurt more.
    """
    device = image.device
    h, w = image.shape[-2:]
    n = h * w
    region_t = torch.from_numpy(region).to(device)
    base = _class_confidence(model(image), region_t, target_class)[0]
    batch, keys = [], []
    for name, order in orders.items():
        for p in fractions:
            k = int(round(p * n))
            flat = torch.ones(n, dtype=torch.bool)
            flat[torch.from_numpy(order[:k].copy())] = False
            keep = flat.view(1, 1, h, w).to(device)
            batch.append(torch.where(keep, image, torch.full_like(image, fill)))
            keys.append(name)
    out: Dict[str, list] = {name: [] for name in orders}
    for s in range(0, len(batch), batch_size):
        chunk = torch.cat(batch[s : s + batch_size], dim=0)
        conf = _class_confidence(model(chunk), region_t, target_class)
        for key, c in zip(keys[s : s + batch_size], conf):
            out[key].append(float(c))
    return {k: np.array([1.0] + [v / max(base, EPS) for v in vals]) for k, vals in out.items()}


def curve_auc(curve: np.ndarray, fractions: Sequence[float]) -> float:
    """Area under a deletion curve, normalised to the x-range (so 1.0 = no drop at all)."""
    xs = np.array([0.0] + list(fractions))
    trapezoid = getattr(np, "trapezoid", None) or np.trapz  # numpy 2 renamed trapz
    return float(trapezoid(curve, xs) / xs[-1])
