"""Synthetic H&E-like histopathology images.

Pure-NumPy generator (no torch / skimage needed) used for:
  * the smoke test (verify the whole pipeline without downloading PanNuke),
  * the demo app when no real data is present,
  * teaching: colour is correlated with class, so a model can actually learn it.

Tissue-specific class priors are baked in so the fairness audit has real signal
(different tissues end up with different class compositions, just like PanNuke).
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

# Rough H&E RGB appearance per semantic class label (0..5).
_CLASS_RGB = {
    0: (235, 210, 222),   # Background   - pale pink
    1: (120, 55, 130),    # Neoplastic   - dark purple, large/irregular
    2: (70, 95, 175),     # Inflammatory - small, bluish
    3: (205, 150, 175),   # Connective   - pink, elongated
    4: (155, 150, 140),   # Dead         - faded grey (degraded)
    5: (175, 85, 150),    # Epithelial   - magenta
}

# Per-tissue priors over the 5 foreground classes (Neo, Inf, Con, Dead, Epi).
_TISSUE_PRIORS = {
    "Breast": [0.45, 0.12, 0.18, 0.07, 0.18],
    "Colon": [0.20, 0.12, 0.13, 0.05, 0.50],
    "Lung": [0.25, 0.30, 0.15, 0.18, 0.12],
}


def generate_synthetic(
    n: int = 64,
    image_size: int = 256,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Return (images uint8 [N,H,W,3], masks uint8 [N,H,W], tissue types [N])."""
    rng = np.random.default_rng(seed)
    tissues = list(_TISSUE_PRIORS)
    images = np.zeros((n, image_size, image_size, 3), np.uint8)
    masks = np.zeros((n, image_size, image_size), np.uint8)
    types: List[str] = []
    for i in range(n):
        tissue = tissues[i % len(tissues)]
        priors = np.asarray(_TISSUE_PRIORS[tissue], dtype=float)
        images[i], masks[i] = _make_one(rng, image_size, priors)
        types.append(tissue)
    return images, masks, types


def _make_one(rng: np.random.Generator, size: int, priors: np.ndarray):
    img = np.tile(np.asarray(_CLASS_RGB[0], np.float32), (size, size, 1))
    img += rng.normal(0, 6, (size, size, 3))
    msk = np.zeros((size, size), np.uint8)
    for _ in range(int(rng.integers(30, 70))):
        cls = int(rng.choice(np.arange(1, 6), p=priors))
        cy, cx = rng.integers(0, size, size=2)
        base = 6.0 if cls == 2 else 9.0          # inflammatory cells are smaller
        ry = rng.uniform(base * 0.7, base * 1.6)
        rx = rng.uniform(base * 0.7, base * 1.6)
        if cls == 3:                             # connective is elongated
            rx *= 1.8
        _stamp(img, msk, cls, cy, cx, ry, rx, rng)
    return np.clip(img, 0, 255).astype(np.uint8), msk


def _stamp(img, msk, cls, cy, cx, ry, rx, rng):
    size = msk.shape[0]
    y0, y1 = max(0, int(cy - ry - 1)), min(size, int(cy + ry + 2))
    x0, x1 = max(0, int(cx - rx - 1)), min(size, int(cx + rx + 2))
    if y0 >= y1 or x0 >= x1:
        return
    yy, xx = np.ogrid[y0:y1, x0:x1]
    ell = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0
    color = np.clip(np.asarray(_CLASS_RGB[cls], np.float32) + rng.normal(0, 8, 3), 0, 255)
    img[y0:y1, x0:x1][ell] = color
    msk[y0:y1, x0:x1][ell] = cls
