"""Data-drift detection on image-level appearance statistics.

In deployed pathology AI the most common silent failure is **covariate shift**:
a new scanner or staining protocol changes the colour distribution, and accuracy
quietly drops. We summarise each image with cheap colour statistics and track the
**Population Stability Index (PSI)** of those features against a training-time
reference distribution.

PSI rule of thumb:  < 0.1 stable | 0.1-0.2 moderate shift | > 0.2 significant shift.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

FEATURE_NAMES: List[str] = [
    "R_mean", "G_mean", "B_mean",
    "R_std", "G_std", "B_std",
    "gray_mean", "gray_std",
]


def extract_features(images: np.ndarray) -> np.ndarray:
    """Per-image colour-statistics features. images: (N,H,W,3) uint8 -> (N,8)."""
    x = np.asarray(images).astype(np.float32)
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    gray = x.mean(axis=-1)
    return np.stack(
        [
            r.mean(axis=(1, 2)), g.mean(axis=(1, 2)), b.mean(axis=(1, 2)),
            r.std(axis=(1, 2)), g.std(axis=(1, 2)), b.std(axis=(1, 2)),
            gray.mean(axis=(1, 2)), gray.std(axis=(1, 2)),
        ],
        axis=1,
    )


def _severity(psi: float) -> str:
    if psi < 0.1:
        return "stable"
    if psi < 0.2:
        return "moderate"
    return "significant"


class DriftDetector:
    """Quantile-binned PSI per feature, averaged to an overall drift score."""

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self.features = []  # list of (edges, ref_frac) per feature column

    def fit(self, reference: np.ndarray) -> "DriftDetector":
        reference = np.asarray(reference, dtype=float)
        self.features = []
        for j in range(reference.shape[1]):
            edges = np.unique(np.quantile(reference[:, j], np.linspace(0, 1, self.n_bins + 1)))
            if edges.size < 2:
                edges = np.array([reference[:, j].min(), reference[:, j].max() + 1e-6])
            ref_frac = self._frac(reference[:, j], edges)
            self.features.append((edges, ref_frac))
        return self

    @staticmethod
    def _frac(x: np.ndarray, edges: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        # Additive (Laplace) smoothing keeps empty bins from blowing up the log-ratio,
        # so identical distributions score ~0 and PSI stays in a sane range.
        x = np.clip(x, edges[0], edges[-1])
        counts, _ = np.histogram(x, bins=edges)
        counts = counts.astype(float) + alpha
        return counts / counts.sum()

    def psi(self, features: np.ndarray) -> Dict[str, object]:
        if not self.features:
            raise RuntimeError("DriftDetector.fit must be called before psi().")
        features = np.asarray(features, dtype=float)
        per_feature = {}
        for j, (edges, ref_frac) in enumerate(self.features):
            cur = self._frac(features[:, j], edges)
            per_feature[FEATURE_NAMES[j]] = float(np.sum((cur - ref_frac) * np.log(cur / ref_frac)))
        overall = float(np.mean(list(per_feature.values())))
        return {
            "per_feature": per_feature,
            "overall_psi": overall,
            "severity": _severity(overall),
            "flag": overall >= 0.2,
        }
