"""Confusion-matrix-based segmentation metrics.

A single accumulated confusion matrix yields per-class IoU and Dice (F1), pixel
accuracy, and mean-over-foreground-classes summaries. Pure NumPy internals so the
metrics are unit-testable without a GPU and reusable by the fairness audit
(compute the same metrics per tissue subgroup).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


def _to_numpy(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


class SegMetrics:
    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        self.reset()

    def reset(self) -> None:
        self.cm = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, pred: np.ndarray, target: np.ndarray) -> None:
        """Accumulate predictions. `pred`/`target` are integer label maps (any shape)."""
        pred = _to_numpy(pred).reshape(-1)
        target = _to_numpy(target).reshape(-1)
        valid = (target >= 0) & (target < self.num_classes)
        idx = self.num_classes * target[valid].astype(np.int64) + pred[valid].astype(np.int64)
        self.cm += np.bincount(idx, minlength=self.num_classes ** 2).reshape(
            self.num_classes, self.num_classes
        )

    def update_logits(self, logits, target) -> None:
        """Convenience: accept raw logits (B,C,H,W) and argmax over the class dim."""
        logits = _to_numpy(logits)
        pred = logits.argmax(axis=1)
        self.update(pred, target)

    def compute(self) -> Dict[str, object]:
        cm = self.cm.astype(np.float64)
        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp
        eps = 1e-7
        iou = tp / (tp + fp + fn + eps)
        dice = 2 * tp / (2 * tp + fp + fn + eps)
        support = cm.sum(axis=1)
        present = support > 0
        fg = np.zeros(self.num_classes, dtype=bool)
        fg[1:] = present[1:]                       # foreground = non-background present classes
        result = {
            "per_class_iou": {self.class_names[i]: float(iou[i]) for i in range(self.num_classes)},
            "per_class_dice": {self.class_names[i]: float(dice[i]) for i in range(self.num_classes)},
            "support": {self.class_names[i]: int(support[i]) for i in range(self.num_classes)},
            "mean_iou_fg": float(iou[fg].mean()) if fg.any() else 0.0,
            "mean_dice_fg": float(dice[fg].mean()) if fg.any() else 0.0,
            "mean_iou_all": float(iou[present].mean()) if present.any() else 0.0,
            "mean_dice_all": float(dice[present].mean()) if present.any() else 0.0,
            "pixel_accuracy": float(tp.sum() / (cm.sum() + eps)),
        }
        return result

    @property
    def mean_dice_fg(self) -> float:
        return self.compute()["mean_dice_fg"]
