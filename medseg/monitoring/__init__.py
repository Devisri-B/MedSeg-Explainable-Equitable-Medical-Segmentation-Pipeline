"""Model monitoring: data-drift detection + performance alerting (lazy API)."""
from __future__ import annotations

import importlib

__all__ = ["DriftDetector", "extract_features", "PerformanceMonitor", "corrupt_images"]

_LAZY = {
    "DriftDetector": "medseg.monitoring.drift",
    "extract_features": "medseg.monitoring.drift",
    "PerformanceMonitor": "medseg.monitoring.monitor",
    "corrupt_images": "medseg.monitoring.monitor",
}


def __getattr__(name: str):
    if name in _LAZY:
        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
