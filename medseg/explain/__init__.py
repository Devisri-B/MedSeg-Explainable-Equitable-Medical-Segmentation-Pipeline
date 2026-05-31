"""Explainability: Seg-Grad-CAM attributions and predictive uncertainty.

Lazy attribute loading (PEP 562) keeps `from medseg.explain import SegGradCAM`
working while avoiding an eager import of the CLI submodule, so
`python -m medseg.explain.seg_gradcam` runs without a runpy double-import warning.
"""
from __future__ import annotations

import importlib

__all__ = [
    "SegGradCAM",
    "explanation_panel",
    "predictive_entropy",
    "confidence",
    "mc_dropout_uncertainty",
]

_LAZY = {
    "SegGradCAM": "medseg.explain.seg_gradcam",
    "explanation_panel": "medseg.explain.seg_gradcam",
    "predictive_entropy": "medseg.explain.uncertainty",
    "confidence": "medseg.explain.uncertainty",
    "mc_dropout_uncertainty": "medseg.explain.uncertainty",
}


def __getattr__(name: str):
    if name in _LAZY:
        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
