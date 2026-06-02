"""Fairness / algorithmic-bias auditing across subgroups (lazy API)."""
from __future__ import annotations

import importlib

__all__ = ["run_audit", "disparity"]

_LAZY = {"run_audit": "medseg.fairness.audit", "disparity": "medseg.fairness.audit"}


def __getattr__(name: str):
    if name in _LAZY:
        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
