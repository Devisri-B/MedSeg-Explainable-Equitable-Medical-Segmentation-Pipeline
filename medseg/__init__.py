"""medseg: Responsible semantic segmentation for histopathology.

A dataset-agnostic framework that pairs a multi-class segmentation model with the
"responsible AI" layer expected in regulated healthcare settings:
explainability, fairness / bias auditing, and live performance monitoring.

The default task is multi-class nucleus segmentation on PanNuke (19 tissue types,
5 nucleus classes + background), but every component is written to generalise to
other medical-imaging modalities (radiology, ophthalmology, dermatology, ...).
"""
from __future__ import annotations

__version__ = "0.1.0"

# Canonical PanNuke semantic classes. The list index *is* the integer pixel label.
#   - "Dead"       -> degraded / necrotic / apoptotic cells (a "degraded" state)
#   - "Neoplastic" -> tumour / diseased cells
# These two classes are what let us reason about "healthy vs. degraded biological
# states" and compute a tissue-degradation index (see medseg/quantify.py).
CLASS_NAMES = [
    "Background",
    "Neoplastic",
    "Inflammatory",
    "Connective",
    "Dead",
    "Epithelial",
]

NUM_CLASSES = len(CLASS_NAMES)

# Classes that represent non-healthy tissue, used by the quantification module.
DISEASED_CLASSES = ("Neoplastic",)
DEGRADED_CLASSES = ("Dead",)
