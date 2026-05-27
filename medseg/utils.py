"""Small shared utilities: seeding, JSON IO, mask colorising, overlays, plots."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Fixed colour per semantic class (RGB), used everywhere for consistent visuals.
CLASS_COLORS = np.array(
    [
        [0, 0, 0],        # 0 Background
        [228, 26, 28],    # 1 Neoplastic   (red)
        [55, 126, 184],   # 2 Inflammatory (blue)
        [77, 175, 74],    # 3 Connective   (green)
        [152, 78, 163],   # 4 Dead         (purple)
        [255, 127, 0],    # 5 Epithelial   (orange)
    ],
    dtype=np.uint8,
)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def ensure_dir(path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _json_default(o: Any):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def save_json(obj: Any, path) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def load_json(path) -> Any:
    with open(path) as f:
        return json.load(f)


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Map an integer label map (H,W) to an RGB image (H,W,3)."""
    mask = np.asarray(mask)
    palette = CLASS_COLORS
    if mask.max(initial=0) >= len(palette):
        # extend palette deterministically if more classes than defaults
        extra = np.random.default_rng(0).integers(0, 255, (mask.max() + 1 - len(palette), 3))
        palette = np.vstack([palette, extra]).astype(np.uint8)
    return palette[mask]


def overlay_mask(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Blend a colourised mask over an RGB uint8 image (background left untouched)."""
    image = np.asarray(image).astype(np.float32)
    color = colorize_mask(mask).astype(np.float32)
    fg = np.asarray(mask) > 0
    out = image.copy()
    out[fg] = (1 - alpha) * image[fg] + alpha * color[fg]
    return np.clip(out, 0, 255).astype(np.uint8)


def plot_history(history: List[Dict[str, Any]], path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    loss = [h.get("train_loss") for h in history]
    val_dice = [h.get("val_dice") for h in history]

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(epochs, loss, "C0-o", ms=3, label="train loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("train loss", color="C0")
    ax2 = ax1.twinx()
    vx = [e for e, v in zip(epochs, val_dice) if v is not None]
    vy = [v for v in val_dice if v is not None]
    ax2.plot(vx, vy, "C1-s", ms=3, label="val Dice (fg)")
    ax2.set_ylabel("val Dice (fg)", color="C1")
    fig.suptitle("Training curves")
    fig.tight_layout()
    ensure_dir(Path(path).parent)
    fig.savefig(path, dpi=120)
    plt.close(fig)
