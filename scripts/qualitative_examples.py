"""Save a qualitative panel (input | ground truth | prediction) for a trained run.

This is usually the single most compelling figure for a portfolio/resume: it shows
the model's segmentation next to the annotation on real tissue.

Usage:
    python scripts/qualitative_examples.py --run outputs/pannuke_improved --num 6 --tta
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

from medseg.data.dataset import build_datasets, denormalize
from medseg.evaluate import load_run, tta_probs
from medseg.utils import CLASS_COLORS, ensure_dir, overlay_mask


def _pick_indices(ds, num: int, seed: int) -> list:
    """Prefer examples that contain the rare 'Dead' class (label 4), then fill randomly."""
    rng = np.random.default_rng(seed)
    n = len(ds)
    have_dead = []
    if hasattr(ds, "masks"):
        masks = np.asarray(ds.masks)
        have_dead = [i for i in range(n) if (masks[i] == 4).any()]
    chosen = list(rng.permutation(have_dead))[: max(num // 2, 1)] if have_dead else []
    rest = [i for i in rng.permutation(n) if i not in set(chosen)]
    chosen += rest[: num - len(chosen)]
    return chosen[:num]


def main() -> None:
    ap = argparse.ArgumentParser(description="Qualitative segmentation panel.")
    ap.add_argument("--run", required=True)
    ap.add_argument("--num", type=int, default=6)
    ap.add_argument("--split", default="test", choices=["test", "val", "train"])
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    model, cfg, _ = load_run(args.run, device=(torch.device(args.device) if args.device else None))
    device = next(model.parameters()).device
    train_ds, val_ds, test_ds = build_datasets(cfg)
    ds = {"train": train_ds, "val": val_ds, "test": test_ds}[args.split]

    idxs = _pick_indices(ds, args.num, args.seed)
    n = len(idxs)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = axes[None, :]

    for r, i in enumerate(idxs):
        item = ds[int(i)]
        img = denormalize(item["image"])
        gt = item["mask"].numpy()
        with torch.no_grad():
            x = item["image"].unsqueeze(0).to(device)
            probs = tta_probs(model, x) if args.tta else torch.softmax(model(x), 1)
            pred = probs.argmax(1)[0].cpu().numpy()
        axes[r, 0].imshow(img)
        axes[r, 1].imshow(overlay_mask(img, gt))
        axes[r, 2].imshow(overlay_mask(img, pred))
        axes[r, 0].set_ylabel(str(item["tissue"]), fontsize=9)
        if r == 0:
            axes[r, 0].set_title("input")
            axes[r, 1].set_title("ground truth")
            axes[r, 2].set_title("prediction")
        for c in range(3):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])

    handles = [
        mpatches.Patch(color=np.array(CLASS_COLORS[k]) / 255.0, label=cfg.data.class_names[k])
        for k in range(1, len(cfg.data.class_names))
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))

    out = ensure_dir(Path(args.run) / "report")
    path = out / f"qualitative_{args.split}{'_tta' if args.tta else ''}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"[qualitative] saved {path}")


if __name__ == "__main__":
    main()
