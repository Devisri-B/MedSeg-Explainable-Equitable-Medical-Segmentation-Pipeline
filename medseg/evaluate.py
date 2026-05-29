"""Evaluation: per-class metrics on a held-out split, plus run (re)loading.

`evaluate_loader` is shared with training (validation each epoch) and with the
fairness audit (it can break metrics down per tissue subgroup).
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from medseg.config import Config, load_config
from medseg.data.dataset import build_loaders
from medseg.metrics import SegMetrics
from medseg.models import build_model, get_device
from medseg.utils import ensure_dir, save_json


@torch.no_grad()
def evaluate_loader(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    num_classes: int,
    class_names,
    by_tissue: bool = False,
) -> Dict[str, object]:
    model.eval()
    overall = SegMetrics(num_classes, class_names)
    per_tissue: Dict[str, SegMetrics] = defaultdict(lambda: SegMetrics(num_classes, class_names))

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].numpy()
        logits = model(images)
        pred = logits.argmax(dim=1).cpu().numpy()
        overall.update(pred, masks)
        if by_tissue:
            for i, tissue in enumerate(batch["tissue"]):
                per_tissue[tissue].update(pred[i], masks[i])

    result = overall.compute()
    if by_tissue:
        result["per_tissue"] = {t: m.compute() for t, m in sorted(per_tissue.items())}
    return result


def load_run(run_dir, device: Optional[torch.device] = None, weights: str = "best_model.pth"):
    """Reload a trained model + its config from an output run directory."""
    run_dir = Path(run_dir)
    ckpt = torch.load(run_dir / weights, map_location="cpu", weights_only=False)
    cfg = load_config(overrides=ckpt["config"])
    device = device or get_device(cfg.train.device)
    # Load weights from checkpoint -> no need to re-download pretrained encoder.
    model = build_model(cfg.data.num_classes, cfg.model.encoder, encoder_weights=None)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, cfg, ckpt


def _plot_confusion(cm: np.ndarray, class_names, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cmn = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Row-normalised confusion matrix")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a trained MedSeg-RAI run.")
    ap.add_argument("--run", required=True, help="output run directory (e.g. outputs/pannuke_unet)")
    ap.add_argument("--split", default="test", choices=["test", "val", "train"])
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = get_device(args.device) if args.device else None
    model, cfg, _ = load_run(args.run, device=device)
    device = next(model.parameters()).device
    class_names = cfg.data.class_names

    train_loader, val_loader, test_loader, _ = build_loaders(cfg)
    loader = {"train": train_loader, "val": val_loader, "test": test_loader}[args.split]

    metrics = evaluate_loader(model, loader, device, cfg.data.num_classes, class_names, by_tissue=True)

    print(f"\n=== {args.split} metrics ({cfg.data.name}) ===")
    print(f"mean Dice (fg): {metrics['mean_dice_fg']:.4f}   "
          f"mean IoU (fg): {metrics['mean_iou_fg']:.4f}   "
          f"pixel acc: {metrics['pixel_accuracy']:.4f}")
    print("\nper-class Dice:")
    for name, val in metrics["per_class_dice"].items():
        print(f"  {name:<14s} {val:.4f}")

    out_dir = ensure_dir(Path(args.run) / "evaluation")
    save_json(metrics, out_dir / f"{args.split}_metrics.json")
    # Reuse the overall confusion matrix for a plot.
    overall = SegMetrics(cfg.data.num_classes, class_names)
    for batch in loader:
        logits = model(batch["image"].to(device))
        overall.update(logits.argmax(1).cpu().numpy(), batch["mask"].numpy())
    _plot_confusion(overall.cm, class_names, out_dir / f"{args.split}_confusion.png")
    print(f"\nSaved report + confusion matrix to {out_dir}")


if __name__ == "__main__":
    main()
