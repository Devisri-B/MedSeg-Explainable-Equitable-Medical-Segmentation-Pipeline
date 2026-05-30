"""Quantification: turn a semantic segmentation into interpretable readouts.

This is the bridge from "pixels" to "biology". For each image we report:
  * per-class **object counts** (connected components = individual nuclei),
  * per-class **area fractions**,
  * a **tissue-degradation index** = degraded (Dead) area / total cellular area, and
  * a **neoplastic fraction** = tumour area / total cellular area.

These are the kinds of summary statistics used to compare healthy vs. degraded
tissue and to track therapeutic response over a cohort.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from medseg import CLASS_NAMES, DEGRADED_CLASSES, DISEASED_CLASSES


def _count_objects(binary_mask: np.ndarray) -> Optional[int]:
    """Count connected components; returns None if no labelling backend is present."""
    try:
        from skimage.measure import label

        return int(label(binary_mask, connectivity=2).max())
    except Exception:
        try:
            from scipy.ndimage import label as ndlabel

            return int(ndlabel(binary_mask)[1])
        except Exception:
            return None


def quantify_mask(
    mask: np.ndarray,
    class_names: List[str] = CLASS_NAMES,
    microns_per_pixel: Optional[float] = None,
) -> Dict[str, object]:
    mask = np.asarray(mask)
    image_px = int(mask.size)
    areas, counts = {}, {}
    for idx, name in enumerate(class_names):
        m = mask == idx
        areas[name] = int(m.sum())
        counts[name] = 0 if idx == 0 else _count_objects(m)

    fg = int((mask > 0).sum())
    degraded = sum(areas.get(n, 0) for n in DEGRADED_CLASSES)
    diseased = sum(areas.get(n, 0) for n in DISEASED_CLASSES)
    healthy = max(fg - degraded - diseased, 0)
    denom = fg if fg > 0 else 1

    out: Dict[str, object] = {
        "areas_px": areas,
        "object_counts": counts,
        "image_px": image_px,
        "foreground_px": fg,
        "foreground_fraction": fg / image_px,
        "degraded_area_px": int(degraded),
        "diseased_area_px": int(diseased),
        "degradation_index": degraded / denom,      # fraction of cells that are necrotic/dead
        "neoplastic_fraction": diseased / denom,     # fraction of cells that are tumour
        "healthy_fraction": healthy / denom,
    }
    if microns_per_pixel:
        scale = microns_per_pixel ** 2
        out["areas_um2"] = {k: v * scale for k, v in areas.items()}
    return out


def quantify_batch(masks, types=None, class_names: List[str] = CLASS_NAMES):
    """Return a per-image pandas DataFrame of quantification readouts."""
    import pandas as pd

    rows = []
    for i, mask in enumerate(masks):
        q = quantify_mask(mask, class_names)
        row = {
            "index": i,
            "tissue": (types[i] if types is not None else "NA"),
            "foreground_fraction": q["foreground_fraction"],
            "degradation_index": q["degradation_index"],
            "neoplastic_fraction": q["neoplastic_fraction"],
            "healthy_fraction": q["healthy_fraction"],
        }
        for name in class_names:
            row[f"count_{name}"] = q["object_counts"][name]
            row[f"areafrac_{name}"] = q["areas_px"][name] / q["image_px"]
        rows.append(row)
    return pd.DataFrame(rows)


def _plot_summary(df, out_dir: Path, class_names: List[str]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(df["degradation_index"], bins=20, color="#9850a3")
    axes[0].set_title("Tissue-degradation index (Dead / cellular area)")
    axes[0].set_xlabel("degradation index")
    axes[0].set_ylabel("# images")

    by_tissue = df.groupby("tissue")[[f"areafrac_{c}" for c in class_names[1:]]].mean()
    bottom = np.zeros(len(by_tissue))
    for c in class_names[1:]:
        axes[1].bar(by_tissue.index, by_tissue[f"areafrac_{c}"], bottom=bottom, label=c)
        bottom += by_tissue[f"areafrac_{c}"].values
    axes[1].set_title("Mean class composition by tissue")
    axes[1].set_ylabel("area fraction")
    axes[1].legend(fontsize=7)
    axes[1].tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out_dir / "quantification_summary.png", dpi=120)
    plt.close(fig)


def main() -> None:
    import torch

    from medseg.data.dataset import build_loaders
    from medseg.evaluate import load_run
    from medseg.utils import ensure_dir, save_json

    ap = argparse.ArgumentParser(description="Quantify predictions of a trained run.")
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="test", choices=["test", "val", "train"])
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    model, cfg, _ = load_run(args.run, device=(torch.device(args.device) if args.device else None))
    device = next(model.parameters()).device
    loaders = dict(zip(["train", "val", "test"], build_loaders(cfg)[:3]))
    loader = loaders[args.split]

    pred_masks, tissues = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["image"].to(device))
            pred_masks.extend(list(logits.argmax(1).cpu().numpy()))
            tissues.extend(list(batch["tissue"]))

    df = quantify_batch(pred_masks, tissues, cfg.data.class_names)
    out_dir = ensure_dir(Path(args.run) / "quantification")
    df.to_csv(out_dir / f"{args.split}_quantification.csv", index=False)
    summary = {
        "n_images": len(df),
        "mean_degradation_index": float(df["degradation_index"].mean()),
        "mean_neoplastic_fraction": float(df["neoplastic_fraction"].mean()),
        "by_tissue_degradation_index": df.groupby("tissue")["degradation_index"].mean().to_dict(),
    }
    save_json(summary, out_dir / f"{args.split}_quantification_summary.json")
    try:
        _plot_summary(df, out_dir, cfg.data.class_names)
    except Exception as exc:  # noqa: BLE001
        print(f"[quantify] plot skipped: {exc}")

    print(f"[quantify] mean degradation index = {summary['mean_degradation_index']:.3f}")
    print(f"[quantify] saved CSV + summary to {out_dir}")


if __name__ == "__main__":
    main()
