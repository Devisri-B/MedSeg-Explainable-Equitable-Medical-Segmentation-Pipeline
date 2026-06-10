"""Fairness / algorithmic-bias audit for segmentation.

A model can have excellent *aggregate* Dice while quietly failing on a subgroup
(a tissue type, a staining condition, a scanner). In healthcare that is a patient-
safety issue, and it maps directly to the IEEE 7000-series concern of
**Algorithmic Bias**. This module measures performance *per subgroup* and reports
standard disparity statistics:

  * worst-group Dice            (the number that actually bounds patient risk)
  * gap   = best - worst
  * ratio = worst / best        (>= 0.8 is a common "4/5ths rule" comfort zone)
  * std / coefficient of variation across groups

Subgroups audited: tissue type (the clinically meaningful axis) and stain
brightness bins (a proxy for scanner / staining-protocol shift).
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import torch

from medseg.metrics import SegMetrics


def _brightness_bin(img_tensor: torch.Tensor) -> str:
    from medseg.data.dataset import denormalize

    mean_intensity = float(denormalize(img_tensor).mean())
    if mean_intensity < 110:
        return "dark"
    if mean_intensity < 160:
        return "medium"
    return "bright"


def disparity(scores: Dict[str, float]) -> Dict[str, object]:
    """Disparity statistics over a {group: score} mapping (higher score = better)."""
    if not scores:
        return {}
    vals = np.array(list(scores.values()), dtype=float)
    best, worst = float(vals.max()), float(vals.min())
    ratio = worst / best if best > 0 else 0.0
    return {
        "per_group": {g: float(s) for g, s in scores.items()},
        "n_groups": len(scores),
        "best": best,
        "worst": worst,
        "best_group": max(scores, key=scores.get),
        "worst_group": min(scores, key=scores.get),
        "gap": best - worst,
        "ratio": ratio,
        "std": float(vals.std()),
        "mean": float(vals.mean()),
        "cv": float(vals.std() / (vals.mean() + 1e-8)),
        "flagged": ratio < 0.8,            # >20% relative drop for the worst group
    }


@torch.no_grad()
def run_audit(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    num_classes: int,
    class_names: List[str],
    min_support_images: int = 3,
) -> Dict[str, object]:
    model.eval()
    overall = SegMetrics(num_classes, class_names)
    by: Dict[str, Dict[str, SegMetrics]] = {
        "tissue": defaultdict(lambda: SegMetrics(num_classes, class_names)),
        "brightness": defaultdict(lambda: SegMetrics(num_classes, class_names)),
    }
    counts: Dict[str, Dict[str, int]] = {"tissue": defaultdict(int), "brightness": defaultdict(int)}

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].numpy()
        pred = model(images).argmax(dim=1).cpu().numpy()
        for i in range(pred.shape[0]):
            overall.update(pred[i], masks[i])
            t = batch["tissue"][i]
            by["tissue"][t].update(pred[i], masks[i])
            counts["tissue"][t] += 1
            b = _brightness_bin(batch["image"][i])
            by["brightness"][b].update(pred[i], masks[i])
            counts["brightness"][b] += 1

    report: Dict[str, object] = {
        "headline_dice_fg": overall.compute()["mean_dice_fg"],
        "attributes": {},
    }
    for attr, groups in by.items():
        scores, full = {}, {}
        for g, m in groups.items():
            if counts[attr][g] < min_support_images:
                continue
            res = m.compute()
            scores[g] = res["mean_dice_fg"]
            full[g] = {"mean_dice_fg": res["mean_dice_fg"], "n_images": counts[attr][g],
                       "per_class_dice": res["per_class_dice"]}
        d = disparity(scores)
        d["per_group_detail"] = full
        report["attributes"][attr] = d
    return report


def _plot(attr: str, d: Dict[str, object], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = sorted(d["per_group"], key=d["per_group"].get)
    vals = [d["per_group"][g] for g in groups]
    colors = ["#d62728" if v == d["worst"] else "#1f77b4" for v in vals]
    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(groups)), 4))
    ax.bar(groups, vals, color=colors)
    ax.axhline(d["mean"], color="gray", ls="--", lw=1, label=f"mean={d['mean']:.3f}")
    ax.set_ylabel("Dice (fg)")
    ax.set_title(f"Per-{attr} Dice  |  worst={d['worst']:.3f} ({d['worst_group']})  ratio={d['ratio']:.2f}")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _markdown(report: Dict[str, object]) -> str:
    lines = ["# Fairness audit", "",
             f"Headline foreground Dice: **{report['headline_dice_fg']:.4f}**", ""]
    for attr, d in report["attributes"].items():
        if not d:
            continue
        verdict = "DISPARITY FLAGGED" if d["flagged"] else "within the 4/5ths comfort zone"
        lines += [
            f"## By {attr}: {verdict}",
            "",
            f"- groups: {d['n_groups']}  |  best **{d['best']:.3f}** ({d['best_group']})  "
            f"|  worst **{d['worst']:.3f}** ({d['worst_group']})",
            f"- gap {d['gap']:.3f}  |  worst/best ratio {d['ratio']:.2f}  |  CV {d['cv']:.2f}",
            "",
            "| group | Dice (fg) | n images |",
            "|---|---|---|",
        ]
        for g in sorted(d["per_group"], key=d["per_group"].get):
            n = d["per_group_detail"][g]["n_images"]
            lines.append(f"| {g} | {d['per_group'][g]:.3f} | {n} |")
        lines.append("")
    lines += [
        "## Interpretation (IEEE Algorithmic Bias)",
        "",
        "A worst/best ratio below 0.8 indicates the model underserves a subgroup by "
        "more than 20% relative Dice. Mitigations to consider: class/group reweighting, "
        "targeted data collection for the worst group, stain normalisation/augmentation, "
        "group-aware thresholds, and explicit disclosure of the disparity in the Model Card.",
    ]
    return "\n".join(lines)


def main() -> None:
    from medseg.data.dataset import build_loaders
    from medseg.evaluate import load_run
    from medseg.utils import ensure_dir, save_json

    ap = argparse.ArgumentParser(description="Run the fairness / bias audit on a trained run.")
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    model, cfg, _ = load_run(args.run, device=(torch.device(args.device) if args.device else None))
    device = next(model.parameters()).device
    loaders = dict(zip(["train", "val", "test"], build_loaders(cfg)[:3]))
    report = run_audit(model, loaders[args.split], device, cfg.data.num_classes, cfg.data.class_names)

    out_dir = ensure_dir(Path(args.run) / "fairness")
    save_json(report, out_dir / "fairness_report.json")
    for attr, d in report["attributes"].items():
        if d:
            _plot(attr, d, out_dir / f"dice_by_{attr}.png")
    (out_dir / "FAIRNESS_SUMMARY.md").write_text(_markdown(report))

    print(f"[fairness] headline Dice(fg)={report['headline_dice_fg']:.4f}")
    for attr, d in report["attributes"].items():
        if d:
            flag = "FLAGGED" if d["flagged"] else "ok"
            print(f"[fairness] {attr:<11s} worst={d['worst']:.3f} ({d['worst_group']}) "
                  f"ratio={d['ratio']:.2f} -> {flag}")
    print(f"[fairness] report + charts written to {out_dir}")


if __name__ == "__main__":
    main()
