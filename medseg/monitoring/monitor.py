"""Performance monitoring with alerting + a drift-simulation demo.

`PerformanceMonitor` consumes a stream of observations (e.g. nightly batches of
incoming slides) and raises alerts when foreground Dice degrades past absolute or
relative thresholds, or when the data-drift PSI crosses its limit. The `main()`
demo simulates progressive staining/scanner drift on the held-out set so you can
*watch* accuracy fall and the alerts trigger — the story you tell for "rigorous
model performance monitoring".
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from medseg.monitoring.drift import extract_features


def corrupt_images(images: np.ndarray, severity: float, seed: int = 0) -> np.ndarray:
    """Emulate a staining/scanner shift: brightness wash, channel imbalance, noise."""
    rng = np.random.default_rng(seed)
    x = images.astype(np.float32)
    x = x * (1 - 0.4 * severity) + 255 * 0.15 * severity      # global wash-out
    x[..., 0] *= 1 + 0.30 * severity                          # red up
    x[..., 2] *= 1 - 0.30 * severity                          # blue down
    x = x + rng.normal(0, 25 * severity, x.shape)             # sensor noise
    return np.clip(x, 0, 255).astype(np.uint8)


class PerformanceMonitor:
    def __init__(
        self,
        baseline_dice: float,
        dice_drop_abs: float = 0.10,
        dice_drop_rel: float = 0.15,
        psi_threshold: float = 0.2,
        log_path: Optional[Path] = None,
    ):
        self.baseline = baseline_dice
        self.dice_drop_abs = dice_drop_abs
        self.dice_drop_rel = dice_drop_rel
        self.psi_threshold = psi_threshold
        self.log_path = Path(log_path) if log_path else None
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log_path.write_text("")
        self.records: List[Dict[str, object]] = []

    def observe(self, step: int, dice_fg: float, psi: float, label: str = "") -> Dict[str, object]:
        alerts = []
        if self.baseline - dice_fg >= self.dice_drop_abs:
            alerts.append("perf_drop_abs")
        if dice_fg <= self.baseline * (1 - self.dice_drop_rel):
            alerts.append("perf_drop_rel")
        if psi >= self.psi_threshold:
            alerts.append("data_drift")
        record = {
            "step": step, "label": label, "dice_fg": round(float(dice_fg), 4),
            "psi": round(float(psi), 4), "alerts": alerts,
            "status": "ALERT" if alerts else "ok",
        }
        self.records.append(record)
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(record) + "\n")
        return record


def _plot(records: List[Dict[str, object]], baseline: float, psi_threshold: float, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [r["step"] for r in records]
    dice = [r["dice_fg"] for r in records]
    psi = [r["psi"] for r in records]
    alert_steps = [r["step"] for r in records if r["alerts"]]

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(steps, dice, "C0-o", label="Dice (fg)")
    ax1.axhline(baseline, color="C0", ls=":", lw=1, label="baseline Dice")
    ax1.set_xlabel("monitoring step (simulated drift severity)")
    ax1.set_ylabel("Dice (fg)", color="C0")
    ax1.set_ylim(0, 1)
    ax2 = ax1.twinx()
    ax2.plot(steps, psi, "C3-s", label="PSI (drift)")
    ax2.axhline(psi_threshold, color="C3", ls=":", lw=1, label="PSI threshold")
    ax2.set_ylabel("PSI", color="C3")
    for s in alert_steps:
        ax1.axvspan(s - 0.25, s + 0.25, color="red", alpha=0.12)
    ax1.set_title("Model performance vs. data drift (red = alert)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    import torch
    from torch.utils.data import DataLoader

    from medseg.data.dataset import HistoDataset, build_loaders
    from medseg.evaluate import evaluate_loader, load_run
    from medseg.monitoring.drift import DriftDetector
    from medseg.utils import ensure_dir, save_json

    ap = argparse.ArgumentParser(description="Simulate drift and monitor performance.")
    ap.add_argument("--run", required=True)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    model, cfg, _ = load_run(args.run, device=(torch.device(args.device) if args.device else None))
    device = next(model.parameters()).device
    _, _, _, (_, _, test_ds) = build_loaders(cfg)

    # Reference = the clean (validation-time) distribution the model was signed off on.
    # Incoming "production" batches below are progressively corrupted and compared to it.
    clean_images = np.asarray(test_ds.images)
    detector = DriftDetector().fit(extract_features(clean_images))

    out_dir = ensure_dir(Path(args.run) / "monitoring")
    monitor = None
    severities = np.linspace(0.0, 0.9, args.steps)
    for step, sev in enumerate(severities):
        corrupted = corrupt_images(np.asarray(test_ds.images), float(sev), seed=step)
        ds = HistoDataset(corrupted, test_ds.masks, test_ds.types, cfg.data.image_size, augment=False)
        loader = DataLoader(ds, batch_size=cfg.data.batch_size, num_workers=0)
        dice = evaluate_loader(model, loader, device, cfg.data.num_classes, cfg.data.class_names)["mean_dice_fg"]
        psi = detector.psi(extract_features(corrupted))["overall_psi"]
        if monitor is None:
            monitor = PerformanceMonitor(baseline_dice=dice, log_path=out_dir / "monitoring_log.jsonl")
        rec = monitor.observe(step, dice, psi, label=f"severity={sev:.2f}")
        print(f"[monitor] step {step} sev {sev:.2f} | Dice {dice:.3f} | PSI {psi:.3f} | {rec['status']}"
              + (f"  -> {rec['alerts']}" if rec["alerts"] else ""))

    _plot(monitor.records, monitor.baseline, monitor.psi_threshold, out_dir / "monitoring.png")
    save_json(
        {
            "baseline_dice_fg": monitor.baseline,
            "thresholds": {
                "dice_drop_abs": monitor.dice_drop_abs,
                "dice_drop_rel": monitor.dice_drop_rel,
                "psi": monitor.psi_threshold,
            },
            "records": monitor.records,
            "first_alert_step": next((r["step"] for r in monitor.records if r["alerts"]), None),
        },
        out_dir / "monitoring_summary.json",
    )
    print(f"[monitor] log + chart written to {out_dir}")


if __name__ == "__main__":
    main()
