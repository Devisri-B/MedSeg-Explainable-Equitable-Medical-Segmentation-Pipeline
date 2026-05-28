"""Training entry point.

    python -m medseg.train --config configs/default.yaml
    python -m medseg.train --data-name synthetic --epochs 5 --run-name smoke

Writes everything needed to reproduce and audit a run to outputs/<run_name>/:
best_model.pth, config.yaml, history.csv, training_curves.png, summary.json.
"""
from __future__ import annotations

import argparse
import csv
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import torch

from medseg.config import Config, load_config, save_config
from medseg.data.dataset import build_loaders
from medseg.evaluate import evaluate_loader
from medseg.losses import CombinedLoss, estimate_class_weights
from medseg.models import build_model, get_device
from medseg.utils import ensure_dir, plot_history, save_json, set_seed

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **k):
        return x


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train MedSeg-RAI segmentation model.")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--epochs", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--device")
    p.add_argument("--data-name", choices=["pannuke", "synthetic"])
    p.add_argument("--data-root")
    p.add_argument("--encoder")
    p.add_argument("--encoder-weights", help="'none' for from-scratch / offline")
    p.add_argument("--run-name")
    p.add_argument("--synthetic-n", type=int)
    p.add_argument("--num-workers", type=int)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--no-class-weights", action="store_true",
                   help="disable median-frequency class weighting")
    return p.parse_args()


def build_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    data, model, train = {}, {}, {}
    if args.data_name:
        data["name"] = args.data_name
    if args.data_root:
        data["root"] = args.data_root
    if args.synthetic_n:
        data["synthetic_n"] = args.synthetic_n
    if args.num_workers is not None:
        data["num_workers"] = args.num_workers
    if args.no_augment:
        data["augment"] = False
    if args.encoder:
        model["encoder"] = args.encoder
    if args.encoder_weights:
        model["encoder_weights"] = None if args.encoder_weights.lower() in ("none", "null") \
            else args.encoder_weights
    if args.epochs:
        train["epochs"] = args.epochs
    if args.lr:
        train["lr"] = args.lr
    if args.batch_size:
        data["batch_size"] = args.batch_size
    if args.device:
        train["device"] = args.device
    if args.run_name:
        train["run_name"] = args.run_name
    out = {}
    if data:
        out["data"] = data
    if model:
        out["model"] = model
    if train:
        out["train"] = train
    return out


def train(cfg: Config, use_class_weights: bool = True) -> Dict[str, Any]:
    set_seed(cfg.train.seed)
    device = get_device(cfg.train.device)
    print(f"[train] device={device.type}  data={cfg.data.name}  encoder={cfg.model.encoder}")

    train_loader, val_loader, test_loader, (train_ds, val_ds, test_ds) = build_loaders(cfg)
    print(f"[train] sizes: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    model = build_model(cfg.data.num_classes, cfg.model.encoder, cfg.model.encoder_weights).to(device)

    class_weights = None
    if use_class_weights:
        class_weights = estimate_class_weights(train_loader, cfg.data.num_classes).to(device)
        print(f"[train] class weights: {[round(w, 2) for w in class_weights.tolist()]}")

    criterion = CombinedLoss(
        ce_weight=cfg.train.ce_weight,
        dice_weight=cfg.train.dice_weight,
        class_weights=class_weights,
        include_background_in_dice=cfg.train.include_background_in_dice,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scheduler = None
    if cfg.train.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.epochs)

    run_dir = ensure_dir(Path(cfg.train.output_dir) / cfg.train.run_name)
    save_config(cfg, run_dir / "config.yaml")

    history, best_dice, best_metrics, patience = [], -1.0, None, 0
    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        running, n = 0.0, 0
        t0 = time.time()
        for batch in tqdm(train_loader, desc=f"epoch {epoch}/{cfg.train.epochs}", leave=False):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            if cfg.train.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optimizer.step()
            running += loss.item() * images.size(0)
            n += images.size(0)
        if scheduler is not None:
            scheduler.step()
        train_loss = running / max(n, 1)

        val_dice = None
        if epoch % cfg.train.val_interval == 0 or epoch == cfg.train.epochs:
            val_metrics = evaluate_loader(model, val_loader, device, cfg.data.num_classes, cfg.data.class_names)
            val_dice = val_metrics["mean_dice_fg"]
            improved = val_dice > best_dice
            if improved:
                best_dice, best_metrics, patience = val_dice, val_metrics, 0
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "config": asdict(cfg),
                        "class_names": cfg.data.class_names,
                        "val_metrics": val_metrics,
                        "epoch": epoch,
                    },
                    run_dir / "best_model.pth",
                )
            else:
                patience += 1
            print(f"epoch {epoch:3d} | loss {train_loss:.4f} | val Dice(fg) {val_dice:.4f} "
                  f"| best {best_dice:.4f} | {time.time() - t0:.1f}s"
                  + ("  *" if improved else ""))

        history.append({"epoch": epoch, "train_loss": train_loss, "val_dice": val_dice})
        if patience >= cfg.train.early_stop_patience:
            print(f"[train] early stopping at epoch {epoch} (no val improvement for "
                  f"{cfg.train.early_stop_patience} checks)")
            break

    # Persist history + curves + summary.
    with open(run_dir / "history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_dice"])
        writer.writeheader()
        writer.writerows(history)
    try:
        plot_history(history, run_dir / "training_curves.png")
    except Exception as exc:  # noqa: BLE001
        print(f"[train] could not plot history: {exc}")

    summary = {
        "run_name": cfg.train.run_name,
        "best_val_dice_fg": best_dice,
        "best_val_metrics": best_metrics,
        "epochs_ran": len(history),
        "device": device.type,
    }
    save_json(summary, run_dir / "summary.json")
    print(f"\n[train] done. best val Dice(fg)={best_dice:.4f}. Artifacts in {run_dir}")
    return summary


def main() -> None:
    args = parse_args()
    config_path = args.config if args.config and Path(args.config).exists() else None
    cfg = load_config(config_path, overrides=build_overrides(args))
    train(cfg, use_class_weights=not args.no_class_weights)


if __name__ == "__main__":
    main()
