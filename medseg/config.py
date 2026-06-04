"""Typed configuration with YAML loading and CLI-override merging.

We deliberately avoid heavyweight config frameworks (Hydra, etc.) so the project
stays easy to read in an interview setting. Everything is plain dataclasses.
"""
from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from medseg import CLASS_NAMES


@dataclass
class DataConfig:
    name: str = "pannuke"                 # dataset label
    root: str = "data/pannuke"            # folder holding the PanNuke folds
    image_size: int = 256
    num_classes: int = 6
    class_names: List[str] = field(default_factory=lambda: list(CLASS_NAMES))
    train_folds: List[int] = field(default_factory=lambda: [1, 2])
    test_fold: int = 3
    val_fraction: float = 0.15            # carved from train_folds for validation
    batch_size: int = 8
    num_workers: int = 2
    augment: bool = True
    stain_aug: bool = False               # HED stain jitter on the training set
    limit: int = 0                        # subsample N images per fold for quick runs/tests (0 = all)


@dataclass
class ModelConfig:
    arch: str = "unet"                    # unet | unetplusplus | deeplabv3plus | fpn | manet
    encoder: str = "resnet34"             # any segmentation-models-pytorch encoder
    encoder_weights: Optional[str] = "imagenet"  # None -> train encoder from scratch
    pretrained: bool = True


@dataclass
class TrainConfig:
    epochs: int = 40
    lr: float = 3e-4
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    scheduler: str = "cosine"             # "cosine" | "none"
    ce_weight: float = 1.0
    dice_weight: float = 1.0              # weight of the overlap (seg) loss term
    include_background_in_dice: bool = False
    seg_loss: str = "dice"               # dice | tversky | focal_tversky
    tversky_alpha: float = 0.3
    tversky_beta: float = 0.7
    focal_gamma: float = 1.3333
    class_weight_scheme: str = "median"  # median | sqrt | none
    class_weight_clip: float = 0.0       # cap on max class weight (0 = no cap)
    select_metric: str = "mean_dice_fg"  # mean_dice_fg | mean_dice_robust
    robust_exclude: List[str] = field(default_factory=list)
    seed: int = 42
    device: str = "auto"                  # "auto" | "mps" | "cuda" | "cpu"
    amp: bool = False                     # keep False on Apple MPS
    grad_clip: float = 0.0                # 0 disables gradient clipping
    early_stop_patience: int = 10
    val_interval: int = 1
    output_dir: str = "outputs"
    run_name: str = "pannuke_unet"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def _merge(dc: Any, d: Optional[Dict[str, Any]], strict: bool = True) -> None:
    """Recursively overwrite dataclass fields from a (possibly nested) dict.

    With strict=False, unknown keys are skipped with a warning instead of raising.
    This lets us reload older checkpoints whose saved config carried fields that
    have since been removed.
    """
    if not d:
        return
    for key, value in d.items():
        if not hasattr(dc, key):
            if strict:
                raise KeyError(f"Unknown config key {key!r} for {type(dc).__name__}")
            warnings.warn(f"Ignoring unknown config key {key!r} for {type(dc).__name__}")
            continue
        current = getattr(dc, key)
        if is_dataclass(current) and isinstance(value, dict):
            _merge(current, value, strict=strict)
        else:
            setattr(dc, key, value)


def load_config(
    path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
    strict: bool = True,
) -> Config:
    """Build a Config from defaults, then a YAML file, then explicit overrides."""
    cfg = Config()
    if path:
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        _merge(cfg, data, strict=strict)
    if overrides:
        _merge(cfg, overrides, strict=strict)
    cfg.data.num_classes = len(cfg.data.class_names)
    return cfg


def save_config(cfg: Config, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(asdict(cfg), f, sort_keys=False)
