"""Typed configuration with YAML loading and CLI-override merging.

We deliberately avoid heavyweight config frameworks (Hydra, etc.) so the project
stays easy to read in an interview setting. Everything is plain dataclasses.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from medseg import CLASS_NAMES


@dataclass
class DataConfig:
    name: str = "pannuke"                 # "pannuke" | "synthetic"
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
    synthetic_n: int = 64                 # only used when name == "synthetic"


@dataclass
class ModelConfig:
    arch: str = "unet"
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
    dice_weight: float = 1.0
    include_background_in_dice: bool = False
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


def _merge(dc: Any, d: Optional[Dict[str, Any]]) -> None:
    """Recursively overwrite dataclass fields from a (possibly nested) dict."""
    if not d:
        return
    for key, value in d.items():
        if not hasattr(dc, key):
            raise KeyError(f"Unknown config key {key!r} for {type(dc).__name__}")
        current = getattr(dc, key)
        if is_dataclass(current) and isinstance(value, dict):
            _merge(current, value)
        else:
            setattr(dc, key, value)


def load_config(
    path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Config:
    """Build a Config from defaults, then a YAML file, then explicit overrides."""
    cfg = Config()
    if path:
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        _merge(cfg, data)
    if overrides:
        _merge(cfg, overrides)
    # Keep num_classes and class_names consistent.
    cfg.data.num_classes = len(cfg.data.class_names)
    return cfg


def save_config(cfg: Config, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(asdict(cfg), f, sort_keys=False)
