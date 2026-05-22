"""Torch Dataset / DataLoader construction for histopathology segmentation.

Returns, per item, a dict:
  image  : FloatTensor (3, H, W)  ImageNet-normalised
  mask   : LongTensor  (H, W)     semantic labels 0..num_classes-1
  tissue : str                    tissue type (used by the fairness audit)
  index  : int

Augmentation uses albumentations when available, with a NumPy fallback so the
project still runs in a minimal environment.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:
    import albumentations as A

    _HAS_ALB = True
except Exception:  # pragma: no cover
    _HAS_ALB = False

from medseg.config import Config
from medseg.data import pannuke, synthetic

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _resize(img: np.ndarray, mask: np.ndarray, size: int):
    if img.shape[0] == size and img.shape[1] == size:
        return img, mask
    from PIL import Image

    img = np.asarray(Image.fromarray(img).resize((size, size), Image.BILINEAR))
    mask = np.asarray(Image.fromarray(mask).resize((size, size), Image.NEAREST))
    return img, mask


def _build_aug(size: int):
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.1, rotate_limit=20,
                border_mode=0, p=0.5,
            ),
            A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
            A.HueSaturationValue(10, 15, 10, p=0.3),   # robustness to stain variation
        ]
    )


def _np_aug(img: np.ndarray, mask: np.ndarray):
    if np.random.rand() < 0.5:
        img, mask = img[:, ::-1], mask[:, ::-1]
    if np.random.rand() < 0.5:
        img, mask = img[::-1], mask[::-1]
    k = np.random.randint(4)
    if k:
        img, mask = np.rot90(img, k), np.rot90(mask, k)
    return np.ascontiguousarray(img), np.ascontiguousarray(mask)


class HistoDataset(Dataset):
    def __init__(self, images, masks, types, image_size=256, augment=False, normalize=True):
        assert len(images) == len(masks) == len(types)
        self.images = images
        self.masks = masks
        self.types = list(types)
        self.image_size = image_size
        self.augment = augment
        self.normalize = normalize
        self._aug = _build_aug(image_size) if (augment and _HAS_ALB) else None

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img = np.ascontiguousarray(self.images[idx])
        msk = np.ascontiguousarray(self.masks[idx]).astype(np.int64)
        img, msk = _resize(img, msk.astype(np.uint8), self.image_size)
        msk = msk.astype(np.int64)
        if self._aug is not None:
            out = self._aug(image=img, mask=msk)
            img, msk = out["image"], out["mask"]
        elif self.augment:
            img, msk = _np_aug(img, msk)

        img_t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
        if self.normalize:
            img_t = (img_t - IMAGENET_MEAN) / IMAGENET_STD
        msk_t = torch.from_numpy(np.ascontiguousarray(msk)).long()
        return {"image": img_t, "mask": msk_t, "tissue": self.types[idx], "index": idx}


def denormalize(img_t: torch.Tensor) -> np.ndarray:
    """Inverse of ImageNet normalisation -> uint8 RGB (H,W,3) for visualisation."""
    x = img_t.detach().cpu() * IMAGENET_STD + IMAGENET_MEAN
    x = (x.clamp(0, 1) * 255).permute(1, 2, 0).numpy().astype(np.uint8)
    return x


def _stratified_split(types: Sequence[str], test_size: float, seed: int):
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(types))
    strat = list(types)
    # Stratification needs >= 2 members per class; otherwise fall back to a plain split.
    _, counts = np.unique(strat, return_counts=True)
    if len(counts) < 2 or counts.min() < 2:
        strat = None
    return train_test_split(idx, test_size=test_size, random_state=seed, stratify=strat)


def build_datasets(cfg: Config) -> Tuple[HistoDataset, HistoDataset, HistoDataset]:
    d = cfg.data
    if d.name == "synthetic":
        imgs, msks, types = synthetic.generate_synthetic(d.synthetic_n, d.image_size, seed=cfg.train.seed)
        tr, tmp = _stratified_split(types, test_size=0.3, seed=cfg.train.seed)
        va, te = _stratified_split([types[i] for i in tmp], test_size=0.5, seed=cfg.train.seed)
        va, te = tmp[va], tmp[te]

        def make(ix, aug):
            return HistoDataset(imgs[ix], msks[ix], [types[i] for i in ix], d.image_size, aug)

        return make(tr, d.augment), make(va, False), make(te, False)

    # PanNuke
    tr_imgs, tr_msks, tr_types = pannuke.load_folds(d.root, d.train_folds)
    te_imgs, te_msks, te_types = pannuke.load_folds(d.root, [d.test_fold])
    tr, va = _stratified_split(tr_types, test_size=d.val_fraction, seed=cfg.train.seed)
    train_ds = HistoDataset(tr_imgs[tr], tr_msks[tr], [tr_types[i] for i in tr], d.image_size, d.augment)
    val_ds = HistoDataset(tr_imgs[va], tr_msks[va], [tr_types[i] for i in va], d.image_size, False)
    test_ds = HistoDataset(te_imgs, te_msks, te_types, d.image_size, False)
    return train_ds, val_ds, test_ds


def build_loaders(cfg: Config):
    train_ds, val_ds, test_ds = build_datasets(cfg)
    common = dict(batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers, pin_memory=False)
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)
    return train_loader, val_loader, test_loader, (train_ds, val_ds, test_ds)
