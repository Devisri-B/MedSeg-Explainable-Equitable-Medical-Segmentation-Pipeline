"""Torch Dataset / DataLoader construction for PanNuke histopathology segmentation.

Returns, per item, a dict:
  image  : FloatTensor (3, H, W)  ImageNet-normalised
  mask   : LongTensor  (H, W)     semantic labels 0..num_classes-1
  tissue : str                    tissue type (used by the fairness audit)
  index  : int

Augmentation uses albumentations when available, with a NumPy fallback so the
project still runs in a minimal environment. H&E stain jitter (HED colour
perturbation) is applied to the training set to harden the model against the
scanner and lab stain variation that is the main domain shift in histopathology.
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
from medseg.data import pannuke

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _resize(img: np.ndarray, mask: np.ndarray, size: int):
    if img.shape[0] == size and img.shape[1] == size:
        return img, mask
    from PIL import Image

    img = np.array(Image.fromarray(img).resize((size, size), Image.BILINEAR))
    mask = np.array(Image.fromarray(mask).resize((size, size), Image.NEAREST))
    return img, mask


def hed_jitter(img: np.ndarray, rng=np.random, sigma: float = 0.05, bias: float = 0.05) -> np.ndarray:
    """Tellez-style HED stain augmentation.

    Deconvolve RGB into Haematoxylin/Eosin/DAB stain channels, randomly scale and
    shift each, then recompose. Simulates realistic stain and scanner variation.
    """
    try:
        from skimage.color import hed2rgb, rgb2hed
    except Exception:
        return img  # skimage missing -> no-op
    hed = rgb2hed(img.astype(np.float32) / 255.0)
    for c in range(3):
        hed[..., c] = hed[..., c] * (1.0 + rng.uniform(-sigma, sigma)) + rng.uniform(-bias, bias)
    rgb = hed2rgb(hed)
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


def _build_aug(size: int):
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(scale=(0.9, 1.1), translate_percent=(0.0, 0.05),
                     rotate=(-20, 20), p=0.5),
            A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
            A.HueSaturationValue(10, 15, 10, p=0.3),
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
    def __init__(self, images, masks, types, image_size=256, augment=False,
                 normalize=True, stain_aug=False):
        assert len(images) == len(masks) == len(types)
        self.images = images
        self.masks = masks
        self.types = list(types)
        self.image_size = image_size
        self.augment = augment
        self.normalize = normalize
        self.stain_aug = stain_aug
        self._aug = _build_aug(image_size) if (augment and _HAS_ALB) else None

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img = np.ascontiguousarray(self.images[idx])
        msk = np.ascontiguousarray(self.masks[idx]).astype(np.int64)
        img, msk = _resize(img, msk.astype(np.uint8), self.image_size)
        msk = msk.astype(np.int64)
        if self.augment and self.stain_aug:
            img = hed_jitter(img)
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


def _maybe_limit(imgs, msks, types, limit: int, seed: int):
    """Randomly subsample to `limit` images (used for quick runs and tests)."""
    if not limit or limit <= 0 or limit >= len(imgs):
        return imgs, msks, types
    idx = np.random.default_rng(seed).permutation(len(imgs))[:limit]
    return imgs[idx], msks[idx], [types[i] for i in idx]


def build_datasets(cfg: Config) -> Tuple[HistoDataset, HistoDataset, HistoDataset]:
    d = cfg.data
    stain = getattr(d, "stain_aug", False)
    limit = getattr(d, "limit", 0)

    tr_imgs, tr_msks, tr_types = pannuke.load_folds(d.root, d.train_folds)
    te_imgs, te_msks, te_types = pannuke.load_folds(d.root, [d.test_fold])
    tr_imgs, tr_msks, tr_types = _maybe_limit(tr_imgs, tr_msks, tr_types, limit, cfg.train.seed)
    te_imgs, te_msks, te_types = _maybe_limit(te_imgs, te_msks, te_types, limit, cfg.train.seed + 1)

    tr, va = _stratified_split(tr_types, test_size=d.val_fraction, seed=cfg.train.seed)
    train_ds = HistoDataset(tr_imgs[tr], tr_msks[tr], [tr_types[i] for i in tr],
                            d.image_size, d.augment, stain_aug=stain)
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
