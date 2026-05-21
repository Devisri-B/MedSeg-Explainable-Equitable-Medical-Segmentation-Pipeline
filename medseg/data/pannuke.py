"""PanNuke loading and PanNuke instance-mask -> semantic-label conversion.

PanNuke (Gamper et al., 2019/2020) ships as 3 folds, each with:
  images.npy  (N, 256, 256, 3)  float RGB in [0, 255]
  masks.npy   (N, 256, 256, 6)  per-channel *instance* maps
  types.npy   (N,)              tissue-type string per image (19 tissues)

The 6 mask channels are, in order:
  0 Neoplastic | 1 Inflammatory | 2 Connective | 3 Dead | 4 Epithelial | 5 Background

We collapse these to a single semantic label map matching medseg.CLASS_NAMES:
  0 Background | 1 Neoplastic | 2 Inflammatory | 3 Connective | 4 Dead | 5 Epithelial
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

PANNUKE_CHANNELS = [
    "Neoplastic",
    "Inflammatory",
    "Connective",
    "Dead",
    "Epithelial",
    "Background",
]


def masks_to_semantic(masks: np.ndarray) -> np.ndarray:
    """(N,H,W,6) instance channels -> (N,H,W) uint8 semantic labels 0..5."""
    masks = np.asarray(masks)
    if masks.ndim != 4 or masks.shape[-1] < 6:
        raise ValueError(f"Expected PanNuke masks (N,H,W,6); got {masks.shape}")
    inst = masks[..., :5]                       # foreground class channels
    any_fg = (inst > 0).any(axis=-1)
    # For a foreground pixel only one class channel is non-zero, so argmax selects it.
    cls = np.argmax(inst, axis=-1).astype(np.uint8) + 1   # 1..5
    return np.where(any_fg, cls, 0).astype(np.uint8)


def _find_npy(root: str | Path, fold: int, kind: str) -> Path:
    """Locate `<kind>.npy` for a fold, tolerant of the messy official folder layout."""
    root = Path(root)
    cands = sorted(root.glob(f"**/{kind}.npy"))
    if not cands:
        raise FileNotFoundError(
            f"No {kind}.npy found under {root}. Run scripts/download_data.py first."
        )
    tag = f"fold{fold}"
    marked = [c for c in cands if tag in str(c).lower().replace(" ", "").replace("_", "")]
    chosen = marked or cands
    if len(chosen) > 1:
        warnings.warn(f"Multiple {kind}.npy candidates for fold {fold}; using {chosen[0]}")
    return chosen[0]


def load_fold(root: str | Path, fold: int) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    images = np.load(_find_npy(root, fold, "images"))
    masks = np.load(_find_npy(root, fold, "masks"))
    types = np.load(_find_npy(root, fold, "types"), allow_pickle=True)
    images = np.clip(images, 0, 255).astype(np.uint8)
    # masks.npy may be the official 6-channel instance maps, or already-semantic
    # (N,H,W) label maps produced by our Hugging Face converter (prepare_from_hf).
    semantic = masks_to_semantic(masks) if masks.ndim == 4 else masks.astype(np.uint8)
    types = [str(t) for t in np.asarray(types).reshape(-1)]
    return images, semantic, types


def load_folds(root: str | Path, folds: Sequence[int]):
    imgs, msks, typs = [], [], []
    for f in folds:
        a, b, c = load_fold(root, f)
        imgs.append(a)
        msks.append(b)
        typs += c
    return np.concatenate(imgs, 0), np.concatenate(msks, 0), typs


def is_available(root: str | Path) -> bool:
    try:
        _find_npy(root, 1, "images")
        return True
    except Exception:
        return False


# Default Hugging Face mirror. RationAI/PanNuke stores, per image: an RGB `image`,
# a list of per-instance binary masks (`instances`), a parallel list of category
# labels (`categories`, ClassLabel order: Neoplastic, Inflammatory, Connective,
# Dead, Epithelial), and a `tissue` ClassLabel. Category index c maps to our
# semantic label c+1 (0 is Background).
DEFAULT_HF_REPO = "RationAI/PanNuke"
_OFFICIAL_URL = "https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke"


def prepare_from_hf(
    root: str | Path,
    repo_id: str = DEFAULT_HF_REPO,
    folds: Sequence[int] = (1, 2, 3),
    max_per_fold: int | None = None,
) -> bool:
    """Download PanNuke via the 🤗 datasets library and cache it as our .npy format.

    Converts per-instance masks + category labels into a single semantic label map
    per image, then writes images.npy / masks.npy / types.npy under root/fold{n}/.
    """
    from datasets import load_dataset

    try:
        from tqdm import tqdm
    except Exception:  # pragma: no cover
        def tqdm(x, **k):
            return x

    root = Path(root)
    for fold in folds:
        out = root / f"fold{fold}"
        if (out / "images.npy").exists():
            print(f"[pannuke] fold{fold} already cached at {out}")
            continue
        out.mkdir(parents=True, exist_ok=True)
        print(f"[pannuke] loading {repo_id} split fold{fold} via 🤗 datasets ...")
        ds = load_dataset(repo_id, split=f"fold{fold}")
        tissue_names = ds.features["tissue"].names
        n = len(ds) if max_per_fold is None else min(len(ds), max_per_fold)

        images = np.zeros((n, 256, 256, 3), np.uint8)
        masks = np.zeros((n, 256, 256), np.uint8)
        types: List[str] = []
        for i in tqdm(range(n), desc=f"fold{fold}"):
            ex = ds[i]
            images[i] = np.asarray(ex["image"].convert("RGB"), np.uint8)
            sem = np.zeros((256, 256), np.uint8)
            for inst, cat in zip(ex["instances"], ex["categories"]):
                sem[np.asarray(inst) > 0] = int(cat) + 1
            masks[i] = sem
            types.append(tissue_names[int(ex["tissue"])])

        np.save(out / "images.npy", images)
        np.save(out / "masks.npy", masks)
        np.save(out / "types.npy", np.array(types, dtype=object), allow_pickle=True)
        print(f"[pannuke] fold{fold}: cached {n} images -> {out}")
    return is_available(root)


def download(root: str | Path, repo_id: str | None = None, max_per_fold: int | None = None) -> bool:
    """Download + cache PanNuke. Returns True if the data is ready afterward."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    try:
        import datasets  # noqa: F401
    except Exception:
        datasets = None

    if datasets is not None:
        try:
            return prepare_from_hf(root, repo_id or DEFAULT_HF_REPO, max_per_fold=max_per_fold)
        except Exception as exc:  # noqa: BLE001
            print(f"[pannuke] Hugging Face datasets path failed: {exc}")

    print(
        "\n[pannuke] Could not auto-download. Options:\n"
        "  1) pip install datasets   (then re-run — pulls RationAI/PanNuke automatically)\n"
        f"  2) Manual: download the 3 folds from {_OFFICIAL_URL}\n"
        f"     and place images.npy / masks.npy / types.npy under {root}/fold1 (etc.)\n"
    )
    return is_available(root)
