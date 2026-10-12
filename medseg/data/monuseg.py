"""MoNuSeg (Kumar et al., 2017; MICCAI 2018 challenge) as an external test set.

44-51 H&E images of 1000x1000 pixels at 40x, from several organs and hospitals, with
every nucleus outlined. It has NO nucleus types, only "nucleus or not", so a model
trained on PanNuke is tested here as a foreground detector. We cut each image into a
4x4 grid of 256x256 tiles (the size the model was trained on).

Cached as data/monuseg/{images,masks,types,source}.npy. Run once:
    python -m medseg.data.monuseg
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

HF_REPO = "RationAI/MoNuSeg"
TILE = 256
OFFSETS = (0, 248, 496, 744)         # 4 tiles per side; neighbours overlap by at most 8 px


def prepare(root: str | Path = "data/monuseg") -> Path:
    from datasets import load_dataset

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "images.npy").exists():
        return root
    ds = load_dataset(HF_REPO)
    imgs, msks, types, source = [], [], [], []
    for split in ("train", "test"):
        names = ds[split].features["tissue"].names
        for k, ex in enumerate(ds[split]):
            img = np.asarray(ex["image"].convert("RGB"), np.uint8)
            fg = np.zeros(img.shape[:2], np.uint8)
            for inst in ex["instances"]:
                fg[np.asarray(inst) > 0] = 1
            tissue = names[int(ex["tissue"])]
            for y in OFFSETS:
                for x in OFFSETS:
                    imgs.append(img[y:y + TILE, x:x + TILE])
                    msks.append(fg[y:y + TILE, x:x + TILE])
                    types.append(tissue)
                    source.append(f"{split}{k}")
    np.save(root / "images.npy", np.stack(imgs))
    np.save(root / "masks.npy", np.stack(msks))
    np.save(root / "types.npy", np.array(types))
    np.save(root / "source.npy", np.array(source))
    return root


def load(root: str | Path = "data/monuseg"):
    root = Path(root)
    prepare(root)
    return (np.load(root / "images.npy"), np.load(root / "masks.npy"),
            [str(t) for t in np.load(root / "types.npy")], np.load(root / "source.npy"))


if __name__ == "__main__":
    im, mk, ty, src = load()
    print(im.shape, mk.shape, {t: ty.count(t) for t in sorted(set(ty))})
