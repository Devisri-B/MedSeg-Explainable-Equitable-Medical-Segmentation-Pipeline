"""Explanation audit: measure how trustworthy the model's explanations are, per subgroup.

The existing fairness audit asks "does the model segment every subgroup equally well?".
This audit asks the next question: "when the model explains itself, is the explanation
equally *trustworthy* for every subgroup?" It also records the label-free signals
(uncertainty, explanation stability, method agreement) later used to ask whether an
explanation can warn us that a segmentation is wrong.

For each audited image we explain the model's dominant predicted nucleus class with
several attribution methods and record, per method:

  loc_enrich   how much more weight sits inside the explained region than chance
  loc_gt_mass  share of weight inside the ground-truth mask of that class
  conc         how spread out the map is (0 concentrated .. 1 uniform)
  stab_rho     rank correlation with the map computed after a small stain shift
  del_auc      area under the deletion curve (lower = hiding "important" pixels hurts more)
  del_gain_*   how much faster than random deletion hurts (positive = informative)

Run:
    python -m medseg.explain.xai_audit --run outputs/pannuke_resnet50 --per-tissue 34

Results are appended row by row to <run>/xai_audit/rows.csv, so an interrupted run
resumes where it stopped.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from medseg.data import pannuke
from medseg.data.dataset import IMAGENET_MEAN, IMAGENET_STD, hed_jitter
from medseg.evaluate import load_run
from medseg.explain import xai_metrics as xm
from medseg.explain.attribution import MultiAttribution, default_layers
from medseg.utils import ensure_dir, set_seed

DELETION_METHODS = ("cam@dec", "hires@dec", "cam@enc4", "hires@enc4", "gxi", "logit", "lrp")  # "lrp" only when LRP is on
FRACTIONS = (0.10, 0.30)
MIN_REGION_PIXELS = 200          # skip classes the model barely predicts
STAIN_SIGMA, STAIN_BIAS = 0.08, 0.08  # ~1.6x the strength used in training augmentation


def brightness_bin(mean_intensity: float) -> str:
    """Same thresholds as medseg/fairness/audit.py (mean of raw 0..255 pixel values)."""
    if mean_intensity < 110:
        return "dark"
    if mean_intensity < 160:
        return "medium"
    return "bright"


def to_tensor(img_u8: np.ndarray) -> torch.Tensor:
    """uint8 (H, W, 3) -> normalised float tensor (1, 3, H, W), as the model was trained."""
    t = torch.from_numpy(np.ascontiguousarray(img_u8)).permute(2, 0, 1).float() / 255.0
    return ((t - IMAGENET_MEAN) / IMAGENET_STD).unsqueeze(0)


def select_images(types: List[str], intensities: np.ndarray, per_tissue: int, seed: int) -> np.ndarray:
    """Random sample of up to `per_tissue` images from every tissue type (seeded)."""
    rng = np.random.RandomState(seed)
    types_arr = np.asarray(types)
    chosen: List[int] = []
    for t in sorted(set(types)):
        idx = np.flatnonzero(types_arr == t)
        rng.shuffle(idx)
        chosen += idx[:per_tissue].tolist()
    return np.array(sorted(chosen))


def pick_target_class(pred: np.ndarray, num_classes: int, forced: Optional[int] = None) -> Optional[int]:
    """Most common predicted foreground class with enough pixels (or the forced class)."""
    counts = np.bincount(pred.ravel(), minlength=num_classes)
    if forced is not None:
        return forced if counts[forced] >= MIN_REGION_PIXELS else None
    counts[0] = 0
    c = int(counts.argmax())
    return c if counts[c] >= MIN_REGION_PIXELS else None


def audit_one(
    model, attr: MultiAttribution, img_u8: np.ndarray, gt: np.ndarray, num_classes: int,
    seed: int, forced_class: Optional[int] = None, lrp=None,
) -> Optional[Dict[str, float]]:
    """All measurements for one image. Returns None if there is nothing to explain."""
    rng = np.random.RandomState(seed)
    x = to_tensor(img_u8)

    # --- explain the prediction (one forward + one backward for all methods) ---
    with torch.no_grad():
        logits0 = model(x)
    pred = logits0.argmax(1)[0].numpy()
    c = pick_target_class(pred, num_classes, forced_class)
    if c is None:
        return None
    region = pred == c
    maps, _ = attr(x, c, region=torch.from_numpy(region))
    if not maps:
        return None
    row_lrp: Dict[str, float] = {}
    if lrp is not None:
        l_map, l_total, l_score = lrp(x, c, torch.from_numpy(region))
        if l_map is not None:
            maps["lrp"] = l_map
            row_lrp["lrp_retained"] = l_total / l_score if l_score else float("nan")
    gt_c = gt == c

    prob = F.softmax(logits0, dim=1)[0]
    ent = (-(prob * torch.log(prob + 1e-8)).sum(0) / np.log(num_classes)).numpy()
    row: Dict[str, float] = {
        "target_class": c,
        "region_frac": float(region.mean()),
        "gt_frac": float(gt_c.mean()),
        "dice_c": xm.dice_binary(region, gt_c),
        "entropy_region": float(ent[region].mean()),
        "entropy_img": float(ent.mean()),
        "conf_region": float(prob[c].numpy()[region].mean()),
    }
    fg = [k for k in range(1, num_classes) if (gt == k).any() or (pred == k).any()]
    d = [xm.dice_binary(pred == k, gt == k) for k in fg]
    row["dice_img_fg"] = float(np.nanmean(d)) if d else float("nan")

    # --- 1 localization, 4 concentration ---
    for name, m in maps.items():
        row[f"loc_enrich|{name}"] = xm.enrichment(m, region)
        row[f"loc_gt_mass|{name}"] = xm.mass_fraction(m, gt_c) if gt_c.any() else float("nan")
        row[f"conc|{name}"] = xm.concentration(m)

    # --- agreement between methods (do they tell the same story?) ---
    small = {k: xm.downsample(v) for k, v in maps.items()}
    row["agree|cam-hires"] = xm.spearman(small["cam@dec"], small["hires@dec"])
    row["agree|cam-gxi"] = xm.spearman(small["cam@dec"], small["gxi"])
    row["agree|hires-gxi"] = xm.spearman(small["hires@dec"], small["gxi"])
    if "lrp" in small:
        row["agree|cam-lrp"] = xm.spearman(small["cam@dec"], small["lrp"])
        row["agree|gxi-lrp"] = xm.spearman(small["gxi"], small["lrp"])
        row.update(row_lrp)
    # Circularity check: how closely does each method just re-draw the model's own output?
    for name in ("cam@dec", "hires@dec", "cam@enc4", "hires@enc4", "gxi", "lrp"):
        if name not in small:
            continue
        row[f"agree_logit|{name}"] = xm.spearman(small[name], small["logit"])

    # --- 3 stability under a small stain shift (same region, same target class) ---
    jit = hed_jitter(img_u8, rng=np.random.RandomState(seed + 1), sigma=STAIN_SIGMA, bias=STAIN_BIAS)
    maps_j, logits_j = attr(to_tensor(jit), c, region=torch.from_numpy(region))
    if lrp is not None and maps_j and "lrp" in maps:
        maps_j["lrp"] = lrp(to_tensor(jit), c, torch.from_numpy(region))[0]
    pred_j = logits_j.argmax(1)[0].numpy()
    row["stain_pred_dice"] = xm.dice_binary(pred_j == c, region)
    for name in maps:
        row[f"stab_rho|{name}"] = (
            xm.spearman(small[name], xm.downsample(maps_j[name]))
            if (maps_j and maps_j.get(name) is not None) else float("nan")
        )

    # --- 2 faithfulness: deletion test against two random baselines ---
    del_methods = [n for n in DELETION_METHODS if n in maps]
    orders = {n: xm._order_desc(maps[n], rng) for n in del_methods}
    orders["random"] = rng.permutation(region.size)
    orders["region_random"] = xm._region_random_order(region, rng)
    curves = xm.deletion_curves(model, x, region, c, orders, FRACTIONS)
    auc = {n: xm.curve_auc(v, FRACTIONS) for n, v in curves.items()}
    row["del_auc|random"] = auc["random"]
    row["del_auc|region_random"] = auc["region_random"]
    for n in del_methods:
        row[f"del_auc|{n}"] = auc[n]
        row[f"del_gain_global|{n}"] = auc["random"] - auc[n]
        row[f"del_gain_region|{n}"] = auc["region_random"] - auc[n]
    return row


def load_dataset_for_audit(args):
    """Images, labels, tissue names. PanNuke folds, or MoNuSeg tiles (labels are 0/1 foreground)."""
    if args.dataset == "monuseg":
        from medseg.data import monuseg

        images, sem, types, _ = monuseg.load(args.data_root or "data/monuseg")
        return images, sem, types
    return pannuke.load_fold(args.data_root or "data/pannuke", args.fold)


def run(args) -> Path:
    set_seed(args.seed)
    torch.set_num_threads(args.threads)
    model, cfg, _ = load_run(args.run, device=torch.device("cpu"))
    num_classes = cfg.data.num_classes
    if args.dataset == "monuseg":
        # MoNuSeg only says "nucleus or not": explain the model's foreground score.
        from medseg.explain.binary_fg import BinaryForeground

        model, num_classes = BinaryForeground(model).eval(), 2
    images, sem, types = load_dataset_for_audit(args)
    intens = images.reshape(len(images), -1).mean(axis=1)
    if args.only_dark:
        idx_all = np.array([i for i in range(len(images)) if brightness_bin(float(intens[i])) == "dark"])
    else:
        idx_all = select_images(types, intens, args.per_tissue, args.seed)
    out_dir = ensure_dir(Path(args.run) / "xai_audit")
    if args.exclude_csv:
        ex = out_dir / args.exclude_csv
        if ex.exists():
            skip = set(pd.read_csv(ex)["index"])
            idx_all = np.array([i for i in idx_all if int(i) not in skip])
    if args.limit:
        idx_all = idx_all[: args.limit]
    csv = out_dir / args.out_name
    skipped_csv = out_dir / (Path(args.out_name).stem + "_skipped.csv")   # too little foreground to explain
    done = set(pd.read_csv(csv)["index"]) if csv.exists() else set()
    if skipped_csv.exists():
        done |= set(pd.read_csv(skipped_csv)["index"])
    todo = [i for i in idx_all if int(i) not in done]
    print(f"[xai-audit] {len(idx_all)} images selected, {len(done)} already done, {len(todo)} to run", flush=True)

    attr = MultiAttribution(model, default_layers(model))
    lrp = None
    if args.lrp:
        from medseg.explain.lrp import LRPAttribution

        lrp = LRPAttribution(model)
    t0 = time.time()
    for n, i in enumerate(todo):
        row = audit_one(model, attr, images[i], sem[i], num_classes, seed=args.seed + int(i),
                        forced_class=args.force_class, lrp=lrp)
        meta = dict(index=int(i), tissue=types[i], mean_intensity=float(intens[i]),
                    brightness=brightness_bin(float(intens[i])))
        if row is None:
            # Skipped rows go to their own file: a short row appended to the results would
            # shift every column after it.
            pd.DataFrame([meta]).to_csv(skipped_csv, mode="a", header=not skipped_csv.exists(), index=False)
            continue
        row.update(meta)
        pd.DataFrame([row]).to_csv(csv, mode="a", header=not csv.exists(), index=False)
        if (n + 1) % 10 == 0:
            el = time.time() - t0
            print(f"[xai-audit] {n + 1}/{len(todo)}  {el / (n + 1):.1f}s per image  "
                  f"eta {(len(todo) - n - 1) * el / (n + 1) / 60:.0f} min", flush=True)
    attr.remove()
    print(f"[xai-audit] wrote {csv}")
    return csv


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-subgroup explanation-quality audit.")
    ap.add_argument("--run", required=True)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--dataset", choices=["pannuke", "monuseg"], default="pannuke")
    ap.add_argument("--lrp", action="store_true", help="also compute LRP (about 2x slower)")
    ap.add_argument("--only-dark", action="store_true", help="audit every dark-stained image in the fold")
    ap.add_argument("--exclude-csv", default=None, help="skip images already listed in this results file")
    ap.add_argument("--fold", type=int, default=3)
    ap.add_argument("--per-tissue", type=int, default=34)
    ap.add_argument("--limit", type=int, default=0, help="only the first N selected images (quick test)")
    ap.add_argument("--force-class", type=int, default=None, help="explain this class instead of the dominant one")
    ap.add_argument("--out-name", default="rows.csv")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
