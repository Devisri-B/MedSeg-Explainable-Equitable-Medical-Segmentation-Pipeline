"""Add LRP to an audit that was already run without it.

Re-running every method for every image would cost hours. Instead, for the images already
in rows.csv, this computes only the LRP columns (same definitions as xai_audit) and
writes them to rows_lrp.csv, to be joined on `index`. The random-pixel deletion
baselines are reused from rows.csv, so LRP is compared with exactly the same baselines
as the other methods.

Run:  python -m medseg.explain.xai_lrp_addon --run outputs/pannuke_resnet50
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from medseg.data import pannuke
from medseg.evaluate import load_run
from medseg.explain import xai_metrics as xm
from medseg.explain.attribution import MultiAttribution, default_layers
from medseg.explain.lrp import LRPAttribution
from medseg.explain.xai_audit import (FRACTIONS, STAIN_BIAS, STAIN_SIGMA, hed_jitter, to_tensor)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data-root", default="data/pannuke")
    ap.add_argument("--fold", type=int, default=3)
    ap.add_argument("--rows", default="rows.csv")
    ap.add_argument("--out-name", default="rows_lrp.csv")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    model, cfg, _ = load_run(a.run, device=torch.device("cpu"))
    images, sem, _ = pannuke.load_fold(a.data_root, a.fold)
    out_dir = Path(a.run) / "xai_audit"
    rows = pd.read_csv(out_dir / a.rows)
    csv = out_dir / a.out_name
    done = set(pd.read_csv(csv)["index"]) if csv.exists() else set()
    todo = rows[~rows["index"].isin(done)]
    if a.limit:
        todo = todo.head(a.limit)
    attr = MultiAttribution(model, default_layers(model))
    lrp = LRPAttribution(model)
    t0 = time.time()
    for n, (_, r) in enumerate(todo.iterrows()):
        i, c = int(r["index"]), int(r["target_class"])
        seed = a.seed + i
        rng = np.random.RandomState(seed)
        x = to_tensor(images[i])
        with torch.no_grad():
            pred = model(x).argmax(1)[0].numpy()
        region = pred == c
        maps, _ = attr(x, c, region=torch.from_numpy(region))
        m, total, score = lrp(x, c, torch.from_numpy(region))
        if m is None or not maps:
            continue
        gt_c = sem[i] == c
        small = {k: xm.downsample(v) for k, v in {**maps, "lrp": m}.items()}
        row = {"index": i, "lrp_retained": total / score if score else float("nan"),
               "loc_enrich|lrp": xm.enrichment(m, region),
               "loc_gt_mass|lrp": xm.mass_fraction(m, gt_c) if gt_c.any() else float("nan"),
               "conc|lrp": xm.concentration(m),
               "agree_logit|lrp": xm.spearman(small["lrp"], small["logit"]),
               "agree|cam-lrp": xm.spearman(small["cam@dec"], small["lrp"]),
               "agree|gxi-lrp": xm.spearman(small["gxi"], small["lrp"])}
        jit = hed_jitter(images[i], rng=np.random.RandomState(seed + 1), sigma=STAIN_SIGMA, bias=STAIN_BIAS)
        mj = lrp(to_tensor(jit), c, torch.from_numpy(region))[0]
        row["stab_rho|lrp"] = xm.spearman(small["lrp"], xm.downsample(mj)) if mj is not None else float("nan")
        order = xm._order_desc(m, rng)
        curve = xm.deletion_curves(model, x, region, c, {"lrp": order}, FRACTIONS)
        auc = xm.curve_auc(curve["lrp"], FRACTIONS)
        row["del_auc|lrp"] = auc
        row["del_gain_global|lrp"] = r["del_auc|random"] - auc
        row["del_gain_region|lrp"] = r["del_auc|region_random"] - auc
        pd.DataFrame([row]).to_csv(csv, mode="a", header=not csv.exists(), index=False)
        if (n + 1) % 10 == 0:
            el = time.time() - t0
            print(f"[lrp-addon] {n + 1}/{len(todo)} {el / (n + 1):.1f}s/img eta {(len(todo) - n - 1) * el / (n + 1) / 60:.0f} min", flush=True)
    print("[lrp-addon] done", csv)


if __name__ == "__main__":
    main()
