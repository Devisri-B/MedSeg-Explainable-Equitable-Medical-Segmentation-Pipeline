"""Is the LRP result an accident of one rule recipe?

LRP has choices (which rule on which layer). If the faithfulness of LRP changed a lot with
the recipe, one recipe would be a weak basis for a claim. This runs the three recipes in
medseg.explain.lrp on the same images and reports the same measures as the main audit.

Run:  python -m medseg.explain.xai_lrp_ablation --run outputs/pannuke_resnet50 --n 60
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
from medseg.explain.xai_audit import FRACTIONS, to_tensor


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data-root", default="data/pannuke")
    ap.add_argument("--fold", type=int, default=3)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    model, cfg, _ = load_run(a.run, device=torch.device("cpu"))
    images, sem, _ = pannuke.load_fold(a.data_root, a.fold)
    out_dir = Path(a.run) / "xai_audit"
    rows = pd.read_csv(out_dir / "rows.csv").sample(n=a.n, random_state=a.seed)
    csv = out_dir / "lrp_ablation.csv"
    done = set(pd.read_csv(csv)["index"]) if csv.exists() else set()
    attr = MultiAttribution(model, default_layers(model))
    lrps = {r: LRPAttribution(model, r) for r in LRPAttribution.RULES}
    t0 = time.time()
    for n, (_, r) in enumerate(rows[~rows["index"].isin(done)].iterrows()):
        i, c = int(r["index"]), int(r["target_class"])
        rng = np.random.RandomState(a.seed + i)
        x = to_tensor(images[i])
        with torch.no_grad():
            pred = model(x).argmax(1)[0].numpy()
        region = pred == c
        maps, _ = attr(x, c, region=torch.from_numpy(region))
        if not maps:
            continue
        small_logit = xm.downsample(maps["logit"])
        orders = {"random": rng.permutation(region.size), "region_random": xm._region_random_order(region, rng)}
        row = {"index": i}
        for rule, lrp in lrps.items():
            m, total, score = lrp(x, c, torch.from_numpy(region))
            if m is None:
                continue
            orders[rule] = xm._order_desc(m, rng)
            row[f"retained|{rule}"] = total / score if score else float("nan")
            row[f"agree_logit|{rule}"] = xm.spearman(xm.downsample(m), small_logit)
            row[f"loc_enrich|{rule}"] = xm.enrichment(m, region)
        curves = xm.deletion_curves(model, x, region, c, orders, FRACTIONS)
        auc = {k: xm.curve_auc(v, FRACTIONS) for k, v in curves.items()}
        for rule in LRPAttribution.RULES:
            if rule in auc:
                row[f"gain_region|{rule}"] = auc["region_random"] - auc[rule]
        pd.DataFrame([row]).to_csv(csv, mode="a", header=not csv.exists(), index=False)
        if (n + 1) % 10 == 0:
            print(f"[lrp-ablation] {n + 1} {(time.time() - t0) / (n + 1):.1f}s/img", flush=True)


if __name__ == "__main__":
    main()
