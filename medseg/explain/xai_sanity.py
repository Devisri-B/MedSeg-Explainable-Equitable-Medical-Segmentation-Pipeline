"""Parameter-randomization sanity check (Adebayo et al., 2018), for segmentation.

A real explanation must depend on what the model learned. So if we destroy the learned
weights, the explanation should change a lot. If it barely changes, the method is
drawing something about the *image* (edges, colours), not about the model.

We re-initialise the model in stages, from the output end towards the input:
    head        only the final segmentation head
    decoder     the decoder and the head
    all         every layer (a completely untrained network)
and measure, for each method, the rank correlation between the explanation of the trained
model and the explanation of the randomised model. Values near 1 after randomisation are a
red flag; values near 0 are what a faithful method should show.

The pixels being explained are fixed to the trained model's predicted region, so the
comparison isolates the explanation rather than a changed prediction.

That choice has a side effect. A map that is zero outside the explained region (HiResCAM at
the last decoder block is one) shares its outline with the trained map even when the weights
are random, which inflates the correlation. So two extra numbers are reported:
    rho_in_region   correlation computed only on pixels inside the explained region, which
                    removes the shared outline
    region_corr     correlation of the trained map with the plain region mask, which says
                    how much of a map's layout is simply the region

Run:  python -m medseg.explain.xai_sanity --run outputs/pannuke_resnet50 --n 40
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from medseg.data import pannuke
from medseg.evaluate import load_run
from medseg.explain import xai_metrics as xm
from medseg.explain.attribution import MultiAttribution, default_layers
from medseg.explain.xai_audit import pick_target_class, select_images, to_tensor
from medseg.utils import ensure_dir, set_seed

METHODS = ["cam@dec", "hires@dec", "cam@enc4", "hires@enc4", "gxi", "logit"]


def reinit(module: torch.nn.Module) -> None:
    """Replace learned weights with fresh random ones (same shapes, default init)."""
    for m in module.modules():
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()


def randomised_copy(model: torch.nn.Module, stage: str) -> torch.nn.Module:
    m = copy.deepcopy(model)
    if stage == "head":
        reinit(m.segmentation_head)
    elif stage == "decoder":
        reinit(m.segmentation_head)
        reinit(m.decoder)
    elif stage == "all":
        reinit(m)
    else:
        raise ValueError(stage)
    return m.eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data-root", default="data/pannuke")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    set_seed(args.seed)
    torch.set_num_threads(2)

    model, cfg, _ = load_run(args.run, device=torch.device("cpu"))
    images, sem, types = pannuke.load_fold(args.data_root, 3)
    rng = np.random.RandomState(args.seed + 99)
    idx = rng.choice(len(images), size=args.n * 2, replace=False)

    attr_t = MultiAttribution(model, default_layers(model))
    stages = ["head", "decoder", "all"]
    rand_models = {s: randomised_copy(model, s) for s in stages}
    attr_r = {s: MultiAttribution(rm, default_layers(rm)) for s, rm in rand_models.items()}

    rows = []
    for i in idx:
        if len(rows) >= args.n * len(stages):
            break
        x = to_tensor(images[i])
        with torch.no_grad():
            pred = model(x).argmax(1)[0].numpy()
        c = pick_target_class(pred, cfg.data.num_classes)
        if c is None:
            continue
        region = torch.from_numpy(pred == c)
        maps_t, _ = attr_t(x, c, region=region)
        for s in stages:
            maps_r, _ = attr_r[s](x, c, region=region)
            if not maps_r:
                continue
            row = {"index": int(i), "stage": s}
            inside = (pred == c)
            for m in METHODS:
                row[m] = xm.spearman(xm.downsample(maps_t[m]), xm.downsample(maps_r[m]))
                row[f"in_region|{m}"] = xm.spearman(maps_t[m][inside], maps_r[m][inside])
                row[f"region_corr|{m}"] = xm.spearman(xm.downsample(maps_t[m]), xm.downsample(inside.astype(float)))
            rows.append(row)
        if len({r["index"] for r in rows}) >= args.n:
            break

    df = pd.DataFrame(rows)
    out = ensure_dir(Path(args.run) / "xai_audit" / "analysis")
    df.to_csv(out / "sanity_rows.csv", index=False)
    summary = df.groupby("stage")[METHODS].agg(["mean", "std"]).round(3)
    res = {s: {m: float(df[df.stage == s][m].mean()) for m in METHODS} for s in stages}
    res_in = {s: {m: float(df[df.stage == s][f"in_region|{m}"].mean()) for m in METHODS} for s in stages}
    reg = {m: float(df[df.stage == "head"][f"region_corr|{m}"].mean()) for m in METHODS}
    (out / "sanity.json").write_text(json.dumps({"n_images": int(df["index"].nunique()), "mean_rho": res,
                                                 "mean_rho_in_region": res_in, "trained_map_vs_region_mask": reg}, indent=2))
    print(summary.to_string())
    print(json.dumps({'mean_rho': res, 'mean_rho_in_region': res_in, 'trained_map_vs_region_mask': reg}, indent=2))


if __name__ == "__main__":
    main()
