"""Where in the network should a class-activation map be taken?

The audit found two opposite problems. A map taken at the last decoder block is almost a
copy of the prediction (high agreement with the model's own score map, little faithfulness
beyond it). A map taken at the deepest encoder stage points at pixels whose removal
hurts the model LESS than removing random pixels. This script sweeps the layer in
between and asks whether any depth gives a map that is both informative and not a copy.

For every layer it computes Seg-Grad-CAM and a HiResCAM-style map, and measures:
    agree_logit   rank correlation with the model's own score map (1 = a copy)
    enrichment    map weight inside the explained region versus chance
    gain          deletion gain over random pixels inside the region (> 0 = informative)

Layers: encoder stages layer1 to layer4 (output strides 4, 8, 16, 32) and the decoder.

Run:  python -m medseg.explain.xai_depth_sweep --run outputs/pannuke_resnet50 --n 150
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
from medseg.explain.attribution import MultiAttribution
from medseg.explain.xai_audit import FRACTIONS, pick_target_class, to_tensor
from medseg.models import default_target_layer


def sweep_layers(model: torch.nn.Module):
    enc = model.encoder
    return {"enc1": enc.layer1, "enc2": enc.layer2, "enc3": enc.layer3, "enc4": enc.layer4,
            "dec": default_target_layer(model)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data-root", default="data/pannuke")
    ap.add_argument("--fold", type=int, default=3)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--rows", default="rows.csv", help="draw the images from this audit's results")
    ap.add_argument("--out-name", default="depth_sweep.csv")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    model, cfg, _ = load_run(a.run, device=torch.device("cpu"))
    images, sem, _ = pannuke.load_fold(a.data_root, a.fold)
    out_dir = Path(a.run) / "xai_audit"
    rows = pd.read_csv(out_dir / a.rows)
    pick = rows.sample(n=min(a.n, len(rows)), random_state=a.seed)
    csv = out_dir / a.out_name
    done = set(pd.read_csv(csv)["index"]) if csv.exists() else set()
    layers = sweep_layers(model)
    attr = MultiAttribution(model, layers)
    t0 = time.time()
    todo = pick[~pick["index"].isin(done)]
    for n, (_, r) in enumerate(todo.iterrows()):
        i, c = int(r["index"]), int(r["target_class"])
        rng = np.random.RandomState(a.seed + i)
        x = to_tensor(images[i])
        with torch.no_grad():
            pred = model(x).argmax(1)[0].numpy()
        region = pred == c
        maps, _ = attr(x, c, region=torch.from_numpy(region))
        if not maps:
            continue
        names = [k for k in maps if k != "logit" and k != "gxi"]
        orders = {k: xm._order_desc(maps[k], rng) for k in names}
        orders["random"] = rng.permutation(region.size)
        orders["region_random"] = xm._region_random_order(region, rng)
        curves = xm.deletion_curves(model, x, region, c, orders, FRACTIONS)
        auc = {k: xm.curve_auc(v, FRACTIONS) for k, v in curves.items()}
        small_logit = xm.downsample(maps["logit"])
        row = {"index": i, "tissue": r["tissue"], "brightness": r["brightness"]}
        for k in names:
            row[f"agree_logit|{k}"] = xm.spearman(xm.downsample(maps[k]), small_logit)
            row[f"loc_enrich|{k}"] = xm.enrichment(maps[k], region)
            row[f"gain_region|{k}"] = auc["region_random"] - auc[k]
        pd.DataFrame([row]).to_csv(csv, mode="a", header=not csv.exists(), index=False)
        if (n + 1) % 10 == 0:
            el = time.time() - t0
            print(f"[depth] {n + 1}/{len(todo)} {el / (n + 1):.1f}s/img eta {(len(todo) - n - 1) * el / (n + 1) / 60:.0f} min", flush=True)
    attr.remove()


if __name__ == "__main__":
    main()


def summarise(run: str, csv_name: str = "depth_sweep.csv") -> pd.DataFrame:
    """Mean and bootstrap 95% interval of each measure at each depth; writes a table and a figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(run) / "xai_audit"
    d = pd.read_csv(out / csv_name)
    depths = ["enc1", "enc2", "enc3", "enc4", "dec"]
    rng = np.random.RandomState(0)
    rows = []
    for tag in ("cam", "hires"):
        for layer in depths:
            r = {"map": tag, "layer": layer, "n": int(len(d))}
            for key, col in (("gain", f"gain_region|{tag}@{layer}"), ("agree_logit", f"agree_logit|{tag}@{layer}"),
                             ("enrich", f"loc_enrich|{tag}@{layer}")):
                v = d[col].dropna().values
                bs = [v[rng.randint(0, len(v), len(v))].mean() for _ in range(1000)]
                r[key], r[key + "_lo"], r[key + "_hi"] = float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
            rows.append(r)
    tbl = pd.DataFrame(rows)
    tbl.to_csv(out / "depth_sweep_summary.csv", index=False)

    colors = {"cam": "#3b3a8c", "hires": "#b04a73"}
    names = {"cam": "Seg-Grad-CAM", "hires": "HiResCAM-style"}
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), sharex=True)
    for ax, key, title, ylab in ((axes[0], "gain", "Faithfulness: deletion gain", "gain over random pixels in region (0 = no better)"),
                                 (axes[1], "agree_logit", "Circularity: agreement with the model's own score map", "rank correlation (1 = a copy)")):
        for tag in ("cam", "hires"):
            t = tbl[tbl["map"] == tag]
            ax.errorbar(range(len(depths)), t[key], yerr=[t[key] - t[key + "_lo"], t[key + "_hi"] - t[key]], marker="o",
                        color=colors[tag], capsize=2, label=names[tag])
        ax.set_xticks(range(len(depths)))
        ax.set_xticklabels(["enc1\n(stride 4)", "enc2\n(8)", "enc3\n(16)", "enc4\n(32)", "decoder\n(output side)"], fontsize=8)
        ax.set_title(title, fontsize=10); ax.set_ylabel(ylab, fontsize=8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].axhline(0, color="#777", lw=0.8)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(out / "fig5_depth_sweep.png", dpi=160); plt.close(fig)
    return tbl
