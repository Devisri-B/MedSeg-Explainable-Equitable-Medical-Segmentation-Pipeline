"""Figures for the explanation-audit paper.

    heatmaps   real attribution maps for a few audited PanNuke images (model A), one row per image:
               H&E, ground truth, prediction, the control, and every attribution map, with the
               per-image audit numbers printed under each map.
    summary    one-glance overview of the study (data, models, maps, tests, findings).
    depth      the depth sweep: deletion gain against agreement with the score map, per layer.

Run:
    python -m medseg.viz.paper_figures heatmaps --run outputs/pannuke_resnet50 --indices 2199,841,2165 --out paper/fig_heatmaps.png
    python -m medseg.viz.paper_figures candidates --run outputs/pannuke_resnet50 --out paper/candidates.png
    python -m medseg.viz.paper_figures summary --out paper/summary_image.png
    python -m medseg.viz.paper_figures depth --out paper/fig_depth.png
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch

MAPS = [("logit", "Score map\n(control)"), ("cam@dec", "Seg-Grad-CAM\ndecoder"), ("hires@dec", "HiResCAM-style\ndecoder"),
        ("cam@enc4", "Seg-Grad-CAM\nencoder"), ("gxi", "Gradient\n× Input"), ("lrp", "LRP\n(ε, z+, flat)")]
CLASS_NAMES = ["background", "neoplastic", "inflammatory", "connective", "dead", "epithelial"]


def _blend(img, rgb, alpha):
    a = alpha[..., None]
    return np.clip((1 - a) * img + a * rgb, 0, 255).astype(np.uint8)


def compute_maps(model, img: np.ndarray):
    """Prediction, explained class and every attribution map for one image."""
    from medseg.explain.attribution import MultiAttribution, default_layers
    from medseg.explain.lrp import LRPAttribution
    from medseg.explain.xai_audit import pick_target_class, to_tensor

    x = to_tensor(img)
    with torch.no_grad():
        pred = model(x).argmax(1)[0].numpy()
    c = pick_target_class(pred, 6)
    region = torch.from_numpy(pred == c)
    attr = MultiAttribution(model, default_layers(model))
    maps, _ = attr(x, c, region=region)
    attr.remove()
    maps["lrp"] = LRPAttribution(model)(x, c, region)[0]
    return pred, c, maps


def _cases(df: pd.DataFrame, which: str, k: int, seed: int = 0) -> pd.DataFrame:
    d = df[df["del_gain_region|lrp"].notna() & df["region_frac"].between(0.04, 0.35)]
    if which == "bright":
        d = d[(d["brightness"] == "bright") & (d["dice_c"] > 0.85)]
    elif which == "dark":
        d = d[(d["brightness"] == "dark") & (d["dice_c"] > 0.8)]
    else:
        d = d[d["dice_c"] < 0.4]
    return d.sample(n=min(k, len(d)), random_state=seed)


def _load(run: str):
    from medseg.data import pannuke
    from medseg.evaluate import load_run
    from medseg.explain.xai_replication import STUDIES, load_study

    df = load_study(STUDIES["A"])
    model, _, _ = load_run(run, device=torch.device("cpu"))
    images, sem, _ = pannuke.load_fold("data/pannuke", 3)
    return df, model, images, sem


def draw(rows: List[pd.Series], images, sem, model, out: Path, labels: List[str]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from medseg.utils import CLASS_COLORS

    inferno = matplotlib.colormaps["inferno"]
    ncol = 3 + len(MAPS)
    fig, axes = plt.subplots(len(rows), ncol, figsize=(0.78 * ncol, 1.0 * len(rows) + 0.42))
    axes = np.atleast_2d(axes)
    for r, (row, lab) in enumerate(zip(rows, labels)):
        i = int(row["index"])
        img, gt = images[i], sem[i]
        pred, c, maps = compute_maps(model, img)
        base = img.astype(np.float32)
        gray = np.repeat(base.mean(-1, keepdims=True), 3, -1) * 0.85 + 30
        region = pred == c
        tiles = [img, _blend(base, CLASS_COLORS[gt].astype(np.float32), (gt > 0) * 0.6),
                 _blend(base, CLASS_COLORS[pred].astype(np.float32), (pred > 0) * 0.6)]
        for k, _ in MAPS:
            m = maps[k]
            m = np.clip(m / max(np.percentile(m, 99), 1e-8), 0, 1)   # display only: clip at the 99th percentile
            tiles.append(_blend(gray, (inferno(m)[..., :3] * 255).astype(np.float32), 0.25 + 0.75 * m))
        for j, t in enumerate(tiles):
            ax = axes[r, j]
            ax.imshow(t)
            if j >= 3:
                ax.contour(region, levels=[0.5], colors="#7fe3ff", linewidths=0.3)
                k = MAPS[j - 3][0]
                txt = f"Δ {row[f'del_gain_region|{k}']:+.2f}"
                if k != "logit":
                    txt += f"   r {row[f'agree_logit|{k}']:.2f}"
                ax.set_xlabel(txt, fontsize=5.3, labelpad=1.2)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.3)
            if r == 0:
                ax.set_title(["H&E", "Ground truth", "Prediction"][j] if j < 3 else MAPS[j - 3][1], fontsize=5.8, pad=2)
        axes[r, 0].set_ylabel(f"{lab}\n{row['tissue']}, Dice {row['dice_c']:.2f}\nexplained: {CLASS_NAMES[c]}",
                              fontsize=5.3, labelpad=2)
    pass
    fig.subplots_adjust(left=0.065, right=0.997, top=0.92, bottom=0.05, wspace=0.04, hspace=0.3)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=450)
    plt.close(fig)


def summary(out: Path, rep: dict, run: str = "outputs/pannuke_resnet50", index: int = 2199) -> None:
    """One-glance summary for the submission form: pipeline, a real example, and the four findings."""
    import textwrap

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    from medseg.utils import CLASS_COLORS

    ink, muted, accent, warm, line, panel = "#1f2440", "#5b6178", "#3b3a8c", "#b04a73", "#c9cbe0", "#f4f4fa"
    s = rep["studies"]

    def gain(k, m):
        return next(r for r in s[k]["methods"] if r["method"] == m)["gain_vs_region_random"]

    circ = [s[k]["circularity_cam_dec"]["mean"] for k in "ABCD"]
    lrp = [gain(k, "lrp") for k in "ABCD"]
    enc = [gain(k, "cam@enc4") for k in "ABCD"]
    pool = rep["pooled_dark_vs_bright"]["stab_rho|cam@dec"]["prob_dark_higher"]
    fd = [s[k]["failure_detection"]["sets"]["uncertainty + explanation"]["gain_over_uncertainty"] for k in "ABD"]

    fig = plt.figure(figsize=(12, 6.75))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 12); ax.set_ylim(0, 6.75); ax.axis("off")
    ax.text(0.35, 6.28, "Do attribution heatmaps explain a segmentation network, or redraw its prediction?",
            fontsize=16.5, weight="bold", color=ink)
    ax.text(0.35, 5.93, "A control-based audit of six attribution maps for nucleus segmentation in H&E histopathology",
            fontsize=11, color=muted)

    # pipeline
    steps = [("DATA", "PanNuke, 3 test folds\nMoNuSeg, 200 tiles"),
             ("MODELS", "U-Net++ / ResNet-50\nFPN, U-Net / ResNet-18"),
             ("MAPS", "6 attribution maps\n+ control: score map"),
             ("TESTS", "faithfulness, stability,\ncopying, failure detection")]
    w, gap, y0, h = 1.62, 0.2, 4.55, 1.05
    for n, (head, body) in enumerate(steps):
        x = 0.35 + n * (w + gap)
        ax.add_patch(FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0.02,rounding_size=0.07", fc=panel, ec=line, lw=0.9))
        ax.text(x + 0.12, y0 + h - 0.25, head, fontsize=8.8, weight="bold", color=accent)
        ax.text(x + 0.12, y0 + h - 0.45, body, fontsize=8.6, color=ink, va="top", linespacing=1.35)
        if n < len(steps) - 1:
            ax.annotate("", xy=(x + w + gap - 0.02, y0 + h / 2), xytext=(x + w + 0.02, y0 + h / 2),
                        arrowprops=dict(arrowstyle="-|>", color=muted, lw=1))
    ax.text(0.35, 4.3, "4 studies  ·  3,180 image audits  ·  per-tissue and per-stain groups", fontsize=9, color=muted)

    # a real example
    df, model, images, sem = _load(run)
    row = df[df["index"] == index].iloc[0]
    img = images[index]
    pred, c, maps = compute_maps(model, img)
    import matplotlib as mpl
    inferno = mpl.colormaps["inferno"]
    base = img.astype(np.float32)
    gray = np.repeat(base.mean(-1, keepdims=True), 3, -1) * 0.85 + 30
    tiles = [("H&E", img, ""), ("Prediction", _blend(base, CLASS_COLORS[pred].astype(np.float32), (pred > 0) * 0.6), "")]
    for k, name in (("logit", "Score map\n(control)"), ("cam@dec", "Seg-Grad-CAM\ndecoder"),
                    ("cam@enc4", "Seg-Grad-CAM\nencoder"), ("lrp", "LRP")):
        m = np.clip(maps[k] / max(np.percentile(maps[k], 99), 1e-8), 0, 1)
        t = _blend(gray, (inferno(m)[..., :3] * 255).astype(np.float32), 0.25 + 0.75 * m)
        cap = f"\u0394 {row[f'del_gain_region|{k}']:+.2f}" + ("" if k == "logit" else f"  r {row[f'agree_logit|{k}']:.2f}")
        tiles.append((name, t, cap))
    tw, tg, ty = 1.08, 0.1, 1.95
    for n, (name, t, cap) in enumerate(tiles):
        x = 0.35 + n * (tw + tg)
        iax = fig.add_axes([x / 12, ty / 6.75, tw / 12, tw / 6.75 * 12 / 12 * (12 / 6.75) / (12 / 6.75)])
        iax.imshow(t); iax.set_xticks([]); iax.set_yticks([])
        for sp in iax.spines.values():
            sp.set_linewidth(0.4); sp.set_color(line)
        ax.text(x + tw / 2, ty + tw * (12 / 6.75) * (6.75 / 12) + 0.08, name, ha="center", va="bottom", fontsize=7.8, color=ink)
        if cap:
            ax.text(x + tw / 2, ty - 0.08, cap, ha="center", va="top", fontsize=7.6, color=ink)
    ax.text(0.35, 3.72, f"Example (PanNuke {row['tissue']}, Dice {row['dice_c']:.2f}): the decoder map is the score map again (r = 1.00)",
            fontsize=9.2, color=ink, weight="bold")
    ax.text(0.35, 1.3, "\u0394 = deletion gain over random pixels in the region (> 0 informative);  r = correlation with the score map (1 = copy)",
            fontsize=7.8, color=muted)

    # findings
    finds = [("Decoder maps redraw the prediction",
              f"Correlation with the model's own score map is {min(circ):.2f} to {max(circ):.2f} in all four studies, "
              "and deletion faithfulness is within 0.013 of the control."),
             ("Deeper maps and LRP are not faithful",
              f"Worse than hiding random pixels in the region (encoder {max(enc):+.2f} to {min(enc):+.2f}, "
              f"LRP {max(lrp):+.2f} to {min(lrp):+.2f}). No depth is informative and not a copy."),
             ("Stability hides a stain effect",
              f"Maps are more stable in dark stains (P = {pool:.2f}) with no consistent Dice gap; "
              "it survives adjustment for copying in 3 of 4 studies."),
             ("Failure detection is model-dependent",
              f"Adding explanation signals to uncertainty: AUROC {fd[0]:+.3f} on the strongest model, "
              f"{fd[1]:+.3f} and {fd[2]:+.3f} on weaker ones.")]
    fx, fw, fh, top = 7.55, 4.1, 1.18, 5.62
    for n, (head, body) in enumerate(finds):
        y = top - (n + 1) * fh - n * 0.1
        ax.add_patch(FancyBboxPatch((fx, y), fw, fh, boxstyle="round,pad=0.02,rounding_size=0.07", fc="white", ec=line, lw=0.9))
        ax.add_patch(plt.Circle((fx + 0.32, y + fh - 0.3), 0.17, color=warm if n == 3 else accent))
        ax.text(fx + 0.32, y + fh - 0.31, str(n + 1), fontsize=10, weight="bold", color="white", ha="center", va="center")
        ax.text(fx + 0.6, y + fh - 0.31, head, fontsize=9.8, weight="bold", color=ink, va="center")
        ax.text(fx + 0.18, y + fh - 0.58, textwrap.fill(body, 62), fontsize=8.5, color=ink, va="top", linespacing=1.35)
    ax.text(0.35, 0.78, "Code, results, checkpoints and an interactive explorer:\n"
            "github.com/Devisri-B/MedSeg-Explainable-Equitable-Medical-Segmentation-Pipeline",
            fontsize=8.2, color=accent, va="center", linespacing=1.45)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def depth(out: Path, csv: str = "outputs/pannuke_resnet50/xai_audit/depth_sweep_summary.csv") -> None:
    """Depth sweep: deletion gain against agreement with the score map, one point per layer."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = pd.read_csv(csv)
    plt.rcParams.update({"font.size": 7, "font.family": "serif"})
    fig, ax = plt.subplots(figsize=(3.35, 2.45))
    ax.axhspan(0, 0.08, xmin=0, xmax=0.55, color="#e9f3ee", zorder=0)
    ax.text(0.02, 0.055, "wanted: informative\nand not a copy", fontsize=6.3, color="#2f6f55", va="center")
    cols = {"cam": "#3b3a8c", "hires": "#b04a73"}
    names = {"cam": "Seg-Grad-CAM", "hires": "HiResCAM-style"}
    for m in ("cam", "hires"):
        t = d[d["map"] == m]
        ax.errorbar(t["agree_logit"], t["gain"], yerr=[t["gain"] - t["gain_lo"], t["gain_hi"] - t["gain"]],
                    xerr=[t["agree_logit"] - t["agree_logit_lo"], t["agree_logit_hi"] - t["agree_logit"]],
                    fmt="o", ms=3.2, color=cols[m], elinewidth=0.6, capsize=1.2, label=names[m])
        for _, r in t.iterrows():
            dx, dy = (0.03, -0.018) if m == "cam" else (-0.03, 0.05)
            ax.text(r["agree_logit"] + dx, r["gain"] + dy, r["layer"], fontsize=5.8, color=cols[m], va="center",
                    ha="left" if m == "cam" else "center")
    ax.axhline(0, color="#777", lw=0.6)
    ax.set_xlim(-0.02, 1.1); ax.set_ylim(-0.42, 0.09)
    ax.set_xlabel("Agreement with the model's own score map (1 = a copy)")
    ax.set_ylabel("Deletion gain over region-random")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=6.3, loc="lower right")
    fig.tight_layout(pad=0.3)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=400)
    plt.close(fig)


def main() -> None:
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["heatmaps", "candidates", "summary", "depth"])
    ap.add_argument("--run", default="outputs/pannuke_resnet50")
    ap.add_argument("--out", required=True)
    ap.add_argument("--indices", default="", help="comma-separated image indices (rows) for 'heatmaps'")
    ap.add_argument("--labels", default="Bright stain,Dark stain,Failure (Dice < 0.5)")
    ap.add_argument("--replication", default="outputs/replication/replication.json")
    a = ap.parse_args()
    torch.set_num_threads(2)
    if a.what == "depth":
        depth(Path(a.out))
        return
    if a.what == "summary":
        summary(Path(a.out), json.loads(Path(a.replication).read_text()), a.run)
        return
    df, model, images, sem = _load(a.run)
    if a.what == "candidates":
        rows, labels = [], []
        for which in ("bright", "dark", "fail"):
            for _, r in _cases(df, which, 3).iterrows():
                rows.append(r); labels.append(f"{which} #{int(r['index'])}")
        draw(rows, images, sem, model, Path(a.out), labels)
        return
    idx = [int(v) for v in a.indices.split(",")]
    rows = [df[df["index"] == i].iloc[0] for i in idx]
    draw(rows, images, sem, model, Path(a.out), a.labels.split(","))


if __name__ == "__main__":
    main()
