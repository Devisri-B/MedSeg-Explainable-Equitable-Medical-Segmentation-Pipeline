"""Do the explanation-audit findings repeat on another model, fold and dataset?

One audit on one model could be a quirk of that model. This script runs the same
analysis on several "studies" and puts the headline numbers side by side:

    A  PanNuke fold 3, U-Net++ / ResNet-50   (the main model)
    B  PanNuke fold 1, FPN / ResNet-18       (another decoder, encoder, training folds)
    C  MoNuSeg tiles,  model A as a nucleus-vs-background detector (new dataset, new labs)
    D  PanNuke fold 2, U-Net / ResNet-18     (a third decoder and a third test fold)

For each study it reports the five checks that carry the paper's claims:
    1  circularity       decoder Seg-Grad-CAM vs the model's own score map (rank correlation)
    2  faithfulness      deletion gain over region-random pixels, per method, with 95% CI,
                         and Seg-Grad-CAM minus the control (paired bootstrap)
    3  LRP               same measures for LRP, and how much relevance it keeps
    4  stain groups      dark vs bright: stability of decoder Seg-Grad-CAM, and Dice
    5  failure detection AUROC of uncertainty, of explanation signals, and the change when
                         explanation signals are added to uncertainty

Run (after the audits):
    python -m medseg.explain.xai_replication --out outputs/replication
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats

from medseg.explain.xai_analysis import (LABELS, METHODS, boot_ci, dark_vs_bright, failure_detection,
                                         group_tests, load_rows, method_table)

STUDIES = {
    "A": dict(name="A: PanNuke fold 3, U-Net++ / ResNet-50", run="outputs/pannuke_resnet50",
              base="rows.csv", addon="rows_lrp.csv", extra=["rows_dark.csv"]),
    "B": dict(name="B: PanNuke fold 1, FPN / ResNet-18", run="outputs/pannuke_fpn_r18_fold1",
              base="rows.csv", addon=None, extra=["rows_dark.csv"]),
    "C": dict(name="C: MoNuSeg tiles, model A (foreground)", run="outputs/pannuke_resnet50",
              base="rows_monuseg.csv", addon=None, extra=[]),
    "D": dict(name="D: PanNuke fold 2, U-Net / ResNet-18", run="outputs/pannuke_unet_r18_fold2",
              base="rows.csv", addon=None, extra=[]),
}


def load_study(spec: Dict) -> pd.DataFrame:
    d = Path(spec["run"]) / "xai_audit"
    if not (d / spec["base"]).exists():
        return pd.DataFrame()
    df = load_rows(d / spec["base"])
    if spec["addon"] and (d / spec["addon"]).exists():
        add = pd.read_csv(d / spec["addon"])
        df = df.merge(add, on="index", how="left", suffixes=("", "_new"))
        for c in [c for c in df.columns if c.endswith("_new")]:
            df[c[:-4]] = df[c]
            df = df.drop(columns=c)
    for extra in spec["extra"]:
        if (d / extra).exists():
            df = pd.concat([df, load_rows(d / extra)], ignore_index=True)
    return df.drop_duplicates("index").reset_index(drop=True)


def paired_diff(a: pd.Series, b: pd.Series, n: int = 2000, seed: int = 0) -> Dict[str, float]:
    ok = a.notna() & b.notna()
    d = (a[ok] - b[ok]).values
    if len(d) < 5:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
    rng = np.random.RandomState(seed)
    bs = [d[rng.randint(0, len(d), len(d))].mean() for _ in range(n)]
    return {"mean": float(d.mean()), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5))}


def summarise(df: pd.DataFrame) -> Dict[str, object]:
    out: Dict[str, object] = {"n": int(len(df)), "n_tissues": int(df["tissue"].nunique()),
                              "brightness": df["brightness"].value_counts().to_dict(),
                              "mean_dice": boot_ci(df["dice_c"].dropna())}
    mt = method_table(df)
    out["methods"] = mt.round(4).to_dict(orient="records")
    out["circularity_cam_dec"] = boot_ci(df["agree_logit|cam@dec"].dropna())
    out["cam_dec_minus_control_gain"] = paired_diff(df["del_gain_region|cam@dec"], df["del_gain_region|logit"])
    out["stability_cam_dec_minus_control"] = paired_diff(df["stab_rho|cam@dec"], df["stab_rho|logit"])
    if "lrp_retained" in df:
        out["lrp_retained_relevance"] = boot_ci(df["lrp_retained"].dropna())
    for col in ("stab_rho|cam@dec", "dice_c", "agree_logit|hires@dec"):
        if col in df:
            out[f"dark_vs_bright|{col}"] = dark_vs_bright(df, col)
    if "agree_logit|cam@dec" in df:
        out["dark_effect_stability_cam_dec_adjusted_for_copying"] = dark_effect_adjusted(df, "stab_rho|cam@dec", "agree_logit|cam@dec")
    out["tissue_test_stability"] = group_tests(df, "tissue", "stab_rho|cam@dec")
    out["failure_detection"] = failure_detection(df)
    return out


def dark_effect_adjusted(df: pd.DataFrame, col: str, control: str, n: int = 1000, seed: int = 0) -> Dict[str, float]:
    """Dark-vs-bright difference in `col`, before and after adjusting for `control`.

    Least squares on [1, is_dark, control] using only dark and bright images. If the dark
    coefficient shrinks towards 0 once the control is included, the group difference is
    explained by the control (here: by how closely the map copies the prediction).
    """
    d = df[df["brightness"].isin(["dark", "bright"]) & df[col].notna() & df[control].notna()]
    if (d["brightness"] == "dark").sum() < 8:
        return {}
    y = d[col].values
    dark = (d["brightness"] == "dark").astype(float).values
    ctl = d[control].values

    def coefs(idx):
        X0 = np.c_[np.ones(len(idx)), dark[idx]]
        X1 = np.c_[X0, ctl[idx]]
        return (np.linalg.lstsq(X0, y[idx], rcond=None)[0][1], np.linalg.lstsq(X1, y[idx], rcond=None)[0][1])

    full = coefs(np.arange(len(y)))
    rng = np.random.RandomState(seed)
    bs = np.array([coefs(rng.randint(0, len(y), len(y))) for _ in range(n)])
    return {"n_dark": int(dark.sum()), "n_bright": int(len(y) - dark.sum()),
            "dark_effect_raw": float(full[0]), "raw_lo": float(np.percentile(bs[:, 0], 2.5)), "raw_hi": float(np.percentile(bs[:, 0], 97.5)),
            "dark_effect_adjusted": float(full[1]), "adj_lo": float(np.percentile(bs[:, 1], 2.5)), "adj_hi": float(np.percentile(bs[:, 1], 97.5))}


def pooled_dark_effect(frames: Dict[str, pd.DataFrame], col: str) -> Dict[str, float]:
    """Dark vs bright across studies: values are turned into within-study percentile ranks first."""
    dark, bright = [], []
    for df in frames.values():
        if col not in df or df[col].notna().sum() < 20:
            continue
        r = df[col].rank(pct=True)
        dark += list(r[(df["brightness"] == "dark") & df[col].notna()])
        bright += list(r[(df["brightness"] == "bright") & df[col].notna()])
    if len(dark) < 5 or len(bright) < 5:
        return {}
    u = stats.mannwhitneyu(dark, bright, alternative="two-sided")
    return {"n_dark": len(dark), "n_bright": len(bright), "prob_dark_higher": float(u.statistic / (len(dark) * len(bright))),
            "p": float(u.pvalue)}


def forest_figure(summ: Dict[str, Dict], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    studies = list(summ)
    methods = [m for m in METHODS if any(any(r["method"] == m for r in summ[s]["methods"]) for s in studies)]
    cols = {"A": "#3b3a8c", "B": "#b04a73", "C": "#2f8f83", "D": "#c28a2c"}
    fig, ax = plt.subplots(figsize=(8.5, 0.24 * len(methods) * len(studies) + 1.8))
    y = 0
    ticks, labels = [], []
    for m in methods:
        for s in studies:
            r = next((r for r in summ[s]["methods"] if r["method"] == m), None)
            if r is None:
                continue
            ax.errorbar(r["gain_vs_region_random"], y, xerr=[[r["gain_vs_region_random"] - r["gain_vs_region_random_lo"]],
                                                              [r["gain_vs_region_random_hi"] - r["gain_vs_region_random"]]],
                        fmt="o", color=cols.get(s, "k"), capsize=2, label=s if m == methods[0] else None)
            y += 1
        ticks.append(y - len(studies) / 2 - 0.5); labels.append(LABELS[m])
        y += 0.6
    ax.axvline(0, color="#777", lw=0.8)
    ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=9); ax.invert_yaxis()
    ax.set_xlabel("Deletion gain over random pixels in the region (95% CI); 0 = no better than random", fontsize=9)
    ax.legend(title="Study", frameon=False, fontsize=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/replication")
    ap.add_argument("--studies", default="A,B,C,D")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    frames: Dict[str, pd.DataFrame] = {}
    for k in a.studies.split(","):
        df = load_study(STUDIES[k])
        if len(df):
            frames[k] = df
            print(f"[replication] study {k}: {len(df)} images")
    summ = {k: summarise(df) for k, df in frames.items()}
    for k in summ:
        summ[k]["name"] = STUDIES[k]["name"]
    pooled = {col: pooled_dark_effect(frames, col) for col in ("stab_rho|cam@dec", "dice_c", "agree_logit|hires@dec")}
    (out / "replication.json").write_text(json.dumps({"studies": summ, "pooled_dark_vs_bright": pooled}, indent=2, default=float))
    rows = []
    for k, s in summ.items():
        for r in s["methods"]:
            rows.append({"study": k, **r})
    pd.DataFrame(rows).to_csv(out / "replication_methods.csv", index=False)
    forest_figure(summ, out / "fig4_replication.png")
    print(json.dumps({k: {"n": s["n"], "circ": s["circularity_cam_dec"]["mean"], "gain_diff": s["cam_dec_minus_control_gain"],
                          "fd": {n: round(v["auroc"], 3) for n, v in s["failure_detection"].get("sets", {}).items()}}
                      for k, s in summ.items()}, indent=1, default=float))


if __name__ == "__main__":
    main()
