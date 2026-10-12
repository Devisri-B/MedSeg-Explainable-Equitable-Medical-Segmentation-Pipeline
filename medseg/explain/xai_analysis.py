"""Turn the explanation-audit table into findings: method comparison, subgroup gaps, failure signals.

Input : <run>/xai_audit/rows.csv  (one row per audited image, from xai_audit.py)
Output: <run>/xai_audit/analysis/*.csv, *.json, *.png and summary.md

Three questions, each answered with a number and a confidence interval:

Q1  Do the explanation methods add information beyond the model's own output map?
    (circularity: rank correlation with the `logit` control; faithfulness gain over it)
Q2  Is explanation quality equal across tissue types and stain brightness, or does it
    fall where segmentation is already worst?
Q3  Can label-free explanation signals warn that a segmentation is wrong, beyond what
    uncertainty alone already tells us?
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from medseg.fairness.audit import disparity
from medseg.utils import ensure_dir

METHODS = ["cam@dec", "hires@dec", "cam@enc4", "hires@enc4", "gxi", "lrp", "logit"]   # "logit" must stay last
LABELS = {
    "cam@dec": "Seg-Grad-CAM (decoder)", "hires@dec": "HiResCAM-style (decoder)",
    "cam@enc4": "Seg-Grad-CAM (encoder)", "hires@enc4": "HiResCAM-style (encoder)",
    "gxi": "Gradient x Input", "lrp": "LRP (epsilon, z+, flat)", "logit": "Control: model's own score map",
}
FAIL_THRESHOLD = 0.5   # an explained class counts as a failure if its Dice is below this


# ----------------------------------------------------------------------- utilities
def load_rows(csv: Path) -> pd.DataFrame:
    return pd.read_csv(csv).reset_index(drop=True)


def boot_ci(x: Sequence[float], n: int = 2000, seed: int = 0) -> Dict[str, float]:
    """Mean with a 95% percentile-bootstrap interval (NaNs dropped)."""
    a = np.asarray(x, dtype=float)
    a = a[~np.isnan(a)]
    if len(a) == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = np.random.RandomState(seed)
    means = a[rng.randint(0, len(a), (n, len(a)))].mean(axis=1)
    return {"mean": float(a.mean()), "lo": float(np.percentile(means, 2.5)),
            "hi": float(np.percentile(means, 97.5)), "n": int(len(a))}


def wilcoxon_vs(a: pd.Series, b: pd.Series) -> Dict[str, float]:
    """Paired signed-rank test of a - b (rows with NaN dropped)."""
    d = (a - b).dropna()
    if len(d) < 10 or (d == 0).all():
        return {"median_diff": float(d.median()) if len(d) else float("nan"), "p": float("nan"), "n": int(len(d))}
    return {"median_diff": float(d.median()), "p": float(stats.wilcoxon(d).pvalue), "n": int(len(d))}


# --------------------------------------------------------------- Q1: method comparison
def method_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per method: localization, concentration, stability, faithfulness, circularity."""
    rows = []
    for m in METHODS:
        if f"loc_enrich|{m}" not in df:      # e.g. LRP was not run for this audit
            continue
        r = {"method": m, "label": LABELS[m]}
        for key, col in [
            ("enrichment", f"loc_enrich|{m}"), ("gt_mass", f"loc_gt_mass|{m}"),
            ("concentration", f"conc|{m}"), ("stability_rho", f"stab_rho|{m}"),
            ("deletion_auc", f"del_auc|{m}"),
            ("gain_vs_random", f"del_gain_global|{m}"), ("gain_vs_region_random", f"del_gain_region|{m}"),
            ("agree_with_logit", f"agree_logit|{m}"),
        ]:
            if col in df:
                ci = boot_ci(df[col])
                r[key] = ci["mean"]; r[key + "_lo"] = ci["lo"]; r[key + "_hi"] = ci["hi"]
        if m != "logit" and f"del_gain_region|{m}" in df:
            w = wilcoxon_vs(df[f"del_gain_region|{m}"], df["del_gain_region|logit"])
            r["vs_logit_median_diff"] = w["median_diff"]; r["vs_logit_p"] = w["p"]
        rows.append(r)
    return pd.DataFrame(rows)


# -------------------------------------------------------------- Q2: subgroup analysis
def subgroup_table(df: pd.DataFrame, by: str, metrics: Dict[str, str], min_n: int = 8) -> pd.DataFrame:
    rows = []
    for g, sub in df.groupby(by):
        if len(sub) < min_n:
            continue
        r = {by: g, "n": len(sub)}
        for name, col in metrics.items():
            ci = boot_ci(sub[col])
            r[name] = ci["mean"]; r[name + "_lo"] = ci["lo"]; r[name + "_hi"] = ci["hi"]
        rows.append(r)
    return pd.DataFrame(rows).sort_values(by).reset_index(drop=True)


def group_tests(df: pd.DataFrame, by: str, col: str) -> Dict[str, float]:
    """Kruskal-Wallis across groups: does this metric differ between groups at all?"""
    groups = [g[col].dropna().values for _, g in df.groupby(by) if g[col].notna().sum() >= 8]
    if len(groups) < 2:
        return {"H": float("nan"), "p": float("nan"), "groups": len(groups)}
    try:
        h = stats.kruskal(*groups)
    except (ValueError, AttributeError):   # e.g. every value identical
        return {"H": float("nan"), "p": float("nan"), "groups": len(groups)}
    return {"H": float(h.statistic), "p": float(h.pvalue), "groups": len(groups)}


def gap_report(tbl: pd.DataFrame, by: str, name: str) -> Dict[str, object]:
    """Best-minus-worst gap and worst/best ratio (same definitions as the fairness audit)."""
    d = disparity({str(r[by]): float(r[name]) for _, r in tbl.iterrows() if r[name] == r[name]})
    return {k: d[k] for k in ("n_groups", "best", "worst", "best_group", "worst_group", "gap", "ratio", "cv") if k in d}


def dark_vs_bright(df: pd.DataFrame, col: str) -> Dict[str, float]:
    """Effect size for dark vs bright images: P(dark value > bright value) via Mann-Whitney."""
    a = df.loc[df.brightness == "dark", col].dropna()
    b = df.loc[df.brightness == "bright", col].dropna()
    if len(a) < 5 or len(b) < 5:
        return {"n_dark": len(a), "n_bright": len(b)}
    u = stats.mannwhitneyu(a, b)
    return {"n_dark": len(a), "n_bright": len(b), "median_dark": float(a.median()),
            "median_bright": float(b.median()), "p": float(u.pvalue),
            "prob_dark_greater": float(u.statistic / (len(a) * len(b)))}


# ----------------------------------------------------- Q3: label-free failure detection
UNCERTAINTY = ["entropy_region", "entropy_img", "conf_region", "region_frac"]
PRED_STABILITY = ["stain_pred_dice"]
EXPLANATION = (
    [f"stab_rho|{m}" for m in METHODS] + ["agree|cam-hires", "agree|cam-gxi", "agree|hires-gxi"]
    + [f"agree_logit|{m}" for m in METHODS[:-1]] + [f"conc|{m}" for m in METHODS]
    + [f"loc_enrich|{m}" for m in METHODS]
)
EXPLANATION_DELETION = [f"del_gain_region|{m}" for m in METHODS] + [f"del_gain_global|{m}" for m in METHODS]
FEATURE_SETS = {
    "uncertainty": UNCERTAINTY,
    "uncertainty + stain-shift prediction stability": UNCERTAINTY + PRED_STABILITY,
    "explanation signals only": EXPLANATION,
    "uncertainty + explanation": UNCERTAINTY + EXPLANATION,
    "uncertainty + stain-shift + explanation": UNCERTAINTY + PRED_STABILITY + EXPLANATION,
    "all of the above + deletion test": UNCERTAINTY + PRED_STABILITY + EXPLANATION + EXPLANATION_DELETION,
}


def _oof_scores(X: np.ndarray, y: np.ndarray, repeats: int, seed: int) -> np.ndarray:
    """Out-of-fold failure probability for every image, averaged over repeated 5-fold CV."""
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=seed)
    acc = np.zeros(len(y)); cnt = np.zeros(len(y))
    for tr, te in cv.split(X, y):
        pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(C=0.3, class_weight="balanced", max_iter=2000))
        pipe.fit(X[tr], y[tr])
        acc[te] += pipe.predict_proba(X[te])[:, 1]; cnt[te] += 1
    return acc / np.maximum(cnt, 1)


def failure_detection(df: pd.DataFrame, repeats: int = 10, seed: int = 0, n_boot: int = 1000) -> Dict[str, object]:
    d = df[df["dice_c"].notna()].reset_index(drop=True)
    y = (d["dice_c"].values < FAIL_THRESHOLD).astype(int)
    out: Dict[str, object] = {"n": int(len(d)), "n_fail": int(y.sum()), "threshold": FAIL_THRESHOLD, "sets": {}}
    if y.sum() < 10 or (len(y) - y.sum()) < 10:
        out["note"] = "too few failures or successes for a meaningful test"
        return out
    scores = {}
    for name, cols in FEATURE_SETS.items():
        cols = [c for c in cols if c in d.columns]
        scores[name] = _oof_scores(d[cols].values.astype(float), y, repeats, seed)
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(y), (n_boot, len(y)))
    base = "uncertainty"
    for name, s in scores.items():
        boots, diffs = [], []
        for b in idx:
            if y[b].min() == y[b].max():
                continue
            boots.append(roc_auc_score(y[b], s[b]))
            diffs.append(roc_auc_score(y[b], s[b]) - roc_auc_score(y[b], scores[base][b]))
        out["sets"][name] = {
            "auroc": float(roc_auc_score(y, s)),
            "auroc_lo": float(np.percentile(boots, 2.5)), "auroc_hi": float(np.percentile(boots, 97.5)),
            "gain_over_uncertainty": float(roc_auc_score(y, s) - roc_auc_score(y, scores[base])),
            "gain_lo": float(np.percentile(diffs, 2.5)), "gain_hi": float(np.percentile(diffs, 97.5)),
        }
    # single-feature view: which one signal alone tracks Dice best?
    singles = {}
    for c in UNCERTAINTY + PRED_STABILITY + EXPLANATION:
        if c in d and d[c].notna().sum() > 30:
            rho = stats.spearmanr(d[c], d["dice_c"], nan_policy="omit").statistic
            singles[c] = float(rho)
    out["single_spearman_with_dice"] = dict(sorted(singles.items(), key=lambda kv: -abs(kv[1])))
    return out


# ------------------------------------------------------------------------- figures
def make_figures(df, mt, tissue_tbl, bright_tbl, fd, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1. circularity + faithfulness gain per method
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    m2 = mt[mt.method != "logit"]
    ax[0].barh(m2.label, m2.agree_with_logit, xerr=[m2.agree_with_logit - m2.agree_with_logit_lo,
                                                       m2.agree_with_logit_hi - m2.agree_with_logit], color="#4c72b0")
    ax[0].set_xlim(0, 1.05); ax[0].invert_yaxis()
    ax[0].set_xlabel("rank correlation with the model's own score map")
    ax[0].set_title("Does the method re-draw the prediction?")
    ax[1].barh(mt.label, mt.gain_vs_region_random,
               xerr=[mt.gain_vs_region_random - mt.gain_vs_region_random_lo,
                     mt.gain_vs_region_random_hi - mt.gain_vs_region_random], color="#dd8452")
    ax[1].axvline(0, color="k", lw=0.8); ax[1].invert_yaxis(); ax[1].set_yticklabels([])
    ax[1].set_xlabel("deletion gain over region-random order (higher = more informative)")
    ax[1].set_title("Does the ranking inside the region matter?")
    fig.tight_layout(); fig.savefig(out / "fig1_methods.png", dpi=140); plt.close(fig)

    # 2. subgroup: segmentation quality vs explanation quality
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for a, tbl, by in ((ax[0], tissue_tbl, "tissue"), (ax[1], bright_tbl, "brightness")):
        a.errorbar(tbl["dice_c"], tbl["stability"], xerr=[tbl["dice_c"] - tbl["dice_c_lo"], tbl["dice_c_hi"] - tbl["dice_c"]],
                   yerr=[tbl["stability"] - tbl["stability_lo"], tbl["stability_hi"] - tbl["stability"]],
                   fmt="o", color="#4c72b0", alpha=0.8)
        for _, r in tbl.iterrows():
            a.annotate(str(r[by]), (r["dice_c"], r["stability"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
        a.set_xlabel("Dice of the explained class"); a.set_ylabel("explanation stability under stain shift (rho)")
        a.set_title(f"By {by}")
    fig.tight_layout(); fig.savefig(out / "fig2_subgroups.png", dpi=140); plt.close(fig)

    # 3. failure detection
    if fd.get("sets"):
        names = list(fd["sets"]); v = [fd["sets"][n]["auroc"] for n in names]
        lo = [fd["sets"][n]["auroc"] - fd["sets"][n]["auroc_lo"] for n in names]
        hi = [fd["sets"][n]["auroc_hi"] - fd["sets"][n]["auroc"] for n in names]
        fig, ax = plt.subplots(figsize=(8, 3.8))
        ax.barh(names, v, xerr=[lo, hi], color="#55a868"); ax.invert_yaxis()
        ax.axvline(0.5, color="k", lw=0.8, ls="--"); ax.set_xlim(0.4, 1.0)
        ax.set_xlabel(f"AUROC for flagging Dice < {FAIL_THRESHOLD} without labels")
        fig.tight_layout(); fig.savefig(out / "fig3_failure_detection.png", dpi=140); plt.close(fig)


# ---------------------------------------------------------------------------- main
def run(args) -> Path:
    run_dir = Path(args.run)
    df = load_rows(run_dir / "xai_audit" / args.rows)
    out = ensure_dir(run_dir / "xai_audit" / "analysis")
    # primary decoder-layer HiResCAM stability + faithfulness are reported per subgroup
    sub_metrics = {
        "dice_c": "dice_c", "stability": "stab_rho|hires@enc4", "stability_cam_dec": "stab_rho|cam@dec",
        "gain_region_hires_enc4": "del_gain_region|hires@enc4", "gain_region_gxi": "del_gain_region|gxi",
        "agree_logit_hires_dec": "agree_logit|hires@dec", "entropy": "entropy_region",
    }
    mt = method_table(df); mt.to_csv(out / "methods.csv", index=False)
    tissue_tbl = subgroup_table(df, "tissue", sub_metrics); tissue_tbl.to_csv(out / "by_tissue.csv", index=False)
    bright_tbl = subgroup_table(df, "brightness", sub_metrics); bright_tbl.to_csv(out / "by_brightness.csv", index=False)
    fd = failure_detection(df, repeats=args.repeats)
    summary = {
        "n_images": int(len(df)), "n_tissues": int(df.tissue.nunique()),
        "brightness_counts": df.brightness.value_counts().to_dict(),
        "dice_gap_by_tissue": gap_report(tissue_tbl, "tissue", "dice_c"),
        "stability_gap_by_tissue": gap_report(tissue_tbl, "tissue", "stability"),
        "tests": {f"{by}|{name}": group_tests(df, by, col)
                  for by in ("tissue", "brightness") for name, col in sub_metrics.items()},
        "dark_vs_bright": {name: dark_vs_bright(df, col) for name, col in sub_metrics.items()},
        "spearman_group_dice_vs_group_stability_tissue": float(
            stats.spearmanr(tissue_tbl["dice_c"], tissue_tbl["stability"]).statistic),
        "failure_detection": fd,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    make_figures(df, mt, tissue_tbl, bright_tbl, fd, out)
    print(mt.round(3).to_string(index=False))
    print(json.dumps({k: summary[k] for k in ("n_images", "dice_gap_by_tissue", "stability_gap_by_tissue",
                                              "spearman_group_dice_vs_group_stability_tissue")}, indent=2, default=float))
    print(json.dumps(fd.get("sets", fd), indent=2))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyse the explanation audit.")
    ap.add_argument("--run", required=True)
    ap.add_argument("--rows", default="rows.csv")
    ap.add_argument("--repeats", type=int, default=10)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
