# Explanation audit results

Every number in the paper comes from the files in this folder, so the analysis can be checked without retraining a model or re-running inference. The central unit is **one row per audited image**: for each image we explain the model's prediction with several heatmap methods and record how each map behaves (see [Reading a `rows*.csv` file](#reading-a-rowscsv-file)).

- Start with the [four studies](#the-four-studies) to see which files belong to which experiment.
- Use [Load a study](#load-a-study-in-python) to get a clean table in a few lines of pandas.
- The interactive [`explorer.html`](pannuke_resnet50/xai_audit/explorer.html) shows **Study A only** (659 images). Studies B, C and D are in the CSVs below.

---

## The four studies

| Study | Model and data | Folder | Images audited |
|:---:|:---|:---|:---:|
| **A** | U-Net++ / ResNet-50 on PanNuke fold 3 (the tuned main model) | `pannuke_resnet50/xai_audit/` | **659** |
| **B** | FPN / ResNet-18 on PanNuke fold 1 (3 epochs) | `pannuke_fpn_r18_fold1/` | **1,333** |
| **C** | Model A applied to MoNuSeg, an external dataset (foreground only) | `pannuke_resnet50/xai_audit/rows_monuseg.csv` | **200** |
| **D** | U-Net / ResNet-18 on PanNuke fold 2 (4 epochs) | `pannuke_unet_r18_fold2/` | **988** |
| | | **Total** | **3,180** |

Study C reuses Model A's folder because it is the same model on different data.

### Why the file row counts do not equal the image counts

A study's images can be split over several files. Combine them and drop duplicate `index` values to get the audited set:

| Study | Files combined | Rows | Result |
|:---:|:---|:---|:---:|
| A | `rows.csv` + `rows_dark.csv` (`rows_lrp.csv` adds columns, not rows) | 622 + 37 | 659 |
| B | `rows.csv` + `rows_dark.csv` | 1,313 + 43 = 1,356, of which 23 images appear in both | 1,333 |
| C | `rows_monuseg.csv` | 200 | 200 |
| D | `rows.csv` | 988 | 988 |

`rows_dark.csv` holds extra dark-stain images added so the dark group is large enough to compare with the others. The repository's own loader does the same combine and de-duplicate step (`load_study` in `medseg/explain/xai_replication.py`).

---

## What is where

```
results/
├── pannuke_resnet50/xai_audit/          Study A, plus Study C
│   ├── rows.csv                         per-image audit rows (622)
│   ├── rows_lrp.csv                     the LRP columns for those 622 images
│   ├── rows_dark.csv                    extra dark-stain images (37)
│   ├── rows_monuseg.csv                 Study C: 200 MoNuSeg tiles
│   ├── rows*_skipped.csv                images that could not be explained
│   ├── depth_sweep.csv, depth_sweep_summary.csv, fig5_depth_sweep.png
│   ├── lrp_ablation.csv
│   ├── analysis/                        tables, figures and summary.json for Study A
│   └── explorer.html                    interactive explorer (opens offline)
├── pannuke_fpn_r18_fold1/               Study B: training files + xai_audit/
├── pannuke_unet_r18_fold2/              Study D: training files + xai_audit/
└── replication/                         all four studies side by side
```

### Study A (`pannuke_resnet50/xai_audit/`)

| File | Holds |
|:---|:---|
| `rows.csv` | One row per audited image: 65 columns covering segmentation quality, uncertainty and every metric for the six gradient-free and gradient-based maps. LRP is not in this file. |
| `rows_lrp.csv` | `index` plus 11 LRP columns for the same 622 images. Join it to `rows.csv` on `index`. |
| `rows_dark.csv` | Extra dark-stain images. These already contain the LRP columns (76 columns). |
| `rows_monuseg.csv` | Study C. Same 76 columns. The five nucleus classes are merged into one foreground score because MoNuSeg has no nucleus types. |
| `rows_skipped.csv`, `rows_dark_skipped.csv` | Images that were skipped, listed with `index`, `tissue`, `mean_intensity` and `brightness`. See [Skipped images](#skipped-images). |
| `depth_sweep.csv` | 100 images. Agreement with the score map, localization and deletion gain for Grad-CAM and HiResCAM-style maps taken at each encoder stage (`enc1` to `enc4`) and at the decoder (`dec`). |
| `depth_sweep_summary.csv` | The same sweep averaged per method and layer, with 95% bootstrap intervals (`_lo`, `_hi`). Source of `fig5_depth_sweep.png`. |
| `lrp_ablation.csv` | 60 images. Compares three LRP rule sets (`eps_plus_flat`, `eps_plus`, `eps_gamma_box`) on relevance retained, agreement with the score map, localization and deletion gain. |
| `analysis/methods.csv` | One row per method (the decoder and encoder Grad-CAM and HiResCAM-style maps, Gradient × Input, and the control): enrichment, ground-truth mass, concentration, stability, deletion gain against both random baselines, agreement with the control, and a paired test against the control (`vs_logit_median_diff`, `vs_logit_p`). Most columns have a 95% interval. |
| `analysis/by_tissue.csv` | One row per tissue (19): number of images, Dice and stability, with intervals. |
| `analysis/by_brightness.csv` | The same for the three stain groups (dark, medium, bright). |
| `analysis/summary.json` | Image and tissue counts, subgroup gaps, statistical tests, the dark-versus-bright comparison, and failure-detection results (AUROC). |
| `analysis/sanity_rows.csv`, `analysis/sanity.json` | Parameter-randomization sanity check on 40 images at three stages (`head`, `decoder`, `all`), giving 120 rows. Records how much each method's map changes when the model is randomized. |
| `analysis/fig1_methods.png`, `fig2_subgroups.png`, `fig3_failure_detection.png` | Paper figures made from the tables above. |
| `explorer.html` | The interactive explorer. Same page as the live site. Study A only. |

### Studies B and D

Each folder holds the model's training record and its audit rows:

| File | Holds |
|:---|:---|
| `config.yaml` | Training configuration |
| `history.csv` | `epoch`, `train_loss`, `val_dice` (3 epochs for B, 4 for D) |
| `summary.json` | Best validation scores |
| `training_curves.png` | Plot of `history.csv` |
| `xai_audit/rows.csv` | Per-image audit rows (76 columns, LRP included) |
| `xai_audit/rows_dark.csv` | Extra dark-stain images (Study B only) |
| `xai_audit/*_skipped.csv` | Skipped images |

### Replication (`replication/`)

| File | Holds |
|:---|:---|
| `replication.json` | Per-study summary (copying, deletion gain, stability, failure detection) and the pooled dark-versus-bright test across studies |
| `replication_methods.csv` | One row per study and method (4 studies × 7 methods = 28 rows) |
| `fig4_replication.png` | Forest plot comparing the studies |

---

## Reading a `rows*.csv` file

**One row is one image.** Column names have the form `metric|map`, where the part after `|` is one of the **map** codes below.

### Maps

| Code | Map |
|:---|:---|
| `cam@dec` | Seg-Grad-CAM, final decoder block |
| `hires@dec` | HiResCAM-style, final decoder block |
| `cam@enc4` | Seg-Grad-CAM, deepest encoder stage |
| `hires@enc4` | HiResCAM-style, deepest encoder stage |
| `gxi` | Gradient × Input |
| `lrp` | Layer-wise Relevance Propagation |
| `logit` | **The control**: ReLU of the model's own score for the explained class. It uses no gradients. |

### Identification and image properties

| Column | Meaning |
|:---|:---|
| `index` | Position of the image within its dataset (the PanNuke fold, or the MoNuSeg tile list). Unique key. |
| `tissue` | Tissue type (19 in PanNuke) |
| `mean_intensity` | Mean pixel value of the raw image, 0 to 255 |
| `brightness` | Stain group from `mean_intensity`: **dark** below 110, **medium** from 110 to 160, **bright** from 160 up |

### The explained class and how well it was segmented

Each image is explained for **one class**: the most common predicted nucleus class, provided the model predicts at least 200 pixels of it.

| Column | Meaning |
|:---|:---|
| `target_class` | Class index that was explained (0 is background) |
| `region_frac`, `gt_frac` | Fraction of the image covered by the predicted region and by the ground truth of that class |
| `dice_c` | **Dice of the explained class**: overlap between the predicted region and the ground truth. This is the quality score used throughout. A **failure** is `dice_c < 0.5`. |
| `dice_img_fg` | Mean Dice over all foreground classes in the image |
| `entropy_region` | Model **uncertainty** (normalized entropy) inside the predicted region. The standard failure warning the heatmaps are compared against. |
| `entropy_img`, `conf_region` | Entropy over the whole image, and mean probability of the explained class inside the region |

### Per-map measures

| Column pattern | Meaning |
|:---|:---|
| `loc_enrich\|map` | How much more weight the map puts inside the explained region than chance would |
| `loc_gt_mass\|map` | Share of the map's weight that falls inside the ground-truth mask |
| `conc\|map` | How spread out the map is: 0 is concentrated, 1 is uniform |
| `agree_logit\|map` | **Agreement with the control**: rank (Spearman) correlation with the `logit` map. At **0.95 or more, the map is counted as a copy of the prediction.** |
| `agree\|a-b` | Agreement between two methods, for example `agree\|cam-hires` |
| `stab_rho\|map` | **Stability**: rank correlation between the map and the map recomputed after a small stain shift |
| `stain_pred_dice` | Overlap between the prediction before and after that stain shift (one value per image) |
| `del_auc\|map` | Area under the deletion curve: hide the pixels the map ranks highest and track the model's score. Lower means hiding them hurt more. |
| `del_auc\|random`, `del_auc\|region_random` | The same measure for two baselines: random pixels anywhere, and random pixels inside the explained region |
| `del_gain_global\|map` | Random baseline minus the map. Positive means informative. |
| `del_gain_region\|map` | **Deletion gain**: region-random baseline minus the map. Positive means the map ranks pixels better than chance inside the region. Negative means worse than chance. |
| `lrp_retained` | LRP only: total relevance divided by the explained score. A check on how much relevance is conserved. |

Not every column exists for every map. The control `logit` has no `agree_logit` (it would be compared with itself), and `lrp` columns live in `rows_lrp.csv` for Study A.

### Skipped images

If the model predicts **no** nucleus class covering at least 200 pixels, there is nothing to explain and the image is skipped. Skipped images are kept in separate `*_skipped.csv` files (`index`, `tissue`, `mean_intensity`, `brightness`) so they never appear as short rows that would shift the columns of the main files.

| Study | Skipped |
|:---:|:---|
| A | 18 (`rows_skipped.csv`) and 13 dark (`rows_dark_skipped.csv`) |
| B | 35 and 7 dark |
| D | 24 |

Study C has no skipped file.

---

## Load a study in Python

```python
import pandas as pd

r = "results/pannuke_resnet50/xai_audit/"
a = pd.read_csv(r + "rows.csv").merge(pd.read_csv(r + "rows_lrp.csv"), on="index")  # add the LRP columns
a = pd.concat([a, pd.read_csv(r + "rows_dark.csv")]).drop_duplicates("index")        # add dark-stain images

print(len(a))                                    # 659 images
print((a.dice_c < 0.5).sum())                    # 75 failures
print((a["agree_logit|cam@dec"] >= 0.95).mean()) # 0.994: decoder Grad-CAM copies the control on 99% of images
print(a["del_gain_region|hires@enc4"].mean())    # -0.355: encoder maps rank pixels worse than random
print(a["del_gain_region|lrp"].mean())           # -0.309: so does LRP
```

Studies B and D need only `rows.csv` (plus `rows_dark.csv` for B), because their files already include LRP. Study C is `rows_monuseg.csv` on its own.

---

## Regenerating the results

The analysis scripts read from `outputs/<run>/xai_audit/`. To rebuild tables from these saved rows, copy the `rows*.csv` files back under that path, then run:

```bash
# Study A tables, figures and summary.json
python -m medseg.explain.xai_analysis --run outputs/pannuke_resnet50

# Side-by-side synthesis of all four studies
python -m medseg.explain.xai_replication --out results/replication --studies A,B,C,D
```

Re-running the audits themselves, the depth sweep, the LRP ablation and the explorer build needs the model checkpoints and datasets. The commands are in the [main README](../README.md#reproducing-the-experiments).
