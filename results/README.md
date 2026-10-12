# Explanation audit results

Everything the audit measured, kept so the models do not have to be re-run.

| Folder | Study | What it holds |
|---|---|---|
| `pannuke_resnet50/xai_audit/` | A: PanNuke fold 3, U-Net++ / ResNet-50 | `rows.csv` (one row per audited image), `rows_lrp.csv` (LRP columns added afterwards), `rows_dark.csv` (extra dark-stain images), `rows_monuseg.csv` (study C: 200 MoNuSeg tiles scored with this model), `depth_sweep*.csv`, `lrp_ablation.csv`, `analysis/` (tables, figures, `summary.json`), `explorer.html` (open in a browser, no server needed) |
| `pannuke_fpn_r18_fold1/` | B: PanNuke fold 1, FPN / ResNet-18 | training `config.yaml`, `history.csv`, `summary.json`, `training_curves.png`, and `xai_audit/rows*.csv` |
| `pannuke_unet_r18_fold2/` | D: PanNuke fold 2, U-Net / ResNet-18 | same as B |
| `replication/` | all four studies side by side | `replication.json`, `replication_methods.csv`, `fig4_replication.png` |

Column names in the `rows*.csv` files: `loc_enrich|<map>` localization, `del_gain_region|<map>` deletion gain over
region-random pixels, `stab_rho|<map>` stability under stain jitter, `agree_logit|<map>` agreement with the model's own
score map, `conc|<map>` concentration; `<map>` is one of `cam@dec`, `hires@dec`, `cam@enc4`, `hires@enc4`, `gxi`, `lrp`, `logit`.
Regenerate the combined tables with
`python -m medseg.explain.xai_replication --out results/replication --studies A,B,C,D` after copying the rows files
back under `outputs/<run>/xai_audit/`.
