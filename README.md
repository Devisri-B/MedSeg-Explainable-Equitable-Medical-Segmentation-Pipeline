# Heatmaps That Redraw the Prediction

**A control-based audit of attribution maps for nucleus segmentation in H&E histopathology**

This repository contains the code, per-image results and trained checkpoints for the paper
*Heatmaps That Redraw the Prediction: A Control-Based Audit of Attribution Maps for Nucleus
Segmentation in Histopathology* (D. S. Bandaru, 2026). It also contains the segmentation pipeline
the audit is built on (MedSeg-RAI): a multi-class nucleus segmentation model with uncertainty,
fairness, monitoring and governance tooling.

![Attribution maps for three PanNuke images](assets/heatmap_audit_examples.jpg)

*Attribution maps from the main model (U-Net++ / ResNet-50) for a bright-stain, a dark-stain and a
failure case. Under each map: Δ, deletion gain over random pixels in the explained region (above 0 =
informative), and r, rank correlation with the model's own score map (1 = a copy).*

## Summary

Attribution heatmaps are often shown beside medical segmentations as evidence that a model can be
trusted. We test that assumption by scoring six attribution maps against a gradient-free control,
the network's own class-score map. A map that is no better than this control is a copy of the
prediction, however convincing it looks.

| Component | Details |
|---|---|
| Attribution maps | Seg-Grad-CAM and a HiResCAM-style map (decoder and deepest encoder stage), Gradient × Input, layer-wise relevance propagation (LRP) |
| Control | ReLU of the model's own class-score map (no gradients) |
| Measurements | localization, deletion faithfulness against a region-restricted random baseline, stability under stain shift, agreement with the control, parameter randomization, depth sweep, label-free failure detection |
| Studies | three PanNuke models with different decoders, encoders and held-out folds, and model A on 200 MoNuSeg tiles: 3,180 image audits |

### Key findings

1. **Decoder-level maps redraw the prediction.** Seg-Grad-CAM at the decoder correlates 0.95 to 0.99
   with the score map in every study, and its deletion faithfulness is within 0.013 of the control's.
2. **Looking deeper does not help.** Encoder-level maps and LRP rank pixels worse than random pixels
   inside the region in all four studies (deletion gain −0.06 to −0.38), for all three LRP rules
   tested. No encoder depth gives a map that is both informative and not a copy.
3. **Stability hides a stain effect.** Maps are more stable in dark-stained images (pooled probability
   0.73 over 179 dark images) with no consistent difference in Dice; the gap survives adjustment for
   copying in three of four studies.
4. **Failure detection is model-dependent.** Adding explanation signals to predictive uncertainty
   changes AUROC by −0.010 on the strongest model and by +0.034 and +0.014 on two weaker ones.

Deletion gain over region-random pixels, mean (95% bootstrap interval):

| Map | A: U-Net++ / R50, fold 3 (n = 659) | B: FPN / R18, fold 1 (n = 1,333) | C: MoNuSeg (n = 200) | D: U-Net / R18, fold 2 (n = 988) |
|---|---|---|---|---|
| Score map (control) | 0.023 (0.012, 0.035) | −0.045 (−0.053, −0.036) | −0.015 (−0.038, 0.008) | −0.009 (−0.017, −0.001) |
| Seg-Grad-CAM, decoder | 0.021 (0.010, 0.033) | −0.043 (−0.051, −0.034) | −0.025 (−0.049, −0.002) | −0.022 (−0.030, −0.015) |
| HiResCAM-style, decoder | 0.018 (0.008, 0.029) | −0.035 (−0.043, −0.028) | −0.015 (−0.038, 0.007) | 0.010 (0.004, 0.017) |
| Seg-Grad-CAM, encoder | −0.366 (−0.384, −0.349) | −0.115 (−0.127, −0.103) | −0.296 (−0.329, −0.263) | −0.147 (−0.162, −0.133) |
| HiResCAM-style, encoder | −0.355 (−0.372, −0.337) | −0.077 (−0.088, −0.066) | −0.304 (−0.337, −0.272) | −0.055 (−0.069, −0.041) |
| Gradient × Input | −0.109 (−0.123, −0.095) | 0.016 (0.005, 0.026) | −0.154 (−0.175, −0.131) | 0.025 (0.015, 0.034) |
| LRP (ε, z⁺, flat) | −0.309 (−0.330, −0.288) | −0.221 (−0.234, −0.207) | −0.375 (−0.411, −0.340) | −0.172 (−0.186, −0.158) |

Full method, tables and limitations: [docs/EXPLANATION_AUDIT.md](docs/EXPLANATION_AUDIT.md).

<img src="assets/depth_sweep.png" width="440" alt="Depth sweep">

*Depth sweep (model A, 100 images): every encoder stage is worse than random; only the decoder is
above zero, and there the map is a near-copy of the prediction.*

## Installation

```bash
git clone https://github.com/Devisri-B/MedSeg-Explainable-Equitable-Medical-Segmentation-Pipeline
cd MedSeg-Explainable-Equitable-Medical-Segmentation-Pipeline
git lfs install && git lfs pull        # downloads the trained checkpoints
python -m venv .venv && source .venv/bin/activate
pip install -e ".[full]"
pytest -q tests
```

Python 3.10 or newer. Everything runs on CPU; a GPU only speeds up training.

## Data

| Dataset | Use | How to get it |
|---|---|---|
| [PanNuke](https://arxiv.org/abs/2003.10778) | training and studies A, B, D (folds 1 to 3, 19 tissues, 5 nucleus types) | `python scripts/download_data.py --root data/pannuke` (from the Hugging Face Hub) |
| [MoNuSeg](https://monuseg.grand-challenge.org/) | study C (external labs; foreground only) | prepared automatically on the first `--dataset monuseg` run |

Both datasets are released for non-commercial research; please cite the original authors.

## Trained checkpoints

| File (in `checkpoints/`, Git LFS) | Study | Architecture | Trained on → tested on |
|---|---|---|---|
| `study_A_unetpp_resnet50_fold3.pth` | A, C | U-Net++ / ResNet-50 | PanNuke folds 1+2 → fold 3 (C: MoNuSeg) |
| `study_B_fpn_resnet18_fold1.pth` | B | FPN / ResNet-18 | folds 2+3 → fold 1 |
| `study_D_unet_resnet18_fold2.pth` | D | U-Net / ResNet-18 | folds 1+3 → fold 2 |

Each checkpoint stores its own training configuration. To use one, copy it to
`outputs/<run>/best_model.pth`; `medseg.evaluate.load_run("outputs/<run>")` then restores the
model. Checksums are in `checkpoints/SHA256SUMS.txt`. Models B and D were trained for 3 and 4 epochs
and serve as audit subjects rather than tuned segmenters.

## Reproducing the audit

```bash
# place the checkpoints where the scripts expect them
mkdir -p outputs/pannuke_resnet50 outputs/pannuke_fpn_r18_fold1 outputs/pannuke_unet_r18_fold2
cp checkpoints/study_A_unetpp_resnet50_fold3.pth outputs/pannuke_resnet50/best_model.pth
cp checkpoints/study_B_fpn_resnet18_fold1.pth   outputs/pannuke_fpn_r18_fold1/best_model.pth
cp checkpoints/study_D_unet_resnet18_fold2.pth  outputs/pannuke_unet_r18_fold2/best_model.pth

# studies A to D
python -m medseg.explain.xai_audit --run outputs/pannuke_resnet50     --fold 3 --per-tissue 34  --lrp
python -m medseg.explain.xai_audit --run outputs/pannuke_fpn_r18_fold1 --fold 1 --per-tissue 100 --lrp
python -m medseg.explain.xai_audit --run outputs/pannuke_resnet50 --dataset monuseg --per-tissue 25 --lrp --out-name rows_monuseg.csv
python -m medseg.explain.xai_audit --run outputs/pannuke_unet_r18_fold2 --fold 2 --per-tissue 60  --lrp

# additional checks (model A)
python -m medseg.explain.xai_sanity       --run outputs/pannuke_resnet50 --n 40
python -m medseg.explain.xai_depth_sweep  --run outputs/pannuke_resnet50 --n 100
python -m medseg.explain.xai_lrp_ablation --run outputs/pannuke_resnet50 --n 60

# tables, figures and the interactive explorer
python -m medseg.explain.xai_analysis    --run outputs/pannuke_resnet50
python -m medseg.explain.xai_replication --out outputs/replication --studies A,B,C,D
python -m medseg.viz.explorer            --run outputs/pannuke_resnet50
python -m medseg.viz.paper_figures heatmaps --indices 2199,841,2165 --out paper/fig_heatmaps.png
```

Each audit resumes from its own results file if interrupted. Rare dark-stain images can be added
with `--only-dark --exclude-csv rows.csv --out-name rows_dark.csv`. To retrain models B and D, use
`configs/audit_fpn_r18_fold1.yaml` and `configs/audit_unet_r18_fold2.yaml` with `python -m medseg.train`.

## Results included in this repository

[`results/`](results/README.md) holds every measured number, so the analysis can be checked without
re-running any model: one row per audited image for each study, the analysis tables, the replication
summary (`results/replication/replication.json`), training curves for models B and D, and
`results/pannuke_resnet50/xai_audit/explorer.html`, a self-contained interactive explorer that links
per-tissue and per-stain scorecards to the maps of every image (open it in any browser).

## The segmentation pipeline (MedSeg-RAI)

The audited model comes from a full segmentation pipeline for five nucleus types plus background
across 19 tissue types.

| Component | Location |
|---|---|
| U-Net / U-Net++ / FPN training with Focal-Tversky loss, HED stain augmentation and TTA | [`medseg/train.py`](medseg/train.py), [`medseg/models/`](medseg/models), [`configs/`](configs) |
| Evaluation, per-class Dice and IoU, confusion matrix | [`medseg/evaluate.py`](medseg/evaluate.py), [`medseg/metrics.py`](medseg/metrics.py) |
| Nucleus counts, area fractions and a tissue-degradation index | [`medseg/quantify.py`](medseg/quantify.py) |
| Attribution maps, LRP, uncertainty and the explanation audit | [`medseg/explain/`](medseg/explain) |
| Per-tissue and per-stain fairness audit | [`medseg/fairness/`](medseg/fairness) |
| Data-drift and performance monitoring | [`medseg/monitoring/`](medseg/monitoring) |
| Interactive demo and Hugging Face deployment | [`medseg/app/`](medseg/app), [`deploy/`](deploy) |
| Model card, datasheet, ethics and regulatory notes | [`docs/`](docs) |

On the PanNuke fold-3 test split (2,722 images, with test-time augmentation) the main model reaches a
mean foreground Dice of 0.644 and pixel accuracy of 0.919; see [docs/RESULTS.md](docs/RESULTS.md).
A live demo runs at [huggingface.co/spaces/Devisri515/medseg-rai](https://huggingface.co/spaces/Devisri515/medseg-rai).

```bash
python -m medseg.train    --config configs/improved.yaml --run-name pannuke_resnet50
python -m medseg.evaluate --run outputs/pannuke_resnet50 --tta
bash scripts/make_report.sh outputs/pannuke_resnet50
```

## Repository structure

```
medseg/
  data/          PanNuke and MoNuSeg loaders, augmentation
  models/        U-Net, U-Net++, FPN builders
  explain/       attribution maps, LRP, audit metrics, audit runner, analysis, replication
  viz/           interactive explorer and paper figures
  fairness/      subgroup audit
  monitoring/    drift detection
  app/           Gradio demo
configs/         training configurations
checkpoints/     trained models for studies A, B, D (Git LFS)
results/         per-image audit results, tables, figures, explorer
docs/            explanation-audit method, results, model card, datasheet, governance
deploy/          Hugging Face Space
scripts/         data download and report generation
tests/           unit and smoke tests
```

## Limitations

The audit covers H&E nuclei on two datasets and three models; models B and D are short-trained.
LRP relevance is not conserved in this setup (input relevance is −14% to +16% of the explained
score). Deletion fills pixels with the ImageNet mean colour, and stain brightness is a proxy for
scanner and laboratory. Attribution measures model sensitivity, not tissue causation. This is
research code on public data, not a medical device.

## Citation

```bibtex
@misc{bandaru2026heatmaps,
  author = {Bandaru, Devi Sri},
  title  = {Heatmaps That Redraw the Prediction: A Control-Based Audit of Attribution Maps
            for Nucleus Segmentation in Histopathology},
  year   = {2026},
  note   = {Code: https://github.com/Devisri-B/MedSeg-Explainable-Equitable-Medical-Segmentation-Pipeline}
}
```

Please also cite PanNuke (Gamper et al., 2019, 2020), MoNuSeg (Kumar et al., 2017) and Zennit
(Anders et al., 2021) when using this work.

## License

Code is released under the [MIT License](LICENSE). The datasets keep their own licences.
