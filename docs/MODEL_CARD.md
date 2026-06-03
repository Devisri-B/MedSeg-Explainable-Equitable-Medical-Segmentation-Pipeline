# Model Card — MedSeg-RAI Histopathology Segmenter

Following the *Model Cards for Model Reporting* framework (Mitchell et al., 2019).
This card documents a **research/portfolio** model. It is **not** an FDA-cleared
medical device and must not be used for diagnosis or treatment decisions.

## Model details
- **Developer:** (your name) — portfolio project.
- **Date / version:** v0.1.
- **Architecture:** U-Net with an ImageNet-pretrained encoder (ResNet-34 by default
  via `segmentation-models-pytorch`); a dependency-free `MiniUNet` is used when that
  package is unavailable.
- **Task:** multi-class semantic segmentation, 6 classes — Background, Neoplastic,
  Inflammatory, Connective, Dead, Epithelial.
- **Inputs:** 256×256 RGB H&E patches, ImageNet-normalised.
- **Outputs:** per-pixel class map + softmax probabilities (used for uncertainty).
- **Loss:** Cross-Entropy + soft Dice; median-frequency class weighting for imbalance.
- **Compute:** trains on a single Apple-Silicon GPU (Metal/MPS), CUDA, or CPU.

## Intended use
- **Intended:** methods demonstration; research on tissue/nucleus quantification,
  explainability, fairness auditing, and drift monitoring.
- **Out of scope:** any clinical, diagnostic, or treatment use; non-H&E modalities
  without retraining; whole-slide inference without appropriate tiling/QC.

## Factors
- **Subgroups evaluated:** tissue type (up to 19 in PanNuke) and stain-brightness
  bins (a proxy for scanner/staining-protocol variation).
- **Instrumentation/environment:** H&E staining; scanner and lab vary by source site.

## Metrics
- **Primary:** mean foreground Dice and IoU (background excluded).
- **Secondary:** per-class Dice/IoU, pixel accuracy, confusion matrix.
- **Quantification readouts:** per-class object counts, area fractions, and a
  *tissue-degradation index* (Dead-class area / total cellular area).
- **Fairness:** worst-group Dice, best–worst gap, worst/best ratio (4/5ths rule), CV.

## Evaluation data
- **Default:** PanNuke Fold 3 (held out from training Folds 1–2). See
  [DATASHEET.md](DATASHEET.md).
- **Illustrative smoke run:** a synthetic H&E generator (no PHI) used to verify the
  pipeline; numbers below are from that synthetic run and are **not** clinical results.

## Quantitative analysis (illustrative — synthetic demo)
| class | Dice |
|---|---|
| Background | 0.988 |
| Neoplastic | 0.990 |
| Inflammatory | 0.966 |
| Connective | 0.984 |
| Dead | 0.976 |
| Epithelial | 0.986 |

Mean foreground Dice ≈ 0.98 on synthetic data. **TODO:** replace with PanNuke Fold-3
results after a full training run (`python -m medseg.train --config configs/default.yaml`).
Fairness audit on synthetic data shows worst/best tissue ratio ≈ 1.00 (balanced by
construction); on PanNuke, expect real disparities — report them here honestly.

## Ethical considerations
- Histopathology data can encode site/scanner/stain biases that become model biases;
  we therefore audit per-subgroup performance and document drift sensitivity.
- A confident-looking segmentation can mislead; uncertainty maps and Grad-CAM are
  provided so outputs are inspectable rather than taken on trust.
- See [ETHICS_IEEE.md](ETHICS_IEEE.md) for the IEEE-aligned assessment and
  [REGULATORY.md](REGULATORY.md) for FDA/HIPAA considerations.

## Caveats & recommendations
- Validate on data from the *target* deployment site before any real use.
- Monitor data drift continuously (see `medseg/monitoring`); retrain/recalibrate when
  PSI or Dice breach thresholds, under a predetermined change-control plan.
- Report worst-group, not just average, performance.
