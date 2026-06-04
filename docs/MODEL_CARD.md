# Model Card: MedSeg-RAI Histopathology Segmenter

This card follows the Model Cards for Model Reporting framework (Mitchell et al., 2019).
It documents a research and portfolio model. It is not an FDA-cleared medical device and
must not be used for diagnosis or treatment.

## Model details
- Developer: portfolio project (add your name).
- Version: v0.2, trained June 2026.
- Architecture: U-Net++ with an ImageNet-pretrained ResNet-50 encoder, built with
  segmentation-models-pytorch. The framework also supports U-Net, DeepLabV3+, FPN, and
  MAnet, plus a dependency-free MiniUNet fallback. The baseline run used a ResNet-34 U-Net.
- Task: multi-class semantic segmentation, 6 classes: Background, Neoplastic, Inflammatory,
  Connective, Dead, Epithelial.
- Inputs: 256 by 256 RGB H&E patches, ImageNet-normalised.
- Outputs: per-pixel class map and softmax probabilities (used for uncertainty).
- Loss: Cross-Entropy plus Focal-Tversky, with square-root class weighting clipped at 10 to
  handle imbalance. The baseline used Cross-Entropy plus Dice.
- Training: 100 epochs, AdamW, cosine schedule, HED stain augmentation, checkpoint selected
  on the robust mean Dice (Dead excluded). Hardware was a single Apple Silicon GPU (MPS),
  about 9 minutes per epoch for the ResNet-50 model.

## Intended use
- Intended: methods demonstration and research on tissue and nucleus quantification,
  explainability, fairness auditing, and drift monitoring.
- Out of scope: any clinical, diagnostic, or treatment use; non-H&E modalities without
  retraining; whole-slide inference without appropriate tiling and quality control.

## Factors
- Subgroups evaluated: tissue type (19 in PanNuke) and stain-brightness bins, a proxy for
  scanner and staining-protocol variation.
- Environment: H&E staining, with scanner and lab varying by source site.

## Metrics
- Primary: mean foreground Dice and IoU, background excluded.
- Secondary: per-class Dice and IoU, pixel accuracy, confusion matrix.
- Robust mean: foreground Dice with the Dead class excluded, since that class is rare and
  noisy and would otherwise mask performance on the rest.
- Quantification readouts: per-class object counts, area fractions, and a tissue-degradation
  index (Dead area over total cellular area).
- Fairness: worst-group Dice, best-minus-worst gap, worst-over-best ratio (four-fifths rule),
  and coefficient of variation.

## Evaluation data
- PanNuke fold 3, held out from training folds 1 and 2 (2722 test images). See
  [DATASHEET.md](DATASHEET.md).

## Quantitative results (PanNuke fold 3, with test-time augmentation)
Headline: mean foreground Dice 0.644, robust mean 0.714, mean foreground IoU 0.492, pixel
accuracy 0.919.

| class | Dice |
|---|---|
| Background | 0.962 |
| Neoplastic | 0.794 |
| Epithelial | 0.757 |
| Inflammatory | 0.671 |
| Connective | 0.633 |
| Dead | 0.364 |

Development progress: the baseline U-Net with a ResNet-34 encoder reached 0.554 mean
foreground Dice with Dead at 0.152. The final model reaches 0.644 with Dead at 0.364. Full
details and figures are in [RESULTS.md](RESULTS.md).

## Fairness findings
The audit runs without TTA, so its overall figure is 0.635. It shows real disparities and
flags them:
- By tissue: best is Testis at 0.714, worst is Uterus at 0.396. The gap is 0.318 and the
  worst-over-best ratio is 0.56, which fails the four-fifths rule. Several groups have small
  test counts (for example Pancreatic 28, Kidney 41), so their numbers are noisy.
- By stain brightness: bright 0.651, medium 0.599, dark 0.454. The model is clearly weaker on
  dark slides, which is a stain and scanner effect.

These gaps are reported rather than averaged away. They point to the next work: more coverage
for the weak tissues and stronger stain normalisation.

## Quantification findings
Across 2722 test images the mean neoplastic fraction is about 0.41 of cellular area. The mean
degradation index is about 0.006 overall but reaches 0.24 for Lung, which is consistent with
lung samples carrying more necrotic tissue.

## Ethical considerations
- Histopathology data carries site, scanner, and stain biases that can become model biases,
  so the project audits per-subgroup performance and documents drift sensitivity.
- A confident-looking segmentation can mislead, so uncertainty maps and Grad-CAM keep outputs
  inspectable.
- See [ETHICS_IEEE.md](ETHICS_IEEE.md) for the IEEE-aligned assessment and
  [REGULATORY.md](REGULATORY.md) for FDA and HIPAA considerations.

## Caveats and recommendations
- Validate on data from the target deployment site before any real use.
- Monitor data drift continuously (see `medseg/monitoring`) and retrain or recalibrate when
  drift or Dice thresholds are breached, under a predetermined change-control plan.
- Report worst-group performance, not just the average.
- The Dead class remains the weakest and would benefit from more labelled examples.
