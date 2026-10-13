# Explanation audit: can the model's explanations be trusted, and for whom?

This document provides the technical protocols, audit metrics, and extended results for the paper:
> **Heatmaps That Redraw the Prediction: A Control-Based Audit of Attribution Maps for Nucleus Segmentation in Histopathology**  
> Devi Sri Bandaru and Deepshikha Bhati (Department of Computer Science, Kent State University, 2026)

The fairness audit (`medseg/fairness/`) asks whether the model segments every subgroup
equally well. This audit asks the follow-up question: **when the model explains itself, is
the explanation equally trustworthy for every subgroup, and can it warn us when the
segmentation is wrong?**

A heatmap that looks sensible is not evidence that it is faithful. Here every explanation
is turned into numbers that can be compared across methods and across tissue and stain
groups.

## What is measured

For each audited image, the model's most common predicted nucleus class is explained with
six attribution maps, plus one control:

| Map | What it is |
|---|---|
| `cam@dec`, `cam@enc4` | Seg-Grad-CAM at the last decoder block and the deepest encoder stage. Channels are weighted by their gradient averaged over the image. |
| `hires@dec`, `hires@enc4` | HiResCAM-style: gradient times activation at every pixel, no averaging. |
| `gxi` | Gradient x Input, a pixel-level map. |
| `lrp` | Layer-wise Relevance Propagation (zennit; epsilon rule, z+ on convolutions, flat on the first layer, batch-norm merged, residual additions handled by the ResNet canonizer). The seed is the class score map inside the explained region. |
| `logit` | **Control, not an explanation.** The model's own class-score map, ReLU(logit of the class). No gradients. A method that is no better than this is re-drawing the prediction. |

Per map, four kinds of measurement are recorded (`medseg/explain/xai_metrics.py`):

1. **Localization.** How much more of the map's weight sits inside the explained region than
   chance would put there (`loc_enrich`), and the share inside the ground-truth mask.
2. **Faithfulness (deletion test).** Hide the top 10% and 30% most important pixels and
   re-run the model. The map is informative if the model's confidence falls faster than when
   random pixels are hidden. Two random baselines are used: fully random, and random but
   restricted to the explained region first (a stricter test of whether the *ranking inside
   the region* matters). The headline number is the **deletion gain over region-random**
   (above 0 = informative, below 0 = worse than hiding random pixels in the region).
3. **Stability.** Re-explain the same pixels after a small random stain shift (HED jitter,
   about 1.6 times the training augmentation strength) and compute the rank correlation of
   the two maps.
4. **Concentration.** Normalised entropy of the map (0 = concentrated, 1 = spread evenly).

Label-free signals (uncertainty, stability, agreement between methods) are then tested as
failure detectors: can they flag images where the explained class has Dice below 0.5, and do
they add anything to uncertainty alone? (Repeated stratified cross-validation, bootstrap
intervals.)

Extra checks, each a separate script:

| Script | Question |
|---|---|
| `xai_sanity.py` | Do explanations change when the network is re-initialised? (head, decoder, whole network) |
| `xai_depth_sweep.py` | At which depth of the network should a class-activation map be taken? (encoder stages 1 to 4 and decoder) |
| `xai_lrp_ablation.py` | Do the conclusions about LRP depend on the LRP rule? (epsilon-plus-flat, epsilon-plus, epsilon-gamma-box) |
| `xai_replication.py` | Do the findings repeat on another model, another test fold and another dataset? |

### Four studies

| Study | Data | Model | Images |
|---|---|---|---|
| A | PanNuke fold 3 | U-Net++ / ResNet-50 (main model; trained on folds 1 and 2) | 659 |
| B | PanNuke fold 1 | FPN / ResNet-18 (trained on folds 2 and 3, 3 epochs) | 1,333 |
| C | MoNuSeg tiles (other labs, different dataset) | model A used as a nucleus-versus-background detector | 200 |
| D | PanNuke fold 2 | U-Net / ResNet-18 (trained on folds 1 and 3, 4 epochs) | 988 |

For C the five nucleus classes are merged into one foreground score (log-sum-exp of the class
logits, `medseg/explain/binary_fg.py`), because MoNuSeg has no nucleus-type labels.
Images were sampled per tissue; the stain-brightness "dark" group was topped up
(`--only-dark`) for A and B because it is rare.

## Run it

```bash
python -m medseg.explain.xai_audit    --run outputs/pannuke_resnet50 --per-tissue 34 --lrp
python -m medseg.explain.xai_audit    --run outputs/pannuke_fpn_r18_fold1 --fold 1 --per-tissue 100 --lrp
python -m medseg.explain.xai_audit    --run outputs/pannuke_resnet50 --dataset monuseg --per-tissue 25 --lrp --out-name rows_monuseg.csv
python -m medseg.explain.xai_sanity   --run outputs/pannuke_resnet50 --n 40
python -m medseg.explain.xai_depth_sweep  --run outputs/pannuke_resnet50 --n 100
python -m medseg.explain.xai_lrp_ablation --run outputs/pannuke_resnet50 --n 60
python -m medseg.explain.xai_analysis --run outputs/pannuke_resnet50
python -m medseg.explain.xai_replication --out outputs/replication --studies A,B,C,D
python -m medseg.viz.explorer         --run outputs/pannuke_resnet50
pytest -q tests/test_xai_audit.py
```

Training configs for the second and third model are `configs/audit_fpn_r18_fold1.yaml` and
`configs/audit_unet_r18_fold2.yaml`. Outputs go to `<run>/xai_audit/` and `outputs/replication/`
(`replication.json`, `replication_methods.csv`, `fig4_replication.png`).

## Results

Deletion gain over region-random pixels, mean (95% bootstrap interval). Above 0 = the map ranks
pixels better than chance; the control is the model's own score map.

| Map | A (n=659) | B (n=1,333) | C (n=200) | D (n=988) |
|---|---|---|---|---|
| Seg-Grad-CAM, decoder | 0.021 (0.010, 0.033) | -0.043 (-0.051, -0.034) | -0.025 (-0.049, -0.002) | -0.022 (-0.030, -0.015) |
| HiResCAM-style, decoder | 0.018 (0.008, 0.029) | -0.035 (-0.043, -0.028) | -0.015 (-0.038, 0.007) | 0.010 (0.004, 0.017) |
| Seg-Grad-CAM, encoder | -0.366 (-0.384, -0.349) | -0.115 (-0.127, -0.103) | -0.296 (-0.329, -0.263) | -0.147 (-0.162, -0.133) |
| HiResCAM-style, encoder | -0.355 (-0.372, -0.337) | -0.077 (-0.088, -0.066) | -0.304 (-0.337, -0.272) | -0.055 (-0.069, -0.041) |
| Gradient x Input | -0.109 (-0.123, -0.095) | 0.016 (0.005, 0.026) | -0.154 (-0.175, -0.131) | 0.025 (0.015, 0.034) |
| LRP (epsilon, z+, flat) | -0.309 (-0.330, -0.288) | -0.221 (-0.234, -0.207) | -0.375 (-0.411, -0.340) | -0.172 (-0.186, -0.158) |
| Score map (control) | 0.023 (0.012, 0.035) | -0.045 (-0.053, -0.036) | -0.015 (-0.038, 0.008) | -0.009 (-0.017, -0.001) |

What repeats in all four studies:

1. **Circularity.** Decoder Seg-Grad-CAM has rank correlation 0.991 (A), 0.977 (B), 0.946 (C) and
   0.963 (D) with the model's own score map, and its deletion gain is within 0.013 of the
   control's (paired difference: A -0.002, B +0.002, C -0.010, D -0.013). It re-draws the prediction.
2. **Encoder-level maps and LRP are worse than random pixels in the region** in every study
   (gain -0.06 to -0.38), even though they put more weight inside the region than chance does
   (LRP localization enrichment 3.3 to 4.0 in A, B and D). They look localized and are not what
   deleting pixels says the model uses.
3. **Stain-shift stability is higher in dark images** (probability a dark image has a more stable
   map than a bright one: A 0.77, B 0.74, C 0.86, D 0.64; pooled 0.73, p < 1e-20; 179 dark images
   in total) while Dice is not different (pooled 0.52, p = 0.32). Part of this is copying: after
   adjusting for how closely the map copies the score map, the dark effect is +0.045 (A, 0.013 to
   0.078), +0.078 (B, 0.061 to 0.092), +0.090 (C, 0.036 to 0.162) and +0.020 (D, -0.029 to 0.067,
   not distinguishable from 0). So the stability gap survives the adjustment in three of four studies.

What does not repeat:

- **Gradient x Input** has a small positive deletion gain on B (0.016) and D (0.025) and a
  negative one on A and C. No claim is made in either direction.
- **Failure detection (Dice below 0.5).** AUROC of uncertainty alone and the change when explanation
  signals are added (95% interval):

  | Study | Failures | Uncertainty | + explanation signals | + explanation + deletion test |
  |---|---|---|---|---|
  | A | 75 of 659 | 0.872 | -0.010 (-0.039, 0.015) | -0.019 (-0.048, 0.008) |
  | B | 330 of 1,333 | 0.780 | +0.034 (0.012, 0.057) | +0.050 (0.027, 0.074) |
  | D | 214 of 988 | 0.849 | +0.014 (0.001, 0.028) | +0.026 (0.009, 0.045) |

  C has no images below Dice 0.5, so no test is possible. Explanation signals help on the two
  weaker models and not on the strongest one.

Depth sweep (model A, 100 images, deletion gain / agreement with the score map): Seg-Grad-CAM at encoder
stages 1 to 4 has gain -0.18, -0.17, -0.15, -0.35 and agreement 0.10, 0.19, 0.41, 0.14; at the
decoder gain +0.036 and agreement 0.99. No depth gives a map that is both informative and not a copy.

LRP rule ablation (model A, 60 images): epsilon-plus-flat, epsilon-plus and epsilon-gamma-box give
deletion gains of -0.34, -0.31 and -0.19. The sign does not depend on the rule.

Parameter randomization (model A, 40 images), correlation inside the explained region after the whole
network is re-initialised: CAM family -0.02 to 0.07, Gradient x Input 0.21. The unrestricted
HiResCAM-decoder value of 0.79 comes from the shared region outline (0.94 correlation with the
region mask).

## Limits

- Two datasets (PanNuke H&E tiles; 200 MoNuSeg tiles, evaluated as foreground only) and three models. Models B and D were trained for 3 and 4 epochs and are
  weaker than A (foreground Dice about 0.5); results on them describe weak models, not a tuned one.
- **LRP does not conserve relevance here.** The total relevance at the input, as a fraction of the
  score being explained, is 0.003 (A), -0.144 (B), -0.049 (C) and 0.162 (D). The LRP maps are
  therefore not a decomposition of the score in the strict sense, and their negative deletion gain
  should be read with that in mind.
- Per-tissue estimates rest on at most about 100 images and carry wide intervals; the dark group has
  33 to 63 images per PanNuke study.
- Deletion replaces pixels with the ImageNet mean colour. That is an artificial image, and
  deleting a pixel also removes the evidence at that pixel, which is why the region-random
  baseline is reported next to the plain random one.
- Stain brightness is a proxy for scanner and lab variation, not a record of either.
- Explanations are computed for one class per image (the most common predicted nucleus class).
- Attribution measures sensitivity of the model, not causation in the tissue.
- Research code on public data, not a medical device.
