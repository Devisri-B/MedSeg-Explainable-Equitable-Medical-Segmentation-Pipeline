---
title: MedSeg-RAI Histopathology Segmentation
emoji: 🔬
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 6.16.0
app_file: app.py
pinned: false
license: mit
---

# MedSeg-RAI: histopathology segmentation with responsible AI

Multi-class nucleus segmentation on H&E histopathology (PanNuke), with Seg-Grad-CAM
explanations, per-pixel uncertainty, and tissue quantification. Upload an H&E patch or
pick an example, choose a Grad-CAM class, and run.

The model is a U-Net++ with a ResNet-50 encoder. On the PanNuke held-out test split it
reaches a mean foreground Dice of 0.644 and pixel accuracy of 0.919. Weights are
downloaded at startup from the model repo `__MODEL_REPO__`.

Not a medical device. Research and portfolio use only.

Source code and full write-up: https://github.com/Devisri-B/EquiSeg-Explainable-Equitable-Medical-Segmentation-Pipeline
