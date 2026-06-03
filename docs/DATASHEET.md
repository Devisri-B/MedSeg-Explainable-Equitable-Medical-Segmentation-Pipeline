# Datasheet — PanNuke (and the synthetic fallback)

Following *Datasheets for Datasets* (Gebru et al., 2018/2021). Documenting data
provenance is itself a responsible-AI practice (transparency + privacy).

## Motivation
- **Purpose:** PanNuke supports nucleus instance segmentation and classification
  across many tissue types — a strong benchmark for multi-class biological
  segmentation and, because tissue labels are provided, for *fairness analysis*.
- **Created by:** Gamper, Alemi Koohbanani, et al. (Tissue Image Analytics Centre,
  University of Warwick).

## Composition
- **Instances:** ~7,900 image patches of 256×256 px, H&E-stained.
- **Classes (nuclei):** Neoplastic, Inflammatory, Connective/Soft-tissue, Dead,
  Epithelial — plus Background. The **Dead** class encodes degraded/necrotic tissue,
  central to the "healthy vs. degraded" analysis in this project.
- **Tissue types:** 19 (breast, colon, lung, bladder, kidney, prostate, etc.).
- **Splits:** distributed as 3 folds; we train on Folds 1–2 and test on Fold 3.
- **Label format:** per-channel instance maps `(N,256,256,6)`; we collapse these to a
  single semantic label map (see `medseg/data/pannuke.py::masks_to_semantic`).
- **Known imbalances/biases:** tissue types are unevenly represented; the Dead class
  is rare; staining/scanner characteristics vary by source institution. These are
  exactly the conditions the fairness audit and drift monitor are designed to surface.

## Collection process
- Curated from multiple public histopathology sources; nuclei semi-automatically
  pre-segmented then expert-refined. See the PanNuke papers for the full protocol.

## Preprocessing / labeling (in this repo)
- Images clipped to `uint8`; instance channels argmax-collapsed to semantic labels.
- ImageNet normalisation; augmentation = flips/rotations + brightness/contrast/hue
  jitter (the hue jitter is deliberate — it improves robustness to stain variation).

## Uses
- **This project:** segmentation, quantification, explainability, fairness, monitoring.
- **Should not be used for:** clinical decisions; training that ignores the
  non-commercial license; cross-tissue claims without per-tissue evaluation.

## Distribution & license
- PanNuke is released for **non-commercial academic research**
  (CC BY-NC-SA 4.0). Cite the original papers; do not redistribute commercially.
- Official source: `https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke`.

## Privacy / PHI
- PanNuke patches are de-identified tissue images with no direct patient identifiers,
  consistent with HIPAA Safe-Harbor expectations for research imagery (see
  [REGULATORY.md](REGULATORY.md)). Using a public, consented research dataset avoids
  handling PHI in a portfolio context.

## Synthetic fallback (`medseg/data/synthetic.py`)
- A pure-NumPy generator produces H&E-like images with tissue-specific class
  compositions. It contains **no real patient data**, runs without any download, and
  exists so the full pipeline (and the smoke test) is reproducible anywhere.

## Citation
> Gamper J., Alemi Koohbanani N., et al. *PanNuke: an open pan-cancer histology
> dataset for nuclei instance segmentation and classification.* (2019); and the
> extended *PanNuke Dataset Extension, Insights and Baselines* (2020).
