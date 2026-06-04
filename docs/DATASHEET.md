# Datasheet: PanNuke

This follows Datasheets for Datasets (Gebru et al., 2018 and 2021). Documenting data
provenance is itself a responsible-AI practice that supports transparency and privacy.

## Motivation
- Purpose: PanNuke supports nucleus instance segmentation and classification across many
  tissue types. It is a strong benchmark for multi-class biological segmentation, and because
  tissue labels are provided, it also supports fairness analysis.
- Created by: Gamper, Alemi Koohbanani, and colleagues at the Tissue Image Analytics Centre,
  University of Warwick.

## Composition
- Instances: about 7,900 image patches of 256 by 256 pixels, H&E-stained.
- Classes (nuclei): Neoplastic, Inflammatory, Connective or Soft-tissue, Dead, and Epithelial,
  plus Background. The Dead class encodes degraded or necrotic tissue, which is central to the
  healthy versus degraded analysis in this project.
- Tissue types: 19 (breast, colon, lung, bladder, kidney, prostate, and more).
- Splits: distributed as 3 folds. This project trains on folds 1 and 2 and tests on fold 3.
  After a 15 percent validation split, that is about 4,402 training images, 777 validation
  images, and 2,722 test images.
- Label format: per-channel instance maps of shape (N, 256, 256, 6), which this project
  collapses to a single semantic label map (`medseg/data/pannuke.py`, masks_to_semantic).
- Known imbalances and biases: tissue types are unevenly represented, the Dead class is rare,
  and staining and scanner characteristics vary by source institution. These are the conditions
  the fairness audit and drift monitor are designed to surface, and the audit does find real
  gaps across tissue and stain brightness (see [RESULTS.md](RESULTS.md)).

## Collection process
- Curated from multiple public histopathology sources. Nuclei were semi-automatically
  pre-segmented and then expert-refined. See the PanNuke papers for the full protocol.

## Access in this repo
- The data is pulled from the Hugging Face Hub mirror (RationAI/PanNuke) and cached locally as
  numpy arrays by `scripts/download_data.py` and `medseg/data/pannuke.py` (prepare_from_hf). The
  raw images are never committed to this repository.

## Preprocessing and labeling in this repo
- Images are clipped to uint8, and instance channels are argmax-collapsed to semantic labels.
- ImageNet normalisation is applied. Training augmentation includes flips and rotations,
  brightness and contrast jitter, and HED stain jitter, which perturbs the Haematoxylin, Eosin,
  and DAB stain channels to improve robustness to stain variation.
- A `limit` option can subsample a few images per fold for quick runs and the test suite.

## Uses
- This project: segmentation, quantification, explainability, fairness, and monitoring.
- Should not be used for: clinical decisions, training that ignores the non-commercial license,
  or cross-tissue claims without per-tissue evaluation.

## Distribution and license
- PanNuke is released for non-commercial academic research (CC BY-NC-SA 4.0). Cite the original
  papers and do not redistribute commercially.
- Official source: https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke

## Privacy and PHI
- PanNuke patches are de-identified tissue images with no direct patient identifiers, which is
  consistent with HIPAA Safe-Harbor expectations for research imagery (see
  [REGULATORY.md](REGULATORY.md)). Using a public, consented research dataset avoids handling
  protected health information in a portfolio context.

## Citation
- Gamper J., Alemi Koohbanani N., et al. PanNuke: an open pan-cancer histology dataset for nuclei
  instance segmentation and classification (2019), and the extended PanNuke Dataset Extension,
  Insights and Baselines (2020).
