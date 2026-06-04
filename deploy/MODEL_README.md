---
license: cc-by-nc-sa-4.0
library_name: pytorch
pipeline_tag: image-segmentation
tags:
  - histopathology
  - medical-imaging
  - semantic-segmentation
  - pannuke
  - pytorch
---

# MedSeg-RAI: PanNuke nucleus segmentation (U-Net++ ResNet-50)

Trained weights (`best_model.pth`) for the MedSeg-RAI demo. U-Net++ with a ResNet-50
encoder, 6-class semantic segmentation of H&E nuclei: Background, Neoplastic,
Inflammatory, Connective, Dead, Epithelial. Trained on PanNuke folds 1 and 2.

Test results on PanNuke fold 3 (with test-time augmentation): mean foreground Dice
0.644, robust mean 0.714, pixel accuracy 0.919.

The checkpoint is a torch dict containing the state_dict, the run config, and the class
names. Load it with the project's `medseg.evaluate.load_run`.

The model was trained on PanNuke (CC BY-NC-SA 4.0), so these weights inherit the same
non-commercial, share-alike terms. Not a medical device; research and portfolio use only.

Live demo and source: see the Hugging Face Space and the GitHub repository.
