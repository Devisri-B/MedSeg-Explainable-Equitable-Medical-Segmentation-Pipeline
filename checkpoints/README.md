# Trained checkpoints

| File | Study | Architecture | Test fold | Size |
|---|---|---|---|---|
| `study_A_unetpp_resnet50_fold3.pth` | A (main model) | U-Net++ / ResNet-50 | PanNuke fold 3 | 196 MB |
| `study_B_fpn_resnet18_fold1.pth` | B | FPN / ResNet-18, trained on folds 2+3 for 3 epochs | PanNuke fold 1 | 52 MB |
| `study_D_unet_resnet18_fold2.pth` | D | U-Net / ResNet-18, trained on folds 1+3 for 4 epochs | PanNuke fold 2 | 57 MB |

Study A's weights were downloaded from the Hugging Face repo `Devisri515/medseg-rai-pannuke` (file `best_model.pth`) and
audited as they are; this copy is byte-identical to that download (checksums in `SHA256SUMS.txt`).
Models B and D were trained for this audit with `configs/audit_fpn_r18_fold1.yaml` and `configs/audit_unet_r18_fold2.yaml`;
their training curves and settings are in `results/pannuke_fpn_r18_fold1/` and `results/pannuke_unet_r18_fold2/`.
Models B and D are short-trained (foreground Dice about 0.5), so use them as audit subjects, not as tuned segmenters.

The files are stored with Git LFS. After cloning run `git lfs pull` (install with `git lfs install` first), or download
the files from the GitHub page.

To use one: create `outputs/<run>/`, copy the file there as `best_model.pth`, and call
`medseg.evaluate.load_run("outputs/<run>")`. Each checkpoint stores its own training config and weights
(`ckpt["config"]`, `ckpt["state_dict"]`), so no separate config file is needed. The audit commands in
`docs/EXPLANATION_AUDIT.md` then work unchanged.
