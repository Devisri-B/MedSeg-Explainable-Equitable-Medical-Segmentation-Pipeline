"""Seg-Grad-CAM: Grad-CAM adapted to semantic segmentation.

Classic Grad-CAM explains a single classification logit. For segmentation we sum
the target-class logit over a region of interest (by default, all pixels predicted
as that class) and backprop that scalar to a decoder feature map. The result is a
heatmap of "which input regions drove this class here", directly answering the
"decision explainability" requirement.

Reference: Vinogradova, Dibrov & Myers, "Towards Interpretable Semantic
Segmentation via Gradient-weighted Class Activation Mapping" (AAAI 2020).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from medseg.models import default_target_layer


class SegGradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: Optional[torch.nn.Module] = None):
        self.model = model
        self.target = target_layer or default_target_layer(model)
        self._acts: Optional[torch.Tensor] = None
        self._grads: Optional[torch.Tensor] = None
        self._fh = self.target.register_forward_hook(self._forward_hook)
        self._bh = self.target.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inputs, output):
        self._acts = output

    def _backward_hook(self, module, grad_input, grad_output):
        self._grads = grad_output[0].detach()

    def __call__(
        self,
        image: torch.Tensor,
        target_class: int,
        region: Union[str, torch.Tensor] = "pred",
    ) -> np.ndarray:
        """Return a [0,1] heatmap (H,W) for `target_class`. image is (1,3,H,W)."""
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = self.model(image)
            if isinstance(region, str) and region == "pred":
                selection = logits.argmax(dim=1) == target_class
            elif region is None or (isinstance(region, str) and region == "all"):
                selection = torch.ones_like(logits[:, 0], dtype=torch.bool)
            else:
                selection = region.to(logits.device).bool()
            if int(selection.sum()) == 0:
                self.model.zero_grad(set_to_none=True)
                return np.zeros(image.shape[-2:], dtype=np.float32)
            score = logits[:, target_class][selection].sum()
            score.backward()

        weights = self._grads.mean(dim=(2, 3), keepdim=True)          # (1,K,1,1)
        cam = F.relu((weights * self._acts.detach()).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.detach().cpu().numpy()

    def remove(self) -> None:
        self._fh.remove()
        self._bh.remove()


def explanation_panel(model, sample, device, class_index: Optional[int] = None, save_path=None):
    """Build a 4-panel figure: input | prediction | Seg-Grad-CAM | uncertainty."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from medseg.data.dataset import denormalize
    from medseg.explain.uncertainty import predictive_entropy
    from medseg.utils import overlay_mask

    image = sample["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(image)
        pred = logits.argmax(dim=1)[0].cpu().numpy()
        entropy = predictive_entropy(logits)[0].cpu().numpy()
    rgb = denormalize(sample["image"])

    if class_index is None:
        vals, counts = np.unique(pred[pred > 0], return_counts=True)
        class_index = int(vals[counts.argmax()]) if len(vals) else 1

    cam_gen = SegGradCAM(model)
    cam = cam_gen(image, class_index)
    cam_gen.remove()

    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    axes[0].imshow(rgb)
    axes[0].set_title("input")
    axes[1].imshow(overlay_mask(rgb, pred))
    axes[1].set_title("prediction")
    axes[2].imshow(rgb)
    axes[2].imshow(cam, cmap="jet", alpha=0.5)
    axes[2].set_title(f"Seg-Grad-CAM (class {class_index})")
    im = axes[3].imshow(entropy, cmap="magma", vmin=0, vmax=1)
    axes[3].set_title("uncertainty (entropy)")
    for ax in axes:
        ax.axis("off")
    fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
        plt.close(fig)
        return None
    return fig


def main() -> None:
    from medseg.data.dataset import build_loaders
    from medseg.evaluate import load_run
    from medseg.utils import ensure_dir

    ap = argparse.ArgumentParser(description="Generate Seg-Grad-CAM + uncertainty panels.")
    ap.add_argument("--run", required=True)
    ap.add_argument("--num", type=int, default=6)
    ap.add_argument("--class-index", type=int, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    model, cfg, _ = load_run(args.run, device=(torch.device(args.device) if args.device else None))
    device = next(model.parameters()).device
    *_, (_, _, test_ds) = build_loaders(cfg)

    out_dir = ensure_dir(Path(args.run) / "explanations")
    n = min(args.num, len(test_ds))
    for i in range(n):
        explanation_panel(model, test_ds[i], device, class_index=args.class_index,
                          save_path=out_dir / f"explain_{i:02d}.png")
    print(f"[explain] wrote {n} explanation panels to {out_dir}")


if __name__ == "__main__":
    main()
