"""Several backpropagation-based attribution maps from ONE forward/backward pass.

Why this file exists
--------------------
`seg_gradcam.py` explains a prediction with a single method (Seg-Grad-CAM) at a
single layer. To ask *how trustworthy* an explanation is, we need to compare several
methods on exactly the same prediction. This module computes them together:

  cam@<layer>    Seg-Grad-CAM. Each feature channel is weighted by its gradient
                 AVERAGED over the whole image, then the weighted channels are summed.
  hires@<layer>  HiResCAM-style map. Gradient and activation are multiplied at every
                 pixel (no averaging), then summed over channels. It keeps the
                 spatial detail that the averaging step in Grad-CAM throws away.
  gxi            Gradient x Input. A pixel-level map: how much would the class score
                 change if this input pixel changed, times the pixel's value.
  logit          A CONTROL, not an explanation: the model's own class-score map,
                 ReLU(logit of the target class). It uses no gradients at all. If a
                 real attribution method is no better than this, it is just
                 re-drawing the model's output instead of explaining it.

All three explain the same scalar: the target-class logit summed over the pixels
that the model predicted as that class (or any region you pass in).

The maps are returned at input resolution, scaled to [0, 1].
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F


def _minmax(x: torch.Tensor) -> torch.Tensor:
    x = x - x.min()
    return x / (x.max() + 1e-8)


def encoder_last_stage(model: torch.nn.Module) -> torch.nn.Module:
    """Deepest encoder stage of a segmentation-models-pytorch ResNet-style model."""
    enc = getattr(model, "encoder", None)
    if enc is not None and hasattr(enc, "layer4"):
        return enc.layer4
    raise ValueError("Model has no encoder.layer4; pass `layers` explicitly.")


class MultiAttribution:
    """Compute Seg-Grad-CAM, HiResCAM-style and Gradient x Input maps together.

    Parameters
    ----------
    model:   a segmentation network returning logits of shape (B, C, H, W).
    layers:  mapping {short_name: nn.Module} of layers whose activations the two CAM
             methods use, e.g. {"dec": model.decoder, "enc4": model.encoder.layer4}.
    """

    def __init__(self, model: torch.nn.Module, layers: Dict[str, torch.nn.Module]):
        self.model = model
        self.layers = layers
        self._acts: Dict[str, torch.Tensor] = {}
        self._grads: Dict[str, torch.Tensor] = {}
        self._handles = []
        for name, module in layers.items():
            self._handles.append(module.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name: str):
        def hook(module, inputs, output):
            out = output[0] if isinstance(output, (tuple, list)) else output
            self._acts[name] = out
            # A tensor hook captures d(score)/d(out) during the backward pass. Passes
            # run under torch.no_grad() (plain inference) produce tensors that do not
            # track gradients, and PyTorch refuses hooks on those, so we skip them.
            if out.requires_grad:
                out.register_hook(lambda g, n=name: self._grads.__setitem__(n, g))

        return hook

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    @staticmethod
    def region_of(logits: torch.Tensor, target_class: int) -> torch.Tensor:
        """Boolean (B, H, W) mask of pixels the model assigns to `target_class`."""
        return logits.argmax(dim=1) == target_class

    def __call__(
        self,
        image: torch.Tensor,
        target_class: int,
        region: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, np.ndarray], torch.Tensor]:
        """Return ({method: (H, W) map in [0,1]}, logits) for one image (1, 3, H, W).

        `region` (H, W) bool selects which pixels' logits are summed. If None, the
        pixels predicted as `target_class` are used. Returns empty maps if the region
        is empty.
        """
        self.model.zero_grad(set_to_none=True)
        self._acts.clear()
        self._grads.clear()
        x = image.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            logits = self.model(x)
            if region is None:
                sel = self.region_of(logits, target_class)
            else:
                sel = region.to(logits.device).bool().unsqueeze(0)
            hw = image.shape[-2:]
            if int(sel.sum()) == 0:
                return {}, logits.detach()
            score = logits[:, target_class][sel].sum()
            score.backward()

        maps: Dict[str, np.ndarray] = {}
        for name in self.layers:
            a = self._acts[name].detach()
            g = self._grads[name]
            weights = g.mean(dim=(2, 3), keepdim=True)
            cam = F.relu((weights * a).sum(dim=1, keepdim=True))
            hires = F.relu((g * a).sum(dim=1, keepdim=True))
            for tag, m in (("cam", cam), ("hires", hires)):
                m = F.interpolate(m, size=hw, mode="bilinear", align_corners=False)[0, 0]
                maps[f"{tag}@{name}"] = _minmax(m).cpu().numpy()
        gxi = (x.grad.detach() * x.detach()).abs().sum(dim=1)[0]
        maps["gxi"] = _minmax(gxi).cpu().numpy()
        maps["logit"] = _minmax(F.relu(logits.detach()[0, target_class])).cpu().numpy()
        return maps, logits.detach()


def default_layers(model: torch.nn.Module) -> Dict[str, torch.nn.Module]:
    from medseg.models import default_target_layer

    layers = {"dec": default_target_layer(model)}
    try:
        layers["enc4"] = encoder_last_stage(model)
    except ValueError:
        pass
    return layers
