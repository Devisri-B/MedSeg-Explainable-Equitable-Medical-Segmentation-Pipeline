"""U-Net builder with an Apple-Silicon-aware device selector.

Primary path uses `segmentation-models-pytorch` (ImageNet-pretrained encoders).
If that package is unavailable, we fall back to a compact built-in U-Net so the
project remains fully runnable with only torch installed.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def get_device(prefer: str = "auto") -> torch.device:
    """Select compute device. 'auto' prefers Apple MPS, then CUDA, then CPU."""
    if prefer and prefer != "auto":
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# Friendly architecture names -> segmentation-models-pytorch decoder classes.
_SMP_ARCH = {
    "unet": "Unet",
    "unetplusplus": "UnetPlusPlus",
    "unet++": "UnetPlusPlus",
    "deeplabv3plus": "DeepLabV3Plus",
    "deeplabv3+": "DeepLabV3Plus",
    "fpn": "FPN",
    "manet": "MAnet",
}


def build_model(
    num_classes: int,
    encoder: str = "resnet34",
    encoder_weights: str | None = "imagenet",
    in_channels: int = 3,
    arch: str = "unet",
) -> nn.Module:
    """Build a segmentation model.

    `arch` selects the segmentation-models-pytorch decoder family (Unet, Unet++,
    DeepLabV3+, FPN, MAnet). Falls back to the dependency-free MiniUNet *only* if
    segmentation-models-pytorch is not installed.
    """
    arch_key = (arch or "unet").lower()
    try:
        import segmentation_models_pytorch as smp
    except Exception as exc:  # noqa: BLE001
        print(f"[model] segmentation-models-pytorch unavailable ({exc}); "
              f"using built-in MiniUNet.")
        model = MiniUNet(in_channels, num_classes)
        model.medseg_backend = "mini"
        model.medseg_arch = "miniunet"
        return model

    cls_name = _SMP_ARCH.get(arch_key)
    if cls_name is None:
        raise ValueError(f"Unknown arch {arch!r}; choose from {sorted(_SMP_ARCH)}")
    model = getattr(smp, cls_name)(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
    )
    model.medseg_backend = "smp"
    model.medseg_arch = arch_key
    return model


class _DoubleConv(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class MiniUNet(nn.Module):
    """A small but real 4-level U-Net (input H,W must be divisible by 8)."""

    def __init__(self, in_channels: int = 3, num_classes: int = 6, base: int = 32):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8]
        self.d1 = _DoubleConv(in_channels, c[0])
        self.d2 = _DoubleConv(c[0], c[1])
        self.d3 = _DoubleConv(c[1], c[2])
        self.d4 = _DoubleConv(c[2], c[3])
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(c[3], c[2], 2, stride=2)
        self.u3 = _DoubleConv(c[3], c[2])
        self.up2 = nn.ConvTranspose2d(c[2], c[1], 2, stride=2)
        self.u2 = _DoubleConv(c[2], c[1])
        self.up1 = nn.ConvTranspose2d(c[1], c[0], 2, stride=2)
        self.u1 = _DoubleConv(c[1], c[0])
        self.head = nn.Conv2d(c[0], num_classes, 1)

    def forward(self, x):
        x1 = self.d1(x)
        x2 = self.d2(self.pool(x1))
        x3 = self.d3(self.pool(x2))
        x4 = self.d4(self.pool(x3))
        y = self.u3(torch.cat([self.up3(x4), x3], dim=1))
        y = self.u2(torch.cat([self.up2(y), x2], dim=1))
        y = self.u1(torch.cat([self.up1(y), x1], dim=1))
        return self.head(y)


def find_last_conv(model: nn.Module) -> nn.Module:
    """Return the last Conv2d before the segmentation head (Grad-CAM target layer)."""
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    if not convs:
        raise ValueError("No Conv2d layer found in model.")
    # Skip a final 1x1 classification conv if a larger-kernel conv precedes it.
    for conv in reversed(convs):
        if conv.kernel_size != (1, 1):
            return conv
    return convs[-1]


def default_target_layer(model: nn.Module) -> nn.Module:
    """Best Grad-CAM target: the last *feature* map before the classification head.

    For segmentation-models-pytorch we hook the final decoder block (its output is
    a rich feature map, not the low-dim logits). For the built-in MiniUNet we hook
    the last decoder conv block. Falls back to the last non-1x1 conv otherwise.
    """
    decoder = getattr(model, "decoder", None)
    if decoder is not None:
        blocks = getattr(decoder, "blocks", None)
        if blocks is not None and len(blocks) > 0:
            return blocks[-1]
        return decoder
    if hasattr(model, "u1"):           # MiniUNet
        return model.u1
    return find_last_conv(model)
