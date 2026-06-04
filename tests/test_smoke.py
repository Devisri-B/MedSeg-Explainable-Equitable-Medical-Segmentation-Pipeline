"""Smoke tests.

The logic tests use small hand-made arrays or random tensors and run anywhere.
The end-to-end training test trains for one epoch on a tiny slice of real PanNuke
and skips if the dataset has not been downloaded. torch-dependent tests skip if
torch is absent. Run with `pytest -q` or `python -m tests.test_smoke`.
"""
from __future__ import annotations

import numpy as np
import pytest

from medseg.data import pannuke
from medseg.fairness.audit import disparity
from medseg.metrics import SegMetrics
from medseg.monitoring.drift import DriftDetector, extract_features
from medseg.monitoring.monitor import corrupt_images
from medseg.quantify import quantify_mask

DATA_ROOT = "data/pannuke"


# ----------------------------- pure NumPy -----------------------------

def test_pannuke_mask_to_semantic():
    inst = np.zeros((1, 8, 8, 6), dtype=np.uint8)
    inst[0, 2:5, 2:5, 3] = 7          # an instance of the "Dead" channel (idx 3)
    sem = pannuke.masks_to_semantic(inst)
    assert sem.shape == (1, 8, 8)
    assert (sem[0, 2:5, 2:5] == 4).all()   # Dead -> label 4
    assert sem.sum() == 4 * 9               # only that 3x3 block is foreground


def test_metrics_perfect_and_partial():
    m = SegMetrics(num_classes=2)
    target = np.array([[0, 0], [1, 1]])
    m.update(target.copy(), target.copy())
    assert pytest.approx(m.compute()["per_class_dice"]["class_1"], abs=1e-5) == 1.0

    m2 = SegMetrics(num_classes=2)
    pred = np.array([[0, 1], [1, 1]])
    m2.update(pred, target)               # one background pixel misclassified
    assert pytest.approx(m2.compute()["per_class_dice"]["class_1"], abs=1e-5) == 0.8


def test_quantify_degradation_index():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[:, :4] = 5     # Epithelial (healthy) -> 40 px
    mask[:, 4:6] = 4    # Dead (degraded)      -> 20 px
    q = quantify_mask(mask)
    assert q["foreground_px"] == 60
    assert pytest.approx(q["degradation_index"], abs=1e-6) == 20 / 60


def test_disparity_flagging():
    d = disparity({"breast": 0.90, "lung": 0.60})
    assert d["worst_group"] == "lung"
    assert pytest.approx(d["ratio"], abs=1e-6) == 0.6 / 0.9
    assert d["flagged"] is True          # ratio < 0.8


def test_drift_detector():
    rng = np.random.default_rng(1)
    imgs = rng.integers(0, 256, size=(16, 64, 64, 3), dtype=np.uint8)
    feats = extract_features(imgs)
    det = DriftDetector().fit(feats)
    psi_same = det.psi(feats)["overall_psi"]
    psi_shift = det.psi(extract_features(corrupt_images(imgs, severity=0.8, seed=2)))
    assert psi_same < 1e-6
    assert psi_shift["overall_psi"] > 0.1 and psi_shift["flag"] in (True, False)
    assert psi_shift["overall_psi"] > psi_same


# ----------------------------- torch path -----------------------------

def test_model_forward_and_loss():
    torch = pytest.importorskip("torch")
    from medseg.losses import CombinedLoss
    from medseg.models.unet import MiniUNet

    model = MiniUNet(in_channels=3, num_classes=6)
    x = torch.randn(2, 3, 64, 64)
    y = torch.randint(0, 6, (2, 64, 64))
    logits = model(x)
    assert logits.shape == (2, 6, 64, 64)
    loss = CombinedLoss()(logits, y)
    loss.backward()
    assert torch.isfinite(loss)


def test_gradcam_and_uncertainty():
    torch = pytest.importorskip("torch")
    from medseg.explain.seg_gradcam import SegGradCAM
    from medseg.explain.uncertainty import predictive_entropy
    from medseg.models.unet import MiniUNet

    model = MiniUNet(3, 6).eval()
    image = torch.randn(1, 3, 64, 64)
    cam = SegGradCAM(model)(image, target_class=1, region="all")
    assert cam.shape == (64, 64)
    assert cam.min() >= 0.0 and cam.max() <= 1.0 + 1e-5

    ent = predictive_entropy(model(image))
    assert ent.shape == (1, 64, 64)
    assert float(ent.max()) <= 1.0 + 1e-5


def test_train_integration(tmp_path):
    pytest.importorskip("torch")
    if not pannuke.is_available(DATA_ROOT):
        pytest.skip(f"PanNuke not found under {DATA_ROOT}; run scripts/download_data.py")
    from medseg.config import load_config
    from medseg.train import train

    cfg = load_config(overrides={
        "data": {"root": DATA_ROOT, "train_folds": [3], "test_fold": 3, "limit": 24,
                 "image_size": 128, "batch_size": 4, "num_workers": 0, "augment": False},
        "model": {"arch": "unet", "encoder": "resnet18", "encoder_weights": None},
        "train": {"epochs": 1, "device": "cpu", "output_dir": str(tmp_path),
                  "run_name": "smoke", "early_stop_patience": 99},
    })
    summary = train(cfg, use_class_weights=True)
    assert (tmp_path / "smoke" / "best_model.pth").exists()
    assert 0.0 <= summary["best_val_dice_fg"] <= 1.0


if __name__ == "__main__":
    import sys
    import tempfile

    failures = 0
    tests = [
        test_pannuke_mask_to_semantic, test_metrics_perfect_and_partial,
        test_quantify_degradation_index, test_disparity_flagging, test_drift_detector,
        test_model_forward_and_loss, test_gradcam_and_uncertainty,
    ]
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL  {t.__name__}: {exc}")
    try:
        with tempfile.TemporaryDirectory() as d:
            test_train_integration(__import__("pathlib").Path(d))
        print("PASS  test_train_integration")
    except Exception as exc:  # noqa: BLE001
        failures += 1
        print(f"FAIL  test_train_integration: {exc}")
    sys.exit(1 if failures else 0)
