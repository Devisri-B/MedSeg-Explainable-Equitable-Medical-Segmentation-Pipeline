"""Tests for the explanation-audit code. They use the small built-in MiniUNet, so they run
anywhere in a few seconds and need no data or trained weights."""
import numpy as np
import pytest
import torch

from medseg.explain import xai_metrics as xm
from medseg.explain.attribution import MultiAttribution
from medseg.models import MiniUNet


def _model_and_image(seed=0):
    torch.manual_seed(seed)
    model = MiniUNet(3, 6).eval()
    x = torch.randn(1, 3, 64, 64)
    return model, x


def test_multi_attribution_returns_all_methods_in_unit_range():
    model, x = _model_and_image()
    attr = MultiAttribution(model, {"dec": model.u1})
    region = torch.zeros(64, 64, dtype=torch.bool)
    region[10:40, 10:40] = True
    maps, logits = attr(x, 1, region=region)
    assert set(maps) == {"cam@dec", "hires@dec", "gxi", "logit"}
    for m in maps.values():
        assert m.shape == (64, 64)
        assert 0.0 <= m.min() and m.max() <= 1.0 + 1e-6
    attr.remove()


def test_empty_region_gives_no_maps():
    model, x = _model_and_image()
    attr = MultiAttribution(model, {"dec": model.u1})
    maps, _ = attr(x, 1, region=torch.zeros(64, 64, dtype=torch.bool))
    assert maps == {}
    attr.remove()


def test_attribution_hooks_survive_plain_inference():
    # A forward pass under no_grad must not crash the hooks (regression test).
    model, x = _model_and_image()
    attr = MultiAttribution(model, {"dec": model.u1})
    with torch.no_grad():
        model(x)
    attr.remove()


def test_enrichment_and_concentration():
    mask = np.zeros((32, 32), bool)
    mask[:16, :16] = True                                  # 25% of the image
    inside = mask.astype(float)
    assert abs(xm.enrichment(inside, mask) - 4.0) < 1e-6    # all weight in 25% area -> 4x chance
    uniform = np.ones((32, 32))
    assert abs(xm.enrichment(uniform, mask) - 1.0) < 1e-6   # spread evenly -> chance level
    assert abs(xm.concentration(uniform) - 1.0) < 1e-6
    spike = np.zeros((32, 32)); spike[3, 3] = 1.0
    assert xm.concentration(spike) < 1e-6


def test_spearman_and_downsample():
    a = np.arange(64.0).reshape(8, 8)
    assert abs(xm.spearman(a, a * 3 + 1) - 1.0) < 1e-9
    assert np.isnan(xm.spearman(a, np.ones_like(a)))
    assert xm.downsample(np.ones((64, 64)), 4).shape == (16, 16)


def test_dice_binary():
    a = np.zeros((8, 8), bool); a[:4] = True
    assert xm.dice_binary(a, a) == 1.0
    assert xm.dice_binary(a, ~a) == 0.0
    assert np.isnan(xm.dice_binary(np.zeros_like(a), np.zeros_like(a)))


def test_deletion_curve_starts_at_one_and_beats_nothing_baseline():
    model, x = _model_and_image()
    with torch.no_grad():
        pred = model(x).argmax(1)[0].numpy()
    c = int(np.bincount(pred.ravel()).argmax())
    region = pred == c
    rng = np.random.RandomState(0)
    orders = {"random": rng.permutation(region.size), "region_random": xm._region_random_order(region, rng)}
    curves = xm.deletion_curves(model, x, region, c, orders, fractions=(0.1, 0.3))
    for v in curves.values():
        assert v[0] == 1.0 and len(v) == 3 and np.all(np.isfinite(v))
    auc = xm.curve_auc(curves["random"], (0.1, 0.3))
    assert np.isfinite(auc)


def test_region_random_order_visits_region_first():
    region = np.zeros((16, 16), bool); region[:4, :4] = True
    order = xm._region_random_order(region, np.random.RandomState(1))
    assert set(order[: region.sum()]) == set(np.flatnonzero(region.ravel()))
    assert len(set(order)) == region.size


def test_binary_foreground_wrapper_shapes_and_values():
    smp = pytest.importorskip("segmentation_models_pytorch")

    from medseg.explain.binary_fg import BinaryForeground

    torch.manual_seed(0)
    model = smp.Unet("resnet18", encoder_weights=None, classes=6).eval()
    x = torch.randn(1, 3, 64, 64)
    wrapped = BinaryForeground(model).eval()
    with torch.no_grad():
        base = model(x)
        out = wrapped(x)
    assert out.shape == (1, 2, 64, 64)
    assert torch.allclose(out[:, 0], base[:, 0])
    assert torch.allclose(out[:, 1], torch.logsumexp(base[:, 1:], dim=1), atol=1e-6)
    # logsumexp is at least the biggest nucleus-class logit
    assert bool((out[:, 1] >= base[:, 1:].max(dim=1).values - 1e-6).all())


def test_lrp_returns_unit_range_map_and_none_for_empty_region():
    pytest.importorskip("zennit")
    from medseg.explain.lrp import LRPAttribution

    model, x = _model_and_image()
    lrp = LRPAttribution(model)
    region = torch.zeros(64, 64, dtype=torch.bool)
    region[10:40, 10:40] = True
    m, total, score = lrp(x, 1, region)
    assert m.shape == (64, 64) and 0.0 <= m.min() and m.max() <= 1.0 + 1e-6
    assert np.isfinite(total) and np.isfinite(score)
    none_map, _, _ = lrp(x, 1, torch.zeros(64, 64, dtype=torch.bool))
    assert none_map is None


def test_lrp_does_not_modify_the_original_model():
    pytest.importorskip("zennit")
    from medseg.explain.lrp import LRPAttribution

    model, x = _model_and_image()
    with torch.no_grad():
        before = model(x).clone()
    lrp = LRPAttribution(model)
    region = torch.ones(64, 64, dtype=torch.bool)
    lrp(x, 2, region)
    with torch.no_grad():
        after = model(x)
    assert torch.allclose(before, after)


def test_lrp_and_cams_work_through_binary_foreground_wrapper():
    # Regression test: the wrapper used to register encoder/decoder twice, and LRP then
    # failed with "backward through the graph a second time".
    smp = pytest.importorskip("segmentation_models_pytorch")
    pytest.importorskip("zennit")

    from medseg.explain.binary_fg import BinaryForeground
    from medseg.explain.attribution import default_layers
    from medseg.explain.lrp import LRPAttribution

    torch.manual_seed(0)
    model = BinaryForeground(smp.Unet("resnet18", encoder_weights=None, classes=6)).eval()
    x = torch.randn(1, 3, 64, 64)
    region = torch.zeros(64, 64, dtype=torch.bool)
    region[8:40, 8:40] = True
    maps, _ = MultiAttribution(model, default_layers(model))(x, 1, region=region)
    assert {"cam@dec", "cam@enc4", "gxi", "logit"} <= set(maps)
    m, _, _ = LRPAttribution(model)(x, 1, region)
    assert m is not None and m.shape == (64, 64)
