"""Build the Explanation Audit Explorer: a single-file interactive page (no server needed).

It links four views of the explanation audit:
  * filters   stain brightness and tissue type (bar = segmentation quality of that group)
  * scatter   segmentation quality vs a chosen explanation signal, one dot per image
  * scorecard how each explanation method behaves on the images currently selected
  * case      the image, ground truth, prediction and every explanation for one image

Run (after medseg.explain.xai_audit):
    python -m medseg.viz.explorer --run outputs/pannuke_resnet50

Writes <run>/xai_audit/explorer.html (standalone) and explorer_body.html (page body only).
"""
from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from medseg.data import pannuke
from medseg.evaluate import load_run
from medseg.explain.attribution import MultiAttribution, default_layers
from medseg.explain.xai_analysis import FAIL_THRESHOLD, LABELS, METHODS, load_rows
from medseg.explain.xai_audit import pick_target_class, to_tensor
from medseg.utils import CLASS_COLORS

TILE = 88
TILE_LABELS = ["Image", "Ground truth", "Prediction", "Grad-CAM dec.", "HiRes dec.", "Grad-CAM enc.",
               "HiRes enc.", "Grad x Input", "LRP", "Score map (control)"]
SIGNALS = [
    ("entropy_region", "Model uncertainty (entropy)", "Uncertainty is the usual failure warning."),
    ("stain_pred_dice", "Prediction stability under stain shift", "Overlap between the prediction before and after a stain shift."),
    ("stab_rho|hires@enc4", "Explanation stability (HiRes, encoder)", "Rank correlation of the map before and after a stain shift."),
    ("stab_rho|cam@dec", "Explanation stability (Grad-CAM, decoder)", "Rank correlation of the map before and after a stain shift."),
    ("agree|cam-hires", "Agreement: Grad-CAM vs HiResCAM", "Do two methods tell the same story?"),
    ("agree|cam-gxi", "Agreement: Grad-CAM vs Gradient x Input", "Do a region-level and a pixel-level method agree?"),
    ("agree_logit|hires@dec", "HiResCAM (decoder) vs model's own score map", "High means the map mostly re-draws the prediction."),
    ("stab_rho|lrp", "Explanation stability (LRP)", "Rank correlation of the LRP map before and after a stain shift."),
    ("agree_logit|lrp", "LRP vs model's own score map", "High means the LRP map mostly re-draws the prediction."),
    ("conc|hires@enc4", "Spread of the map (HiRes, encoder)", "0 is concentrated, 1 is spread evenly."),
    ("del_gain_region|hires@enc4", "Deletion gain over region-random (HiRes, encoder)", "Needs 16 extra model runs per image."),
    ("del_gain_region|lrp", "Deletion gain over region-random (LRP)", "Needs extra model runs per image."),
]


def _blend(img: np.ndarray, rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    a = alpha[..., None]
    return np.clip((1 - a) * img + a * rgb, 0, 255).astype(np.uint8)


def _tile(a: np.ndarray) -> Image.Image:
    return Image.fromarray(a).resize((TILE, TILE), Image.BILINEAR)


def composite(attr: MultiAttribution, lrp, img: np.ndarray, gt: np.ndarray, num_classes: int) -> str | None:
    import matplotlib

    inferno = matplotlib.colormaps["inferno"]
    x = to_tensor(img)
    with torch.no_grad():
        pred = attr.model(x).argmax(1)[0].numpy()
    c = pick_target_class(pred, num_classes)
    if c is None:
        return None
    maps, _ = attr(x, c, region=torch.from_numpy(pred == c))
    if lrp is not None:
        maps["lrp"] = lrp(x, c, torch.from_numpy(pred == c))[0]
    base = img.astype(np.float32)

    def lab(mask: np.ndarray) -> np.ndarray:
        return _blend(base, CLASS_COLORS[mask].astype(np.float32), (mask > 0).astype(np.float32) * 0.55)

    def heat(m: np.ndarray) -> np.ndarray:
        col = (inferno(m)[..., :3] * 255).astype(np.float32)
        return _blend(base, col, 0.8 * m)

    keys = ("cam@dec", "hires@dec", "cam@enc4", "hires@enc4", "gxi") + (("lrp",) if lrp is not None else ()) + ("logit",)
    tiles = [img, lab(gt), lab(pred)] + [heat(maps[k]) for k in keys]
    strip = Image.new("RGB", (TILE * len(tiles), TILE))
    for i, t in enumerate(tiles):
        strip.paste(_tile(t), (i * TILE, 0))
    buf = io.BytesIO()
    strip.save(buf, format="JPEG", quality=82)
    return base64.b64encode(buf.getvalue()).decode()


def pick_cases(df: pd.DataFrame, per_tissue: int = 6) -> set:
    """Per tissue: the 2 lowest-Dice, 2 highest-Dice and 2 random images."""
    keep = set()
    rng = np.random.RandomState(0)
    for _, g in df.groupby("tissue"):
        g = g.sort_values("dice_c")
        keep |= set(g["index"].head(per_tissue // 3))
        keep |= set(g["index"].tail(per_tissue // 3))
        rest = g["index"].values[per_tissue // 3: len(g) - per_tissue // 3]
        if len(rest):
            keep |= set(rng.choice(rest, size=min(per_tissue // 3, len(rest)), replace=False))
    return keep


def build(run: str, data_root: str = "data/pannuke", per_tissue: int = 6) -> Path:
    run_dir = Path(run)
    from medseg.explain.lrp import LRPAttribution
    from medseg.explain.xai_replication import STUDIES, load_study

    df = load_study(STUDIES["A"])          # main audit with the LRP columns joined on
    df = df[df["dice_c"].notna()].reset_index(drop=True)
    has_lrp = "stab_rho|lrp" in df.columns
    model, cfg, _ = load_run(run, device=torch.device("cpu"))
    images, sem, _ = pannuke.load_fold(data_root, 3)
    attr = MultiAttribution(model, default_layers(model))
    lrp = LRPAttribution(model) if has_lrp else None
    chosen = pick_cases(df, per_tissue)
    keys = sorted({k for k, _, _ in SIGNALS} | {f"{p}{m}" for p in ("agree_logit|", "stab_rho|", "del_gain_region|") for m in METHODS}
                  | {"entropy_region"})
    rows = []
    for _, r in df.iterrows():
        i = int(r["index"])
        v = {k: (None if pd.isna(r.get(k, np.nan)) else round(float(r[k]), 4)) for k in keys if k in df.columns}
        img = composite(attr, lrp, images[i], sem[i], cfg.data.num_classes) if i in chosen else None
        rows.append({"i": i, "t": r["tissue"], "b": r["brightness"], "d": round(float(r["dice_c"]), 4), "v": v, "img": img})
    attr.remove()
    data = {
        "rows": rows, "fail_threshold": FAIL_THRESHOLD, "default_x": "entropy_region",
        "signals": [{"key": k, "label": l, "hint": h} for k, l, h in SIGNALS if k in df.columns],
        "methods": [{"key": m, "label": LABELS[m]} for m in METHODS if f"stab_rho|{m}" in df.columns],
        "tile_labels": [t for t in TILE_LABELS if has_lrp or t != "LRP"],
        "lede": (f"{len(rows)} PanNuke test images, sampled across {df.tissue.nunique()} tissue types. Each dot is one image: "
                 "how well the model segmented the nucleus class it was explaining, against a signal that needs no ground truth. "
                 "Select a stain group or tissue to see whether the explanation signals still track segmentation quality there."),
        "footer": ("Research prototype on a public dataset, not a medical device. Failure means the Dice of the explained class is below "
                   f"{FAIL_THRESHOLD}. Explanations are computed for the most common predicted nucleus class in each image."),
    }
    body = (Path(__file__).parent / "explorer_template.html").read_text().replace("/*DATA*/null", json.dumps(data, separators=(",", ":")))
    out_dir = run_dir / "xai_audit"
    (out_dir / "explorer_body.html").write_text(body)
    (out_dir / "explorer.html").write_text(
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"></head><body style=\"margin:0\">"
        + body + "</body></html>")
    print(f"[explorer] {len(rows)} images, {sum(r['img'] is not None for r in rows)} with pictures -> {out_dir / 'explorer.html'}")
    return out_dir / "explorer.html"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data-root", default="data/pannuke")
    ap.add_argument("--per-tissue", type=int, default=6)
    a = ap.parse_args()
    build(a.run, a.data_root, a.per_tissue)


if __name__ == "__main__":
    main()
