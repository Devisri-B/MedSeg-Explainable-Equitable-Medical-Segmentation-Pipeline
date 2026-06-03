"""Interactive demo: upload an H&E patch -> segmentation + quantification +
Seg-Grad-CAM + uncertainty, all in the browser.

    python -m medseg.app.gradio_app --run outputs/pannuke_unet

The `run_inference` function is deliberately free of any Gradio dependency so it can
be unit-tested and reused; only `build_demo`/`main` need Gradio installed.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from medseg import CLASS_NAMES
from medseg.quantify import quantify_mask
from medseg.utils import ensure_dir, overlay_mask

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def _preprocess(image_rgb: np.ndarray, size: int, device):
    from PIL import Image

    img = Image.fromarray(np.asarray(image_rgb).astype(np.uint8)).convert("RGB").resize(
        (size, size), Image.BILINEAR
    )
    arr = np.asarray(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy((arr - IMAGENET_MEAN) / IMAGENET_STD)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0).float().to(device)
    return tensor, np.asarray(img, dtype=np.uint8)


def _heatmap(arr01: np.ndarray, cmap: str = "magma") -> np.ndarray:
    import matplotlib

    rgba = matplotlib.colormaps[cmap](np.clip(arr01, 0, 1))
    return (rgba[..., :3] * 255).astype(np.uint8)


def run_inference(model, image_rgb, device, size=256, class_names=CLASS_NAMES, gradcam_class=None):
    """Gradio-free inference core. Returns overlay/gradcam/entropy images + readouts."""
    from medseg.explain.seg_gradcam import SegGradCAM
    from medseg.explain.uncertainty import predictive_entropy

    tensor, rgb = _preprocess(image_rgb, size, device)
    with torch.no_grad():
        logits = model(tensor)
        pred = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
        entropy = predictive_entropy(logits)[0].cpu().numpy()

    if gradcam_class is None:
        vals, counts = np.unique(pred[pred > 0], return_counts=True)
        gradcam_class = int(vals[counts.argmax()]) if len(vals) else 1

    cam_gen = SegGradCAM(model)
    cam = cam_gen(tensor, int(gradcam_class))
    cam_gen.remove()

    return {
        "overlay": overlay_mask(rgb, pred),
        "gradcam": (0.5 * rgb + 0.5 * _heatmap(cam, "jet")).astype(np.uint8),
        "entropy": _heatmap(entropy, "magma"),
        "pred": pred,
        "quant": quantify_mask(pred, class_names),
        "gradcam_class": int(gradcam_class),
    }


def _load_model(run):
    from medseg.models import build_model, get_device

    if run and (Path(run) / "best_model.pth").exists():
        from medseg.evaluate import load_run

        model, cfg, _ = load_run(run)
        device = next(model.parameters()).device
        return model, device, cfg.data.image_size, list(cfg.data.class_names), True

    device = get_device("auto")
    model = build_model(len(CLASS_NAMES), encoder_weights=None).to(device).eval()
    return model, device, 256, list(CLASS_NAMES), False


def build_demo(run=None):
    import gradio as gr
    from PIL import Image

    from medseg.data import synthetic

    model, device, size, class_names, trained = _load_model(run)

    ex_dir = ensure_dir("assets/examples")
    example_imgs, _, _ = synthetic.generate_synthetic(3, size, seed=7)
    example_paths = []
    for i, im in enumerate(example_imgs):
        p = ex_dir / f"example_{i}.png"
        Image.fromarray(im).save(p)
        example_paths.append(str(p))

    def predict(image, cls_choice):
        if image is None:
            return None, None, None, None, "Upload or pick an example image first."
        gc = None if cls_choice == "auto" else class_names.index(cls_choice)
        out = run_inference(model, np.asarray(image), device, size, class_names, gc)
        q = out["quant"]
        table = [[n, q["object_counts"][n], round(q["areas_px"][n] / q["image_px"], 4)]
                 for n in class_names]
        summary = (
            f"**Tissue-degradation index:** {q['degradation_index']:.3f}  |  "
            f"**Neoplastic fraction:** {q['neoplastic_fraction']:.3f}  |  "
            f"**Foreground:** {q['foreground_fraction']:.3f}  |  "
            f"**Grad-CAM class:** {class_names[out['gradcam_class']]}"
        )
        return out["overlay"], out["gradcam"], out["entropy"], table, summary

    banner = "" if trained else (
        "\n\n> ⚠️ **No trained model found** — predictions are random. "
        "Train first (`python -m medseg.train ...`) or pass `--run <dir>`."
    )
    with gr.Blocks(title="MedSeg-RAI") as demo:
        gr.Markdown("# MedSeg-RAI — Histopathology segmentation + responsible AI" + banner)
        with gr.Row():
            with gr.Column():
                inp = gr.Image(type="numpy", label="H&E patch")
                cls = gr.Dropdown(["auto"] + class_names[1:], value="auto", label="Grad-CAM class")
                btn = gr.Button("Segment & analyse", variant="primary")
                gr.Examples(example_paths, inputs=inp)
            with gr.Column():
                out_overlay = gr.Image(label="Segmentation overlay")
                out_cam = gr.Image(label="Seg-Grad-CAM (why this class, here)")
                out_ent = gr.Image(label="Uncertainty (entropy)")
        out_table = gr.Dataframe(headers=["class", "object_count", "area_fraction"],
                                 label="Quantification")
        out_text = gr.Markdown()
        btn.click(predict, [inp, cls], [out_overlay, out_cam, out_ent, out_table, out_text])
    return demo


def main():
    ap = argparse.ArgumentParser(description="Launch the MedSeg-RAI demo.")
    ap.add_argument("--run", default="outputs/synthetic_demo")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()
    build_demo(args.run).launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
