"""Hugging Face Space entrypoint for the MedSeg-RAI demo.

Downloads the trained weights from the companion model repo at startup, then launches
the Gradio app. The 187 MB checkpoint lives in a model repo, not in the Space repo.
The placeholder below is filled in by deploy/deploy_hf.py at upload time.
"""
import os
from pathlib import Path

from huggingface_hub import hf_hub_download

from medseg.app.gradio_app import build_demo

MODEL_REPO = os.environ.get("MEDSEG_MODEL_REPO", "__MODEL_REPO__")


def _build():
    model_dir = Path("model")
    model_dir.mkdir(exist_ok=True)
    hf_hub_download(repo_id=MODEL_REPO, filename="best_model.pth", local_dir=str(model_dir))
    examples = sorted(str(p) for p in Path("examples").glob("*.png"))
    return build_demo(run=str(model_dir), example_paths=examples or None)


demo = _build()

if __name__ == "__main__":
    demo.launch()
