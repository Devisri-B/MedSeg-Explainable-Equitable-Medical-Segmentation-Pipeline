"""Publish MedSeg-RAI to Hugging Face.

Creates two repos under your account:
  * a model repo that holds the 187 MB weights (best_model.pth), and
  * a Gradio Space that runs the demo and downloads those weights at startup.

Usage (you are already logged in via `hf auth login`):

    python deploy/deploy_hf.py \
        --model-repo <user>/medseg-rai-pannuke \
        --space      <user>/medseg-rai \
        --weights    outputs/pannuke_resnet50/best_model.pth \
        --data-root  data/pannuke

After a code change, refresh only the Space (no re-upload of the weights):

    python deploy/deploy_hf.py --model-repo <user>/medseg-rai-pannuke \
        --space <user>/medseg-rai --skip-weights
"""
from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import numpy as np
from huggingface_hub import HfApi, whoami

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def _write_examples(data_root: str, dst: Path, n: int = 3) -> None:
    from PIL import Image

    from medseg.data import pannuke

    dst.mkdir(parents=True, exist_ok=True)
    if not pannuke.is_available(data_root):
        print("[deploy] local PanNuke not found; the Space will run without bundled examples")
        return
    imgs, _, _ = pannuke.load_fold(data_root, 3)
    for i, j in enumerate(np.linspace(0, len(imgs) - 1, n).astype(int)):
        Image.fromarray(imgs[int(j)]).save(dst / f"example_{i}.png")
    print(f"[deploy] added {n} real example patches")


def main() -> None:
    ap = argparse.ArgumentParser(description="Deploy MedSeg-RAI to Hugging Face.")
    ap.add_argument("--model-repo", required=True, help="e.g. <user>/medseg-rai-pannuke")
    ap.add_argument("--space", required=True, help="e.g. <user>/medseg-rai")
    ap.add_argument("--weights", default="outputs/pannuke_resnet50/best_model.pth")
    ap.add_argument("--data-root", default="data/pannuke")
    ap.add_argument("--private", action="store_true", help="make both repos private")
    ap.add_argument("--skip-weights", action="store_true",
                    help="only update the Space code; leave the model repo untouched")
    args = ap.parse_args()

    api = HfApi()
    print("[deploy] logged in as", whoami()["name"])

    # 1) Model repo with the weights + a model card.
    if args.skip_weights:
        print("[deploy] --skip-weights: leaving the model repo untouched")
    else:
        if not Path(args.weights).exists():
            raise SystemExit(f"weights not found: {args.weights}")
        api.create_repo(args.model_repo, repo_type="model", exist_ok=True, private=args.private)
        print("[deploy] uploading weights (187 MB, may take a minute) ...")
        api.upload_file(path_or_fileobj=args.weights, path_in_repo="best_model.pth",
                        repo_id=args.model_repo, repo_type="model")
        api.upload_file(path_or_fileobj=str(HERE / "MODEL_README.md"), path_in_repo="README.md",
                        repo_id=args.model_repo, repo_type="model")
        print("[deploy] weights -> https://huggingface.co/" + args.model_repo)

    # 2) Space with the app and the medseg package.
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        shutil.copytree(ROOT / "medseg", tmp / "medseg",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        (tmp / "app.py").write_text(
            (HERE / "app.py").read_text().replace("__MODEL_REPO__", args.model_repo)
        )
        shutil.copy(HERE / "requirements.txt", tmp / "requirements.txt")
        (tmp / "README.md").write_text(
            (HERE / "README.md").read_text().replace("__MODEL_REPO__", args.model_repo)
        )
        _write_examples(args.data_root, tmp / "examples")

        api.create_repo(args.space, repo_type="space", space_sdk="gradio",
                        exist_ok=True, private=args.private)
        print("[deploy] uploading Space files ...")
        api.upload_folder(folder_path=str(tmp), repo_id=args.space, repo_type="space")

    print("[deploy] space -> https://huggingface.co/spaces/" + args.space)
    print("[deploy] the Space will rebuild for a few minutes, then go live.")


if __name__ == "__main__":
    main()
