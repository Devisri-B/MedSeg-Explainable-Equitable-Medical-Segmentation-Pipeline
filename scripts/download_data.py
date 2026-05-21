#!/usr/bin/env python
"""Download the PanNuke dataset.

    python scripts/download_data.py --root data/pannuke
    python scripts/download_data.py --root data/pannuke --repo-id <hf-dataset-id>

Tries Hugging Face Hub mirrors, otherwise prints the official academic link.
"""
import argparse

from medseg.data import pannuke


def main() -> None:
    ap = argparse.ArgumentParser(description="Download PanNuke into a local folder.")
    ap.add_argument("--root", default="data/pannuke")
    ap.add_argument("--repo-id", default=None, help="explicit Hugging Face dataset id to pull")
    ap.add_argument("--max-per-fold", type=int, default=None,
                    help="cap images per fold (handy for a quick first run)")
    args = ap.parse_args()

    pannuke.download(args.root, args.repo_id, max_per_fold=args.max_per_fold)
    if pannuke.is_available(args.root):
        images, semantic, types = pannuke.load_fold(args.root, 1)
        tissues = sorted(set(types))
        print(f"\n[ok] Fold 1: {len(images)} images of shape {images.shape[1:]} | "
              f"{len(tissues)} tissue types: {tissues[:6]}{' ...' if len(tissues) > 6 else ''}")
        print(f"[ok] semantic label range: {semantic.min()}..{semantic.max()} "
              f"(0=Background ... 5=Epithelial)")
    else:
        print("\n[!] Data not found after download step — see the instructions above.")


if __name__ == "__main__":
    main()
