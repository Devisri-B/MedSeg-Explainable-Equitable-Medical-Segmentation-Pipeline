#!/usr/bin/env bash
# Generate every resume-ready figure + metrics file for a trained MedSeg-RAI run,
# then list what was produced.
#
# Usage:
#   bash scripts/make_report.sh [run_dir]
#   PY=python bash scripts/make_report.sh outputs/pannuke_improved
set -e

RUN="${1:-outputs/pannuke_improved}"
PY="${PY:-./.venv/bin/python}"

echo ">> [1/5] evaluation (+TTA): per-class metrics + confusion matrix"
$PY -m medseg.evaluate --run "$RUN" --split test --tta

echo ">> [2/5] fairness audit: Dice per tissue subgroup"
$PY -m medseg.fairness.audit --run "$RUN" --split test || echo "   (fairness skipped)"

echo ">> [3/5] quantification: counts, area fractions, tissue-degradation index"
$PY -m medseg.quantify --run "$RUN" --split test || echo "   (quantify skipped)"

echo ">> [4/5] explainability: Seg-Grad-CAM overlays"
$PY -m medseg.explain.seg_gradcam --run "$RUN" --num 8 || echo "   (grad-cam skipped)"

echo ">> [5/5] qualitative panel: input | ground truth | prediction"
$PY scripts/qualitative_examples.py --run "$RUN" --num 6 --tta || echo "   (qualitative skipped)"

echo
echo "=== figures produced under $RUN ==="
find "$RUN" -name '*.png' | sort
