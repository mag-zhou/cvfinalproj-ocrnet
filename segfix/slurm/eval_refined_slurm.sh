#!/bin/bash
# Phase 5: standalone mIoU + boundary F-score on refined and baseline preds.
#SBATCH -J segfix_eval
#SBATCH -p mit_normal
#SBATCH -A mit_general
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 1:00:00
#SBATCH -o logs/segfix_eval_%j.out
#SBATCH -e logs/segfix_eval_%j.err

set -euo pipefail

REPO_ROOT="/orcd/scratch/orcd/003/janetguo/cvfinalproj-ocrnet"
cd "$REPO_ROOT"

PYTHON_BIN="/home/janetguo/.conda/envs/mmseg/bin/python"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

mkdir -p logs

GT_DIR="${GT_DIR:-data/ade/ADEChallengeData2016/annotations/validation}"
OUTPUT_ROOT="${OUTPUT_ROOT:-work_dirs/segfix_refined_baseline}"

echo "==== refined predictions ===="
"$PYTHON_BIN" segfix/eval_refined.py \
    --pred-dir "$OUTPUT_ROOT/predictions" \
    --gt-dir   "$GT_DIR"

if [ -d "$OUTPUT_ROOT/predictions_baseline" ]; then
    echo ""
    echo "==== baseline (un-refined) ===="
    "$PYTHON_BIN" segfix/eval_refined.py \
        --pred-dir "$OUTPUT_ROOT/predictions_baseline" \
        --gt-dir   "$GT_DIR"
fi
