#!/bin/bash
# Run mIoU + boundary F1 on both the refined and un-refined predictions
# produced by refine_slurm.sh. Same metric implementation, same edge cases.
#
# Optional env vars:
#   GT_DIR        default data/ade/ADEChallengeData2016/annotations/validation
#   OUTPUT_ROOT   default work_dirs/segfix_refined_baseline_50pct
#SBATCH -J segfix_eval
#SBATCH -p mit_normal_gpu
#SBATCH -A mit_general
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 1:00:00
#SBATCH -o logs/segfix_eval_%j.out
#SBATCH -e logs/segfix_eval_%j.err

set -euo pipefail

module load miniforge
source /orcd/software/core/001/pkg/miniforge/25.11.0-0/etc/profile.d/conda.sh
conda activate mmseg

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

mkdir -p logs

GT_DIR="${GT_DIR:-data/ade/ADEChallengeData2016/annotations/validation}"
OUTPUT_ROOT="${OUTPUT_ROOT:-work_dirs/segfix_refined_baseline_50pct}"

echo "==== refined predictions ===="
python segfix/eval_refined.py \
    --pred-dir "$OUTPUT_ROOT/predictions" \
    --gt-dir   "$GT_DIR"

if [ -d "$OUTPUT_ROOT/predictions_baseline" ]; then
    echo ""
    echo "==== baseline (un-refined) ===="
    python segfix/eval_refined.py \
        --pred-dir "$OUTPUT_ROOT/predictions_baseline" \
        --gt-dir   "$GT_DIR"
fi
