#!/bin/bash
# Run mIoU + boundary F1 on both the refined and un-refined predictions
# produced by refine_slurm.sh. Same metric implementation, same edge cases.
#
# Optional env vars:
#   GT_DIR        default data/ade/ADEChallengeData2016/annotations/validation
#   OUTPUT_ROOT   default work_dirs/segfix_refined_baseline_50pct
#SBATCH -J segfix_eval
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 1:00:00
#SBATCH -o logs/segfix_eval_%j.out
#SBATCH -e logs/segfix_eval_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

mkdir -p logs

GT_DIR="${GT_DIR:-data/ade/ADEChallengeData2016/annotations/validation}"
OUTPUT_ROOT="${OUTPUT_ROOT:-work_dirs/segfix_refined_baseline_50pct}"

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
