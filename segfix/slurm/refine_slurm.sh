#!/bin/bash
# Run the trained baseline OCRNet + trained SegFix offset model on the
# validation set. Produces two directories of PNG predictions:
#   $OUTPUT/predictions/           refined (SegFix-applied)
#   $OUTPUT/predictions_baseline/  un-refined, for apples-to-apples eval
#
# Required env vars (set before sbatch / before bash):
#   SEG_CONFIG          path to the OCRNet baseline config (50pct)
#   SEG_CHECKPOINT      path to the trained OCRNet baseline .pth
#   OFFSET_CHECKPOINT   path to the trained SegFix .pth (segfix/configs/segfix_r18_ade20k_50pct.py)
# Optional:
#   OUTPUT              default work_dirs/segfix_refined_baseline_50pct
#   BOUNDARY_THRESH     default 0.5
#SBATCH -J segfix_refine
#SBATCH -c 8
#SBATCH -G 1
#SBATCH --mem=32G
#SBATCH -t 2:00:00
#SBATCH -o logs/segfix_refine_%j.out
#SBATCH -e logs/segfix_refine_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENCV_OPENCL_RUNTIME=disabled
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

mkdir -p logs

# Required (no defaults so we fail loudly if not provided)
: "${SEG_CONFIG:?must set SEG_CONFIG (e.g. configs/ocrnet/.../baseline_50pct.py)}"
: "${SEG_CHECKPOINT:?must set SEG_CHECKPOINT (path to trained OCRNet .pth)}"
: "${OFFSET_CHECKPOINT:?must set OFFSET_CHECKPOINT (path to trained SegFix .pth)}"

OFFSET_CONFIG="${OFFSET_CONFIG:-segfix/configs/segfix_r18_ade20k_50pct.py}"
OUTPUT="${OUTPUT:-work_dirs/segfix_refined_baseline_50pct}"
BOUNDARY_THRESH="${BOUNDARY_THRESH:-0.5}"

echo "REPO_ROOT         : $REPO_ROOT"
echo "Seg config        : $SEG_CONFIG"
echo "Seg checkpoint    : $SEG_CHECKPOINT"
echo "Offset config     : $OFFSET_CONFIG"
echo "Offset checkpoint : $OFFSET_CHECKPOINT"
echo "Output            : $OUTPUT"
echo "Boundary thresh   : $BOUNDARY_THRESH"

"$PYTHON_BIN" segfix/refine.py \
    --seg-config        "$SEG_CONFIG" \
    --seg-checkpoint    "$SEG_CHECKPOINT" \
    --offset-config     "$OFFSET_CONFIG" \
    --offset-checkpoint "$OFFSET_CHECKPOINT" \
    --output            "$OUTPUT" \
    --boundary-thresh   "$BOUNDARY_THRESH"
