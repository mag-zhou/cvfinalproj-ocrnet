#!/bin/bash
# Train the SegFix offset model on 50% ADE20K. ~2h on a single H200; longer
# on smaller cards (A100 / 3090 / 4090). 1 GPU is enough; the model is small.
#
# Portable: no hardcoded paths. Set REPO_ROOT and PYTHON_BIN as env vars to
# override the auto-detected defaults if needed. SLURM directives are kept
# but harmless if you `bash` this script directly instead of `sbatch`-ing it.
#SBATCH -J segfix_50pct
#SBATCH -c 8
#SBATCH -G 1
#SBATCH --mem=32G
#SBATCH -t 6:00:00
#SBATCH -o logs/train_segfix_50pct_%j.out
#SBATCH -e logs/train_segfix_50pct_%j.err

set -euo pipefail

# Resolve repo root from this script's location: segfix/slurm/X.sh -> ../..
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$REPO_ROOT"

CONFIG="segfix/configs/segfix_r18_ade20k_50pct.py"
WORK_DIR="${WORK_DIR:-work_dirs/segfix_r18_ade20k_50pct}"

PYTHON_BIN="${PYTHON_BIN:-python}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENCV_OPENCL_RUNTIME=disabled
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

mkdir -p logs "$WORK_DIR"

echo "REPO_ROOT : $REPO_ROOT"
echo "Python    : $PYTHON_BIN"
echo "Config    : $CONFIG"
echo "Work dir  : $WORK_DIR"

"$PYTHON_BIN" -c "import mmengine, mmcv, mmseg, segfix; print('Preflight OK', mmengine.__version__, mmcv.__version__, mmseg.__version__)"

if [ -f "$WORK_DIR/last_checkpoint" ]; then
    LAST="$(cat "$WORK_DIR/last_checkpoint")"
    echo "Resuming from: $LAST"
    RESUME_FLAG="--resume"
else
    echo "No checkpoint found - starting fresh."
    RESUME_FLAG=""
fi

"$PYTHON_BIN" tools/train.py "$CONFIG" \
    --work-dir "$WORK_DIR" \
    --launcher none \
    $RESUME_FLAG
