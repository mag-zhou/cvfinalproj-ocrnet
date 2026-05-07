#!/bin/bash
# Train the SegFix offset model on 50% ADE20K. ~2h on a single H200; longer
# on smaller cards (A100 / 3090 / 4090). 1 GPU is enough; the model is small.
#
# Portable: no hardcoded paths. Set REPO_ROOT and PYTHON_BIN as env vars to
# override the auto-detected defaults if needed. SLURM directives are kept
# but harmless if you `bash` this script directly instead of `sbatch`-ing it.
#SBATCH -J segfix_50pct
#SBATCH -p mit_normal_gpu
#SBATCH -A mit_general
#SBATCH -c 8
#SBATCH -G 1
#SBATCH --mem=32G
#SBATCH -t 6:00:00
#SBATCH -o logs/train_segfix_50pct_%j.out
#SBATCH -e logs/train_segfix_50pct_%j.err

set -euo pipefail

CONFIG="segfix/configs/segfix_r18_ade20k_50pct.py"
WORK_DIR="${WORK_DIR:-work_dirs/segfix_r18_ade20k_50pct}"

module load miniforge
source /orcd/software/core/001/pkg/miniforge/25.11.0-0/etc/profile.d/conda.sh
conda activate mmseg

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENCV_OPENCL_RUNTIME=disabled
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

mkdir -p logs "$WORK_DIR"

echo "PWD       : $(pwd)"
echo "Python    : $(which python)"
echo "Config    : $CONFIG"
echo "Work dir  : $WORK_DIR"

python -c "import mmengine, mmcv, mmseg, segfix; print('Preflight OK', mmengine.__version__, mmcv.__version__, mmseg.__version__)"

if [ -f "$WORK_DIR/last_checkpoint" ]; then
    LAST="$(cat "$WORK_DIR/last_checkpoint")"
    echo "Resuming from: $LAST"
    RESUME_FLAG="--resume"
else
    echo "No checkpoint found - starting fresh."
    RESUME_FLAG=""
fi

python tools/train.py "$CONFIG" \
    --work-dir "$WORK_DIR" \
    --launcher none \
    $RESUME_FLAG
