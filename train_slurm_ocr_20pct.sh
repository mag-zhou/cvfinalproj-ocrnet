#!/bin/bash
#SBATCH -J ocr_20pct
#SBATCH -p mit_normal_gpu
#SBATCH -A mit_general
#SBATCH -c 8
#SBATCH -G 1
#SBATCH --mem=32G
#SBATCH -t 6:00:00
#SBATCH -o logs/train_20pct_%j.out
#SBATCH -e logs/train_20pct_%j.err

set -euo pipefail

REPO_ROOT="/orcd/scratch/orcd/003/janetguo/cvfinalproj-ocrnet"
cd "$REPO_ROOT"

CONFIG="configs/ocrnet/ocrnet_r50-d8_1xb8-40k_ade20k-512x512-20pct.py"
WORK_DIR="work_dirs/ocrnet_r50_ade20k_20pct"

PYTHON_BIN="/home/janetguo/.conda/envs/mmseg/bin/python"

# Avoid libgomp/pthread explosions on shared nodes.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENCV_OPENCL_RUNTIME=disabled

mkdir -p logs "$WORK_DIR"

echo "PWD: $(pwd)"
echo "Python: $PYTHON_BIN"
"$PYTHON_BIN" -c "import mmengine, mmcv, mmseg; print('Preflight OK', mmengine.__version__, mmcv.__version__, mmseg.__version__)"

if [ -f "$WORK_DIR/last_checkpoint" ]; then
    LAST="$(cat "$WORK_DIR/last_checkpoint")"
    echo "Resuming from: $LAST"
    RESUME_FLAG="--resume"
else
    echo "No checkpoint found - starting fresh"
    RESUME_FLAG=""
fi

"$PYTHON_BIN" tools/train.py "$CONFIG" \
    --work-dir "$WORK_DIR" \
    --launcher none \
    $RESUME_FLAG
