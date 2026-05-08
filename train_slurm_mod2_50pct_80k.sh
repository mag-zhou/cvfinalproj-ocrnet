#!/bin/bash
#SBATCH -J mod2_50_80k
#SBATCH -p mit_normal_gpu
#SBATCH -A mit_general
#SBATCH -c 8
#SBATCH -G 1
#SBATCH --mem=32G
#SBATCH -t 6:00:00
#SBATCH -o logs/train_mod2_50pct_80k_%j.out
#SBATCH -e logs/train_mod2_50pct_80k_%j.err

set -euo pipefail

REPO_ROOT="/orcd/scratch/orcd/003/janetguo/cvfinalproj-ocrnet"
cd "$REPO_ROOT"

CONFIG="configs/ocrnet/boundary/ocrnet_r50_mod2_weighted_ce_50pct_80k.py"
WORK_DIR="work_dirs/ocrnet_r50_mod2_weighted_ce_50pct_80k"

PYTHON_BIN="/home/janetguo/.conda/envs/mmseg/bin/python"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENCV_OPENCL_RUNTIME=disabled

mkdir -p logs "$WORK_DIR"

echo "PWD: $(pwd)"
echo "Python: $PYTHON_BIN"
"$PYTHON_BIN" -c "import mmengine, mmcv, mmseg; print('Preflight OK', mmengine.__version__, mmcv.__version__, mmseg.__version__)"

# Fresh run: starts from iter 0 the first time. If SLURM preempts and
# re-submits, last_checkpoint exists and we resume normally (continuing iter
# counter, optimizer state, and the single 80k PolyLR schedule).
if [ -f "$WORK_DIR/last_checkpoint" ]; then
    LAST="$(cat "$WORK_DIR/last_checkpoint")"
    echo "Resuming from: $LAST"
    RESUME_FLAG="--resume"
else
    echo "No checkpoint found - starting fresh from iter 0"
    RESUME_FLAG=""
fi

"$PYTHON_BIN" tools/train.py "$CONFIG" \
    --work-dir "$WORK_DIR" \
    --launcher none \
    $RESUME_FLAG
