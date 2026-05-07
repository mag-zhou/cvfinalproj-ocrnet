#!/bin/bash
#SBATCH -J mod3_50ext
#SBATCH -p mit_normal_gpu
#SBATCH -A mit_general
#SBATCH -c 8
#SBATCH -G 1
#SBATCH --mem=32G
#SBATCH -t 6:00:00
#SBATCH -o logs/train_mod3_50pct_ext_%j.out
#SBATCH -e logs/train_mod3_50pct_ext_%j.err

set -euo pipefail

REPO_ROOT="/orcd/scratch/orcd/003/janetguo/cvfinalproj-ocrnet"
cd "$REPO_ROOT"

CONFIG="configs/ocrnet/boundary/ocrnet_r50_mod3_modulated_ocr_50pct_ext.py"
WORK_DIR="work_dirs/ocrnet_r50_mod3_modulated_ocr_50pct_ext"

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

# First sbatch: no checkpoint in WORK_DIR yet, so we start from `load_from`
# (set in the config) -- a *fresh* PolyLR over 40k iters, weights warm-started
# from iter_40000.pth of the original run.
# If slurm preempts and re-submits, last_checkpoint exists in WORK_DIR and we
# resume normally (continuing iter counter, optimizer state, LR schedule).
if [ -f "$WORK_DIR/last_checkpoint" ]; then
    LAST="$(cat "$WORK_DIR/last_checkpoint")"
    echo "Resuming from: $LAST"
    RESUME_FLAG="--resume"
else
    echo "First run: warm-starting weights from load_from set in config."
    RESUME_FLAG=""
fi

"$PYTHON_BIN" tools/train.py "$CONFIG" \
    --work-dir "$WORK_DIR" \
    --launcher none \
    $RESUME_FLAG
