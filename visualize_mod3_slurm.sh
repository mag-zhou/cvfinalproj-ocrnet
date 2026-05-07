#!/bin/bash
#SBATCH -J mod3_viz
#SBATCH -p mit_normal_gpu
#SBATCH -A mit_general
#SBATCH -c 4
#SBATCH -G 1
#SBATCH --mem=16G
#SBATCH -t 0:30:00
#SBATCH -o logs/viz_mod3_%j.out
#SBATCH -e logs/viz_mod3_%j.err

set -euo pipefail

REPO_ROOT="/orcd/scratch/orcd/003/janetguo/cvfinalproj-ocrnet"
cd "$REPO_ROOT"

PYTHON_BIN="/home/janetguo/.conda/envs/mmseg/bin/python"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENCV_OPENCL_RUNTIME=disabled

mkdir -p logs my_figures/mod3_20vs50pct

echo "PWD: $(pwd)"
echo "Python: $PYTHON_BIN"
"$PYTHON_BIN" -c "import mmengine, mmcv, mmseg; print('Preflight OK', mmengine.__version__, mmcv.__version__, mmseg.__version__)"

"$PYTHON_BIN" visualize_mod3.py \
    --num-images 8 \
    --seed 42 \
    --output-dir my_figures/mod3_20vs50pct
