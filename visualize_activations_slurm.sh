#!/bin/bash
#SBATCH -J act_viz
#SBATCH -p mit_normal_gpu
#SBATCH -A mit_general
#SBATCH -c 4
#SBATCH -G 1
#SBATCH --mem=24G
#SBATCH -t 0:30:00
#SBATCH -o logs/viz_activations_%j.out
#SBATCH -e logs/viz_activations_%j.err

set -euo pipefail

REPO_ROOT="/orcd/scratch/orcd/003/janetguo/cvfinalproj-ocrnet"
cd "$REPO_ROOT"

PYTHON_BIN="/home/janetguo/.conda/envs/mmseg/bin/python"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENCV_OPENCL_RUNTIME=disabled

mkdir -p logs my_figures/activations

echo "PWD: $(pwd)"
echo "Python: $PYTHON_BIN"
"$PYTHON_BIN" -c "import mmengine, mmcv, mmseg; print('Preflight OK', mmengine.__version__, mmcv.__version__, mmseg.__version__)"

# Default to ADE_val_00000229 (the one currently being inspected) but allow
# overriding via the IMAGE env var for sbatch --export. Tweak --reduce / --alpha
# via REDUCE / ALPHA env vars (l2 default).
IMAGE="${IMAGE:-ADE_val_00000229}"
REDUCE="${REDUCE:-l2}"
ALPHA="${ALPHA:-0.55}"

EXTRA_ARGS=()
if [[ "${SHARED_ROW_NORM:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--shared-row-norm)
fi

"$PYTHON_BIN" visualize_activations.py \
    --image "$IMAGE" \
    --reduce "$REDUCE" \
    --alpha "$ALPHA" \
    --output-dir my_figures/activations \
    "${EXTRA_ARGS[@]}"
