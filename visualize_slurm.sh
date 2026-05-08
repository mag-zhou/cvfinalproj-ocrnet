#!/bin/bash
#SBATCH -J vis_50pct_80k
#SBATCH -p mit_normal_gpu
#SBATCH -A mit_general
#SBATCH -c 4
#SBATCH -G 1
#SBATCH --mem=16G
#SBATCH -t 0:30:00
#SBATCH -o logs/visualize_%j.out
#SBATCH -e logs/visualize_%j.err

set -euo pipefail

module load miniforge
source /orcd/software/core/001/pkg/miniforge/25.11.0-0/etc/profile.d/conda.sh
conda activate mmseg

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

mkdir -p logs

OUTPUT_DIR="${OUTPUT_DIR:-report_figures_50pct_80k}"
NUM_IMAGES="${NUM_IMAGES:-40}"
SEED="${SEED:-42}"

echo "===== ENV DIAGNOSTICS ====="
echo "PWD          : $(pwd)"
echo "HOSTNAME     : $(hostname)"
echo "Python       : $(which python)"
echo "CONDA_PREFIX : ${CONDA_PREFIX:-unset}"
echo "Has mmcv?    :"
python -c "import mmcv; print('  mmcv', mmcv.__version__, 'at', mmcv.__file__)" || echo "  mmcv MISSING"
echo "Has mmengine?:"
python -c "import mmengine; print('  mmengine', mmengine.__version__)" || echo "  mmengine MISSING"
echo "Has mmseg?   :"
python -c "import mmseg; print('  mmseg', mmseg.__version__, 'at', mmseg.__file__)" || echo "  mmseg MISSING"
echo "==========================="
echo "Output    : $OUTPUT_DIR"
echo "Num imgs  : $NUM_IMAGES"
echo "Seed      : $SEED"

python visualize_comparison.py \
    --num-images "$NUM_IMAGES" \
    --seed "$SEED" \
    --output-dir "$OUTPUT_DIR"
