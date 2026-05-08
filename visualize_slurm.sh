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

# conda activate sets CONDA_PREFIX but PATH may not get updated under SLURM —
# use the env's python directly to be safe.
PYTHON_BIN="${CONDA_PREFIX}/bin/python"

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

mkdir -p logs

OUTPUT_DIR="${OUTPUT_DIR:-report_figures_50pct_80k}"
NUM_IMAGES="${NUM_IMAGES:-40}"
SEED="${SEED:-42}"

echo "PWD          : $(pwd)"
echo "HOSTNAME     : $(hostname)"
echo "Python       : $PYTHON_BIN"
echo "CONDA_PREFIX : ${CONDA_PREFIX:-unset}"
echo "Output       : $OUTPUT_DIR"
echo "Num imgs     : $NUM_IMAGES"
echo "Seed         : $SEED"

"$PYTHON_BIN" -c "import mmcv, mmengine, mmseg; print('Preflight OK', mmcv.__version__, mmengine.__version__, mmseg.__version__)"

"$PYTHON_BIN" visualize_comparison.py \
    --num-images "$NUM_IMAGES" \
    --seed "$SEED" \
    --output-dir "$OUTPUT_DIR"
