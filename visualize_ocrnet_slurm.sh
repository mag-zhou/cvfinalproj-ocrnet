#!/bin/bash
#SBATCH -J vis_ocrnet
#SBATCH -p mit_normal_gpu
#SBATCH -A mit_general
#SBATCH -c 4
#SBATCH -G 1
#SBATCH --mem=16G
#SBATCH -t 0:15:00
#SBATCH -o logs/visualize_ocrnet_%j.out
#SBATCH -e logs/visualize_ocrnet_%j.err

set -euo pipefail

module load miniforge
source /orcd/software/core/001/pkg/miniforge/25.11.0-0/etc/profile.d/conda.sh
conda activate mmseg

PYTHON_BIN="${CONDA_PREFIX}/bin/python"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

mkdir -p logs

OUTPUT_DIR="${OUTPUT_DIR:-report_figures_ocrnet}"
IMAGE_IDS="${IMAGE_IDS:-229 286 502 1519}"

echo "PWD          : $(pwd)"
echo "Python       : $PYTHON_BIN"
echo "CONDA_PREFIX : ${CONDA_PREFIX:-unset}"
echo "Output       : $OUTPUT_DIR"
echo "Image IDs    : $IMAGE_IDS"

"$PYTHON_BIN" -c "import mmcv, mmengine, mmseg; print('Preflight OK', mmcv.__version__, mmengine.__version__, mmseg.__version__)"

"$PYTHON_BIN" visualize_ocrnet.py \
    --image-ids $IMAGE_IDS \
    --output-dir "$OUTPUT_DIR"
