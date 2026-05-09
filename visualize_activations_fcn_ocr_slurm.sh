#!/bin/bash
#SBATCH -J vis_act_fcn_ocr
#SBATCH -p mit_normal_gpu
#SBATCH -A mit_general
#SBATCH -c 4
#SBATCH -G 1
#SBATCH --mem=24G
#SBATCH -t 0:30:00
#SBATCH -o logs/visualize_activations_fcn_ocr_%j.out
#SBATCH -e logs/visualize_activations_fcn_ocr_%j.err

set -euo pipefail

module load miniforge
source /orcd/software/core/001/pkg/miniforge/25.11.0-0/etc/profile.d/conda.sh
conda activate mmseg

# conda activate sets CONDA_PREFIX but PATH may not get updated under SLURM —
# use the env's python directly to be safe.
PYTHON_BIN="${CONDA_PREFIX}/bin/python"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

mkdir -p logs

OUTPUT_DIR="${OUTPUT_DIR:-my_figures/activations_fcn_ocr}"
IMAGE_IDS="${IMAGE_IDS:-52 229 286 458 502 564 1000 1310 1519 1850}"

echo "PWD          : $(pwd)"
echo "Python       : $PYTHON_BIN"
echo "CONDA_PREFIX : ${CONDA_PREFIX:-unset}"
echo "Output       : $OUTPUT_DIR"
echo "Image IDs    : $IMAGE_IDS"

"$PYTHON_BIN" -c "import mmcv, mmengine, mmseg; print('Preflight OK', mmcv.__version__, mmengine.__version__, mmseg.__version__)"

"$PYTHON_BIN" visualize_activations_fcn_ocr.py \
    --image-ids $IMAGE_IDS \
    --output-dir "$OUTPUT_DIR"
