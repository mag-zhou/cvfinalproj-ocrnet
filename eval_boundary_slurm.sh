#!/bin/bash
#SBATCH -J eval_bf
#SBATCH -p mit_normal_gpu
#SBATCH -A mit_general
#SBATCH -c 4
#SBATCH -G 1
#SBATCH --mem=24G
#SBATCH -t 1:00:00
#SBATCH -o logs/eval_bf_%j.out
#SBATCH -e logs/eval_bf_%j.err

set -euo pipefail

module load miniforge
source /orcd/software/core/001/pkg/miniforge/25.11.0-0/etc/profile.d/conda.sh
conda activate mmseg

PYTHON_BIN="${CONDA_PREFIX}/bin/python"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

mkdir -p logs

FCN_CONFIG="configs/fcn/fcn_r50-d8_eval_bf_50pct_80k.py"
FCN_CKPT="${FCN_CKPT:-work_dirs/fcn_r50_ade20k_50pct_ext/iter_40000.pth}"
FCN_WORK_DIR="${FCN_WORK_DIR:-work_dirs/eval_bf_fcn_50pct_80k}"

OCR_CONFIG="configs/ocrnet/ocrnet_r50-d8_eval_bf_50pct_80k.py"
OCR_CKPT="${OCR_CKPT:-work_dirs/ocrnet_r50_ade20k_50pct_ext/iter_40000.pth}"
OCR_WORK_DIR="${OCR_WORK_DIR:-work_dirs/eval_bf_ocrnet_50pct_80k}"

echo "PWD          : $(pwd)"
echo "Python       : $PYTHON_BIN"
echo "CONDA_PREFIX : ${CONDA_PREFIX:-unset}"
echo "FCN ckpt     : $FCN_CKPT"
echo "OCR ckpt     : $OCR_CKPT"

"$PYTHON_BIN" -c "import mmcv, mmengine, mmseg; print('Preflight OK', mmcv.__version__, mmengine.__version__, mmseg.__version__)"

echo ""
echo "============================================================"
echo "  FCN  (ResNet-50, 50% / 80k)"
echo "============================================================"
"$PYTHON_BIN" tools/test.py "$FCN_CONFIG" "$FCN_CKPT" --work-dir "$FCN_WORK_DIR"

echo ""
echo "============================================================"
echo "  OCRNet  (ResNet-50, 50% / 80k)"
echo "============================================================"
"$PYTHON_BIN" tools/test.py "$OCR_CONFIG" "$OCR_CKPT" --work-dir "$OCR_WORK_DIR"

echo ""
echo "Done. Look for 'mIoU' and 'mBoundaryF1_3 / _5 / _9' lines above."
