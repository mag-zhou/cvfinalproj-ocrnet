#!/bin/bash
# Phase 4: run baseline OCRNet + SegFix offset model, write refined preds.
#SBATCH -J segfix_refine
#SBATCH -p mit_normal_gpu
#SBATCH -A mit_general
#SBATCH -c 8
#SBATCH -G 1
#SBATCH --mem=32G
#SBATCH -t 2:00:00
#SBATCH -o logs/segfix_refine_%j.out
#SBATCH -e logs/segfix_refine_%j.err

set -euo pipefail

REPO_ROOT="/orcd/scratch/orcd/003/janetguo/cvfinalproj-ocrnet"
cd "$REPO_ROOT"

PYTHON_BIN="/home/janetguo/.conda/envs/mmseg/bin/python"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENCV_OPENCL_RUNTIME=disabled
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

mkdir -p logs

# Required: edit the four paths below to point at your trained checkpoints.
SEG_CONFIG="${SEG_CONFIG:-configs/ocrnet/boundary/ocrnet_r50_baseline_20pct.py}"
SEG_CHECKPOINT="${SEG_CHECKPOINT:-work_dirs/ocrnet_r50_ade20k_20pct/iter_40000.pth}"
OFFSET_CONFIG="${OFFSET_CONFIG:-segfix/configs/segfix_r18_ade20k_20pct.py}"
OFFSET_CHECKPOINT="${OFFSET_CHECKPOINT:-work_dirs/segfix_r18_ade20k_20pct/iter_20000.pth}"
OUTPUT="${OUTPUT:-work_dirs/segfix_refined_baseline}"
BOUNDARY_THRESH="${BOUNDARY_THRESH:-0.5}"

echo "Seg checkpoint    : $SEG_CHECKPOINT"
echo "Offset checkpoint : $OFFSET_CHECKPOINT"
echo "Output            : $OUTPUT"
echo "Boundary thresh   : $BOUNDARY_THRESH"

"$PYTHON_BIN" segfix/refine.py \
    --seg-config        "$SEG_CONFIG" \
    --seg-checkpoint    "$SEG_CHECKPOINT" \
    --offset-config     "$OFFSET_CONFIG" \
    --offset-checkpoint "$OFFSET_CHECKPOINT" \
    --output            "$OUTPUT" \
    --boundary-thresh   "$BOUNDARY_THRESH"
