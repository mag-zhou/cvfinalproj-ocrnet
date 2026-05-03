#!/bin/bash
#SBATCH -J mmseg_train
#SBATCH -p mit_normal_gpu
#SBATCH -A mit_general
#SBATCH -c 8
#SBATCH -G 1
#SBATCH --mem=32G
#SBATCH -t 6:00:00
#SBATCH -o logs/train_%j.out
#SBATCH -e logs/train_%j.err

# --- Edit these ---
CONFIG="configs/ocrnet/ocrnet_r50-d8_1xb8-40k_ade20k-512x512-20pct.py"
WORK_DIR="work_dirs/ocrnet_r50_ade20k_20pct"
# ------------------

module load miniforge
source activate mmseg

mkdir -p logs "$WORK_DIR"

python tools/train.py "$CONFIG" \
    --work-dir "$WORK_DIR" \
    --launcher none
