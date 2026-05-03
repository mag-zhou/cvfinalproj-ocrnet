#!/bin/bash
#SBATCH -J mmseg_train
#SBATCH -p mit_normal_gpu
#SBATCH -c 8
#SBATCH -G 1
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -o logs/train_%j.out
#SBATCH -e logs/train_%j.err

# --- Edit these ---
CONFIG="configs/pspnet/pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py"
WORK_DIR="work_dirs/pspnet_r50_cityscapes"
# ------------------

module load miniforge
source activate mmseg

mkdir -p logs "$WORK_DIR"

python tools/train.py "$CONFIG" \
    --work-dir "$WORK_DIR" \
    --launcher none
