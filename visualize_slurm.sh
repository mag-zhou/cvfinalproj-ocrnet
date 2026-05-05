#!/bin/bash
#SBATCH -J mmseg_viz
#SBATCH -p mit_normal_gpu
#SBATCH -A mit_general
#SBATCH -c 4
#SBATCH -G 1
#SBATCH --mem=16G
#SBATCH -t 0:30:00
#SBATCH -o logs/viz_%j.out
#SBATCH -e logs/viz_%j.err

module load miniforge
source /orcd/software/core/001/pkg/miniforge/25.11.0-0/etc/profile.d/conda.sh
conda activate mmseg

mkdir -p logs report_figures

python visualize_comparison.py \
    --fcn-config configs/fcn/fcn_r50-d8_1xb8-40k_ade20k-512x512-20pct.py \
    --fcn-ckpt   "work_dirs/fcn_r50_ade20k_20pct/best_mIoU_*.pth" \
    --ocr-config configs/ocrnet/ocrnet_r50-d8_1xb8-40k_ade20k-512x512-20pct.py \
    --ocr-ckpt   work_dirs/ocrnet_r50_ade20k_20pct/best_mIoU_iter_36000.pth \
    --num-images 12 \
    --seed 42 \
    --output-dir report_figures
