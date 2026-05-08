#!/bin/bash
#SBATCH -J mmseg_50pct
#SBATCH -p mit_normal_gpu
#SBATCH -A mit_general
#SBATCH -c 8
#SBATCH -G 1
#SBATCH --mem=32G
#SBATCH -t 6:00:00
#SBATCH -o logs/train_%j.out
#SBATCH -e logs/train_%j.err

CONFIG="configs/ocrnet/ocrnet_r50-d8_1xb8-40k_ade20k-512x512-50pct.py"
WORK_DIR="work_dirs/ocrnet_r50_ade20k_50pct"

module load miniforge
source /orcd/software/core/001/pkg/miniforge/25.11.0-0/etc/profile.d/conda.sh
conda activate mmseg

mkdir -p logs "$WORK_DIR"

# Auto-resume: if a checkpoint exists from a previous run, continue from it
if [ -f "$WORK_DIR/last_checkpoint" ]; then
    LAST=$(cat "$WORK_DIR/last_checkpoint")
    echo "Resuming from: $LAST"
    RESUME_FLAG="--resume"
else
    echo "No checkpoint found — starting fresh"
    RESUME_FLAG=""
fi

python tools/train.py "$CONFIG" \
    --work-dir "$WORK_DIR" \
    --launcher none \
    $RESUME_FLAG
