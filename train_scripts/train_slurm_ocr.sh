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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
cd "$REPO_ROOT"

# --- Edit these ---
CONFIG="configs/ocrnet/ocrnet_r50-d8_1xb8-40k_ade20k-512x512-20pct.py"
WORK_DIR="work_dirs/ocrnet_r50_ade20k_20pct"
# ------------------

module load miniforge
source /orcd/software/core/001/pkg/miniforge/25.11.0-0/etc/profile.d/conda.sh
conda activate mmseg

mkdir -p logs "$WORK_DIR"

python tools/train.py "$CONFIG" \
    --work-dir "$WORK_DIR" \
    --launcher none
