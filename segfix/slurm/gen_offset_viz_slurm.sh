#!/bin/bash
# Phase 1 sanity-check: write magnitude + HSV-direction PNGs for ~10 images.
# Training does NOT use these files; this is purely for visual inspection.
#SBATCH -J segfix_viz
#SBATCH -p mit_normal
#SBATCH -A mit_general
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 0:30:00
#SBATCH -o logs/segfix_viz_%j.out
#SBATCH -e logs/segfix_viz_%j.err

set -euo pipefail

REPO_ROOT="/orcd/scratch/orcd/003/janetguo/cvfinalproj-ocrnet"
cd "$REPO_ROOT"

PYTHON_BIN="/home/janetguo/.conda/envs/mmseg/bin/python"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

mkdir -p logs

"$PYTHON_BIN" segfix/gen_offset_gt.py \
    --visualize \
    --max-images 10 \
    --workers 4
