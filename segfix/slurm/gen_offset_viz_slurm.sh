#!/bin/bash
# OPTIONAL sanity-check: write magnitude + HSV-direction PNGs for ~10
# training labels. Training does NOT use these files; this is purely for
# visual inspection ("are the offsets pointing inward and small in
# interiors?").
#SBATCH -J segfix_viz
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 0:30:00
#SBATCH -o logs/segfix_viz_%j.out
#SBATCH -e logs/segfix_viz_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

mkdir -p logs

"$PYTHON_BIN" segfix/gen_offset_gt.py \
    --visualize \
    --max-images 10 \
    --workers 4
