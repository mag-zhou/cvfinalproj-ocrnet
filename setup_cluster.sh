#!/bin/bash
# Run this ONCE on the Engaging login node to set up your environment.
# Usage: bash setup_cluster.sh

set -e

module load miniforge

# Create conda env (skip if already exists)
if ! conda env list | grep -q "^mmseg "; then
    mamba create -n mmseg python=3.12 -y
fi

source activate mmseg

# Install PyTorch with CUDA 12.1 (compatible with A100/H200/L40S)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install OpenMMLab dependencies
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"

# Install mmsegmentation from this repo (editable)
pip install -v -e .

echo ""
echo "Setup complete. Verify with:"
echo "  python -c \"import torch; print(torch.__version__, torch.cuda.is_available())\""
echo "  python -c \"import mmseg; print(mmseg.__version__)\""
