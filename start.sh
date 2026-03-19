#!/bin/bash
set -e

echo "[Hydra] Installing dependencies..."

# Fix torch/torchvision compatibility (RunPod pods sometimes have mismatched versions)
CUDA_TAG=$(python3 -c "import torch; print('cu' + torch.version.cuda.replace('.','')[:3])" 2>/dev/null || echo "cu124")
echo "[Hydra] Detected CUDA: $CUDA_TAG"
pip install --quiet torchvision --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"

# Install diffusers from GitHub main (needed for ZImagePipeline & QwenImageEditPlusPipeline)
pip install --quiet git+https://github.com/huggingface/diffusers.git

# Install other deps
# --ignore-installed for blinker: RunPod's system blinker lacks RECORD file
pip install --quiet flask Pillow accelerate sentencepiece transformers --ignore-installed blinker

echo "[Hydra] Starting server..."
cd "$(dirname "$0")"
python3 server.py
