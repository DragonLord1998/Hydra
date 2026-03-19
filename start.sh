#!/bin/bash
set -e

echo "[Hydra] Installing dependencies..."

# Install diffusers from GitHub main (needed for ZImagePipeline & QwenImageEditPlusPipeline)
pip install --quiet git+https://github.com/huggingface/diffusers.git

# Install other deps (torch/torchvision are pre-installed on RunPod with CUDA — don't reinstall)
# --ignore-installed for blinker: RunPod's system blinker lacks RECORD file
pip install --quiet flask Pillow accelerate sentencepiece transformers --ignore-installed blinker

echo "[Hydra] Starting server..."
cd "$(dirname "$0")"
python3 server.py
