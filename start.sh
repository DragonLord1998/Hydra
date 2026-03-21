#!/bin/bash
set -e

echo "[Hydra] Installing dependencies..."

# Fix torch/torchvision compatibility (RunPod pods sometimes have mismatched versions)
# Must uninstall first to remove stale .so files, then reinstall from PyTorch CUDA index
CUDA_TAG=$(python3 -c "import torch; print('cu' + torch.version.cuda.replace('.','')[:3])" 2>/dev/null || echo "cu124")
echo "[Hydra] Detected CUDA: $CUDA_TAG — reinstalling torchvision..."
pip uninstall -y torchvision 2>/dev/null || true
pip install --quiet torchvision --no-deps --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"

# Install diffusers from GitHub main (needed for ZImagePipeline & QwenImageEditPlusPipeline)
pip install --quiet git+https://github.com/huggingface/diffusers.git

# Install other deps
# --ignore-installed for blinker: RunPod's system blinker lacks RECORD file
pip install --quiet flask Pillow accelerate sentencepiece transformers --ignore-installed blinker

# SAM 3D Body — pose extraction
echo "[Hydra] Installing SAM 3D Body..."
pip install --quiet sam-3d-body 2>/dev/null || \
  pip install --quiet git+https://github.com/facebookresearch/sam3d.git 2>/dev/null || \
  echo "[Hydra] WARNING: SAM 3D Body install failed — pose extraction will be unavailable"

# Pre-download SAM 3D Body checkpoint
if python3 -c "import sam_3d_body" 2>/dev/null; then
  echo "[Hydra] Downloading SAM 3D Body checkpoint..."
  python3 -c "
from huggingface_hub import snapshot_download
import os
snapshot_download('facebook/sam-3d-body-dinov3', local_dir='checkpoints/sam-3d-body-dinov3', token=os.environ.get('HF_TOKEN'))
print('[Hydra] SAM 3D Body checkpoint ready.')
" 2>/dev/null || echo "[Hydra] WARNING: SAM 3D checkpoint download failed — will retry on first use"
fi

# Pre-download AnyPose LoRAs for Qwen pose transfer
echo "[Hydra] Downloading AnyPose LoRAs..."
python3 -c "
from huggingface_hub import hf_hub_download
import os
token = os.environ.get('HF_TOKEN')
hf_hub_download('lightx2v/Qwen-Image-Edit-2511-Lightning', 'Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors', token=token)
hf_hub_download('lilylilith/AnyPose', '2511-AnyPose-base-000006250.safetensors', token=token)
hf_hub_download('lilylilith/AnyPose', '2511-AnyPose-helper-00006000.safetensors', token=token)
print('[Hydra] AnyPose LoRAs ready.')
" 2>/dev/null || echo "[Hydra] WARNING: AnyPose LoRA download failed — will retry on first use"

echo "[Hydra] Starting server..."
cd "$(dirname "$0")"
python3 server.py
