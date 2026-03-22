#!/bin/bash
set -e

echo "[Hydra] Installing dependencies..."

# Fix torch/torchvision compatibility (RunPod pods sometimes have mismatched versions)
CUDA_TAG=$(python3 -c "import torch; print('cu' + torch.version.cuda.replace('.','')[:3])" 2>/dev/null || echo "cu124")
echo "[Hydra] Detected CUDA: $CUDA_TAG — reinstalling torchvision..."
pip uninstall -y torchvision 2>/dev/null || true
pip install --quiet torchvision --no-deps --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"

# Install diffusers from GitHub main (needed for Flux2Pipeline)
pip install --quiet git+https://github.com/huggingface/diffusers.git

# Install other deps
pip install --quiet flask Pillow accelerate sentencepiece transformers numpy --ignore-installed blinker

# TensorRT-LLM visual_gen — NVFP4 quantization (Blackwell GPUs only)
echo "[Hydra] Installing TensorRT-LLM visual_gen for NVFP4..."
pip install --quiet tensorrt-llm 2>/dev/null || \
  echo "[Hydra] NOTE: TensorRT-LLM not available — Flux 2 will run in BF16 (still works, just no NVFP4 speedup)"

# SAM 3D Body — pose extraction
echo "[Hydra] Installing SAM 3D Body dependencies..."
pip install --quiet pytorch-lightning pyrender opencv-python yacs scikit-image einops timm \
  dill pandas rich hydra-core hydra-submitit-launcher hydra-colorlog pyrootutils networkx==3.2.1 \
  roma joblib cython xtcocotools loguru optree fvcore pycocotools huggingface_hub 2>/dev/null || true
pip install --quiet 'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' \
  --no-build-isolation --no-deps 2>/dev/null || true
echo "[Hydra] Installing SAM 3D Body..."
pip install --quiet git+https://github.com/facebookresearch/sam-3d-body.git 2>/dev/null || \
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

# SeedVR2 7B Sharp — image upscaling
SEEDVR2_DIR="/workspace/seedvr2"
if [ ! -d "$SEEDVR2_DIR/.git" ]; then
  echo "[Hydra] Cloning SeedVR2..."
  git clone --depth 1 https://github.com/ByteDance-Seed/SeedVR.git "$SEEDVR2_DIR" 2>/dev/null || \
    echo "[Hydra] WARNING: SeedVR2 clone failed — upscaling will be unavailable"
fi
if [ -d "$SEEDVR2_DIR" ]; then
  echo "[Hydra] Installing SeedVR2 dependencies..."
  pip install --quiet -r "$SEEDVR2_DIR/requirements.txt" 2>/dev/null || true
  pip install --quiet flash_attn --no-build-isolation 2>/dev/null || true

  echo "[Hydra] Downloading SeedVR2 7B Sharp checkpoint..."
  python3 -c "
from huggingface_hub import hf_hub_download
import os
token = os.environ.get('HF_TOKEN')
hf_hub_download('ByteDance-Seed/SeedVR2-7B', 'seedvr2_ema_7b_sharp.pth',
                local_dir='$SEEDVR2_DIR/ckpts', token=token)
hf_hub_download('ByteDance-Seed/SeedVR2-3B', 'ema_vae.pth',
                local_dir='$SEEDVR2_DIR/ckpts', token=token)
hf_hub_download('ByteDance-Seed/SeedVR2-3B', 'pos_emb.pt',
                local_dir='$SEEDVR2_DIR', token=token)
hf_hub_download('ByteDance-Seed/SeedVR2-3B', 'neg_emb.pt',
                local_dir='$SEEDVR2_DIR', token=token)
print('[Hydra] SeedVR2 7B Sharp checkpoint ready.')
" 2>/dev/null || echo "[Hydra] WARNING: SeedVR2 checkpoint download failed — will retry on first use"

  ln -sf seedvr2_ema_7b_sharp.pth "$SEEDVR2_DIR/ckpts/seedvr2_ema_7b.pth"
  cp "$(dirname "$0")/seedvr2_cli.py" "$SEEDVR2_DIR/inference_cli.py"
fi

echo "[Hydra] Starting server..."
cd "$(dirname "$0")"
python3 server.py
