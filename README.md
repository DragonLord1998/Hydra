# Hydra — Character Developer

Hydra is a standalone character development tool powered by **Z-Image** generation and **Qwen-Image-Edit-2511** instruction-based image editing. Generate character images with your trained LoRAs, then iteratively refine them with natural language edits.

## Features

- **Z-Image Generation** — Two model variants:
  - **De-Turbo** (default) — 25 steps, CFG 2.5, cfg normalization enabled
  - **Foundation** — 50 steps, CFG 4.0, full quality baseline
- **Qwen Image Editing** — Instruction-based editing via Qwen-Image-Edit-2511. Generate an image, then switch to edit mode and describe changes in plain English.
- **LoRA Support** — Upload `.safetensors` LoRA files (trained with [Chimera](https://github.com/DragonLord1998/Chimera) or any Z-Image compatible trainer). Hot-swaps LoRAs without restarting.
- **Live Latent Previews** — See the image form in real-time during diffusion (every 2 steps via SSE streaming).
- **Model Loading UI** — Loading overlay with model name shown while downloading/initializing models.
- **VRAM Management** — Only one model loaded at a time via `enable_model_cpu_offload()`. Automatically swaps between Z-Image and Qwen pipelines to fit on 24GB+ GPUs.

## Workflow

```
1. Generate  →  Create a character image with Z-Image (green mode)
2. Edit      →  Refine the image with Qwen edits (blue mode)
3. Repeat    →  Keep editing until satisfied
```

Toggle between modes by clicking the glowing circle next to the prompt bar:
- **Green** = Generate mode (Z-Image)
- **Blue** = Edit mode (Qwen-Image-Edit)

## Quick Start

```bash
# Single command — installs deps and starts server
bash start.sh
# → http://0.0.0.0:7862
```

### Environment Variables (all optional)

```bash
export ZIMAGE_DETURBO_PATH="/workspace/models/z_image"    # Local De-Turbo model
export ZIMAGE_BASE_PATH="Tongyi-MAI/Z-Image"              # Foundation model (HF repo)
export QWEN_EDIT_MODEL="Qwen/Qwen-Image-Edit-2511"        # Qwen editor (HF repo)
export HF_TOKEN="hf_..."                                   # HuggingFace token (for gated models)
export HYDRA_API_KEY="your-secret-key"                     # Optional API key auth
```

## API

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Serves the UI |
| `/api/generate` | POST | Generate an image (`{prompt, model}`) |
| `/api/edit` | POST | Edit the current image (`{prompt}`) |
| `/api/upload-lora` | POST | Upload a LoRA (multipart: `lora` file + `trigger_word`) |
| `/api/status` | GET | Current server state (loaded model, LoRA, has image) |
| `/api/stream` | GET | SSE stream — latent previews, model status, errors |
| `/outputs/<filename>` | GET | Serve generated images |

## Architecture

```
hydra/
├── server.py          — Flask server, model lifecycle, SSE streaming, API routes
├── start.sh           — RunPod startup (installs deps from git main, launches server)
├── requirements.txt   — Python dependencies
├── static/
│   ├── index.html     — Single-page UI
│   ├── app.js         — Frontend logic (SSE, mode toggle, LoRA upload, previews)
│   └── style.css      — Dark theme with loading/preview overlays
├── loras/             — Uploaded LoRA files (auto-created)
└── outputs/           — Generated images (auto-created)
```

## Requirements

- Python 3.10+
- NVIDIA GPU with 24GB+ VRAM (RTX 3090, RTX 4090, etc.)
- CUDA toolkit
- `diffusers` from GitHub main branch (for ZImagePipeline & QwenImageEditPlusPipeline)

## Related

- **[Chimera](https://github.com/DragonLord1998/Chimera)** — Character LoRA Creator. Train Z-Image LoRAs from a single input image, then load them in Hydra.
