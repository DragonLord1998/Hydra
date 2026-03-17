# Hydra — Character Developer

Hydra is a standalone character development tool powered by **Z-Image** generation and **Qwen-Image-Edit-2511** instruction-based image editing. Generate character images with your trained LoRAs, then iteratively refine them with natural language edits.

## Features

- **Z-Image Generation** — Two model variants:
  - **De-Turbo** (default) — 25 steps, CFG 2.5, cfg normalization enabled
  - **Foundation** — 50 steps, CFG 4.0, full quality baseline
- **Qwen Image Editing** — Instruction-based editing via Qwen-Image-Edit-2511. Generate an image, then switch to edit mode and describe changes in plain English.
- **LoRA Support** — Upload `.safetensors` LoRA files (trained with [Chimera](https://github.com/DragonLord1998/Chimera) or any Z-Image compatible trainer). Hot-swaps LoRAs without restarting.
- **VRAM Management** — Only one model loaded at a time. Automatically swaps between Z-Image and Qwen pipelines to fit on 24GB+ GPUs.

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
# Install dependencies
pip install flask torch diffusers transformers pillow accelerate sentencepiece

# Set model paths (optional — defaults shown)
export ZIMAGE_DETURBO_PATH="/workspace/models/z_image"
export ZIMAGE_BASE_PATH="Tongyi-MAI/Z-Image"
export QWEN_EDIT_MODEL="Qwen/Qwen-Image-Edit-2511"

# Start Hydra
python server.py
# → http://0.0.0.0:7861
```

## API

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Serves the UI |
| `/api/generate` | POST | Generate an image (`{prompt, model}`) |
| `/api/edit` | POST | Edit the current image (`{prompt}`) |
| `/api/upload-lora` | POST | Upload a LoRA (multipart: `lora` file + `trigger_word`) |
| `/api/status` | GET | Current server state (loaded model, LoRA, has image) |
| `/outputs/<filename>` | GET | Serve generated images |

## Architecture

```
hydra/
├── server.py          — Flask server, model lifecycle, API routes
├── static/
│   ├── index.html     — Single-page UI
│   ├── app.js         — Frontend logic (mode toggle, LoRA upload, prompt submission)
│   └── style.css      — Dark theme
├── loras/             — Uploaded LoRA files (auto-created)
└── outputs/           — Generated images (auto-created)
```

## Requirements

- Python 3.10+
- NVIDIA GPU with 24GB+ VRAM (RTX 3090, RTX 4090, etc.)
- CUDA toolkit
- `diffusers` with Z-Image and Qwen-Image-Edit support (latest from main branch)

## Related

- **[Chimera](https://github.com/DragonLord1998/Chimera)** — Character LoRA Creator. Train Z-Image LoRAs from a single input image, then load them in Hydra.
