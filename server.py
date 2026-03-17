"""
Hydra — Character Developer

Z-Image generation (De-Turbo or Foundation) with LoRA support +
Qwen-Image-Edit-2511 instruction-based image editing.
"""

import gc
import logging
import os
import threading
import time
import uuid
from pathlib import Path

import torch
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image
from werkzeug.utils import secure_filename

logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static")

# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
LORA_DIR = BASE_DIR / "loras"
OUTPUT_DIR = BASE_DIR / "outputs"
LORA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

ZIMAGE_DETURBO_PATH = os.environ.get("ZIMAGE_DETURBO_PATH", "/workspace/models/z_image")
ZIMAGE_BASE_PATH = os.environ.get("ZIMAGE_BASE_PATH", "Tongyi-MAI/Z-Image")
QWEN_EDIT_MODEL = os.environ.get("QWEN_EDIT_MODEL", "Qwen/Qwen-Image-Edit-2511")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model presets: (steps, guidance_scale, cfg_normalization)
ZIMAGE_PRESETS = {
    "deturbo": {"steps": 25, "cfg": 2.5, "cfg_norm": True},
    "base":    {"steps": 50, "cfg": 4.0, "cfg_norm": False},
}

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_zimage_pipe = None
_zimage_variant: str | None = None       # "deturbo" or "base"
_qwen_pipe = None
_current_lora: dict | None = None        # {"path", "name", "trigger"}
_current_image_path: str | None = None   # last generated/edited image


# ---------------------------------------------------------------------------
# Model lifecycle
# ---------------------------------------------------------------------------

def _unload_zimage():
    global _zimage_pipe, _zimage_variant
    if _zimage_pipe is not None:
        del _zimage_pipe
        _zimage_pipe = None
        _zimage_variant = None
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("[Hydra] Z-Image pipeline unloaded.")


def _unload_qwen():
    global _qwen_pipe
    if _qwen_pipe is not None:
        del _qwen_pipe
        _qwen_pipe = None
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("[Hydra] Qwen-Image-Edit pipeline unloaded.")


def _load_zimage(variant: str = "deturbo"):
    global _zimage_pipe, _zimage_variant

    # If already loaded with the right variant, skip
    if _zimage_pipe is not None and _zimage_variant == variant:
        return

    # Need a different variant — unload current
    _unload_zimage()
    _unload_qwen()

    from diffusers import ZImagePipeline

    model_path = ZIMAGE_DETURBO_PATH if variant == "deturbo" else ZIMAGE_BASE_PATH
    label = "De-Turbo" if variant == "deturbo" else "Foundation"

    logger.info("[Hydra] Loading Z-Image %s from %s ...", label, model_path)
    print(f"[Hydra] Loading Z-Image {label} from {model_path} ...")

    _zimage_pipe = ZImagePipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
    )
    _zimage_pipe.to(DEVICE)
    _zimage_variant = variant

    if _current_lora:
        logger.info("[Hydra] Loading LoRA: %s", _current_lora["name"])
        _zimage_pipe.load_lora_weights(_current_lora["path"])

    print(f"[Hydra] Z-Image {label} pipeline ready.")


def _load_qwen():
    global _qwen_pipe
    if _qwen_pipe is not None:
        return

    _unload_zimage()

    from diffusers import QwenImageEditPlusPipeline

    logger.info("[Hydra] Loading Qwen-Image-Edit from %s ...", QWEN_EDIT_MODEL)
    print(f"[Hydra] Loading Qwen-Image-Edit from {QWEN_EDIT_MODEL} ...")

    _qwen_pipe = QwenImageEditPlusPipeline.from_pretrained(
        QWEN_EDIT_MODEL,
        torch_dtype=torch.bfloat16,
    )
    _qwen_pipe.to(DEVICE)

    print("[Hydra] Qwen-Image-Edit pipeline ready.")


# ---------------------------------------------------------------------------
# Routes — static
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/outputs/<filename>")
def serve_output(filename):
    return send_from_directory(str(OUTPUT_DIR), filename)


# ---------------------------------------------------------------------------
# Routes — API
# ---------------------------------------------------------------------------

@app.route("/api/upload-lora", methods=["POST"])
def upload_lora():
    global _current_lora

    file = request.files.get("lora")
    trigger = (request.form.get("trigger_word") or "chrx").strip()

    if not file or not file.filename.endswith(".safetensors"):
        return jsonify({"error": "Upload a .safetensors file"}), 400

    filename = secure_filename(file.filename)
    save_path = LORA_DIR / filename
    file.save(str(save_path))

    _current_lora = {"path": str(save_path), "name": filename, "trigger": trigger}

    # Hot-swap LoRA if Z-Image is already loaded
    with _lock:
        if _zimage_pipe is not None:
            try:
                _zimage_pipe.unload_lora_weights()
            except Exception:
                pass
            _zimage_pipe.load_lora_weights(str(save_path))
            print(f"[Hydra] LoRA hot-swapped: {filename}")

    return jsonify({"name": filename, "trigger": trigger})


@app.route("/api/generate", methods=["POST"])
def generate():
    global _current_image_path

    data = request.get_json(silent=True) or {}
    prompt = (data.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    variant = data.get("model", "deturbo")
    if variant not in ZIMAGE_PRESETS:
        variant = "deturbo"

    preset = ZIMAGE_PRESETS[variant]
    seed = data.get("seed", int(time.time()) % (2**32))

    with _lock:
        _load_zimage(variant)

        generator = torch.Generator(DEVICE).manual_seed(seed)

        result = _zimage_pipe(
            prompt=prompt,
            height=1024,
            width=1024,
            num_inference_steps=preset["steps"],
            guidance_scale=preset["cfg"],
            cfg_normalization=preset["cfg_norm"],
            generator=generator,
            max_sequence_length=512,
        ).images[0]

        filename = f"{uuid.uuid4().hex[:12]}.png"
        out = OUTPUT_DIR / filename
        result.save(str(out))
        _current_image_path = str(out)

    return jsonify({"image_url": f"/outputs/{filename}", "seed": seed, "model": variant})


@app.route("/api/edit", methods=["POST"])
def edit_image():
    global _current_image_path

    data = request.get_json(silent=True) or {}
    instruction = (data.get("prompt") or "").strip()
    if not instruction:
        return jsonify({"error": "Edit instruction is required"}), 400

    if not _current_image_path or not os.path.isfile(_current_image_path):
        return jsonify({"error": "Generate an image first"}), 400

    with _lock:
        _load_qwen()

        source = Image.open(_current_image_path).convert("RGB")

        with torch.inference_mode():
            result = _qwen_pipe(
                image=[source],
                prompt=instruction,
                true_cfg_scale=4.0,
                negative_prompt=" ",
                num_inference_steps=40,
                guidance_scale=1.0,
                num_images_per_prompt=1,
                generator=torch.manual_seed(int(time.time()) % (2**32)),
            ).images[0]

        filename = f"{uuid.uuid4().hex[:12]}.png"
        out = OUTPUT_DIR / filename
        result.save(str(out))
        _current_image_path = str(out)

    return jsonify({"image_url": f"/outputs/{filename}"})


@app.route("/api/status")
def status():
    return jsonify({
        "lora": _current_lora,
        "mode": (
            "generate" if _zimage_pipe
            else "edit" if _qwen_pipe
            else None
        ),
        "zimage_variant": _zimage_variant,
        "has_image": bool(
            _current_image_path and os.path.isfile(_current_image_path)
        ),
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    print("[Hydra] Character Developer — http://0.0.0.0:7862")
    app.run(host="0.0.0.0", port=7862, debug=False, threaded=True)
