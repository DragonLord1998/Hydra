"""
Hydra — Character Developer

Z-Image generation (De-Turbo or Foundation) with LoRA support +
Qwen-Image-Edit-2511 instruction-based image editing.
"""

import base64
import gc
import io
import json
import logging
import os
import queue
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import torch
from flask import Flask, Response, jsonify, request, send_from_directory
from PIL import Image
from werkzeug.utils import secure_filename

logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024  # 2 GB upload limit

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
HF_TOKEN = os.environ.get("HF_TOKEN")
HYDRA_API_KEY = os.environ.get("HYDRA_API_KEY")  # optional auth
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_SSE_CONNECTIONS = 10
MAX_PROMPT_LENGTH = 2000

import functools


def require_auth(f):
    """Optional API key auth — only enforced if HYDRA_API_KEY is set."""
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        if HYDRA_API_KEY and request.headers.get("X-API-Key") != HYDRA_API_KEY:
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated


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

# SSE subscribers
_subscribers: list[queue.Queue] = []
_sub_lock = threading.Lock()


def _broadcast(event_type: str, data: dict, priority: bool = False) -> None:
    """Push an SSE event to all connected subscribers.

    When *priority* is True, drop oldest events from full queues to make room
    (ensures critical events like model_status are never silently lost).
    """
    payload = f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
    with _sub_lock:
        for q in list(_subscribers):
            if priority and q.full():
                try:
                    q.get_nowait()  # drop oldest to make room
                except queue.Empty:
                    pass
            try:
                q.put_nowait(payload)
            except queue.Full:
                pass


def _latents_to_preview(latents: torch.Tensor, size: int = 256) -> str | None:
    """Approximate RGB preview from latents without VAE decode. Returns data URI."""
    try:
        if latents.dim() == 3:
            # Packed format (B, seq_len, channels) — common in Flux/Z-Image
            _b, seq_len, _c = latents.shape
            h = w = int(seq_len ** 0.5)
            if h * w < seq_len:
                h += 1
            usable = min(h * w, seq_len)
            lat = latents[0, :usable, :3].detach().float().cpu()
            rows = min(h, int(usable ** 0.5) + 1)
            lat = lat.reshape(rows, -1, 3)[:h, :w, :]
        elif latents.dim() == 4:
            # Spatial format (B, C, H, W)
            lat = latents[0, :3].detach().float().cpu().permute(1, 2, 0)
        else:
            return None

        vmin, vmax = lat.min(), lat.max()
        if (vmax - vmin) > 1e-8:
            lat = (lat - vmin) / (vmax - vmin)
        else:
            lat = torch.full_like(lat, 0.5)

        rgb = (lat.numpy() * 255).clip(0, 255).astype(np.uint8)
        del lat
        preview = Image.fromarray(rgb).resize((size, size), Image.LANCZOS)
        del rgb
        buf = io.BytesIO()
        preview.save(buf, format="JPEG", quality=60)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return None


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

    # Resolve model path — fall back to HuggingFace if local deturbo path missing
    if variant == "deturbo" and os.path.isdir(ZIMAGE_DETURBO_PATH):
        model_path = ZIMAGE_DETURBO_PATH
        label = "De-Turbo"
    elif variant == "deturbo":
        model_path = ZIMAGE_BASE_PATH
        label = "De-Turbo (fallback to Foundation weights)"
        logger.warning(
            "[Hydra] Local De-Turbo path %s not found — using %s from HuggingFace",
            ZIMAGE_DETURBO_PATH, ZIMAGE_BASE_PATH,
        )
        print(f"[Hydra] Local De-Turbo not found at {ZIMAGE_DETURBO_PATH}, using {ZIMAGE_BASE_PATH}")
    else:
        model_path = ZIMAGE_BASE_PATH
        label = "Foundation"

    logger.info("[Hydra] Loading Z-Image %s from %s ...", label, model_path)
    print(f"[Hydra] Loading Z-Image {label} from {model_path} ...")
    _broadcast("model_status", {"action": "loading", "name": f"Z-Image {label}"}, priority=True)

    _zimage_pipe = ZImagePipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        token=HF_TOKEN,
    )
    _zimage_pipe.enable_model_cpu_offload()
    _zimage_variant = variant

    if _current_lora:
        logger.info("[Hydra] Loading LoRA: %s", _current_lora["name"])
        _zimage_pipe.load_lora_weights(_current_lora["path"])

    print(f"[Hydra] Z-Image {label} pipeline ready.")
    _broadcast("model_status", {"action": "ready", "name": f"Z-Image {label}"}, priority=True)


def _load_qwen():
    global _qwen_pipe
    if _qwen_pipe is not None:
        return

    _unload_zimage()

    from diffusers import QwenImageEditPlusPipeline

    logger.info("[Hydra] Loading Qwen-Image-Edit from %s ...", QWEN_EDIT_MODEL)
    print(f"[Hydra] Loading Qwen-Image-Edit from {QWEN_EDIT_MODEL} ...")
    _broadcast("model_status", {"action": "loading", "name": "Qwen-Image-Edit"}, priority=True)

    _qwen_pipe = QwenImageEditPlusPipeline.from_pretrained(
        QWEN_EDIT_MODEL,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        token=HF_TOKEN,
    )
    _qwen_pipe.enable_model_cpu_offload()

    print("[Hydra] Qwen-Image-Edit pipeline ready.")
    _broadcast("model_status", {"action": "ready", "name": "Qwen-Image-Edit"}, priority=True)


# ---------------------------------------------------------------------------
# Routes — static
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/outputs/<filename>")
def serve_output(filename):
    import re as _re
    if not _re.match(r"^[a-f0-9]{12}\.png$", filename):
        return "Not found", 404
    return send_from_directory(str(OUTPUT_DIR), filename)


# ---------------------------------------------------------------------------
# Routes — API
# ---------------------------------------------------------------------------

@app.route("/api/upload-lora", methods=["POST"])
@require_auth
def upload_lora():
    global _current_lora

    file = request.files.get("lora")
    trigger = (request.form.get("trigger_word") or "chrx").strip()

    if not file or not file.filename.endswith(".safetensors"):
        return jsonify({"error": "Upload a .safetensors file"}), 400

    # Validate safetensors header (8-byte LE length + JSON metadata)
    import struct
    header = file.stream.read(8)
    if len(header) < 8:
        return jsonify({"error": "Invalid safetensors file"}), 400
    meta_len = struct.unpack("<Q", header)[0]
    if meta_len > 100 * 1024 * 1024:  # metadata > 100MB is suspicious
        return jsonify({"error": "Invalid safetensors file"}), 400
    file.stream.seek(0)

    filename = secure_filename(file.filename)
    save_path = LORA_DIR / filename
    file.save(str(save_path))

    # Assign inside lock to prevent race with generate/edit
    with _lock:
        _current_lora = {"path": str(save_path), "name": filename, "trigger": trigger}
        if _zimage_pipe is not None:
            try:
                _zimage_pipe.unload_lora_weights()
            except Exception:
                pass
            _zimage_pipe.load_lora_weights(str(save_path))
            print(f"[Hydra] LoRA hot-swapped: {filename}")

    return jsonify({"name": filename, "trigger": trigger})


@app.route("/api/generate", methods=["POST"])
@require_auth
def generate():
    global _current_image_path

    data = request.get_json(silent=True) or {}
    prompt = (data.get("prompt") or "").strip()[:MAX_PROMPT_LENGTH]
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
        total_steps = preset["steps"]

        def _on_step(pipe, step_index, timestep, cb_kwargs):
            if (step_index + 1) % 2 == 0:
                latents = cb_kwargs.get("latents")
                if latents is not None:
                    b64 = _latents_to_preview(latents)
                    if b64:
                        _broadcast("preview", {
                            "step": step_index + 1,
                            "total": total_steps,
                            "image": b64,
                        })
            return cb_kwargs

        try:
            with torch.inference_mode():
                result = _zimage_pipe(
                    prompt=prompt,
                    height=1024,
                    width=1024,
                    num_inference_steps=total_steps,
                    guidance_scale=preset["cfg"],
                    cfg_normalization=preset["cfg_norm"],
                    generator=generator,
                    callback_on_step_end=_on_step,
                    callback_on_step_end_tensor_inputs=["latents"],
                ).images[0]
        except Exception as exc:
            _broadcast("error", {"message": str(exc)}, priority=True)
            return jsonify({"error": f"Generation failed: {exc}"}), 500
        finally:
            torch.cuda.empty_cache()
            gc.collect()

        filename = f"{uuid.uuid4().hex[:12]}.png"
        out = OUTPUT_DIR / filename
        result.save(str(out))
        _current_image_path = str(out)

    return jsonify({"image_url": f"/outputs/{filename}", "seed": seed, "model": variant})


@app.route("/api/edit", methods=["POST"])
@require_auth
def edit_image():
    global _current_image_path

    data = request.get_json(silent=True) or {}
    instruction = (data.get("prompt") or "").strip()[:MAX_PROMPT_LENGTH]
    if not instruction:
        return jsonify({"error": "Edit instruction is required"}), 400

    if not _current_image_path or not os.path.isfile(_current_image_path):
        return jsonify({"error": "Generate an image first"}), 400

    with _lock:
        _load_qwen()

        source = Image.open(_current_image_path).convert("RGB")

        qwen_steps = 20

        def _on_edit_step(pipe, step_index, timestep, cb_kwargs):
            if (step_index + 1) % 2 == 0:
                latents = cb_kwargs.get("latents")
                if latents is not None:
                    b64 = _latents_to_preview(latents)
                    if b64:
                        _broadcast("preview", {
                            "step": step_index + 1,
                            "total": qwen_steps,
                            "image": b64,
                        })
            return cb_kwargs

        try:
            with torch.inference_mode():
                result = _qwen_pipe(
                    image=[source],
                    prompt=instruction,
                    true_cfg_scale=4.0,
                    negative_prompt=" ",
                    num_inference_steps=qwen_steps,
                    guidance_scale=1.0,
                    num_images_per_prompt=1,
                    generator=torch.Generator(DEVICE).manual_seed(int(time.time()) % (2**32)),
                    callback_on_step_end=_on_edit_step,
                    callback_on_step_end_tensor_inputs=["latents"],
                ).images[0]
        except Exception as exc:
            _broadcast("error", {"message": str(exc)}, priority=True)
            return jsonify({"error": f"Edit failed: {exc}"}), 500
        finally:
            del source
            torch.cuda.empty_cache()
            gc.collect()

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
# SSE stream
# ---------------------------------------------------------------------------

@app.route("/api/stream")
def stream():
    with _sub_lock:
        if len(_subscribers) >= MAX_SSE_CONNECTIONS:
            return Response("Too many connections", status=429)
        q: queue.Queue = queue.Queue(maxsize=32)
        _subscribers.append(q)

    def generate():
        try:
            while True:
                try:
                    event = q.get(timeout=30)
                    yield event
                except queue.Empty:
                    yield "event: heartbeat\ndata: {}\n\n"
        except GeneratorExit:
            pass
        finally:
            with _sub_lock:
                if q in _subscribers:
                    _subscribers.remove(q)

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


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
