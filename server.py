"""
Hydra — Character Developer

Z-Image generation (De-Turbo or Foundation), Flux SRPO, with LoRA support +
Qwen-Image-Edit-2511 instruction-based image editing.
TAESD-accelerated live latent previews for Flux-based pipelines.
"""

import base64
import functools
import gc
import io
import json
import logging
import os
import queue
import re
import struct
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
LORA_DIR = Path(os.environ.get("LORA_DIR", str(BASE_DIR / "loras")))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", str(BASE_DIR / "outputs")))
LORA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

ZIMAGE_BASE_PATH = os.environ.get("ZIMAGE_BASE_PATH", "Tongyi-MAI/Z-Image")
SRPO_MODEL_PATH = os.environ.get("SRPO_MODEL_PATH", "vladmandic/flux.1-dev-SRPO")
QWEN_EDIT_MODEL = os.environ.get("QWEN_EDIT_MODEL", "Qwen/Qwen-Image-Edit-2511")
TAESD_MODEL = os.environ.get("TAESD_MODEL", "madebyollin/taef1")
HF_TOKEN = os.environ.get("HF_TOKEN")
HYDRA_API_KEY = os.environ.get("HYDRA_API_KEY")  # optional auth
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_SSE_CONNECTIONS = 10
MAX_PROMPT_LENGTH = 2000


def require_auth(f):
    """Optional API key auth — only enforced if HYDRA_API_KEY is set."""
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        if HYDRA_API_KEY and request.headers.get("X-API-Key") != HYDRA_API_KEY:
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated


# Model presets: steps, guidance_scale, cfg_normalization, pipeline type
MODEL_PRESETS = {
    "base":    {"steps": 50, "cfg": 4.0, "cfg_norm": False, "pipeline": "zimage"},
    "srpo":    {"steps": 50, "cfg": 3.5, "cfg_norm": False, "pipeline": "flux"},
}

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_gen_pipe = None                        # ZImagePipeline or FluxPipeline
_gen_variant: str | None = None         # "deturbo", "base", or "srpo"
_qwen_pipe = None
_taesd = None                           # AutoencoderTiny for live previews
_current_lora: dict | None = None       # {"path", "name", "trigger"}
_current_image_path: str | None = None  # last generated/edited image

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


# ---------------------------------------------------------------------------
# TAESD — tiny autoencoder for live latent previews (Flux-based pipelines)
# ---------------------------------------------------------------------------

def _load_taesd():
    """Lazy-load TAESD (~2MB). Stays resident on GPU."""
    global _taesd
    if _taesd is not None:
        return

    from diffusers import AutoencoderTiny

    logger.info("[Hydra] Loading TAESD from %s ...", TAESD_MODEL)
    _taesd = AutoencoderTiny.from_pretrained(
        TAESD_MODEL, torch_dtype=torch.bfloat16, token=HF_TOKEN,
    ).to(DEVICE)
    _taesd.eval()
    logger.info("[Hydra] TAESD ready.")


def _unpack_latents(latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Unpack Flux/Z-Image packed latents (B, seq, 64) → spatial (B, 16, H, W)."""
    batch_size, _num_patches, channels = latents.shape
    vae_scale_factor = 8
    h = 2 * (height // (vae_scale_factor * 2))
    w = 2 * (width // (vae_scale_factor * 2))
    latents = latents.view(batch_size, h // 2, w // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    latents = latents.reshape(batch_size, channels // 4, h, w)
    return latents


def _taesd_preview(latents: torch.Tensor, height: int, width: int,
                   size: int = 512) -> str | None:
    """Decode packed Flux latents via TAESD → JPEG data URI."""
    try:
        _load_taesd()
        with torch.inference_mode():
            spatial = _unpack_latents(
                latents[:1].detach(), height, width,
            ).to(device=DEVICE, dtype=torch.bfloat16)
            decoded = _taesd.decode(spatial, return_dict=False)[0]  # (1, 3, H, W)
            del spatial

        # [-1, 1] → [0, 255]
        img = decoded[0].clamp(-1, 1).add(1).div(2)
        img = img.float().cpu().permute(1, 2, 0).numpy()
        del decoded
        rgb = (img * 255).clip(0, 255).astype(np.uint8)
        del img
        preview = Image.fromarray(rgb).resize((size, size), Image.LANCZOS)
        del rgb
        buf = io.BytesIO()
        preview.save(buf, format="JPEG", quality=70)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        logger.debug("[Hydra] TAESD preview failed, falling back to raw", exc_info=True)
        return None


def _raw_latents_preview(latents: torch.Tensor, size: int = 256) -> str | None:
    """Approximate RGB preview from raw latent channels (fallback for Qwen)."""
    try:
        if latents.dim() == 3:
            # Packed format (B, seq_len, channels) — Flux/Z-Image
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
        elif latents.dim() == 5:
            # Video format (B, C, T, H, W) — Qwen
            lat = latents[0, :3, 0].detach().float().cpu().permute(1, 2, 0)
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

def _unload_gen():
    global _gen_pipe, _gen_variant
    if _gen_pipe is not None:
        del _gen_pipe
        _gen_pipe = None
        _gen_variant = None
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("[Hydra] Generation pipeline unloaded.")


def _unload_qwen():
    global _qwen_pipe
    if _qwen_pipe is not None:
        del _qwen_pipe
        _qwen_pipe = None
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("[Hydra] Qwen-Image-Edit pipeline unloaded.")


def _load_gen(variant: str = "deturbo"):
    global _gen_pipe, _gen_variant

    # If already loaded with the right variant, skip
    if _gen_pipe is not None and _gen_variant == variant:
        return

    # Need a different variant — unload current
    _unload_gen()
    _unload_qwen()

    preset = MODEL_PRESETS[variant]

    if preset["pipeline"] == "flux":
        # SRPO — uses FluxPipeline
        from diffusers import FluxPipeline

        model_path = SRPO_MODEL_PATH
        label = "SRPO"

        logger.info("[Hydra] Loading Flux SRPO from %s ...", model_path)
        print(f"[Hydra] Loading Flux SRPO from {model_path} ...")
        _broadcast("model_status", {"action": "loading", "name": f"Flux {label}"}, priority=True)

        _gen_pipe = FluxPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            token=HF_TOKEN,
        )
    else:
        # Z-Image (base/foundation)
        from diffusers import ZImagePipeline

        model_path = ZIMAGE_BASE_PATH
        label = "Foundation"

        logger.info("[Hydra] Loading Z-Image %s from %s ...", label, model_path)
        print(f"[Hydra] Loading Z-Image {label} from {model_path} ...")
        _broadcast("model_status", {"action": "loading", "name": f"Z-Image {label}"}, priority=True)

        _gen_pipe = ZImagePipeline.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            token=HF_TOKEN,
        )

    _gen_pipe.enable_model_cpu_offload()
    _gen_variant = variant

    if _current_lora:
        logger.info("[Hydra] Loading LoRA: %s", _current_lora["name"])
        _gen_pipe.load_lora_weights(_current_lora["path"])

    print(f"[Hydra] {label} pipeline ready.")
    _broadcast("model_status", {"action": "ready", "name": label}, priority=True)


def _load_qwen():
    global _qwen_pipe
    if _qwen_pipe is not None:
        return

    _unload_gen()

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
    if not re.match(r"^[a-f0-9]{12}\.png$", filename):
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
        if _gen_pipe is not None:
            try:
                _gen_pipe.unload_lora_weights()
            except Exception:
                pass
            _gen_pipe.load_lora_weights(str(save_path))
            print(f"[Hydra] LoRA hot-swapped: {filename}")

    return jsonify({"name": filename, "trigger": trigger})


ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


@app.route("/api/upload-image", methods=["POST"])
@require_auth
def upload_image():
    global _current_image_path

    file = request.files.get("image")
    if not file or not file.filename:
        return jsonify({"error": "No image file provided"}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        return jsonify({"error": "Supported formats: PNG, JPG, WEBP"}), 400

    # Read and validate as an actual image
    try:
        img = Image.open(file.stream)
        img.verify()
        file.stream.seek(0)
        img = Image.open(file.stream).convert("RGB")
    except Exception:
        return jsonify({"error": "Invalid image file"}), 400

    filename = f"{uuid.uuid4().hex[:12]}.png"
    out = OUTPUT_DIR / filename
    img.save(str(out))

    with _lock:
        _current_image_path = str(out)

    return jsonify({"image_url": f"/outputs/{filename}"})


@app.route("/api/generate", methods=["POST"])
@require_auth
def generate():
    global _current_image_path

    data = request.get_json(silent=True) or {}
    prompt = (data.get("prompt") or "").strip()[:MAX_PROMPT_LENGTH]
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    variant = data.get("model", "srpo")
    if variant not in MODEL_PRESETS:
        variant = "srpo"

    preset = MODEL_PRESETS[variant]
    seed = data.get("seed", int(time.time()) % (2**32))

    # User-configurable resolution and steps (with sane clamping)
    try:
        width = min(max(int(data.get("width", 1024)), 256), 2048)
        height = min(max(int(data.get("height", 1024)), 256), 2048)
        total_steps = min(max(int(data.get("steps", preset["steps"])), 1), 100)
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid width, height, or steps value"}), 400

    with _lock:
        _load_gen(variant)

        generator = torch.Generator(DEVICE).manual_seed(seed)

        def _on_step(pipe, step_index, timestep, cb_kwargs):
            if (step_index + 1) % 2 == 0:
                latents = cb_kwargs.get("latents")
                if latents is not None:
                    b64 = _taesd_preview(latents, height, width)
                    if not b64:
                        b64 = _raw_latents_preview(latents)
                    if b64:
                        _broadcast("preview", {
                            "step": step_index + 1,
                            "total": total_steps,
                            "image": b64,
                        })
            return cb_kwargs

        # Build pipeline kwargs (SRPO/Flux doesn't use cfg_normalization)
        pipe_kwargs = dict(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=total_steps,
            guidance_scale=preset["cfg"],
            generator=generator,
            callback_on_step_end=_on_step,
            callback_on_step_end_tensor_inputs=["latents"],
        )
        if preset["pipeline"] == "zimage":
            pipe_kwargs["cfg_normalization"] = preset["cfg_norm"]

        try:
            with torch.inference_mode():
                result = _gen_pipe(**pipe_kwargs).images[0]
        except Exception as exc:
            logger.exception("[Hydra] Generation failed")
            _broadcast("error", {"message": "Generation failed"}, priority=True)
            return jsonify({"error": "Generation failed"}), 500
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

    try:
        qwen_steps = min(max(int(data.get("steps", 20)), 1), 100)
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid steps value"}), 400

    with _lock:
        _load_qwen()

        source = Image.open(_current_image_path).convert("RGB")

        def _on_edit_step(pipe, step_index, timestep, cb_kwargs):
            if (step_index + 1) % 2 == 0:
                latents = cb_kwargs.get("latents")
                if latents is not None:
                    b64 = _raw_latents_preview(latents)
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
            logger.exception("[Hydra] Edit failed")
            _broadcast("error", {"message": "Edit failed"}, priority=True)
            return jsonify({"error": "Edit failed"}), 500
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
    image_url = None
    if _current_image_path and os.path.isfile(_current_image_path):
        image_url = f"/outputs/{Path(_current_image_path).name}"
    return jsonify({
        "lora": _current_lora,
        "mode": (
            "generate" if _gen_pipe
            else "edit" if _qwen_pipe
            else None
        ),
        "gen_variant": _gen_variant,
        "has_image": image_url is not None,
        "image_url": image_url,
        "busy": _lock.locked(),
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
