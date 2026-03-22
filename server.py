"""
Hydra -- Character Developer

Flux 2 local inference (pre-quantized checkpoint via diffusers).
Single model for generation + editing.
TAESD-accelerated live latent previews.
SAM 3D Body for local pose extraction.
SeedVR2 for local upscaling.
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

# Model paths
FLUX2_MODEL = os.environ.get("FLUX2_MODEL", "black-forest-labs/FLUX.2-dev")
FLUX2_NVFP4_REPO = os.environ.get("FLUX2_NVFP4_REPO", "black-forest-labs/FLUX.2-dev-NVFP4")
FLUX2_BNB4_REPO = os.environ.get("FLUX2_BNB4_REPO", "diffusers/FLUX.2-dev-bnb-4bit")
TAESD_MODEL = os.environ.get("TAESD_MODEL", "madebyollin/taef1")
SEEDVR2_CLI = os.environ.get("SEEDVR2_CLI", "/workspace/seedvr2/inference_cli.py")
HF_TOKEN = os.environ.get("HF_TOKEN")
HYDRA_API_KEY = os.environ.get("HYDRA_API_KEY")  # optional auth

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_SSE_CONNECTIONS = 10
MAX_PROMPT_LENGTH = 2000
MAX_PIXELS = 4_000_000  # 4 megapixels

SAM3D_CHECKPOINT = os.environ.get(
    "SAM3D_CHECKPOINT",
    str(BASE_DIR / "checkpoints" / "sam-3d-body-dinov3"),
)


def require_auth(f):
    """Optional API key auth -- only enforced if HYDRA_API_KEY is set."""
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        if HYDRA_API_KEY and request.headers.get("X-API-Key") != HYDRA_API_KEY:
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_busy_lock = threading.Lock()
_busy = False
_flux_pipe = None                       # Flux2Pipeline with NVFP4 transformer
_taesd = None                           # AutoencoderTiny for live previews
_current_lora: dict | None = None       # {"path", "name", "trigger"}
_current_image_path: str | None = None  # last generated/edited image
_sam3d_estimator = None                 # SAM3DBodyEstimator

# SSE subscribers
_subscribers: list[queue.Queue] = []
_sub_lock = threading.Lock()


def _broadcast(event_type: str, data: dict, priority: bool = False) -> None:
    """Push an SSE event to all connected subscribers."""
    payload = f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
    with _sub_lock:
        for q in list(_subscribers):
            if priority and q.full():
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass
            try:
                q.put_nowait(payload)
            except queue.Full:
                pass


def _try_acquire_busy():
    """Atomically check and set _busy."""
    global _busy
    with _busy_lock:
        if _busy:
            return False
        _busy = True
        return True


def _release_busy():
    """Release the busy flag."""
    global _busy
    with _busy_lock:
        _busy = False


# ---------------------------------------------------------------------------
# Resolution validation (Flux 2: max 4MP, multiples of 16, max 4096 per dim)
# ---------------------------------------------------------------------------

def _validate_resolution(width: int, height: int) -> tuple[int, int]:
    """Clamp to valid Flux 2 resolution."""
    width = max(256, min(4096, width))
    height = max(256, min(4096, height))
    width = (width // 16) * 16
    height = (height // 16) * 16
    pixels = width * height
    if pixels > MAX_PIXELS:
        scale = (MAX_PIXELS / pixels) ** 0.5
        width = (int(width * scale) // 16) * 16
        height = (int(height * scale) // 16) * 16
    return max(256, width), max(256, height)


# ---------------------------------------------------------------------------
# TAESD -- tiny autoencoder for live latent previews
# ---------------------------------------------------------------------------

def _load_taesd():
    """Lazy-load TAESD (~2MB). Stays resident on GPU."""
    global _taesd
    if _taesd is not None:
        return

    import logging as _logging
    from diffusers import AutoencoderTiny

    logger.info("[Hydra] Loading TAESD from %s ...", TAESD_MODEL)
    _diffusers_logger = _logging.getLogger("diffusers.configuration_utils")
    _prev_level = _diffusers_logger.level
    _diffusers_logger.setLevel(_logging.ERROR)
    try:
        _taesd = AutoencoderTiny.from_pretrained(
            TAESD_MODEL, torch_dtype=torch.bfloat16, token=HF_TOKEN,
        ).to(DEVICE)
    finally:
        _diffusers_logger.setLevel(_prev_level)
    _taesd.eval()
    logger.info("[Hydra] TAESD ready.")


def _unpack_latents(latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Unpack Flux packed latents (B, seq, 64) -> spatial (B, 16, H, W)."""
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
    """Decode packed Flux latents via TAESD -> JPEG data URI."""
    try:
        _load_taesd()
        with torch.inference_mode():
            lat = latents[:1].detach()
            if lat.ndim == 3:
                spatial = _unpack_latents(lat, height, width)
            else:
                spatial = lat
            spatial = spatial.to(device=DEVICE, dtype=torch.bfloat16)
            decoded = _taesd.decode(spatial, return_dict=False)[0]
            del spatial

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
    """Approximate RGB preview from raw latent channels (fallback)."""
    try:
        if latents.dim() == 3:
            _b, seq_len, _c = latents.shape
            h = w = int(seq_len ** 0.5)
            if h * w < seq_len:
                h += 1
            usable = min(h * w, seq_len)
            lat = latents[0, :usable, :3].detach().float().cpu()
            rows = min(h, int(usable ** 0.5) + 1)
            lat = lat.reshape(rows, -1, 3)[:h, :w, :]
        elif latents.dim() == 4:
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
# MHR70 skeleton -- SAM 3D Body joint definitions
# ---------------------------------------------------------------------------

MHR70_NAMES = [
    "nose", "left-eye", "right-eye", "left-ear", "right-ear",
    "left-shoulder", "right-shoulder", "left-elbow", "right-elbow",
    "left-hip", "right-hip", "left-knee", "right-knee",
    "left-ankle", "right-ankle", "left-big-toe-tip", "left-small-toe-tip",
    "left-heel", "right-big-toe-tip", "right-small-toe-tip", "right-heel",
    "right-thumb-tip", "right-thumb-first-joint", "right-thumb-second-joint",
    "right-thumb-third-joint", "right-index-tip", "right-index-first-joint",
    "right-index-second-joint", "right-index-third-joint", "right-middle-tip",
    "right-middle-first-joint", "right-middle-second-joint",
    "right-middle-third-joint", "right-ring-tip", "right-ring-first-joint",
    "right-ring-second-joint", "right-ring-third-joint", "right-pinky-tip",
    "right-pinky-first-joint", "right-pinky-second-joint",
    "right-pinky-third-joint", "right-wrist", "left-thumb-tip",
    "left-thumb-first-joint", "left-thumb-second-joint",
    "left-thumb-third-joint", "left-index-tip", "left-index-first-joint",
    "left-index-second-joint", "left-index-third-joint", "left-middle-tip",
    "left-middle-first-joint", "left-middle-second-joint",
    "left-middle-third-joint", "left-ring-tip", "left-ring-first-joint",
    "left-ring-second-joint", "left-ring-third-joint", "left-pinky-tip",
    "left-pinky-first-joint", "left-pinky-second-joint",
    "left-pinky-third-joint", "left-wrist", "left-olecranon",
    "right-olecranon", "left-cubital-fossa", "right-cubital-fossa",
    "left-acromion", "right-acromion", "neck",
]

MHR70_BODY_BONES = [
    (0, 69), (69, 5), (69, 6), (5, 7), (6, 8), (7, 62), (8, 41),
    (69, 9), (69, 10), (9, 11), (10, 12), (11, 13), (12, 14),
    (13, 15), (14, 18), (9, 10), (5, 6),
]

MHR70_DRAGGABLE = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 41, 62, 69]

MHR70_IK_CHAINS = [
    (5, 7, 62), (6, 8, 41), (9, 11, 13), (10, 12, 14),
]


# ---------------------------------------------------------------------------
# SAM 3D Body -- pose extraction (local)
# ---------------------------------------------------------------------------

def _load_sam3d():
    global _sam3d_estimator
    if _sam3d_estimator is not None:
        return

    _unload_flux()

    try:
        from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
    except ImportError:
        raise RuntimeError("SAM 3D Body not installed. pip install sam-3d-body")

    logger.info("[Hydra] Loading SAM 3D Body from %s ...", SAM3D_CHECKPOINT)
    _broadcast("model_status", {"action": "loading", "name": "SAM 3D Body"}, priority=True)

    ckpt_dir = Path(SAM3D_CHECKPOINT)
    if not ckpt_dir.is_dir():
        from huggingface_hub import snapshot_download
        ckpt_dir = Path(snapshot_download(SAM3D_CHECKPOINT, token=HF_TOKEN))

    ckpt_path = ckpt_dir / "model.ckpt"
    mhr_path = ckpt_dir / "assets" / "mhr_model.pt"

    model, model_cfg = load_sam_3d_body(
        str(ckpt_path), device=DEVICE, mhr_path=str(mhr_path),
    )
    _sam3d_estimator = SAM3DBodyEstimator(model, model_cfg)

    logger.info("[Hydra] SAM 3D Body ready.")
    _broadcast("model_status", {"action": "ready", "name": "SAM 3D Body"}, priority=True)


def _unload_sam3d():
    global _sam3d_estimator
    if _sam3d_estimator is not None:
        del _sam3d_estimator
        _sam3d_estimator = None
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("[Hydra] SAM 3D Body unloaded.")


# ---------------------------------------------------------------------------
# Flux 2 model lifecycle
# ---------------------------------------------------------------------------

def _unload_flux():
    global _flux_pipe
    if _flux_pipe is not None:
        del _flux_pipe
        _flux_pipe = None
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("[Hydra] Flux 2 pipeline unloaded.")


def _load_flux():
    """Load Flux 2 from the pre-quantized BnB 4-bit checkpoint."""
    global _flux_pipe
    if _flux_pipe is not None:
        return

    _unload_sam3d()

    from diffusers import Flux2Pipeline
    from transformers import Mistral3ForConditionalGeneration

    logger.info("[Hydra] Loading Flux 2 (4-bit) from %s ...", FLUX2_BNB4_REPO)
    _broadcast("model_status", {"action": "loading", "name": "Flux 2 (4-bit)"}, priority=True)

    # The BnB 4-bit text encoder (~14 GB) + transformer (~17 GB) together
    # exceed 32 GB VRAM.  Work around this by loading sequentially:
    #   1. Load text encoder to CUDA, then move to CPU (frees VRAM)
    #   2. Load pipeline (transformer) without text encoder
    #   3. Re-attach text encoder and enable cpu_offload
    # cpu_offload moves each component to GPU only during its forward pass.
    logger.info("[Hydra] Loading text encoder...")
    _broadcast("model_status", {"action": "loading", "name": "Text encoder"}, priority=True)
    text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
        FLUX2_BNB4_REPO, subfolder="text_encoder",
        torch_dtype=torch.bfloat16, token=HF_TOKEN,
    )
    text_encoder.to("cpu")
    torch.cuda.empty_cache()
    gc.collect()

    logger.info("[Hydra] Loading transformer pipeline...")
    _broadcast("model_status", {"action": "loading", "name": "Transformer"}, priority=True)
    _flux_pipe = Flux2Pipeline.from_pretrained(
        FLUX2_BNB4_REPO,
        text_encoder=None,
        torch_dtype=torch.bfloat16,
        token=HF_TOKEN,
    )
    _flux_pipe.text_encoder = text_encoder
    _flux_pipe.enable_model_cpu_offload()

    # BF16 VAE: skip costly fp32 upcast (bf16 has same exponent range as fp32)
    if _flux_pipe.vae is not None and _flux_pipe.dtype == torch.bfloat16:
        _flux_pipe.vae.config.force_upcast = False

    # Apply LoRA if one was uploaded
    if _current_lora:
        logger.info("[Hydra] Loading LoRA: %s", _current_lora["name"])
        _flux_pipe.load_lora_weights(_current_lora["path"])

    logger.info("[Hydra] Flux 2 pipeline ready.")
    _broadcast("model_status", {"action": "ready", "name": "Flux 2"}, priority=True)


# ---------------------------------------------------------------------------
# Routes -- static
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
# Routes -- API
# ---------------------------------------------------------------------------

@app.route("/api/upload-lora", methods=["POST"])
@require_auth
def upload_lora():
    global _current_lora

    file = request.files.get("lora")
    trigger = (request.form.get("trigger_word") or "chrx").strip()

    if not file or not file.filename.endswith(".safetensors"):
        return jsonify({"error": "Upload a .safetensors file"}), 400

    header = file.stream.read(8)
    if len(header) < 8:
        return jsonify({"error": "Invalid safetensors file"}), 400
    meta_len = struct.unpack("<Q", header)[0]
    if meta_len > 100 * 1024 * 1024:
        return jsonify({"error": "Invalid safetensors file"}), 400
    file.stream.seek(0)

    filename = secure_filename(file.filename)
    save_path = LORA_DIR / filename
    file.save(str(save_path))

    with _lock:
        _current_lora = {"path": str(save_path), "name": filename, "trigger": trigger}
        if _flux_pipe is not None:
            try:
                _flux_pipe.unload_lora_weights()
            except Exception:
                pass
            _flux_pipe.load_lora_weights(str(save_path))
            logger.info("[Hydra] LoRA hot-swapped: %s", filename)

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

    if not _try_acquire_busy():
        return jsonify({"error": "Generation already in progress"}), 429

    data = request.get_json(silent=True) or {}
    prompt = (data.get("prompt") or "").strip()[:MAX_PROMPT_LENGTH]
    if not prompt:
        _release_busy()
        return jsonify({"error": "Prompt is required"}), 400

    seed = data.get("seed", int(time.time()) % (2**32))

    try:
        width = int(data.get("width", 1024))
        height = int(data.get("height", 1024))
        total_steps = min(max(int(data.get("steps", 50)), 1), 100)
        guidance = min(max(float(data.get("cfg", 4.0)), 0), 20)
    except (ValueError, TypeError):
        _release_busy()
        return jsonify({"error": "Invalid width, height, steps, or cfg value"}), 400

    width, height = _validate_resolution(width, height)

    try:
        with _lock:
            _load_flux()
            _load_taesd()

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
                if step_index + 1 >= total_steps:
                    _broadcast("model_status", {
                        "action": "loading", "name": "Decoding final image...",
                    }, priority=True)
                return cb_kwargs

            try:
                with torch.inference_mode():
                    result = _flux_pipe(
                        prompt=prompt,
                        height=height,
                        width=width,
                        num_inference_steps=total_steps,
                        guidance_scale=guidance,
                        generator=generator,
                        callback_on_step_end=_on_step,
                        callback_on_step_end_tensor_inputs=["latents"],
                    ).images[0]
            except Exception:
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

        _broadcast("model_status", {"action": "ready", "name": "Flux 2"}, priority=True)
        return jsonify({"image_url": f"/outputs/{filename}", "seed": seed})
    finally:
        _release_busy()


@app.route("/api/edit", methods=["POST"])
@require_auth
def edit_image():
    global _current_image_path

    if not _try_acquire_busy():
        return jsonify({"error": "Generation already in progress"}), 429

    data = request.get_json(silent=True) or {}
    instruction = (data.get("prompt") or "").strip()[:MAX_PROMPT_LENGTH]
    if not instruction:
        _release_busy()
        return jsonify({"error": "Edit instruction is required"}), 400

    source_image = data.get("source_image")
    if source_image:
        fname = Path(source_image).name
        if re.match(r"^[a-f0-9]{12}\.png$", fname):
            candidate = OUTPUT_DIR / fname
            if candidate.is_file():
                _current_image_path = str(candidate)

    if not _current_image_path or not os.path.isfile(_current_image_path):
        _release_busy()
        return jsonify({"error": "Generate or upload an image first"}), 400

    try:
        edit_steps = min(max(int(data.get("steps", 50)), 1), 100)
    except (ValueError, TypeError):
        _release_busy()
        return jsonify({"error": "Invalid steps value"}), 400

    try:
        with _lock:
            _load_flux()

            source = Image.open(_current_image_path).convert("RGB")

            def _on_edit_step(pipe, step_index, timestep, cb_kwargs):
                if (step_index + 1) % 2 == 0:
                    latents = cb_kwargs.get("latents")
                    if latents is not None:
                        b64 = _raw_latents_preview(latents)
                        if b64:
                            _broadcast("preview", {
                                "step": step_index + 1,
                                "total": edit_steps,
                                "image": b64,
                            })
                return cb_kwargs

            try:
                with torch.inference_mode():
                    result = _flux_pipe(
                        image=[source],
                        prompt=instruction,
                        num_inference_steps=edit_steps,
                        guidance_scale=2.5,
                        generator=torch.Generator(DEVICE).manual_seed(
                            int(time.time()) % (2**32)
                        ),
                        callback_on_step_end=_on_edit_step,
                        callback_on_step_end_tensor_inputs=["latents"],
                    ).images[0]
            except Exception:
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

        _broadcast("model_status", {"action": "ready", "name": "Flux 2"}, priority=True)
        return jsonify({"image_url": f"/outputs/{filename}"})
    finally:
        _release_busy()


@app.route("/api/upscale", methods=["POST"])
@require_auth
def upscale_image():
    """Upscale an image using SeedVR2."""
    import shutil
    import subprocess
    import tempfile

    data = request.get_json(silent=True) or {}
    source = data.get("source_image")
    if not source:
        return jsonify({"error": "source_image is required"}), 400

    fname = Path(source).name
    if not re.match(r"^[a-f0-9]{12}\.png$", fname):
        return jsonify({"error": "Invalid source image"}), 400

    source_path = OUTPUT_DIR / fname
    if not source_path.is_file():
        return jsonify({"error": "Source image not found"}), 400

    try:
        target_res = min(max(int(data.get("resolution", 1080)), 256), 4096)
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid resolution"}), 400

    if not Path(SEEDVR2_CLI).is_file():
        return jsonify({"error": "SeedVR2 not installed"}), 500

    _broadcast("model_status", {"action": "loading", "name": "SeedVR2 Upscaler"}, priority=True)

    tmp_dir = tempfile.mkdtemp(prefix="hydra_upscale_")
    try:
        cmd = [
            "python3", str(SEEDVR2_CLI),
            str(source_path),
            "--output", tmp_dir,
            "--output_format", "png",
            "--resolution", str(target_res),
            "--dit_offload_device", "cpu",
            "--vae_offload_device", "cpu",
        ]
        logger.info("[Hydra] Upscaling %s at %dp", fname, target_res)
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300,
            cwd=str(Path(SEEDVR2_CLI).parent),
        )
        if proc.returncode != 0:
            stderr = proc.stderr[-1000:] if proc.stderr else ""
            stdout = proc.stdout[-1000:] if proc.stdout else ""
            logger.error("[Hydra] SeedVR2 failed (rc=%d):\nstderr: %s\nstdout: %s",
                         proc.returncode, stderr, stdout)
            _broadcast("error", {"message": "Upscale failed"}, priority=True)
            return jsonify({"error": "Upscale failed"}), 500

        result_pngs = []
        for root, _dirs, files in os.walk(tmp_dir):
            for f in files:
                if f.lower().endswith(".png"):
                    result_pngs.append(Path(root) / f)

        if not result_pngs:
            _broadcast("error", {"message": "Upscale produced no output"}, priority=True)
            return jsonify({"error": "Upscale produced no output"}), 500

        out_filename = f"{uuid.uuid4().hex[:12]}.png"
        out_path = OUTPUT_DIR / out_filename
        shutil.move(str(result_pngs[0]), str(out_path))

        _broadcast("model_status", {"action": "ready", "name": "SeedVR2 Upscaler"}, priority=True)
        return jsonify({"image_url": f"/outputs/{out_filename}"})

    except subprocess.TimeoutExpired:
        _broadcast("error", {"message": "Upscale timed out"}, priority=True)
        return jsonify({"error": "Upscale timed out"}), 500
    except Exception:
        logger.exception("[Hydra] Upscale error")
        _broadcast("error", {"message": "Upscale failed"}, priority=True)
        return jsonify({"error": "Upscale failed"}), 500
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.route("/api/extract-pose", methods=["POST"])
@require_auth
def extract_pose():
    """Extract 3D human pose from an image using SAM 3D Body."""
    data = request.get_json(silent=True) or {}
    source = data.get("source_image")
    if not source:
        return jsonify({"error": "source_image is required"}), 400

    fname = Path(source).name
    if not re.match(r"^[a-f0-9]{12}\.png$", fname):
        return jsonify({"error": "Invalid source image"}), 400

    source_path = OUTPUT_DIR / fname
    if not source_path.is_file():
        return jsonify({"error": "Source image not found"}), 400

    with _lock:
        try:
            _load_sam3d()
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 500

        try:
            outputs = _sam3d_estimator.process_one_image(
                str(source_path), bbox_thr=0.5, use_mask=True,
            )
        except Exception:
            logger.exception("[Hydra] Pose extraction failed")
            _broadcast("error", {"message": "Pose extraction failed"}, priority=True)
            return jsonify({"error": "Pose extraction failed"}), 500

    if not outputs:
        return jsonify({"error": "No human detected in image"}), 400

    person = outputs[0]
    kp3d = person["pred_keypoints_3d"]

    if torch.is_tensor(kp3d):
        kp3d = kp3d.detach().cpu().numpy()
    kp3d = kp3d.tolist()

    joints = []
    for i, name in enumerate(MHR70_NAMES):
        if i < len(kp3d):
            joints.append({
                "name": name,
                "index": i,
                "position": kp3d[i],
                "draggable": i in MHR70_DRAGGABLE,
            })

    return jsonify({
        "joints": joints,
        "bones": MHR70_BODY_BONES,
        "ik_chains": MHR70_IK_CHAINS,
        "draggable_indices": MHR70_DRAGGABLE,
    })


@app.route("/api/generate-posed", methods=["POST"])
@require_auth
def generate_posed():
    """Re-generate a character in a new pose using Flux 2 multi-reference."""
    global _current_image_path

    if not _try_acquire_busy():
        return jsonify({"error": "Generation already in progress"}), 429

    data = request.get_json(silent=True) or {}
    prompt = (data.get("prompt") or "").strip()[:MAX_PROMPT_LENGTH]
    character_image = data.get("character_image")
    pose_image_b64 = data.get("pose_image")

    if not character_image:
        _release_busy()
        return jsonify({"error": "character_image is required"}), 400
    if not pose_image_b64:
        _release_busy()
        return jsonify({"error": "pose_image is required"}), 400

    char_fname = Path(character_image).name
    if not re.match(r"^[a-f0-9]{12}\.png$", char_fname):
        _release_busy()
        return jsonify({"error": "Invalid character image"}), 400
    char_path = OUTPUT_DIR / char_fname
    if not char_path.is_file():
        _release_busy()
        return jsonify({"error": "Character image not found"}), 400

    if len(pose_image_b64) > 5 * 1024 * 1024:
        _release_busy()
        return jsonify({"error": "Pose image too large"}), 400
    try:
        if "," in pose_image_b64:
            pose_image_b64 = pose_image_b64.split(",", 1)[1]
        pose_bytes = base64.b64decode(pose_image_b64)
        pose_img = Image.open(io.BytesIO(pose_bytes)).convert("RGB")
    except Exception:
        _release_busy()
        return jsonify({"error": "Invalid pose image"}), 400

    try:
        steps = min(max(int(data.get("steps", 50)), 1), 100)
    except (ValueError, TypeError):
        _release_busy()
        return jsonify({"error": "Invalid steps value"}), 400

    if not prompt:
        prompt = (
            "Make the person in image 1 do the exact same pose of the person in image 2. "
            "Keep the style, identity, clothing, and background of image 1. "
            "The new pose should be pixel accurate to the pose in image 2. "
            "Match the position of arms, legs, head, and torso exactly."
        )

    try:
        with _lock:
            _unload_sam3d()
            _load_flux()

            char_img = Image.open(str(char_path)).convert("RGB")

            try:
                with torch.inference_mode():
                    result = _flux_pipe(
                        image=[char_img, pose_img],
                        prompt=prompt,
                        num_inference_steps=steps,
                        guidance_scale=2.5,
                        generator=torch.Generator(DEVICE).manual_seed(
                            int(time.time()) % (2**32)
                        ),
                    ).images[0]
            except Exception:
                logger.exception("[Hydra] Posed generation failed")
                _broadcast("error", {"message": "Posed generation failed"}, priority=True)
                return jsonify({"error": "Posed generation failed"}), 500
            finally:
                del char_img, pose_img
                torch.cuda.empty_cache()
                gc.collect()

            filename = f"{uuid.uuid4().hex[:12]}.png"
            out = OUTPUT_DIR / filename
            result.save(str(out))
            _current_image_path = str(out)

        _broadcast("model_status", {"action": "ready", "name": "Flux 2"}, priority=True)
        return jsonify({"image_url": f"/outputs/{filename}"})
    finally:
        _release_busy()


@app.route("/api/status")
def status():
    image_url = None
    if _current_image_path and os.path.isfile(_current_image_path):
        image_url = f"/outputs/{Path(_current_image_path).name}"
    return jsonify({
        "lora": _current_lora,
        "has_image": image_url is not None,
        "image_url": image_url,
        "busy": _busy,
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
    logger.info("[Hydra] Character Developer (Flux 2) -- http://0.0.0.0:7862")

    # Pre-load Flux 2 at startup so the first request is instant
    logger.info("[Hydra] Pre-loading Flux 2...")
    _load_flux()
    _load_taesd()

    app.run(host="0.0.0.0", port=7862, debug=False, threaded=True)
