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
SEEDVR2_CLI = os.environ.get("SEEDVR2_CLI", "/workspace/seedvr2/inference_cli.py")
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
_sam3d_estimator = None                 # SAM3DBodyEstimator
_anypose_loaded = False                 # whether AnyPose LoRAs are on Qwen

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

    import logging as _logging
    from diffusers import AutoencoderTiny

    logger.info("[Hydra] Loading TAESD from %s ...", TAESD_MODEL)
    # Suppress spurious "config attributes {'block_out_channels': ...} will be ignored" warning
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
    """Decode packed Flux/Z-Image latents via TAESD → JPEG data URI."""
    try:
        _load_taesd()
        with torch.inference_mode():
            lat = latents[:1].detach()
            if lat.ndim == 3:
                # Packed format (B, seq, 64) — need to unpack to spatial
                spatial = _unpack_latents(lat, height, width)
            else:
                # Already spatial (B, C, H, W)
                spatial = lat
            spatial = spatial.to(device=DEVICE, dtype=torch.bfloat16)

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
# MHR70 skeleton — SAM 3D Body joint definitions
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

# Major body bones (excluding fingers for the pose editor)
MHR70_BODY_BONES = [
    (0, 69),   # nose → neck
    (69, 5),   # neck → left-shoulder
    (69, 6),   # neck → right-shoulder
    (5, 7),    # left-shoulder → left-elbow
    (6, 8),    # right-shoulder → right-elbow
    (7, 62),   # left-elbow → left-wrist
    (8, 41),   # right-elbow → right-wrist
    (69, 9),   # neck → left-hip (via spine)
    (69, 10),  # neck → right-hip (via spine)
    (9, 11),   # left-hip → left-knee
    (10, 12),  # right-hip → right-knee
    (11, 13),  # left-knee → left-ankle
    (12, 14),  # right-knee → right-ankle
    (13, 15),  # left-ankle → left-big-toe
    (14, 18),  # right-ankle → right-big-toe
    (9, 10),   # left-hip → right-hip (pelvis)
    (5, 6),    # left-shoulder → right-shoulder
]

# Joints that are draggable in the pose editor (major body joints)
MHR70_DRAGGABLE = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 41, 62, 69]

# IK chains: (root, mid, end) — for two-bone IK solving
MHR70_IK_CHAINS = [
    (5, 7, 62),   # left arm: shoulder → elbow → wrist
    (6, 8, 41),   # right arm: shoulder → elbow → wrist
    (9, 11, 13),  # left leg: hip → knee → ankle
    (10, 12, 14), # right leg: hip → knee → ankle
]

SAM3D_CHECKPOINT = os.environ.get(
    "SAM3D_CHECKPOINT",
    str(BASE_DIR / "checkpoints" / "sam-3d-body-dinov3"),
)


# ---------------------------------------------------------------------------
# SAM 3D Body — pose extraction
# ---------------------------------------------------------------------------

def _load_sam3d():
    global _sam3d_estimator
    if _sam3d_estimator is not None:
        return

    _unload_gen()
    _unload_qwen()

    try:
        from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
    except ImportError:
        logger.warning("[Hydra] sam_3d_body not installed — pose extraction unavailable")
        raise RuntimeError("SAM 3D Body not installed. pip install sam-3d-body")

    logger.info("[Hydra] Loading SAM 3D Body from %s ...", SAM3D_CHECKPOINT)
    _broadcast("model_status", {"action": "loading", "name": "SAM 3D Body"}, priority=True)

    ckpt_dir = Path(SAM3D_CHECKPOINT)
    if not ckpt_dir.is_dir():
        # HuggingFace download
        from huggingface_hub import snapshot_download
        ckpt_dir = Path(snapshot_download(SAM3D_CHECKPOINT, token=HF_TOKEN))

    ckpt_path = ckpt_dir / "model.ckpt"
    mhr_path = ckpt_dir / "assets" / "mhr_model.pt"

    model, model_cfg = load_sam_3d_body(
        str(ckpt_path), device=DEVICE, mhr_path=str(mhr_path),
    )
    _sam3d_estimator = SAM3DBodyEstimator(model, model_cfg)

    print("[Hydra] SAM 3D Body ready.")
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
    global _qwen_pipe, _anypose_loaded
    if _qwen_pipe is not None:
        del _qwen_pipe
        _qwen_pipe = None
        _anypose_loaded = False
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
    _unload_sam3d()

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

    # Z-Image's VAE ships with force_upcast=True (fp16 causes NaN/black images).
    # BF16 has the same exponent range as fp32 so it's numerically stable —
    # skip the costly fp32 cast to speed up the final decode significantly.
    if _gen_pipe.vae is not None and _gen_pipe.dtype == torch.bfloat16:
        _gen_pipe.vae.config.force_upcast = False

    _gen_pipe.enable_model_cpu_offload()
    _gen_variant = variant

    if _current_lora:
        logger.info("[Hydra] Loading LoRA: %s", _current_lora["name"])
        _gen_pipe.load_lora_weights(_current_lora["path"])

    # Warmup — first inference triggers CUDA kernel compilation and
    # cpu_offload hook setup; run a tiny throw-away pass so the
    # first real generation isn't penalised.
    logger.info("[Hydra] Warmup pass (%s)...", label)
    _broadcast("model_status", {"action": "loading", "name": f"Warming up {label}..."}, priority=True)
    try:
        with torch.inference_mode():
            _gen_pipe(
                prompt="warmup",
                height=256,
                width=256,
                num_inference_steps=1,
                guidance_scale=0.0,
            )
    except Exception:
        logger.debug("[Hydra] Warmup failed (non-fatal)", exc_info=True)
    torch.cuda.empty_cache()
    gc.collect()

    print(f"[Hydra] {label} pipeline ready.")
    _broadcast("model_status", {"action": "ready", "name": label}, priority=True)


def _load_qwen():
    global _qwen_pipe
    if _qwen_pipe is not None:
        return

    _unload_gen()
    _unload_sam3d()

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
        guidance = min(max(float(data.get("cfg", preset["cfg"])), 0), 20)
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid width, height, steps, or cfg value"}), 400

    with _lock:
        _load_gen(variant)
        _load_taesd()

        generator = torch.Generator(DEVICE).manual_seed(seed)

        def _on_step(pipe, step_index, timestep, cb_kwargs):
            if (step_index + 1) % 2 == 0:
                latents = cb_kwargs.get("latents")
                if latents is not None:
                    if step_index == 0:
                        logger.info("[Hydra] Callback latent shape: %s (ndim=%d)", latents.shape, latents.ndim)
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
                _broadcast("model_status", {"action": "loading", "name": "Decoding final image..."}, priority=True)
            return cb_kwargs

        # Build pipeline kwargs (SRPO/Flux doesn't use cfg_normalization)
        pipe_kwargs = dict(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=total_steps,
            guidance_scale=guidance,
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

    # Allow selecting a specific source image from the canvas
    source_image = data.get("source_image")
    if source_image:
        fname = Path(source_image).name
        if re.match(r"^[a-f0-9]{12}\.png$", fname):
            candidate = OUTPUT_DIR / fname
            if candidate.is_file():
                _current_image_path = str(candidate)

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

    # Use a temp directory for SeedVR2 output, then move result to OUTPUT_DIR
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

        # Find the output PNG — SeedVR2 creates files in the output dir
        result_pngs = []
        for root, _dirs, files in os.walk(tmp_dir):
            for f in files:
                if f.lower().endswith(".png"):
                    result_pngs.append(Path(root) / f)

        if not result_pngs:
            logger.error("[Hydra] SeedVR2 produced no output in %s", tmp_dir)
            _broadcast("error", {"message": "Upscale produced no output"}, priority=True)
            return jsonify({"error": "Upscale produced no output"}), 500

        # Move result to OUTPUT_DIR with a proper filename
        out_filename = f"{uuid.uuid4().hex[:12]}.png"
        out_path = OUTPUT_DIR / out_filename
        shutil.move(str(result_pngs[0]), str(out_path))

        _broadcast("model_status", {"action": "ready", "name": "SeedVR2 Upscaler"}, priority=True)
        logger.info("[Hydra] Upscale complete: %s → %s", fname, out_filename)
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

    # Take the first (most confident) detection
    person = outputs[0]
    kp3d = person["pred_keypoints_3d"]  # (70, 3) tensor or ndarray

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
    """Re-generate a character in a new pose using Qwen + AnyPose LoRAs."""
    global _current_image_path, _anypose_loaded

    data = request.get_json(silent=True) or {}
    prompt = (data.get("prompt") or "").strip()[:MAX_PROMPT_LENGTH]
    character_image = data.get("character_image")
    pose_image_b64 = data.get("pose_image")  # base64 data URI from Three.js

    if not character_image:
        return jsonify({"error": "character_image is required"}), 400
    if not pose_image_b64:
        return jsonify({"error": "pose_image is required"}), 400

    # Resolve character image
    char_fname = Path(character_image).name
    if not re.match(r"^[a-f0-9]{12}\.png$", char_fname):
        return jsonify({"error": "Invalid character image"}), 400
    char_path = OUTPUT_DIR / char_fname
    if not char_path.is_file():
        return jsonify({"error": "Character image not found"}), 400

    # Decode pose reference image from base64 (cap at 5MB to prevent abuse)
    if len(pose_image_b64) > 5 * 1024 * 1024:
        return jsonify({"error": "Pose image too large"}), 400
    try:
        if "," in pose_image_b64:
            pose_image_b64 = pose_image_b64.split(",", 1)[1]
        pose_bytes = base64.b64decode(pose_image_b64)
        pose_img = Image.open(io.BytesIO(pose_bytes)).convert("RGB")
    except Exception:
        return jsonify({"error": "Invalid pose image"}), 400

    try:
        steps = min(max(int(data.get("steps", 4)), 1), 20)
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid steps value"}), 400

    if not prompt:
        prompt = (
            "Make the person in image 1 do the exact same pose of the person in image 2. "
            "Keep the style, identity, clothing, and background of image 1. "
            "The new pose should be pixel accurate to the pose in image 2. "
            "Match the position of arms, legs, head, and torso exactly."
        )

    with _lock:
        _unload_sam3d()
        _load_qwen()

        # Load AnyPose LoRAs if not already loaded
        if not _anypose_loaded:
            logger.info("[Hydra] Loading AnyPose LoRAs ...")
            _broadcast("model_status", {"action": "loading", "name": "AnyPose LoRAs"}, priority=True)
            try:
                _qwen_pipe.load_lora_weights(
                    "lightx2v/Qwen-Image-Edit-2511-Lightning",
                    weight_name="Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
                    adapter_name="fast",
                )
                _qwen_pipe.load_lora_weights(
                    "lilylilith/AnyPose",
                    weight_name="2511-AnyPose-base-000006250.safetensors",
                    adapter_name="base",
                )
                _qwen_pipe.load_lora_weights(
                    "lilylilith/AnyPose",
                    weight_name="2511-AnyPose-helper-00006000.safetensors",
                    adapter_name="helper",
                )
                _anypose_loaded = True
                _broadcast("model_status", {"action": "ready", "name": "AnyPose LoRAs"}, priority=True)
            except Exception:
                logger.exception("[Hydra] Failed to load AnyPose LoRAs")
                _broadcast("error", {"message": "AnyPose LoRA loading failed"}, priority=True)
                return jsonify({"error": "AnyPose LoRA loading failed"}), 500

        _qwen_pipe.set_adapters(
            ["fast", "base", "helper"],
            adapter_weights=[1.0, 0.7, 0.7],
        )

        char_img = Image.open(str(char_path)).convert("RGB")

        try:
            with torch.inference_mode():
                result = _qwen_pipe(
                    image=[char_img, pose_img],
                    prompt=prompt,
                    true_cfg_scale=4.0,
                    negative_prompt=" ",
                    num_inference_steps=steps,
                    guidance_scale=1.0,
                    num_images_per_prompt=1,
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
