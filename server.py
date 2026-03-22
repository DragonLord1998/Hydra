"""
Hydra -- Character Developer

Flux 2 (NVFP4) generation & editing via fal.ai API -- single unified model.
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
import urllib.request
from pathlib import Path

import fal_client
from flask import Flask, Response, jsonify, request, send_from_directory
from PIL import Image
from werkzeug.utils import secure_filename

# Optional: torch/numpy for SAM 3D Body pose extraction
try:
    import torch
    import numpy as np
    HAS_TORCH = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    HAS_TORCH = False
    DEVICE = "cpu"

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

SEEDVR2_CLI = os.environ.get("SEEDVR2_CLI", "/workspace/seedvr2/inference_cli.py")
HF_TOKEN = os.environ.get("HF_TOKEN")
HYDRA_API_KEY = os.environ.get("HYDRA_API_KEY")  # optional auth

MAX_SSE_CONNECTIONS = 10
MAX_PROMPT_LENGTH = 2000
MAX_PIXELS = 4_000_000  # 4 megapixels

# Flux 2 API endpoints (fal.ai -- backed by FLUX.2-dev / NVFP4)
FAL_GENERATE = "fal-ai/flux-2"
FAL_EDIT = "fal-ai/flux-2/edit"
FAL_LORA_GENERATE = "fal-ai/flux-2/lora"
FAL_LORA_EDIT = "fal-ai/flux-2/lora/edit"

# SAM 3D Body checkpoint
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
_busy = False                           # True while a fal.ai request is in flight
_current_lora: dict | None = None       # {"path", "name", "trigger", "fal_url"}
_current_image_path: str | None = None  # last generated/edited image
_sam3d_estimator = None                 # SAM3DBodyEstimator (local)

# SSE subscribers
_subscribers: list[queue.Queue] = []
_sub_lock = threading.Lock()


def _broadcast(event_type: str, data: dict, priority: bool = False) -> None:
    """Push an SSE event to all connected subscribers.

    When *priority* is True, drop oldest events from full queues to make room.
    """
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


# ---------------------------------------------------------------------------
# Resolution validation (Flux 2: max 4MP, multiples of 16, max 4096 per dim)
# ---------------------------------------------------------------------------

def _validate_resolution(width: int, height: int) -> tuple[int, int]:
    """Clamp to valid Flux 2 resolution."""
    width = max(256, min(4096, width))
    height = max(256, min(4096, height))
    # Round to multiples of 16
    width = (width // 16) * 16
    height = (height // 16) * 16
    # Enforce 4MP limit
    pixels = width * height
    if pixels > MAX_PIXELS:
        scale = (MAX_PIXELS / pixels) ** 0.5
        width = (int(width * scale) // 16) * 16
        height = (int(height * scale) // 16) * 16
    return max(256, width), max(256, height)


# ---------------------------------------------------------------------------
# fal.ai helpers
# ---------------------------------------------------------------------------

def _fal_progress_callback():
    """Create a callback for fal.ai queue updates that broadcasts SSE."""
    def on_update(update):
        if hasattr(update, "position"):
            _broadcast("model_status", {
                "action": "loading",
                "name": f"Flux 2 (queued #{update.position})",
            }, priority=True)
        elif hasattr(update, "logs"):
            _broadcast("model_status", {
                "action": "loading",
                "name": "Flux 2 (generating...)",
            }, priority=True)
    return on_update


def _upload_to_fal(local_path: str) -> str:
    """Upload a local file to fal.ai temporary storage."""
    return fal_client.upload_file(local_path)


ALLOWED_FAL_HOSTS = {"fal.media", "v3.fal.media", "fal-cdn.batuhan.co", "storage.googleapis.com"}


def _download_fal_image(url: str) -> tuple[str, Path]:
    """Download image from fal.ai CDN and save to OUTPUT_DIR."""
    import urllib.parse
    parsed = urllib.parse.urlparse(url)
    if parsed.hostname not in ALLOWED_FAL_HOSTS:
        raise ValueError(f"Unexpected image host: {parsed.hostname}")
    filename = f"{uuid.uuid4().hex[:12]}.png"
    out_path = OUTPUT_DIR / filename
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=60) as resp:
        out_path.write_bytes(resp.read())
    return f"/outputs/{filename}", out_path


def _require_fal_key():
    """Return an error response if FAL_KEY is not configured."""
    if not os.environ.get("FAL_KEY"):
        return jsonify({"error": "FAL_KEY not configured. Set your fal.ai API key: export FAL_KEY=your_key"}), 503
    return None


def _try_acquire_busy():
    """Atomically check and set _busy. Returns True if acquired, False if already busy."""
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
# SAM 3D Body -- pose extraction (local, needs torch)
# ---------------------------------------------------------------------------

def _load_sam3d():
    global _sam3d_estimator
    if _sam3d_estimator is not None:
        return

    if not HAS_TORCH:
        raise RuntimeError("torch not installed -- pose extraction unavailable")

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
        if HAS_TORCH:
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("[Hydra] SAM 3D Body unloaded.")


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

    # Validate safetensors header
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

    # Upload to fal.ai storage so the API can access it
    try:
        fal_url = _upload_to_fal(str(save_path))
        logger.info("[Hydra] LoRA uploaded to fal.ai: %s", filename)
    except Exception as exc:
        logger.error("[Hydra] LoRA fal.ai upload failed: %s", exc)
        return jsonify({"error": "LoRA saved but fal.ai upload failed — it will not be applied to generation"}), 500

    with _lock:
        _current_lora = {
            "path": str(save_path),
            "name": filename,
            "trigger": trigger,
            "fal_url": fal_url,
        }

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
    global _current_image_path, _busy

    fal_err = _require_fal_key()
    if fal_err:
        return fal_err

    if not _try_acquire_busy():
        return jsonify({"error": "Generation already in progress"}), 429

    data = request.get_json(silent=True) or {}
    prompt = (data.get("prompt") or "").strip()[:MAX_PROMPT_LENGTH]
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    seed = data.get("seed", int(time.time()) % (2**32))

    try:
        width = int(data.get("width", 1024))
        height = int(data.get("height", 1024))
        steps = min(max(int(data.get("steps", 28)), 1), 50)
        guidance = min(max(float(data.get("cfg", 3.5)), 1), 20)
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid width, height, steps, or cfg value"}), 400

    width, height = _validate_resolution(width, height)

    # Choose endpoint: LoRA or standard
    has_lora = _current_lora and _current_lora.get("fal_url")
    endpoint = FAL_LORA_GENERATE if has_lora else FAL_GENERATE

    args = {
        "prompt": prompt,
        "image_size": {"width": width, "height": height},
        "num_inference_steps": steps,
        "guidance_scale": guidance,
        "seed": seed,
        "num_images": 1,
        "output_format": "png",
    }

    if has_lora:
        args["loras"] = [{"path": _current_lora["fal_url"], "scale": 1.0}]

    _broadcast("model_status", {"action": "loading", "name": "Flux 2"}, priority=True)

    try:
        result = fal_client.subscribe(
            endpoint,
            arguments=args,
            with_logs=True,
            on_queue_update=_fal_progress_callback(),
        )

        images = result.get("images") or result.get("output", {}).get("images")
        if not images:
            _broadcast("error", {"message": "No images returned"}, priority=True)
            return jsonify({"error": "No images returned"}), 500

        local_url, local_path = _download_fal_image(images[0]["url"])

        with _lock:
            _current_image_path = str(local_path)

        _broadcast("model_status", {"action": "ready", "name": "Flux 2"}, priority=True)
        return jsonify({
            "image_url": local_url,
            "seed": result.get("seed", seed),
        })
    except Exception as exc:
        logger.exception("[Hydra] Generation failed")
        _broadcast("error", {"message": "Generation failed"}, priority=True)
        return jsonify({"error": f"Generation failed: {exc}"}), 500
    finally:
        _release_busy()


@app.route("/api/edit", methods=["POST"])
@require_auth
def edit_image():
    global _current_image_path, _busy

    fal_err = _require_fal_key()
    if fal_err:
        return fal_err

    if not _try_acquire_busy():
        return jsonify({"error": "Generation already in progress"}), 429

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
                with _lock:
                    _current_image_path = str(candidate)

    if not _current_image_path or not os.path.isfile(_current_image_path):
        return jsonify({"error": "Generate or upload an image first"}), 400

    try:
        steps = min(max(int(data.get("steps", 28)), 1), 50)
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid steps value"}), 400

    _broadcast("model_status", {"action": "loading", "name": "Flux 2 Edit"}, priority=True)

    try:
        # Upload source image to fal.ai storage
        fal_image_url = _upload_to_fal(_current_image_path)

        has_lora = _current_lora and _current_lora.get("fal_url")
        endpoint = FAL_LORA_EDIT if has_lora else FAL_EDIT

        args = {
            "prompt": instruction,
            "image_url": fal_image_url,
            "num_inference_steps": steps,
            "guidance_scale": 3.5,
            "output_format": "png",
        }

        if has_lora:
            args["loras"] = [{"path": _current_lora["fal_url"], "scale": 1.0}]

        result = fal_client.subscribe(
            endpoint,
            arguments=args,
            with_logs=True,
            on_queue_update=_fal_progress_callback(),
        )

        images = result.get("images") or result.get("output", {}).get("images")
        if not images:
            _broadcast("error", {"message": "No images returned"}, priority=True)
            return jsonify({"error": "No images returned"}), 500

        local_url, local_path = _download_fal_image(images[0]["url"])

        with _lock:
            _current_image_path = str(local_path)

        _broadcast("model_status", {"action": "ready", "name": "Flux 2 Edit"}, priority=True)
        return jsonify({"image_url": local_url})
    except Exception as exc:
        logger.exception("[Hydra] Edit failed")
        _broadcast("error", {"message": "Edit failed"}, priority=True)
        return jsonify({"error": f"Edit failed: {exc}"}), 500
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
            logger.error("[Hydra] SeedVR2 produced no output in %s", tmp_dir)
            _broadcast("error", {"message": "Upscale produced no output"}, priority=True)
            return jsonify({"error": "Upscale produced no output"}), 500

        out_filename = f"{uuid.uuid4().hex[:12]}.png"
        out_path = OUTPUT_DIR / out_filename
        shutil.move(str(result_pngs[0]), str(out_path))

        _broadcast("model_status", {"action": "ready", "name": "SeedVR2 Upscaler"}, priority=True)
        logger.info("[Hydra] Upscale complete: %s -> %s", fname, out_filename)
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

    if HAS_TORCH and torch.is_tensor(kp3d):
        kp3d = kp3d.detach().cpu().numpy()
    if hasattr(kp3d, "tolist"):
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
    """Re-generate a character in a new pose using Flux 2 Edit."""
    global _current_image_path, _busy

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

    # Decode pose reference image from base64
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
        steps = min(max(int(data.get("steps", 28)), 1), 50)
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid steps value"}), 400

    # Build a composite image: character (left) + pose skeleton (right)
    # This gives Flux 2 both references in a single image
    char_img = Image.open(str(char_path)).convert("RGB")
    cw, ch = char_img.size
    pose_img = pose_img.resize((cw, ch), Image.LANCZOS)
    composite = Image.new("RGB", (cw * 2, ch))
    composite.paste(char_img, (0, 0))
    composite.paste(pose_img, (cw, 0))

    # Save composite temporarily
    comp_filename = f"{uuid.uuid4().hex[:12]}.png"
    comp_path = OUTPUT_DIR / comp_filename
    composite.save(str(comp_path))

    if not prompt:
        prompt = (
            "Recreate the person from the left half of this image in the exact pose "
            "shown by the skeleton on the right half. Keep the identity, clothing, "
            "style, and background of the person on the left. Output only the person "
            "in the new pose, not the side-by-side layout."
        )

    fal_err = _require_fal_key()
    if fal_err:
        comp_path.unlink(missing_ok=True)
        return fal_err

    if not _try_acquire_busy():
        comp_path.unlink(missing_ok=True)
        return jsonify({"error": "Generation already in progress"}), 429

    _broadcast("model_status", {"action": "loading", "name": "Flux 2 Pose"}, priority=True)

    try:
        fal_url = _upload_to_fal(str(comp_path))

        args = {
            "prompt": prompt,
            "image_url": fal_url,
            "num_inference_steps": steps,
            "guidance_scale": 3.5,
            "output_format": "png",
        }

        result = fal_client.subscribe(
            FAL_EDIT,
            arguments=args,
            with_logs=True,
            on_queue_update=_fal_progress_callback(),
        )

        images = result.get("images") or result.get("output", {}).get("images")
        if not images:
            _broadcast("error", {"message": "No images returned"}, priority=True)
            return jsonify({"error": "No images returned"}), 500

        local_url, local_path = _download_fal_image(images[0]["url"])

        with _lock:
            _current_image_path = str(local_path)

        _broadcast("model_status", {"action": "ready", "name": "Flux 2 Pose"}, priority=True)
        return jsonify({"image_url": local_url})
    except Exception as exc:
        logger.exception("[Hydra] Posed generation failed")
        _broadcast("error", {"message": "Posed generation failed"}, priority=True)
        return jsonify({"error": f"Posed generation failed: {exc}"}), 500
    finally:
        _release_busy()
        comp_path.unlink(missing_ok=True)


@app.route("/api/status")
def status():
    image_url = None
    if _current_image_path and os.path.isfile(_current_image_path):
        image_url = f"/outputs/{Path(_current_image_path).name}"
    return jsonify({
        "lora": {
            "name": _current_lora["name"],
            "trigger": _current_lora["trigger"],
        } if _current_lora else None,
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

    if not os.environ.get("FAL_KEY"):
        logger.warning("[Hydra] FAL_KEY not set -- generation/editing will fail.")
        logger.warning("[Hydra] Set your fal.ai API key: export FAL_KEY=your_key_here")

    logger.info("[Hydra] Character Developer (Flux 2) -- http://0.0.0.0:7862")
    app.run(host="0.0.0.0", port=7862, debug=False, threaded=True)
