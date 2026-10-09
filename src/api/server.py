"""
PRISM OpenAI-compatible API server.

Serves the PRISM UnifiedTransformer behind /v1/chat/completions so any
OpenAI-compatible client (e.g. lmms-eval's `openai` chat backend) can drive
it for standardized multimodal evaluation. Single image-per-request for the
smoke-test path; other modalities are stubbed.

Env:
  PRISM_CHECKPOINT   path to the checkpoint dir (contains model.safetensors)
  PRISM_BACKBONE     optional override of the backbone HF id
"""

import os
import sys

# Run as a bare script (`python src/api/server.py`), sys.path[0] is src/api/,
# so the package root has to go on the path before any src.* import — including
# _repo_paths below, which is stdlib-only and safe to import this early.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src._repo_paths import PROJECT_ROOT

# HF cache must be set before any transformers/huggingface_hub imports.
_MODELS_CACHE = os.path.join(PROJECT_ROOT, "models_cache")
if os.path.isdir(_MODELS_CACHE):
    os.environ["HF_HUB_CACHE"] = _MODELS_CACHE
    os.environ["HUGGINGFACE_HUB_CACHE"] = _MODELS_CACHE
    os.environ["TRANSFORMERS_CACHE"] = _MODELS_CACHE
    # Only force offline when the cache actually has content; an empty or
    # partially-populated models_cache/ would otherwise make cold-start
    # tokenizer fetches fail opaquely. Operators can still opt in explicitly
    # by exporting HF_HUB_OFFLINE=1 before invoking the server.
    try:
        if any(os.scandir(_MODELS_CACHE)):
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
    except OSError:
        pass

import base64
import contextlib
import io
import logging
import re
import threading
import time
import uuid
from typing import Any

import torch
import torch.nn as nn
import uvicorn
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel
from safetensors.torch import load_file
from src.config import ModelConfig
from src.model import UnifiedTransformer
from torchvision import transforms
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


# ============================================================
# Device / dtype (mirrors src/ui/app.py)
# ============================================================
def detect_device() -> str:
    if torch.cuda.is_available():
        logger.info("[Device] NVIDIA CUDA: %s", torch.cuda.get_device_name(0))
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        logger.info("[Device] Intel XPU")
        return "xpu"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("[Device] Apple Silicon MPS")
        return "mps"
    logger.info("[Device] CPU")
    return "cpu"


device = detect_device()
model_dtype = torch.bfloat16 if device in ("cuda", "xpu") else torch.float32

# PRISM_CHECKPOINT is required at server start. We deliberately do NOT default
# to a baked-in run path: those go stale across hosts and silently 500 every
# request with FileNotFoundError if the directory was rotated away.
CHECKPOINT_DIR = os.environ.get("PRISM_CHECKPOINT")
SERVED_MODEL_ID = os.environ.get("PRISM_SERVED_MODEL_ID", "prism-mm-v1")

IMG_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

DATA_URI_RE = re.compile(r"^data:image/[^;]+;base64,", re.IGNORECASE)


# ============================================================
# Legacy projector for older checkpoints (mirrors src/ui/app.py)
# ============================================================
class LegacyModalityProjector(nn.Module):
    def __init__(self, input_dim: int, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )
        self.modality_embedding = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

    def forward(self, x):
        x = self.net(x)
        x = x + self.modality_embedding
        return x


# ============================================================
# Model state
# ============================================================
model: UnifiedTransformer | None = None
tokenizer = None
generate_lock = threading.Lock()


def _read_hydra_config(ckpt_dir: str) -> dict[str, Any]:
    run_dir = ckpt_dir
    for _ in range(4):
        hydra_path = os.path.join(run_dir, ".hydra", "config.yaml")
        if os.path.exists(hydra_path):
            try:
                import yaml

                with open(hydra_path) as f:
                    cfg = yaml.safe_load(f)
                return {
                    "backbone_id": cfg.get("model", {}).get("backbone_id"),
                    "modalities": cfg.get("model", {}).get("modalities"),
                    "d_img": cfg.get("model", {}).get("d_img", 768),
                }
            except Exception as e:
                logger.warning("[Config] Failed to parse %s: %s", hydra_path, e)
                return {}
        run_dir = os.path.dirname(run_dir)
    return {}


def load_model() -> None:
    global model, tokenizer
    if model is not None:
        return

    if not CHECKPOINT_DIR:
        raise RuntimeError(
            "PRISM_CHECKPOINT env var is not set. Point it at a checkpoint "
            "directory containing model.safetensors before starting the server."
        )

    logger.info("[PRISM] Checkpoint: %s", CHECKPOINT_DIR)
    logger.info("[PRISM] Device: %s (%s)", device, model_dtype)

    hydra_cfg = _read_hydra_config(CHECKPOINT_DIR)
    backbone_id = os.environ.get(
        "PRISM_BACKBONE",
        hydra_cfg.get("backbone_id", "allenai/OLMo-1B-0724-hf"),
    )
    d_img = hydra_cfg.get("d_img", 768)
    modalities = hydra_cfg.get("modalities", ["text", "image"])
    logger.info("[PRISM] Backbone: %s", backbone_id)
    logger.info("[PRISM] Modalities (config): %s", modalities)

    ckpt_file = CHECKPOINT_DIR
    if os.path.isdir(ckpt_file):
        ckpt_file = os.path.join(ckpt_file, "model.safetensors")
    if not os.path.exists(ckpt_file):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_file}")

    logger.info("[PRISM] Loading weights from %s", ckpt_file)
    state_dict = load_file(ckpt_file, device="cpu")
    clean_sd = {k.removeprefix("module."): v for k, v in state_dict.items()}

    d_model = None
    if "backbone.model.embed_tokens.weight" in clean_sd:
        d_model = clean_sd["backbone.model.embed_tokens.weight"].shape[1]
    if d_model is None:
        for k, v in clean_sd.items():
            if k.startswith("projectors.image.") and k.endswith(".bias") and v.dim() == 1:
                d_model = v.shape[0]
                break
    if d_model is None:
        d_model = 2048
    logger.info("[PRISM] Detected d_model=%s", d_model)

    uses_legacy = any(k.startswith("projectors.image.net.") for k in clean_sd)
    logger.info("[PRISM] Projector format: %s", "legacy" if uses_legacy else "current")

    tokenizer = AutoTokenizer.from_pretrained(backbone_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    server_modalities = ["text"]
    if "image" in modalities:
        server_modalities.append("image")

    config = ModelConfig(
        llm_backbone_id=backbone_id,
        freeze_backbone=True,
        freeze_encoders=True,
        d_model=d_model,
        d_text=d_model,
        d_img=d_img,
        modalities=server_modalities,
    )
    _model = UnifiedTransformer(config)

    if uses_legacy and "image" in server_modalities:
        logger.info("[PRISM] Swapping image projector to legacy format")
        _model.projectors["image"] = LegacyModalityProjector(d_img, d_model)

    missing, _unexpected = _model.load_state_dict(clean_sd, strict=False)
    proj_missing = [k for k in missing if "projectors.image" in k]
    if proj_missing:
        logger.warning("[PRISM] %d image projector keys missing", len(proj_missing))
        for k in proj_missing[:5]:
            logger.warning("  - %s", k)
    else:
        logger.info("[PRISM] Image projector weights loaded OK")

    _model.to(device=device, dtype=model_dtype)
    _model.eval()
    _model.tokenizer = tokenizer

    if "image" in server_modalities:
        try:
            img_enc = _model.encoders["image"]
            enc_name = getattr(img_enc, "model_name", None) or type(img_enc).__name__
            logger.info("[PRISM] Image encoder: %s", enc_name)
            if "siglip" not in str(enc_name).lower():
                logger.warning(
                    "[PRISM] image encoder is not SigLIP; IMG_TRANSFORM "
                    "normalization may not match training preprocessing."
                )
        except Exception as e:
            logger.warning("[PRISM] Could not introspect image encoder: %s", e)

    model = _model
    logger.info("[PRISM] Model ready on %s", device)


# ============================================================
# OpenAI schema
# ============================================================
class ChatMessage(BaseModel):
    role: str
    content: str | list[dict[str, Any]]


class ChatCompletionRequest(BaseModel):
    model: str | None = SERVED_MODEL_ID
    messages: list[ChatMessage]
    max_tokens: int | None = 64
    max_completion_tokens: int | None = None
    temperature: float | None = 0.0
    top_p: float | None = 1.0


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionResponseChoice]
    usage: Usage


# ============================================================
# Helpers
# ============================================================
def _decode_image(url: str) -> Image.Image:
    if url.startswith("data:"):
        b64 = DATA_URI_RE.sub("", url, count=1)
        raw = base64.b64decode(b64)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    raise HTTPException(
        status_code=400,
        detail="Only data: URIs are supported for image_url in this server.",
    )


def _extract_prompt_and_image(content) -> tuple[str, Image.Image | None]:
    if isinstance(content, str):
        return content, None
    text_parts: list[str] = []
    img: Image.Image | None = None
    for part in content:
        ptype = part.get("type")
        if ptype == "text":
            text_parts.append(part.get("text", ""))
        elif ptype == "image_url" and img is None:
            url = (part.get("image_url") or {}).get("url", "")
            if url:
                img = _decode_image(url)
    return "\n".join(t for t in text_parts if t), img


# ============================================================
# FastAPI app
# ============================================================
@contextlib.asynccontextmanager
async def _lifespan(_app: FastAPI):
    load_model()
    yield


app = FastAPI(
    title="PRISM API",
    description="OpenAI-compatible server for PRISM multimodal model",
    lifespan=_lifespan,
)


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": SERVED_MODEL_ID, "object": "model", "owned_by": "prism-team"}],
    }


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    if model is None or tokenizer is None:
        load_model()

    if not request.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")

    prompt_text, pil_img = _extract_prompt_and_image(request.messages[-1].content)
    if not prompt_text:
        prompt_text = "Describe this image." if pil_img is not None else ""

    input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
    inputs: dict[str, Any] = {"text": input_ids}
    if pil_img is not None:
        img_tensor = IMG_TRANSFORM(pil_img).unsqueeze(0).to(device, dtype=model_dtype)
        inputs["image"] = img_tensor

    max_new_tokens = request.max_completion_tokens or request.max_tokens or 64
    temperature = request.temperature if request.temperature is not None else 0.0
    do_sample = temperature > 0.0

    gen_kwargs = dict(
        inputs=inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        top_p=request.top_p if request.top_p is not None else 1.0,
    )
    if do_sample:
        gen_kwargs["temperature"] = max(temperature, 1e-6)

    # UnifiedTransformer.generate() is not threadsafe; serialize.
    with generate_lock:
        with torch.no_grad():
            outputs = model.generate(**gen_kwargs)

    # backbone.generate(inputs_embeds=...) returns ONLY new tokens — decode directly.
    new_tokens = outputs[0]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    prompt_tokens = int(input_ids.numel())
    completion_tokens = int(new_tokens.numel())

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:24]}",
        created=int(time.time()),
        model=request.model or SERVED_MODEL_ID,
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content=text),
                finish_reason="stop",
            )
        ],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
