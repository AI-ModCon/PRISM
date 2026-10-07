"""
PRISM Chat UI — Gradio-based Multimodal Inference Interface

Uses model.generate() with TextIteratorStreamer for proper KV-cached,
streaming generation. Auto-detects d_model and projector format from
the checkpoint.

Supports NVIDIA CUDA, Intel XPU (Aurora), Apple Silicon (MPS), and CPU.
"""

import os
import re
import sys
import threading

# HF cache — must be set before any transformers/huggingface_hub imports
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
_models_cache = os.path.join(_project_root, "models_cache")
if os.path.isdir(_models_cache):
    os.environ["HF_HUB_CACHE"] = _models_cache
    os.environ["HUGGINGFACE_HUB_CACHE"] = _models_cache
    os.environ["TRANSFORMERS_CACHE"] = _models_cache
    # Avoid hanging on compute nodes with no internet access
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

sys.path.insert(0, _project_root)

# Load .env if available
try:
    from dotenv import load_dotenv
    _env_path = os.path.join(_project_root, ".env")
    if os.path.exists(_env_path):
        load_dotenv(_env_path, override=False)
except ImportError:
    pass

import gradio as gr
import torch
import torch.nn as nn
from PIL import Image
from safetensors.torch import load_file
from src.config import ModelConfig
from src.model import UnifiedTransformer
from torchvision import transforms
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
)


# ============================================================
# Device Detection
# ============================================================
def detect_device() -> str:
    if torch.cuda.is_available():
        print(f"[Device] NVIDIA CUDA: {torch.cuda.get_device_name(0)}")
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        print("[Device] Intel XPU")
        return "xpu"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("[Device] Apple Silicon MPS")
        return "mps"
    print("[Device] CPU")
    return "cpu"


device = detect_device()
model_dtype = torch.bfloat16 if device in ("cuda", "xpu") else torch.float32

# ============================================================
# Configuration
# ============================================================
DEFAULT_CHECKPOINT = os.path.join(
    _project_root,
    "outputs/SMOKE-TEST/2026-03-02/15-47-00/checkpoints/step_500",
)
CHECKPOINT_DIR = os.environ.get("PRISM_CHECKPOINT", DEFAULT_CHECKPOINT)
IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


# ============================================================
# Legacy projector for backward compatibility with older checkpoints
# ============================================================
class LegacyModalityProjector(nn.Module):
    """Old-style projector using nn.Sequential (pre-refactor checkpoints)."""

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
# Model Loading
# ============================================================
model = None
text_model = None
tokenizer = None


def _read_hydra_config(ckpt_dir):
    """Read backbone_id and modalities from the run's Hydra config."""
    # Walk up from checkpoints/step_N to the run dir containing .hydra/
    run_dir = ckpt_dir
    for _ in range(4):  # step_N -> checkpoints -> timestamp -> date -> run_dir
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
                print(f"[Config] Failed to parse {hydra_path}: {e}")
                return {}
        run_dir = os.path.dirname(run_dir)
    return {}


def load_model():
    """Load PRISM model from checkpoint with auto-detection."""
    global model, tokenizer, siglip_processor
    if model is not None:
        return

    print(f"[PRISM] Checkpoint: {CHECKPOINT_DIR}")
    print(f"[PRISM] Device: {device} ({model_dtype})")

    # --- 1. Read Hydra config for backbone_id ---
    hydra_cfg = _read_hydra_config(CHECKPOINT_DIR)
    backbone_id = os.environ.get(
        "PRISM_BACKBONE",
        hydra_cfg.get("backbone_id", "allenai/OLMo-1B-0724-hf"),
    )
    d_img = hydra_cfg.get("d_img", 768)
    modalities = hydra_cfg.get("modalities", ["text", "image"])
    print(f"[PRISM] Backbone: {backbone_id}")
    print(f"[PRISM] Modalities: {modalities}")

    # --- 2. Load checkpoint and auto-detect dimensions ---
    ckpt_file = CHECKPOINT_DIR
    if os.path.isdir(ckpt_file):
        ckpt_file = os.path.join(ckpt_file, "model.safetensors")
    if not os.path.exists(ckpt_file):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_file}")

    print(f"[PRISM] Loading weights from {ckpt_file}")
    state_dict = load_file(ckpt_file, device="cpu")

    # Strip module. prefix (DDP/FSDP)
    clean_sd = {}
    for k, v in state_dict.items():
        clean_sd[k.removeprefix("module.")] = v

    # Auto-detect d_model
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
    print(f"[PRISM] Detected d_model={d_model}")

    # Detect projector format
    uses_legacy = any(k.startswith("projectors.image.net.") for k in clean_sd)
    print(f"[PRISM] Projector format: {'legacy (nn.Sequential)' if uses_legacy else 'current (fc1/fc2)'}")

    # --- 3. Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(backbone_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- 4. Build model ---
    # Only load image modality for the UI (avoids torch_geometric etc.)
    ui_modalities = ["text"]
    if "image" in modalities:
        ui_modalities.append("image")

    config = ModelConfig(
        llm_backbone_id=backbone_id,
        freeze_backbone=True,
        freeze_encoders=True,
        d_model=d_model,
        d_text=d_model,
        d_img=d_img,
        modalities=ui_modalities,
    )
    model = UnifiedTransformer(config)

    # Swap projector if checkpoint uses legacy format
    if uses_legacy and "image" in ui_modalities:
        print("[PRISM] Swapping image projector to legacy format")
        model.projectors["image"] = LegacyModalityProjector(d_img, d_model)

    # --- 6. Load weights ---
    missing, unexpected = model.load_state_dict(clean_sd, strict=False)
    proj_missing = [k for k in missing if "projectors.image" in k]
    if proj_missing:
        print(f"[PRISM] WARNING: {len(proj_missing)} image projector keys missing!")
        for k in proj_missing[:5]:
            print(f"  - {k}")
    else:
        print("[PRISM] Image projector weights loaded OK")

    # --- 7. Move to device ---
    model.to(device=device, dtype=model_dtype)
    model.eval()
    model.tokenizer = tokenizer
    print(f"[PRISM] Model ready on {device}")


def load_text_model():
    """Load standalone backbone for text-only chat."""
    global text_model, tokenizer
    if text_model is not None:
        return

    hydra_cfg = _read_hydra_config(CHECKPOINT_DIR)
    backbone_id = os.environ.get(
        "PRISM_BACKBONE",
        hydra_cfg.get("backbone_id", "allenai/OLMo-1B-0724-hf"),
    )

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(backbone_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    print(f"[Text-Only] Loading {backbone_id}")
    text_model = AutoModelForCausalLM.from_pretrained(
        backbone_id, trust_remote_code=True, torch_dtype=model_dtype
    )
    text_model.to(device)
    text_model.eval()
    print(f"[Text-Only] Ready on {device}")


# ============================================================
# Inference — Multimodal (streaming via TextIteratorStreamer)
# ============================================================
def generate_multimodal_response(
    message, history, img_file, table_file, ts_file, geo_file, graph_file,
    temperature=0.4,
):
    load_model()

    prompt_text = str(message) if message else ""
    if not prompt_text:
        prompt_text = "Describe this image in detail."
    print(f"[Inference] Prompt: '{prompt_text[:80]}' (temp={temperature})")

    input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
    inputs = {"text": input_ids}

    # Process image with the standard PRISM transforms (Resize 224, Normalize [0.5]*3).
    if img_file:
        file_path = img_file if isinstance(img_file, str) else img_file.name
        print(f"[Inference] Image: {file_path}")
        try:
            img = Image.open(file_path).convert("RGB")
            img_tensor = IMG_TRANSFORM(img).unsqueeze(0)  # (1, 3, 224, 224)
            inputs["image"] = img_tensor.to(device, dtype=model_dtype)
        except Exception as e:
            print(f"[Inference] Image processing failed: {e}")
            yield f"Error processing image: {e}"
            return

    # Streaming generation
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(
        inputs=inputs,
        max_new_tokens=200,
        do_sample=temperature > 0,
        temperature=max(temperature, 1e-6),
        top_p=0.9,
        repetition_penalty=1.2,
        streamer=streamer,
    )

    thread = threading.Thread(target=_run_generate, args=(generate_kwargs,))
    thread.start()

    generated = ""
    for chunk in streamer:
        generated += chunk
        yield _clean_response(generated)

    thread.join()


def _clean_response(text: str) -> str:
    """Clean model output for display: fix newlines and strip training artifacts."""
    # Replace literal \n (two chars) with actual newline
    text = text.replace("\\n", "\n")
    # Strip known training data marker tokens that leak through generation
    for marker in (
        "<end of image>", "<image>", "<|endoftext|>",
        "<end of text>", "<|im_end|>", "<|im_start|>",
    ):
        text = text.replace(marker, "")
    # Strip leading JSON/structured-data fragments (training data artifacts).
    # Repeatedly strip lines that look like structured data (braces, quotes, colons)
    # rather than natural language.
    lines = text.split("\n")
    while lines and re.match(r"^[\s{}\[\]'\",;:]+$", lines[0]):
        lines.pop(0)
    text = "\n".join(lines)
    # Collapse 3+ consecutive newlines into double newline (paragraph break)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _run_generate(kwargs):
    """Run model.generate in a background thread."""
    try:
        with torch.no_grad():
            model.generate(**kwargs)
    except Exception as e:
        print(f"[Generate Error] {e}")


# ============================================================
# Inference — Text-Only (streaming)
# ============================================================
def generate_text_only(message, history, temperature=0.4):
    load_text_model()

    input_ids = tokenizer(message, return_tensors="pt").input_ids.to(device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(
        input_ids=input_ids,
        max_new_tokens=256,
        do_sample=temperature > 0,
        temperature=max(temperature, 1e-6),
        top_p=0.9,
        repetition_penalty=1.2,
        streamer=streamer,
    )

    thread = threading.Thread(target=_run_text_generate, args=(generate_kwargs,))
    thread.start()

    generated = ""
    for chunk in streamer:
        generated += chunk
        yield _clean_response(generated)

    thread.join()


def _run_text_generate(kwargs):
    """Run text_model.generate in a background thread."""
    try:
        with torch.no_grad():
            text_model.generate(**kwargs)
    except Exception as e:
        print(f"[Text Generate Error] {e}")


# ============================================================
# UI Layout
# ============================================================
DEVICE_LABEL = {
    "cuda": "NVIDIA CUDA",
    "xpu": "Intel XPU",
    "mps": "Apple Silicon MPS",
    "cpu": "CPU",
}

# Read checkpoint step name for display
_ckpt_name = os.path.basename(CHECKPOINT_DIR)
_hydra = _read_hydra_config(CHECKPOINT_DIR)
_backbone_display = _hydra.get("backbone_id", "unknown").split("/")[-1]

with gr.Blocks(title="PRISM Chat") as demo:
    gr.Markdown(
        f"# PRISM: Poly-Reasoning Integrated Scientific Multimodal Model\n"
        f"**Backbone**: `{_backbone_display}` | "
        f"**Checkpoint**: `{_ckpt_name}` | "
        f"**Device**: {DEVICE_LABEL.get(device, device)}"
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Configuration")
            temperature_slider = gr.Slider(
                minimum=0.0, maximum=1.0, value=0.4, step=0.05,
                label="Temperature",
                info="Lower = more focused, 0 = greedy",
            )
            gr.CheckboxGroup(
                ["Text", "Image"],
                value=["Text", "Image"],
                label="Active Modalities",
                interactive=False,
            )
            gr.Markdown(
                f"**Checkpoint**: `{_ckpt_name}`\n\n"
                f"**Device**: {DEVICE_LABEL.get(device, device)}\n\n"
                "Override with env vars:\n"
                "- `PRISM_CHECKPOINT`\n"
                "- `PRISM_BACKBONE`\n"
                "- `PRISM_IMAGE_ENCODER`"
            )

        with gr.Column(scale=4):
            with gr.Tabs():
                with gr.TabItem("Multimodal Chat"):
                    with gr.Row():
                        with gr.Column(scale=3):
                            chatbot = gr.Chatbot(label="PRISM Chat")
                            with gr.Row():
                                txt_input = gr.Textbox(
                                    show_label=False,
                                    placeholder="Type your message here...",
                                    scale=4,
                                )
                                btn_submit = gr.Button("Send", scale=1)

                            with gr.Row():
                                btn_img = gr.UploadButton("Image", file_types=["image"])
                                btn_table = gr.UploadButton("Table", file_types=[".csv", ".xlsx"])
                                btn_ts = gr.UploadButton("Time Series", file_types=[".csv", ".json"])
                                btn_geo = gr.UploadButton("Geometry", file_types=[".obj", ".ply"])
                                btn_graph = gr.UploadButton("Graph", file_types=[".json", ".pt"])

                        with gr.Column(scale=1):
                            img_preview = gr.Image(
                                label="Uploaded Image",
                                type="filepath",
                                interactive=False,
                            )
                            btn_clear_img = gr.Button("Clear Image", size="sm")

                    state_img = gr.State(None)
                    state_table = gr.State(None)
                    state_ts = gr.State(None)
                    state_geo = gr.State(None)
                    state_graph = gr.State(None)

                    def upload_image(file):
                        file_path = file if isinstance(file, str) else file.name
                        gr.Info(f"Image uploaded: {os.path.basename(file_path)}")
                        return file, file_path  # state + preview

                    def clear_image():
                        return None, None  # state + preview

                    def upload_file(file, name):
                        file_path = file if isinstance(file, str) else file.name
                        gr.Info(f"{name} uploaded: {os.path.basename(file_path)}")
                        return file

                    btn_img.upload(upload_image, btn_img, [state_img, img_preview])
                    btn_clear_img.click(clear_image, None, [state_img, img_preview])
                    btn_table.upload(lambda f: upload_file(f, "Table"), btn_table, [state_table])
                    btn_ts.upload(lambda f: upload_file(f, "Time Series"), btn_ts, [state_ts])
                    btn_geo.upload(lambda f: upload_file(f, "Geometry"), btn_geo, [state_geo])
                    btn_graph.upload(lambda f: upload_file(f, "Graph"), btn_graph, [state_graph])

                    def user_msg(user_message, history, img=None):
                        if history is None:
                            history = []
                        text = str(user_message) if user_message else ""
                        # Show image inline in chat if one is attached
                        if img:
                            img_path = img if isinstance(img, str) else img.name
                            content = text + "\n" if text else ""
                            content += f"![image]({img_path})"
                            history.append({"role": "user", "content": content})
                        else:
                            history.append({"role": "user", "content": text})
                        return "", history

                    def bot_msg(history, img, table, ts, geo, graph, temp):
                        if not history:
                            return history
                        last_msg = history[-1]
                        content = last_msg["content"] if isinstance(last_msg, dict) else str(last_msg)
                        # Strip markdown image from the prompt sent to model
                        user_message = str(content).split("![image]")[0].strip()

                        history.append({"role": "assistant", "content": ""})
                        for chunk in generate_multimodal_response(
                            user_message, history[:-1], img, table, ts, geo, graph,
                            temperature=temp,
                        ):
                            history[-1]["content"] = _clean_response(chunk)
                            yield history

                    txt_input.submit(
                        user_msg, [txt_input, chatbot, state_img], [txt_input, chatbot], queue=False
                    ).then(
                        bot_msg,
                        [chatbot, state_img, state_table, state_ts, state_geo, state_graph, temperature_slider],
                        chatbot,
                    )
                    btn_submit.click(
                        user_msg, [txt_input, chatbot, state_img], [txt_input, chatbot], queue=False
                    ).then(
                        bot_msg,
                        [chatbot, state_img, state_table, state_ts, state_geo, state_graph, temperature_slider],
                        chatbot,
                    )

                with gr.TabItem("Text-Only Chat"):
                    gr.ChatInterface(
                        fn=generate_text_only,
                        multimodal=False,
                        title=f"Text-Only ({_backbone_display})",
                        description=f"Text input only. Uses {_backbone_display} backbone directly.",
                        additional_inputs=[temperature_slider],
                    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
