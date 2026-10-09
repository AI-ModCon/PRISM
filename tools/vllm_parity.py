"""Parity test: run PRISM via demo's UnifiedTransformer path AND via vLLM
on the same image+prompt with greedy decoding. Print both outputs side-by-side
to localize where divergence happens.

Usage on a compute node:
    bash tools/_vllm_parity_runner.sh \\
        --checkpoint outputs/SMOKE-TEST/.../step_500 \\
        --vllm-model exported/prism-olmo1b-image-step500 \\
        --image test_images/cat.jpg
"""

from __future__ import annotations

import argparse
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

# Make sure vLLM spawn workers can import src.* (matches tools/vllm_serve.py).
_pp = os.environ.get("PYTHONPATH", "")
os.environ["PYTHONPATH"] = (
    _PROJECT_ROOT if not _pp else f"{_PROJECT_ROOT}{os.pathsep}{_pp}"
)

# Register PRISM as a vLLM model class at module import time so that spawn
# workers — which re-import this module — also see the registration.
import src.vllm_plugin  # noqa: E402

src.vllm_plugin.register()

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torchvision import transforms  # noqa: E402

PROMPT = "The image shows"
MAX_NEW = 60


def img_xform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


def run_demo_path(checkpoint_dir: str, image_path: str) -> str:
    """Mirror src/ui/app.py:load_model() + generate_multimodal_response()."""
    from safetensors.torch import load_file
    from src.config import ModelConfig
    from src.model import UnifiedTransformer
    from transformers import AutoTokenizer

    if torch.xpu.is_available():
        device = torch.device("xpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    dtype = torch.bfloat16

    backbone_id = "allenai/OLMo-1B-0724-hf"
    d_img = 768

    ckpt_file = os.path.join(checkpoint_dir, "model.safetensors")
    sd = load_file(ckpt_file, device="cpu")
    sd = {k.removeprefix("module."): v for k, v in sd.items()}

    d_model = sd["backbone.model.embed_tokens.weight"].shape[1]
    print(f"[demo] backbone={backbone_id} d_model={d_model} d_img={d_img}")

    tokenizer = AutoTokenizer.from_pretrained(backbone_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # UnifiedTransformer hardcodes local_files_only=True which fails on a fresh
    # node. The pre-warm of AutoModelForCausalLM below populates the HF cache
    # so the local_files_only path then succeeds — no monkey-patch needed.
    cfg = ModelConfig(
        llm_backbone_id=backbone_id,
        freeze_backbone=True,
        freeze_encoders=True,
        d_model=d_model,
        d_text=d_model,
        d_img=d_img,
        modalities=["text", "image"],
    )
    # Pre-warm the cache so UnifiedTransformer's local_files_only=True can find it.
    from transformers import AutoModelForCausalLM as _AMC
    _ = _AMC.from_pretrained(backbone_id, trust_remote_code=True)
    del _

    model = UnifiedTransformer(cfg)
    if model.backbone is None:
        raise RuntimeError("UnifiedTransformer failed to load HF backbone — "
                           "check the warning above.")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    proj_missing = [k for k in missing if "projectors.image" in k]
    print(f"[demo] missing={len(missing)} unexpected={len(unexpected)} "
          f"projector_missing={len(proj_missing)}")
    if proj_missing[:3]:
        print("[demo] sample proj missing:", proj_missing[:3])

    model.to(device=device, dtype=dtype)
    model.eval()
    model.tokenizer = tokenizer

    img = Image.open(image_path).convert("RGB")
    img_tensor = img_xform()(img).unsqueeze(0).to(device, dtype=dtype)

    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to(device)
    inputs = {"text": input_ids, "image": img_tensor}

    print(f"[demo] prompt tokens: {input_ids[0].tolist()}")

    with torch.no_grad():
        out = model.generate(
            inputs,
            max_new_tokens=MAX_NEW,
            do_sample=False,
        )
    if hasattr(out, "sequences"):
        out = out.sequences[0]
    elif isinstance(out, torch.Tensor):
        out = out[0]
    else:
        out = out[0]
    text = tokenizer.decode(out, skip_special_tokens=True)
    return text


def run_vllm_path(model_dir: str, image_path: str, prompt_form: str) -> str:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_dir,
        trust_remote_code=True,
        dtype="bfloat16",
        limit_mm_per_prompt={"image": 1},
        enforce_eager=True,
    )

    sampling = SamplingParams(max_tokens=MAX_NEW, temperature=0.0)

    img = Image.open(image_path).convert("RGB")
    req = {
        "prompt": prompt_form,
        "multi_modal_data": {"image": img},
    }
    outs = llm.generate([req], sampling)
    return outs[0].outputs[0].text


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True,
                   help="PRISM training checkpoint dir (model.safetensors)")
    p.add_argument("--vllm-model", required=True,
                   help="Exported vLLM-loadable dir")
    p.add_argument("--image", required=True)
    p.add_argument("--mode", default="both", choices=["both", "demo", "vllm"])
    p.add_argument("--prompt-form", default="<image>The image shows",
                   help="Exact prompt for vLLM (must contain <image>). Default has "
                   "no space after <image> so the text portion tokenizes identically "
                   "to the demo's `tokenizer('The image shows')` -> [510, 2460, 2722].")
    args = p.parse_args()

    if args.mode in ("demo", "both"):
        print("\n========== DEMO PATH ==========")
        demo_text = run_demo_path(args.checkpoint, args.image)
        print(f"[demo] >>> {demo_text!r}")

    if args.mode in ("vllm", "both"):
        print("\n========== VLLM PATH ==========")
        vllm_text = run_vllm_path(args.vllm_model, args.image, args.prompt_form)
        print(f"[vllm] >>> {vllm_text!r}")


if __name__ == "__main__":
    main()
