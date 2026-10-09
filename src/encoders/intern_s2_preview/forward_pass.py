#!/usr/bin/env python3
"""Load Intern-S2 Preview and run a dummy time-series forward pass."""

# 1. Set up imports and runtime checks
import os
import sys
from pathlib import Path

import torch
import transformers
from dotenv import load_dotenv
from packaging.version import Version
from safetensors.torch import load_file

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

print(f"PyTorch: {torch.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"XPU available: {hasattr(torch, 'xpu') and torch.xpu.is_available()}")

required_transformers = Version("5.2.0")
installed_transformers = Version(transformers.__version__.split("+")[0])
assert installed_transformers >= required_transformers, (
    f"Intern-S2 requires transformers>={required_transformers}; "
    f"this environment has {transformers.__version__}."
)

# 2. Resolve model assets and configuration paths
repo_root = Path(__file__).resolve().parents[3]
load_dotenv(repo_root / ".env", override=False)
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

config_path = repo_root / "src/encoders/intern_s2_preview/config.json"
hf_home = os.getenv("HF_HOME")
assert hf_home, f"HF_HOME is not set; expected it in {repo_root / '.env'}"
checkpoint_path = (
    Path(hf_home).expanduser()
    / "intern-s2-preview-timeseries"
    / "model.safetensors"
)
assert config_path.is_file(), f"Missing config: {config_path}"
assert checkpoint_path.is_file(), f"Missing checkpoint: {checkpoint_path}"
print(f"Config: {config_path}")
print(f"Checkpoint: {checkpoint_path}")

# 3. Instantiate the Intern-S2 time-series model
from src.encoders.intern_s2_preview.configuration_interns2_preview import (
    InternS2PreviewTimeSeriesConfig,
)
from src.encoders.intern_s2_preview.modeling_interns2_preview import (
    InternS2PreviewTimeSeriesModel,
)

if hasattr(torch, "xpu") and torch.xpu.is_available():
    device = torch.device("xpu")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

config = InternS2PreviewTimeSeriesConfig.from_json_file(str(config_path))
model = InternS2PreviewTimeSeriesModel(config)
print(f"Constructed {type(model).__name__} for {device}")

# 4. Load checkpoint weights
state_dict = load_file(str(checkpoint_path), device="cpu")
load_result = model.load_state_dict(state_dict, strict=False)
print(f"Loaded {len(state_dict)} tensors")
print(f"Missing keys: {load_result.missing_keys}")
print(f"Unexpected keys: {load_result.unexpected_keys}")
assert not load_result.missing_keys
assert not load_result.unexpected_keys

model_dtype = next(model.parameters()).dtype

# encoder_embed.forward_encoder hardcodes a cast to bfloat16 before the
# transformer encoder, so that submodule's weights must match or matmul dtypes
# mismatch (float32 vs bfloat16). Cast the submodule to bf16 and monkey-patch
# encoder_embed's forward to cast its output back to model_dtype so the rest
# of the (float32) model downstream still matches dtypes. Done here rather
# than editing modeling_interns2_preview.py, which is a downloaded HF file.
model.encoder_embed.transformer_encoder = model.encoder_embed.transformer_encoder.to(torch.bfloat16)
_orig_encoder_embed_forward = model.encoder_embed.forward


def _encoder_embed_forward_cast(*args, **kwargs):
    outputs, output_lens = _orig_encoder_embed_forward(*args, **kwargs)
    return outputs.to(model_dtype), output_lens


model.encoder_embed.forward = _encoder_embed_forward_cast

model = model.to(device).eval()
print(f"Parameters: {sum(parameter.numel() for parameter in model.parameters()):,}")
print(f"Output width: {model.config.out_hidden_size}")

# 5. Create a dummy time-series tensor
batch_size = 1
sequence_length = 512
num_channels = 1

dummy_input = torch.randn(
    batch_size,
    sequence_length,
    num_channels,
    dtype=model_dtype,
    device=device,
)
ts_lens = torch.tensor([sequence_length], dtype=torch.long, device=device)
sampling_rates = torch.tensor([1.0], dtype=torch.float32, device=device)
channel_counts = torch.tensor([num_channels], dtype=torch.long, device=device)
print(
    f"Input: shape={tuple(dummy_input.shape)}, "
    f"dtype={dummy_input.dtype}, device={dummy_input.device}"
)

# 6. Run a forward pass in inference mode
with torch.inference_mode():
    raw_outputs = model(
        time_series_signals=dummy_input,
        ts_lens=ts_lens,
        sr=sampling_rates,
        channels=channel_counts,
    )

projected_embeddings, pad_mask = raw_outputs
valid_embeddings = projected_embeddings[~pad_mask].reshape(
    batch_size, -1, config.out_hidden_size
)

# 7. Inspect output tensor shapes and types
print(f"Raw output type: {type(raw_outputs).__name__}")
for index, output in enumerate(raw_outputs):
    print(
        f"output[{index}]: shape={tuple(output.shape)}, "
        f"dtype={output.dtype}, device={output.device}"
    )
print(f"Valid embeddings: {tuple(valid_embeddings.shape)}")
print(f"First token values: {valid_embeddings[0, 0, :8].float().cpu().tolist()}")

assert valid_embeddings.shape[0] == batch_size
assert valid_embeddings.shape[-1] == config.out_hidden_size
assert valid_embeddings.numel() > 0
assert torch.isfinite(valid_embeddings).all()

# 8. Minimal re-runnable smoke test
from src.encoders.time_series import _load_intern_s2_model


def run_smoke_test(
    checkpoint: Path, target_device: torch.device
) -> torch.Tensor:
    smoke_model = _load_intern_s2_model(checkpoint).to(target_device).eval()
    smoke_dtype = next(smoke_model.parameters()).dtype
    smoke_input = torch.randn(
        1, 512, 1, dtype=smoke_dtype, device=target_device
    )
    with torch.inference_mode():
        embeddings, mask = smoke_model(
            time_series_signals=smoke_input,
            ts_lens=torch.tensor([512], dtype=torch.long, device=target_device),
            sr=torch.tensor([1.0], dtype=torch.float32, device=target_device),
            channels=torch.tensor([1], dtype=torch.long, device=target_device),
        )
    valid = embeddings[~mask].reshape(
        1, -1, smoke_model.config.out_hidden_size
    )
    assert valid.shape == (1, 64, 2048), valid.shape
    assert torch.isfinite(valid).all()
    return valid


del model, state_dict
if device.type == "cuda":
    torch.cuda.empty_cache()
elif device.type == "xpu":
    torch.xpu.empty_cache()

smoke_output = run_smoke_test(checkpoint_path, device)
print(
    f"Smoke test passed: shape={tuple(smoke_output.shape)}, "
    f"dtype={smoke_output.dtype}"
)