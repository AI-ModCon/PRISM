#!/usr/bin/env python3

"""Load Intern-S2 Preview 397B and run a dummy time-series forward pass."""

import os
import sys
from pathlib import Path

import torch
import transformers
from dotenv import load_dotenv
from packaging.version import Version
from safetensors.torch import load_file

# Keep runs deterministic enough for smoke testing.
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

repo_root = Path(__file__).resolve().parents[3]
load_dotenv(repo_root / ".env", override=False)
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

config_path = repo_root / "src/encoders/intern_s2_preview_397b/config.json"
hf_home = os.getenv("HF_HOME")
assert hf_home, f"HF_HOME is not set; expected it in {repo_root / '.env'}"
checkpoint_path = (
    Path(hf_home).expanduser()
    / "intern-s2-preview-397b-timeseries"
    / "model.safetensors"
)
assert config_path.is_file(), f"Missing config: {config_path}"
assert checkpoint_path.is_file(), f"Missing checkpoint: {checkpoint_path}"
print(f"Config: {config_path}")
print(f"Checkpoint: {checkpoint_path}")

from src.encoders.intern_s2_preview_397b.configuration_interns2_preview import (
    InternS2PreviewTimeSeriesConfig,
)
from src.encoders.intern_s2_preview_397b.modeling_interns2_preview import (
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

state_dict = load_file(str(checkpoint_path), device="cpu")
load_result = model.load_state_dict(state_dict, strict=False)
print(f"Loaded {len(state_dict)} tensors")
print(f"Missing keys: {load_result.missing_keys}")
print(f"Unexpected keys: {load_result.unexpected_keys}")
assert not load_result.missing_keys
assert not load_result.unexpected_keys

model = model.to(device).eval()
print(f"Parameters: {sum(parameter.numel() for parameter in model.parameters()):,}")
print(f"Output width: {model.config.out_hidden_size}")

batch_size = 1
sequence_length = 512
num_channels = 1
model_dtype = next(model.parameters()).dtype

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

with torch.inference_mode():
    raw_outputs = model(
        time_series_signals=dummy_input,
        ts_lens=ts_lens,
        sr=sampling_rates,
        channels=channel_counts,
    )

projected_embeddings, pad_mask, encoder_out = raw_outputs
valid_embeddings = projected_embeddings[~pad_mask].reshape(
    batch_size, -1, config.out_hidden_size
)

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

from src.encoders.time_series import _load_intern_s2_model


def run_smoke_test(checkpoint: Path, target_device: torch.device) -> torch.Tensor:
    smoke_model = _load_intern_s2_model(
        checkpoint,
        package="src.encoders.intern_s2_preview_397b",
    ).to(target_device).eval()
    smoke_dtype = next(smoke_model.parameters()).dtype
    smoke_input = torch.randn(
        1, 512, 1, dtype=smoke_dtype, device=target_device
    )
    with torch.inference_mode():
        embeddings, pad_mask, _ = smoke_model(
            time_series_signals=smoke_input,
            ts_lens=torch.tensor([512], dtype=torch.long, device=target_device),
            sr=torch.tensor([1.0], dtype=torch.float32, device=target_device),
            channels=torch.tensor([1], dtype=torch.long, device=target_device),
        )
    valid = embeddings[~pad_mask].reshape(
        1, -1, smoke_model.config.out_hidden_size
    )
    assert valid.shape == (1, 1, 4096) or valid.shape[0] == 1, valid.shape
    assert valid.shape[-1] == 4096, valid.shape
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
