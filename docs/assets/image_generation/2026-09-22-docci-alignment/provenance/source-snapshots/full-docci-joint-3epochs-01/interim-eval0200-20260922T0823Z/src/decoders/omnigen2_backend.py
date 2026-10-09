"""Lazy adapter to the released *official* OmniGen2 implementation.

Supported source/API pin (not a claim of accelerator/checkpoint acceptance):
https://github.com/VectorSpaceLab/OmniGen2/tree/18e6f9d5271b517fcb32e999f10df943ae9b8f20
https://huggingface.co/OmniGen2/OmniGen2/tree/df5dca8a981d74e6c3af214c145f5c735fe72367

Install that checkout on PYTHONPATH with diffusers==0.33.1 and
transformers==4.51.3 in a separate image environment.  No imports, installation,
downloads, or model construction happen at module import or __init__. The
upstream uses its own flow scheduler; substituting diffusers' stock scheduler
changes the time convention and is incorrect.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

DEFAULT_REVISION = "df5dca8a981d74e6c3af214c145f5c735fe72367"
UPSTREAM_REVISION = "18e6f9d5271b517fcb32e999f10df943ae9b8f20"
SUPPORTED_VERSIONS = {"diffusers": "0.33.1", "transformers": "4.51.3"}


def _kernel_policy(device) -> str:
    return (
        "upstream_package_detection"
        if torch.device(device).type == "cuda"
        else "upstream_torch_fallback"
    )


def configure_omnigen2_kernels(device) -> dict[str, Any]:
    """Select existing upstream kernels before importing OmniGen2 model classes.

    Upstream probes package presence, so Intel's installed Triton erroneously
    selects a CUDA-only module that queries torch.cuda during import. There is
    no upstream environment switch. This explicit adapter policy sets only
    OmniGen2's two optional-package flags, selecting its existing torch RMSNorm,
    SwiGLU and SDPA branches on CPU/XPU/MPS. It changes no weights, upstream
    source, global import machinery or torch.cuda behavior.

    The selection lasts for this process: the class imports and a later
    FlashAttention constructor both consult these flags. Switching between
    native CUDA and portable classes requires a fresh process. This is a port
    selection, not evidence of numerical parity with CUDA fused kernels.
    """
    selected = _kernel_policy(device)
    availability = importlib.import_module("omnigen2.utils.import_utils")
    previous = getattr(availability, "_prism_kernel_policy", None)
    if previous is not None and previous != selected:
        raise RuntimeError("OmniGen2 kernel policy changed across devices; use a fresh process")
    consumers = (
        "omnigen2.models.transformers.transformer_omnigen2",
        "omnigen2.models.transformers.block_lumina2",
        "omnigen2.models.attention_processor",
        "omnigen2.ops.triton.layer_norm",
    )
    if previous is None and any(name in sys.modules for name in consumers):
        raise RuntimeError(
            "OmniGen2 classes were imported before kernel policy selection; use a fresh process"
        )
    if not hasattr(availability, "_prism_detected_packages"):
        availability._prism_detected_packages = {
            "triton": bool(availability._triton_available),
            "flash_attn": bool(availability._flash_attn_available),
        }
    if selected == "upstream_torch_fallback":
        availability._triton_available = False
        availability._flash_attn_available = False
    availability._prism_kernel_policy = selected
    return {
        "policy": selected,
        "detected_packages": dict(availability._prism_detected_packages),
        "triton_enabled": bool(availability._triton_available),
        "flash_attn_enabled": bool(availability._flash_attn_available),
        "rms_norm": "torch.nn.RMSNorm"
        if selected == "upstream_torch_fallback"
        else "upstream_detection",
        "swiglu": "upstream_components.swiglu"
        if selected == "upstream_torch_fallback"
        else "upstream_detection",
        "attention": "torch_sdpa"
        if selected == "upstream_torch_fallback"
        else "upstream_detection",
        "conditioner_attention": "sdpa"
        if selected == "upstream_torch_fallback"
        else "transformers_default",
        "scope": "omnigen2_optional_kernel_flags_for_process",
        "cross_kernel_parity": "not_run",
    }


class OmniGen2Backend(nn.Module):
    """Pretrained generator, frozen by default, with a differentiable train path.

    Sampling currently supports one prompt with at most five source images;
    training supports fixed-size BCHW batches. Targets are normalized RGB in
    [-1, 1], as in the official training loader. Training returns a latent
    velocity prediction, **not** a decoded image. Sampling returns native
    pipeline images (PIL by default, or its requested output_type).
    """

    def __init__(
        self,
        model_id: str = "OmniGen2/OmniGen2",
        revision: str = DEFAULT_REVISION,
        local_files_only: bool = True,
        conditioning_dim: int = 2048,
    ) -> None:
        super().__init__()
        if not re.fullmatch(r"[0-9a-f]{40}", revision or ""):
            raise ValueError("OmniGen2 revision must be an immutable 40-character commit SHA")
        if conditioning_dim < 1:
            raise ValueError("conditioning_dim must be positive")
        self.model_id = model_id
        self.revision = revision
        self.local_files_only = local_files_only
        self.conditioning_dim = conditioning_dim
        self.register_buffer("_placement", torch.empty(0), persistent=False)
        self._pipeline = None
        self._train_diffusion = False
        self._gradient_checkpointing = False
        self.last_trace: dict[str, torch.Tensor] = {}

    @property
    def train_diffusion(self) -> bool:
        return self._train_diffusion

    def configure_training(
        self, *, train_diffusion: bool = False, gradient_checkpointing: bool = False
    ) -> OmniGen2Backend:
        """Explicitly choose frozen or fully trainable diffusion-transformer weights.

        This is lazy: calling before ``ensure_loaded`` does not load weights.
        The VAE and native prompt conditioner always remain frozen. Call after
        any enclosing model-wide ``requires_grad_(False)`` operation and before
        constructing the optimizer. Checkpointing is an opt-in upstream feature;
        unsupported requests fail rather than silently changing the run budget.
        """
        if type(train_diffusion) is not bool or type(gradient_checkpointing) is not bool:
            raise TypeError("training configuration flags must be bool")
        if gradient_checkpointing and not train_diffusion:
            raise ValueError("gradient checkpointing requires train_diffusion=True")
        if self._pipeline is not None:
            self._configure_gradient_checkpointing(gradient_checkpointing)
        self._train_diffusion = train_diffusion
        self._gradient_checkpointing = gradient_checkpointing
        self._apply_training_policy()
        return self

    def _configure_gradient_checkpointing(self, enabled: bool) -> None:
        transformer = self.transformer
        if enabled:
            enable = getattr(transformer, "enable_gradient_checkpointing", None)
            if not callable(enable):
                raise RuntimeError("OmniGen2 transformer does not support gradient checkpointing")
            enable()
        elif self._gradient_checkpointing:
            disable = getattr(transformer, "disable_gradient_checkpointing", None)
            if not callable(disable):
                raise RuntimeError("OmniGen2 transformer cannot disable gradient checkpointing")
            disable()

    def _apply_training_policy(self) -> None:
        if self._pipeline is None:
            return
        for name in ("vae", "mllm"):
            component = getattr(self, name)
            component.requires_grad_(False)
            component.eval()
        self.transformer.requires_grad_(self._train_diffusion)
        self.transformer.train(self.training and self._train_diffusion)

    def train(self, mode: bool = True) -> OmniGen2Backend:
        super().train(mode)
        # Mode changes must never silently change a configured optimizer scope.
        for name in ("vae", "mllm"):
            component = getattr(self, name, None)
            if component is not None:
                component.eval()
        transformer = getattr(self, "transformer", None)
        if transformer is not None:
            transformer.train(mode and self._train_diffusion)
        return self

    def provenance(self) -> dict[str, Any]:
        return {
            "backend": "official_omnigen2",
            "model_id": self.model_id,
            "checkpoint_revision": self.revision,
            "upstream_revision": UPSTREAM_REVISION,
            "dependency_pins": dict(SUPPORTED_VERSIONS),
            "local_files_only": self.local_files_only,
            "conditioning_dim": self.conditioning_dim,
            "device_type": self._placement.device.type,
            "kernel_policy": getattr(
                self,
                "_selected_kernels",
                {
                    "policy": _kernel_policy(self._placement.device),
                    "selection": "not_imported",
                    "cross_kernel_parity": "not_run",
                },
            ),
            "objective": "linear_flow_velocity_data_minus_noise",
            "time_sampling": "lognormal_dynamic_shift_v1_options_ft_yml",
            "checkpoint_acceptance": "not_run",
            "training_scope": "connector_and_full_diffusion"
            if self._train_diffusion
            else "connector_only",
            "gradient_checkpointing": self._gradient_checkpointing,
        }

    def preflight(self) -> dict[str, Any]:
        """Check dependency/source metadata only; never import models or fetch files."""
        errors = []
        versions = {}
        for package, expected in SUPPORTED_VERSIONS.items():
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
            if versions[package] != expected:
                errors.append(f"requires {package}=={expected}, found {versions[package]!r}")
        spec = importlib.util.find_spec("omnigen2")
        if spec is None or spec.origin is None:
            errors.append(f"official OmniGen2 checkout at {UPSTREAM_REVISION} is not on PYTHONPATH")
        else:
            root = Path(spec.origin).resolve().parent.parent
            try:
                head = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=root,
                    check=True,
                    text=True,
                    capture_output=True,
                ).stdout.strip()
                if head != UPSTREAM_REVISION:
                    errors.append(f"OmniGen2 source revision {head} != {UPSTREAM_REVISION}")
                clean = subprocess.run(
                    ["git", "diff", "--quiet", "HEAD", "--", "omnigen2"],
                    cwd=root,
                    capture_output=True,
                )
                if clean.returncode != 0:
                    errors.append("OmniGen2 source has local modifications")
            except (OSError, subprocess.CalledProcessError):
                errors.append("cannot verify installed OmniGen2 checkout revision")
        return {
            **self.provenance(),
            "versions": versions,
            "errors": errors,
            "ready_to_load": not errors,
            "weights_checked": False,
        }

    def checkpoint_manifest(self) -> dict[str, Any]:
        """Hash the loaded pinned snapshot, including actual component weights.

        This is intentionally an explicit, potentially expensive operation.
        It never fetches files; callers must first load the backend. The digest
        identifies local file contents, rather than merely a mutable Hub name.
        """
        if self._pipeline is None:
            raise RuntimeError("load the image backend before hashing its checkpoint")
        if hasattr(self, "_checkpoint_manifest"):
            return self._checkpoint_manifest
        from huggingface_hub import snapshot_download

        root = Path(self.model_id)
        if not root.is_dir():
            root = Path(
                snapshot_download(
                    repo_id=self.model_id, revision=self.revision, local_files_only=True
                )
            )
        hashes = {}
        for component in ("transformer", "vae", "mllm", "processor", "scheduler"):
            files = sorted((root / component).rglob("*"))
            if not any(path.is_file() for path in files):
                raise RuntimeError(f"missing cached checkpoint component {component}")
            for path in files:
                if path.is_file():
                    digest = hashlib.sha256()
                    with path.open("rb") as handle:
                        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                            digest.update(chunk)
                    hashes[path.relative_to(root).as_posix()] = digest.hexdigest()
        # A cache directory/location is provenance, not weight identity. The same
        # pinned component bytes copied to another host must retain this digest.
        identity = {"revision": self.revision, "files": hashes}
        encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        self._checkpoint_manifest = {
            "model_id": self.model_id,
            **identity,
            "manifest_sha256": hashlib.sha256(encoded).hexdigest(),
        }
        return self._checkpoint_manifest

    def _load(self) -> None:
        if self._pipeline is not None:
            selected = getattr(self, "_selected_kernels", None)
            if selected is not None and selected["policy"] != _kernel_policy(
                self._placement.device
            ):
                raise RuntimeError(
                    "OmniGen2 loaded kernel policy differs from device; use a fresh process"
                )
            return
        report = self.preflight()
        if report["errors"]:
            raise RuntimeError("OmniGen2 backend unavailable: " + "; ".join(report["errors"]))
        try:
            self._selected_kernels = configure_omnigen2_kernels(self._placement.device)
            from diffusers import AutoencoderKL
            from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel
            from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
            from omnigen2.schedulers.scheduling_flow_match_euler_discrete import (
                FlowMatchEulerDiscreteScheduler,
            )
            from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor

            common = dict(revision=self.revision, local_files_only=self.local_files_only)
            weights = dict(common, torch_dtype=self._placement.dtype)
            # Explicit component loaders avoid executing checkpoint-provided code.
            transformer = OmniGen2Transformer2DModel.from_pretrained(
                self.model_id, subfolder="transformer", **weights
            )
            if transformer.config.text_feat_dim != self.conditioning_dim:
                raise ValueError("checkpoint text_feat_dim does not match conditioning_dim")
            vae = AutoencoderKL.from_pretrained(self.model_id, subfolder="vae", **weights)
            conditioner_options = {}
            if self._selected_kernels["policy"] == "upstream_torch_fallback":
                conditioner_options["attn_implementation"] = "sdpa"
            mllm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_id, subfolder="mllm", **weights, **conditioner_options
            )
            processor = Qwen2_5_VLProcessor.from_pretrained(
                self.model_id, subfolder="processor", **common
            )
            scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                self.model_id, subfolder="scheduler", **common
            )
            pipe = OmniGen2Pipeline(
                transformer=transformer,
                vae=vae,
                mllm=mllm,
                processor=processor,
                scheduler=scheduler,
            )
            pipe.to(device=self._placement.device, dtype=self._placement.dtype)
            pipe.enable_taylorseer = False
            pipe.transformer.enable_teacache = False
            pipe.transformer.enable_taylorseer = False
            # Register modules so device moves and gradient audits work.
            self.transformer, self.vae, self.mllm = transformer, vae, mllm
            self._configure_gradient_checkpointing(self._gradient_checkpointing)
            self._pipeline = pipe
            # Preserve the legacy default of loading all components in eval.
            # An explicit diffusion-training request preserves parent mode.
            if not self._train_diffusion:
                self.eval()
            self._apply_training_policy()
        except Exception as exc:
            self._pipeline = None
            raise RuntimeError(
                f"Could not load pinned OmniGen2 {self.model_id}@{self.revision} "
                f"(local_files_only={self.local_files_only}): {exc}"
            ) from exc

    def ensure_loaded(self) -> OmniGen2Backend:
        """Load pinned components and apply the explicitly configured training scope."""
        self._load()
        return self

    @staticmethod
    def _native(native_context: dict | None, batch_size: int | None = None) -> dict:
        native = dict(native_context or {})
        if "reference_images" in native:
            if "input_images" in native:
                raise ValueError("use reference_images or input_images, not both")
            refs = native.pop("reference_images")
            if not isinstance(refs, list) or any(not isinstance(row, list) for row in refs):
                raise ValueError("reference_images must be an ordered list of lists")
            if batch_size is not None and len(refs) != batch_size:
                raise ValueError("reference_images must match the condition batch size")
            if any(len(row) > 5 for row in refs):
                raise ValueError("OmniGen2 supports at most 5 reference images per example")
            ids = native.pop("source_ids", None)
            if ids is not None and (
                len(ids) != len(refs)
                or any(len(names) != len(images) for names, images in zip(ids, refs, strict=False))
                or any(len(set(names)) != len(names) for names in ids)
            ):
                raise ValueError("source_ids must align with ordered, distinct reference_images")
            mask = native.pop("source_mask", None)
            if mask is not None:
                if (
                    not isinstance(mask, torch.Tensor)
                    or mask.ndim != 2
                    or mask.shape[0] != len(refs)
                ):
                    raise ValueError("source_mask must have shape (B, max_sources)")
                counts = torch.tensor([len(row) for row in refs], device=mask.device)
                expected = (
                    torch.arange(mask.shape[1], device=mask.device)[None, :] < counts[:, None]
                )
                if (counts > mask.shape[1]).any() or not torch.equal(mask, expected):
                    raise ValueError("source_mask must preserve the valid reference-image order")
            # The upstream adds the batch axis itself for B=1.
            native["input_images"] = refs[0] if len(refs) == 1 else refs
        images = native.get("input_images")
        if images is not None:
            if not isinstance(images, list):
                raise ValueError("input_images must be an ordered list or None")
            rows = [images] if batch_size in (None, 1) else images
            if batch_size not in (None, 1) and len(rows) != batch_size:
                raise ValueError("input_images must have one reference list per batch item")
            if any(row is not None and not isinstance(row, list) for row in rows):
                raise ValueError("batched input_images must contain reference lists or None")
            if any(row is not None and len(row) > 5 for row in rows):
                raise ValueError("OmniGen2 supports at most 5 reference images per example")
        permitted = {
            "prompt",
            "negative_prompt",
            "input_images",
            "negative_prompt_embeds",
            "negative_prompt_attention_mask",
        }
        unknown = set(native) - permitted
        if unknown:
            raise ValueError(f"unsupported image native_context keys: {sorted(unknown)}")
        return native

    def _check_condition(self, embeds, mask) -> None:
        if embeds.ndim != 3 or embeds.shape[-1] != self.conditioning_dim:
            raise ValueError("OmniGen2 conditioning must have shape (B, L, conditioning_dim)")
        if mask.shape != embeds.shape[:2] or not torch.all((mask == 0) | (mask == 1)):
            raise ValueError("OmniGen2 attention mask must be binary (B, L)")
        valid = mask.bool()
        if not valid.any(1).all() or (valid[:, 1:] & ~valid[:, :-1]).any():
            raise ValueError("OmniGen2 conditioning requires nonempty, right-padded valid prefixes")

    def _call(self, options: dict, trace: bool):
        self._load()
        modes = [(module, module.training) for module in self.modules()]
        try:
            self.eval()
            return self._call_pipeline(options, trace)
        finally:
            # Restore every submodule's exact mode, including exceptional exits.
            # ``train()`` recursion would overwrite intentionally mixed modes.
            for module, training in modes:
                module.training = training

    def _call_pipeline(self, options: dict, trace: bool):
        pipe = self._pipeline
        self.last_trace = {}
        originals = {}
        hooks = []
        active_prompt = ["positive"]

        def save(name, value):
            if isinstance(value, torch.Tensor):
                self.last_trace[name] = value.detach().cpu().clone()

        if trace:

            def wrap(name, handler):
                originals[name] = getattr(pipe, name)
                original = originals[name]

                def wrapped(*args, **kwargs):
                    result = original(*args, **kwargs)
                    handler(args, kwargs, result)
                    return result

                setattr(pipe, name, wrapped)

            def capture_prompt(args, kwargs, out):
                for name, value in zip(
                    ("condition.positive", "mask.positive", "condition.negative", "mask.negative"),
                    out,
                    strict=False,
                ):
                    save(name, value)

            wrap("encode_prompt", capture_prompt)
            wrap("prepare_latents", lambda a, k, out: save("latents.initial", out))

            def capture_refs(args, kwargs, out):
                for batch, refs in enumerate(out):
                    for index, latent in enumerate(refs or []):
                        save(f"reference.{batch}.{index}", latent)

            wrap("prepare_image", capture_refs)

            def capture_prediction(args, kwargs, out):
                if "prediction.step0" not in self.last_trace:
                    save("prediction.step0", out)
                if "latents.final" not in self.last_trace:
                    index = sum(key.startswith("prediction.branch") for key in self.last_trace)
                    save(f"prediction.branch{index}", out)
                    save(f"condition.branch{index}", kwargs.get("prompt_embeds"))
                    save(f"mask.branch{index}", kwargs.get("prompt_attention_mask"))

            wrap("predict", capture_prediction)
            originals["scheduler.step"] = pipe.scheduler.step

            def step(*args, **kwargs):
                out = originals["scheduler.step"](*args, **kwargs)
                latents = out[0] if isinstance(out, tuple) else out.prev_sample
                if "latents.step0" not in self.last_trace:
                    save("latents.step0", latents)
                save("latents.final", latents)
                return out

            pipe.scheduler.step = step
            if isinstance(pipe.mllm, nn.Module):

                def capture_tokens(module, args, kwargs):
                    prefix = (
                        "positive" if "token_ids.positive" not in self.last_trace else "negative"
                    )
                    active_prompt[0] = prefix
                    save(f"token_ids.{prefix}", kwargs.get("input_ids", args[0] if args else None))
                    save(f"token_mask.{prefix}", kwargs.get("attention_mask"))
                    save(f"position_ids.{prefix}", kwargs.get("position_ids"))

                hooks.append(pipe.mllm.register_forward_pre_hook(capture_tokens, with_kwargs=True))
                if hasattr(pipe.mllm, "get_rope_index"):
                    originals["mllm.get_rope_index"] = pipe.mllm.get_rope_index

                    def rope_index(*args, **kwargs):
                        out = originals["mllm.get_rope_index"](*args, **kwargs)
                        save(f"position_ids.{active_prompt[0]}", out[0])
                        save(f"rope_deltas.{active_prompt[0]}", out[1])
                        return out

                    pipe.mllm.get_rope_index = rope_index
            originals["vae.decode"] = pipe.vae.decode

            def decode(latents, *args, **kwargs):
                # Inverse VAE normalization already applied at this boundary.
                save("latents.vae_input", latents)
                out = originals["vae.decode"](latents, *args, **kwargs)
                save("pixels.vae_output", out[0])
                return out

            pipe.vae.decode = decode
        try:
            output = pipe(**options)
            if trace:
                save("schedule.timesteps", pipe.scheduler.timesteps)
            images = output.images if hasattr(output, "images") else output
            save("pixels", images)
            if trace and "pixels" not in self.last_trace:
                import numpy as np

                save("pixels", torch.from_numpy(np.array(images)))
            return images
        finally:
            for hook in hooks:
                hook.remove()
            for name, original in originals.items():
                if name == "vae.decode":
                    pipe.vae.decode = original
                elif name == "scheduler.step":
                    pipe.scheduler.step = original
                elif name == "mllm.get_rope_index":
                    pipe.mllm.get_rope_index = original
                else:
                    setattr(pipe, name, original)

    @torch.no_grad()
    def generate_reference(self, native_context: dict, *, trace: bool = False, **kwargs):
        native = self._native(native_context, batch_size=1)
        if "prompt" not in native:
            raise ValueError("reference mode requires native_context['prompt']")
        prompt = native["prompt"]
        if not isinstance(prompt, str) and (not isinstance(prompt, list) or len(prompt) != 1):
            raise ValueError("OmniGen2 image sampling currently supports one prompt per call")
        if {"prompt_embeds", "prompt_attention_mask"} & kwargs.keys():
            raise ValueError("reference mode must retain the native prompt conditioner")
        overlap = set(native) & set(kwargs)
        if overlap:
            raise ValueError(f"duplicate native image options: {sorted(overlap)}")
        return self._call({**native, **kwargs}, trace)

    @torch.no_grad()
    def generate_conditioned(
        self, embeds, attention_mask, native_context=None, *, trace=False, **kwargs
    ):
        self._check_condition(embeds, attention_mask)
        if embeds.shape[0] != 1:
            raise ValueError("OmniGen2 image sampling currently supports batch_size=1")
        native = self._native(native_context, batch_size=embeds.shape[0])
        # Upstream encode_prompt iterates prompt even with supplied embeddings.
        # This placeholder is formatted but never encoded for positive context.
        native["prompt"] = [""]
        if {"prompt", "prompt_embeds", "prompt_attention_mask"} & kwargs.keys():
            raise ValueError("PRISM mode owns positive prompt embeddings and attention mask")
        options = {**native, **kwargs}
        negative_embeds = options.get("negative_prompt_embeds")
        negative_mask = options.get("negative_prompt_attention_mask")
        if (negative_embeds is None) != (negative_mask is None):
            raise ValueError(
                "negative_prompt_embeds and negative_prompt_attention_mask must be supplied together"
            )
        if negative_embeds is not None:
            if not isinstance(negative_embeds, torch.Tensor) or not isinstance(
                negative_mask, torch.Tensor
            ):
                raise ValueError("negative prompt embeddings and attention mask must be tensors")
            try:
                self._check_condition(negative_embeds, negative_mask)
            except ValueError as error:
                raise ValueError(f"Invalid negative prompt conditioning: {error}") from error
            if negative_embeds.shape[0] != embeds.shape[0]:
                raise ValueError(
                    "negative prompt conditioning batch size must match positive conditioning"
                )
        self._load()
        device, dtype = self._pipeline.transformer.device, self._pipeline.transformer.dtype
        if negative_embeds is not None:
            options["negative_prompt_embeds"] = negative_embeds.to(device=device, dtype=dtype)
            options["negative_prompt_attention_mask"] = negative_mask.to(device=device)
        return self._call(
            {
                **options,
                "prompt_embeds": embeds.to(device=device, dtype=dtype),
                "prompt_attention_mask": attention_mask.to(device=device),
            },
            trace,
        )

    def training_step(
        self,
        embeds,
        attention_mask,
        targets,
        native_context=None,
        *,
        generator=None,
        noise=None,
        timesteps=None,
        height=None,
        width=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Official linear flow objective, t=0 noise -> t=1 data.

        Default sampling is lognormal + dynamic v1 shift from options/ft.yml.
        Explicit noise/timesteps allow reproducible objective/gradient audits.
        VAE encoding is frozen; transformer forward deliberately retains autograd.
        """
        self._check_condition(embeds, attention_mask)
        if not isinstance(targets, torch.Tensor) or targets.ndim != 4 or targets.shape[1] != 3:
            raise ValueError("image targets must be normalized RGB tensors (B, 3, H, W)")
        if targets.shape[0] != embeds.shape[0]:
            raise ValueError("image target and condition batch sizes must match")
        if (height is not None and height != targets.shape[-2]) or (
            width is not None and width != targets.shape[-1]
        ):
            raise ValueError("image output_spec height/width must match training targets")
        if not torch.isfinite(targets).all() or targets.min() < -1 or targets.max() > 1:
            raise ValueError("image targets must be finite and normalized to [-1, 1]")
        native = self._native(native_context, batch_size=embeds.shape[0])
        self._load()
        pipe = self._pipeline

        dtype, device = pipe.transformer.dtype, pipe.transformer.device
        if any(
            size % (pipe.vae_scale_factor * pipe.transformer.config.patch_size)
            for size in targets.shape[-2:]
        ):
            raise ValueError("image target H/W must be divisible by VAE scale times patch size")
        embeds = embeds.to(device=device, dtype=dtype)
        attention_mask = attention_mask.to(device=device)
        with torch.no_grad():
            posterior = pipe.vae.encode(targets.to(device=device, dtype=pipe.vae.dtype)).latent_dist
            clean = posterior.sample(generator=generator)
            if pipe.vae.config.shift_factor is not None:
                clean = clean - pipe.vae.config.shift_factor
            if pipe.vae.config.scaling_factor is not None:
                clean = clean * pipe.vae.config.scaling_factor
            clean = clean.to(dtype=dtype)
            refs = native.get("input_images")
            if refs is None:
                ref_latents = [None] * targets.shape[0]
            else:
                if targets.shape[0] > 1 and len(refs) != targets.shape[0]:
                    raise ValueError("training input_images must have one list per batch item")
                ref_latents = pipe.prepare_image(
                    images=refs,
                    batch_size=targets.shape[0],
                    num_images_per_prompt=1,
                    max_pixels=1024 * 1024,
                    max_side_length=1024,
                    device=device,
                    dtype=pipe.vae.dtype,
                )
        if noise is None:
            noise = torch.randn(clean.shape, device=device, dtype=dtype, generator=generator)
        else:
            noise = noise.to(device=device, dtype=dtype)
        if noise.shape != clean.shape:
            raise ValueError("flow noise must match encoded target latent shape")
        if timesteps is None:
            t = torch.randn((clean.shape[0],), device=device, generator=generator).sigmoid()
            tokens = (clean.shape[-2] // 2) * (clean.shape[-1] // 2)
            mu = 0.5 + (tokens - 256) * (1.15 - 0.5) / (4096 - 256)
            # Algebraic form of upstream transport.time_shift(mu, 1, t).
            t = t / (torch.exp(t.new_tensor(mu)) * (1 - t) + t)
        else:
            t = timesteps.to(device=device, dtype=torch.float32)
        if t.shape != (clean.shape[0],) or not torch.isfinite(t).all() or ((t < 0) | (t > 1)).any():
            raise ValueError("flow timesteps must be finite (B,) values in [0, 1]")
        t = t.to(dtype=dtype)
        mixing = t[:, None, None, None]
        noisy = mixing * clean + (1 - mixing) * noise
        velocity_target = clean - noise
        freqs_cis = self._rotary_embeddings()
        prediction = pipe.transformer(
            hidden_states=noisy,
            timestep=t,
            text_hidden_states=embeds,
            text_attention_mask=attention_mask,
            freqs_cis=freqs_cis,
            ref_image_hidden_states=ref_latents,
            return_dict=False,
        )
        if prediction.shape != velocity_target.shape:
            raise RuntimeError("OmniGen2 velocity prediction does not match target latent shape")
        return prediction, F.mse_loss(prediction.float(), velocity_target.float())

    def _rotary_embeddings(self):
        from omnigen2.models.transformers.repo import OmniGen2RotaryPosEmbed

        return OmniGen2RotaryPosEmbed.get_freqs_cis(
            self._pipeline.transformer.config.axes_dim_rope,
            self._pipeline.transformer.config.axes_lens,
            theta=10000,
        )
