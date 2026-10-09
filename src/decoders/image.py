"""Image output through a learned connector and a pretrained OmniGen2 generator.

The connector is new PRISM capacity, not a pretrained image head.  ``reference``
mode bypasses it and retains the official native conditioner.  Optional image
dependencies and checkpoint weights are loaded only when a backend is used.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from ..connectors import (
    BackboneFeatures,
    DecoderContext,
    build_bridge,
    build_readout,
    pack_right_padded,
)
from .base import OutputDecoder
from .types import DecoderCondition


def _generator_options(generator, **legacy) -> dict:
    """Normalize the nested API without changing legacy constructor defaults."""
    if generator is None:
        options = {key: value for key, value in legacy.items() if value is not None}
    else:
        if not isinstance(generator, Mapping):
            raise TypeError("generator configuration must be a mapping")
        extra = set(generator) - {"type", *legacy}
        if extra:
            raise ValueError(f"Unknown generator configuration fields: {sorted(extra)}")
        if generator.get("type", "omnigen2") != "omnigen2":
            raise ValueError("Unsupported generator type; only 'omnigen2' is implemented")
        if any(value is not None for value in legacy.values()):
            raise ValueError(
                "Cannot mix nested generator configuration with flat generator options"
            )
        options = {key: value for key, value in generator.items() if key != "type"}
    options.setdefault("model_id", "OmniGen2/OmniGen2")
    options.setdefault("revision", None)
    options.setdefault("local_files_only", True)
    options.setdefault("conditioning_dim", None)
    if not isinstance(options["model_id"], str) or not options["model_id"]:
        raise ValueError("generator model_id must be a nonempty string")
    if type(options["local_files_only"]) is not bool:
        raise ValueError("generator local_files_only must be bool")
    dim = options["conditioning_dim"]
    if dim is not None and (type(dim) is not int or dim < 1):
        raise ValueError("generator conditioning_dim must be a positive integer")
    return options


class ImageDecoder(OutputDecoder):
    """Image output through a learned connector into a pretrained generator.

    Reads all valid backbone states, bridges them to the generator's
    conditioning width, packs them right-padded, and hands them to a generator
    backend (an ``OmniGen2Backend`` unless one is injected). The connector is
    the only trainable part by default; the pretrained generator is frozen
    unless ``configure_training`` opts in.

    Two modes are selected through ``output_spec["mode"]`` (or a per-call
    keyword): ``"prism"`` (the default) conditions the generator on the
    connector output, and ``"reference"`` bypasses the connector and uses the
    backend's own native prompt conditioner.
    """

    output_kind = "image"
    loss_kind = "flow_matching"
    response_encoding = "png"
    # The generator backend is duck-typed: an ``nn.Module`` that additionally
    # exposes ``generate_reference`` / ``generate_conditioned`` /
    # ``training_step`` (validated at call time, as the getattr probes below
    # do for the optional parts). Declaring it keeps those lookups off
    # ``nn.Module.__getattr__``, which reports submodules as ``Tensor | Module``.
    backend: Any

    def __init__(
        self,
        d_model: int,
        backend: nn.Module | None = None,
        model_id: str | None = None,
        revision: str | None = None,
        local_files_only: bool | None = None,
        conditioning_dim: int | None = None,
        readout: Mapping | None = None,
        bridge: Mapping | None = None,
        generator: Mapping | None = None,
    ) -> None:
        super().__init__()
        # The shared factories also serve scientific heads. Keep this route's
        # accepted architecture and reported conditioning contract unchanged.
        if isinstance(readout, Mapping) and readout.get("type", "select") != "select":
            raise ValueError("Image decoder requires the all-valid select readout")
        if isinstance(bridge, Mapping) and bridge.get("type", "layernorm_linear") != "layernorm_linear":
            raise ValueError("Image decoder requires a layernorm_linear bridge")
        # The readout is parameter-free, preserving legacy initialization RNG.
        self.readout = build_readout(readout, d_model=d_model)
        generator_options = _generator_options(
            generator,
            model_id=model_id,
            revision=revision,
            local_files_only=local_files_only,
            conditioning_dim=conditioning_dim,
        )
        conditioning_dim = generator_options["conditioning_dim"]
        if backend is None:
            from .omnigen2_backend import DEFAULT_REVISION, OmniGen2Backend

            backend = OmniGen2Backend(
                model_id=generator_options["model_id"],
                revision=generator_options["revision"] or DEFAULT_REVISION,
                local_files_only=generator_options["local_files_only"],
                conditioning_dim=conditioning_dim or 2048,
            )
        if not isinstance(backend, nn.Module):
            raise TypeError("image backend must be an nn.Module")
        inferred_dim = getattr(backend, "conditioning_dim", None)
        conditioning_dim = conditioning_dim or inferred_dim
        if not isinstance(conditioning_dim, int) or conditioning_dim < 1:
            raise ValueError("conditioning_dim must be set or provided by the backend")
        if inferred_dim is not None and inferred_dim != conditioning_dim:
            raise ValueError("conditioning_dim does not match the backend")
        self.d_model = d_model
        self.conditioning_dim = conditioning_dim
        # Keep connector.0/1 keys and tensor-forward for audited old checkpoints
        # and feature-alignment runners. No duplicate parameter registration.
        self.connector = build_bridge(bridge, input_dim=d_model, output_dim=conditioning_dim)
        self.backend = backend
        if getattr(self.backend, "train_diffusion", False):
            self.backend.train(self.training)
        else:
            self.backend.requires_grad_(False)
            self.backend.eval()

    def configure_training(
        self, *, train_diffusion: bool = False, gradient_checkpointing: bool = False
    ) -> ImageDecoder:
        """Train the connector and optionally the full pretrained diffusion model.

        Default construction and ``configure_training()`` retain connector-only
        behavior. No pretrained weights are allocated here. Configure after any
        model-wide freezing and before building the optimizer.
        """
        configure = getattr(self.backend, "configure_training", None)
        if not callable(configure):
            raise TypeError("image backend does not expose explicit training configuration")
        configure(train_diffusion=train_diffusion, gradient_checkpointing=gradient_checkpointing)
        self.connector.requires_grad_(True)
        self.train(self.training)
        return self

    def train(self, mode: bool = True) -> ImageDecoder:
        """Set training mode, holding the generator in eval mode unless opted in.

        This changes module modes only, never ``requires_grad``: the generator
        follows ``mode`` once ``configure_training(train_diffusion=True)`` has
        been called, and is forced to eval otherwise.

        Args:
            mode: whether to put this module in training mode. Default: ``True``.

        Returns:
            ``self``, as for ``nn.Module.train``.
        """
        super().train(mode)
        # Only explicitly opted-in diffusion weights follow the parent's mode.
        self.backend.train(mode if getattr(self.backend, "train_diffusion", False) else False)
        return self

    def conditioning_contract(self) -> dict:
        """Describe the implemented route independently of checkpoint file paths."""
        return {
            "schema_version": 1,
            "readout": {"type": "select", "layers": "final", "positions": "all_valid"},
            "bridge": {
                "type": "layernorm_linear",
                "input_dim": self.d_model,
                "output_dim": self.conditioning_dim,
            },
            "layout": {"type": "right_padded", "preserves_valid_order": True},
            "generator": {"type": "omnigen2", "conditioning_dim": self.conditioning_dim},
        }

    def prepare_condition(self, condition: DecoderCondition) -> DecoderContext:
        """Read source states, bridge their representation, and pack generator inputs."""
        features = BackboneFeatures(
            hidden_states=condition.hidden_states,
            attention_mask=condition.attention_mask,
            modality_spans=condition.modality_spans,
            provenance=condition.provenance,
        )
        context = pack_right_padded(self.connector.connect(self.readout(features)))
        context.provenance["conditioning_contract"] = self.conditioning_contract()
        return context

    def connect(self, condition: DecoderCondition) -> tuple[torch.Tensor, torch.Tensor]:
        """Compatibility tuple API for existing generation and diagnostic runners."""
        context = self.prepare_condition(condition)
        return context.tokens, context.attention_mask

    @staticmethod
    def _options(condition: DecoderCondition, kwargs: dict) -> tuple[str, dict]:
        options = dict(condition.output_spec)
        options.update(kwargs)
        mode = options.pop("mode", "prism")
        if mode not in ("prism", "reference"):
            raise ValueError("image mode must be 'prism' or 'reference'")
        return mode, options

    @torch.no_grad()
    def generate_condition(self, condition: DecoderCondition, **kwargs: Any) -> Any:
        """Sample an image from the backend.

        Args:
            condition: backbone states plus their attention mask, the
                ``native_context`` handed to the backend, and an
                ``output_spec`` of generation options.
            **kwargs: generation options that override ``output_spec``,
                including ``mode`` (``"prism"`` or ``"reference"``).

        Returns:
            The backend's native sampling output — for OmniGen2, PIL images
            unless a different ``output_type`` was requested.

        Raises:
            ValueError: if ``mode`` is neither ``"prism"`` nor ``"reference"``.
        """
        mode, options = self._options(condition, kwargs)
        if mode == "reference":
            return self.backend.generate_reference(dict(condition.native_context), **options)
        embeds, mask = self.connect(condition)
        return self.backend.generate_conditioned(
            embeds, mask, native_context=dict(condition.native_context), **options
        )

    def forward_condition(
        self, condition: DecoderCondition, targets: torch.Tensor | None = None, **kwargs: Any
    ) -> tuple[Any, torch.Tensor | None]:
        """Train the connector against image targets, or sample when unsupervised.

        Args:
            condition: backbone states, attention mask, ``native_context`` and
                ``output_spec`` (see ``generate_condition``).
            targets: normalized RGB image targets ``(B, 3, H, W)``, or ``None``
                to sample instead of training. Default: ``None``.
            **kwargs: generation/training options that override ``output_spec``.

        Returns:
            ``(prediction, loss)``. With targets this is the backend's
            ``training_step`` result — for OmniGen2, a latent velocity
            prediction and its flow-matching loss. Without targets it is the
            sampled output paired with ``None``.

        Raises:
            ValueError: if targets are supplied while ``mode != "prism"``.
        """
        if targets is None:
            return self.generate_condition(condition, **kwargs), None
        mode, options = self._options(condition, kwargs)
        if mode != "prism":
            raise ValueError("image connector training requires mode='prism'")
        embeds, mask = self.connect(condition)
        # Autograd trains the connector through the generator, and its weights on opt-in.
        return self.backend.training_step(
            embeds, mask, targets, native_context=dict(condition.native_context), **options
        )

    def forward(self, hidden_states: torch.Tensor, targets=None, **kwargs: Any):
        """Wrap plain tensors into a ``DecoderCondition`` and run the typed path.

        Args:
            hidden_states: backbone states ``(B, T, d_model)``.
            targets: normalized RGB image targets ``(B, 3, H, W)``, or ``None``
                to sample instead of training. Default: ``None``.
            **kwargs: ``attention_mask`` (defaults to all-valid),
                ``native_context`` and ``output_spec`` (each defaulting to
                ``{}``) populate the condition; the rest are forwarded as
                generation/training options.

        Returns:
            ``(prediction, loss)`` from ``forward_condition``; ``loss`` is
            ``None`` when ``targets`` is ``None``.
        """
        mask = kwargs.pop("attention_mask", None)
        if mask is None:
            mask = torch.ones(
                hidden_states.shape[:2], device=hidden_states.device, dtype=torch.bool
            )
        condition = DecoderCondition(
            hidden_states=hidden_states,
            attention_mask=mask,
            native_context=kwargs.pop("native_context", {}),
            output_spec=kwargs.pop("output_spec", {}),
        )
        return self.forward_condition(condition, targets=targets, **kwargs)

    @torch.no_grad()
    def generate(self, hidden_states: torch.Tensor, **kwargs: Any):
        """Sample an image from plain tensors.

        Args:
            hidden_states: backbone states ``(B, T, d_model)``.
            **kwargs: as for ``forward``.

        Returns:
            The backend's native sampling output — the prediction half of
            ``forward``, called with no targets.
        """
        return self.forward(hidden_states, **kwargs)[0]
