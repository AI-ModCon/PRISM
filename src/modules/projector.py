"""Modality projection into the shared backbone dimension.

Stage 2 of the PRISM input path: ``ModalityProjector`` maps modality-specific
encoder features to ``d_model``, applies one of several configurable
normalization modes (the knob behind the projector ablation studies), and,
unless ``modality_embed_pos`` is ``"none"``, adds a learnable modality
embedding before or after that normalization. ``RMSNorm`` is the LLaMA/OLMo-style
normalization one of those modes selects.
"""

import logging
import math

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (used in LLaMA, OLMo)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Scale the input by its inverse root-mean-square, then by ``weight``.

        Args:
            x: Input activations of shape (B, T, D), where ``D`` is the ``dim``
                the module was constructed with.

        Returns:
            The normalized activations of shape (B, T, D). Unlike LayerNorm the
            mean is not subtracted and no bias is added.
        """
        # x: (B, T, D)
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        x = x / rms
        return x * self.weight


class ModalityProjector(nn.Module):
    """
    Stage 2: Token Harmonization & Projection.
    Projects modality-specific features to the shared model dimension
    and adds a learnable modality embedding.

    Configurable Normalization Modes (for ablation studies):
    - none: No normalization (simple MLP output)
    - layernorm: Standard LayerNorm on output (original/default behavior)
    - rmsnorm: RMSNorm (like LLaMA/OLMo, no centering)
    - l2_token: Per-token L2 normalization to target_norm
    - l2_sequence: Sequence-level L2 normalization (preserves token differences)
    - scale_only: Learned scalar multiplier only (no normalization)
    - match_text_stats: Match token norm distribution to text embeddings
    - match_text_elemstats: Match element-wise distribution to text embeddings

    Modality Embedding Position:
    - before_norm: Added before normalization (affects direction)
    - after_norm: Added after normalization (original behavior)
    - none: No modality embedding
    """

    VALID_NORM_MODES = {
        "none",
        "layernorm",
        "rmsnorm",
        "l2_token",
        "l2_sequence",
        "scale_only",
        "match_text_stats",
        "match_text_elemstats",
    }
    VALID_EMBED_POSITIONS = {"before_norm", "after_norm", "none"}

    # IsoFLOP capacity-ablation variant table. Each entry maps a label to
    # (hidden_mult, num_layers) for the per-modality scaling-law study.
    VARIANT_MAP: dict[str, tuple[int, int]] = {
        "BASE": (1, 2),
        "W2X":  (2, 2),
        "W4X":  (4, 2),
        "D2X":  (1, 4),
        "D4X":  (1, 8),
    }

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        norm_mode: str = "layernorm",
        target_norm: float = 0.25,
        modality_embed_pos: str = "after_norm",
        modality_embed_scale: float = 0.02,  # Initialization scale for modality embedding
        # Text statistics matching parameters
        text_norm_mean: float = 0.25,
        text_norm_std: float = 0.05,
        text_elem_mean: float = 0.0,
        text_elem_std: float = 0.006,
        norm_clip_min: float = 0.1,
        norm_clip_max: float = 0.5,
        # IsoFLOP capacity knobs. (1, 2) == legacy fc1/fc2 path (bit-identical
        # state_dict keys so existing checkpoints still load). Any other shape
        # uses a ModuleList — incompatible with old checkpoints by design.
        hidden_mult: int = 1,
        num_layers: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.norm_mode = norm_mode
        self.target_norm = target_norm
        self.modality_embed_pos = modality_embed_pos
        self.modality_embed_scale = modality_embed_scale
        self.hidden_mult = int(hidden_mult)
        self.num_layers = int(num_layers)

        # Text statistics for match_text_* modes
        self.text_norm_mean = text_norm_mean
        self.text_norm_std = text_norm_std
        self.text_elem_mean = text_elem_mean
        self.text_elem_std = text_elem_std
        self.norm_clip_min = norm_clip_min
        self.norm_clip_max = norm_clip_max

        # Validate configuration
        if norm_mode not in self.VALID_NORM_MODES:
            raise ValueError(
                f"Invalid norm_mode '{norm_mode}'. Must be one of {self.VALID_NORM_MODES}"
            )
        if modality_embed_pos not in self.VALID_EMBED_POSITIONS:
            raise ValueError(
                f"Invalid modality_embed_pos '{modality_embed_pos}'. Must be one of {self.VALID_EMBED_POSITIONS}"
            )
        if self.num_layers < 2:
            raise ValueError(
                f"ModalityProjector.num_layers must be >= 2 (got {self.num_layers})"
            )
        if self.hidden_mult < 1:
            raise ValueError(
                f"ModalityProjector.hidden_mult must be >= 1 (got {self.hidden_mult})"
            )

        if (self.hidden_mult, self.num_layers) == (1, 2):
            # Legacy path — preserve exact state_dict keys (fc1/fc2/act) so
            # existing checkpoints continue to load bit-identically.
            self.fc1 = nn.Linear(input_dim, d_model)
            self.act = nn.GELU()
            self.fc2 = nn.Linear(d_model, d_model)

            nn.init.xavier_uniform_(self.fc1.weight)
            nn.init.zeros_(self.fc1.bias)
            nn.init.xavier_uniform_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)
        else:
            # Capacity-ablation path. Hidden width = d_model * hidden_mult;
            # depth = num_layers Linears with GELU between (no activation
            # after the final projection). State-dict keys are `layers.0`,
            # `layers.2`, ... and incompatible with legacy checkpoints by
            # design (this only fires for ablation variants).
            h = d_model * self.hidden_mult
            dims = (
                [(input_dim, h)]
                + [(h, h)] * (self.num_layers - 2)
                + [(h, d_model)]
            )
            layers: list[nn.Module] = []
            for i, (in_dim, out_dim) in enumerate(dims):
                linear = nn.Linear(in_dim, out_dim)
                nn.init.xavier_uniform_(linear.weight)
                nn.init.zeros_(linear.bias)
                layers.append(linear)
                if i < len(dims) - 1:
                    layers.append(nn.GELU())
            self.layers = nn.ModuleList(layers)

        # Initialize normalization layer based on mode
        self.final_norm = None
        self.output_scale = None

        if norm_mode == "layernorm":
            self.final_norm = nn.LayerNorm(d_model)
        elif norm_mode == "rmsnorm":
            self.final_norm = RMSNorm(d_model)
        elif norm_mode in ["l2_token", "l2_sequence", "scale_only"]:
            # Learnable scale parameter
            self.output_scale = nn.Parameter(torch.ones(1) * target_norm)
        # match_text_stats and match_text_elemstats don't need additional parameters

        # Modality embedding (if enabled)
        if modality_embed_pos != "none":
            # Use configurable scale for modality embedding initialization
            # Default 0.02 gives norm ~0.9 for d_model=2048
            # For text-matched norms (~0.25), use scale ~0.0055
            self.modality_embedding = nn.Parameter(
                torch.randn(1, 1, d_model) * modality_embed_scale
            )
            # Log the expected norm for debugging
            expected_norm = math.sqrt(d_model) * modality_embed_scale
            logger.info(
                f"  Modality embedding: scale={modality_embed_scale:.4f}, expected_norm={expected_norm:.4f}"
            )
        else:
            self.register_parameter("modality_embedding", None)

        logger.info(
            f"ModalityProjector initialized: norm_mode={norm_mode}, target_norm={target_norm}, embed_pos={modality_embed_pos}"
        )
        if norm_mode in ["match_text_stats", "match_text_elemstats"]:
            logger.info(
                f"  Text stats: norm_mean={text_norm_mean}, norm_std={text_norm_std}, "
                f"elem_mean={text_elem_mean}, elem_std={text_elem_std}, clip=[{norm_clip_min}, {norm_clip_max}]"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with configurable normalization.

        Args:
            x: (B, T_m, D_m) - e.g., (B, 196, 768) for ViT patches

        Returns:
            (B, T_m, d_model) - Projected embeddings
        """
        # MLP projection
        if hasattr(self, "layers"):
            # Capacity-ablation path (hidden_mult != 1 or num_layers != 2).
            # `self.layers` interleaves Linear/GELU/Linear/GELU/.../Linear.
            for layer in self.layers:
                x = layer(x)
        else:
            x = self.fc1(x)
            x = self.act(x)
            x = self.fc2(x)  # (B, T_m, d_model)

        # Add modality embedding BEFORE normalization (if configured)
        if self.modality_embed_pos == "before_norm" and self.modality_embedding is not None:
            x = x + self.modality_embedding

        # Apply normalization based on mode
        x = self._apply_normalization(x)

        # Add modality embedding AFTER normalization (if configured)
        if self.modality_embed_pos == "after_norm" and self.modality_embedding is not None:
            x = x + self.modality_embedding

        return x

    def _apply_normalization(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the configured normalization mode."""

        if self.norm_mode == "none":
            # No normalization - raw MLP output
            return x

        elif self.norm_mode == "layernorm":
            # Standard LayerNorm (original behavior)
            return self.final_norm(x)

        elif self.norm_mode == "rmsnorm":
            # RMSNorm (like LLaMA/OLMo)
            return self.final_norm(x)

        elif self.norm_mode == "l2_token":
            # Per-token L2 normalization: each token has exactly target_norm
            x_norm = x.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            x = x / x_norm  # Unit norm per token
            x = x * self.output_scale  # Scale to target
            return x

        elif self.norm_mode == "l2_sequence":
            # Sequence-level L2 normalization: preserves relative token differences
            # Normalize so the mean token norm equals target_norm
            token_norms = x.norm(dim=-1, keepdim=True)  # (B, T, 1)
            mean_norm = token_norms.mean(dim=1, keepdim=True).clamp(min=1e-6)  # (B, 1, 1)
            x = x / mean_norm * self.output_scale
            return x

        elif self.norm_mode == "scale_only":
            # Just a learned scalar multiplier, no normalization
            return x * self.output_scale

        elif self.norm_mode == "match_text_stats":
            # Match token norm distribution to text embedding statistics
            # This preserves relative token differences while matching the overall norm distribution
            token_norms = x.norm(dim=-1, keepdim=True)  # (B, T, 1)

            # Compute current statistics
            current_mean = token_norms.mean()
            current_std = token_norms.std().clamp(min=1e-6)

            # Standardize and rescale to match text distribution
            normalized_norms = (token_norms - current_mean) / current_std
            target_norms = normalized_norms * self.text_norm_std + self.text_norm_mean

            # Clip to prevent extreme values
            target_norms = target_norms.clamp(min=self.norm_clip_min, max=self.norm_clip_max)

            # Scale each token to target norm while preserving direction
            x = x / token_norms.clamp(min=1e-6) * target_norms
            return x

        elif self.norm_mode == "match_text_elemstats":
            # Match element-wise distribution to text embedding statistics
            # Similar to LayerNorm but scaled to match text mean/std instead of 0/1

            # Compute current element-wise statistics
            current_mean = x.mean(dim=-1, keepdim=True)
            current_std = x.std(dim=-1, keepdim=True).clamp(min=1e-6)

            # Standardize to zero mean, unit variance
            x_normalized = (x - current_mean) / current_std

            # Rescale to match text element statistics
            x = x_normalized * self.text_elem_std + self.text_elem_mean

            # The above gives us element-wise matching, but we should also check
            # that the resulting token norms are reasonable
            token_norms = x.norm(dim=-1, keepdim=True)

            # Clip norms if they're outside expected range
            scale_factor = torch.ones_like(token_norms)
            too_small = token_norms < self.norm_clip_min
            too_large = token_norms > self.norm_clip_max

            scale_factor = torch.where(
                too_small,
                self.norm_clip_min / token_norms.clamp(min=1e-6),
                scale_factor,
            )
            scale_factor = torch.where(
                too_large,
                self.norm_clip_max / token_norms.clamp(min=1e-6),
                scale_factor,
            )

            x = x * scale_factor
            return x

        else:
            raise ValueError(f"Unknown norm_mode: {self.norm_mode}")

    def get_debug_stats(self, x: torch.Tensor) -> dict:
        """
        Compute debug statistics for logging.
        Call this on the OUTPUT of forward() for accurate stats.
        """
        with torch.no_grad():
            token_norms = x.norm(dim=-1)  # (B, T)
            return {
                "norm_mean": token_norms.mean().item(),
                "norm_std": token_norms.std().item(),
                "norm_min": token_norms.min().item(),
                "norm_max": token_norms.max().item(),
                "elem_std": x.std().item(),
                "scale": self.output_scale.item() if self.output_scale is not None else None,
            }
