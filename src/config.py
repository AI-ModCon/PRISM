"""Dataclass configuration for PRISM training and model construction.

Two dataclasses own the knobs the framework reads: ``TrainingConfig`` for the
optimization loop and ``ModelConfig`` for the architecture that
``UnifiedTransformer`` builds. ``PRISM_CONFIGS`` maps a preset name (e.g.
``"prism-nano"``) to a dict of ``ModelConfig`` fields, and
``ModelConfig.from_preset`` turns one into a config object.

The Hydra tree under ``src/conf/`` is the other entry point: ``src/train.py``
reads it and passes the fields through to these dataclasses explicitly, so a
new field has to be wired up in both places.
"""

import os
from dataclasses import dataclass, field
from typing import Any

from .modalities import ALL_MODALITIES, Modality, parse_modality

# ts_projector values whose encoder accepts heterogeneous per-sample lengths
# (list[Tensor(T_i,V_i)]) instead of a single pre-padded Tensor(B,T,V); the
# data pipeline must skip its fixed max_ts_length pad/truncate step for these.
DYNAMIC_LENGTH_TS_PROJECTORS = ("timeomni", "intern_s2", "intern_s2_397b")


def _default_device() -> str:
    # torch is deliberately not a declared dependency (HPC installs use the
    # system build, every install here is --no-deps), so importing this module
    # must not require it — `prism --help` goes through it. Resolved per
    # instance rather than once at import, which is fine: nothing constructs
    # TrainingConfig in a hot loop.
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TrainingConfig:
    """Hyperparameters and run settings for the PRISM training loop.

    Covers the optimizer and scheduler, checkpoint/resume paths, Weights &
    Biases logging, the periodic eval and visualization gates, the
    differential (per-zone) learning rates and warmups, the freeze switches
    for the LLM and the vision tower, and the CALVIN VLA data path.

    ``src/train.py`` populates it -- mostly from the Hydra ``training``
    group, with ``wandb_*`` from ``wandb``, ``sweep_id``/``preset`` from
    ``exp`` and ``output_dir`` from the Hydra run directory. ``ZoneATrainer``,
    ``ZoneAVLATrainer`` and ``train_native_ddp`` all read it. Many fields
    carry an inline or preceding comment; the rest are named by their
    identifier alone.
    """

    task: str = "vlm"
    batch_size: int = 4
    learning_rate: float = 1e-4
    max_steps: int = 100
    finite_epoch_steps: int = 0  # Reset finite iterable datasets every N optimizer steps
    log_every_n_steps: int = 10
    eval_every_n_steps: int = 100  # Run eval every 100 steps
    # Master gate for periodic in-training evaluation. Default false so
    # existing runs are bit-identical (the eval block has been disabled in
    # trainer_zone_a for OOM reasons). IsoFLOP runs flip this on to emit
    # per-family eval loss into perf.jsonl.
    eval_enabled: bool = False
    validation_batches: int = 8  # Number of validation batches per eval event
    viz_every_n_steps: int = 500  # Run batch visualization every 500 steps
    warmup_steps: int = 10
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    device: str = field(default_factory=_default_device)
    wandb_project: str = "prism-training"
    wandb_entity: str | None = None
    wandb_mode: str = "online"
    wandb_run_name: str = "zone-a-debug"
    wandb_run_id: str | None = None
    save_every_n_steps: int = 1000  # Default save interval
    resume_from_checkpoint: str | None = None  # Path to checkpoint directory to resume from
    # Model-only resume: load weights but create fresh optimizer/scheduler.
    # Use for stage transitions (e.g., projector-only -> E2E fine-tuning).
    # When set, resume_from_checkpoint is ignored.
    resume_weights_only: str | None = None
    output_dir: str = "checkpoints"  # Default output directory
    vocab_size: int = 30522  # Default, should be updated

    # Sweep metadata — populated by run_sweep.py via exp.sweep_id / exp.preset
    # Hydra overrides. Surfaces in perf.jsonl so tools/perf_aggregate.py can
    # group multiple runs.
    sweep_id: str | None = None
    preset: str | None = None
    # Resolved system.seed (default 42 in src/conf/config.yaml). Stamped into
    # perf.jsonl's startup_param_count event so the IsoFLOP collector can group
    # variance-floor replicas.
    seed: int = 42

    # Molmo / Advanced Optimization
    scheduler_type: str = "cosine"  # "cosine", "cosine_with_min_lr", "molmo_layered", "wsd"
    min_lr_ratio: float = 0.1  # Floor for cosine decay (0.0 to 1.0)
    wsd_decay_ratio: float = 0.1  # Fraction of max_steps used for WSD decay; 0 disables decay
    wsd_decay_steps: int | None = None  # Explicit WSD decay window; 0 disables decay

    # Differential Learning Rates
    lr_connector: float | None = None  # If None, uses base learning_rate
    lr_vit: float | None = None
    lr_llm: float | None = None
    lr_dna: float | None = None  # DNA encoder LR (typically 10x lower)

    # Differential Warmup
    warmup_steps_connector: int = 200
    warmup_steps_main: int = 2000

    # Freezing Overrides
    freeze_llm: bool = True
    freeze_vit: bool = True
    freeze_connector: bool = False  # If True, projector is loaded but not updated (Stage 2)
    data_num_workers: int = 4
    calvin_root: str | None = os.environ.get(
        "PRISM_CALVIN_ROOT", "/flare/<project>/<user>/vla_training/calvin_dataset"
    )
    calvin_split: str = "train"
    calvin_max_chunk: int = 20
    calvin_text_max_length: int = 128
    calvin_strict_integrity: bool = False
    calvin_max_skipped_fraction: float = 1.0
    # CALVIN data path selector. "map" = the legacy CalvinVLADataset;
    # "webdataset" = ModalityAwareWebDatasetWrapper over shards from
    # applications/vla/shard_calvin_vla.py. VLA is Accelerate-only on both paths.
    calvin_loader: str = "map"
    # Override the dataset-group `path` for one-off smokes against /tmp.
    calvin_webdataset_root: str | None = None
    calvin_webdataset_storage: str = "lustre"  # or "daos"

    # BioReason DNA/KEGG training knobs
    bioreason_dataset: str | None = None
    dna_truncation_per_side: int = 1024
    max_dna_length: int = 1024
    kegg_eval_max_examples: int | None = None
    use_reasoning_traces: bool = True
    use_class_weights: bool = False
    class_weight_max: float = 10.0
    # LoRA (applied to the LLM backbone via apply_lora_torchtune — see
    # src/utils/lora_utils.py). Used by ZoneATrainer and ZoneDTrainer.
    lora_enabled: bool = False
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: list[str] | None = None  # Auto-detected if None
    # Alpha to use when merging a resumed SFT-stage LoRA adapter into the base
    # weights before creating fresh GRPO-rank LoRALinear layers (the two
    # stages' adapters aren't rank-compatible for direct loading). Defaults to
    # lora_alpha if unset.
    resume_lora_alpha: float | None = None
    # GRPO: when resume_weights_only is set but no lora_adapter.pt exists at
    # that path, the default is to fail fast (starting GRPO from an
    # untrained base is a costly silent failure, not a recoverable one).
    # Set True only for the rare intentional case of starting GRPO from
    # base weights with no prior SFT adapter.
    allow_missing_resume_adapter: bool = False

    # GRPO Configuration
    grpo_num_generations: int = 8
    grpo_max_completion_length: int = 800
    grpo_temperature: float = 1.0
    grpo_top_p: float = 0.95
    grpo_top_k: int = 20
    grpo_beta: float = 0.0
    grpo_reward_functions: list[str] | None = None


@dataclass
class ModelConfig:
    """Architecture and modality settings for ``UnifiedTransformer``.

    A single dataclass describes the whole model: the shared width
    ``d_model`` and the per-modality input dims (``d_text``, ``d_img``, ...),
    which modalities are active, the optional Hugging Face causal-LM backbone
    (``llm_backbone_id``) and what stays frozen, the ``ModalityProjector``
    normalization and capacity ablation knobs, the time-series encoder
    selection, and which output decoders get built.

    ``__post_init__`` coerces ``modalities`` to ``Modality`` members,
    normalizes and validates ``decoder_loss_weights``, and maps the legacy
    ``is_vla=True`` flag onto ``output_decoders = ["text", "action"]``.

    Where a field needs explaining it carries an inline or preceding
    comment; the rest are named by their identifier alone. The named entries
    in ``PRISM_CONFIGS`` are partial field dicts for common bases -- note
    that the smaller ones set only ``hf_model_id`` and leave
    ``llm_backbone_id`` at ``None``, so they build the local stack rather
    than an HF backbone.
    """

    # Dimensions
    d_model: int = 1280  # Default to ~0.6B (prism-nano)
    vocab_size: int = 65536  # NanoChat Tokenizer

    # Modality Input Dimensions (Simulated)
    d_text: int = 1280  # Will be updated by preset
    d_table: int = 768
    d_ts: int = 512
    d_img: int = 1152  # SigLIP-large native dim (use 768 for SigLIP2-base)
    d_geo: int = 512
    d_graph: int = 768
    d_dna: int = 1024  # Nucleotide Transformer hidden dim

    # DNA encoder settings
    dna_model_name: str = "InstaDeepAI/nucleotide-transformer-2.5b-multi-species"
    dna_is_evo2: bool = False  # If True, DNAEncoder loads an Evo2 checkpoint instead of NT

    # Image encoder / processor
    image_encoder_id: str = "google/siglip2-base-patch16-224"
    image_processor_id: str | None = None
    image_processor_strict: bool = False
    image_size: int = 224
    image_mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    image_std: tuple[float, float, float] = (0.5, 0.5, 0.5)

    # Transformer / MoE
    num_layers: int = 20
    num_heads: int = 10
    num_experts: int = 8
    num_experts_per_token: int = 2
    dropout: float = 0.1
    max_seq_len: int = 8192
    mlp_type: str = "swiglu"  # "swiglu" (Llama), "gelu" (Pythia), "relusquared" (NanoChat)
    hf_model_id: str | None = None  # Legacy: For weight initialization of custom blocks

    # HF Backbone Support
    llm_backbone_id: str | None = None  # If set, wraps this AutoModelForCausalLM as the decoder
    llm_tokenizer_id: str | None = (
        None  # If set, uses this tokenizer (overrides backbone if both set)
    )
    freeze_backbone: bool = True  # If True, freezes the backbone weights
    freeze_encoders: bool = True  # If True, freezes all modality encoders (Zone 1)
    text_dropout: float = 0.0  # Text-Only Dropout (Input Masking)
    is_vla: bool = False
    pose_dim: int = 15
    action_dim: int = 7

    # === Output decoders ===
    # Which output decoders the model builds, by registry name (see
    # src/decoders/__init__.py DECODERS). Default ["text"] reproduces the
    # legacy text-only behavior exactly. is_vla=True is mapped to
    # ["text", "action"] in __post_init__ for back-compat — old configs need
    # not set this field.
    output_decoders: list[str] = field(default_factory=lambda: ["text"])
    # Per-decoder kwargs, keyed by decoder name (e.g. {"time_series": {...}}).
    decoder_configs: dict = field(default_factory=dict)
    decoder_loss_weights: dict[str, float] = field(default_factory=dict)

    # Attention implementation for HF backbone. Options: "sdpa", "eager", "flash_attention_2"
    # On Intel XPU, "sdpa" uses F.scaled_dot_product_attention which has a known
    # UR_RESULT_ERROR_OUT_OF_RESOURCES leak in the Level Zero runtime. Use "eager"
    # (manual matmul+softmax) as a workaround for affected models (e.g., LlamaForCausalLM).
    # Models with custom attention (e.g., OLMo-3) are typically unaffected.
    attn_implementation: str = "sdpa"
    # When enabled, mask padded variable-length modality tokens in HF attention.
    # Keep disabled on XPU unless the selected attention implementation has been
    # validated: nontrivial masks can re-enable a known resource-heavy path.
    mask_padded_modality_tokens: bool = False

    # Modality specific settings
    # Typed as list[Modality] so the enum is the source of truth. YAML/CLI
    # strings are coerced to Modality in __post_init__. Because
    # Modality(str, Enum), comparisons like `Modality.IMAGE in modalities`
    # work whether elements are enum members or plain strings.
    modalities: list[Modality] = field(default_factory=lambda: list(ALL_MODALITIES))

    # Timeseries specific
    is_timeseries: bool = False
    ts_projector: str = "linear"  # "linear", "moirai", "intern_s2", "intern_s2_397b", or "timeomni"
    ts_encoder_id: str = "Salesforce/moirai-2.0-R-small"
    # If False, builds the moirai/intern_s2/intern_s2_397b encoder architecture from its
    # config but skips loading pretrained weights (random init, for ablation studies).
    ts_load_pretrained: bool = True
    intern_s2_sampling_rate: float = 1.0
    # Moirai and linear both only handle univariate.
    # We handle multivariate by "flatting" the time series (T_ts, Num_Vars) -> (T_ts * Num_Vars) and projecting to d_ts.
    ts_variates: int = 1
    max_ts_length: int = (
        512  # Maximum length of time series data (T_ts) that the model can handle per instance
    )
    # Whether to apply scale-loc normalization in the TimeSeriesEncoder (for Moirai, necessary due to patch-specific normalization)
    # or in the StreamingDataset (for general projectors, e.g., linear)
    normalize_ts_in_encoder: bool = True
    # TimeOmni-specific patching knobs. For ts_projector="timeomni", max_ts_length
    # is treated as a dynamic budget computed from these values (see train.py):
    #   max_ts_length = max(timeomni_stride or timeomni_patch_len) * (timeomni_max_patches - 1)
    timeomni_patch_len: int | list[int] = 16
    timeomni_stride: int | list[int] | None = None
    timeomni_d_model: int = 512
    timeomni_dropout: float = 0.1
    timeomni_ts_tokens: int = 100
    timeomni_max_patches: int = 100

    # Forecast horizon (number of future steps) for the time_series output
    # decoder (Phase 2). Only used when "time_series" is in output_decoders;
    # Overridable via decoder_configs["time_series"]["generator"]["horizon"]
    # or the legacy flat decoder_configs["time_series"]["horizon"].
    ts_forecast_horizon: int = 96

    # Interleave specific
    is_interleaved_qa: bool = (
        False  # If True, use interleaved modality encoding for questions, text-only answers
    )
    modality_start_end_token_indices: dict | None = (
        None  # Dict mapping modality name to (start_token_id, end_token_id) for interleaving
    )

    # Merged-length guard (issue #123). When set, the interleave-QA merge path
    # (_merge_text_input_ids_with_modality_embeds) checks the assembled merged
    # sequence length against this limit BEFORE the backbone forward, converting
    # a silent XPU GPU page-fault (see #120) into an actionable Python error.
    # Covers only merges that flow through that path (interleaved QA); the
    # non-interleaved concat path in forward() does not reach this check.
    # None = disabled (default), so non-training callers/tests are unaffected.
    max_merged_seq_length: int | None = None
    # Guard behavior when the limit is exceeded: "error" (raise, default in
    # training) or "warn" (log once and proceed).
    merged_seq_length_guard: str = "error"

    # === Projector Configuration (Ablation-Ready) ===
    # Normalization mode for ModalityProjector
    # Options: "none", "layernorm", "rmsnorm", "l2_token", "l2_sequence", "scale_only",
    #          "match_text_stats", "match_text_elemstats"
    #   - none: No normalization (simple MLP output)
    #   - layernorm: Standard LayerNorm on output (original behavior)
    #   - rmsnorm: RMSNorm (like LLaMA/OLMo, no centering)
    #   - l2_token: Per-token L2 normalization to target_norm
    #   - l2_sequence: Sequence-level L2 normalization (preserves token differences)
    #   - scale_only: Learned scalar multiplier only (no normalization)
    #   - match_text_stats: Match token norm distribution to text (mean/std of norms)
    #   - match_text_elemstats: Match element-wise distribution to text (like LayerNorm but scaled)
    projector_norm_mode: str = "layernorm"  # Default: original behavior

    # Target norm for l2_token, l2_sequence, and scale_only modes
    projector_target_norm: float = 0.25

    # Where to add the modality embedding
    # Options: "before_norm", "after_norm", "none"
    projector_modality_embed_pos: str = "after_norm"  # Default: original behavior

    # Modality embedding initialization scale
    # Default 0.02 gives embedding norm ~0.9 for d_model=2048, which is 3.5x larger than text norms!
    # For text-matched norms, use: target_norm / sqrt(d_model) ≈ 0.25/45.25 ≈ 0.0055
    projector_modality_embed_scale: float = 0.02  # Default: original behavior (norm ~0.9)

    # === Text Statistics Matching (for match_text_stats and match_text_elemstats modes) ===
    # These values are based on observed OLMo-1B text embedding statistics
    projector_text_norm_mean: float = 0.25  # Target mean of token norms
    projector_text_norm_std: float = 0.05  # Target std of token norms
    projector_text_elem_mean: float = 0.0  # Target mean of elements (for elemstats)
    projector_text_elem_std: float = 0.006  # Target std of elements (for elemstats)
    projector_norm_clip_min: float = 0.1  # Min norm after matching (prevents near-zero)
    projector_norm_clip_max: float = 0.5  # Max norm after matching (prevents outliers)

    # IsoFLOP capacity knobs. (1, 2) reproduces the legacy 2-layer MLP
    # projector exactly (state-dict bit-compatible). Any other pair selects
    # the ablation path in ModalityProjector (hidden width = d_model * mult,
    # depth = num_layers). Variants documented in ModalityProjector.VARIANT_MAP.
    projector_hidden_mult: int = 1
    projector_num_layers: int = 2

    def __post_init__(self) -> None:
        # Coerce YAML / CLI / preset strings (and OmegaConf ListConfig elements)
        # into Modality enum members. Hydra structured configs hand us plain
        # strings even when the field is declared list[Modality]; presets
        # below also use string literals.
        self.modalities = [parse_modality(m) for m in self.modalities]

        # Normalize output_decoders (OmegaConf hands us a ListConfig of str).
        self.output_decoders = [str(d) for d in self.output_decoders]
        import math

        self.decoder_loss_weights = {str(k): float(v) for k, v in self.decoder_loss_weights.items()}
        if any(not math.isfinite(v) or v < 0 for v in self.decoder_loss_weights.values()):
            raise ValueError("decoder_loss_weights must be finite and nonnegative")
        if set(self.decoder_loss_weights) - set(self.output_decoders):
            raise ValueError("Loss weights must name configured output decoders")
        # Back-compat: is_vla=True implies the action regression decoder.
        # Old VLA configs only set is_vla, never output_decoders, so map it
        # to ["text", "action"] here. Idempotent if already listed.
        if self.is_vla and "action" not in self.output_decoders:
            self.output_decoders = ["text", "action"]

    @classmethod
    def from_preset(cls, preset_name: str):
        """Build a ``ModelConfig`` from a named entry in ``PRISM_CONFIGS``.

        Args:
            preset_name: Hyphenated preset key, e.g. ``"prism-nano"`` or
                ``"prism-olmo3-7b"``. These are not the underscored Hydra
                YAML filenames under ``src/conf/model/``.

        Returns:
            A new ``ModelConfig`` built by passing the preset dict as keyword
            arguments. Fields the preset names take its values; every other
            field keeps its dataclass default, and ``__post_init__`` runs as
            usual.

        Raises:
            ValueError: If ``preset_name`` is not a key of ``PRISM_CONFIGS``.
                The message lists the available keys.

        Example:
            >>> ModelConfig.from_preset("prism-nano").d_model
            1280
        """
        if preset_name not in PRISM_CONFIGS:
            raise ValueError(
                f"Unknown preset: {preset_name}. Available: {list(PRISM_CONFIGS.keys())}"
            )

        config_dict = PRISM_CONFIGS[preset_name]
        return cls(**config_dict)


# --- Configuration Presets ---
PRISM_CONFIGS: dict[str, dict[str, Any]] = {
    "prism-nano": {
        # Base: NanoChat Speedrun (~0.6B) | Full MoE: ~2.5B
        "d_model": 1280,
        "num_layers": 20,
        "num_heads": 10,
        "vocab_size": 65536,
        "d_text": 1280,
        "mlp_type": "relusquared",
        "hf_model_id": "sdobson/nanochat",  # 561M params, speedrun aligned
    },
    "prism-micro": {
        # Base: Pythia-410M (~0.4B) | Full MoE: ~1.5B
        "d_model": 1024,
        "num_layers": 24,  # Pythia-410M has 24 layers
        "num_heads": 16,
        "vocab_size": 32000,
        "d_text": 1024,
        "hf_model_id": "EleutherAI/pythia-410m",
    },
    "prism-mini": {
        # Base: Custom (~0.8B) | Full MoE: ~3.0B
        "d_model": 1536,
        "num_layers": 10,
        "num_heads": 24,
        "vocab_size": 32000,
        "d_text": 1536,
        "hf_model_id": None,  # Custom, no pretrained base
    },
    "prism-small": {
        # Base: Llama-3.2-1B (~1.2B) | Full MoE: ~4.5B
        "d_model": 2048,
        "num_layers": 16,
        "num_heads": 32,
        "vocab_size": 128256,
        "d_text": 2048,
        "hf_model_id": "meta-llama/Llama-3.2-1B",
    },
    "prism-granite-2b": {
        # Base: IBM Granite 3.2 2B (~2.5B) | Full MoE: ~9.5B
        "d_model": 2048,
        "num_layers": 40,
        "num_heads": 32,
        "vocab_size": 49152,
        "d_text": 2048,
        "hf_model_id": "ibm-granite/granite-3.2-2b-instruct",
    },
    "prism-phi4-mini": {
        # Base: Microsoft Phi-4-mini (~3.8B) | Full MoE: ~14B
        "d_model": 3072,
        "num_layers": 32,
        "num_heads": 24,
        "vocab_size": 200064,
        "d_text": 3072,
        "hf_model_id": "microsoft/Phi-4-mini-instruct",
    },
    "prism-base": {
        # Base: Custom (~3B) | Full MoE: ~11B
        "d_model": 4096,
        "num_layers": 12,
        "num_heads": 32,
        "vocab_size": 32000,
        "d_text": 4096,
        "hf_model_id": None,
    },
    "prism-auroragpt-2b": {
        # Base: AuroraGPT-2B (~2B) - Llama architecture, Gemma tokenizer (256K vocab)
        # Pretrained from scratch on 7T+ tokens via Megatron-DeepSpeed (SophiaG optimizer)
        # Architecture: 12 layers, 16 heads, 4 KV heads (GQA), FFN 11008, SwiGLU
        # NOTE: Text+image only to match pixmo training and enable static_graph DDP
        "d_model": 2048,
        "num_layers": 12,
        "num_heads": 16,
        "vocab_size": 256000,
        "d_text": 2048,
        "d_img": 768,  # SigLIP2-base native dim (768, not 1152 which is SigLIP-large)
        "modalities": ["text", "image"],
        "llm_backbone_id": os.environ.get(
            "PRISM_AURORAGPT_2B_CHECKPOINT",
            # Placeholder: point this at your own AuroraGPT-2B checkpoint,
            # e.g. /lus/flare/projects/<project>/<path>/AuroraGPT-2B-.../global_stepNNNNNN
            "/lus/flare/projects/<project>/<path>/AuroraGPT-2B-checkpoint/global_stepNNNNNN",
        ),
        "freeze_backbone": True,
        "freeze_encoders": True,
    },
    "prism-olmo3-7b": {
        # Base: OLMo 3 7B (~7B)
        "d_model": 4096,
        "num_layers": 32,
        "num_heads": 32,
        "vocab_size": 100278,
        "d_text": 4096,
        "llm_backbone_id": "allenai/Olmo-3-7B-Instruct",
        "freeze_backbone": True,
        "freeze_encoders": True,
    },
    "prism-nemotron-30b": {
        # Base: Nvidia Nemotron 3 Nano 30B (~30B)
        # Note: Auto-detects hidden_size (~6144 or 7168 likely)
        "d_model": 7168,  # Placeholder, will be overwritten by backbone config
        "num_layers": 32,  # Placeholder
        "num_heads": 32,  # Placeholder
        "vocab_size": 32000,  # Placeholder
        "d_text": 4096,  # Projector input (if using text encoder) or tuned
        "llm_backbone_id": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        "freeze_backbone": True,
    },
    "prism-olmo-ts-7b": {
        "llm_backbone_id": "allenai/OLMo-7B-0724-hf",
        "modalities": ["text", "time_series"],
        "freeze_backbone": True,
        "freeze_encoders": True,
        "is_timeseries": True,
    },
    "prism-olmo-ts-1b-interleaved": {
        "llm_backbone_id": "allenai/OLMo-1B-0724-hf",
        "llm_tokenizer_id": os.environ.get(
            "PRISM_OLMO1B_INTERLEAVED_TOKENIZER",
            # Custom tokenizer with interleaving tokens added; build it with
            # the committed tokenizers/ directory, or point this at your own path.
            "/flare/<project>/<user>/BaseMM_PRISM/tokenizers/prism-olmo-1b-interleaved",
        ),
        "modalities": ["text", "time_series"],
        "freeze_backbone": True,
        "freeze_encoders": True,
        "is_timeseries": True,
        "ts_variates": 1,
        "max_ts_length": 512,
        "normalize_ts_in_encoder": True,  # Apply scale-loc normalization in the TimeSeriesEncoder
        "is_interleaved_qa": True,
        "modality_start_end_token_indices": {
            "time_series": (
                50280,
                50281,
            )  # Example token IDs for start and end of time series features
        },
    },
    "prism-olmo-linear-ts-1b-interleaved": {
        "llm_backbone_id": "allenai/OLMo-1B-0724-hf",
        "llm_tokenizer_id": os.environ.get(
            "PRISM_OLMO1B_INTERLEAVED_TOKENIZER",
            # Custom tokenizer with interleaving tokens added; build it with
            # the committed tokenizers/ directory, or point this at your own path.
            "/flare/<project>/<user>/BaseMM_PRISM/tokenizers/prism-olmo-1b-interleaved",
        ),
        "modalities": ["text", "time_series"],
        "freeze_backbone": True,
        "freeze_encoders": False,
        "is_timeseries": True,
        "ts_variates": 1,
        "normalize_ts_in_encoder": False,  # Apply scale-loc normalization during data loading
        "max_ts_length": 256,
        "is_interleaved_qa": True,
        "modality_start_end_token_indices": {
            "time_series": (
                50280,
                50281,
            )  # Example token IDs for start and end of time series features
        },
    },
    "prism-olmo-linear-ts-7b-interleaved": {
        "llm_backbone_id": "allenai/OLMo-7B-0724-hf",
        "llm_tokenizer_id": os.environ.get(
            "PRISM_OLMO1B_INTERLEAVED_TOKENIZER",
            # Custom tokenizer with interleaving tokens added; build it with
            # the committed tokenizers/ directory, or point this at your own path.
            "/flare/<project>/<user>/BaseMM_PRISM/tokenizers/prism-olmo-1b-interleaved",
        ),
        "modalities": ["text", "time_series"],
        "freeze_backbone": True,
        "freeze_encoders": False,
        "is_timeseries": True,
        "ts_variates": 1,
        "normalize_ts_in_encoder": False,  # Apply scale-loc normalization during data loading
        "max_ts_length": 256,
        "is_interleaved_qa": True,
        "modality_start_end_token_indices": {
            "time_series": (
                50280,
                50281,
            )  # Example token IDs for start and end of time series features
        },
    },
}
