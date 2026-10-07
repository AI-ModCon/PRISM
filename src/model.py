"""The PRISM ``UnifiedTransformer`` and its custom transformer block.

``UnifiedTransformer`` is the model: it encodes every configured modality,
projects each one into the shared backbone width, concatenates (or
interleaves) the results into one sequence, and runs that sequence through a
Hugging Face causal-LM backbone -- or, when no backbone is configured,
through the local ``TransformerBlock`` stack defined here.

Only the backbone path reaches the output decoders in ``src/decoders/``. The
no-backbone path ends at the local ``CausalLMHead`` (``self.head``, from
``src/modules/heads.py``) and an inline cross-entropy, and the structured
decoding entry points refuse it with ``NotImplementedError``.

The class docstring walks those construction stages and the two forward
paths. Everything the module builds is driven by ``ModelConfig``
(``src/config.py``).
"""

import logging
import random
from collections.abc import Mapping

import torch
import torch.nn as nn

from .config import ModelConfig
from .decoders import (
    DECODERS,
    GeometryDecoder,
    GraphDecoder,
    ImageDecoder,
    LMHeadDecoder,
    RegressionDecoder,
    TimeSeriesDecoder,
)
from .decoders.conditioning import align_text_targets, compile_inputs
from .decoders.types import DecoderCondition, DecoderResult
from .hf_cache import load_cached_tokenizer
from .modalities import Modality

logger = logging.getLogger(__name__)
from .encoders import (
    DNAEncoder,
    GeometryEncoder,
    GraphEncoder,
    ImageEncoder,
    TableEncoder,
    TextEncoder,
    TimeSeriesEncoder,
)
from .modules import (
    CausalLMHead,
    CausalSelfAttention,
    ModalityProjector,
    MoELayer,
    PerceiverResampler,
)
from .modules.attention import RotaryEmbedding


class TransformerBlock(nn.Module):
    """One pre-norm transformer block with a sparse mixture-of-experts MLP.

    The block PRISM stacks on the no-backbone path (Path B): LayerNorm ->
    ``CausalSelfAttention`` -> residual, then LayerNorm -> ``MoELayer`` ->
    residual. The MoE layer replaces the usual dense feed-forward and returns
    a load-balancing auxiliary loss alongside its output, which this block
    passes straight through to its caller.

    Widths, head count, expert count, dropout, maximum sequence length and
    MLP flavor all come from the ``ModelConfig`` passed to ``__init__``.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(
            config.d_model, config.num_heads, config.max_seq_len, config.dropout
        )
        self.ln2 = nn.LayerNorm(config.d_model)
        self.moe = MoELayer(
            config.d_model,
            config.num_experts,
            config.num_experts_per_token,
            config.dropout,
            config.mlp_type,
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor = None, freqs_cis: tuple = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply attention and the MoE feed-forward, each with a residual.

        Args:
            x: Hidden states, ``(B, T, d_model)``.
            mask: Additive attention mask broadcastable to
                ``(B, 1, T, T)`` -- ``0.0`` to attend, a large negative value
                to mask. Default: ``None``, which makes
                ``CausalSelfAttention`` fall back to its own causal mask.
            freqs_cis: ``(cos, sin)`` rotary tables from ``RotaryEmbedding``,
                applied to the queries and keys. Default: ``None`` (no RoPE).

        Returns:
            ``(x, aux_loss)``: ``x`` is the updated hidden states, same shape
            as the input; ``aux_loss`` is the scalar load-balancing loss the
            block's ``MoELayer`` produced for this call, which the caller is
            expected to add to the training objective.

        Shape:
            - x: ``(B, T, d_model)``
            - output: ``(B, T, d_model)``, scalar
        """
        x = x + self.attn(self.ln1(x), mask=mask, freqs_cis=freqs_cis)
        moe_out, aux_loss = self.moe(self.ln2(x))
        x = x + moe_out
        return x, aux_loss


class UnifiedTransformer(nn.Module):
    """PRISM's multimodal model: encode every modality, fuse, decode.

    One ``ModelConfig`` describes the whole network. ``__init__`` builds it in
    stages, most of them marked by a ``# --- Stage N ---`` comment below:

    - **Stage 0 - backbone (optional).** When ``config.llm_backbone_id`` is
      set, an ``AutoModelForCausalLM`` is loaded from the local Hugging Face
      cache (``local_files_only=True``) in bfloat16 on CUDA when
      ``torch.cuda.is_bf16_supported()``, or on any available XPU (assumed
      there, not probed), and otherwise in float16. Under
      ``torch.distributed`` local rank 0 loads
      first and the other ranks follow after a barrier, so they read a warm
      cache. The tokenizer (``config.llm_tokenizer_id`` when given, else the
      backbone id) lands on ``self.backbone_tokenizer`` and drives a
      **grow-only** ``resize_token_embeddings``, seeded so every rank
      initializes new rows identically. ``config.freeze_backbone`` then
      freezes the weights, and ``self.backbone_dim`` picks up the backbone's
      ``hidden_size``. A failure here raises ``RuntimeError``. With no
      backbone id, ``self.backbone`` stays ``None`` and ``self.backbone_dim``
      remains ``config.d_model``.
    - **Stage 1 - modality encoders,** plus the Stage 2 projectors that feed
      on them. For each modality in ``config.modalities`` an encoder goes into
      the ``self.encoders`` ``nn.ModuleDict`` and a projector to
      ``self.backbone_dim`` into ``self.projectors``, under the modality's
      own name. Text, image and time series use ``ModalityProjector``; table,
      geometry and graph use ``PerceiverResampler`` with 32 latents, giving
      those modalities a fixed token budget. Text is the exception: an
      external ``TextEncoder`` is built only when there is no backbone, since
      otherwise the backbone's own input-embedding layer embeds the text ids.
      ``config.freeze_encoders`` freezes every encoder, and the frozen state
      is cached in ``self._encoder_frozen_cache`` so the forward pass can skip
      an O(params) check.
    - **Output decoders.** ``config.output_decoders`` names what gets built
      into ``self.decoders``; an unknown name raises ``ValueError``.
      ``self.text_decoder`` is always present, even if the config omits
      ``"text"``. The VLA action head is the one exception: it stays at
      ``self.action_head`` so existing checkpoints keep matching.
    - **Stages 3 and 4 - local transformer stack,** built only when there is
      no backbone: ``self.blocks`` (``config.num_layers`` instances of
      ``TransformerBlock``), ``self.ln_f``, rotary embeddings in
      ``self.rope``, and the ``CausalLMHead`` at ``self.head``.

    ``forward`` mirrors that split. It encodes and projects each modality
    present in ``inputs``, then either interleaves the modality tokens into
    the text ids at their start/end marker tokens
    (``config.is_interleaved_qa``) or concatenates them as a prefix with the
    text last, and runs the fused sequence through whichever stack exists --
    **Path A** the HF backbone, **Path B** the local blocks. Both paths return
    ``(logits, loss)``. Two variants return something else: VLA mode
    (``config.is_vla``) returns ``(pred_action, loss, per_dim_mse)``, and
    passing ``requested_outputs`` routes to the decoder registry and returns a
    ``DecoderResult``. ``generate`` delegates to ``backbone.generate`` on
    Path A.

    Attributes:
        config: The ``ModelConfig`` the model was built from.
        backbone: The HF causal LM, or ``None`` on the custom path.
        backbone_dim: Shared width every projector targets -- the backbone's
            ``hidden_size``, or ``config.d_model`` with no backbone.
        encoders: ``nn.ModuleDict`` of per-modality encoders, keyed by
            modality name.
        projectors: ``nn.ModuleDict`` of per-modality projectors into
            ``backbone_dim``, keyed the same way.
        decoders: ``nn.ModuleDict`` of configured output decoders.
        text_decoder: The ``LMHeadDecoder`` that carries the shifted
            cross-entropy on the backbone path. Always set, even with no
            backbone -- but the no-backbone path never reads it.
        global_step: Step counter the trainer writes onto the model.
            ``forward`` only reads it, to gate the ``% 100`` debug logs and
            to name the step in the NaN-loss report.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.is_vla = config.is_vla
        self._backbone_load_error = None

        # --- Stage 0: Backbone (Optional) ---
        self.backbone = None
        self.backbone_dim = config.d_model
        self.global_step = 0  # For logging purposes

        if config.llm_backbone_id:
            logger.info(f"Initializing HF Backbone: {config.llm_backbone_id}...")
            try:
                import os

                import torch.distributed as dist
                from transformers import AutoModelForCausalLM

                # Trust remote code needed for models like Nemotron / Mamba
                # Use float16 even on CPU for large models (30B+) to fit in RAM
                # Prefer bfloat16 for OLMo/Llama-3 if available (MPS/Mac supports it usually)
                # Enable BF16 for CUDA or XPU (Aurora)
                use_bf16 = False
                if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                    use_bf16 = True
                # Check for XPU (Intel Extension for PyTorch)
                elif hasattr(torch, "xpu") and torch.xpu.is_available():
                    use_bf16 = True

                dtype = torch.bfloat16 if use_bf16 else torch.float16
                # On MPS, bfloat16 is supported in recent torch, but safe fallback is float16.
                # However, OLMo is trained in bf16.

                # --- Fix for HuggingFace cache race condition on multi-node ---
                # Only local_rank 0 on each node loads first to populate the cache,
                # then other ranks load after a barrier to avoid metadata contention.
                # CRITICAL: All ranks must participate in each barrier call.
                local_rank = int(
                    os.environ.get("PALS_LOCAL_RANKID", os.environ.get("LOCAL_RANK", "0"))
                )

                # Load backbone with proper synchronization
                backbone_loaded = False
                if dist.is_initialized():
                    if local_rank == 0:
                        # Local rank 0 loads first (populates /tmp cache)
                        try:
                            self.backbone = AutoModelForCausalLM.from_pretrained(
                                config.llm_backbone_id,
                                trust_remote_code=True,
                                torch_dtype=dtype,
                                low_cpu_mem_usage=True,
                                local_files_only=True,
                                attn_implementation=config.attn_implementation,
                            )
                            logger.info(
                                f"Local rank 0: Backbone loaded (attn={config.attn_implementation}), "
                                "signaling other ranks..."
                            )
                            backbone_loaded = True
                        except Exception as e:
                            logger.error(f"Local rank 0: Error loading backbone: {e}")
                            self.backbone = None

                    # ALL ranks synchronize here - rank 0 has loaded (or failed), others are ready
                    dist.barrier()

                    # Non-zero local ranks now load from cache
                    if local_rank != 0:
                        logger.info(f"Rank {local_rank}: Loading backbone from cache...")
                        try:
                            self.backbone = AutoModelForCausalLM.from_pretrained(
                                config.llm_backbone_id,
                                trust_remote_code=True,
                                torch_dtype=dtype,
                                low_cpu_mem_usage=True,
                                local_files_only=True,
                                attn_implementation=config.attn_implementation,
                            )
                            backbone_loaded = True
                        except Exception as e:
                            logger.error(f"Rank {local_rank}: Error loading backbone: {e}")
                            self.backbone = None

                    # ALL ranks synchronize again after loading
                    dist.barrier()

                    # Load tokenizer. This ALSO drives resize_token_embeddings
                    # below, so it MUST match the tokenizer the data pipeline
                    # uses (config.llm_tokenizer_id) — not the base backbone
                    # tokenizer. Using the base tokenizer here shrinks the
                    # embedding table below the custom modality-token ids
                    # (e.g. <ts>=50280) that appear in every prompt, causing an
                    # out-of-bounds embedding gather on XPU and a delayed GPU
                    # write page-fault (issue #117). Mirror the single-process
                    # branch below, which already resolves llm_tokenizer_id.
                    tokenizer_id = (
                        config.llm_tokenizer_id
                        if config.llm_tokenizer_id
                        else config.llm_backbone_id
                    )
                    if backbone_loaded:
                        try:
                            self.backbone_tokenizer = load_cached_tokenizer(
                                tokenizer_id,
                                trust_remote_code=True,
                            )
                        except Exception as e:
                            logger.warning(
                                f"Could not load backbone tokenizer for debug logging: {e}"
                            )
                else:
                    # Single process mode
                    self.backbone = AutoModelForCausalLM.from_pretrained(
                        config.llm_backbone_id,
                        trust_remote_code=True,
                        torch_dtype=dtype,
                        low_cpu_mem_usage=True,
                        local_files_only=True,
                        attn_implementation=config.attn_implementation,
                    )
                    backbone_loaded = True

                    tokenizer_id = (
                        config.llm_tokenizer_id
                        if config.llm_tokenizer_id
                        else config.llm_backbone_id
                    )
                    self.backbone_tokenizer = load_cached_tokenizer(
                        tokenizer_id,
                        trust_remote_code=True,
                    )

                if self.backbone is None:
                    raise RuntimeError(
                        f"Could not load HF backbone {config.llm_backbone_id!r} "
                        "from the local HuggingFace cache"
                    )
                # Grow the embedding matrix to match a custom tokenizer.
                # PRISM modality tokens (<ts>, <image>, <table>, ...) push
                # len(tokenizer) past the backbone's original vocab.
                # Seed before resize so every rank initializes the new rows
                # identically — FSDP wrap does not set sync_module_states=True
                # (see training/distributed.py), so without this the new rows
                # would diverge across ranks.
                if backbone_loaded and hasattr(self.backbone_tokenizer, "__len__"):
                    # Grow-only resize. We must never shrink below the current
                    # embedding size: doing so drops rows that valid token ids
                    # index into, causing an out-of-bounds embedding gather on
                    # XPU and a delayed GPU write page-fault (issue #117). This
                    # bit us when the distributed path loaded the BASE backbone
                    # tokenizer (len 50280) instead of the custom interleaved
                    # tokenizer (len 50292) whose <ts>=50280 token appears in
                    # every prompt. Target = max(current, len(tokenizer)).
                    _before = len(self.backbone.get_input_embeddings().weight)
                    _target = max(_before, len(self.backbone_tokenizer))
                    if _target != _before:
                        logger.info(
                            f"Resizing token embeddings {_before} -> {_target} "
                            f"(len(tokenizer)={len(self.backbone_tokenizer)}, "
                            f"tokenizer_id={getattr(config, 'llm_tokenizer_id', None)!r})"
                        )
                        # Seed before resize so every rank initializes the new
                        # rows identically — FSDP wrap does not set
                        # sync_module_states=True (see training/distributed.py),
                        # so without this the new rows would diverge across ranks.
                        rng_state = torch.get_rng_state()
                        try:
                            torch.manual_seed(0)
                            self.backbone.resize_token_embeddings(_target)
                        finally:
                            torch.set_rng_state(rng_state)

                if config.freeze_backbone:
                    logger.info("Freezing Backbone weights...")
                    for param in self.backbone.parameters():
                        param.requires_grad = False

                # Update dims
                if hasattr(self.backbone.config, "hidden_size"):
                    self.backbone_dim = self.backbone.config.hidden_size
                    logger.info(f"Backbone hidden dim detected: {self.backbone_dim}")

                # We do NOT initialize blocks/head if using backbone

                # Resolve any string-valued modality_start_end_token_indices entries
                # (e.g. DNA's ("<dna_ref_start>", "<dna_ref_end>")) to integer token
                # ids. Unlike time_series's fixed ids (pre-baked into the custom
                # interleaved tokenizer's vocab at known positions), DNA's splice
                # tokens are authored as literal strings in multimodal.py's prompt
                # renderer and in eval_kegg.py — register them as special tokens
                # if missing, then resolve to ids so
                # _merge_text_input_ids_with_modality_embeds's `input_ids ==
                # start_idx` integer comparison works. Mirrors the grow-only,
                # seeded-RNG resize pattern above (issue #117) so new embedding
                # rows initialize identically across ranks.
                if (
                    self.backbone_tokenizer is not None
                    and config.modality_start_end_token_indices
                ):
                    unresolved = {
                        modality: tags
                        for modality, tags in config.modality_start_end_token_indices.items()
                        if isinstance(tags[0], str) or isinstance(tags[1], str)
                    }
                    if unresolved:
                        missing = sorted(
                            {
                                tok
                                for tags in unresolved.values()
                                for tok in tags
                                if isinstance(tok, str)
                                and tok not in self.backbone_tokenizer.get_vocab()
                            }
                        )
                        if missing:
                            logger.info(
                                f"Registering {len(missing)} new special token(s) "
                                f"{missing} for string-keyed modality_start_end_token_indices entries"
                            )
                            num_added = self.backbone_tokenizer.add_special_tokens(
                                {"additional_special_tokens": missing}
                            )
                            if num_added > 0:
                                rng_state = torch.get_rng_state()
                                try:
                                    torch.manual_seed(0)
                                    self.backbone.resize_token_embeddings(
                                        len(self.backbone_tokenizer)
                                    )
                                finally:
                                    torch.set_rng_state(rng_state)
                                logger.info(
                                    f"Resized token embeddings to {len(self.backbone_tokenizer)} "
                                    f"after adding {num_added} modality special token(s)"
                                )

                        resolved = dict(config.modality_start_end_token_indices)
                        for modality, tags in unresolved.items():
                            resolved[modality] = tuple(
                                self.backbone_tokenizer.convert_tokens_to_ids(tok)
                                if isinstance(tok, str)
                                else tok
                                for tok in tags
                            )
                        config.modality_start_end_token_indices = resolved

            except Exception as e:
                logger.error(f"Error loading Backbone {config.llm_backbone_id}: {e}")
                self._backbone_load_error = e
                raise RuntimeError(
                    f"Could not load HF backbone {config.llm_backbone_id!r} "
                    "with local_files_only=True. Stage the model cache before "
                    "launching or use a config without model.backbone_id. "
                    f"Current cache directories: {os.environ.get('HF_HOME', 'Not set')}, {os.environ.get('TRANSFORMERS_CACHE', 'Not set')}, {os.environ.get('SHARED_HF_HOME', 'Not set')}"
                ) from e

        # --- Stage 1: Modality Encoders ---
        self.encoders = nn.ModuleDict()
        self.projectors = nn.ModuleDict()

        # Determine target dimension for projectors
        target_dim = self.backbone_dim

        # === Projector Configuration (from config, with defaults for backward compat) ===
        proj_norm_mode = getattr(config, "projector_norm_mode", "layernorm")
        proj_target_norm = getattr(config, "projector_target_norm", 0.25)
        proj_embed_pos = getattr(config, "projector_modality_embed_pos", "after_norm")
        proj_embed_scale = getattr(config, "projector_modality_embed_scale", 0.02)
        # Text statistics matching parameters (for match_text_stats and match_text_elemstats modes)
        proj_text_norm_mean = getattr(config, "projector_text_norm_mean", 0.25)
        proj_text_norm_std = getattr(config, "projector_text_norm_std", 0.05)
        proj_text_elem_mean = getattr(config, "projector_text_elem_mean", 0.0)
        proj_text_elem_std = getattr(config, "projector_text_elem_std", 0.006)
        proj_norm_clip_min = getattr(config, "projector_norm_clip_min", 0.1)
        proj_norm_clip_max = getattr(config, "projector_norm_clip_max", 0.5)
        # IsoFLOP capacity knobs (single global pair — per-modality variants
        # can be added later; for PR-1 the same shape applies to every
        # ModalityProjector instance).
        proj_hidden_mult = getattr(config, "projector_hidden_mult", 1)
        proj_num_layers = getattr(config, "projector_num_layers", 2)

        logger.info(
            f"Projector config: norm_mode={proj_norm_mode}, target_norm={proj_target_norm}, embed_pos={proj_embed_pos}, embed_scale={proj_embed_scale}, hidden_mult={proj_hidden_mult}, num_layers={proj_num_layers}"
        )
        if proj_norm_mode in ["match_text_stats", "match_text_elemstats"]:
            logger.info(
                f"  Text stats: norm_mean={proj_text_norm_mean}, norm_std={proj_text_norm_std}, "
                f"elem_mean={proj_text_elem_mean}, elem_std={proj_text_elem_std}, clip=[{proj_norm_clip_min}, {proj_norm_clip_max}]"
            )

        # Text
        # Only load external text encoder (SmolLM/etc) if we DO NOT have a backbone.
        # If backbone exists, we use its native embedding layer (see _process_multimodal_embeddings).
        if Modality.TEXT in config.modalities and self.backbone is None:
            logger.info("Initializing External Text Encoder...")
            self.encoders["text"] = TextEncoder(d_text=config.d_text)
            self.projectors["text"] = ModalityProjector(
                config.d_text,
                target_dim,
                norm_mode=proj_norm_mode,
                target_norm=proj_target_norm,
                modality_embed_pos=proj_embed_pos,
                modality_embed_scale=proj_embed_scale,
                text_norm_mean=proj_text_norm_mean,
                text_norm_std=proj_text_norm_std,
                text_elem_mean=proj_text_elem_mean,
                text_elem_std=proj_text_elem_std,
                norm_clip_min=proj_norm_clip_min,
                norm_clip_max=proj_norm_clip_max,
                hidden_mult=proj_hidden_mult,
                num_layers=proj_num_layers,
            )

        # Table
        if Modality.TABLE in config.modalities:
            self.encoders["table"] = TableEncoder(d_table=config.d_table)
            self.projectors["table"] = PerceiverResampler(
                input_dim=config.d_table,
                d_model=target_dim,
                num_latents=32,
                num_layers=2,
            )


        # Time Series
        if Modality.TIME_SERIES in config.modalities:
            self.encoders["time_series"] = TimeSeriesEncoder(
                encoder_type=config.ts_projector, 
                d_ts=config.d_ts,
                num_vars=config.ts_variates,
                model_name=config.ts_encoder_id,
                max_ts_length=config.max_ts_length,
                is_interleaved=bool(self.config.is_interleaved_qa),
                intern_s2_sampling_rate=config.intern_s2_sampling_rate,
                timeomni_patch_len=config.timeomni_patch_len,
                timeomni_stride=config.timeomni_stride,
                timeomni_d_model=config.timeomni_d_model,
                timeomni_dropout=config.timeomni_dropout,
                timeomni_ts_tokens=config.timeomni_ts_tokens,
                timeomni_max_patches=config.timeomni_max_patches,
                load_pretrained=config.ts_load_pretrained,
            )

            # Use actual encoder output dim if available (Moirai might be 1024, config 512)
            ts_input_dim = getattr(self.encoders["time_series"], "hidden_dim", config.d_ts)

            self.projectors["time_series"] = ModalityProjector(
                ts_input_dim,
                target_dim,
                hidden_mult=proj_hidden_mult,
                num_layers=proj_num_layers,
            )

        # Image
        if Modality.IMAGE in config.modalities:
            self.encoders["image"] = ImageEncoder(
                d_img=config.d_img,
                model_name=config.image_encoder_id,
            )
            self.projectors["image"] = ModalityProjector(
                config.d_img,
                target_dim,
                norm_mode=proj_norm_mode,
                target_norm=proj_target_norm,
                modality_embed_pos=proj_embed_pos,
                modality_embed_scale=proj_embed_scale,
                text_norm_mean=proj_text_norm_mean,
                text_norm_std=proj_text_norm_std,
                text_elem_mean=proj_text_elem_mean,
                text_elem_std=proj_text_elem_std,
                norm_clip_min=proj_norm_clip_min,
                norm_clip_max=proj_norm_clip_max,
                hidden_mult=proj_hidden_mult,
                num_layers=proj_num_layers,
            )

        # Geometry
        if Modality.GEOMETRY in config.modalities:
            self.encoders["geometry"] = GeometryEncoder(d_geo=config.d_geo)
            geo_input_dim = getattr(self.encoders["geometry"], "hidden_dim", config.d_geo)

            self.projectors["geometry"] = PerceiverResampler(
                input_dim=geo_input_dim,
                d_model=target_dim,
                num_latents=32,
                num_layers=2,
            )

        # Graph (requires torch_geometric)
        if Modality.GRAPH in config.modalities:
            self.encoders["graph"] = GraphEncoder(input_dim=32, d_graph=config.d_graph)
            self.projectors["graph"] = PerceiverResampler(
                input_dim=config.d_graph, d_model=target_dim, num_latents=32, num_layers=2
            )

        # DNA (Nucleotide Transformer / Evo2)
        if Modality.DNA in config.modalities:
            dna_model_name = getattr(
                config,
                "dna_model_name",
                "InstaDeepAI/nucleotide-transformer-2.5b-multi-species",
            )
            dna_is_evo2 = getattr(config, "dna_is_evo2", False)

            self.encoders["dna"] = DNAEncoder(
                d_dna=config.d_dna,
                model_name=dna_model_name,
                dna_is_evo2=dna_is_evo2,
                max_dna_length=getattr(config, "max_dna_length", None),
            )
            # Output dimension from the encoder (d_dna after its internal projection)
            dna_input_dim = getattr(self.encoders["dna"], "hidden_dim", config.d_dna)

            self.projectors["dna"] = ModalityProjector(
                dna_input_dim,
                target_dim,
                norm_mode=proj_norm_mode,
                target_norm=proj_target_norm,
                modality_embed_pos=proj_embed_pos,
                modality_embed_scale=proj_embed_scale,
                text_norm_mean=proj_text_norm_mean,
                text_norm_std=proj_text_norm_std,
                text_elem_mean=proj_text_elem_mean,
                text_elem_std=proj_text_elem_std,
                norm_clip_min=proj_norm_clip_min,
                norm_clip_max=proj_norm_clip_max,
                hidden_mult=proj_hidden_mult,
                num_layers=proj_num_layers,
            )
        # === Output decoders (Phase 1) ===
        # self.decoders is the config-driven home for output decoders owned by
        # this module (currently just text; Phase 2+ adds time_series, etc.).
        # The VLA action decoder is the one legacy exception — it stays as
        # self.action_head (built in the VLA block below) because the VLA
        # trainer and existing checkpoints hard-code the `action_head.*`
        # parameter name.
        self.decoders = nn.ModuleDict()
        decoder_cfgs = config.decoder_configs or {}
        for _dec_name in config.output_decoders:
            if _dec_name == "text":
                # Parameter-less on the HF-backbone path (the LM head lives in
                # the backbone); carries the shifted-CE loss.
                self.decoders["text"] = LMHeadDecoder()
            elif _dec_name in ("action", "regression"):
                # Owned by the VLA block as self.action_head — skip here to
                # avoid double-registering parameters.
                continue
            elif _dec_name == "time_series":
                # Direct multi-horizon quantile forecaster. Reads pooled
                # backbone hidden states; defaults pull from the time-series
                # model config so a bare ["text","time_series"] works.
                ts_cfg = dict(decoder_cfgs.get("time_series", {}))
                if ts_cfg.get("generator") is not None:
                    # Keep native nesting: injecting flat defaults would mix
                    # the legacy and explicit constructor contracts.
                    if isinstance(ts_cfg["generator"], Mapping):
                        generator_cfg = dict(ts_cfg["generator"])
                        generator_cfg.setdefault("horizon", config.ts_forecast_horizon)
                        generator_cfg.setdefault("num_vars", config.ts_variates)
                        ts_cfg["generator"] = generator_cfg
                else:
                    ts_cfg.setdefault("horizon", config.ts_forecast_horizon)
                    ts_cfg.setdefault("num_vars", config.ts_variates)
                self.decoders["time_series"] = TimeSeriesDecoder(
                    d_model=target_dim, **ts_cfg
                )
            elif _dec_name == "image":
                self.decoders["image"] = ImageDecoder(
                    d_model=target_dim, **dict(decoder_cfgs.get("image", {}))
                )
            elif _dec_name == "geometry":
                # Field-regression head (inverse of GeometryEncoder/Walrus).
                # num_points/num_channels are field-specific — must be given in
                # decoder_configs["geometry"].
                geo_cfg = dict(decoder_cfgs.get("geometry", {}))
                if "num_points" not in geo_cfg or "num_channels" not in geo_cfg:
                    raise ValueError(
                        "geometry decoder requires decoder_configs['geometry'] "
                        "with 'num_points' and 'num_channels'."
                    )
                self.decoders["geometry"] = GeometryDecoder(
                    d_model=target_dim, **geo_cfg
                )
            elif _dec_name == "graph":
                # Per-node head (inverse of GraphEncoder/GraphMAE2) — node
                # feature/property prediction only, never structure generation.
                # out_dim is task-specific — must be given in
                # decoder_configs["graph"].
                graph_cfg = dict(decoder_cfgs.get("graph", {}))
                if "out_dim" not in graph_cfg:
                    raise ValueError(
                        "graph decoder requires decoder_configs['graph'] with "
                        "'out_dim' (and optionally task='regression'|'classification')."
                    )
                self.decoders["graph"] = GraphDecoder(d_model=target_dim, **graph_cfg)
            else:
                raise ValueError(
                    f"Unknown output decoder '{_dec_name}'. Known: "
                    f"{sorted(DECODERS)} (additional decoders land in Phase 2+)."
                )

        # self.text_decoder is the forward path's handle on the text decoder.
        # Always present so the non-VLA path works even if a config omits
        # "text" from output_decoders.
        if "text" not in self.decoders:
            self.decoders["text"] = LMHeadDecoder()
        self.text_decoder = self.decoders["text"]

        if self.is_vla:
            if self.backbone is None:
                raise RuntimeError(
                    f"VLA mode requires a loadable llm_backbone_id. "
                    f"backbone_id={config.llm_backbone_id}, error={self._backbone_load_error}"
                )
            if Modality.IMAGE not in config.modalities:
                raise RuntimeError("VLA mode requires 'image' in model.modalities")
            self.pose_embed = nn.Sequential(
                nn.Linear(config.pose_dim, target_dim),
                nn.ReLU(),
                nn.Linear(target_dim, target_dim),
            )
            self.pose_modality_embedding = nn.Parameter(torch.randn(1, 1, target_dim) * 0.02)
            # Action regression head, now expressed via the OutputDecoder
            # registry. Same MLP (Linear->ReLU->Linear) and init order as the
            # old inline nn.Sequential, so numerics are unchanged. The attribute
            # keeps the name `action_head` so the VLA trainer and existing
            # checkpoints keep working; the inner MLP moves from `action_head.*`
            # to `action_head.head.*`.
            self.action_head = RegressionDecoder(target_dim, config.action_dim)


        # Freeze Encoders if requested (Zone 1)
        if config.freeze_encoders:
            logger.info("Freezing Modality Encoders...")
            for param in self.encoders.parameters():
                param.requires_grad = False


        # Cache frozen state for each encoder (avoids O(params) check in forward)
        self._encoder_frozen_cache = {
            modality: not any(p.requires_grad for p in encoder.parameters())
            for modality, encoder in self.encoders.items()
        }

        # --- Stage 3: Unified Transformer (Legacy / Custom) ---
        if self.backbone is None:
            self.blocks = nn.ModuleList(
                [TransformerBlock(config) for _ in range(config.num_layers)]
            )
            self.ln_f = nn.LayerNorm(config.d_model)


            # RoPE
            self.rope = RotaryEmbedding(config.d_model // config.num_heads)


            # --- Stage 4: Output Head ---
            self.head = CausalLMHead(config.d_model, config.vocab_size)


    def get_input_embeddings(self, inputs):
        """Helper to get embeddings from varied inputs."""
        # Retrieve input_ids from 'text' if available
        if "text" in inputs:
            if self.backbone:
                return self.backbone.get_input_embeddings()(inputs["text"])
            else:
                # Custom implementation doesn't have a separate embedding layer usually stored here
                # In custom impl, Encoders handle embedding.
                # However, for text, TextEncoder output IS the embedding.
                pass
        return None

    def _resolve_pad_id(self) -> int | None:
        """Resolve pad_token_id from whichever tokenizer is attached.

        `backbone_tokenizer` is set only on Path A (HF backbone load).
        `tokenizer` is set unconditionally by src/train.py:642 for both
        paths. Prefer backbone_tokenizer when present (avoids a mismatch if
        Path A loads a backbone with its own tokenizer that differs from
        the training tokenizer), skipping any tokenizer whose
        `pad_token_id` is None, and fall back to the training tokenizer so
        Path B's pad-mask actually engages.
        """
        for attr in ("backbone_tokenizer", "tokenizer"):
            tok = getattr(self, attr, None)
            if tok is not None:
                pad_id = getattr(tok, "pad_token_id", None)
                if pad_id is not None:
                    return pad_id
        return None

    def _process_multimodal_embeddings(self, inputs):
        """Encodes and projects all modalities into a list of (name, tensor).

        Args:
            inputs: Dict mapping modality name to input tensor.

        Returns:
            List of (modality_name, projected_embedding) tuples.
        """
        embeddings = []
        self._modality_pad_masks = {}
        for modality in self.config.modalities:
            if modality in inputs:
                # Special handling for text if backbone is present
                if modality == "text" and self.backbone is not None:
                    # Use backbone's own embedding layer directly
                    # This ensures tokenizer compatibility (e.g. OLMo input -> OLMo embedding)
                    embed_layer = self.backbone.get_input_embeddings()
                    projected = embed_layer(inputs[modality])

                    # --- Text-Only Dropout (Molmo Paper) ---
                    # Dropping entire text tokens (masking) to force reliance on image.
                    # Only apply during training.
                    if self.training and getattr(self.config, "text_dropout", 0.0) > 0.0:
                        mask_prob = self.config.text_dropout
                        # Token-level dropout (Molmo paper): zero entire text token vectors
                        # to force reliance on image/modality tokens during training.
                        # mask shape: (B, T_text, 1) — broadcasts over embedding dim
                        mask = torch.bernoulli(
                            torch.ones_like(projected[:, :, :1]) * (1 - mask_prob)
                        )
                        projected = projected * mask
                        # Alternative: element-wise dropout (may suit non-VLM modalities)
                        # projected = F.dropout(projected, p=mask_prob, training=True)
                    # ---------------------------------------

                    embeddings.append((modality, projected))
                    continue

                # DNA: reference/variant sequences arrive as two independent
                # sub-inputs under one "dna" entry. They share the single
                # DNAEncoder/projector registered above, but each gets encoded
                # and projected separately (early fusion) and emitted as its
                # own (name, embedding) entry — "dna_reference"/"dna_variant"
                # — so the merge step below can splice each into its own
                # token span (see _PARENT_MODALITY in
                # _merge_text_input_ids_with_modality_embeds).
                if modality == Modality.DNA and isinstance(inputs[modality], dict):
                    raw_features = inputs[modality]
                    if "dna_reference" in raw_features and "dna_variant" in raw_features:
                        for sub_modality in ("dna_reference", "dna_variant"):
                            feature = raw_features[sub_modality]
                            if (
                                isinstance(feature, list | tuple)
                                and len(feature) == 2
                                and all(isinstance(x, torch.Tensor) for x in feature)
                            ):
                                encoder_input = {
                                    "input_ids": feature[0],
                                    "attention_mask": feature[1],
                                }
                            elif isinstance(feature, list | tuple) and len(feature) == 1:
                                encoder_input = feature[0]
                            else:
                                encoder_input = feature

                            sub_encoded = self.encoders[modality](encoder_input)
                            if getattr(self, "_encoder_frozen_cache", {}).get(modality, False):
                                sub_encoded = sub_encoded.detach()

                            try:
                                projected = self.projectors[modality](sub_encoded)
                            except Exception as e:
                                logger.error(f"Error projecting {sub_modality}: {e}")
                                logger.error(f"Feature Shape: {sub_encoded.shape}")
                                logger.error(f"Projector: {self.projectors[modality]}")
                                raise e

                            embeddings.append((sub_modality, projected))
                        continue  # dna_reference/dna_variant handled above; skip the generic path below

                # 1. Encode
                # Forward through encoder unconditionally for static_graph compatibility.
                # Detach output of frozen encoders to prevent autograd graph building through them.
                # This achieves the same memory/compute savings as torch.no_grad() without
                # changing the forward graph structure (which breaks DDP static_graph).
                encoder_input = inputs[modality]
                reference_shape = None
                reference_valid = None
                if modality == "image" and encoder_input.ndim == 5:
                    reference_shape = encoder_input.shape[:2]
                    reference_valid = inputs.get("image_mask")
                    if reference_valid is None:
                        reference_valid = torch.ones(reference_shape, device=encoder_input.device, dtype=torch.bool)
                    if reference_valid.shape != reference_shape or not torch.all((reference_valid == 0) | (reference_valid == 1)):
                        raise ValueError("image_mask must be binary with shape (B, N)")
                    reference_valid = reference_valid.to(device=encoder_input.device, dtype=torch.bool).flatten()
                    if not reference_valid.any():
                        continue
                    # Padding must not affect projector batch statistics or
                    # consume encoder work, even before the sequence is merged.
                    encoder_input = encoder_input.flatten(0, 1)[reference_valid]
                features = self.encoders[modality](encoder_input)
                pad_mask = getattr(self.encoders[modality], "_last_pad_mask", None)


                # Detach if encoder is frozen (avoids backward through encoder)
                # Use cached frozen state to avoid O(params) check every forward
                if getattr(self, "_encoder_frozen_cache", {}).get(modality, False):
                    features = features.detach()


                # 2. Project
                try:
                    projected = self.projectors[modality](features)
                except Exception as e:
                    logger.error(f"Error projecting {modality}: {e}")
                    logger.error(f"Feature Shape: {features.shape}")
                    logger.error(f"Projector: {self.projectors[modality]}")
                    raise e
                if reference_shape is not None:
                    per_reference, width = projected.shape[1:]
                    scattered = projected.new_zeros((reference_shape[0] * reference_shape[1], per_reference, width))
                    scattered[reference_valid.to(projected.device)] = projected
                    projected = scattered.reshape(reference_shape[0], reference_shape[1] * per_reference, width)
                if pad_mask is not None:
                    if pad_mask.shape != projected.shape[:2]:
                        raise ValueError(
                            f"{modality} pad mask shape {tuple(pad_mask.shape)} does not "
                            f"match projected features {tuple(projected.shape)}"
                        )
                    self._modality_pad_masks[modality] = pad_mask.detach()
                embeddings.append((modality, projected))
        return embeddings

    def _build_prefix_content_mask(
        self, embeddings: list[tuple[str, torch.Tensor]]
    ) -> torch.Tensor:
        """Build a per-sample valid-token mask in concatenated prefix order."""
        masks = []
        for modality, features in embeddings:
            pad_mask = getattr(self, "_modality_pad_masks", {}).get(modality)
            if pad_mask is None:
                valid = torch.ones(
                    features.shape[:2], dtype=torch.bool, device=features.device
                )
            else:
                if pad_mask.shape != features.shape[:2]:
                    raise ValueError(
                        f"{modality} pad mask shape {tuple(pad_mask.shape)} does not "
                        f"match features {tuple(features.shape)}"
                    )
                valid = ~pad_mask.to(device=features.device, dtype=torch.bool)
            masks.append(valid)
        if not masks:
            raise ValueError("Cannot build a content mask without embeddings")
        return torch.cat(masks, dim=1)

    def _merge_text_input_ids_with_modality_embeds(
        self,
        input_ids,
        input_embeds,
        embeddings_list,
        metadata,
        pad_id=None,
        modality_pad_masks=None,
    ):
        """
        Replaces the special modality tokens in input_ids with the corresponding
        modality features, and returns merged embeddings and masks.

        Args:
            embeddings_list: List of (modality_name, embedding_tensor) tuples. Includes text modality embeddings.
            labels: Optional labels aligned with input_ids (text tokens only).
            pad_id: Optional pad token id used to drop padding from the merge.
        Returns:
            final_embedding: (B, T_total, D_model)
            final_attention_mask: (B, T_total)
            position_ids: (B, T_total)
            final_labels: (B, T_total) or None
        """
        embeddings_by_modality = {m: e for m, e in embeddings_list}
        if "text" not in embeddings_by_modality:
            raise ValueError("Interleaving requires text embeddings to be present.")

        batch_size, sequence_length = input_ids.shape
        embed_dim = input_embeds.shape[-1]

        start_any = torch.zeros_like(input_ids, dtype=torch.bool)
        end_any = torch.zeros_like(input_ids, dtype=torch.bool)
        start_modality_idx = torch.full_like(input_ids, -1, dtype=torch.long)

        modalities = []

        # DNA sub-modalities (dna_reference, dna_variant) are keyed under the
        # parent "dna" entry in config.modalities/self.encoders — they get
        # their own start/end token pair (see modality_start_end_token_indices)
        # but share the single DNAEncoder/projector instance and the
        # config.modalities/["dna"] gate.
        _PARENT_MODALITY = {"dna_reference": "dna", "dna_variant": "dna"}

        for modality, (start_idx, end_idx) in (
            self.config.modality_start_end_token_indices or {}
        ).items():
            parent_modality = _PARENT_MODALITY.get(modality, modality)
            if parent_modality not in self.config.modalities or modality == "text":
                continue
            if modality not in embeddings_by_modality:
                raise ValueError(f"Missing embeddings for modality '{modality}'.")

            start_mask = input_ids == start_idx
            end_mask = input_ids == end_idx

            if (start_any & start_mask).any() or (end_any & end_mask).any():
                raise ValueError("Overlapping modality start/end tokens detected.")

            modalities.append(modality)
            start_any |= start_mask
            end_any |= end_mask
            start_modality_idx[start_mask] = len(modalities) - 1

        pad_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        if pad_id is not None:
            pad_mask = input_ids == pad_id

        slot_size = torch.ones_like(input_ids, dtype=torch.long)
        slot_size[end_any] = 0
        slot_size[pad_mask] = 0  # Padding tokens don't occupy output slots

        for i, modality in enumerate(modalities):
            start_mask = start_modality_idx == i
            if start_mask.any():
                encoder_key = _PARENT_MODALITY.get(modality, modality)
                tokens_per = int(self.encoders[encoder_key].tokens_per_instance())
                slot_size[start_mask] = tokens_per

        # sanity checks
        for modality, (start_idx, end_idx) in (
            self.config.modality_start_end_token_indices or {}
        ).items():
            if modality not in modalities:
                continue
            start_mask = input_ids == start_idx
            end_mask = input_ids == end_idx
            if torch.any(start_mask.sum(dim=-1) != end_mask.sum(dim=-1)):
                raise ValueError(f"Mismatched start/end tokens for modality '{modality}'.")
            if torch.any(start_mask[:, :-1] & (input_ids[:, 1:] != end_idx)):
                raise ValueError(f"Start/end tokens for '{modality}' must be adjacent.")
            if torch.any(start_mask[:, -1]):
                raise ValueError(f"Start token for '{modality}' cannot be last in the sequence.")

        new_token_positions = torch.cumsum(slot_size, dim=-1) - slot_size
        total_len = slot_size.sum(dim=-1)
        max_len = int(total_len.max().item()) if batch_size > 0 else 0

        # Merged-length guard (issue #123). The merged sequence (text +
        # modality-expanded slots) is what the backbone attends over. When it
        # grows too large, attention is O(T^2) and on XPU this surfaces as a
        # silent GPU write page-fault (drm_neo.cpp) under DDP+oneCCL rather than
        # a clean OutOfMemoryError — extremely expensive to diagnose (see #120,
        # which crashed at step 1040). The configured limit is the collator
        # MAX_SEQ_LENGTH (default 2048), a conservative proxy that fires well
        # before the real attention-memory/OOM cliff (~4608 on a 64GB tile) —
        # NOT the backbone positional context; do not raise it to the backbone
        # context or the fault can reappear. PR #122 caps ts_qa variates
        # upstream, but only in that one collator path; this belt-and-suspenders
        # check catches an over-length merge from any interleaved-QA data path
        # that reaches this merge. (The non-interleaved concat path in forward()
        # does not flow through here and is not covered.) No-op when unset.
        merged_limit = self.config.max_merged_seq_length
        if merged_limit is not None and max_len > merged_limit:
            worst = int(total_len.argmax().item())
            worst_len = int(total_len[worst].item())
            msg = (
                f"Merged sequence length {worst_len} (batch item {worst}) exceeds "
                f"max_merged_seq_length={merged_limit}. Oversized merged sequences "
                f"drive O(T^2) attention memory; on XPU this surfaces as a silent "
                f"GPU page-fault (issue #120/#123). Modalities present: {modalities}. "
                f"Sample metadata (prompt_len target_len): {metadata[worst]!r}."
            )
            if self.config.merged_seq_length_guard == "warn":
                logger.warning(msg + " Proceeding anyway (guard mode='warn').")
            else:
                raise ValueError(msg)

        final_embedding = torch.zeros(
            batch_size, max_len, embed_dim, dtype=input_embeds.dtype, device=input_embeds.device
        )
        final_attention_mask = torch.zeros(
            batch_size, max_len, dtype=torch.long, device=input_embeds.device
        )

        text_mask = ~(start_any | end_any | pad_mask)
        batch_indices, text_indices = torch.where(text_mask)
        text_positions = new_token_positions[batch_indices, text_indices]

        final_embedding[batch_indices, text_positions] = input_embeds[batch_indices, text_indices]
        final_attention_mask[batch_indices, text_positions] = 1

        tokens_per_instance = {
            modality: int(
                self.encoders[_PARENT_MODALITY.get(modality, modality)].tokens_per_instance()
            )
            for modality in modalities
        }
        expected_tokens = {
            modality: (start_modality_idx == i).sum(dim=-1) * tokens_per_instance[modality]
            for i, modality in enumerate(modalities)
        }

        for modality in modalities:
            modality_features = embeddings_by_modality[modality]
            if modality_features.shape[0] != batch_size:
                raise ValueError(f"Batch size mismatch for modality '{modality}'.")
            if int(expected_tokens[modality].max().item()) > modality_features.shape[1]:
                raise ValueError(f"Not enough tokens in embeddings for modality '{modality}'.")
            if modality_features.shape[-1] != embed_dim:
                raise ValueError(f"Embedding dim mismatch for modality '{modality}'.")

        labels = torch.zeros(
            batch_size, max_len, device=input_embeds.device
        ).long()  # Placeholder for labels if needed later
        _, T_total, _ = final_embedding.shape
        # Create labels filled with -100 (ignore index)
        full_labels = torch.full(
            (batch_size, T_total), -100, dtype=labels.dtype, device=labels.device
        )

        # for each element of the batch, copy in the modality features
        for b_idx in range(batch_size):
            cursors = {modality: 0 for modality in modalities}
            span_positions = torch.nonzero(start_any[b_idx], as_tuple=False).squeeze(-1).tolist()

            # For QA SFT, need to mask only the answer tokens, but we don't have a good way to identify them.
            # So use the lengths of the tokenized Q and A from the metadata
            # to accomplish this here.
            #
            # "[dna_bioreason]" sentinel — projector-only stage: full causal LM
            # loss over every text token in the merged sequence (question +
            # answer). DNA slot positions stay -100 (no discrete target exists
            # for a continuous embedding).
            if metadata[b_idx] == "[dna_bioreason]":
                text_positions_b = new_token_positions[b_idx][text_mask[b_idx]]
                text_ids_b = input_ids[b_idx][text_mask[b_idx]]
                full_labels[b_idx, text_positions_b] = text_ids_b
            else:
                prompt_len, target_len = metadata[b_idx].split(" ")
                prompt_len = int(prompt_len)
                target_len = int(target_len)
                old_len = prompt_len + target_len
                # calculate new prompt len
                prompt_growth = total_len[b_idx] - old_len
                new_prompt_len = prompt_len + prompt_growth
                labels[b_idx, new_prompt_len : new_prompt_len + target_len] = (
                    1  # Mark label positions (after prompt)
                )
                # copy in answer tokens where labels == 1

                full_labels[b_idx, labels[b_idx] == 1] = input_ids[
                    b_idx, prompt_len : prompt_len + target_len
                ]

            for pos in span_positions:
                modality_index = int(start_modality_idx[b_idx, pos].item())
                if modality_index < 0:
                    continue
                modality = modalities[modality_index]
                tokens_per = tokens_per_instance[modality]
                # indices in the final_embedding tensor for copying to
                start_out = int(new_token_positions[b_idx, pos].item())
                end_out = start_out + tokens_per
                # indices in this modalitiy's tensor to copy from
                start_feat = cursors[modality]
                end_feat = start_feat + tokens_per
                features = embeddings_by_modality[modality][b_idx, start_feat:end_feat, :]
                if features.shape[0] != tokens_per:
                    raise ValueError(f"Unexpected token count for modality '{modality}'.")

                final_embedding[b_idx, start_out:end_out] = features
                final_attention_mask[b_idx, start_out:end_out] = 1
                if modality_pad_masks and modality in modality_pad_masks:
                    modality_pad_mask = modality_pad_masks[modality][b_idx]
                    final_attention_mask[b_idx, start_out:end_out] = (
                        ~modality_pad_mask[start_feat:end_feat]
                    ).to(dtype=final_attention_mask.dtype, device=final_attention_mask.device)
                cursors[modality] = end_feat

            for modality in modalities:
                expected = int(expected_tokens[modality][b_idx].item())
                if cursors[modality] != expected:
                    raise ValueError(
                        f"Token count mismatch for modality '{modality}': expected {expected}, got {cursors[modality]}."
                    )

        position_ids = final_attention_mask.cumsum(-1) - 1
        position_ids = position_ids.masked_fill(final_attention_mask == 0, 0)

        return final_embedding, final_attention_mask, position_ids, full_labels

    # ------------------------------------------------------------------
    # Auxiliary (non-text) output decoders
    # ------------------------------------------------------------------
    # Targets are passed in `inputs` under "<decoder_name>_target", e.g.
    # inputs["time_series_target"] of shape (B, H, V). text/action are handled
    # by their own paths and never treated as auxiliary here.
    _AUX_DECODER_NAMES = ("time_series", "geometry", "graph")

    def _collect_aux_decoder_targets(
        self, inputs: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Return {decoder_name: target} for built aux decoders with a target."""
        found: dict[str, torch.Tensor] = {}
        for name in self._AUX_DECODER_NAMES:
            if name not in self.decoders:
                continue
            target = inputs.get(f"{name}_target")
            if target is not None:
                found[name] = target
        return found

    def _aux_decoder_loss(
        self,
        hidden: torch.Tensor,
        aux_targets: dict[str, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sum the losses of all auxiliary decoders that have a target."""
        total = None
        for name, target in aux_targets.items():
            _, dec_loss = self.decoders[name](
                hidden, targets=target, attention_mask=attention_mask
            )
            if dec_loss is None:
                continue
            total = dec_loss if total is None else total + dec_loss
        if total is None:
            return torch.zeros((), device=hidden.device, dtype=hidden.dtype)
        return total

    def _forward_vla(self, inputs: dict[str, torch.Tensor], labels: torch.Tensor | None = None):
        required_keys = ["text", "image_head", "image_wrist", "pose"]
        missing = [k for k in required_keys if k not in inputs]
        if missing:
            raise RuntimeError(f"Missing VLA input keys: {missing}")

        if "image" not in self.encoders or "image" not in self.projectors:
            raise RuntimeError("VLA mode requires image encoder and image projector")

        text_input_ids = inputs["text"]
        text_embeddings = self.backbone.get_input_embeddings()(text_input_ids)

        image_head = inputs["image_head"]
        image_wrist = inputs["image_wrist"]
        pose = inputs["pose"]
        if pose.ndim != 2:
            raise RuntimeError(f"Pose tensor must be (B, D), got {tuple(pose.shape)}")

        head_features = self.encoders["image"](image_head)
        wrist_features = self.encoders["image"](image_wrist)
        if getattr(self, "_encoder_frozen_cache", {}).get("image", False):
            head_features = head_features.detach()
            wrist_features = wrist_features.detach()

        head_tokens = self.projectors["image"](head_features)
        wrist_tokens = self.projectors["image"](wrist_features)
        pose_token = self.pose_embed(pose).unsqueeze(1) + self.pose_modality_embedding

        x = torch.cat([head_tokens, wrist_tokens, pose_token, text_embeddings], dim=1)
        prefix_len = head_tokens.shape[1] + wrist_tokens.shape[1] + 1

        target_device = next(self.backbone.parameters()).device
        x = x.to(dtype=self.backbone.dtype, device=target_device)

        text_attention_mask = inputs.get("text_attention_mask")
        if text_attention_mask is None:
            text_attention_mask = torch.ones(
                text_input_ids.shape,
                dtype=torch.long,
                device=text_input_ids.device,
            )
        text_attention_mask = text_attention_mask.to(device=target_device, dtype=torch.long)
        prefix_mask = torch.ones((x.shape[0], prefix_len), dtype=torch.long, device=target_device)
        attention_mask = torch.cat([prefix_mask, text_attention_mask], dim=1)

        outputs = self.backbone(
            inputs_embeds=x,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states[-1]

        text_lengths = text_attention_mask.sum(dim=1) - 1
        text_lengths = text_lengths.clamp(min=0)
        batch_indices = torch.arange(hidden_states.shape[0], device=hidden_states.device)
        token_indices = prefix_len + text_lengths
        action_features = hidden_states[batch_indices, token_indices]
        pred_action = self.action_head.predict(action_features)

        target_action = inputs.get("action") if labels is None else labels
        if target_action is None:
            return pred_action, None, None
        target_action = target_action.to(device=pred_action.device, dtype=pred_action.dtype)
        if target_action.shape != pred_action.shape:
            raise RuntimeError(
                f"Action target shape {tuple(target_action.shape)} != prediction shape {tuple(pred_action.shape)}"
            )

        loss, per_dim_mse = self.action_head.loss_terms(pred_action, target_action)
        return pred_action, loss, per_dim_mse

    def _forward_interleave_qa(
        self,
        inputs: dict[str, torch.Tensor],
        embeddings: list[tuple[str, torch.Tensor]],
        modality_pad_masks=None,
    ):
        if "text" not in inputs:
            raise ValueError("Interleaved QA requires 'text' input_ids.")
        if "_metadata" not in inputs:
                raise ValueError("Interleaved QA requires '_metadata' input.")
        
        embeddings_by_modality = {m: e for m, e in embeddings}
        input_ids = inputs["text"]
        input_embeds = embeddings_by_modality["text"]

        pad_id = self._resolve_pad_id()

        x, attention_mask, position_ids, full_labels = (
            self._merge_text_input_ids_with_modality_embeds(
                input_ids=input_ids,
                input_embeds=input_embeds,
                embeddings_list=embeddings,
                metadata=inputs["_metadata"],
                pad_id=pad_id,
                modality_pad_masks=modality_pad_masks,
            )
        )
        return x, attention_mask, position_ids, full_labels


    def _output_condition(self, inputs):
        if self.is_vla or self.backbone is None:
            raise NotImplementedError("Structured decoding currently requires the eager HF causal-LM backbone")
        embeddings = self._process_multimodal_embeddings(inputs)
        x, mask, spans, text_positions = compile_inputs(self, inputs, embeddings)
        weight = self.backbone.get_input_embeddings().weight
        x = x.to(device=weight.device, dtype=weight.dtype)
        mask = mask.to(weight.device)
        position_ids = mask.long().cumsum(-1).sub(1).clamp_min(0)
        outputs = self.backbone(
            inputs_embeds=x, attention_mask=mask, position_ids=position_ids,
            output_hidden_states=True, return_dict=True, use_cache=False,
        )
        condition = DecoderCondition(
            hidden_states=outputs.hidden_states[-1], attention_mask=mask,
            modality_spans=spans,
            provenance={"backbone": self.config.llm_backbone_id, "fusion": "interleave" if self.config.is_interleaved_qa else "prefix"},
        )
        return condition, outputs.logits, text_positions.to(weight.device), x

    def _route_outputs(self, inputs, targets=None, requested_outputs=None,
                       output_specs=None, native_context=None, decoder_kwargs=None,
                       generate=False):
        names = list(self.config.output_decoders if requested_outputs is None else requested_outputs)
        if not names or len(set(names)) != len(names):
            raise ValueError("requested_outputs must contain distinct decoder names")
        missing = set(names) - set(self.decoders)
        if missing:
            raise ValueError(f"Output decoders are not configured: {sorted(missing)}")
        targets = targets or {}
        if set(targets) - set(names):
            raise ValueError("Every target must belong to a requested output")
        condition, logits, text_positions, x = self._output_condition(inputs)
        result = DecoderResult(provenance=condition.provenance)
        for name in names:
            kwargs = dict((decoder_kwargs or {}).get(name, {}))
            if name == "text":
                if generate:
                    # HF decoder-only generation reads the last position;
                    # left-pad the compacted batch so it is valid in every row.
                    generation_x = torch.zeros_like(x)
                    generation_mask = torch.zeros_like(condition.attention_mask)
                    for row in range(x.shape[0]):
                        length = int(condition.attention_mask[row].sum())
                        generation_x[row, -length:] = x[row, :length]
                        generation_mask[row, -length:] = True
                    result.predictions[name] = self.backbone.generate(
                        inputs_embeds=generation_x, attention_mask=generation_mask, **kwargs,
                    )
                    continue
                labels = align_text_targets(targets[name], text_positions, condition.attention_mask) if name in targets else None
                prediction, loss = self.decoders[name](logits, targets=labels, **kwargs)
            else:
                decoder_condition = DecoderCondition(
                    hidden_states=condition.hidden_states,
                    attention_mask=condition.attention_mask,
                    modality_spans=condition.modality_spans,
                    native_context=dict((native_context or {}).get(name, {})),
                    output_spec=dict((output_specs or {}).get(name, {})),
                    provenance=condition.provenance,
                )
                if generate:
                    result.predictions[name] = self.decoders[name].generate_condition(decoder_condition, **kwargs)
                    continue
                prediction, loss = self.decoders[name].forward_condition(
                    decoder_condition, targets=targets.get(name), **kwargs,
                )
            result.predictions[name] = prediction
            if loss is not None:
                if loss.ndim != 0 or not torch.isfinite(loss):
                    raise ValueError(f"Decoder {name} produced a nonfinite or nonscalar loss")
                result.losses[name] = loss
                weighted = loss * self.config.decoder_loss_weights.get(name, 1.0)
                result.loss = weighted if result.loss is None else result.loss + weighted
        return result

    def forward_outputs(self, inputs, targets=None, requested_outputs=None,
                        output_specs=None, native_context=None, decoder_kwargs=None):
        """Decode named outputs without implicitly supervising instruction text."""
        return self._route_outputs(inputs, targets, requested_outputs, output_specs,
                                   native_context, decoder_kwargs)

    @torch.no_grad()
    def predict(self, inputs, requested_outputs=None, output_specs=None,
                native_context=None, decoder_kwargs=None):
        """Generate native outputs (including an image denoising loop)."""
        return self._route_outputs(inputs, requested_outputs=requested_outputs,
                                   output_specs=output_specs, native_context=native_context,
                                   decoder_kwargs=decoder_kwargs, generate=True)

    def forward(
        self, inputs: dict[str, torch.Tensor], labels: torch.Tensor | None = None,
        *, requested_outputs=None, targets=None, output_specs=None,
        native_context=None, decoder_kwargs=None,
    ):
        """
        Args:
            inputs: Dictionary mapping modality name to input tensor.
            labels: Optional labels for training (SFT). shape (B, T_text) usually.
        Returns:
            logits: (B, T_total, Vocab_Size) OR Loss object if labels provided to backbone
        """
        if requested_outputs is not None:
            if labels is not None:
                raise ValueError("Use explicit targets with requested_outputs, not legacy labels")
            return self.forward_outputs(inputs, targets, requested_outputs, output_specs,
                                        native_context, decoder_kwargs)
        if targets is not None or any(v is not None for v in (output_specs, native_context, decoder_kwargs)):
            raise ValueError("Structured decoding arguments require requested_outputs")
        if self.is_vla:
            return self._forward_vla(inputs, labels=labels)

        # 1. Get Multimodal Embeddings
        # Does a for loop over modalities
        embeddings = self._process_multimodal_embeddings(inputs)

        if not embeddings:
            raise ValueError("No valid modalities found in inputs")

        attention_mask = None
        position_ids = None
        full_labels = None
        is_interleaved_qa = bool(self.config.is_interleaved_qa)

        if is_interleaved_qa:
            text_embeddings = None
            x, attention_mask, position_ids, full_labels = self._forward_interleave_qa(
                inputs,
                embeddings=embeddings,
                modality_pad_masks=getattr(self, "_modality_pad_masks", None),
            )
        else:
            # Reorder embeddings: Non-text first, Text last
            ordered_embedding_pairs = [
                (m, e) for m, e in embeddings if m != "text"
            ] + [(m, e) for m, e in embeddings if m == "text"]
            prefix_embeddings = [e for modality, e in ordered_embedding_pairs if modality != "text"]
            text_embeddings = [e for modality, e in ordered_embedding_pairs if modality == "text"]

            ordered_embeddings = prefix_embeddings + text_embeddings

            # Concatenate tokens
            x = torch.cat(ordered_embeddings, dim=1)  # (B, T_total, D_model)
            attention_mask = self._build_prefix_content_mask(ordered_embedding_pairs)

        # --- DEBUG: Embedding Statistics (Check for "Blind Prior") ---
        if self.training and random.random() < 0.01:  # Sample 1% of steps
            with torch.no_grad():
                t_norm = 0.0
                if text_embeddings:
                    t_norm = text_embeddings[0].norm(dim=-1).mean().item()

                for m, e in embeddings:
                    if m != "text":
                        # Basic stats
                        token_norms = e.norm(dim=-1)  # (B, T)
                        p_norm_mean = token_norms.mean().item()
                        p_norm_std = token_norms.std().item()
                        p_norm_min = token_norms.min().item()
                        p_norm_max = token_norms.max().item()
                        p_elem_std = (
                            e.std().item()
                        )  # Per-element std (important for variance preservation)

                        # Get projector config info
                        extra_info = ""
                        if m in self.projectors:
                            proj = self.projectors[m]
                            # Get norm mode
                            if hasattr(proj, "norm_mode"):
                                extra_info += f", Mode={proj.norm_mode}"
                            # Get scale if available
                            if hasattr(proj, "output_scale") and proj.output_scale is not None:
                                extra_info += f", Scale={proj.output_scale.item():.6f}"

                        logger.info(
                            f"[DEBUG] {m.upper()} Proj: NormMean={p_norm_mean:.4f}, NormStd={p_norm_std:.4f}, "
                            f"NormRange=[{p_norm_min:.4f}, {p_norm_max:.4f}], ElemStd={p_elem_std:.4f} "
                            f"(TextNorm={t_norm:.4f}){extra_info}"
                        )
        # -------------------------------------------------------------

        # --- Path A: HF Backbone ---
        if self.backbone:
            # If labels provided, we need to pad them to match total sequence length?
            # Usually multimodal models mask out the image tokens in the loss.
            # Only text tokens (target) have labels.
            # We assume 'labels' passed in are for the 'text' part.

            if not is_interleaved_qa:
                full_labels = None
                if labels is not None:
                    B, T_total, _ = x.shape
                    # Create labels filled with -100 (ignore index)
                    full_labels = torch.full(
                        (B, T_total), -100, dtype=labels.dtype, device=labels.device
                    )

                    # Fill in the text part at the end
                    if text_embeddings:
                        text_len = text_embeddings[0].shape[1]
                        label_len = labels.shape[1]

                        if text_len != label_len:
                            logger.warning(
                                f"[Warning] Text Embed ({text_len}) != Label ({label_len}). Truncating mismatch to end-align."
                            )

                        # Safe Assignment (Min Length)
                        assign_len = min(text_len, label_len)
                        if assign_len > 0:
                            # clone(): without a pad id the slice below is a
                            # view of the caller's `labels` (batch["text"]), and
                            # the prompt masking would corrupt the input batch.
                            assigned = labels[:, -assign_len:].clone()
                            # Mask pad tokens to -100. Without this, cross_entropy
                            # rewards predicting pad tokens — captions are ~30% of a
                            # 512-token sequence, so loss collapses to ~0 within 30
                            # steps regardless of model size (Stage A round 1).
                            pad_id_local = self._resolve_pad_id()
                            if pad_id_local is not None:
                                assigned = assigned.masked_fill(
                                    assigned == pad_id_local, -100
                                )
                            # Mask the prompt so the loss is answer-only. The
                            # interleaved path already does this via
                            # new_prompt_len; the prefix path previously did
                            # not, so `labels=batch["text"]` supervised the
                            # question too. On SciTS that is ~75% of supervised
                            # tokens across only 4 distinct question templates —
                            # loss drops by memorising them, independent of the
                            # time series. `_prompt_len` is 0 for non-QA data
                            # (captioning), which supervises everything as before.
                            prompt_lens = inputs.get("_prompt_len")
                            if prompt_lens is not None:
                                for _b in range(assigned.shape[0]):
                                    _pl = int(prompt_lens[_b])
                                    if _pl <= 0:
                                        continue
                                    # Keep >=1 supervised token: a sample whose
                                    # answer was truncated away would otherwise
                                    # be all -100 and produce NaN loss. Must
                                    # clamp against this SAMPLE's real (unpadded)
                                    # content length, not the batch's padded
                                    # width (assigned.shape[1]) — pad tokens
                                    # were already masked to -100 above, so a
                                    # row shorter than the batch max has no
                                    # real tokens left of that padded bound,
                                    # and clamping to it is a no-op that lets
                                    # the whole row go to -100 anyway.
                                    _real_len = int((assigned[_b] != -100).sum().item())
                                    if _real_len <= 0:
                                        continue
                                    _pl = min(_pl, _real_len - 1)
                                    if _pl > 0:
                                        assigned[_b, :_pl] = -100
                            full_labels[:, -assign_len:] = assigned
                else:
                    # CRITICAL: No text embeddings means no labels to assign
                    # This happens when text is empty/missing. Log warning.
                    # The loss will be NaN if all labels are -100, so we should skip this batch.
                    logger.warning(
                        "[Warning] No text embeddings found! Labels cannot be assigned. "
                        "Batch will have all-ignored labels (loss may be NaN)."
                    )

            # Forward Backbone with inputs_embeds
            # Ensure x matches backbone dtype and device
            # We robustly find the device of the first parameter to satisfy accelerate/pipeline splitting
            target_device = next(self.backbone.parameters()).device
            x = x.to(dtype=self.backbone.dtype, device=target_device)

            # DEBUG LOG
            if hasattr(self, "global_step") and self.global_step % 100 == 0:
                logger.info(f"[MODEL] Forwarding Backbone: {type(self.backbone).__name__}")

            # CRITICAL: use_cache=False during training. Without this, HuggingFace
            # creates a DynamicCache storing KV states for every layer. These tensors
            # participate in the autograd graph (because enable_input_require_grads()
            # makes inputs_embeds require grads), and backprop through cached KV states
            # causes GPU segfaults on Intel XPU (page fault in drm_neo.cpp).
            # KV caching is only useful for autoregressive generation, not training.
            #
            # CRITICAL: Do NOT pass labels to the backbone. HuggingFace's internal
            # ForCausalLMLoss calls logits.float() to upcast BF16 logits to FP32.
            # With 256K vocab (AuroraGPT), this creates a massive tensor:
            #   BS=4, seq=2048: 4 * 2048 * 256K * 4B = 8.2 GB (+ 8.2 GB for grads)
            # This causes UR_RESULT_ERROR_OUT_OF_RESOURCES on Intel XPU.
            # Instead, we compute the loss ourselves using F.cross_entropy which
            # handles BF16 inputs via a fused kernel (no full FP32 materialization).
            #
            # CRITICAL: Bypass transformers' mask construction to avoid Intel XPU
            # UR_RESULT_ERROR_OUT_OF_RESOURCES. In transformers >= 4.57, both the
            # vmap-based mask construction AND the SDPA kernel itself can exhaust
            # Intel Unified Runtime resources after many forward passes.
            #
            # Strategy depends on attn_implementation:
            # - "sdpa": Pass 2D all-ones mask → _ignore_causal_mask_sdpa returns True
            #   on XPU → create_causal_mask returns None → SDPA uses is_causal=True
            #   (Note: SDPA kernel may still leak UR resources on some models)
            # - "eager": Pass pre-computed 4D causal mask → early_exit in
            #   _preprocess_mask_arguments (ndim==4 check) → bypasses vmap entirely.
            #   Eager attention uses matmul+softmax, avoiding the SDPA kernel leak.
            #   The 4D mask must be additive float format (0.0=attend, -inf=masked).
            B_fwd, T_fwd = x.shape[:2]
            mask_padded = bool(getattr(self.config, "mask_padded_modality_tokens", False))
            if getattr(self.backbone.config, "_attn_implementation", "sdpa") == "eager":
                # Pre-computed 4D causal mask: lower-triangular 0.0, upper -inf
                # Shape: (1, 1, T, T) — broadcasts over batch and heads
                causal_mask_fwd = torch.zeros(
                    1, 1, T_fwd, T_fwd, device=x.device, dtype=x.dtype
                )
                causal_mask_fwd.masked_fill_(
                    ~torch.tril(
                        torch.ones(T_fwd, T_fwd, device=x.device, dtype=torch.bool)
                    ),
                    torch.finfo(x.dtype).min,
                )
                if mask_padded:
                    causal_mask_fwd = causal_mask_fwd.masked_fill(
                        ~attention_mask.to(device=x.device, dtype=torch.bool)[:, None, None, :],
                        torch.finfo(x.dtype).min,
                    )
                attention_mask_fwd = causal_mask_fwd
            else:
                # 2D all-ones mask skips packed sequence detection and enables
                # is_causal=True fast path on XPU (no mask materialization)
                attention_mask_fwd = (
                    attention_mask.to(device=x.device, dtype=torch.long)
                    if mask_padded
                    else torch.ones(B_fwd, T_fwd, device=x.device, dtype=torch.long)
                )
            # Only request hidden states when an auxiliary (non-text) decoder
            # actually has a target in this batch — keeps the default text
            # path allocation-identical to before.
            aux_targets = self._collect_aux_decoder_targets(inputs)
            outputs = self.backbone(
                inputs_embeds=x,
                attention_mask=attention_mask_fwd,
                labels=None,  # Don't pass labels — compute loss ourselves
                return_dict=True,
                use_cache=False,
                output_hidden_states=bool(aux_targets),
            )

            logits = outputs.logits

            # Compute cross-entropy loss via the text OutputDecoder. This is
            # the same shifted next-token CE as before (no FP32 upcast,
            # ignore_index=-100) — LMHeadDecoder owns no params, so numerics
            # and checkpoint keys are unchanged.
            logits, loss = self.text_decoder(logits, targets=full_labels)

            # Auxiliary output decoders (e.g. time_series). Each adds its own
            # loss from the backbone's last hidden state. No-op unless a
            # matching target was supplied, so text-only training is unchanged.
            if aux_targets:
                hidden = outputs.hidden_states[-1]
                aux_loss = self._aux_decoder_loss(
                    hidden,
                    aux_targets,
                    attention_mask=attention_mask.to(device=hidden.device),
                )
                loss = aux_loss if loss is None else loss + aux_loss

            if loss is not None:
                # DEBUG LOG
                if hasattr(self, "global_step") and self.global_step % 100 == 0:
                    logger.info(f"[MODEL] Computed Loss: {loss.item()}")

                # NaN TRAP
                if torch.isnan(loss):
                    logger.error(
                        f"!!! NaN LOSS DETECTED !!! Global Step: {getattr(self, 'global_step', 'Unknown')}"
                    )
                    logger.error(f"Input Shape: {x.shape}")
                    logger.error(
                        f"Label Shape: {full_labels.shape if full_labels is not None else 'None'}"
                    )

                    # Check for NaN/Inf in inputs
                    input_nan = torch.isnan(x).any().item()
                    input_inf = torch.isinf(x).any().item()
                    logger.error(f"Input has NaN: {input_nan}, has Inf: {input_inf}")

                    # Check labels
                    if full_labels is not None:
                        label_min = (
                            full_labels[full_labels != -100].min().item()
                            if (full_labels != -100).any()
                            else -1
                        )
                        label_max = (
                            full_labels[full_labels != -100].max().item()
                            if (full_labels != -100).any()
                            else -1
                        )
                        logger.error(f"Label range (non-ignored): [{label_min}, {label_max}]")

                    # Log Metadata if available
                    if "_metadata" in inputs:
                        logger.error(f"FAILING BATCH METADATA: {inputs['_metadata']}")
                    elif "metadata" in inputs:
                        logger.error(f"FAILING BATCH METADATA: {inputs['metadata']}")
                    else:
                        logger.error(f"Inputs Keys: {list(inputs.keys())}")

                return logits, loss
            else:
                return logits, torch.tensor(0.0, device=x.device)

        # --- Path B: Custom Implementation ---
        else:
            # ... (Original Logic) ...
            B, T_total, _ = x.shape


            # 1. Generate RoPE frequencies
            freqs_cis = self.rope(x, seq_len=T_total)

            if is_interleaved_qa:
                causal = (
                    torch.triu(torch.ones(T_total, T_total, device=x.device), diagonal=1) * -1e9
                )
                mask = causal.unsqueeze(0).unsqueeze(0).expand(B, 1, T_total, T_total)
                if attention_mask is not None:
                    key_mask = (attention_mask == 0).view(B, 1, 1, T_total)
                    mask = mask.masked_fill(key_mask, -1e9)
            else:
                # 2. Construct Prefix Attention Mask (Simplified)
                mask = torch.full((1, 1, T_total, T_total), -1e9, device=x.device)
                prefix_len = sum(e.shape[1] for e in prefix_embeddings)

                if prefix_len > 0:
                    mask[:, :, :prefix_len, :prefix_len] = 0
                if text_embeddings:
                    text_len = text_embeddings[0].shape[1]
                    if prefix_len > 0:
                        mask[:, :, prefix_len:, :prefix_len] = 0
                    causal_mask = (
                        torch.triu(torch.ones(text_len, text_len, device=x.device), diagonal=1)
                        * -1e9
                    )
                    mask[:, :, prefix_len:, prefix_len:] = causal_mask
                else:
                    mask.fill_(0.0)

            # Transformer Backbone
            total_aux_loss = 0.0
            for block in self.blocks:
                x, aux_loss = block(x, mask=mask, freqs_cis=freqs_cis)
                total_aux_loss += aux_loss


            x = self.ln_f(x)
            logits = self.head(x)


            # Compute SFT loss manually if labels provided
            loss = total_aux_loss
            if labels is not None:
                # Need to align labels
                # Standard causal masking: predict next token.
                # Shift logits and labels
                shift_logits = logits[..., :-1, :].contiguous()

                if is_interleaved_qa:
                    if full_labels is None:
                        raise ValueError("Interleaved labels are missing.")
                else:
                    full_labels = torch.full(
                        (B, T_total), -100, dtype=labels.dtype, device=labels.device
                    )
                    if text_embeddings:
                        text_len = text_embeddings[0].shape[1]
                        assigned = labels
                        # Same as Path A pad-mask (line 858).
                        pad_id_local = self._resolve_pad_id()
                        if pad_id_local is not None:
                            assigned = assigned.masked_fill(
                                assigned == pad_id_local, -100
                            )
                        full_labels[:, -text_len:] = assigned

                shift_labels = full_labels[..., 1:].contiguous()

                loss_fct = nn.CrossEntropyLoss()
                ce_loss = loss_fct(
                    shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1)
                )
                loss += ce_loss

            return logits, loss

    def generate(self, inputs, max_new_tokens=20, **kwargs):
        """
        Multimodal Generation Wrapper.
        Constructs embeddings and delegates to backbone.generate() or custom loop.

        Assumption:
            - The argument `inputs` contains only the prefix prompt which the model should complete.
                For example, in a VQA setting, `inputs` would contain the question and the associated image features,
                but not the answer (which is what we want to generate).
        """
        is_interleaved_qa = bool(self.config.is_interleaved_qa)
       
        if "table" in inputs:
            logger.warning("!!! DEBUG: Table Input Detected in Forward Pass !!!")

        # 1. Get Embeddings
        embeddings = self._process_multimodal_embeddings(inputs)
        attention_mask = None

        if is_interleaved_qa:
            inputs_embeds, attention_mask, _, _ = self._forward_interleave_qa(inputs, embeddings)
        else:
            prefix_embeddings = [e for m, e in embeddings if m != "text"]
            text_embeddings = [e for m, e in embeddings if m == "text"]
            ordered_embeddings = prefix_embeddings + text_embeddings
            inputs_embeds = torch.cat(ordered_embeddings, dim=1)

        # Path A: HF Backbone
        if self.backbone:
            # Convert to backbone dtype and device
            target_device = next(self.backbone.parameters()).device
            inputs_embeds = inputs_embeds.to(dtype=self.backbone.dtype, device=target_device)

            # Get pad_token_id from tokenizer if available
            pad_token_id = kwargs.pop("pad_token_id", None)
            if (
                pad_token_id is None
                and hasattr(self, "tokenizer")
                and self.tokenizer is not None
            ):
                pad_token_id = (
                    self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
                )

            # Create attention mask, properly masking any pad tokens in the text.
            # The prefix (image/modality) tokens are always valid (no padding).
            # The text tokens may have padding from batched tokenization
            # (tokenizer(..., padding=True)). Pad tokens in the input embeddings
            # should NOT be attended to — otherwise the model sees spurious tokens
            # that differ between single-sample and batched generation, causing
            # batch-composition-dependent outputs.
            B, T, D = inputs_embeds.shape
            attention_mask = torch.ones(B, T, device=target_device, dtype=torch.long)

            # Mask out pad tokens in the text portion
            if "text" in inputs and pad_token_id is not None:
                text_ids = inputs["text"]
                if isinstance(text_ids, torch.Tensor) and text_ids.dim() >= 1:
                    # text_ids shape: (B, T_text)
                    # Prefix length = total sequence - text length
                    prefix_len = T - text_ids.shape[-1]
                    # Create mask: 0 where text token == pad_token_id, 1 elsewhere
                    text_mask = (text_ids != pad_token_id).long().to(target_device)
                    attention_mask[:, prefix_len:] = text_mask

            # --- Fix: Temporarily disable gradient checkpointing for generation ---
            # HF generate() with inputs_embeds REQUIRES use_cache=True for the KV
            # caching loop. If gradient checkpointing is active, HF will permanently
            # set model.config.use_cache=False, breaking this and all future generate
            # calls. We save/restore the state to prevent this side-effect.
            gc_was_enabled = getattr(self.backbone, "is_gradient_checkpointing", False)
            original_use_cache = getattr(self.backbone.config, "use_cache", True)

            try:
                if gc_was_enabled:
                    self.backbone.gradient_checkpointing_disable()
                # Force use_cache=True for proper KV caching during generation
                self.backbone.config.use_cache = True

                return self.backbone.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=pad_token_id,
                    **kwargs,
                )
            finally:
                # Restore original state so training is unaffected
                if gc_was_enabled:
                    self.backbone.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs={"use_reentrant": False}
                    )
                self.backbone.config.use_cache = original_use_cache

        # Path B: Custom Loop (Simple Greedy)
        else:
            logger.warning("Warning: Using slow custom generation loop.")
            curr_embeds = inputs_embeds

            for _ in range(max_new_tokens):
                logits, _ = self.forward(
                    {"_precomputed": curr_embeds}
                )  # Need to handle precomputed in forward?
                # Actually forward expects dict of modalities.
                # This is tricky without rewriting forward to accept inputs_embeds directly.
                # For now, custom generation is broken/unsupported in this quick refactor unless we add direct support.
                break
            return None  # Placeholder

    def load_pretrained_weights(self, hf_model_id: str):
        """Do nothing -- the custom-stack weight loader is a retired stub.

        The real implementation (MoE upcycling from a dense HF checkpoint)
        was dropped when the HF-backbone path landed. The body now logs and
        returns when ``self.backbone`` is set, and falls through to ``pass``
        otherwise, so it never writes a parameter either way. No call site
        remains under ``src/``.

        Args:
            hf_model_id: Hugging Face model id the weights would come from.
                Unused by the current body.

        Returns:
            ``None``, and the model is left unchanged.
        """
        # Only for custom backbone
        if self.backbone:
            logger.info("Skipping load_pretrained_weights (Backbone already loaded).")
            return


        # ... (Previous Logic preserved if needed) ...
        pass
