import logging
import os
import random
import sys
from typing import Any, cast

import hydra
import numpy as np
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

# Path Safety
# Prioritize local source code over installed packages.
# This file lives at src/train.py, so the repo root is one level up.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import DYNAMIC_LENGTH_TS_PROJECTORS, ModelConfig, TrainingConfig
from src.data.calvin_vla import CalvinVLADataset
from src.data.collate import BucketedCollator, MultimodalCollator
from src.data.multi_webdataset import (
    BucketedMultiWebDatasetWrapper,
    ModalityAwareWebDatasetWrapper,
    MultiWebDatasetWrapper,
    load_daos_config,
)
from src.data.multimodal import StreamingMultimodalDataset
from src.data.vla_collate import VLACollator
from src.hf_cache import load_cached_tokenizer
from src.modalities import Modality
from src.model import UnifiedTransformer
from src.training.distributed import (
    load_model_weights_only,
    setup_distributed,
)
from src.training.trainer_native import train_native_ddp
from src.training.trainer_zone_a import ZoneATrainer
from src.training.trainer_zone_a_vla import ZoneAVLATrainer
from src.utils.banner import print_prism_banner

logger = logging.getLogger(__name__)


def _coerce_str_or_none(value):
    """Return value as a plain str, or None if the value is None or an empty
    OmegaConf scalar container.

    Defensive: `cfg.exp.get("sweep_id", None)` should return None when the
    YAML value is `null`, but smoke runs have surfaced cases where it
    returns an empty `DictConfig({})` instead — likely a Hydra defaults-merge
    edge case. Without this coerce, the empty DictConfig hits `json.dumps`
    in perf_log.log_perf_record, gets stringified to "{}" by the default=str
    fallback, and pollutes perf.jsonl rows.
    """
    if value is None:
        return None
    # OmegaConf empty containers are falsy and not str-typed; treat as None.
    try:
        if not isinstance(value, str) and len(value) == 0:
            return None
    except TypeError:
        pass
    return str(value)


def _is_main_env() -> bool:
    """Return True when running on rank-0 (or single-process mode)."""
    rank = int(os.environ.get("RANK", os.environ.get("PMI_RANK", "0")))
    return rank == 0


def _resolve_dataset_overrides(cfg) -> dict | None:
    """Extract `data.dataset_overrides` from Hydra cfg as a plain dict.

    The per-modality smoke yamls under `src/conf/data/per_modality_smoke/*.yaml`
    declare per-dataset skip overrides; this helper unwraps OmegaConf
    containers so the streaming dataset constructor receives a normal Python
    dict. Returns None when no overrides are configured. Misconfigured
    overrides (non-mapping types) raise — silently dropping them would let a
    bad sweep yaml produce nonsense rows in perf.jsonl.
    """
    if cfg is None or not hasattr(cfg, "get"):
        return None
    data_cfg = cfg.get("data", None)
    if data_cfg is None:
        return None
    if not hasattr(data_cfg, "get"):
        return None
    overrides = data_cfg.get("dataset_overrides", None)
    if overrides is None:
        return None
    # OmegaConf -> plain dict so .update() in the constructor doesn't see
    # ListConfig/DictConfig types that misbehave with arbitrary keys.
    if isinstance(overrides, DictConfig):
        return OmegaConf.to_container(overrides, resolve=True)  # type: ignore[return-value]
    if isinstance(overrides, dict):
        return overrides
    raise TypeError(
        f"data.dataset_overrides must be a mapping, got {type(overrides).__name__}"
    )


def _coerce_int_list(value: Any) -> list[int]:
    if isinstance(value, int):
        return [int(value)]
    if isinstance(value, list | tuple):
        return [int(x) for x in value]
    try:
        return [int(x) for x in list(value)]
    except TypeError:
        return [int(value)]


def _resolve_timeomni_strides(
    patch_lens: list[int], stride_cfg: Any | None
) -> list[int]:
    if stride_cfg is None:
        return list(patch_lens)
    strides = _coerce_int_list(stride_cfg)
    if len(strides) == 1 and len(patch_lens) > 1:
        strides = strides * len(patch_lens)
    if len(strides) != len(patch_lens):
        raise ValueError(
            f"timeomni_stride len={len(strides)} must match "
            f"timeomni_patch_len len={len(patch_lens)} or be scalar"
        )
    return [int(x) for x in strides]


def _local_shards_active(
    local_shards_dir: str, allowed_modality: str, model_modalities
) -> bool:
    """Gate for the legacy LOCAL_SHARDS_DIR fast path in main().

    Returns True iff the staged shard directory exists AND its modality is in
    the model's modality list. See PR #90 — closes the symmetric leak of PR
    #73's WEBDATASET_LOCAL_PATH gate at the train.py legacy fast path.
    """
    if not local_shards_dir or not os.path.isdir(local_shards_dir):
        return False
    return allowed_modality in list(model_modalities)


def _rprint(*args, **kwargs):
    """Rank-aware info-log: only emits on rank 0.

    Backwards-compatible signature with the previous print()-based helper,
    so existing call sites that used flush=True or sep=... still work.
    Drops print-only kwargs (`flush`, `file`, `sep`, `end`) silently; the
    message is joined like print would and routed through `logger.info`.
    """
    if not _is_main_env():
        return
    sep = kwargs.pop("sep", " ")
    # Drop print-only kwargs that logger doesn't accept
    for k in ("flush", "file", "end"):
        kwargs.pop(k, None)
    msg = sep.join(str(a) for a in args)
    logger.info(msg)



def analyze_model(model, config):
    if not _is_main_env():
        return
    logger.info("\n" + "=" * 50)
    logger.info("       MODEL ANALYSIS REPORT")
    logger.info("=" * 50)

    # 1. Projector Weights
    logger.info("\n--- Trainable Projector Weights ---")
    total_proj_params = 0
    for name, proj in model.projectors.items():
        params = sum(p.numel() for p in proj.parameters() if p.requires_grad)
        logger.info(f"  {name.ljust(15)}: {params:,} parameters")
        total_proj_params += params
    logger.info(f"  TOTAL TRAINABLE: {total_proj_params:,}")


    # 2. Encoder Embedding Sizes
    logger.info("\n--- Encoder Output Dims (Embedding Sizes) ---")
    logger.info(f"  Image (SigLIP) : {config.d_img}")
    logger.info(f"  Table          : {config.d_table}")
    logger.info(f"  TimeSeries     : {config.d_ts}")
    logger.info(f"  Geometry       : {config.d_geo}")
    logger.info(f"  Graph          : {config.d_graph}")
    logger.info(f"  DNA            : {config.d_dna}")


    # 3. Joint Input Size
    logger.info("\n--- Joint Input to Transformer ---")
    logger.info(f"  Joint Dimension (d_model): {config.d_text}")
    logger.info(f"  Backbone                 : {config.llm_backbone_id}")
    logger.info("=" * 50 + "\n")

    logger.info("=" * 50 + "\n")


def analyze_dataset(dataset, train_config):
    if not _is_main_env():
        return
    logger.info("\n" + "=" * 50)
    logger.info("       DATA ANALYSIS REPORT")
    logger.info("=" * 50)

    logger.info("=" * 50)

    # 1. Training Volume
    seq_len_text = 128
    seq_len_modal = 64

    total_samples = train_config.max_steps * train_config.batch_size
    tokens_per_sample = seq_len_text + seq_len_modal
    tokens_per_batch = train_config.batch_size * tokens_per_sample

    est_total_tokens = total_samples * tokens_per_sample

    logger.info("\n--- Training Volume ---")
    logger.info(f"  Max Steps          : {train_config.max_steps:,}")
    logger.info(f"  Batch Size         : {train_config.batch_size}")
    logger.info(f"  Tokens per Batch   : {tokens_per_batch:,} (Approx)")
    logger.info(f"  Total Samples      : {total_samples:,}")
    logger.info(f"  Total Seen Tokens  : {est_total_tokens:,}")


    # 2. Modality Streams Status
    logger.info("\n--- Active Data Streams ---")
    if hasattr(dataset, "load_status"):
        for k, v in dataset.load_status.items():
            logger.info(f"  {k.ljust(20)}: {v}")
    elif hasattr(dataset, "get_stats"):
        # MultiWebDatasetWrapper provides stats via get_stats()
        stats = dataset.get_stats()
        for ds in stats.get("datasets", []):
            status = f"Active (weight={ds['weight']:.2f}, {ds['samples']} samples)"
            logger.info(f"  {ds['name'].ljust(20)}: {status}")
    else:
        logger.info("  (Dataset stats not available)")
    logger.info("=" * 50 + "\n")


def analyze_vla_dataset(dataset, train_config):
    if not _is_main_env():
        return
    logger.info("\n" + "=" * 50)
    logger.info("       CALVIN VLA DATA REPORT")
    logger.info("=" * 50)
    logger.info(f"  Max Steps          : {train_config.max_steps:,}")
    logger.info(f"  Batch Size         : {train_config.batch_size}")
    logger.info(f"  Split              : {dataset.split}")
    logger.info(f"  Max Chunk          : {dataset.max_chunk}")
    logger.info(f"  Episodes Included  : {len(dataset.episode_ids):,}")
    logger.info(f"  Samples Included   : {len(dataset):,}")
    logger.info("=" * 50 + "\n")


@hydra.main(config_path="conf", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    if _is_main_env():
        print_prism_banner("Foundational Multimodal Training")
        logger.info(f"--- Starting PRISM Training [{cfg.exp.id}] ---")
        logger.info(OmegaConf.to_yaml(cfg))

    # 0. Determine execution path
    use_native_fsdp = os.environ.get("USE_NATIVE_FSDP", "0") == "1"
    use_native_ddp = os.environ.get("USE_NATIVE_DDP", "0") == "1"

    # Defaults for Accelerate path (setup_distributed is native-only).
    def _env_int(keys, default):
        for key in keys:
            val = os.environ.get(key, "")
            if val and val.strip():
                try:
                    return int(val)
                except ValueError:
                    continue
        return default

    rank = _env_int(["RANK", "PMI_RANK", "PALS_RANKID"], 0)
    world_size = _env_int(["WORLD_SIZE", "PMI_SIZE", "PALS_SIZE"], 1)
    local_rank = _env_int(["LOCAL_RANK", "PMI_LOCAL_RANK", "PALS_LOCAL_RANKID"], 0)
    device = cfg.training.get("device", "cpu")

    # Only call setup_distributed() for native DDP/FSDP paths.
    # For Accelerate path, Accelerator handles distributed init internally.
    # Calling both causes "Fatal error in internal_Init_thread" (double MPI init).
    if use_native_fsdp or use_native_ddp:
        # 0. Setup Distributed Environment (mpi4py-based for multi-node)
        rank, world_size, local_rank, device = setup_distributed()
        if rank == 0:
            logger.info(
                f"[Distributed] Rank {rank}/{world_size}, Local Rank {local_rank}, Device: {device}"
            )

    # Seed plumbing: `system.seed` lives in src/conf/config.yaml (default 42) but
    # no code read it before. Stage A IsoFLOP's variance-floor measurement needs
    # the BASE@3e17 seed-replicas to actually diverge, so seed every RNG the
    # trainer touches. Set the same base seed on every rank — downstream
    # samplers/collators add `rank` themselves where they need rank-specific
    # streams (e.g. DistributedSampler), and starting from identical RNG state
    # makes runs reproducible across world-size changes.
    _seed_cfg = cfg.get("system", None)
    seed = int(_seed_cfg.get("seed", 42)) if _seed_cfg is not None else 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if rank == 0:
        logger.info(f"[Seed] Set RNG state from system.seed={seed}")

    # 1. Configuration Mapping
    from hydra.core.hydra_config import HydraConfig

    try:
        hydra_out = HydraConfig.get().runtime.output_dir
    except Exception:
        hydra_out = os.getcwd()

    ts_projector = str(cfg.model.get("ts_projector", "linear"))
    timeomni_patch_lens = _coerce_int_list(cfg.model.get("timeomni_patch_len", 16))
    timeomni_strides = _resolve_timeomni_strides(
        timeomni_patch_lens, cfg.model.get("timeomni_stride", None)
    )
    timeomni_max_patches = int(cfg.model.get("timeomni_max_patches", 100))
    ts_max_length = int(cfg.model.get("max_ts_length", 512))
    if ts_projector == "timeomni":
        # TimeOmni patch embedding applies right replication padding by stride
        # before unfold, yielding:
        #   num_patches = floor(T / stride) + 1    (when patch_len == stride)
        # To cap at `timeomni_max_patches`, budget T as (max_patches - 1) * stride.
        largest_patch_idx = max(
            range(len(timeomni_patch_lens)), key=lambda i: timeomni_patch_lens[i]
        )
        budget_stride = int(timeomni_strides[largest_patch_idx])
        ts_max_length = budget_stride * max(1, timeomni_max_patches - 1)
        if rank == 0:
            logger.info(
                f"[TimeOmni] max_ts_length computed as stride*(max_patches-1) "
                f"= {budget_stride}*({timeomni_max_patches}-1)={ts_max_length}"
            )

    model_config = ModelConfig(
        llm_backbone_id=cfg.model.backbone_id,
        llm_tokenizer_id=cfg.model.get("tokenizer_id", cfg.model.backbone_id),
        freeze_backbone=cfg.model.freeze_backbone,
        freeze_encoders=cfg.model.freeze_encoders,
        is_vla=cfg.model.get("is_vla", False),
        pose_dim=cfg.model.get("pose_dim", 15),
        action_dim=cfg.model.get("action_dim", 7),
        modalities=list(cfg.model.modalities),
        is_interleaved_qa=cfg.model.get("is_interleaved_qa", False),
        # Merged-length guard (issue #123): bound the interleave-merged sequence
        # by the same MAX_SEQ_LENGTH the collators use, so an over-length merge
        # from the interleaved-QA path fails loud instead of triggering a silent
        # XPU page-fault (#120). Overridable via MERGED_SEQ_LENGTH_GUARD=warn.
        max_merged_seq_length=int(os.environ.get("MAX_SEQ_LENGTH", "2048")),
        merged_seq_length_guard=os.environ.get("MERGED_SEQ_LENGTH_GUARD", "error"),
        # Dimensions overrides
        d_img=cfg.model.get("d_img", 1152),
        d_text=cfg.model.get("d_text", 1280),  # Usually matches backbone
        d_table=cfg.model.get("d_table", 768),
        d_ts=cfg.model.get("d_ts", 512),
        d_geo=cfg.model.get("d_geo", 512),
        d_graph=cfg.model.get("d_graph", 768),
        d_dna=cfg.model.get("d_dna", 1024),
        dna_model_name=cfg.model.get(
            "dna_model_name", "InstaDeepAI/nucleotide-transformer-2.5b-multi-species"
        ),
        dna_is_evo2=cfg.model.get("dna_is_evo2", False),
        image_encoder_id=cfg.model.get(
            "image_encoder_id", "google/siglip2-base-patch16-224"
        ),
        image_processor_id=cfg.model.get("image_processor_id", None),
        image_processor_strict=cfg.model.get("image_processor_strict", False),
        image_size=cfg.model.get("image_size", 224),
        image_mean=tuple(cfg.model.get("image_mean", (0.5, 0.5, 0.5))),
        image_std=tuple(cfg.model.get("image_std", (0.5, 0.5, 0.5))),
        # === Timeseries ===
        is_timeseries=cfg.model.get("is_timeseries", False),
        ts_encoder_id=cfg.model.get(
            "ts_encoder_id", "Salesforce/moirai-2.0-R-small"
        ),
        intern_s2_sampling_rate=cfg.model.get("intern_s2_sampling_rate", 1.0),
        # Honor the selected encoder; otherwise an unsupported full preset
        # would silently fall back to ModelConfig's linear input encoder.
        ts_projector=cfg.model.get("ts_projector", "linear"),
        ts_variates=cfg.model.get("ts_variates", 1),
        ts_forecast_horizon=cfg.model.get("ts_forecast_horizon", 96),
        max_ts_length=cfg.model.get("max_ts_length", 512),
        normalize_ts_in_encoder=cfg.model.get("normalize_ts_in_encoder", False),
        timeomni_patch_len=cfg.model.get("timeomni_patch_len", 16),
        timeomni_stride=cfg.model.get("timeomni_stride", None),
        timeomni_d_model=cfg.model.get("timeomni_d_model", 512),
        timeomni_dropout=cfg.model.get("timeomni_dropout", 0.1),
        timeomni_ts_tokens=cfg.model.get("timeomni_ts_tokens", 100),
        timeomni_max_patches=timeomni_max_patches,
        modality_start_end_token_indices=cfg.model.get("modality_start_end_token_indices", {}),
        # Pass Text Dropout from training or model config (YAML adds it to training usually, check both)
        text_dropout=cfg.training.get("text_dropout", 0.0),
        # === Projector Configuration (Ablation-Ready) ===
        projector_norm_mode=cfg.model.get("projector_norm_mode", "layernorm"),
        projector_target_norm=cfg.model.get("projector_target_norm", 0.25),
        projector_modality_embed_pos=cfg.model.get("projector_modality_embed_pos", "after_norm"),
        projector_modality_embed_scale=cfg.model.get("projector_modality_embed_scale", 0.02),
        # === Text Statistics Matching (for match_text_stats and match_text_elemstats modes) ===
        projector_text_norm_mean=cfg.model.get("projector_text_norm_mean", 0.25),
        projector_text_norm_std=cfg.model.get("projector_text_norm_std", 0.05),
        projector_text_elem_mean=cfg.model.get("projector_text_elem_mean", 0.0),
        projector_text_elem_std=cfg.model.get("projector_text_elem_std", 0.006),
        projector_norm_clip_min=cfg.model.get("projector_norm_clip_min", 0.1),
        projector_norm_clip_max=cfg.model.get("projector_norm_clip_max", 0.5),
        # IsoFLOP projector capacity knobs: PR-98 (commit 3ef83a6) added the
        # fields to ModelConfig dataclass + every src/conf/model/*.yaml, but
        # missed the explicit Hydra→ModelConfig wire-through here. Without
        # these two lines, every variant runs as BASE (hm=1, nl=2) no
        # matter what `model.projector_hidden_mult=N` says on the cmd line.
        # Surfaced during the Smoke 4 replay on 2026-05-27.
        projector_hidden_mult=cfg.model.get("projector_hidden_mult", 1),
        projector_num_layers=cfg.model.get("projector_num_layers", 2),
        # Attention implementation: "sdpa" (default), "eager" (avoids SDPA UR resource leak on XPU)
        attn_implementation=cfg.model.get("attn_implementation", "sdpa"),
        # Output decoders. Default ["text"] preserves text-only behavior; the
        # is_vla=True back-compat mapping to ["text","action"] happens in
        # ModelConfig.__post_init__, so legacy VLA configs need not set this.
        output_decoders=list(cfg.model.get("output_decoders", ["text"])),
        decoder_loss_weights=dict(cfg.model.get("decoder_loss_weights") or {}),
        # to_container()'s declared return type is a broad union (dict | list |
        # str | None | Any); decoder_configs is a mapping in every model schema
        # (ModelConfig.decoder_configs is a dict), so the result is a dict here.
        decoder_configs=cast(
            dict,
            OmegaConf.to_container(
                cfg.model.get("decoder_configs", {}), resolve=True
            ),
        )
        if cfg.model.get("decoder_configs", None) is not None
        else {},
    )


    # Allow override of dimensions if specified in cfg (not implemented fully in hydra yaml yet, relying on defaults)


    train_config = TrainingConfig(
        batch_size=cfg.training.batch_size,
        max_steps=cfg.training.max_steps,
        finite_epoch_steps=cfg.training.get("finite_epoch_steps", 0),
        learning_rate=cfg.training.get("learning_rate", 1e-4),
        weight_decay=cfg.training.get("weight_decay", 0.01),
        warmup_steps=cfg.training.get("warmup_steps", 10),
        device=cfg.training.device,
        vocab_size=model_config.vocab_size,
        gradient_accumulation_steps=cfg.training.get("gradient_accumulation_steps", 1),
        save_every_n_steps=cfg.training.get("save_every_n_steps", 1000),
        viz_every_n_steps=cfg.training.get("viz_every_n_steps", 500),
        wandb_project=cfg.wandb.project,
        wandb_entity=cfg.wandb.get("entity", None),
        wandb_mode=cfg.wandb.get("mode", "online"),
        wandb_run_name=f"{cfg.exp.id}-{cfg.exp.variant}",
        wandb_run_id=cfg.training.get("wandb_run_id", None),
        resume_from_checkpoint=cfg.training.get("resume_from_checkpoint", None),
        resume_weights_only=cfg.training.get("resume_weights_only", None),
        # Force checkpoints to Hydra Output Dir
        output_dir=os.path.join(hydra_out, "checkpoints"),
        # Molmo / Advanced Optimization mappings
        scheduler_type=cfg.training.get("scheduler_type", "cosine"),
        min_lr_ratio=cfg.training.get("min_lr_ratio", 0.1),
        wsd_decay_ratio=cfg.training.get("wsd_decay_ratio", 0.1),
        wsd_decay_steps=cfg.training.get("wsd_decay_steps", None),
        # Differential Learning Rates
        lr_connector=cfg.training.get("lr_connector", None),
        lr_vit=cfg.training.get("lr_vit", None),
        lr_llm=cfg.training.get("lr_llm", None),
        # Differential Warmup
        warmup_steps_connector=cfg.training.get("warmup_steps_connector", 200),
        warmup_steps_main=cfg.training.get("warmup_steps_main", 2000),
        # Freezing Overrides
        freeze_llm=cfg.training.get("freeze_llm", True),
        freeze_vit=cfg.training.get("freeze_vit", True),
        freeze_connector=cfg.training.get("freeze_connector", False),
        task=cfg.training.get("task", "vlm"),
        data_num_workers=cfg.training.get("data_num_workers", 4),
        validation_batches=cfg.training.get("validation_batches", 8),
        calvin_root=cfg.training.get("calvin_root", None),
        calvin_split=cfg.training.get("calvin_split", "train"),
        calvin_max_chunk=cfg.training.get("calvin_max_chunk", 20),
        calvin_text_max_length=cfg.training.get("calvin_text_max_length", 128),
        calvin_strict_integrity=cfg.training.get("calvin_strict_integrity", False),
        calvin_max_skipped_fraction=cfg.training.get("calvin_max_skipped_fraction", 1.0),
        calvin_loader=cfg.training.get("calvin_loader", "map"),
        calvin_webdataset_root=cfg.training.get("calvin_webdataset_root", None),
        calvin_webdataset_storage=cfg.training.get(
            "calvin_webdataset_storage", "lustre"
        ),
        # Coerce to plain str|None — observed in smoke runs that `cfg.exp.get`
        # can return an empty `DictConfig({})` for nullable scalar fields
        # under some Hydra defaults shapes, which then surfaces in perf.jsonl
        # as `"sweep_id": "{}"` after json.dumps's default=str fallback.
        sweep_id=_coerce_str_or_none(cfg.exp.get("sweep_id", None)),
        preset=_coerce_str_or_none(cfg.exp.get("preset", None)),
        seed=seed,
        # BioReason DNA/KEGG training knobs
        bioreason_dataset=cfg.training.get("bioreason_dataset", None),
        dna_truncation_per_side=cfg.training.get("dna_truncation_per_side", 1024),
        max_dna_length=cfg.training.get("max_dna_length", 1024),
        kegg_eval_max_examples=cfg.training.get("kegg_eval_max_examples", None),
        use_reasoning_traces=cfg.training.get("use_reasoning_traces", True),
        use_class_weights=cfg.training.get("use_class_weights", False),
        class_weight_max=cfg.training.get("class_weight_max", 10.0),
        lora_enabled=cfg.training.get("lora_enabled", False),
        lora_r=cfg.training.get("lora_r", 32),
        lora_alpha=cfg.training.get("lora_alpha", 64),
        lora_dropout=cfg.training.get("lora_dropout", 0.05),
        lora_target_modules=cfg.training.get("lora_target_modules", None),
        resume_lora_alpha=cfg.training.get("resume_lora_alpha", None),
        allow_missing_resume_adapter=cfg.training.get("allow_missing_resume_adapter", False),
        grpo_num_generations=cfg.training.get("grpo_num_generations", 8),
        grpo_max_completion_length=cfg.training.get("grpo_max_completion_length", 800),
        grpo_temperature=cfg.training.get("grpo_temperature", 1.0),
        grpo_top_p=cfg.training.get("grpo_top_p", 0.95),
        grpo_top_k=cfg.training.get("grpo_top_k", 20),
        grpo_beta=cfg.training.get("grpo_beta", 0.0),
        grpo_reward_functions=cfg.training.get("grpo_reward_functions", None),
    )
    if train_config.wandb_mode:
        os.environ["WANDB_MODE"] = train_config.wandb_mode
    if train_config.wandb_entity:
        os.environ["WANDB_ENTITY"] = train_config.wandb_entity

    # Inject eval_interval logic
    if getattr(cfg.training, "debug_mode", False) or cfg.training.max_steps < 100:
        _rprint("Debug Mode Detected: Setting Eval Interval to 2 steps")
        train_config.eval_every_n_steps = 2
    else:
        train_config.eval_every_n_steps = getattr(cfg.training, "eval_every_n_steps", 500)

    # IsoFLOP eval gate: PR-98 (commit 3ef83a6) added `eval_enabled` to the
    # TrainingConfig dataclass + every src/conf/training/*.yaml, but missed
    # the explicit Hydra→TrainingConfig wire-through here. Without this
    # line, the trainer reads the dataclass default (False) and the eval
    # gate never fires no matter what the YAML / CLI says. Surfaced during
    # the Smoke 4 replay on 2026-05-27.
    train_config.eval_enabled = getattr(cfg.training, "eval_enabled", False)

    # Debug print
    _rprint(
        f"Traing Config: Steps={train_config.max_steps}, EvalEvery={train_config.eval_every_n_steps}"
    )

    # 2. Model Initialization
    _rprint("Initializing Model...", flush=True)
    model = UnifiedTransformer(model_config)


    if model.backbone:
        _rprint("Enabling Gradient Checkpointing on Backbone...", flush=True)
        model.backbone.gradient_checkpointing_enable()
        model.backbone.enable_input_require_grads()




        # CRITICAL: Disable KV caching during training. Without this, HuggingFace
        # creates a DynamicCache on every forward pass, storing KV states for all
        # layers. These tensors are part of the autograd graph (via
        # enable_input_require_grads), and backprop through them causes GPU segfaults
        # on Intel XPU. KV caching is only needed for autoregressive generation.
        # The forward() call in model.py also passes use_cache=False explicitly,
        # but setting it here ensures the config-level default is also overridden.
        if hasattr(model.backbone, "config"):
            model.backbone.config.use_cache = False
            logger.info("Set backbone config.use_cache = False for training")

        # Gradient checkpointing: controls activation memory during backward pass.
        # Even for frozen-backbone projector-only training, gradient checkpointing
        # is REQUIRED when enable_input_require_grads() is used, because the entire
        # backward graph through ALL backbone layers is retained for gradient flow
        # back to the projector. Without checkpointing, all intermediate activations
        # are kept in memory, which can exhaust GPU resources on Intel XPU (causing
        # UR_RESULT_ERROR_OUT_OF_RESOURCES). Checkpointing trades recompute for memory
        # by discarding and recomputing layer activations during backward.
        #
        # NOTE: gradient_checkpointing_enable() also sets use_cache=False internally,
        # which is important — see the use_cache=False fix above.
        # Use train_config.freeze_llm as the authoritative freeze decision.
        backbone_will_train = not train_config.freeze_llm
        gc_freq = int(os.environ.get("GRAD_CKPT_FREQ", "1"))
        if gc_freq == 0:
            logger.info("Gradient Checkpointing: DISABLED (GRAD_CKPT_FREQ=0)")
        else:
            # use_reentrant=False is REQUIRED for DDP static_graph compatibility.
            # The default (use_reentrant=True) doesn't trigger autograd hooks in
            # the order DDP expects, causing "Empty bucket specified" errors
            # during _rebuild_buckets() on the second forward pass.
            model.backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            inner_model = getattr(model.backbone, "model", model.backbone)
            layers = getattr(inner_model, "layers", None)
            if layers is not None and gc_freq > 1:
                n_ckpt = 0
                for i, layer in enumerate(layers):
                    if hasattr(layer, "gradient_checkpointing"):
                        if i % gc_freq == 0:
                            layer.gradient_checkpointing = True
                            n_ckpt += 1
                        else:
                            layer.gradient_checkpointing = False
                logger.info(
                    f"Gradient Checkpointing: {n_ckpt}/{len(layers)} layers "
                    f"(every {gc_freq} layers)",
                )
            else:
                n_layers = len(layers) if layers else "?"
                mode = (
                    "E2E" if backbone_will_train else "projector-only (frozen backbone)"
                )
                logger.info(
                    f"Gradient Checkpointing: ALL {n_layers} layers ({mode})",
                )

    # --- Apply Training-Time Freeze Overrides ---
    # The model config may have freeze_encoders=True, but training config can override
    # with freeze_vit=False to enable encoder training with differential LR.
    #
    # `freeze_vit` is legacy naming from the vision-only era, but it governs the
    # modality encoder generally — so unfreeze every registered encoder, not just
    # "image". Hardcoding "image" meant a time-series (or graph/table/geometry)
    # run logged "No image encoder found to unfreeze!" and left the encoder
    # frozen, making freeze_vit=False a silent no-op for every non-image
    # modality. It only appeared to work for TS because the TimeOmni model
    # configs already set freeze_encoders=false at construction.
    if not train_config.freeze_vit:
        logger.info("\n*** UNFREEZING ENCODERS (freeze_vit=False) ***")
        encoders = getattr(model, "encoders", None) or {}
        if encoders:
            total_unfrozen = 0
            for modality, encoder in encoders.items():
                num_unfrozen = 0
                for param in encoder.parameters():
                    param.requires_grad = True
                    num_unfrozen += param.numel()
                # Update the frozen cache
                if hasattr(model, "_encoder_frozen_cache"):
                    model._encoder_frozen_cache[modality] = False
                total_unfrozen += num_unfrozen
                logger.info(
                    f"  Unfroze {num_unfrozen:,} parameters in {modality} encoder"
                )
            logger.info(f"  Total unfrozen: {total_unfrozen:,}")
            logger.info(f"  Encoder LR will be: {train_config.lr_vit}")
        else:
            logger.warning("  WARNING: No modality encoders found to unfreeze!")
    else:
        logger.info("Encoders remain FROZEN (freeze_vit=True)")

    # --- Apply Training-Time LLM Freeze Override ---
    # The model config sets freeze_backbone=True, but training config can override
    # with freeze_llm=False to enable E2E training with differential LR
    if not train_config.freeze_llm:
        logger.info("\n*** UNFREEZING LLM BACKBONE (freeze_llm=False) ***")
        if hasattr(model, "backbone") and model.backbone is not None:
            num_unfrozen = 0
            for param in model.backbone.parameters():
                param.requires_grad = True
                num_unfrozen += param.numel()
            logger.info(f"  Unfroze {num_unfrozen:,} parameters in LLM backbone")
            logger.info(f"  LLM LR will be: {train_config.lr_llm}")
        else:
            logger.warning("  WARNING: No backbone found to unfreeze!")
    else:
        logger.info("LLM Backbone remains FROZEN (freeze_llm=True)")

    try:
        analyze_model(model, model_config)
    except Exception as e:
        _rprint(f"Analysis Failed: {e}", flush=True)

    task_type = cfg.training.get("task", "vlm")
    is_vla_task = bool(model_config.is_vla or task_type == "vla_calvin")
    if model_config.is_vla and task_type != "vla_calvin":
        _rprint(
            "Warning: model.is_vla=True while training.task!=vla_calvin. Enabling VLA path.",
            flush=True,
        )
    if is_vla_task and not model_config.is_vla:
        raise RuntimeError("training.task=vla_calvin requires model.is_vla=true")

    is_bio_task = task_type in ("bioreason_sft", "bioreason_grpo", "bioreason_projector")
    # StreamingMultimodalDataset kwargs specific to BioReason DNA/KEGG runs.
    # is_sft=False for the GRPO stage (no assistant turn appended; answer kept
    # for reward computation), True for SFT/projector (assistant turn
    # appended, loss masked to the answer). is_projector_only selects the
    # "[dna_bioreason]" full-sequence-loss sentinel for the projector-only
    # alignment stage, distinct from is_sft — see multimodal.py's
    # _finalize_dna_item / model.py's label-masking branch.
    bio_dataset_kwargs = (
        {
            "use_reasoning_traces": train_config.use_reasoning_traces,
            "model_name": "dna-llm" if Modality.DNA in model_config.modalities else "llm",
            "is_sft": task_type != "bioreason_grpo",
            "task": task_type,
            "use_class_weights": train_config.use_class_weights,
            "class_weight_max": train_config.class_weight_max,
            "is_projector_only": task_type == "bioreason_projector",
        }
        if is_bio_task
        else {}
    )

    # 3. Tokenizer / Data
    _rprint("Initializing Tokenizer...", flush=True)

    # Heterogeneous: AutoTokenizer subclass on the happy path, DummyTokenizer
    # only for custom/no-HF-backbone tests and legacy dummy-data flows.
    tokenizer: Any = None

    # --- Fix for HuggingFace cache race condition on multi-node ---
    # Only local_rank 0 loads first to populate the cache, then others load.
    # CRITICAL: All ranks must participate in each barrier call.
    tokenizer_id = model_config.llm_tokenizer_id if model_config.llm_tokenizer_id else model_config.llm_backbone_id
    requires_real_tokenizer = bool(tokenizer_id)
    if tokenizer_id and dist.is_initialized():
        if local_rank == 0:
            # Local rank 0 loads tokenizer first (populates /tmp cache)
            try:
                tokenizer = load_cached_tokenizer(
                    tokenizer_id,
                    trust_remote_code=True,
                )
            except Exception as e:
                logger.error(f"Could not load tokenizer from tokenizer_id={tokenizer_id}: {e}")
                tokenizer = None
        else:
            tokenizer = None

        # ALL ranks synchronize here - rank 0 has loaded, others are ready
        dist.barrier()

        # Non-zero local ranks now load from cache
        if local_rank != 0 and tokenizer is None:
            try:
                tokenizer = load_cached_tokenizer(
                    tokenizer_id,
                    trust_remote_code=True,
                )
            except Exception as e:
                logger.error(f"Rank {rank} could not load tokenizer_id={tokenizer_id}: {e}")
                tokenizer = None

        # ALL ranks synchronize again after loading
        dist.barrier()
    elif tokenizer_id:
        # Single process mode
        try:
            tokenizer = load_cached_tokenizer(
                tokenizer_id,
                trust_remote_code=True,
            )
        except Exception:
            tokenizer = None

    if tokenizer is None and requires_real_tokenizer:
        if is_vla_task:
            raise RuntimeError(
                f"VLA mode requires tokenizer load from "
                f"llm_backbone_id={model_config.llm_backbone_id} "
                f"(tried tokenizer_id={tokenizer_id!r} with local_files_only=True). "
                "Stage the tokenizer/model cache before launching."
            )
        raise RuntimeError(
            f"Could not load tokenizer {tokenizer_id!r} with local_files_only=True. "
            "Stage the tokenizer/model cache before launching; refusing to fall "
            "back to DummyTokenizer for real HF training."
        )

    # Fallback to dummy tokenizer only when no real HF tokenizer was configured.
    if tokenizer is None:

        class DummyTokenizer:
            def __init__(self):
                self.pad_token_id = 0
                self.vocab_size = 32000

            def __call__(self, text, **kwargs):
                return type("obj", (object,), {"input_ids": torch.randint(0, 32000, (1, 20))})

        tokenizer = DummyTokenizer()

    # Critical: Attach tokenizer to model for Evaluators/Trainer usage
    setattr(model, "tokenizer", tokenizer)  # noqa: B010 — nn.Module attr typed as Tensor|Module

    # DNA special-token fix: UnifiedTransformer.__init__ registers any
    # string-keyed modality_start_end_token_indices entries (e.g. DNA's
    # <dna_ref_start>/<dna_ref_end>/<dna_var_start>/<dna_var_end>) as special
    # tokens on model.backbone_tokenizer and resizes the embedding table. The
    # `tokenizer` loaded above is a *separate* instance that never had those
    # tokens added — it would encode "<dna_ref_start>" as several wrong
    # sub-word-piece ids, so _merge_text_input_ids_with_modality_embeds's
    # exact-id lookup would silently never match, DNA embeddings would never
    # get spliced into the sequence, and the projector would receive zero
    # gradient — no crash, just silently wrong training. Adopt
    # model.backbone_tokenizer as authoritative whenever it has grown beyond
    # the base vocab (i.e. special tokens were added), so the tokenizer that
    # encodes the actual training data matches the one the model resolved
    # its splice-token ids against.
    if (
        hasattr(model, "backbone_tokenizer")
        and model.backbone_tokenizer is not None
        and len(model.backbone_tokenizer) > len(tokenizer)
    ):
        tokenizer = model.backbone_tokenizer
        setattr(model, "tokenizer", tokenizer)  # noqa: B010
        _rprint(
            f"Adopted model.backbone_tokenizer (vocab={len(tokenizer)}) — "
            "modality special tokens (e.g. DNA splice tokens) now encoded correctly",
            flush=True,
        )

    # Branches assign concrete types that don't share a common base.
    dataset: Any
    collator: Any

    # Read once at the top so both branches (VLA and non-VLA) see the same
    # value and the downstream `if use_bucketed_collator:` worker-count
    # check at line ~741 is always defined. VLA uses its own collator and
    # always wants `train_config.data_num_workers`, so we force False there.
    use_bucketed_collator = (
        os.environ.get("USE_BUCKETED_COLLATOR", "1") == "1" and not is_vla_task
    )

    if is_vla_task:
        # Select map-style (legacy CalvinVLADataset) vs webdataset
        # (ModalityAwareWebDatasetWrapper over shards from
        # applications/vla/shard_calvin_vla.py). VLA is Accelerate-only either way —
        # the ZoneAVLATrainer init below raises if USE_NATIVE_DDP /
        # USE_NATIVE_FSDP is set.
        loader_choice = (train_config.calvin_loader or "map").lower()
        if loader_choice not in ("map", "webdataset"):
            raise ValueError(
                f"training.calvin_loader must be 'map' or 'webdataset', got {loader_choice!r}"
            )

        if loader_choice == "map":
            if train_config.calvin_root is None:
                raise RuntimeError("VLA mode requires training.calvin_root to be set")
            _rprint(
                f"Initializing CALVIN VLA Dataset from {train_config.calvin_root}...",
                flush=True,
            )
            dataset = CalvinVLADataset(
                root_dir=train_config.calvin_root,
                tokenizer=tokenizer,
                split=train_config.calvin_split,
                max_chunk=train_config.calvin_max_chunk,
                text_max_length=train_config.calvin_text_max_length,
                strict_integrity=train_config.calvin_strict_integrity,
                max_skipped_fraction=train_config.calvin_max_skipped_fraction,
                model_config=model_config,
            )
            analyze_vla_dataset(dataset, train_config)
        else:
            # webdataset path. Either point at a `calvin` group in
            # daos/lustre_datasets.yaml or override with calvin_webdataset_root.
            storage = (train_config.calvin_webdataset_storage or "lustre").lower()
            if storage == "daos":
                config_path = "src/conf/data/daos_datasets.yaml"
                dataset_root = os.environ.get("DATASET_ROOT") or os.environ.get("DAOS_MOUNT")
                ds_config = load_daos_config(config_path)
            else:
                config_path = "src/conf/data/lustre_datasets.yaml"
                ds_config = load_daos_config(config_path)
                # lustre_datasets.yaml uses the historical `dastr` top-level
                # key for its prefix (kept for compatibility with existing
                # configs); MultiWebDataset reads `daos.mount_base`, so
                # forward the lustre prefix through the dataset root argument.
                dataset_root = ds_config.get("dastr", {}).get(
                    "mount_base"
                ) or ds_config.get("daos", {}).get("mount_base")
            override_root = train_config.calvin_webdataset_root
            if override_root:
                calvin_group = ds_config.get("groups", {}).get("calvin", {})
                for ds_spec in calvin_group.get("datasets", {}).values():
                    ds_spec["path"] = override_root
            _rprint(
                f"Initializing CALVIN VLA WebDataset (storage={storage}, "
                f"world_size={world_size}, rank={rank})...",
                flush=True,
            )
            dataset = ModalityAwareWebDatasetWrapper(
                tokenizer=tokenizer,
                modalities=["vla"],
                config=ds_config,
                groups="calvin",
                daos_mount=dataset_root,
                world_size=world_size,
                rank=rank,
                max_length=train_config.calvin_text_max_length,
                model_config=model_config,
            )
            # analyze_vla_dataset peeks into the dataset, which an
            # IterableDataset can't service without consuming samples — log a
            # minimal banner instead. NOTE: strict_integrity /
            # max_skipped_fraction gates are NOT applied on this path.
            _rprint(
                f"[VLA webdataset] active_modalities={dataset.active_modalities}, "
                f"stats={dataset.get_stats()}",
                flush=True,
            )
        collator = VLACollator(tokenizer)
    else:
        _rprint("Initializing Streaming Datasets...", flush=True)

        # 4. DataLoader with parallel workers for multi-node performance
        # Use BucketedCollator for sorting samples by length within each batch
        # (use_bucketed_collator hoisted above so VLA path also sees it).
        # Hard cap on sequence length in collator - prevents backward pass spikes
        # from outlier-length sequences at epoch boundaries.
        # For projector training, 512 is sufficient; for E2E, match max_seq_length.
        collator_max_seq = int(os.environ.get("MAX_SEQ_LENGTH", "2048"))
        _is_dynamic_length_ts = model_config.ts_projector in DYNAMIC_LENGTH_TS_PROJECTORS
        if use_bucketed_collator and not _is_dynamic_length_ts:
            collator = BucketedCollator(
                tokenizer,
                sort_within_batch=True,
                log_efficiency=True,
                max_seq_length=collator_max_seq,
            )
            logger.info(f"Using BucketedCollator with max_seq_length={collator_max_seq}")
        else:
            if use_bucketed_collator and _is_dynamic_length_ts:
                logger.info(
                    f"Disabling BucketedCollator for ts_projector={model_config.ts_projector}; "
                    "using MultimodalCollator with time_series passthrough for dynamic batching."
                )
            # Interleaved-QA samples carry <ts>/<ts/> token pairs and
            # prompt/target metadata that must stay in sync; tail-truncating
            # them at collate time would desync those invariants and crash the
            # merge in model.py. That path is capped at the variate level in
            # _process_ts_qa instead, so pass max_seq_length=None here. For
            # plain (non-interleaved) batches, apply the same cap as the
            # bucketed collator (issue #120).
            collator_cap = None if model_config.is_interleaved_qa else collator_max_seq
            collator = MultimodalCollator(
                tokenizer,
                max_seq_length=collator_cap,
                passthrough_time_series=_is_dynamic_length_ts,
            )
            logger.info(
                f"Using MultimodalCollator with max_seq_length={collator_cap} "
                f"(is_interleaved_qa={model_config.is_interleaved_qa}, "
                f"passthrough_time_series={_is_dynamic_length_ts})"
            )
        # Check if we should use direct MultiWebDataset mode
        # This bypasses StreamingMultimodalDataset which has integration issues
        # with multi-dataset loading (Issue 14 in the Feb-2026 bring-up debug
        # journal, since retired; see git history for the original writeup)
        use_multi_dataset = os.environ.get("USE_MULTI_DATASET", "0") == "1"
        dataset_root = os.environ.get("DATASET_ROOT") or os.environ.get("DAOS_MOUNT")
        dataset_groups = os.environ.get("DATASET_GROUPS", "all")
        dataset_config = os.environ.get("DATASET_CONFIG")
        requested_non_text_modalities = [
            str(m) for m in list(cfg.model.modalities) if str(m) != "text"
        ]
        primary_non_text_modality = (
            requested_non_text_modalities[0]
            if len(requested_non_text_modalities) == 1
            else None
        )
        _enable_all_modalities = os.environ.get("ENABLE_ALL_MODALITIES", "0") == "1"
        if _enable_all_modalities and _is_main_env():
            logger.info(
                "ENABLE_ALL_MODALITIES=1 → allow_dummy_data=True. Missing datasets will "
                "fall back to modality-correct dummy tensors instead of crashing. "
                "Every skip=true entry in datasets_config.json whose modality is in "
                "model.modalities will be activated and marked fallback_dummy=true at "
                "runtime (entries for other modalities are left skipped; the existing "
                "fallback_dummy field, if any, is overwritten for activated entries)."
            )

        if use_multi_dataset and dataset_root:
            logger.info("*** MULTI-DATASET MODE ***")
            logger.info(f"  DATASET_ROOT: {dataset_root}")
            logger.info(f"  DATASET_GROUPS: {dataset_groups}")
            logger.info(f"  DATASET_CONFIG: {dataset_config}")

            # Parse proportion overrides from environment
            proportion_overrides = {}
            proportions_str = os.environ.get("DATASET_PROPORTIONS", "")
            if proportions_str:
                for pair in proportions_str.split(","):
                    if ":" in pair:
                        ds_name, prop = pair.split(":", 1)
                        try:
                            proportion_overrides[ds_name.strip()] = float(prop.strip())
                        except ValueError:
                            pass
                if proportion_overrides:
                    logger.info(f"  Proportion overrides: {proportion_overrides}")

            # Load config
            config = (
                load_daos_config(dataset_config)
                if dataset_config
                else load_daos_config("src/conf/data/daos_datasets.yaml")
            )

            # Parse groups
            groups = dataset_groups.split(",") if "," in dataset_groups else dataset_groups

            # Check if bucketing is enabled (improves throughput for mixed-length datasets)
            use_bucketing = os.environ.get("USE_BUCKETING", "0") == "1"
            bucket_buffer_size = int(os.environ.get("BUCKET_BUFFER_SIZE", "1000"))
            webdataset_shuffle_buffer = int(
                os.environ.get("WEBDATASET_SHUFFLE_BUFFER", "32768")
            )
            webdataset_resampled = os.environ.get("WEBDATASET_RESAMPLED", "1") == "1"
            webdataset_partition_by = os.environ.get(
                "WEBDATASET_PARTITION_BY", "global"
            )
            logger.info(f"  WebDataset shuffle_buffer={webdataset_shuffle_buffer}")
            if not webdataset_resampled:
                logger.info("  WebDataset finite mode: resampled=False")
            if webdataset_partition_by != "global":
                logger.info(f"  WebDataset partition_by={webdataset_partition_by}")

            # MultiWebDatasetWrapper defaults to the legacy image tuple spec.
            # For single non-image runs (e.g. text+time_series) use the
            # modality-aware wrapper so tuple decoding matches shard contents.
            use_modality_aware = (
                primary_non_text_modality in {"time_series", "graph", "vla"}
            )

            if use_modality_aware and use_bucketing:
                logger.warning(
                    "  USE_BUCKETING=1 requested with modality-aware WebDataset; "
                    "bucketing is currently only wired to MultiWebDatasetWrapper. "
                    "Proceeding with non-bucketed modality-aware loader."
                )

            # `is not None` is implied by use_modality_aware (None is not in the
            # set), but stated so the modalities= list below type-checks.
            if use_modality_aware and primary_non_text_modality is not None:
                logger.info(
                    "  Using ModalityAwareWebDatasetWrapper for modality "
                    f"{primary_non_text_modality}"
                )
                dataset = ModalityAwareWebDatasetWrapper(
                    tokenizer=tokenizer,
                    modalities=[primary_non_text_modality],
                    config=config,
                    groups=groups,
                    daos_mount=dataset_root,
                    proportion_overrides=proportion_overrides if proportion_overrides else None,
                    world_size=world_size,
                    rank=rank,
                    max_length=2048,
                    batch_size=train_config.batch_size,
                    max_steps=train_config.max_steps,
                    model_config=model_config,
                )

            elif use_bucketing:
                logger.info(f"  Using BucketedMultiWebDatasetWrapper (buffer_size={bucket_buffer_size})")
                # Create BucketedMultiWebDatasetWrapper for length-sorted batching
                dataset = BucketedMultiWebDatasetWrapper(
                    tokenizer=tokenizer,
                    config=config,
                    groups=groups,
                    daos_mount=dataset_root,
                    proportion_overrides=proportion_overrides if proportion_overrides else None,
                    world_size=world_size,
                    rank=rank,
                    max_length=2048,
                    batch_size=train_config.batch_size,
                    max_steps=train_config.max_steps,
                    buffer_size=bucket_buffer_size,
                    num_buckets=8,
                    shuffle_buckets=True,
                    shuffle_buffer=webdataset_shuffle_buffer,
                    partition_by=webdataset_partition_by,
                    resampled=webdataset_resampled,
                    model_config=model_config,
                )
            else:
                # Create MultiWebDatasetWrapper (standard mode)
                dataset = MultiWebDatasetWrapper(
                    tokenizer=tokenizer,
                    config=config,
                    groups=groups,
                    daos_mount=dataset_root,
                    proportion_overrides=proportion_overrides if proportion_overrides else None,
                    world_size=world_size,
                    rank=rank,
                    max_length=2048,
                    batch_size=train_config.batch_size,
                    max_steps=train_config.max_steps,
                    shuffle_buffer=webdataset_shuffle_buffer,
                    partition_by=webdataset_partition_by,
                    resampled=webdataset_resampled,
                    model_config=model_config,
                )

            # Log stats
            stats = dataset.get_stats()
            logger.info(
                f"  Loaded {stats['num_datasets']} datasets, {stats['total_samples']} total samples",
            )
            for ds in stats["datasets"]:
                logger.info(
                    f"    - {ds['name']}: {ds['samples']} samples, weight={ds['weight']:.2f}",
                )

        else:
            # Legacy mode: check for launcher-staged local shards first.
            #
            # The LOCAL_SHARDS_DIR path below builds an image-only
            # MultiWebDatasetWrapper reader. Only enter it when the model
            # actually asks for image — otherwise a non-image sweep cell
            # (text_ts, text_graph, text_table) silently trains on the
            # staged pixmo image shards. PR #73 closed the equivalent leak
            # in multimodal.py's WEBDATASET_LOCAL_PATH gate; this is the
            # symmetric gate for the local-shards fast path the sweep
            # harness actually hits.
            allowed_shard_modality = os.environ.get(
                "WEBDATASET_LOCAL_MODALITY", "image"
            )
            local_shards_dir = os.environ.get("LOCAL_SHARDS_DIR", "")
            resolved_modalities = getattr(model_config, "modalities", cfg.model.modalities)
            local_shards_active = _local_shards_active(
                local_shards_dir, allowed_shard_modality, resolved_modalities
            )
            if local_shards_dir and os.path.isdir(local_shards_dir) and not local_shards_active:
                _rprint(
                    f"Ignoring LOCAL_SHARDS_DIR={local_shards_dir}: "
                    f"staged modality={allowed_shard_modality} not in "
                    f"model.modalities={list(resolved_modalities)}. "
                    f"Falling through to StreamingMultimodalDataset.",
                    flush=True,
                )
            if local_shards_active:
                import glob

                # Sanity-check that .tar files actually exist before
                # constructing the wrapper. MultiWebDatasetWrapper would
                # log "0 shards" and continue with an empty pipeline
                # otherwise, leaving the trainer to hang at first batch.
                shard_files = sorted(glob.glob(os.path.join(local_shards_dir, "*.tar")))
                if shard_files:
                    _rprint(
                        f"Using {len(shard_files)} local webdataset shards from "
                        f"{local_shards_dir} via MultiWebDatasetWrapper "
                        f"(local-rank partition)",
                        flush=True,
                    )
                    # Inline LocalShardDataset has been replaced by
                    # MultiWebDatasetWrapper with local_shards_dir set:
                    #   - reads local_manifest.json (falls back to glob)
                    #   - partitions shards by LOCAL_WORLD_SIZE (each
                    #     node's tmpfs subset divided across its ranks)
                    #   - uses the same (jpg;png;jpeg;webp;gif, txt, json)
                    #     tuple spec MultiWebDataset already had for image
                    #     modality (no .json-vs-.txt drift possible)
                    #   - caption format / sample dict shape match the
                    #     normal DAOS path: {image: tensor, text: tokens
                    #     (unpadded; padded at collate), _metadata: str}.
                    resampled = os.environ.get("WEBDATASET_RESAMPLED", "1") == "1"
                    if allowed_shard_modality == "image":
                        dataset = MultiWebDatasetWrapper(
                            tokenizer=tokenizer,
                            local_shards_dir=local_shards_dir,
                            world_size=world_size,
                            rank=rank,
                            max_length=int(os.environ.get("MAX_SEQ_LENGTH", "2048")),
                            modalities=("image",),
                            model_config=model_config,
                            resampled=resampled,
                        )
                    else:
                        _rprint(
                            f"Using modality-aware local shard loader for modality="
                            f"{allowed_shard_modality}",
                            flush=True,
                        )
                        dataset = ModalityAwareWebDatasetWrapper(
                            tokenizer=tokenizer,
                            modalities=[allowed_shard_modality],
                            local_shards_dir=local_shards_dir,
                            world_size=world_size,
                            rank=rank,
                            max_length=int(os.environ.get("MAX_SEQ_LENGTH", "2048")),
                            model_config=model_config,
                            resampled=resampled,
                        )
                else:
                    _rprint(
                        f"WARNING: LOCAL_SHARDS_DIR={local_shards_dir} exists but has no .tar files, falling back to StreamingMultimodalDataset",
                        flush=True,
                    )
                    dataset = StreamingMultimodalDataset(
                        tokenizer=tokenizer,
                        batch_size=train_config.batch_size,
                        max_steps=train_config.max_steps,
                        model_config=model_config,
                        hf_token=os.environ.get("HF_TOKEN"),
                        allow_dummy_data=_enable_all_modalities,
                        verbosity=getattr(
                            logging, cfg.training.get("verbosity", "INFO").upper(), logging.INFO
                        ),
                        dataset_overrides=_resolve_dataset_overrides(cfg),
                        max_seq_length=collator_max_seq,
                        **bio_dataset_kwargs,
                    )
            else:
                # Pure legacy mode: Use StreamingMultimodalDataset
                dataset = StreamingMultimodalDataset(
                    tokenizer=tokenizer,
                    batch_size=train_config.batch_size,
                    max_steps=train_config.max_steps,
                    model_config=model_config,
                    hf_token=os.environ.get("HF_TOKEN"),
                    allow_dummy_data=_enable_all_modalities,
                    verbosity=getattr(
                        logging, cfg.training.get("verbosity", "INFO").upper(), logging.INFO
                    ),
                    dataset_overrides=_resolve_dataset_overrides(cfg),
                    max_seq_length=collator_max_seq,
                    **bio_dataset_kwargs,
                )

    # Compare the modalities the dataloader will actually emit against
    # `model.modalities`. The pre-existing strict validator in
    # `src/data/multimodal.py` only catches datasets that fail to load when
    # skip=False — silently-skipped datasets and image-only WebDataset wrappers
    # both pass through with no signal that the model is asking for modalities
    # the dataloader can never produce. Logs a WARNING on mismatch (also
    # records to perf.jsonl); set `data.strict_modality_check=true` to upgrade
    # to RuntimeError when running scaling sweeps where wrong numbers are
    # worse than no numbers. Mismatch detection runs on every rank (deterministic
    # from the same model+dataset state) so strict-mode raise happens everywhere
    # instead of leaving non-rank-0 ranks to hang at the next collective.
    try:
        from src.utils.perf_log import log_perf_record as _log_perf_record
        from src.utils.perf_log import model_modalities as _model_modalities

        _model_mods = _model_modalities(model) or []
        _ds_active = getattr(dataset, "active_modalities", None)
        _ds_mods = sorted(str(m) for m in _ds_active) if _ds_active else []
        _model_mods_sorted = sorted(_model_mods)
        # text is always implicitly emitted by every dataset; ignore it in the
        # diff so a model with modalities=[text,image] vs a wrapper reporting
        # {image} doesn't trip a spurious warning.
        _model_non_text = [m for m in _model_mods_sorted if m != "text"]
        _ds_non_text = [m for m in _ds_mods if m != "text"]
        _strict = bool(cfg.get("data", {}).get("strict_modality_check", False))
        _mismatch = bool(_model_non_text) and _model_non_text != _ds_non_text
        if _mismatch:
            msg = (
                f"DATALOADER/MODEL MODALITY MISMATCH: dataloader={_ds_mods} "
                f"model={_model_mods_sorted}"
            )
            if rank == 0:
                logger.warning(msg)
                _log_perf_record(
                    getattr(train_config, "output_dir", None),
                    {
                        "event": "startup_modality_check",
                        "site": "train.startup",
                        "dataloader_modalities": _ds_mods,
                        "model_modalities": _model_mods_sorted,
                        "mismatch": True,
                        "sweep_id": getattr(train_config, "sweep_id", None),
                        "preset": getattr(train_config, "preset", None),
                    },
                )
            if _strict:
                raise RuntimeError(msg)
        elif rank == 0:
            # Record the matched view too — the per-modality aggregator uses
            # this record's dataloader_modalities/model_modalities columns to
            # self-validate every sweep row.
            _log_perf_record(
                getattr(train_config, "output_dir", None),
                {
                    "event": "startup_modality_check",
                    "site": "train.startup",
                    "dataloader_modalities": _ds_mods,
                    "model_modalities": _model_mods_sorted,
                    "mismatch": False,
                    "sweep_id": getattr(train_config, "sweep_id", None),
                    "preset": getattr(train_config, "preset", None),
                },
            )
    except RuntimeError:
        raise
    except Exception as _e_mod_check:  # noqa: BLE001
        logger.debug(f"[startup] modality check skipped: {_e_mod_check}")

    # 4. DataLoader
    use_pin_memory = torch.cuda.is_available() and not (
        hasattr(torch, "xpu") and torch.xpu.is_available()
    )
    # Both bucketed and non-bucketed paths honor train_config.data_num_workers
    # (default 4 from TrainingConfig). The previous bucketed-only hard-code
    # silently ignored `+training.data_num_workers=N` from Hydra for every
    # launcher that defaults USE_BUCKETED_COLLATOR=1 — there is no coupling
    # between BucketedCollator and worker count.
    n_workers = train_config.data_num_workers
    # Operator override: DL_NUM_WORKERS env wins over config so smokes /
    # scaling studies can sweep worker counts from launcher scripts that
    # don't touch Hydra. Empty/unset = leave default. Negative values are
    # rejected; non-numeric values fall back with a warning.
    _dl_workers_env = os.environ.get("DL_NUM_WORKERS", "").strip()
    if _dl_workers_env:
        try:
            _dl_workers_val = int(_dl_workers_env)
            if _dl_workers_val < 0:
                raise ValueError(f"must be >= 0, got {_dl_workers_val}")
            n_workers = _dl_workers_val
        except ValueError as e:
            _rprint(
                f"[Rank {rank}] WARNING: ignoring DL_NUM_WORKERS={_dl_workers_env!r} ({e}); "
                f"using default n_workers={n_workers}",
                flush=True,
            )
    _rprint(
        f"[Rank {rank}] Creating DataLoader: num_workers={n_workers}, pin_memory={use_pin_memory}",
        flush=True,
    )
    train_loader = DataLoader(
        dataset,
        batch_size=train_config.batch_size,
        collate_fn=collator,
        num_workers=n_workers,
        persistent_workers=True if n_workers > 0 else False,
        prefetch_factor=2 if n_workers > 0 else None,
        pin_memory=use_pin_memory,
    )
    _rprint(f"[Rank {rank}] DataLoader created successfully", flush=True)

    # Critical: Attach tokenizer to model for Evaluators/Trainer usage
    setattr(model, "tokenizer", tokenizer)  # noqa: B010 — nn.Module attr typed as Tensor|Module

    analyze_dataset(dataset, train_config)


    # --- Resume from checkpoint (model weights only, for stage transitions) ---
    # This loads weights BEFORE DDP wrapping and creates a fresh optimizer.
    # Use for transitioning from projector-only to E2E training.
    if train_config.resume_weights_only:
        logger.info("\n*** LOADING WEIGHTS FROM CHECKPOINT (model-only, fresh optimizer) ***")
        logger.info(f"  Checkpoint: {train_config.resume_weights_only}")
        prev_step = load_model_weights_only(
            model, train_config.resume_weights_only, device=device
        )
        logger.info(f"  Previous training ended at step {prev_step}")
        logger.info("  Starting fresh training with new optimizer/scheduler config")
        # Note: We intentionally do NOT resume the step counter.
        # Stage transitions start fresh (step 0) with new LR schedule.

    # 5. Trainer
    _rprint("Initializing Trainer...")

    if is_vla_task and (use_native_fsdp or use_native_ddp):
        raise RuntimeError(
            "VLA mode currently supports Accelerate path only (unset USE_NATIVE_DDP/FSDP)."
        )
    if is_bio_task and (use_native_fsdp or use_native_ddp):
        raise RuntimeError(
            "BioReason DNA tasks currently support Accelerate path only (unset USE_NATIVE_DDP/FSDP)."
        )

    # Check for native distributed mode (bypasses Accelerate for timing comparison)
    if use_native_fsdp:
        _rprint("*** NATIVE FSDP MODE - Bypassing Accelerate ***")
        os.environ.setdefault("DIST_STRATEGY", "fsdp")
        train_native_ddp(model, train_config, train_loader, rank, world_size, local_rank, device)
    elif use_native_ddp:
        _rprint("*** NATIVE DDP MODE - Bypassing Accelerate ***")
        train_native_ddp(model, train_config, train_loader, rank, world_size, local_rank, device)
    else:
        trainer: Any
        if is_vla_task:
            trainer = ZoneAVLATrainer(model, train_config, train_loader)
        elif task_type == "bioreason_grpo":
            # Deferred import: trainer_grpo.py ships in a later PR than
            # the rest of the bioreason dispatch wiring, so importing it lazily
            # here (rather than at module top-level) keeps this file loadable
            # before that PR lands.
            from src.training.trainer_grpo import ZoneDTrainer

            # ref_model defaults to None — ZoneDTrainer deepcopies self.model
            # internally when not provided (see trainer_grpo.py).
            trainer = ZoneDTrainer(model, train_config, train_loader, tokenizer=tokenizer)
        else:
            # bioreason_sft / bioreason_projector (is_bio_task) fall through
            # here too -- ZoneATrainer handles DNA/LoRA/KEGG-eval via
            # config-gated branches (lora_enabled, freeze_connector,
            # bioreason_dataset), no longer needing a separate trainer class.
            trainer = ZoneATrainer(model, train_config, train_loader)
        # 6. Train
        _rprint(f"Starting Training Loop for {train_config.max_steps} steps...")
        trainer.train()


if __name__ == "__main__":
    main()
