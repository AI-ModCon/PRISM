"""Native DDP/FSDP training loop — bypasses Accelerate.

Used for timing comparison and for production runs where Accelerate's
overhead is undesirable. See `src.training.trainer_zone_a.ZoneATrainer`
for the Accelerate-based path.

Behavior is selected by environment variables; `src/train.py:main()`
sets these before calling `train_native_ddp`:
  USE_NATIVE_DDP=1   → DDP path
  USE_NATIVE_FSDP=1  → FSDP path (also sets DIST_STRATEGY=fsdp)
"""

import contextlib
import logging
import os
import sys

import torch
import wandb

from src.config import DYNAMIC_LENGTH_TS_PROJECTORS
from src.training.distributed import (
    load_model_weights_only,
    save_native_ddp_checkpoint,
    wrap_model_distributed,
)
from src.utils.perf_log import (
    _FlopCounter,
    batch_modality_counts,
    batch_token_count,
    count_parameters,
    log_perf_record,
    model_modalities,
    sequence_stats,
)

logger = logging.getLogger(__name__)


def _optimizer_state_key_kind(optimizer_state: dict) -> str:
    keys = list(optimizer_state.get("state", {}).keys())
    if not keys:
        return "empty"
    if all(isinstance(key, int) for key in keys):
        return "param_id"
    if all(isinstance(key, str) for key in keys):
        return "param_name"
    return "mixed"


def _build_source_optimizer_for_resume(model, config):
    """Recreate pre-wrapper optimizer param ordering for DDP -> FSDP conversion."""
    freeze_llm = getattr(config, "freeze_llm", True)
    freeze_vit = getattr(config, "freeze_vit", True)
    lr_connector = getattr(config, "lr_connector", None)
    lr_vit = getattr(config, "lr_vit", None)

    if freeze_llm and freeze_vit:
        lr_to_use = lr_connector if lr_connector is not None else config.learning_rate
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        return torch.optim.AdamW(
            trainable_params, lr=lr_to_use, weight_decay=config.weight_decay
        )

    if freeze_llm and not freeze_vit:
        encoder_params = []
        projector_params = []
        other_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "encoders" in name:
                encoder_params.append(param)
            elif "projectors" in name:
                projector_params.append(param)
            else:
                other_params.append(param)

        lr_enc = lr_vit if lr_vit is not None else 6e-6
        lr_proj = lr_connector if lr_connector is not None else 2e-4
        param_groups = []
        if encoder_params:
            param_groups.append(
                {"params": encoder_params, "lr": lr_enc, "name": "encoder"}
            )
        if projector_params:
            param_groups.append(
                {"params": projector_params, "lr": lr_proj, "name": "projector"}
            )
        if other_params:
            param_groups.append(
                {"params": other_params, "lr": config.learning_rate, "name": "other"}
            )
        return torch.optim.AdamW(param_groups, weight_decay=config.weight_decay)

    encoder_params = []
    projector_params = []
    llm_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "encoders" in name:
            encoder_params.append(param)
        elif "projectors" in name:
            projector_params.append(param)
        elif (
            "llm" in name
            or "backbone" in name
            or "model.layers" in name
            or "model.embed_tokens" in name
            or "model.norm" in name
            or "lm_head" in name
        ):
            llm_params.append(param)
        else:
            other_params.append(param)

    lr_proj = lr_connector if lr_connector is not None else 2e-4
    lr_enc = lr_vit if lr_vit is not None else 6e-6
    lr_llm_val = getattr(config, "lr_llm", None)
    lr_llm_val = lr_llm_val if lr_llm_val is not None else 2e-5
    param_groups = []
    if projector_params:
        param_groups.append(
            {"params": projector_params, "lr": lr_proj, "name": "projector"}
        )
    if encoder_params:
        param_groups.append(
            {"params": encoder_params, "lr": lr_enc, "name": "encoder"}
        )
    if llm_params:
        param_groups.append({"params": llm_params, "lr": lr_llm_val, "name": "llm"})
    if other_params:
        param_groups.append(
            {"params": other_params, "lr": config.learning_rate, "name": "other"}
        )
    return torch.optim.AdamW(param_groups, weight_decay=config.weight_decay)


def _load_optimizer_state_for_resume(
    optimizer,
    optimizer_state,
    *,
    model,
    unwrapped_model,
    config,
    is_fsdp_model: bool,
    current_dist_strategy: str,
    source_dist_strategy: str | None,
    is_main: bool,
):
    if not is_fsdp_model:
        optimizer.load_state_dict(optimizer_state)
        if is_main:
            logger.info("[Native DDP] Restored optimizer state")
        return

    source_dist_strategy = (source_dist_strategy or "").lower()
    state_key_kind = _optimizer_state_key_kind(optimizer_state)
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    if source_dist_strategy in {"fsdp", "hsdp"}:
        if state_key_kind != "param_name":
            raise RuntimeError(
                "Cannot restore optimizer from this FSDP/HSDP checkpoint: it "
                "contains a rank-local optimizer state instead of a full "
                "param-name keyed state. Relaunch from scratch or load this "
                "checkpoint with training.resume_weights_only."
            )
        sharded_optimizer_state = FSDP.shard_full_optim_state_dict(
            optimizer_state,
            model,
            optim=optimizer,
        )
        optimizer.load_state_dict(sharded_optimizer_state)
        if is_main:
            logger.info(
                "[Native DDP] Sharded and restored full optimizer state "
                f"from {source_dist_strategy} checkpoint"
            )
        return

    if state_key_kind not in {"param_id", "param_name"}:
        raise RuntimeError(
            "Cannot convert optimizer state for FSDP/HSDP resume: "
            f"unsupported optimizer state key kind {state_key_kind!r}"
        )

    from torch.distributed.fsdp import OptimStateKeyType

    if state_key_kind == "param_id":
        source_optimizer = _build_source_optimizer_for_resume(unwrapped_model, config)
        optimizer_state = FSDP.rekey_optim_state_dict(
            optimizer_state,
            OptimStateKeyType.PARAM_NAME,
            unwrapped_model,
            optim=source_optimizer,
        )

    sharded_optimizer_state = FSDP.shard_full_optim_state_dict(
        optimizer_state,
        model,
        optim=optimizer,
    )
    optimizer.load_state_dict(sharded_optimizer_state)
    if is_main:
        logger.info(
            "[Native DDP] Converted and restored optimizer state "
            f"for {current_dist_strategy} resume"
        )


def forward_training_batch(model, batch):
    """Call the public forward entry point so DDP hooks also see native outputs."""
    if "requested_outputs" not in batch:
        logits, loss = model(batch, labels=batch.get("text"))
        return logits, loss, {}
    result = model(
        batch["inputs"], targets=batch.get("targets"),
        requested_outputs=batch["requested_outputs"],
        native_context=batch.get("native_context"),
        output_specs=batch.get("output_specs"),
        decoder_kwargs=batch.get("decoder_kwargs"),
    )
    if result.loss is None:
        raise ValueError("Structured training batch has no supervised output loss")
    return result.predictions.get("text"), result.loss, result.losses


def compute_loss_stats(
    summed_rank_mean: float,
    summed_weighted_loss: float,
    summed_tokens: float,
    world_size: int,
) -> dict[str, float]:
    """Reduce per-rank train-loss scalars into the values logged to W&B.

    Pure function: given the SUM across ranks of
    ``(rank_mean_loss, weighted_loss_sum, loss_tokens)`` and the world size,
    return the token-weighted mean (the true cross-entropy over non-pad text
    tokens), the equal-rank mean (matches the scalar DDP averages for the
    gradient update), and the global token count.

    Kept at module scope, separate from the collective, so it can be unit
    tested without a process group — see tests/test_native_loss_aggregation.py.
    The single-rank case is just ``world_size == 1`` with the local values as
    the "sums", so both paths share this helper.
    """
    rank_mean = (
        summed_rank_mean / world_size if world_size > 0 else float(summed_rank_mean)
    )
    global_tokens = float(max(0.0, summed_tokens))
    token_mean = (
        summed_weighted_loss / global_tokens if global_tokens > 0 else rank_mean
    )
    return {
        "token_mean": float(token_mean),
        "rank_mean": float(rank_mean),
        "global_tokens": global_tokens,
    }


def train_native_ddp(model, config, train_loader, rank, world_size, local_rank, device):
    """Native DDP/FSDP training loop without Accelerate - for timing comparison."""
    import signal
    import time

    import torch.distributed as dist

    # --- Data loading timeout ---
    # Prevents silent hangs where one rank's DAOS/dfuse read stalls indefinitely,
    # causing all other ranks to block on the DDP AllReduce forever.
    # The first batch takes longer due to BucketedMultiWebDatasetWrapper buffer fill
    # (2000 samples per worker). Default: 600s = 10 min.
    # Configurable via DATA_LOAD_TIMEOUT_SEC env var.
    _DATA_LOAD_TIMEOUT = int(os.environ.get("DATA_LOAD_TIMEOUT_SEC", "600"))

    class _DataLoadTimeout(Exception):
        pass

    def _data_load_timeout_handler(signum, frame):
        raise _DataLoadTimeout(
            f"Data loading timed out after {_DATA_LOAD_TIMEOUT}s — "
            f"possible DAOS/dfuse stall on rank {rank}"
        )

    def _fetch_batch(data_iter, train_loader, step, accum_idx):
        """Fetch a batch with a SIGALRM-based timeout to detect hung reads."""
        prev_handler = signal.signal(signal.SIGALRM, _data_load_timeout_handler)
        signal.alarm(_DATA_LOAD_TIMEOUT)
        try:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
        except _DataLoadTimeout:
            logger.error(
                f"[Rank {rank}] Data loading timeout at step {step}, "
                f"accum_idx {accum_idx}. Likely DAOS/dfuse stall. "
                f"Reinitializing data iterator and retrying once."
            )
            # Re-arm alarm for the retry — without this, a hung retry
            # blocks forever (the original alarm was already consumed).
            signal.alarm(_DATA_LOAD_TIMEOUT)
            data_iter = iter(train_loader)
            batch = next(data_iter)
        finally:
            signal.alarm(0)  # Cancel pending alarm
            signal.signal(signal.SIGALRM, prev_handler)
        return batch, data_iter

    def _move_batch_to_device(batch, device):
        from ..decoders.types import move_tensors
        return move_tensors(batch, device)

    def _read_checkpoint_metadata(checkpoint_path: str) -> dict:
        meta_path = os.path.join(checkpoint_path, "training_state.json")
        if not os.path.exists(meta_path):
            return {}
        import json

        with open(meta_path) as f:
            return json.load(f)

    def _load_optimizer_scheduler_state(checkpoint_path: str):
        state_path = os.path.join(checkpoint_path, "training_state.pt")
        if not os.path.exists(state_path):
            return {}
        return torch.load(state_path, map_location="cpu", weights_only=False)

    def _maybe_reset_finite_epoch(data_iter, train_loader, step_num, reason: str):
        finite_epoch_steps = int(getattr(config, "finite_epoch_steps", 0) or 0)
        if finite_epoch_steps <= 0 or step_num <= 0:
            return data_iter
        if step_num % finite_epoch_steps != 0:
            return data_iter
        if is_main:
            epoch = step_num // finite_epoch_steps
            logger.info(
                f"[Native DDP] Finite dataloader epoch boundary "
                f"after epoch {epoch} ({reason}); reshuffling iterator"
            )
        return iter(train_loader)

    def _try_fast_forward_dataloader(skip_batches: int):
        if os.environ.get("PRISM_FAST_DATALOADER_REPLAY", "1") != "1":
            return None
        dataset = getattr(train_loader, "dataset", None)
        fast_forward = getattr(dataset, "fast_forward_batches", None)
        if not callable(fast_forward):
            return None

        num_workers = int(getattr(train_loader, "num_workers", 0) or 0)
        if num_workers != 0:
            raise RuntimeError(
                "PRISM_FAST_DATALOADER_REPLAY requires DataLoader num_workers=0. "
                f"Got num_workers={num_workers}. Set training.data_num_workers=0 "
                "or PRISM_FAST_DATALOADER_REPLAY=0."
            )

        batch_size = int(
            getattr(train_loader, "batch_size", config.batch_size)
            or config.batch_size
        )
        if is_main:
            logger.info(
                f"[Native DDP] Fast-forwarding dataloader: "
                f"{skip_batches} local batch(es), batch_size={batch_size}"
            )
        skipped_samples = fast_forward(skip_batches, batch_size)
        if is_main:
            logger.info(
                f"[Native DDP] Fast dataloader replay positioned stream after "
                f"{skipped_samples} local raw sample(s)"
            )
        return iter(train_loader)

    is_main = rank == 0

    def _loss_token_count(batch: dict, pad_id: int | None) -> int:
        """Count local tokens that contribute to the text loss."""
        try:
            return batch_token_count(batch, pad_id=pad_id)
        except Exception as _e_loss_tokens:  # noqa: BLE001
            logger.debug(f"[perf] loss token count skipped: {_e_loss_tokens}")
            return 0

    def _global_loss_stats(
        local_rank_mean_loss: float,
        local_weighted_loss_sum: float,
        local_loss_tokens: int,
    ) -> dict[str, float]:
        """Aggregate train-loss scalars across ranks for logging.

        `loss` in W&B should reflect the whole distributed batch. The
        token-weighted value is the true cross-entropy over non-pad text
        tokens; `rank_mean` is retained because it matches the equal-rank
        scalar that DDP averages for gradient updates.
        """
        local_tokens = max(0, int(local_loss_tokens))
        if world_size > 1 and dist.is_initialized():
            # fp32, not fp64: oneCCL/xccl on Aurora is finicky about non-fp32
            # collectives and fp64 is emulated/slow on PVC. The magnitudes here
            # (weighted_loss_sum ~ loss * tokens/rank * ranks, order 1e7) stay
            # well inside fp32's exact range, so loss-scalar precision is fine.
            stats = torch.tensor(
                [
                    float(local_rank_mean_loss),
                    float(local_weighted_loss_sum),
                    float(local_tokens),
                ],
                dtype=torch.float32,
                device=device,
            )
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            return compute_loss_stats(
                summed_rank_mean=float(stats[0].item()),
                summed_weighted_loss=float(stats[1].item()),
                summed_tokens=float(stats[2].item()),
                world_size=world_size,
            )

        # Single-rank: local values are the "sums", world_size == 1.
        return compute_loss_stats(
            summed_rank_mean=float(local_rank_mean_loss),
            summed_weighted_loss=float(local_weighted_loss_sum),
            summed_tokens=float(local_tokens),
            world_size=1,
        )

    # === PyTorch Profiler Setup ===
    enable_profiler = os.environ.get("ENABLE_PROFILER", "0") == "1"
    profiler_steps_str = os.environ.get("PROFILER_STEPS", "5,10,15")
    profiler_steps = set(
        int(s.strip()) for s in profiler_steps_str.split(",") if s.strip()
    )
    profiler = None

    if enable_profiler and is_main:
        from torch.profiler import (
            ProfilerActivity,
            profile,
        )

        # Create profiler output directory
        profiler_dir = os.path.join(config.output_dir, "profiler")
        os.makedirs(profiler_dir, exist_ok=True)

        logger.info(f"[Profiler] Enabled. Will profile steps: {sorted(profiler_steps)}")
        logger.info(f"[Profiler] Output directory: {profiler_dir}")

    # === Per-Rank Timing Setup (for straggler detection) ===
    per_rank_timing = os.environ.get("PER_RANK_TIMING", "0") == "1"
    if per_rank_timing:
        logger.info(f"[Rank {rank}] Per-rank timing enabled for straggler detection")

    # === Configurable Log Interval ===
    # LOG_EVERY_N_STEPS: How often to print step summaries (default: 10)
    log_every = int(os.environ.get("LOG_EVERY_N_STEPS", "10"))

    # === Per-Step Watchdog Timer ===
    # Detects hangs faster than the xccl collective timeout (which is 10 min).
    # Each rank sets a SIGALRM before starting a step; if the step takes longer
    # than STEP_WATCHDOG_TIMEOUT seconds, it prints diagnostic info and aborts.
    _step_watchdog_timeout = int(os.environ.get("STEP_WATCHDOG_TIMEOUT", "0"))
    if _step_watchdog_timeout > 0:
        import threading

        _watchdog_step = [0]  # mutable container for current step
        _watchdog_phase = ["init"]  # mutable container for current phase
        _watchdog_timer: list[threading.Timer | None] = [None]

        def _watchdog_bark():
            """Called when a step exceeds the watchdog timeout."""
            s = _watchdog_step[0]
            p = _watchdog_phase[0]
            msg = (
                f"\n{'=' * 60}\n"
                f"[WATCHDOG] Rank {rank} HUNG at step {s}, phase '{p}'\n"
                f"  Timeout: {_step_watchdog_timeout}s exceeded\n"
                f"  This rank is likely blocked on an xccl collective.\n"
                f"{'=' * 60}\n"
            )
            sys.stderr.write(msg)
            sys.stderr.flush()
            # Also try to write to stdout in case stderr is buffered
            logger.error(msg)
            # Abort the process so PBS captures it instead of burning hours
            os._exit(42)

        def _watchdog_start(step_num, phase="step"):
            """Start/restart the watchdog timer for a new step."""
            _watchdog_step[0] = step_num
            _watchdog_phase[0] = phase
            if _watchdog_timer[0] is not None:
                _watchdog_timer[0].cancel()
            t = threading.Timer(_step_watchdog_timeout, _watchdog_bark)
            t.daemon = True
            t.start()
            _watchdog_timer[0] = t

        def _watchdog_stop():
            """Cancel the watchdog timer (step completed successfully)."""
            if _watchdog_timer[0] is not None:
                _watchdog_timer[0].cancel()
                _watchdog_timer[0] = None

        if is_main:
            logger.info(
                f"[Watchdog] Per-step watchdog enabled: {_step_watchdog_timeout}s timeout"
            )
    else:
        # No-op functions when watchdog is disabled
        def _watchdog_start(step_num, phase="step"):
            pass

        def _watchdog_stop():
            pass

    # CRITICAL: Fork DataLoader workers BEFORE HSDP wrapping.
    #
    # HSDP's init_device_mesh() creates multiple CCL communicators, each with
    # CCL_WORKER_COUNT background threads.  If DataLoader workers are forked
    # *after* this, the child processes inherit dead copies of those threads,
    # causing an immediate deadlock when the workers try to read data.
    #
    # By calling iter(train_loader) here (before wrap_model_distributed), the
    # workers fork with only the single global communicator from
    # init_process_group(), which DDP/FSDP prove is safe.  The HSDP mesh
    # communicators created later exist only in the main process.
    #
    # With persistent_workers=True the workers stay alive for the entire run,
    # so this one-time early fork is sufficient.
    dist_strategy = os.environ.get("DIST_STRATEGY", "ddp").lower()
    if dist_strategy == "hsdp":
        if is_main:
            logger.info("[HSDP] Pre-forking DataLoader workers before init_device_mesh...")
        _t_prefork = time.time()
        data_iter = iter(train_loader)
        _dt_prefork = time.time() - _t_prefork
        if is_main:
            logger.info(f"[HSDP] DataLoader workers forked successfully ({_dt_prefork:.1f}s)")
    else:
        data_iter = None  # Will be created later (DDP/FSDP don't need early fork)

    resume_checkpoint = getattr(config, "resume_from_checkpoint", None)
    resume_step = 0
    resume_metadata: dict = {}
    resume_unwrapped_model = model
    if resume_checkpoint:
        resume_checkpoint = os.path.abspath(resume_checkpoint)
        if is_main:
            logger.info(f"[Native DDP] Full resume from checkpoint: {resume_checkpoint}")
        resume_metadata = _read_checkpoint_metadata(resume_checkpoint)
        prev_cfg = resume_metadata.get("config", {})
        if not getattr(config, "wandb_run_id", None):
            previous_wandb_id = prev_cfg.get("wandb_run_id")
            if previous_wandb_id and previous_wandb_id != "None":
                config.wandb_run_id = previous_wandb_id
        resume_step = load_model_weights_only(model, resume_checkpoint, device=device)

    # Wrap model in DDP or FSDP based on configuration
    model = wrap_model_distributed(model, config, rank, world_size, local_rank, device)

    # IsoFLOP FLOP counter (PR-1 parity for native trainer; PR-4 follow-up).
    # The launcher exports CALIBRATION_JSON per cell when --target-flops is
    # set (or operators export it manually). When unset, the counter is a
    # no-op and per-step records get `flops_per_step=None` — the collector
    # treats null as missing rather than a misleading zero.
    flop_counter = _FlopCounter.from_calibration(
        os.environ.get("CALIBRATION_JSON")
    )

    # Detect FSDP wrapping for gradient clipping and checkpoint saving
    is_fsdp_model = False
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as _FSDP_cls

        is_fsdp_model = isinstance(model, _FSDP_cls)
    except ImportError:
        pass

    # Trainable params
    # With FSDP, model.parameters() returns the local shard (1/world_size).
    # We still use these for the optimizer (FSDP handles sharded updates).
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    def _global_numel(params) -> int:
        """Sum numel across ranks.

        Under FSDP each rank holds only its shard, and shards are NOT equal in
        size — a small trainable module (e.g. a TS encoder + projector) can be
        placed entirely on one rank, leaving rank 0 with zero elements. The old
        `local * world_size` extrapolation therefore reported 0 trainable params
        on runs that were training correctly, and silently mis-stated N_active
        (the x-axis of any IsoFLOP fit). An all-reduce is the only correct
        accounting; it must run on every rank, not under an `is_main` guard.
        """
        local = sum(p.numel() for p in params)
        if not (is_fsdp_model and world_size > 1 and dist.is_initialized()):
            return local
        # Reduce over the SHARD group, not the world.
        #
        # Under HSDP / HYBRID_SHARD, parameters are sharded WITHIN a node and
        # REPLICATED across nodes. An all_reduce over the full world therefore
        # counts every parameter once per replica — on a 16-node job that
        # reported 27.7B trainable params for Qwen3-1.7B (exactly 16x). FSDP
        # exposes the intra-node sharding group as `process_group` on the root
        # module; for non-hybrid FULL_SHARD that group IS the world, so the same
        # call stays correct there.
        shard_pg = getattr(model, "process_group", None)
        counter = torch.tensor([local], dtype=torch.long, device=device)
        dist.all_reduce(counter, op=dist.ReduceOp.SUM, group=shard_pg)
        return int(counter.item())

    num_trainable_local = sum(p.numel() for p in trainable_params)
    num_trainable_total = _global_numel(trainable_params)
    if is_main:
        # For FSDP, show both local shard and total
        if is_fsdp_model:
            logger.info(
                f"[Native DDP] Trainable params (local shard): {num_trainable_local:,}"
            )
            logger.info(f"[Native DDP] Trainable params (total): {num_trainable_total:,}")
            logger.info(
                f"[Native DDP] Model size (total trainable): {num_trainable_total * 2 / 1e9:.2f} GB (BF16)"
            )
        else:
            logger.info(f"[Native DDP] Trainable params: {num_trainable_local:,}")
            logger.info(
                f"[Native DDP] Model size (trainable): {num_trainable_local * 2 / 1e9:.2f} GB (BF16)"
            )

    # Optimizer with Differential Learning Rates
    # Determine training mode and set up parameter groups accordingly
    freeze_llm = getattr(config, "freeze_llm", True)
    freeze_vit = getattr(config, "freeze_vit", True)
    lr_connector = getattr(config, "lr_connector", None)
    lr_vit = getattr(config, "lr_vit", None)

    if freeze_llm and freeze_vit:
        # Projector-Only Mode: Single LR (use lr_connector if available)
        lr_to_use = lr_connector if lr_connector is not None else config.learning_rate
        if is_main:
            logger.info(f"[Native DDP] Projector-Only mode. LR: {lr_to_use}")
        optimizer = torch.optim.AdamW(
            trainable_params, lr=lr_to_use, weight_decay=config.weight_decay
        )
    elif freeze_llm and not freeze_vit:
        # Encoder+Projector Mode: Differential LRs
        # Separate encoder params from projector params
        encoder_params = []
        projector_params = []
        other_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "encoders" in name:
                encoder_params.append(param)
            elif "projectors" in name:
                projector_params.append(param)
            else:
                other_params.append(param)

        # Set up parameter groups with differential LRs
        lr_enc = lr_vit if lr_vit is not None else 6e-6
        lr_proj = lr_connector if lr_connector is not None else 2e-4

        param_groups = []
        if encoder_params:
            param_groups.append(
                {"params": encoder_params, "lr": lr_enc, "name": "encoder"}
            )
        if projector_params:
            param_groups.append(
                {"params": projector_params, "lr": lr_proj, "name": "projector"}
            )
        if other_params:
            param_groups.append(
                {"params": other_params, "lr": config.learning_rate, "name": "other"}
            )

        # Counts must be all-reduced (see _global_numel), so compute them on
        # every rank before the is_main guard — a collective inside the guard
        # would hang.
        _n_enc = _global_numel(encoder_params)
        _n_proj = _global_numel(projector_params)
        _n_other = _global_numel(other_params)
        if is_main:
            logger.info("[Native DDP] Encoder+Projector mode with differential LRs:")
            logger.info(f"  Encoder params: {_n_enc:,} @ LR={lr_enc}")
            logger.info(f"  Projector params: {_n_proj:,} @ LR={lr_proj}")
            if other_params:
                logger.info(
                    f"  Other params: {_n_other:,} @ LR={config.learning_rate}"
                )

        optimizer = torch.optim.AdamW(param_groups, weight_decay=config.weight_decay)
    else:
        # E2E mode: Full training with differential LRs for Connector, ViT, LLM
        # Separate params into groups for proper Molmo-style training
        encoder_params = []
        projector_params = []
        llm_params = []
        other_params = []

        _total_seen = 0
        _total_grad = 0
        for name, param in model.named_parameters():
            _total_seen += 1
            if not param.requires_grad:
                continue
            _total_grad += 1
            if "encoders" in name:
                encoder_params.append(param)
            elif "projectors" in name:
                projector_params.append(param)
            elif (
                "llm" in name
                or "backbone" in name
                or "model.layers" in name
                or "model.embed_tokens" in name
                or "model.norm" in name
                or "lm_head" in name
            ):
                llm_params.append(param)
            else:
                other_params.append(param)
        if is_main:
            logger.info(f"[E2E ParamLoop] Seen: {_total_seen}, requires_grad: {_total_grad}")
            logger.info(
                f"[E2E ParamLoop] encoder={len(encoder_params)}, projector={len(projector_params)}, llm={len(llm_params)}, other={len(other_params)}"
            )

        # Set up differential LRs following Molmo recipe:
        # Connector: High LR (2e-4), ViT: Low LR (6e-6), LLM: Medium LR (2e-5)
        lr_proj = lr_connector if lr_connector is not None else 2e-4
        lr_enc = lr_vit if lr_vit is not None else 6e-6
        lr_llm_val = getattr(config, "lr_llm", None)
        lr_llm_val = lr_llm_val if lr_llm_val is not None else 2e-5

        # Build parameter groups in order: [Connector, ViT, LLM, Other]
        # This order matters for the Molmo scheduler which expects:
        # Group 0: Connector (short warmup)
        # Group 1+: ViT/LLM (long warmup)
        param_groups = []

        if projector_params:
            param_groups.append(
                {"params": projector_params, "lr": lr_proj, "name": "projector"}
            )
        if encoder_params:
            param_groups.append(
                {"params": encoder_params, "lr": lr_enc, "name": "encoder"}
            )
        if llm_params:
            param_groups.append({"params": llm_params, "lr": lr_llm_val, "name": "llm"})
        if other_params:
            param_groups.append(
                {"params": other_params, "lr": config.learning_rate, "name": "other"}
            )

        # All-reduced counts (see _global_numel) — must be computed on every
        # rank, outside the is_main guard, or the collective deadlocks.
        _n_proj = _global_numel(projector_params)
        _n_enc = _global_numel(encoder_params)
        _n_llm = _global_numel(llm_params)
        _n_other = _global_numel(other_params)
        if is_main:
            logger.info("[Native DDP] E2E mode with differential LRs (Molmo recipe):")
            logger.info(
                f"  Projector params: {_n_proj:,} ({len(projector_params)} tensors) @ LR={lr_proj}"
            )
            logger.info(
                f"  Encoder params: {_n_enc:,} ({len(encoder_params)} tensors) @ LR={lr_enc}"
            )
            logger.info(
                f"  LLM params: {_n_llm:,} ({len(llm_params)} tensors) @ LR={lr_llm_val}"
            )
            if other_params:
                logger.info(
                    f"  Other params: {_n_other:,} ({len(other_params)} tensors) @ LR={config.learning_rate}"
                )

        optimizer = torch.optim.AdamW(param_groups, weight_decay=config.weight_decay)

    grad_accum = config.gradient_accumulation_steps
    max_steps = config.max_steps

    # Scheduler - Support multiple scheduler types matching trainer_zone_a.py
    from transformers import get_cosine_schedule_with_warmup

    from src.utils.scheduler import (
        get_cosine_with_min_lr,
        get_molmo_scheduler,
        get_wsd_scheduler,
    )

    scheduler_type = getattr(config, "scheduler_type", "cosine")
    min_lr_ratio = getattr(config, "min_lr_ratio", 0.1)
    warmup_steps_connector = getattr(config, "warmup_steps_connector", 200)
    warmup_steps_main = getattr(config, "warmup_steps_main", 2000)

    if scheduler_type == "molmo_layered":
        scheduler = get_molmo_scheduler(
            optimizer,
            max_steps,
            warmup_connector=warmup_steps_connector,
            warmup_main=warmup_steps_main,
            min_lr_ratio=min_lr_ratio,
        )
        if is_main:
            logger.info("[Native DDP] Using Molmo Layered Scheduler:")
            logger.info(f"  Connector warmup: {warmup_steps_connector} steps")
            logger.info(f"  Main warmup: {warmup_steps_main} steps")
            logger.info(f"  Min LR ratio: {min_lr_ratio}, Max steps: {max_steps}")
    elif scheduler_type == "cosine_with_min_lr":
        scheduler = get_cosine_with_min_lr(
            optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=max_steps,
            min_lr_ratio=min_lr_ratio,
        )
        if is_main:
            logger.info("[Native DDP] Using Cosine Scheduler with Min LR Floor:")
            logger.info(f"  Warmup: {config.warmup_steps} steps")
            logger.info(f"  Min LR ratio: {min_lr_ratio}, Max steps: {max_steps}")
    elif scheduler_type == "wsd":
        wsd_decay_ratio = getattr(config, "wsd_decay_ratio", 0.1)
        wsd_decay_steps = getattr(config, "wsd_decay_steps", None)
        scheduler = get_wsd_scheduler(
            optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=max_steps,
            min_lr_ratio=min_lr_ratio,
            decay_ratio=wsd_decay_ratio,
            decay_steps=wsd_decay_steps,
        )
        if is_main:
            logger.info("[Native DDP] Using WSD Scheduler:")
            logger.info(f"  Warmup: {config.warmup_steps} steps")
            logger.info(f"  Decay ratio: {wsd_decay_ratio}")
            logger.info(f"  Decay steps: {wsd_decay_steps}")
            logger.info(f"  Min LR ratio: {min_lr_ratio}, Max steps: {max_steps}")
    else:
        # Default: Standard HF cosine with warmup
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=max_steps,
        )
        if is_main:
            logger.info(
                f"[Native DDP] Initialized Cosine Scheduler: Warmup={config.warmup_steps}, Max={max_steps}"
            )

    if resume_checkpoint:
        resume_state = _load_optimizer_scheduler_state(resume_checkpoint)
        if "optimizer" in resume_state:
            previous_dist_strategy = resume_metadata.get("dist_strategy")
            _load_optimizer_state_for_resume(
                optimizer,
                resume_state["optimizer"],
                model=model,
                unwrapped_model=resume_unwrapped_model,
                config=config,
                is_fsdp_model=is_fsdp_model,
                current_dist_strategy=dist_strategy,
                source_dist_strategy=previous_dist_strategy,
                is_main=is_main,
            )
        else:
            logger.warning(
                f"[Native DDP] No optimizer state found in {resume_checkpoint}"
            )
        if "scheduler" in resume_state:
            scheduler.load_state_dict(resume_state["scheduler"])
            if is_main:
                logger.info("[Native DDP] Restored scheduler state")
        else:
            logger.warning(
                f"[Native DDP] No scheduler state found in {resume_checkpoint}"
            )

    # Timing accumulators
    timing_data = timing_fwd = timing_bwd = timing_opt = timing_allreduce = 0.0
    timing_steps = 0
    # Throughput accumulators (rolling, reset alongside timing accumulators).
    # tokens_per_window counts non-pad text tokens seen across all micro-batches
    # in the current logging window so we can divide by wall-time for tokens/s.
    # last_batch_modality_counts holds the most recent batch's per-modality
    # sample count — this is what gets logged to perf.jsonl (per-step snapshot,
    # not a window mean, since the dataloader composition shouldn't drift
    # within a single window for a stable run).
    tokens_per_window = 0
    last_batch_modality_counts: dict[str, int] = {}
    # tokenizer pad id: discovered lazily below where the model is unwrapped.
    _pad_id_for_tokens = None

    # IsoFLOP startup snapshot — one perf record with per-component param
    # counts so the collector can populate n_total/n_active/n_trainable
    # without re-instantiating the model. Rank-0 only.
    if is_main:
        try:
            _unwrapped_for_count = model.module if hasattr(model, "module") else model
            _unwrapped_cfg_for_count = getattr(_unwrapped_for_count, "config", None)
            log_perf_record(
                getattr(config, "output_dir", None),
                {
                    "event": "startup_param_count",
                    "site": "trainer_native",
                    **count_parameters(_unwrapped_for_count),
                    "sweep_id": getattr(config, "sweep_id", None),
                    "preset": getattr(config, "preset", None),
                    "seed": getattr(config, "seed", None),
                    "modalities": model_modalities(_unwrapped_for_count),
                    "projector_hidden_mult": (
                        getattr(_unwrapped_cfg_for_count, "projector_hidden_mult", 1)
                        if _unwrapped_cfg_for_count is not None else 1
                    ),
                    "projector_num_layers": (
                        getattr(_unwrapped_cfg_for_count, "projector_num_layers", 2)
                        if _unwrapped_cfg_for_count is not None else 2
                    ),
                },
            )
        except Exception as _e_startup:  # noqa: BLE001
            logger.warning(f"[perf] startup_param_count skipped: {_e_startup}")

    # Per-window seq-length samples for IsoFLOP `sequence_stats`. Flushed
    # alongside tokens_per_window in the per-50 perf record below.
    # NOTE: rank-0 only (the populate site below is gated on `if is_main:`).
    # Mirrors the pre-existing `tokens_per_window` extrapolation: the perf
    # row's seq_p* / padding_ratio reflect ONE rank's batch composition, not
    # a global aggregate. Safe for homogeneous text+image runs (every rank
    # processes the same shard distribution); misleading for heterogeneous
    # multimodal batches where different ranks see different modality mixes.
    # Comment-only acknowledgement — fixing this for real means an
    # all-reduce of per-rank seq stats per flush, which is overkill for the
    # IsoFLOP fit. See PR feedback on PR #98.
    seq_lens_window: list[int] = []

    step = resume_step
    # For HSDP, data_iter was created early (before init_device_mesh) to avoid
    # fork-after-CCL deadlock.  For DDP/FSDP, create it now.
    if data_iter is None:
        _t_iter = time.time()
        data_iter = iter(train_loader)
        _dt_iter = time.time() - _t_iter
        if is_main:
            logger.info(f"[DataLoader] iter(train_loader) took {_dt_iter:.1f}s")
    _resampled_stream = os.environ.get("WEBDATASET_RESAMPLED", "1") == "1"
    if resume_checkpoint and step > 0 and _resampled_stream:
        # Replay only reproduces the pre-crash data order for a finite,
        # deterministic stream. With resampled=True the stream is infinite and
        # samples shards with replacement, so re-fetching `step * grad_accum`
        # batches reproduces nothing — it just burns DAOS reads and advances the
        # RNG into a distribution-skewed / partially re-seen state. Skip it; the
        # fresh iterator created above already re-seeds the stream correctly.
        if is_main:
            logger.warning(
                "[Native DDP] Resume with resampled WebDataset: skipping dataloader "
                "replay (inexact for an infinite resampled stream). Training resumes "
                "on a freshly-seeded iterator. Use --finite-webdataset for an exact, "
                "deterministic resume."
            )
    elif resume_checkpoint and step > 0:
        skip_batches = int(step * grad_accum)
        if is_main:
            logger.info(
                f"[Native DDP] Replaying dataloader progress: skipping "
                f"{skip_batches} local batch(es) for completed step {step}"
            )
        fast_skip_batches = skip_batches
        finite_epoch_steps = int(getattr(config, "finite_epoch_steps", 0) or 0)
        if finite_epoch_steps > 0:
            fast_skip_batches = int((step % finite_epoch_steps) * grad_accum)
            if is_main and fast_skip_batches != skip_batches:
                logger.info(
                    f"[Native DDP] Fast replay using current finite-epoch "
                    f"offset: {fast_skip_batches}/{skip_batches} local batch(es)"
                )

        fast_forwarded_iter = _try_fast_forward_dataloader(fast_skip_batches)
        if fast_forwarded_iter is not None:
            data_iter = fast_forwarded_iter
            if is_main:
                logger.info("[Native DDP] Fast dataloader replay complete")
        else:
            _watchdog_start(step, "resume_dataloader_skip")
            skipped = 0
            for replay_step in range(step):
                data_iter = _maybe_reset_finite_epoch(
                    data_iter, train_loader, replay_step, "resume replay"
                )
                for accum_idx in range(grad_accum):
                    _, data_iter = _fetch_batch(
                        data_iter, train_loader, replay_step, accum_idx
                    )
                    skipped += 1
                if is_main and skipped % 500 == 0:
                    logger.info(
                        f"[Native DDP] Resume dataloader skip: {skipped}/{skip_batches}"
                    )
            _watchdog_stop()
            if is_main:
                logger.info("[Native DDP] Dataloader replay skip complete")

    # Per-step backward times for detailed analysis
    per_step_bwd_times = []

    # Initialize variables that are only set on main rank (to avoid NameError)
    visualizer = None
    val_loader = None
    val_iter = None
    unwrapped_model = model.module if hasattr(model, "module") else model
    tokenizer = getattr(unwrapped_model, "tokenizer", None)
    if tokenizer is not None:
        _pad_id_for_tokens = getattr(tokenizer, "pad_token_id", None)

    if is_main:
        logger.info(f"[Native DDP] Training: {max_steps} steps, grad_accum={grad_accum}")
        logger.info(f"[Native DDP] World size: {world_size}, Device: {device}")

        # Explicit WandB Init
        if config.wandb_project:
            logger.info(f"[Native DDP] Initializing WandB: {config.wandb_project}")
            try:
                wandb.init(
                    project=config.wandb_project,
                    name=config.wandb_run_name,
                    config=config.__dict__,
                    entity=config.wandb_entity,
                    mode=config.wandb_mode,
                    id=config.wandb_run_id,
                    resume="allow" if config.wandb_run_id else None,
                    reinit=True,
                )
                if wandb.run:
                    config.wandb_run_id = wandb.run.id
            except Exception as e:
                logger.warning(f"[Native DDP] WandB Init Failed: {e}")

        # Register SIGTERM handler to flush wandb before PBS kills us
        import signal

        def _sigterm_handler(signum, frame):
            import sys as _sys  # local import to avoid closure over shadowed 'sys'

            logger.info(f"\n[Native DDP] Caught signal {signum}, flushing wandb...")
            if wandb.run:
                wandb.finish()
            _sys.exit(0)

        signal.signal(signal.SIGTERM, _sigterm_handler)

        # Initialize Visualizer
        # Unwrap model to get tokenizer (model.module if DDP)
        from src.utils.batch_viz import BatchVisualizer
        visualizer = BatchVisualizer(tokenizer, use_gt_prefix_for_captions=True)
        if tokenizer is None:
            logger.warning(
                "[Native DDP] Warning: Model has no tokenizer attached. Visualization text decoding will fail."
            )

    # --- Validation Loader (for Visualization) ---
    # Uses manifest-based validation shards when available, falling back
    # to a legacy env-var path for compatibility.
    val_loader = None
    val_iter = None
    val_shards = []

    try:
        import webdataset as wds
        from torchvision import transforms

        from src.data.collate import MultimodalCollator

        # Try to get validation shards from the active dataset manifest.
        dataset_root = os.environ.get("DATASET_ROOT") or os.environ.get("DAOS_MOUNT")
        use_multi_dataset = os.environ.get("USE_MULTI_DATASET", "0") == "1"

        if use_multi_dataset and dataset_root:
            # Use manifest-based validation shards from MultiWebDataset
            from src.data.multi_webdataset import (
                MultiWebDataset,
                load_daos_config,
            )

            dataset_config_path = os.environ.get(
                "DATASET_CONFIG", "src/conf/data/daos_datasets.yaml"
            )
            dataset_groups = os.environ.get("DATASET_GROUPS", "all")
            daos_config = load_daos_config(dataset_config_path)
            groups = (
                dataset_groups.split(",")
                if "," in dataset_groups
                else dataset_groups
            )

            # Create a temporary MultiWebDataset instance to use its shard discovery
            # (reuses the existing distributed shard discovery logic)
            temp_mwd = MultiWebDataset(
                config=daos_config,
                groups=groups,
                daos_mount=dataset_root,
                world_size=1,  # Validation uses all shards on rank 0
                rank=0,
            )
            val_shards = temp_mwd.get_all_validation_shard_paths()

            if val_shards:
                logger.info(
                    f"[Native DDP] Found {len(val_shards)} validation shards from dataset manifests"
                )
            else:
                logger.info("[Native DDP] No validation shards found in dataset manifests")

        # Fallback: a directory of validation shards. This is the path used
        # by --webdataset-dir launches (the manifest branch above only fires
        # for USE_MULTI_DATASET=1 + DATASET_ROOT), so without it those runs
        # silently trained with no validation at all.
        # PRISM_VAL_SHARDS_DIR is the same env var SciTSEvaluator reads.
        if not val_shards:
            val_dir = os.environ.get("PRISM_VAL_SHARDS_DIR", "")
            if val_dir and os.path.isdir(val_dir):
                # glob.glob() can hang on dfuse/DAOS mounts; os.listdir() +
                # filter is the safe pattern for shard directories that may
                # live on DAOS.
                val_shards = sorted(
                    os.path.join(val_dir, f)
                    for f in os.listdir(val_dir)
                    if f.endswith(".tar")
                )
                if val_shards:
                    logger.info(
                        f"[Native DDP] Found {len(val_shards)} validation shards "
                        f"in {val_dir}"
                    )

        # Fallback: env-var-supplied legacy path if no DAOS val_shards found.
        if not val_shards:
            legacy_val_tar = os.environ.get("PRISM_LEGACY_VAL_SHARD", "")
            if legacy_val_tar and os.path.exists(legacy_val_tar):
                val_shards = [legacy_val_tar]
                logger.info(
                    f"[Native DDP] Using legacy validation shard: {legacy_val_tar}"
                )

        if val_shards:
            logger.info(
                f"[Native DDP] Initializing Validation Loader with {len(val_shards)} shard(s)"
            )

            val_transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )
            val_text_max_length = int(os.environ.get("MAX_SEQ_LENGTH", "2048"))

            def _prepare_validation_sample(x):
                caption = x[1] if isinstance(x[1], str) else x[1].decode("utf-8")
                if tokenizer is not None:
                    text_value = tokenizer(
                        caption,
                        return_tensors="pt",
                        padding=False,
                        truncation=True,
                        max_length=val_text_max_length,
                    ).input_ids.squeeze(0)
                else:
                    text_value = caption
                return {
                    "image": val_transform(x[0].convert("RGB")),
                    "text": text_value,
                    "metadata": x[2],
                }

            # Which modality do these shards carry? The image pipeline below
            # decodes jpg/png and emits an "image" key; on time-series shards
            # (.ts.npy/.text/.meta.json) it matches nothing, so the loader
            # would yield empty batches. Dispatch on the model config the
            # same way the training path does. Under DDP, `model.config`
            # doesn't forward to the wrapped module — it must be read off
            # `unwrapped_model`, or `_mcfg` is always None and this branch
            # never fires (val loader silently never built on TS jobs).
            _mcfg = getattr(unwrapped_model, "config", None)
            _is_ts_val = bool(getattr(_mcfg, "is_timeseries", False)) or (
                "time_series" in (getattr(_mcfg, "modalities", []) or [])
            )
            _ts_projector = getattr(_mcfg, "ts_projector", "linear")
            _passthrough_ts = _ts_projector in DYNAMIC_LENGTH_TS_PROJECTORS

            if _is_ts_val:
                import io as _io

                import numpy as _np

                def _prepare_ts_validation_sample(x):
                    arr_bytes, txt = x[0], x[1]
                    arr = _np.load(_io.BytesIO(arr_bytes), allow_pickle=False)
                    ts = torch.from_numpy(_np.ascontiguousarray(arr)).float()
                    if ts.dim() == 1:
                        ts = ts.view(-1, 1)
                    caption = txt if isinstance(txt, str) else txt.decode("utf-8")
                    if tokenizer is not None:
                        text_value = tokenizer(
                            caption,
                            return_tensors="pt",
                            padding=False,
                            truncation=True,
                            max_length=val_text_max_length,
                        ).input_ids.squeeze(0)
                    else:
                        text_value = caption
                    # Mirror the training path's answer-only supervision so
                    # val/loss is comparable to the training loss.
                    prompt_len = 0
                    if "Answer:" in caption and tokenizer is not None:
                        head, _, _tail = caption.partition("Answer:")
                        prompt_len = int(
                            tokenizer(
                                head + "Answer:",
                                return_tensors="pt",
                                padding=False,
                                truncation=True,
                                max_length=val_text_max_length,
                            ).input_ids.squeeze(0).shape[0]
                        )
                    return {
                        "time_series": ts,
                        "text": text_value,
                        "_prompt_len": prompt_len,
                        "_metadata": "[val]",
                    }

                # nodesplitter=None so every rank streams the same shard list and
                # therefore performs the SAME NUMBER of eval forwards -- the FSDP
                # all_gather collectives inside the eval forward must line up.
                # But an identical shard list also meant identical *samples* on
                # all 192 ranks, so one eval saw only validation_batches *
                # batch_size unique rows: 32 of the 2,995 available. The
                # 0.6B/1.7B/4B rungs then landed within 0.005 nats of each other,
                # comfortably inside the noise of a 32-sample estimate.
                #
                # _StridedTSValidation hands each rank a disjoint 1/world_size
                # slice while keeping the per-rank forward count equal, so an
                # eval covers world_size x more unique samples at the same cost.
                class _StridedTSValidation(torch.utils.data.IterableDataset):
                    """Every `stride`-th raw sample from `offset`, then decode.

                    The stride is applied BEFORE the decode/tokenize map, so a
                    rank streams the whole shard set as raw tar members but only
                    decodes its own slice. The shards are small enough to sit in
                    each node's page cache after the first eval.

                    Indexing restarts on every ``__iter__``, so a rank draws the
                    SAME subset at every eval step. A subset that drifted from
                    step to step would inject exactly the noise this is meant to
                    remove.
                    """

                    def __init__(self, shards, offset, stride, keys, map_fn):
                        self.shards = shards
                        self.offset = offset
                        self.stride = stride
                        self.keys = keys
                        self.map_fn = map_fn

                    def __iter__(self):
                        # shardshuffle=False is load-bearing, not tidiness: the
                        # slices are only disjoint if every rank enumerates the
                        # shards in the SAME order. A per-rank shuffle would let
                        # ranks double-count some samples and miss others.
                        base = wds.WebDataset(
                            self.shards, nodesplitter=None, shardshuffle=False
                        )
                        for i, sample in enumerate(base):
                            if (i % self.stride) != self.offset:
                                continue
                            try:
                                row = tuple(sample[k] for k in self.keys)
                            except KeyError:
                                continue
                            yield self.map_fn(row)

                if world_size > 1:
                    val_dataset = _StridedTSValidation(
                        val_shards,
                        offset=rank,
                        stride=world_size,
                        keys=("ts.npy", "text"),
                        map_fn=_prepare_ts_validation_sample,
                    )
                    if is_main:
                        logger.info(
                            f"[Native DDP] Validation striped across {world_size} "
                            f"ranks (rank r takes sample i where "
                            f"i % {world_size} == r)"
                        )
                else:
                    val_dataset = (
                        wds.WebDataset(val_shards, nodesplitter=None)
                        .to_tuple("ts.npy", "text")
                        .map(_prepare_ts_validation_sample)
                    )
                val_collate = MultimodalCollator(
                    tokenizer,
                    max_seq_length=val_text_max_length,
                    passthrough_time_series=_passthrough_ts,
                )
            else:
                # nodesplitter=None on purpose: every rank must iterate the SAME
                # validation batches so the FSDP all_gather collectives inside the
                # eval forward line up. Splitting 3 shards across 16 nodes would
                # leave most nodes with zero batches and hang the job.
                val_dataset = (
                    wds.WebDataset(val_shards, nodesplitter=None)
                    .decode("pil")
                    .to_tuple("jpg;png;jpeg;webp;gif", "txt", "json")
                    .map(_prepare_validation_sample)
                )
                # Image+text validation captions (non-interleaved), so the
                # collator-level cap is safe here (issue #120). Captions are
                # already truncated at tokenize time, but pass the same cap
                # for defense in depth.
                val_collate = MultimodalCollator(
                    tokenizer, max_seq_length=val_text_max_length
                )

            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=config.batch_size,
                collate_fn=val_collate,
                num_workers=0,  # Disable workers to avoid multi-node nodesplitter error
            )
            val_iter = iter(val_loader)
            logger.info(
                f"[Native DDP] Validation Loader Ready "
                f"(modality={'time_series' if _is_ts_val else 'image'})."
            )
        else:
            logger.warning("[Native DDP] Warning: No validation shards found")

    except Exception as e_val:
        logger.warning(f"[Native DDP] Warning: Failed to init validation loader: {e_val}")
        import traceback

        traceback.print_exc()

    # Rank-symmetric readiness gate. The try/except above runs independently
    # on every rank, so a transient per-rank failure (a shard read glitch, a
    # DAOS hiccup) can leave val_loader built on some ranks and None on
    # others. The eval-forward block below is entered on every rank that has
    # a val_loader and calls dist.all_reduce() unconditionally inside — if
    # even one rank skips it while others enter, the job deadlocks on that
    # collective. All-reducing a single readiness flag here, once, forces
    # every rank to agree before any of them ever reach that collective.
    #
    # ROOT CAUSE (found 2026-09-10, held 1N debug allocation, job 8817225):
    # oneCCL's ring allreduce silently drops the LAST rank when the buffer is
    # a single fp32 element. Instrumenting this exact call site with a probe
    # matrix over dtype and element count, on 12 ranks, gave:
    #
    #   fp32, numel=1   -> ranks 0..10 reduce to 11.0; rank 11 reduces to 1.0
    #   fp32, numel>=2  -> all 12 ranks reduce to 12.0   (correct)
    #   int64, numel=1  -> all 12 ranks reduce to 12     (correct, 3/3 reps)
    #   fp64,  numel=1  -> all 12 ranks reduce to 12.0   (correct, 2/2 reps)
    #   bf16,  numel=1  -> all 12 ranks reduce to 12.0   (correct)
    #
    # A same-instant fp64 identity probe (each rank contributing 2**rank)
    # reduced to a full 0..11 bitmask on every rank, so rank 11 IS joining the
    # collective and the process group is intact. Only the 4-byte fp32 payload
    # is mishandled, and repeated fp32 numel=1 reduces accumulate into the
    # result (11 -> 22 -> 33 ...) rather than starting fresh.
    #
    # This supersedes the earlier "first cross-node collective / XCCL
    # communicator formation" theory recorded here. That theory predicted 1N
    # would always be correct; the failure reproduces at 1N (job 8817030 and
    # every run above), where there is no cross-node collective at all. The
    # earlier mitigations (int32 MIN, fp32 MIN, fp32 SUM, added barrier) all
    # failed because every one of them kept numel=1 fp32/int32 -- none varied
    # the axis that actually matters. Jobs for the record: 8788586 (1N),
    # 8788736 / 8789128 / 8789134 (2N) all false-negative; 8788597 (1N) passed,
    # i.e. intermittent rather than node-count-determined.
    #
    # Fix: reduce an int64 counter. That is also the semantically correct type
    # for counting ready ranks, and it sidesteps the fp32 scalar path entirely.
    # Do NOT "simplify" this back to a float flag.
    if world_size > 1 and dist.is_initialized():
        dist.barrier()
        _val_ready = torch.tensor(
            [1 if val_loader is not None else 0], dtype=torch.int64, device=device
        )
        dist.all_reduce(_val_ready, op=dist.ReduceOp.SUM)
        _ready_count = int(_val_ready.item())
        _all_ranks_ready = _ready_count >= world_size
        if not _all_ranks_ready and val_loader is not None:
            logger.warning(
                f"[Native DDP] Disabling validation: only {_ready_count}/"
                f"{world_size} ranks built a validation loader "
                f"(see per-rank warnings above)."
            )
            val_loader = None
            val_iter = None
    # ---------------------------------------------

    # === Release cached memory from model loading ===
    # NOTE: torch.xpu.empty_cache() is intentionally NOT called here.
    # Calling empty_cache() after FSDP wrap triggers zeMemAllocDevice/zeMemFree cycles
    # via FSDP's storage.resize_() during subsequent forward passes, leaking Level Zero
    # UR handles. With 13+ FSDP units this crashes training after ~70 iterations
    # (UR_RESULT_ERROR_OUT_OF_RESOURCES); OLMo-3 7B uses 32+ units. The caching
    # allocator reuses freed blocks without touching Level Zero, so omitting
    # empty_cache() is safe. (torchtune Apr 2026)

    # === Memory snapshot before training ===
    if is_main and hasattr(torch, "xpu") and torch.xpu.is_available():
        alloc_gb = torch.xpu.memory_allocated() / 1e9
        reserved_gb = torch.xpu.memory_reserved() / 1e9
        total_gb = torch.xpu.get_device_properties(0).total_memory / 1e9
        logger.info(
            f"[Memory] Before training loop: {alloc_gb:.2f}GB allocated, {reserved_gb:.2f}GB reserved ({total_gb:.0f}GB HBM)"
        )

    # === TRAINING LOOP (ALL RANKS MUST PARTICIPATE) ===
    # Per-rank timing collection for straggler detection
    rank_step_times = []  # Collect (step, rank, phase, time) tuples

    # DDP diagnostic mode: verbose per-microbatch prints for debugging hangs
    ddp_debug = os.environ.get("DDP_DEBUG", "0") == "1"
    ddp_debug_steps = int(
        os.environ.get("DDP_DEBUG_STEPS", "3")
    )  # How many steps to debug
    if ddp_debug and is_main:
        logger.info(f"[DDP_DEBUG] Enabled for first {ddp_debug_steps} steps")

    _first_batch_fetched = False  # Track whether we've logged first-batch timing
    # Hoist perf-related env-var reads out of the inner loop (read once at
    # training-loop entry). Cheap in absolute terms but keeps the hot path clean.
    _fsdp_no_sync_accum = os.environ.get("FSDP_NO_SYNC_ACCUM", "0") == "1"
    _perf_probes_disabled = os.environ.get("PRISM_DISABLE_PERF_PROBES", "0") == "1"
    _grad_norm_interval = int(os.environ.get("GRAD_NORM_INTERVAL", "50"))

    # Hoisted out of the per-step eval blocks below: model modalities don't
    # change during training, and both the training-loss-proxy eval record
    # and the held-out-validation eval record need this same family list.
    # Computing it inline inside a `try/except` gated on `is_main and
    # should_emit_eval_loss` left `_emit_families` referenced-before-assignment
    # (NameError -> caught by a bare except elsewhere, or uncaught and fatal)
    # on the first eval step of a run where that guard's try body threw before
    # reaching the assignment.
    _eval_unwrapped = model.module if hasattr(model, "module") else model
    _eval_modalities = model_modalities(_eval_unwrapped) or []
    # Drop "text" — the trainer's mean_loss IS the text-side next-token loss,
    # but the collector maps modality names (image / time_series / graph) to
    # the loss_* columns. Emitting under the non-text modality is the closer
    # match for IsoFLOP fit consumption.
    _emit_families = [m for m in _eval_modalities if m != "text"] or _eval_modalities

    while step < max_steps:
        data_iter = _maybe_reset_finite_epoch(
            data_iter, train_loader, step, "training"
        )
        step_start_time = time.perf_counter()
        _watchdog_start(step, "start")
        optimizer.zero_grad()
        accum_loss = 0.0
        accum_weighted_loss_sum = 0.0
        accum_loss_tokens = 0

        if ddp_debug and step < ddp_debug_steps and is_main:
            logger.info(
                f"[DDP_DEBUG] Step {step}: zero_grad done, starting accum loop (grad_accum={grad_accum})",
            )

        # OPTIMIZATION: Per-phase syncs add overhead (64+ syncs/step with grad_accum=16)
        # Only sync every phase when DEBUG_SYNC=1, otherwise sync only at step end
        debug_sync = os.environ.get("DEBUG_SYNC", "0") == "1"

        # Production mode: skip ALL non-essential torch.xpu.synchronize() calls.
        # This removes CPU-GPU serialization after backward, grad clip, and optimizer,
        # allowing the CPU to overlap data prefetch with GPU compute. Timing numbers
        # will be inaccurate but throughput improves ~3-8%.
        # Set FSDP_PRODUCTION_MODE=1 to enable.
        _production_mode = os.environ.get("FSDP_PRODUCTION_MODE", "0") == "1"
        if _production_mode and is_main:
            logger.info("[PERF] Production mode: non-essential XPU syncs DISABLED")

        # === PyTorch Profiler: Start profiling for specific steps ===
        should_profile = enable_profiler and is_main and step in profiler_steps
        if should_profile:
            from torch.profiler import ProfilerActivity, profile

            profiler_dir = os.path.join(config.output_dir, "profiler")
            profiler = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.XPU]
                if hasattr(ProfilerActivity, "XPU")
                else [ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                with_stack=True,
                profile_memory=True,
            )
            profiler.__enter__()
            logger.info(f"[Profiler] Started profiling step {step}")

        for accum_idx in range(grad_accum):
            _dbg = ddp_debug and step < ddp_debug_steps and is_main

            # Data fetch (with timeout to detect DAOS/dfuse stalls)
            _watchdog_start(step, f"data_load(accum={accum_idx})")
            if _dbg:
                logger.info(
                    f"[DDP_DEBUG] Step {step} accum {accum_idx}/{grad_accum}: fetching data...",
                )
            t0 = time.perf_counter()
            batch, data_iter = _fetch_batch(data_iter, train_loader, step, accum_idx)
            if not _first_batch_fetched:
                _dt_first_batch = time.perf_counter() - t0
                _first_batch_fetched = True
                if is_main:
                    logger.info(
                        f"[DataLoader] First batch fetch took {_dt_first_batch:.1f}s "
                        f"(includes buffer fill for BucketedMultiWebDatasetWrapper)"
                    )
            batch = _move_batch_to_device(batch, device)
            data_time = time.perf_counter() - t0
            timing_data += data_time

            # tokens_per_window is rank-0-local; the perf record below scales
            # by world_size to mirror how samples_per_sec is aggregated.
            # last_batch_modality_counts is a per-step snapshot so a single
            # row of perf.jsonl shows what the dataloader actually emitted.
            if is_main:
                try:
                    perf_batch = batch.get("inputs", batch)
                    tokens_per_window += batch_token_count(
                        perf_batch,
                        pad_id=_pad_id_for_tokens,
                    )
                    last_batch_modality_counts = batch_modality_counts(perf_batch)
                    # IsoFLOP: per-sample non-pad text lengths for the
                    # sequence_stats helper. Trimmed to text==2D path
                    # (which is every PRISM trainer batch); other shapes
                    # would have already errored in batch_token_count.
                    _text = perf_batch.get("text")
                    if isinstance(_text, torch.Tensor) and _text.dim() == 2:
                        if _pad_id_for_tokens is None:
                            seq_lens_window.extend(
                                [int(_text.shape[1])] * int(_text.shape[0])
                            )
                        else:
                            seq_lens_window.extend(
                                int(n) for n in (
                                    (_text != _pad_id_for_tokens).sum(dim=1).tolist()
                                )
                            )
                except Exception as _e_perf:  # noqa: BLE001
                    # perf telemetry must never kill training
                    logger.debug(f"[perf] modality/token count skipped: {_e_perf}")
            if _dbg:
                logger.info(
                    f"[DDP_DEBUG] Step {step} accum {accum_idx}: data loaded ({data_time:.3f}s)",
                )

            # Per-rank timing: data loading
            if per_rank_timing and step % log_every == 0:
                rank_step_times.append((step, rank, "data", accum_idx, data_time))

            # Use no_sync for DDP gradient accumulation to defer AllReduce
            # to the last micro-batch. For FSDP SHARD_GRAD_OP, do NOT use no_sync:
            # each backward does ReduceScatter which shards gradients to ~1/N size,
            # saving significant memory. Skipping that would keep full-size gradients
            # (14GB for 7B) causing OOM.
            #
            # CRITICAL: Do NOT use no_sync() on step 0 when static_graph=True.
            # DDP with static_graph needs the first backward pass to call
            # prepare_for_backward() (which sets expect_autograd_hooks_=true)
            # before _DDPSink.backward() queues delay_all_reduce ->
            # finalize_backward(). no_sync() sets require_backward_grad_sync=False,
            # which skips prepare_for_backward(), causing an internal assertion
            # failure: "expect_autograd_hooks_ INTERNAL ASSERT FAILED" in
            # reducer.cpp:1660.
            is_last_accum = accum_idx == grad_accum - 1
            # FSDP_NO_SYNC_ACCUM=1 (hoisted above the loop) opts FSDP/HSDP into
            # no_sync() on non-final accumulation microbatches. Default OFF
            # because no_sync() keeps full-size unsharded gradients in HBM
            # (≈14 GB for a 7B model) — only safe when the model is small enough
            # that 2x grad HBM fits. For Qwen3-0.6B (~1.4 GB grads) with ~30 GB
            # headroom this is the primary lever for cutting inter-node sync.
            use_no_sync = (
                not is_last_accum
                and world_size > 1
                and (not is_fsdp_model or _fsdp_no_sync_accum)
                and step > 0  # Let DDP initialize static graph on first step
            )
            sync_ctx = model.no_sync() if use_no_sync else contextlib.nullcontext()
            if _dbg:
                logger.info(
                    f"[DDP_DEBUG] Step {step} accum {accum_idx}: no_sync={use_no_sync}, is_last_accum={is_last_accum}",
                )

            with sync_ctx:
                # Forward
                _watchdog_start(step, f"forward(accum={accum_idx})")
                if _dbg:
                    logger.info(
                        f"[DDP_DEBUG] Step {step} accum {accum_idx}: entering forward...",
                    )
                t1 = time.perf_counter()
                with torch.autocast(
                    device_type=(
                        "xpu"
                        if hasattr(torch, "xpu") and torch.xpu.is_available()
                        else "cuda"
                    ),
                    dtype=torch.bfloat16,
                ):
                    logits, loss, decoder_losses = forward_training_batch(model, batch)
                    if decoder_losses and is_main and step % log_every == 0:
                        logger.info(
                            "Decoder losses: %s",
                            {
                                k: float(v.detach())
                                for k, v in decoder_losses.items()
                            },
                        )
                    unscaled_loss_value = float(loss.detach().item())
                    # Mixed native objectives are weighted per example, not by
                    # prompt token count (instruction text is not a target).
                    loss_tokens = (
                        batch["inputs"]["text"].shape[0]
                        if "requested_outputs" in batch
                        else _loss_token_count(batch, _pad_id_for_tokens)
                    )
                    if loss_tokens > 0:
                        accum_weighted_loss_sum += unscaled_loss_value * loss_tokens
                        accum_loss_tokens += loss_tokens
                    loss = loss / grad_accum
                if debug_sync and hasattr(torch, "xpu") and torch.xpu.is_available():
                    torch.xpu.synchronize()
                fwd_time = time.perf_counter() - t1
                timing_fwd += fwd_time
                if _dbg:
                    logger.info(
                        f"[DDP_DEBUG] Step {step} accum {accum_idx}: forward done ({fwd_time:.3f}s)",
                    )

                # Per-rank timing: forward pass
                if per_rank_timing and step % log_every == 0:
                    rank_step_times.append((step, rank, "fwd", accum_idx, fwd_time))

                # Backward (with detailed per-step timing)
                _watchdog_start(step, f"backward(accum={accum_idx})")
                if _dbg:
                    logger.info(
                        f"[DDP_DEBUG] Step {step} accum {accum_idx}: entering backward...",
                    )
                t2 = time.perf_counter()
                loss.backward()
                if _dbg:
                    logger.info(
                        f"[DDP_DEBUG] Step {step} accum {accum_idx}: backward kernel dispatched, "
                        f"syncing... loss={unscaled_loss_value:.4f}",
                    )
                if hasattr(torch, "xpu") and torch.xpu.is_available() and not _production_mode:
                    torch.xpu.synchronize()
                bwd_time = time.perf_counter() - t2
                timing_bwd += bwd_time
                if _dbg:
                    logger.info(
                        f"[DDP_DEBUG] Step {step} accum {accum_idx}: backward done ({bwd_time:.3f}s)",
                    )

            # Log per-accumulation backward time every 10 steps
            if step % log_every == 0 and is_main:
                per_step_bwd_times.append(bwd_time)

            # Per-rank timing: backward pass
            if per_rank_timing and step % log_every == 0:
                rank_step_times.append((step, rank, "bwd", accum_idx, bwd_time))

            accum_loss += unscaled_loss_value

        # Gradient clipping (this does an AllReduce of norms!)
        # FSDP has its own clip_grad_norm_ that handles sharded gradients correctly
        _dbg_step = ddp_debug and step < ddp_debug_steps and is_main
        if _dbg_step:
            logger.info(
                f"[DDP_DEBUG] Step {step}: all accum done, clipping gradients...",
            )
        t_clip = time.perf_counter()
        if is_fsdp_model:
            model.clip_grad_norm_(1.0)
        else:
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        if hasattr(torch, "xpu") and torch.xpu.is_available() and not _production_mode:
            torch.xpu.synchronize()
        clip_time = time.perf_counter() - t_clip

        # Optimizer step
        if _dbg_step:
            logger.info(
                f"[DDP_DEBUG] Step {step}: clip done ({clip_time:.3f}s), optimizer step...",
            )
        t3 = time.perf_counter()
        optimizer.step()
        scheduler.step()
        if hasattr(torch, "xpu") and torch.xpu.is_available() and not _production_mode:
            torch.xpu.synchronize()
        opt_time = time.perf_counter() - t3
        timing_opt += opt_time

        # === Post-step-0 memory snapshot ===
        # Log peak memory after the first full forward+backward+optimizer step.
        # This is the critical number for sizing batch_size: if reserved > HBM,
        # the backward will fail with UR_RESULT_ERROR_OUT_OF_RESOURCES.
        if step == 0 and is_main and hasattr(torch, "xpu") and torch.xpu.is_available():
            alloc_gb = torch.xpu.memory_allocated() / 1e9
            reserved_gb = torch.xpu.memory_reserved() / 1e9
            total_gb = torch.xpu.get_device_properties(0).total_memory / 1e9
            logger.info(
                f"[Memory] After step 0: {alloc_gb:.2f}GB allocated, "
                f"{reserved_gb:.2f}GB reserved ({total_gb:.0f}GB HBM, "
                f"{total_gb - reserved_gb:.1f}GB headroom)"
            )

        # === Periodic XPU Cache Clearing (DISABLED by default, USE WITH CAUTION) ===
        # WARNING: torch.xpu.empty_cache() combined with FSDP's storage.resize_() cycle
        # leaks Level Zero UR handles. With 13+ FSDP units, this crashes training after
        # ~70 iterations (UR_RESULT_ERROR_OUT_OF_RESOURCES). Root-caused Apr 2026.
        # DO NOT enable PRISM_CACHE_CLEAR_INTERVAL in FSDP training jobs.
        # Only safe for DDP (no storage.resize_() cycles). Default is 0 (off).
        # Set PRISM_CACHE_CLEAR_INTERVAL=N to clear every N steps (0=off).
        cache_clear_interval = int(os.environ.get("PRISM_CACHE_CLEAR_INTERVAL", "0"))
        if cache_clear_interval > 0 and step > 0 and step % cache_clear_interval == 0:
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                import gc

                gc.collect()
                torch.xpu.empty_cache()
                torch.xpu.synchronize()
                if is_main:
                    alloc_mb = torch.xpu.memory_allocated() / 1024**2
                    reserved_mb = torch.xpu.memory_reserved() / 1024**2
                    logger.info(
                        f"  [CACHE-CLEAR] Step {step}: gc.collect() + empty_cache() done. "
                        f"Alloc: {alloc_mb:.0f}MB, Reserved: {reserved_mb:.0f}MB",
                    )

        if _dbg_step:
            step_total = time.perf_counter() - step_start_time
            logger.info(
                f"[DDP_DEBUG] Step {step}: COMPLETE ({step_total:.3f}s total, opt={opt_time:.3f}s)",
            )

        # Step completed successfully - cancel watchdog
        _watchdog_stop()

        # Per-rank timing: optimizer step
        if per_rank_timing and step % log_every == 0:
            rank_step_times.append((step, rank, "opt", 0, opt_time))

        # === PyTorch Profiler: Stop and save profile ===
        if should_profile and profiler is not None:
            profiler.__exit__(None, None, None)
            # Export chrome trace
            profiler_dir = os.path.join(config.output_dir, "profiler")
            trace_file = os.path.join(profiler_dir, f"trace_step_{step}.json")
            profiler.export_chrome_trace(trace_file)
            logger.info(f"[Profiler] Saved trace for step {step} to {trace_file}")

            # Print key averages
            logger.info(f"[Profiler] Step {step} Key Averages:")
            logger.info(
                profiler.key_averages().table(
                    sort_by="self_cpu_time_total", row_limit=20
                )
            )
            profiler = None

        # Measure XCCL latency floor with a 1-element AllReduce probe.
        # This is a HEALTH-CHECK / SENTINEL — it answers "can ranks talk to each
        # other at all, and what is the per-op latency floor on a tiny payload?".
        # It does NOT measure the cost of the per-step gradient AllReduce.
        #
        # For an E2E run with ~1.3 GB of trainable params and ~50 DDP buckets,
        # the per-step AR cost is ~1.5 s (measurable only via a kineto trace's
        # c10d::allreduce_ CPU-op total), while this probe lands at ~30 ms
        # because it pays only XCCL's per-op latency. Conflating the two
        # sends performance investigations in the wrong direction; see
        # scaling-study/results/qwen3_e2e_perf_investigation.md for a worked
        # example.
        #
        # PRISM_DISABLE_PERF_PROBES=1 (hoisted above the loop) skips this — the
        # probe is cheap on its own but the torch.xpu.synchronize() that wraps it
        # drains all in-flight collectives and pollutes throughput for the next
        # logging window. Enable for clean perf runs.
        if step % 50 == 0 and world_size > 1 and not _perf_probes_disabled:
            # numel=2, not 1: a numel=1 fp32 allreduce hits the oneCCL defect
            # documented at the readiness gate (it drops the last rank), so a
            # scalar probe would be timing a degenerate collective rather than
            # the real one.
            test_tensor = torch.ones(2, device=device)
            t_ar = time.perf_counter()
            dist.all_reduce(test_tensor, op=dist.ReduceOp.SUM)
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                torch.xpu.synchronize()
            timing_allreduce = time.perf_counter() - t_ar

        # === Per-Rank Timing: Total step time and cross-rank comparison ===
        step_total_time = time.perf_counter() - step_start_time
        if per_rank_timing and step % log_every == 0:
            rank_step_times.append((step, rank, "total", 0, step_total_time))

            # Every 50 steps, gather timing from all ranks and report stragglers
            if step % 50 == 0 and world_size > 1:
                # Gather step times from all ranks
                step_time_tensor = torch.tensor(
                    [step_total_time], device=device, dtype=torch.float32
                )
                all_step_times = [
                    torch.zeros(1, device=device, dtype=torch.float32)
                    for _ in range(world_size)
                ]
                dist.all_gather(all_step_times, step_time_tensor)

                if is_main:
                    times = [t.item() for t in all_step_times]
                    min_time, max_time = min(times), max(times)
                    mean_time = sum(times) / len(times)
                    slowest_rank = times.index(max_time)
                    fastest_rank = times.index(min_time)
                    straggler_ratio = max_time / min_time if min_time > 0 else 1.0

                    logger.info(f"  [PER-RANK TIMING] Step {step}:")
                    logger.info(f"    Fastest: rank {fastest_rank} ({min_time:.3f}s)")
                    logger.info(f"    Slowest: rank {slowest_rank} ({max_time:.3f}s)")
                    logger.info(
                        f"    Mean: {mean_time:.3f}s, Straggler ratio: {straggler_ratio:.2f}x"
                    )

                    if straggler_ratio > 1.5:
                        logger.warning(
                            f"    WARNING: Significant straggler detected! Rank {slowest_rank} is {straggler_ratio:.2f}x slower"
                        )

        timing_steps += 1
        step += 1
        local_rank_mean_loss = accum_loss / grad_accum
        should_log_train_loss = step % log_every == 0
        eval_enabled = getattr(config, "eval_enabled", False)
        eval_interval = getattr(config, "eval_every_n_steps", 0)
        should_emit_eval_loss = bool(
            eval_enabled and eval_interval and step > 0 and step % eval_interval == 0
        )
        global_loss_stats = None
        if should_log_train_loss or should_emit_eval_loss:
            global_loss_stats = _global_loss_stats(
                local_rank_mean_loss,
                accum_weighted_loss_sum,
                accum_loss_tokens,
            )

        # IsoFLOP eval gate (training-loss proxy variant for native trainer).
        # PR-1's trainer_zone_a has a full evaluator suite; the native trainer
        # doesn't, so we emit the current optimizer step's running training
        # loss as the per-family eval signal. Lives OUTSIDE the `log_every`
        # block so eval cadence is `eval_every_n_steps` exactly, not the LCM
        # of (log_every, eval_every_n_steps) — without this, an
        # eval_every_n_steps=4 + log_every=10 combo would silently fire only
        # at step 40 instead of 4,8,12,…40 (see PR feedback on PR #98).
        # Gated on `eval_enabled` so non-IsoFLOP runs stay bit-identical.
        if is_main and should_emit_eval_loss:
            try:
                _eval_mean_loss = (
                    global_loss_stats["token_mean"]
                    if global_loss_stats is not None
                    else local_rank_mean_loss
                )
                for _family in _emit_families:
                    log_perf_record(
                        getattr(config, "output_dir", None),
                        {
                            "event": "eval",
                            "site": "trainer_native",
                            "step": step,
                            "family": _family,
                            "loss": float(_eval_mean_loss),
                            "loss_source": "train_running_mean",
                            "sweep_id": getattr(config, "sweep_id", None),
                            "preset": getattr(config, "preset", None),
                        },
                    )
            except Exception as _e_eval_perf:  # noqa: BLE001
                logger.debug(
                    f"[perf] eval per-family loss record skipped: {_e_eval_perf}"
                )

            # Held-out validation forward runs on rank 0 only. Under FSDP the
            # model's forward issues all_gather collectives that every rank must
            # join — a rank-0-only forward would deadlock the whole job (the
            # _global_loss_stats all_reduce above already ran on all ranks, so
            # the other ranks have moved on to the next step's collective).
            # DDP replicas hold the full model, so rank-0-only eval is safe there.
            # If FSDP held-out eval is needed later, make it an all-ranks
            # collective with a val loader on every rank.
            if is_fsdp_model and val_loader is not None:
                logger.info(
                    "[Native DDP] Skipping held-out validation forward under FSDP "
                    "(rank-0-only forward would deadlock on all_gather); "
                    "use loss/token_mean for the IsoFLOP loss signal instead."
                )
            if (
                not is_fsdp_model
                and val_loader is not None
                and val_iter is not None
            ):
                eval_batches = max(1, int(getattr(config, "validation_batches", 8)))
                unwrapped_eval_model = model.module if hasattr(model, "module") else model
                was_training = unwrapped_eval_model.training
                val_losses: list[float] = []
                unwrapped_eval_model.eval()
                try:
                    with torch.no_grad():
                        for _ in range(eval_batches):
                            try:
                                val_batch = next(val_iter)
                            except StopIteration:
                                val_iter = iter(val_loader)
                                val_batch = next(val_iter)
                            val_batch = _move_batch_to_device(val_batch, device)
                            with torch.autocast(
                                device_type=(
                                    "xpu"
                                    if hasattr(torch, "xpu") and torch.xpu.is_available()
                                    else "cuda"
                                ),
                                dtype=torch.bfloat16,
                            ):
                                _, val_loss, _ = forward_training_batch(
                                    unwrapped_eval_model,
                                    val_batch,
                                )
                            if val_loss is not None:
                                val_losses.append(float(val_loss.detach().item()))
                except Exception as e_val_loss:  # noqa: BLE001
                    logger.warning(
                        f"[Native DDP] Validation loss failed at step {step}: {e_val_loss}"
                    )
                finally:
                    if was_training:
                        unwrapped_eval_model.train()

                # Reduce over the WORLD, not the shard group: each rank evaluated a
                # disjoint slice of the validation set (see _StridedTSValidation),
                # so the correct figure is a sample-weighted mean over every slice,
                # not any one rank's slice.
                #
                # This collective was previously disabled: it segfaulted 4/4 runs
                # inside XCCL immediately after the eval forward, and an added
                # torch.xpu.synchronize() did not help. That was the same oneCCL
                # numel=1 fp32 defect documented at the readiness gate above (it
                # reduced a scalar fp32 loss and a scalar fp32 count). Reducing a
                # 2-element fp64 [loss_sum, batch_count] buffer instead is clean:
                # verified 2026-09-10 at 2N/24 ranks, no crash, all 24 ranks
                # agreeing on (1185.3770, 192) at step 2 and (1168.7746, 192) at
                # step 4. Keep this fp64 and keep both stats in ONE buffer.
                val_batch_count = len(val_losses)
                val_loss_sum = float(sum(val_losses))
                val_is_global = False
                if world_size > 1 and dist.is_initialized():
                    _val_stats = torch.tensor(
                        [val_loss_sum, float(val_batch_count)],
                        dtype=torch.float64,
                        device=device,
                    )
                    dist.all_reduce(_val_stats, op=dist.ReduceOp.SUM)
                    val_loss_sum = float(_val_stats[0].item())
                    val_batch_count = int(_val_stats[1].item())
                    val_is_global = True
                val_loss_mean = (
                    val_loss_sum / val_batch_count if val_batch_count else None
                )

                if val_loss_mean is not None and is_main:
                    _scope = "all ranks" if val_is_global else "rank 0 local slice"
                    logger.info(
                        f"[Native DDP] Validation loss at step {step}: "
                        f"{val_loss_mean:.4f} over {val_batch_count} batches "
                        f"({val_batch_count * int(getattr(config, 'batch_size', 0) or 0)} "
                        f"samples, {_scope})"
                    )
                    if wandb.run:
                        wandb.log(
                            {
                                "val/loss": val_loss_mean,
                                "eval/loss": val_loss_mean,
                                "val/batches": val_batch_count,
                                "val/samples": val_batch_count
                                * int(getattr(config, "batch_size", 0) or 0),
                            },
                            step=step,
                        )
                    for _family in _emit_families:
                        log_perf_record(
                            getattr(config, "output_dir", None),
                            {
                                "event": "eval",
                                "site": "trainer_native",
                                "step": step,
                                "family": _family,
                                "loss": float(val_loss_mean),
                                "loss_source": "held_out_validation",
                                "sweep_id": getattr(config, "sweep_id", None),
                                "preset": getattr(config, "preset", None),
                            },
                        )

        # Logging
        if should_log_train_loss and is_main:
            global_loss = (
                global_loss_stats["token_mean"]
                if global_loss_stats is not None
                else local_rank_mean_loss
            )
            rank_mean_loss = (
                global_loss_stats["rank_mean"]
                if global_loss_stats is not None
                else local_rank_mean_loss
            )
            logger.info(
                f"Step {step}: Loss {global_loss:.4f} "
                f"(rank_mean={rank_mean_loss:.4f}, rank0={local_rank_mean_loss:.4f}, "
                f"{'examples' if 'requested_outputs' in batch else 'tokens'}="
                f"{int(global_loss_stats['global_tokens']) if global_loss_stats else accum_loss_tokens})"
            )

            # Log LRs during warmup phase to verify scheduler is working
            if step <= 100 or step % 100 == 0:
                lr_strs = []
                for i, pg in enumerate(optimizer.param_groups):
                    group_name = pg.get("name", f"group_{i}")
                    lr_strs.append(f"{group_name}={pg['lr']:.2e}")
                logger.info(f"  [LR] {'  '.join(lr_strs)}")

            # Print per-accumulation backward times
            if per_step_bwd_times:
                bwd_str = ", ".join([f"{t:.3f}s" for t in per_step_bwd_times])
                logger.info(f"  [DETAIL] Per-accum Bwd times: [{bwd_str}]")
                per_step_bwd_times = []

            # Print timing breakdown (Data / Forward / Backward / Optimizer)
            if timing_steps > 0:
                samples_per_step = config.batch_size * grad_accum * world_size
                total_time = timing_data + timing_fwd + timing_bwd + timing_opt
                samples_per_sec = (
                    (samples_per_step * timing_steps) / total_time
                    if total_time > 0
                    else 0
                )
                logger.info(
                    f"  [TIMING] Data: {timing_data / timing_steps:.3f}s | Fwd: {timing_fwd / timing_steps:.3f}s | Bwd: {timing_bwd / timing_steps:.3f}s | Opt: {timing_opt / timing_steps:.3f}s"
                )
                logger.info(
                    f"  [THROUGHPUT] {samples_per_sec:.1f} samples/sec (effective batch: {samples_per_step})"
                )
                # tokens_per_window is rank-0's text-token count across the
                # window. Scale by world_size to mirror samples_per_sec
                # aggregation; assumes ranks process equivalent batches.
                global_tokens_window = tokens_per_window * world_size
                tokens_per_sec = (
                    global_tokens_window / total_time if total_time > 0 else 0.0
                )
                tokens_per_batch = (
                    global_tokens_window / timing_steps if timing_steps > 0 else 0.0
                )

                # Peak memory usage (helps calibrate batch size)
                perf_record: dict = {
                    "site": "trainer_native_per_log",
                    "step": step,
                    "samples_per_sec": samples_per_sec,
                    "samples_per_step": samples_per_step,
                    "tokens_per_sec": tokens_per_sec,
                    "tokens_per_batch": tokens_per_batch,
                    "batch_modality_counts": dict(last_batch_modality_counts),
                    "world_size": world_size,
                    "batch_size": config.batch_size,
                    "grad_accum": grad_accum,
                    "dist_strategy": dist_strategy,
                    "sweep_id": getattr(config, "sweep_id", None),
                    "preset": getattr(config, "preset", None),
                    "modalities": model_modalities(model),
                    "loss": global_loss,
                    "loss_rank_mean": rank_mean_loss,
                    "loss_rank0": local_rank_mean_loss,
                    (
                        "loss_examples_global"
                        if "requested_outputs" in batch
                        else "loss_tokens_global"
                    ): (
                        global_loss_stats["global_tokens"]
                        if global_loss_stats is not None
                        else accum_loss_tokens
                    ),
                }
                if hasattr(torch, "xpu") and torch.xpu.is_available():
                    peak_gb = torch.xpu.max_memory_allocated() / 1e9
                    cur_gb = torch.xpu.memory_allocated() / 1e9
                    hbm_gb = torch.xpu.get_device_properties(0).total_memory / 1e9
                    logger.info(
                        f"  [MEMORY] Peak: {peak_gb:.2f}GB, Current: {cur_gb:.2f}GB (of {hbm_gb:.0f}GB HBM)"
                    )
                    perf_record["peak_gb"] = peak_gb
                    perf_record["current_gb"] = cur_gb

                log_perf_record(getattr(config, "output_dir", None), perf_record)

            # Basic WandB Log (Loss + LR for all param groups)
            if wandb.run:
                log_dict = {
                    "loss": global_loss,
                    "loss/rank_mean": rank_mean_loss,
                    "loss/rank0": local_rank_mean_loss,
                    (
                        "loss/examples_global"
                        if "requested_outputs" in batch
                        else "loss/tokens_global"
                    ): (
                        global_loss_stats["global_tokens"]
                        if global_loss_stats is not None
                        else accum_loss_tokens
                    ),
                }
                # Log LR for each param group (supports differential LRs)
                for i, pg in enumerate(optimizer.param_groups):
                    group_name = pg.get("name", f"group_{i}")
                    log_dict[f"lr/{group_name}"] = pg["lr"]
                # Also log primary LR for backward compatibility
                log_dict["lr"] = optimizer.param_groups[0]["lr"]
                wandb.log(log_dict, step=step)

        # --- Encoder/Projector Gradient Norm Logging ---
        # GRAD_NORM_INTERVAL=0 disables (default 50, hoisted above the loop).
        # Each fire walks every named parameter and calls .norm().item() on its
        # grad — that's a full-model traversal with O(params) host syncs. Cheap
        # on a small model but each .item() drains the XPU queue, polluting
        # throughput for the next window. Disable for clean perf runs.
        if _grad_norm_interval > 0 and step % _grad_norm_interval == 0 and is_main:
            # Compute gradient norms per component to verify training
            grad_norms: dict[str, list[float]] = {}
            for name, param in model.named_parameters():
                if param.grad is not None:
                    norm = param.grad.norm().item()
                    # Categorize by component
                    if "encoders" in name:
                        key = "encoder"
                    elif "projectors" in name:
                        key = "projector"
                    elif "backbone" in name:
                        key = "backbone"
                    else:
                        key = "other"

                    if key not in grad_norms:
                        grad_norms[key] = []
                    grad_norms[key].append(norm)

            # Print summary
            grad_summary = []
            for key, norms in grad_norms.items():
                if norms:
                    mean_norm = sum(norms) / len(norms)
                    max_norm = max(norms)
                    grad_summary.append(
                        f"{key}: mean={mean_norm:.6f}, max={max_norm:.6f}"
                    )

            if grad_summary:
                logger.info(f"  [GRAD NORMS] {'  |  '.join(grad_summary)}")

            # Log to WandB
            if wandb.run:
                for key, norms in grad_norms.items():
                    if norms:
                        wandb.log(
                            {
                                f"grad_norm/{key}_mean": sum(norms) / len(norms),
                                f"grad_norm/{key}_max": max(norms),
                            },
                            step=step,
                        )

        # --- Visualization Logging ---
        viz_interval = getattr(config, "viz_every_n_steps", 500)
        if (
            viz_interval > 0
            and step % viz_interval == 0
            and is_main
            and wandb.run
            and visualizer is not None
        ):
            try:
                import traceback as tb

                # Fetch Validation Batch
                if val_iter is not None:
                    try:
                        val_batch = next(val_iter)
                    except StopIteration:
                        # Restart iterator
                        if val_loader is not None:
                            val_iter = iter(val_loader)
                            val_batch = next(val_iter)
                        else:
                            val_batch = None
                else:
                    val_batch = None

                # Use val_batch if available, else fallback to current train batch.
                viz_batch = val_batch if val_batch is not None else batch
                batch_src = "VAL" if val_batch is not None else "TRAIN (Fallback)"
                logger.info(f"  [Viz] Using {batch_src} batch for visualization.")

                # Move to device
                viz_batch = _move_batch_to_device(viz_batch, device)

                viz_results = visualizer.visualize(viz_batch)

                # Prediction Monitoring
                try:
                    pred_table = visualizer.visualize_predictions(viz_batch, model)
                    if pred_table:
                        viz_results["viz/predictions"] = pred_table
                except Exception as e_gen:
                    logger.error(f"  [Viz] Error generating predictions: {e_gen}")
                    tb.print_exc()

                if viz_results:
                    wandb.log(viz_results, step=step)
                    logger.info(f"  [Viz] Logged {len(viz_results)} visualization items")
            except Exception as e:
                import traceback as tb

                logger.error(f"  [Viz] Error visualizing batch: {e}")
                tb.print_exc()

        if step % 50 == 0 and is_main:
            avg_data = timing_data / timing_steps
            avg_fwd = timing_fwd / timing_steps
            avg_bwd = timing_bwd / timing_steps
            avg_opt = timing_opt / timing_steps
            total = avg_data + avg_fwd + avg_bwd + avg_opt

            mem_alloc = torch.xpu.memory_allocated() / 1024**2 if hasattr(torch, "xpu") else 0
            mem_reserved = torch.xpu.memory_reserved() / 1024**2 if hasattr(torch, "xpu") else 0
            samples_per_sec = (config.batch_size * world_size * grad_accum) / total

            logger.info(
                f"[TIMING Step {step}] Total: {total:.2f}s | "
                f"Data: {avg_data:.2f}s ({100 * avg_data / total:.1f}%) | "
                f"Fwd: {avg_fwd:.2f}s ({100 * avg_fwd / total:.1f}%) | "
                f"Bwd: {avg_bwd:.2f}s ({100 * avg_bwd / total:.1f}%) | "
                f"Opt: {avg_opt:.2f}s ({100 * avg_opt / total:.1f}%) | "
                f"Throughput: {samples_per_sec:.1f} samp/s"
            )
            logger.info(
                f"       [MEMORY] Allocated: {mem_alloc:.0f}MB | Reserved: {mem_reserved:.0f}MB | "
                f"XCCL probe (1-elem AR, NOT per-step grad cost): {timing_allreduce * 1000:.1f}ms | "
                f"Clip time: {clip_time * 1000:.1f}ms"
            )

            global_tokens_window_50 = tokens_per_window * world_size
            tokens_per_sec_50 = (
                global_tokens_window_50 / (total * timing_steps)
                if total > 0 and timing_steps > 0
                else 0.0
            )
            tokens_per_batch_50 = (
                global_tokens_window_50 / timing_steps if timing_steps > 0 else 0.0
            )
            # IsoFLOP per-50 extensions: sequence-length distribution,
            # analytic FLOP accounting, projector capacity knobs. The flop
            # counter advances `timing_steps` at a time (one flush ≈ 50
            # training steps); passing `timing_steps` is critical for
            # cumulative_flops to track actual budget — without it the
            # counter undercounts by the window size (see PR feedback on
            # PR #98). When uncalibrated, both values come out None.
            _seq_stats = sequence_stats(seq_lens_window)
            _fps_window, _cum_flops = flop_counter.step(n_steps=timing_steps)
            _unwrapped_cfg = getattr(
                model.module if hasattr(model, "module") else model,
                "config",
                None,
            )
            log_perf_record(
                getattr(config, "output_dir", None),
                {
                    "site": "trainer_native_per_50",
                    "step": step,
                    "samples_per_sec": samples_per_sec,
                    "tokens_per_sec": tokens_per_sec_50,
                    "tokens_per_batch": tokens_per_batch_50,
                    "batch_modality_counts": dict(last_batch_modality_counts),
                    "total_step_s": total,
                    "data_s": avg_data,
                    "fwd_s": avg_fwd,
                    "bwd_s": avg_bwd,
                    "opt_s": avg_opt,
                    "world_size": world_size,
                    "batch_size": config.batch_size,
                    "grad_accum": grad_accum,
                    "dist_strategy": dist_strategy,
                    "mem_alloc_mb": mem_alloc,
                    "mem_reserved_mb": mem_reserved,
                    # WARNING: this is a 1-element AR latency floor (the test_tensor
                    # probe ~30 lines above), NOT the per-step gradient AR cost. The
                    # real per-step AR for E2E (multi-GB trainable) is hidden inside
                    # bwd_s — `loss.backward()` blocks on the DDP reducer's collective
                    # tail. To measure it, capture a kineto trace and sum c10d::allreduce_
                    # CPU-op time per step (typically 1000x larger than this column).
                    "allreduce_latency_ms": timing_allreduce * 1000,
                    "sweep_id": getattr(config, "sweep_id", None),
                    "preset": getattr(config, "preset", None),
                    "modalities": model_modalities(model),
                    "seq_p50": _seq_stats["seq_p50"],
                    "seq_p95": _seq_stats["seq_p95"],
                    "seq_p99": _seq_stats["seq_p99"],
                    "seq_max": _seq_stats["seq_max"],
                    "padding_ratio": _seq_stats["padding_ratio"],
                    "flops_per_step": _fps_window,
                    "cumulative_flops": _cum_flops,
                    "projector_hidden_mult": (
                        getattr(_unwrapped_cfg, "projector_hidden_mult", 1)
                        if _unwrapped_cfg is not None else 1
                    ),
                    "projector_num_layers": (
                        getattr(_unwrapped_cfg, "projector_num_layers", 2)
                        if _unwrapped_cfg is not None else 2
                    ),
                },
            )

            # Detailed WandB Log (Performance)
            if wandb.run:
                wandb.log(
                    {
                        "perf/iter_per_sec": 1.0 / total if total > 0 else 0,
                        "perf/samples_per_sec": samples_per_sec,
                        "perf/time_per_step_s": total,
                        "perf/data_time_s": avg_data,
                        "perf/forward_time_s": avg_fwd,
                        "perf/backward_time_s": avg_bwd,
                        "perf/optimizer_time_s": avg_opt,
                        "perf/mem_allocated_mb": mem_alloc,
                        "perf/mem_reserved_mb": mem_reserved,
                        "perf/allreduce_latency_ms": timing_allreduce * 1000,
                        "perf/clip_time_ms": clip_time * 1000,
                    },
                    step=step,
                )

            timing_data = timing_fwd = timing_bwd = timing_opt = 0.0
            timing_steps = 0
            tokens_per_window = 0
            seq_lens_window = []

        # --- Checkpoint Saving ---
        save_every = getattr(config, "save_every_n_steps", 0)
        if save_every > 0 and step % save_every == 0:
            # All ranks must barrier before save to ensure consistent state
            if world_size > 1:
                dist.barrier()
            save_native_ddp_checkpoint(model, optimizer, scheduler, step, config, rank)

    # Save final checkpoint (skip if already saved at this step)
    final_save_every = getattr(config, "save_every_n_steps", 0)
    if final_save_every > 0 and step > 0 and step % final_save_every != 0:
        if world_size > 1:
            dist.barrier()
        save_native_ddp_checkpoint(model, optimizer, scheduler, step, config, rank)

    if is_main:
        logger.info(f"[Native DDP] Complete: {step} steps")
        if wandb.run:
            wandb.finish()
