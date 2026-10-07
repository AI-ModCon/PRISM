"""Distributed training setup and model wrapping for PRISM.

Handles DDP, FSDP, and HSDP wrapping, distributed process group initialization,
checkpoint saving, and weight-only resume for stage transitions.
"""

import datetime
import importlib
import json
import os
import time
from functools import partial

import torch
import torch.distributed as dist

from src.decoders import remap_legacy_decoder_keys


def _select_backend() -> str:
    """Pick the distributed backend matching the available accelerator.

    Aurora (XPU) -> xccl, NVIDIA (CUDA) -> nccl, otherwise gloo. The DIST_BACKEND
    env var overrides the auto-detect — set it when running on a system where
    the default would be wrong (e.g. forcing gloo for CPU debugging).
    """
    override = os.environ.get("DIST_BACKEND", "").strip()
    if override:
        return override
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xccl"
    if torch.cuda.is_available():
        return "nccl"
    return "gloo"

# Maps HuggingFace model_type -> (module_path, class_name) for the decoder
# layer used by transformer_auto_wrap_policy in FSDP/HSDP.
DECODER_LAYER_MAP = {
    "olmo": ("transformers.models.olmo.modeling_olmo", "OlmoDecoderLayer"),
    "olmo3": ("transformers.models.olmo3.modeling_olmo3", "Olmo3DecoderLayer"),
    "llama": ("transformers.models.llama.modeling_llama", "LlamaDecoderLayer"),
    "gemma": ("transformers.models.gemma.modeling_gemma", "GemmaDecoderLayer"),
    "gemma2": ("transformers.models.gemma2.modeling_gemma2", "Gemma2DecoderLayer"),
    "mistral": ("transformers.models.mistral.modeling_mistral", "MistralDecoderLayer"),
    "phi3": ("transformers.models.phi3.modeling_phi3", "Phi3DecoderLayer"),
    "qwen2": ("transformers.models.qwen2.modeling_qwen2", "Qwen2DecoderLayer"),
    "qwen3": ("transformers.models.qwen3.modeling_qwen3", "Qwen3DecoderLayer"),
    "granite": ("transformers.models.granite.modeling_granite", "GraniteDecoderLayer"),
}


def _get_decoder_layer_cls(model, is_main):
    """Resolve the decoder layer class for FSDP/HSDP transformer_auto_wrap_policy.

    Looks up the backbone's model_type in DECODER_LAYER_MAP, falling back to
    scanning model modules for a class ending in 'DecoderLayer'.

    Returns:
        The decoder layer class, or None if not found.
    """
    if not (hasattr(model, "backbone") and model.backbone is not None):
        return None

    model_type = getattr(model.backbone.config, "model_type", None)

    if model_type in DECODER_LAYER_MAP:
        module_path, cls_name = DECODER_LAYER_MAP[model_type]
        mod = importlib.import_module(module_path)
        return getattr(mod, cls_name)

    # Fallback: scan model for DecoderLayer
    for module in model.backbone.modules():
        cls = type(module)
        if cls.__name__.endswith("DecoderLayer"):
            return cls

    return None


# --- Multi-Node Distributed Setup ---


def setup_distributed():
    """Initialize distributed training.

    Selects env-only init when EITHER:
      - USE_NATIVE_DDP=1 (set by native-DDP launch scripts), OR
      - DEEPSPEED_ZERO_STAGE is set to a non-empty value (DeepSpeed via
        Accelerate must avoid mpi4py to prevent the MPI re-init crash
        "Fatal error in internal_Init_thread" when DeepSpeed subsequently
        re-initializes torch.distributed).

    Both modes avoid mpi4py, which can conflict with the XCCL backend on Aurora.
    Otherwise falls back to mpi4py for MASTER_ADDR broadcast.
    """
    use_env_only = (
        os.environ.get("USE_NATIVE_DDP", "0") == "1"
        or os.environ.get("DEEPSPEED_ZERO_STAGE", "").strip() != ""
    )

    if use_env_only:
        return _setup_distributed_env_only()
    else:
        return _setup_distributed_mpi4py()


def _setup_distributed_env_only():
    """Initialize distributed using environment variables only (no mpi4py).

    This is required when running inside mpiexec with XCCL backend to avoid
    MPI re-initialization errors ("Fatal error in internal_Init_thread").
    """

    # Helper to get int from env, handling empty strings
    def get_env_int(keys, default):
        for key in keys:
            val = os.environ.get(key, "")
            if val and val.strip():
                try:
                    return int(val)
                except ValueError:
                    continue
        return default

    # Get rank info from environment (set by mpiexec wrapper script).
    # PRISM_REAL_LOCAL_RANK takes priority: it's set when LOCAL_RANK was forced to 0
    # for DeepSpeed (because ZE_AFFINITY_MASK exposes one tile per rank, so DeepSpeed
    # must see LOCAL_RANK=0). The original local rank value is stashed in
    # PRISM_REAL_LOCAL_RANK so device-binding code below sees the correct tile index.
    rank = get_env_int(["RANK", "PMI_RANK", "PALS_RANKID"], 0)
    world_size = get_env_int(
        ["WORLD_SIZE", "PMI_SIZE", "PALS_SIZE", "PALS_LOCAL_SIZE"], 1
    )
    local_rank = get_env_int(
        ["PRISM_REAL_LOCAL_RANK", "LOCAL_RANK", "PMI_LOCAL_RANK", "PALS_LOCAL_RANKID"], 0
    )

    # Set device before init_process_group.
    # When ZE_AFFINITY_MASK (Aurora) or single-GPU CUDA_VISIBLE_DEVICES
    # (Polaris) is set, each rank sees only its own GPU as device 0 — calling
    # set_device(local_rank) would raise "invalid device ordinal".
    device = "cpu"
    use_affinity_mask = "ZE_AFFINITY_MASK" in os.environ
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    cuda_pinned = bool(cvd) and "," not in cvd
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        if use_affinity_mask:
            device = "xpu:0"
        else:
            torch.xpu.set_device(local_rank)
            device = f"xpu:{local_rank}"
    elif torch.cuda.is_available():
        if cuda_pinned:
            torch.cuda.set_device(0)
            device = "cuda:0"
        else:
            torch.cuda.set_device(local_rank)
            device = f"cuda:{local_rank}"

    if world_size > 1 and not dist.is_initialized():
        # Use MASTER_ADDR from environment (set by job script)
        master_addr = os.environ.get("MASTER_ADDR", "localhost")
        master_port = os.environ.get("MASTER_PORT", "29500")
        init_method = f"tcp://{master_addr}:{master_port}"

        backend = _select_backend()

        if rank == 0:
            print(
                f"[Distributed] Env-only mode: backend={backend}, world_size={world_size}, "
                f"init_method={init_method}"
            )

        # device_id rules diverge per backend:
        #   XPU (xccl): MUST omit. Passing it makes xccl eagerly initialize GPU
        #     contexts/streams that fork into DataLoader workers and deadlock on
        #     the first batch read.
        #   CUDA (nccl): SHOULD pass when CUDA_VISIBLE_DEVICES pins one GPU per
        #     rank. Without it NCCL can't infer the rank↔GPU mapping and warns
        #     "using GPU 0 as device used by this process is currently unknown
        #     ... can potentially cause a hang."
        init_kwargs: dict = dict(
            backend=backend,
            init_method=init_method,
            world_size=world_size,
            rank=rank,
            timeout=datetime.timedelta(minutes=10),
        )
        if backend == "nccl" and device.startswith("cuda"):
            init_kwargs["device_id"] = torch.device(device)
        dist.init_process_group(**init_kwargs)

    return rank, world_size, local_rank, device


def _setup_distributed_mpi4py():
    """Initialize distributed training using mpi4py for reliable MASTER_ADDR broadcast."""
    try:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        world_size = comm.Get_size()

        # Get local rank from environment. PRISM_REAL_LOCAL_RANK is checked
        # first for symmetry with the env-only path: if a wrapper forced
        # LOCAL_RANK=0 (e.g. for DeepSpeed under ZE_AFFINITY_MASK), the original
        # local rank is stashed there so device-binding sees the real tile.
        local_rank_vars = [
            "PRISM_REAL_LOCAL_RANK",
            "PALS_LOCAL_RANKID",
            "OMPI_COMM_WORLD_LOCAL_RANK",
            "MPI_LOCALRANKID",
            "LOCAL_RANK",
            "PMI_LOCAL_RANK",
        ]
        local_rank = 0
        for var in local_rank_vars:
            if var in os.environ:
                local_rank = int(os.environ[var])
                break

        # Set device before init_process_group.
        # See env-only branch for cuda_pinned rationale.
        device = "cpu"
        use_affinity_mask = "ZE_AFFINITY_MASK" in os.environ
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        cuda_pinned = bool(cvd) and "," not in cvd
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            if use_affinity_mask:
                device = "xpu:0"
            else:
                torch.xpu.set_device(local_rank)
                device = f"xpu:{local_rank}"
        elif torch.cuda.is_available():
            if cuda_pinned:
                torch.cuda.set_device(0)
                device = "cuda:0"
            else:
                torch.cuda.set_device(local_rank)
                device = f"cuda:{local_rank}"

        if world_size > 1 and not dist.is_initialized():
            # Broadcast MASTER_ADDR from rank 0 (more reliable than env vars)
            if rank == 0:
                master_addr = os.uname()[1]  # hostname
            else:
                master_addr = None
            master_addr = comm.bcast(master_addr, root=0)

            # On Aurora prefer the HSN/Slingshot suffix for inter-node init;
            # other systems (Polaris, dev workstations) use the plain
            # hostname or whatever MASTER_ADDR the launcher already set.
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                master_addr_init = f"{master_addr}.hsn.cm.aurora.alcf.anl.gov"
            else:
                master_addr_init = master_addr

            master_port = os.environ.get("MASTER_PORT", "29500")
            init_method = f"tcp://{master_addr_init}:{master_port}"

            backend = _select_backend()

            if rank == 0:
                print(
                    f"[Distributed] Initializing: backend={backend}, world_size={world_size}, "
                    f"init_method={init_method}"
                )

            # device_id rules — see env-only branch.
            init_kwargs: dict = dict(
                backend=backend,
                init_method=init_method,
                world_size=world_size,
                rank=rank,
                timeout=datetime.timedelta(minutes=10),
            )
            if backend == "nccl" and device.startswith("cuda"):
                init_kwargs["device_id"] = torch.device(device)
            dist.init_process_group(**init_kwargs)

        return rank, world_size, local_rank, device

    except ImportError:
        print("mpi4py not available. Running in single-process mode.")
        return (
            0,
            1,
            0,
            "xpu:0" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu",
        )
    except Exception as e:
        print(f"Distributed setup failed: {e}. Running in single-process mode.")
        return (
            0,
            1,
            0,
            "xpu:0" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu",
        )


# --- Model Wrapping ---


def wrap_model_distributed(model, config, rank, world_size, local_rank, device):
    """Wrap model in DDP or FSDP based on configuration.

    Returns:
        Wrapped model ready for distributed training.
    """
    is_main = rank == 0

    # Check distribution strategy from environment or config
    dist_strategy = os.environ.get("DIST_STRATEGY", "ddp").lower()

    # Move model to device first. Do device + dtype in a single traversal on XPU:
    # the old two-step path (CPU->XPU, then XPU dtype conversion) can leave large
    # Qwen models waiting in the Intel GPU driver for minutes before HSDP wraps.
    target_dtype = (
        torch.bfloat16
        if hasattr(torch, "xpu") and str(device).startswith("xpu")
        else None
    )
    sync_after_move = os.environ.get("PRISM_SYNC_AFTER_MODEL_TO", "0") == "1"
    if is_main:
        dtype_msg = f", dtype={target_dtype}" if target_dtype is not None else ""
        sync_msg = ", sync" if sync_after_move else ", async"
        print(
            f"[Distributed] Moving model to {device}{dtype_msg}{sync_msg} "
            f"before {dist_strategy} wrap...",
            flush=True,
        )
    _t_move = time.time()
    if target_dtype is not None:
        model = model.to(device=device, dtype=target_dtype)
        if sync_after_move:
            torch.xpu.synchronize()
    else:
        model = model.to(device)
        if sync_after_move and str(device).startswith("cuda"):
            torch.cuda.synchronize()
    if is_main:
        msg = f"[Distributed] Model device move complete in {time.time() - _t_move:.1f}s"
        if hasattr(torch, "xpu") and str(device).startswith("xpu"):
            try:
                alloc_gb = torch.xpu.memory_allocated() / 1e9
                reserved_gb = torch.xpu.memory_reserved() / 1e9
                msg += f" ({alloc_gb:.2f}GB allocated, {reserved_gb:.2f}GB reserved)"
            except Exception:
                pass
        print(msg, flush=True)

    # torch.compile: fuse ops and reduce kernel launch overhead.
    # Set TORCH_COMPILE=1 to enable. Applied BEFORE DDP/FSDP wrapping.
    # Compiles only the HF backbone (where >90% of compute lives), avoiding graph
    # breaks from multimodal routing code (random sampling, logger calls, try/except).
    # Verified working on Aurora frameworks 25.190.0 (Triton 3.4.0, PyTorch 2.8.0).
    if os.environ.get("TORCH_COMPILE", "0") == "1":
        compile_backend = os.environ.get("TORCH_COMPILE_BACKEND", "inductor")

        # NOTE: Do NOT use `import torch._dynamo` here — it creates a local binding
        # for `torch` that shadows the global import and causes UnboundLocalError.
        use_dynamic = os.environ.get("TORCH_COMPILE_DYNAMIC", "1") == "1"

        # Multi-node compile deadlock prevention:
        # When GRAD_CKPT_FREQ > 1, alternating layers have different
        # gradient_checkpointing values (True/False). Dynamo treats this as a guard
        # and recompiles for each variant. If ranks hit recompile_limit at different
        # times, some fall back to eager while others stay compiled, causing collective
        # desync and deadlock. Fix: force ALL layers to checkpoint when compiling
        # for multi-node (trades ~5% compute for stability).
        if world_size > 1 and hasattr(model, "backbone"):
            inner = getattr(model.backbone, "model", model.backbone)
            layers = getattr(inner, "layers", None)
            if layers is not None:
                n_fixed = 0
                for layer in layers:
                    if hasattr(layer, "gradient_checkpointing"):
                        if not layer.gradient_checkpointing:
                            layer.gradient_checkpointing = True
                            n_fixed += 1
                if n_fixed > 0 and is_main:
                    print(
                        f"[torch.compile] Normalized gradient_checkpointing: "
                        f"forced {n_fixed} layers True (prevents recompilation deadlock)"
                    )

        if is_main:
            print(f"[torch.compile] Compiling with backend={compile_backend}...")
            print(f"[torch.compile] dynamic={use_dynamic}")
            try:
                import triton
                print(f"[torch.compile] Triton {triton.__version__} at {triton.__file__}")
            except ImportError as e:
                print(f"[torch.compile] WARNING: Triton not importable: {e}")

        if hasattr(model, "backbone") and model.backbone is not None:
            # Compile only the backbone — avoids graph breaks in UnifiedTransformer
            # forward (random.random(), logger calls, try/except, .item() debug stats).
            # The backbone (OLMo/AuroraGPT) has 0 graph breaks and is the dominant
            # compute cost (attention + MLP matmuls across all layers).
            model.backbone = torch.compile(
                model.backbone, backend=compile_backend, dynamic=use_dynamic
            )
            if is_main:
                print(
                    f"[torch.compile] Backbone compiled with dynamic={use_dynamic} "
                    "(first forward will trigger Triton kernel compilation ~90s)"
                )
        else:
            # No backbone (custom transformer path) — compile full model
            model = torch.compile(model, backend=compile_backend, dynamic=use_dynamic)
            if is_main:
                print(f"[torch.compile] Full model compiled with dynamic={use_dynamic}")

    if world_size <= 1:
        if is_main:
            print("[Distributed] Single GPU mode - no wrapping needed")
        return model

    # Device ID handling: when each rank only sees one device (Aurora
    # ZE_AFFINITY_MASK or Polaris CUDA_VISIBLE_DEVICES=$LOCAL_RANK pin),
    # the visible device is index 0 from torch's view. Passing
    # device_ids=[local_rank] then trips DDP's `_streams[device.index]`
    # lookup (IndexError: list index out of range) because that list only
    # has one entry.
    use_affinity_mask = "ZE_AFFINITY_MASK" in os.environ
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    cuda_pinned = bool(cvd) and "," not in cvd
    ddp_device_id = 0 if (use_affinity_mask or cuda_pinned) else local_rank

    if dist_strategy == "fsdp":
        return _wrap_fsdp(model, config, rank, world_size, local_rank, device, is_main)
    elif dist_strategy == "hsdp":
        return _wrap_hsdp(model, config, rank, world_size, local_rank, device, is_main)
    else:
        # Default: DDP
        return _wrap_ddp(model, config, ddp_device_id, is_main)


def _wrap_fsdp(model, config, rank, world_size, local_rank, device, is_main):
    """Wrap model in FSDP for better multi-node scaling with large models.

    FSDP shards model parameters, gradients, and optimizer states across ranks,
    reducing memory per GPU and communication volume for gradient sync.

    Expected improvement for 7B: 63% -> 80%+ scaling efficiency
    """
    try:
        from torch.distributed.fsdp import CPUOffload, MixedPrecision, ShardingStrategy
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp.wrap import (
            size_based_auto_wrap_policy,
            transformer_auto_wrap_policy,
        )
    except ImportError as e:
        if is_main:
            print(f"[FSDP] Import error: {e}. Falling back to DDP.")
        return _wrap_ddp(model, config, 0, is_main)

    if is_main:
        print("[FSDP] Initializing Fully Sharded Data Parallel...")

    # Mixed precision config for communication efficiency
    # BF16 reduces gradient communication by 2x
    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,  # Gradient AllReduce in BF16
        buffer_dtype=torch.bfloat16,
    )

    # CPU offload (optional - saves GPU memory but slower)
    cpu_offload = None
    if os.environ.get("FSDP_CPU_OFFLOAD", "0") == "1":
        cpu_offload = CPUOffload(offload_params=True)
        if is_main:
            print("[FSDP] CPU offload enabled")

    # Sharding strategy
    # FULL_SHARD: Best memory efficiency, highest communication
    # SHARD_GRAD_OP: Shard gradients and optimizer states only (faster)
    # HYBRID_SHARD: FSDP within node, replicate across nodes
    #
    # NOTE: this default ("full_shard") is intentionally different from the HSDP
    # path's default ("shard_grad_op", see _wrap_hsdp below). FSDP is selected
    # when a single shard of params must fit per rank (large models, e.g. 7B+);
    # FULL_SHARD is required to free params between layers. HSDP is selected
    # when params fit per node, so SHARD_GRAD_OP (ZERO2) is the right intra-node
    # default. Both reuse the same env-var name because they're mutually
    # exclusive (only one is wrapping the model in any given run).
    shard_strategy_env = os.environ.get("FSDP_SHARDING", "full_shard").lower()
    shard_map = {
        "full_shard": ShardingStrategy.FULL_SHARD,
        "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
        "no_shard": ShardingStrategy.NO_SHARD,
        "hybrid_shard": ShardingStrategy.HYBRID_SHARD,
    }
    sharding_strategy = shard_map.get(shard_strategy_env, ShardingStrategy.FULL_SHARD)

    # Wrapping strategy:
    # For SHARD_GRAD_OP we use top-level-only wrapping (no auto_wrap_policy).
    # Per-module wrapping causes catastrophic overhead on XPU because each
    # wrapped module does independent communication ops. With top-level-only,
    # FSDP does a single ReduceScatter for gradients (like DDP AllReduce) and
    # shards optimizer states across ranks without per-forward-pass overhead.
    #
    # For FULL_SHARD, per-module wrapping is required for memory efficiency
    # (each module AllGathers only its own params), but this is too slow on XPU.
    if shard_strategy_env == "shard_grad_op":
        wrap_policy = None  # Top-level only
        if is_main:
            print("[FSDP] Using top-level-only wrapping (no auto_wrap_policy)")
            print("[FSDP] Optimizer states sharded across ranks, no AllGather overhead")
    else:
        # Per-module wrapping for FULL_SHARD (not recommended on XPU)
        wrap_policy = partial(
            size_based_auto_wrap_policy,
            min_num_params=1_000_000,
        )

        # Try transformer-specific wrapping for better granularity
        try:
            decoder_layer_cls = _get_decoder_layer_cls(model, is_main)
            if decoder_layer_cls is not None:
                wrap_policy = partial(
                    transformer_auto_wrap_policy,
                    transformer_layer_cls={decoder_layer_cls},
                )
                if is_main:
                    print(
                        f"[FSDP] Using transformer_auto_wrap_policy for {decoder_layer_cls.__name__}"
                    )
        except (ImportError, AttributeError) as e:
            if is_main:
                print(f"[FSDP] Could not get transformer layer class: {e}")

    # Wrap model in FSDP
    #
    # Prefetch settings:
    #   backward_prefetch=BACKWARD_PRE: AllGather next FSDP unit's params during
    #     current unit's backward pass, hiding communication latency behind compute.
    #   forward_prefetch=True: AllGather next unit's params during current unit's
    #     forward pass. Smaller benefit than backward prefetch but essentially free.
    #
    # These are controlled by env vars for A/B testing:
    #   FSDP_BACKWARD_PREFETCH=1 (default on)
    #   FSDP_FORWARD_PREFETCH=1 (default on)
    from torch.distributed.fsdp import BackwardPrefetch

    fsdp_kwargs = dict(
        mixed_precision=mp_policy,
        sharding_strategy=sharding_strategy,
        cpu_offload=cpu_offload,
        device_id=torch.device(device),
        use_orig_params=True,  # Required for named param access in optimizer
        limit_all_gathers=True,
    )

    # Prefetch: hide AllGather latency behind compute
    if os.environ.get("FSDP_BACKWARD_PREFETCH", "1") == "1":
        fsdp_kwargs["backward_prefetch"] = BackwardPrefetch.BACKWARD_PRE
        if is_main:
            print("[FSDP] backward_prefetch=BACKWARD_PRE (hide AllGather in backward)")
    else:
        if is_main:
            print("[FSDP] backward_prefetch DISABLED")

    if os.environ.get("FSDP_FORWARD_PREFETCH", "1") == "1":
        fsdp_kwargs["forward_prefetch"] = True
        if is_main:
            print("[FSDP] forward_prefetch=True (hide AllGather in forward)")
    else:
        if is_main:
            print("[FSDP] forward_prefetch DISABLED")
    if wrap_policy is not None:
        fsdp_kwargs["auto_wrap_policy"] = wrap_policy

    model = FSDP(model, **fsdp_kwargs)

    if is_main:
        print(f"[FSDP] Model wrapped with strategy={shard_strategy_env}")
        # With FSDP, model.parameters() returns sharded params (1/world_size).
        # Multiply by world_size to get the true total.
        local_params = sum(p.numel() for p in model.parameters())
        total_params = local_params * world_size
        print(f"[FSDP] Local shard parameters: {local_params:,} (1/{world_size})")
        print(f"[FSDP] Total parameters (all shards): {total_params:,}")
        # Log parameter names for debugging param group construction
        param_names = [n for n, p in model.named_parameters() if p.requires_grad]
        if param_names:
            print(f"[FSDP] Total named params: {len(param_names)}")
            print(f"[FSDP] Sample param names (first 5): {param_names[:5]}")
            # Categorize params for debugging E2E differential LR matching
            enc_names = [n for n in param_names if "encoders" in n]
            proj_names = [n for n in param_names if "projectors" in n]
            bb_names = [n for n in param_names if "backbone" in n]
            other_names = [
                n
                for n in param_names
                if "encoders" not in n and "projectors" not in n and "backbone" not in n
            ]
            print(
                f"[FSDP] Param categories: encoder={len(enc_names)}, projector={len(proj_names)}, backbone={len(bb_names)}, other={len(other_names)}"
            )
            if enc_names:
                print(f"[FSDP] Encoder params (first 3): {enc_names[:3]}")
            if proj_names:
                print(f"[FSDP] Projector params (first 3): {proj_names[:3]}")
            if other_names:
                print(f"[FSDP] Other params (first 3): {other_names[:3]}")
        # Memory after wrapping
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            alloc_gb = torch.xpu.memory_allocated() / 1e9
            reserved_gb = torch.xpu.memory_reserved() / 1e9
            print(
                f"[FSDP] XPU memory after wrap: {alloc_gb:.2f}GB allocated, {reserved_gb:.2f}GB reserved"
            )

    # Note: empty_cache() intentionally NOT called here. Calling torch.xpu.empty_cache()
    # after FSDP wrap leaks Level Zero UR handles via the zeMemAllocDevice/zeMemFree cycle
    # triggered by FSDP's storage.resize_(). With 13+ FSDP units this crashes training
    # after ~70 iterations. The caching allocator reuses freed blocks without touching
    # Level Zero, so omitting empty_cache() is safe. (torchtune Apr 2026)

    return model


def _wrap_hsdp(model, config, rank, world_size, local_rank, device, is_main):
    """Wrap model in Hybrid Sharded Data Parallel.

    HSDP = FSDP within node + DDP across nodes.
    Best for multi-node training where intra-node bandwidth >> inter-node bandwidth.

    On Aurora: Intra-node tiles share high-BW fabric, inter-node = Slingshot 11.

    Strategy choices:
      --fsdp-sharding shard_grad_op (RECOMMENDED for HSDP):
        _HYBRID_SHARD_ZERO2: shard gradients+optimizer intra-node, DDP inter-node.
        Params stay gathered after forward -> no re-AllGather in backward.
        Uses more memory (~2x shard_grad vs full_shard) but Aurora has 20+ GB headroom.
        Backward = ReduceScatter(intra) + AllReduce(inter) only.

      --fsdp-sharding full_shard:
        HYBRID_SHARD: full_shard intra-node, DDP inter-node.
        Params are freed after forward -> re-AllGathered in backward.
        Backward = AllGather(intra) + ReduceScatter(intra) + AllReduce(inter).
        More communication but lower memory. Use only if HBM-constrained.
    """
    try:
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
        from torch.distributed.fsdp.wrap import (
            size_based_auto_wrap_policy,
            transformer_auto_wrap_policy,
        )
    except ImportError as e:
        if is_main:
            print(f"[HSDP] Import error: {e}. Falling back to DDP.")
        return _wrap_ddp(model, config, 0, is_main)

    if is_main:
        print("[HSDP] Initializing Hybrid Sharded Data Parallel...", flush=True)

    # Calculate mesh dimensions
    # local_world_size = ranks per node (typically 12 on Aurora)
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "12"))
    num_nodes = world_size // local_world_size

    if is_main:
        print(
            f"[HSDP] Topology: {num_nodes} nodes x {local_world_size} ranks/node",
            flush=True,
        )

    # Initialize 2D device mesh: (num_nodes, local_world_size)
    # Dimension 0 = "replicate" (DDP across nodes)
    # Dimension 1 = "shard" (FSDP within node)
    device_type = "xpu" if hasattr(torch, "xpu") else "cuda"
    try:
        mesh = init_device_mesh(
            device_type,
            (num_nodes, local_world_size),
            mesh_dim_names=("replicate", "shard"),
        )
    except Exception as e:
        if is_main:
            print(f"[HSDP] Failed to create device mesh: {e}. Falling back to FSDP.")
        return _wrap_fsdp(model, config, rank, world_size, local_rank, device, is_main)

    # Mixed precision
    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    # Sharding strategy selection:
    # HYBRID_SHARD = FULL_SHARD intra-node + DDP inter-node
    # _HYBRID_SHARD_ZERO2 = SHARD_GRAD_OP intra-node + DDP inter-node
    fsdp_sharding_env = os.environ.get("FSDP_SHARDING", "shard_grad_op").lower()
    if fsdp_sharding_env == "shard_grad_op":
        try:
            sharding = ShardingStrategy._HYBRID_SHARD_ZERO2
            shard_label = "_HYBRID_SHARD_ZERO2 (shard_grad_op intra-node)"
        except AttributeError:
            sharding = ShardingStrategy.HYBRID_SHARD
            shard_label = "HYBRID_SHARD (full_shard intra-node, fallback)"
            if is_main:
                print(
                    "[HSDP] WARNING: _HYBRID_SHARD_ZERO2 not available, "
                    "using HYBRID_SHARD (slower due to AllGather)"
                )
    else:
        sharding = ShardingStrategy.HYBRID_SHARD
        shard_label = "HYBRID_SHARD (full_shard intra-node)"

    # Wrapping policy — use transformer_auto_wrap_policy for per-decoder-layer
    # granularity (consistent with the FSDP path). size_based_auto_wrap_policy(1M)
    # creates too many small FSDP units, each triggering separate collectives.
    # HSDP_FORCE_TRANSFORMER_WRAP=1 forces transformer_auto_wrap_policy even for
    # shard_grad_op. This makes each decoder layer its own FSDP unit, so the
    # inter-node grad ReduceScatter on bwd is per-layer (overlappable with the
    # next layer's bwd compute) instead of one giant collective at the end.
    # Combined with backward_prefetch (default BACKWARD_PRE) this materially
    # cuts the visible per-step comm overhead at high node counts.
    _hsdp_force_xfmr_wrap = os.environ.get("HSDP_FORCE_TRANSFORMER_WRAP", "0") == "1"
    wrap_policy = None
    if fsdp_sharding_env != "shard_grad_op" or _hsdp_force_xfmr_wrap:
        # Per-module wrapping for full_shard (needed to free params between layers)
        decoder_layer_cls = None
        try:
            decoder_layer_cls = _get_decoder_layer_cls(model, is_main)
        except Exception as e:
            if is_main:
                print(f"[HSDP] Could not detect decoder layer class: {e}")

        if decoder_layer_cls is not None:
            wrap_policy = partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={decoder_layer_cls},
            )
            if is_main:
                print(
                    f"[HSDP] Using transformer_auto_wrap_policy for {decoder_layer_cls.__name__}"
                )
        else:
            # Fallback to size-based
            wrap_policy = partial(
                size_based_auto_wrap_policy,
                min_num_params=1_000_000,
            )
            if is_main:
                print("[HSDP] Using size_based_auto_wrap_policy (fallback)")
    elif is_main:
        print("[HSDP] Using top-level-only wrapping (shard_grad_op)", flush=True)

    hsdp_kwargs = dict(
        mixed_precision=mp_policy,
        sharding_strategy=sharding,
        device_mesh=mesh,
        use_orig_params=True,
        limit_all_gathers=True,
    )
    if wrap_policy is not None:
        hsdp_kwargs["auto_wrap_policy"] = wrap_policy

    # Prefetch overlap (mirrors the FSDP path; previously omitted on HSDP).
    # Per-unit wrapping is required for either flag to have effect.
    if wrap_policy is not None:
        from torch.distributed.fsdp import BackwardPrefetch

        if os.environ.get("HSDP_BACKWARD_PREFETCH", "1") == "1":
            hsdp_kwargs["backward_prefetch"] = BackwardPrefetch.BACKWARD_PRE
            if is_main:
                print("[HSDP] backward_prefetch=BACKWARD_PRE")
        if os.environ.get("HSDP_FORWARD_PREFETCH", "1") == "1":
            hsdp_kwargs["forward_prefetch"] = True
            if is_main:
                print("[HSDP] forward_prefetch=True")

    if is_main:
        print("[HSDP] Constructing FSDP wrapper...", flush=True)
    _t_hsdp_wrap = time.time()
    model = FSDP(model, **hsdp_kwargs)

    if is_main:
        print(
            f"[HSDP] Model wrapped: {shard_label}, "
            f"{local_world_size} ranks/node x {num_nodes} nodes "
            f"({time.time() - _t_hsdp_wrap:.1f}s)"
        )

    return model


# Trainable-parameter threshold (MB) separating projector-only from E2E
# training regimes. Used both for DDP bucket sizing in `_wrap_ddp` and for
# the multi-modality projector auto-detect in `_resolve_find_unused`.
_PROJECTOR_ONLY_TRAINABLE_MB_THRESHOLD = 1000


def _resolve_find_unused(
    *,
    trainable_mb: float,
    num_modalities: int,
    is_composite: bool,
    is_interleaved: bool,
    env_override: str | None,
) -> tuple[bool, bool, bool, bool]:
    """Decide whether DDP needs `find_unused_parameters=True`.

    Three auto-detect signals; an env override wins over all:
    - COMPOSITE mode (XCCL allreduce hangs with static_graph=True)
    - Multi-modality projector-only (heterogeneous batches → empty bucket crash)
    - Interleaved-QA E2E training (per-batch modality routing → graph changes
      across microbatches; only surfaces at scale where sampling variance is
      large enough that some ranks see one modality first and others another)

    Returns: (find_unused, is_multimodal_projector_auto, is_composite_auto,
    is_interleaved_auto). The three diagnostic booleans report which
    auto-detect signals were active, independent of whether an env override
    forced a different final value. Callers use them to attribute the
    decision in logs.
    """
    is_multimodal_projector = (
        trainable_mb < _PROJECTOR_ONLY_TRAINABLE_MB_THRESHOLD and num_modalities > 1
    )
    default_find_unused = is_composite or is_multimodal_projector or is_interleaved
    if env_override is not None:
        find_unused = env_override == "1"
    else:
        find_unused = default_find_unused
    return find_unused, is_multimodal_projector, is_composite, is_interleaved


def _wrap_ddp(model, config, ddp_device_id, is_main):
    """Wrap model in standard DDP with optimizations.

    Handles two training regimes:
    - Projector-only (~12MB trainable): Single bucket, static_graph=True
    - E2E (7B+, ~14GB trainable): Standard bucket sizing (25-50MB)

    FLAT mode (12 tiles/node, 64GB each):
        static_graph=True, find_unused_parameters=False — optimal for throughput,
        UNLESS the multi-modality projector auto-detect (below) fires.

    COMPOSITE mode (6 GPUs/node, 128GB each):
        find_unused_parameters=True, static_graph=False — required for reliability.
        static_graph=True causes non-deterministic hangs in XCCL allreduce during
        backward pass (observed as flaky deadlocks: succeeds on some runs, hangs on
        others at varying micro-batch indices). Throughput benchmarks show zero
        overhead from find_unused_parameters — both configs achieve identical
        steady-state step times (~11.4s/step for OLMo-3-7B with grad_accum=6).

    Multi-modality projector-only auto-detect:
        When trainable_mb < 1GB (projector-only regime) AND
        len(model.config.modalities) > 1, find_unused_parameters=True is forced.
        Heterogeneous batches (some ranks see graph, others don't) leave per-modality
        projectors with no gradient flow on some ranks, and DDP's bucket rebuild
        then crashes at the first forward with `RuntimeError: Empty bucket specified`.
        Documented in docs/platforms/aurora_operations.md.

    Interleaved-QA auto-detect:
        When model.config.is_interleaved_qa=True, find_unused_parameters=True is
        forced. Interleaved training routes each batch through one of several
        modality-specific sub-branches (image encoder / TS encoder / text-only).
        Per-rank sampling picks different modalities on different microbatches,
        so DDP sees "param X unused in step 0, used in step 1" — which
        `static_graph=True` forbids and crashes at first backward. Only
        surfaces at multi-node scale where rank-sampling variance is large
        enough to trigger the graph-change check (1N is usually safe;
        4N+ is deterministic). See docs/results/patrick_tsqa_ab.md.

    Frozen parameters are excluded from DDP via _ddp_params_and_buffers_to_ignore
    to prevent empty bucket errors during bucket rebuild.

    Env var overrides:
        PRISM_DDP_FIND_UNUSED: "0" or "1" — override find_unused_parameters
            (wins over all three auto-detect signals)
        PRISM_DDP_GRAD_BUCKET_VIEW: "0" or "1" — override gradient_as_bucket_view
        DDP_BUCKET_CAP_MB: bucket size in MB (default 25)
    """
    from torch.nn.parallel import DistributedDataParallel as DDP

    # Exclude frozen parameters from DDP gradient buckets.
    frozen_params = [
        name for name, param in model.named_parameters() if not param.requires_grad
    ]
    model._ddp_params_and_buffers_to_ignore = frozen_params

    trainable_bytes = sum(
        p.numel() * p.element_size() for p in model.parameters() if p.requires_grad
    )
    trainable_mb = trainable_bytes / (1024 * 1024)
    default_bucket_mb = int(os.environ.get("DDP_BUCKET_CAP_MB", "25"))
    # E2E training (>1GB trainable): use standard bucket sizing for multi-bucket DDP.
    # Projector-only (<1GB): force single bucket for minimal overhead.
    if trainable_mb > _PROJECTOR_ONLY_TRAINABLE_MB_THRESHOLD:
        bucket_cap_mb = default_bucket_mb
    else:
        bucket_cap_mb = max(default_bucket_mb, int(trainable_mb) + 1)

    # COMPOSITE mode: static_graph=True causes flaky XCCL deadlocks (tested extensively:
    # works 50% of the time, hangs at random micro-batch indices in backward allreduce).
    # find_unused_parameters=True is equally performant and fully reliable.
    # FLAT mode: static_graph=True is stable and avoids per-forward scanning.
    is_composite = os.environ.get("ZE_FLAT_DEVICE_HIERARCHY", "").upper() == "COMPOSITE"
    model_cfg = getattr(model, "config", None)
    model_modalities = getattr(model_cfg, "modalities", None) or []
    is_interleaved = bool(getattr(model_cfg, "is_interleaved_qa", False))
    find_unused, is_multimodal_projector, _, _ = _resolve_find_unused(
        trainable_mb=trainable_mb,
        num_modalities=len(model_modalities),
        is_composite=is_composite,
        is_interleaved=is_interleaved,
        env_override=os.environ.get("PRISM_DDP_FIND_UNUSED"),
    )

    # gradient_as_bucket_view=True reduces memory copies; safe in both modes.
    grad_bucket_view = os.environ.get("PRISM_DDP_GRAD_BUCKET_VIEW", "1") == "1"

    model = DDP(
        model,
        device_ids=[ddp_device_id],
        find_unused_parameters=find_unused,
        broadcast_buffers=False,
        static_graph=not find_unused,
        bucket_cap_mb=bucket_cap_mb,
        gradient_as_bucket_view=grad_bucket_view,
    )

    if is_main:
        n_frozen = len(frozen_params)
        n_total = sum(1 for _ in model.parameters())
        print(
            f"[DDP] Wrapped model with bucket_cap_mb={bucket_cap_mb} (trainable: {trainable_mb:.1f}MB)"
        )
        print(
            f"[DDP] Ignored {n_frozen}/{n_frozen + n_total} frozen params via _ddp_params_and_buffers_to_ignore"
        )
        # Attribute which signal(s) flipped find_unused, when relevant. Order
        # matters: check more-specific combinations first so all active signals
        # are reported (useful in post-mortems for runs where multiple auto-
        # detects would independently force find_unused=True).
        if "PRISM_DDP_FIND_UNUSED" in os.environ:
            auto_note = " (env override)"
        else:
            active = []
            if is_composite:
                active.append("COMPOSITE")
            if is_interleaved:
                active.append("interleaved-QA")
            if is_multimodal_projector:
                active.append(
                    f"multi-modality projector "
                    f"({len(model_modalities)} modalities, {trainable_mb:.0f}MB trainable)"
                )
            if active:
                auto_note = f" ({' + '.join(active)} auto-detect)"
            else:
                auto_note = ""
        print(
            f"[DDP] find_unused_parameters={find_unused}, static_graph={not find_unused}"
            f", gradient_as_bucket_view={grad_bucket_view}{auto_note}"
        )

    return model


# --- Checkpointing ---


def save_native_ddp_checkpoint(model, optimizer, scheduler, step, config, rank):
    """Save checkpoint from DDP or FSDP training.

    For DDP: rank 0 saves the full state dict directly.
    For FSDP: all ranks participate in gathering the sharded state dict,
    but only rank 0 writes to disk.

    Saves model weights (with wrapper prefixes stripped), optimizer state,
    scheduler state, and training metadata. Compatible with stage transitions
    (projector-only -> E2E) via load_model_weights_only().

    Args:
        model: DDP or FSDP-wrapped model.
        optimizer: Optimizer with current state.
        scheduler: LR scheduler with current state.
        step: Current training step.
        config: TrainingConfig with output_dir.
        rank: Current process rank (only rank 0 writes files).
    """
    from safetensors.torch import save_file

    # Detect FSDP vs DDP
    is_fsdp = False
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        is_fsdp = isinstance(model, FSDP)
    except ImportError:
        pass

    if is_fsdp:
        # FSDP: all ranks must participate in state_dict gathering
        from torch.distributed.fsdp import (
            FullOptimStateDictConfig,
            FullStateDictConfig,
            StateDictType,
        )

        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        optim_policy = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            save_policy,
            optim_policy,
        ):
            state_dict = model.state_dict()
            optimizer_state = FSDP.optim_state_dict(model, optimizer)
    else:
        # DDP or unwrapped: get state dict from unwrapped model
        unwrapped = model.module if hasattr(model, "module") else model
        state_dict = {
            k: v.contiguous().cpu() for k, v in unwrapped.state_dict().items()
        }
        optimizer_state = optimizer.state_dict()

    # Only rank 0 writes files
    if rank != 0:
        return

    ckpt_dir = os.path.join(config.output_dir, f"step_{step}")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Strip wrapper prefixes ("module.", "_fsdp_wrapped_module.") for clean keys
    clean_state_dict = {}
    for k, v in state_dict.items():
        clean_key = k
        for prefix in ("module.", "_fsdp_wrapped_module."):
            if clean_key.startswith(prefix):
                clean_key = clean_key[len(prefix):]
        clean_state_dict[clean_key] = v.contiguous() if not v.is_contiguous() else v

    # Save model weights as safetensors
    save_file(clean_state_dict, os.path.join(ckpt_dir, "model.safetensors"))

    dist_strategy = os.environ.get("DIST_STRATEGY", "ddp").lower()
    fsdp_sharding = os.environ.get("FSDP_SHARDING")
    checkpoint_world_size = dist.get_world_size() if dist.is_initialized() else 1

    # Save optimizer + scheduler state (for exact resume)
    torch.save(
        {
            "optimizer": optimizer_state,
            "scheduler": scheduler.state_dict(),
            "dist_strategy": dist_strategy,
            "fsdp_sharding": fsdp_sharding,
            "optimizer_state_type": "full" if is_fsdp else "local",
            "world_size": checkpoint_world_size,
            "dataloader": {
                "completed_steps": int(step),
                "completed_microbatches": int(
                    step * getattr(config, "gradient_accumulation_steps", 1)
                ),
                "finite_epoch_steps": int(getattr(config, "finite_epoch_steps", 0)),
                "resume_strategy": "deterministic_replay_skip",
            },
        },
        os.path.join(ckpt_dir, "training_state.pt"),
    )

    # Save metadata (step, config, param group info)
    metadata = {
        "step": step,
        "config": {
            k: str(v) if not isinstance(v, int | float | bool | str | type(None)) else v
            for k, v in config.__dict__.items()
        },
        "param_groups": [
            {"name": pg.get("name", f"group_{i}"), "lr": pg["lr"]}
            for i, pg in enumerate(optimizer.param_groups)
        ],
        "dist_strategy": dist_strategy,
        "fsdp_sharding": fsdp_sharding,
        "optimizer_state_type": "full" if is_fsdp else "local",
        "world_size": checkpoint_world_size,
        "dataloader": {
            "completed_steps": int(step),
            "completed_microbatches": int(
                step * getattr(config, "gradient_accumulation_steps", 1)
            ),
            "finite_epoch_steps": int(getattr(config, "finite_epoch_steps", 0)),
            "resume_strategy": "deterministic_replay_skip",
        },
    }
    with open(os.path.join(ckpt_dir, "training_state.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[Checkpoint] Saved step {step} to {ckpt_dir}")


def load_model_weights_only(model, checkpoint_path, device="cpu"):
    """Load only model weights from a checkpoint, ignoring optimizer/scheduler.

    This enables stage transitions (e.g., projector-only -> E2E) where the
    optimizer structure changes between stages. Handles DDP "module." prefix
    in both directions.

    Args:
        model: UnifiedTransformer (unwrapped, before DDP wrapping).
        checkpoint_path: Path to checkpoint directory containing model.safetensors.
        device: Device to load weights onto.

    Returns:
        step: The training step from the checkpoint metadata (0 if not found).
    """
    from safetensors.torch import load_file

    # Find the model weights file
    safetensors_path = os.path.join(checkpoint_path, "model.safetensors")
    if not os.path.exists(safetensors_path):
        raise FileNotFoundError(
            f"No model.safetensors found in {checkpoint_path}. "
            f"Contents: {os.listdir(checkpoint_path)}"
        )

    print(f"[Resume] Loading model weights from {safetensors_path}")
    state_dict = load_file(safetensors_path, device=str(device))

    # Strip "module." prefix if present (from DDP-saved checkpoints)
    clean_state_dict = {}
    for k, v in state_dict.items():
        clean_key = k[7:] if k.startswith("module.") else k
        clean_state_dict[clean_key] = v

    clean_state_dict = _align_checkpoint_vocab_for_resume(model, clean_state_dict)

    # Remap pre-decoder-refactor keys (e.g. VLA action_head.* -> action_head.head.*)
    # so checkpoints saved before the OutputDecoder refactor load losslessly.
    clean_state_dict = remap_legacy_decoder_keys(clean_state_dict)

    # Load with strict=False to handle frozen params that may not be in checkpoint.
    # PyTorch still raises on same-name tensor shape mismatches, so vocab-sized
    # tensors are aligned above before calling load_state_dict.
    missing, unexpected = model.load_state_dict(clean_state_dict, strict=False)

    # Report what happened
    if missing:
        # Filter out expected missing keys (buffers, frozen params that weren't saved)
        important_missing = [
            k
            for k in missing
            if not any(skip in k for skip in ["_encoder_frozen_cache", "position_ids"])
        ]
        if important_missing:
            print(
                f"[Resume] Missing keys ({len(important_missing)}): {important_missing[:10]}..."
            )
    if unexpected:
        print(f"[Resume] Unexpected keys ({len(unexpected)}): {unexpected[:10]}...")

    loaded_count = len(clean_state_dict) - len(unexpected)
    print(f"[Resume] Loaded {loaded_count} weight tensors successfully")

    # Read step from metadata
    step = 0
    meta_path = os.path.join(checkpoint_path, "training_state.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            metadata = json.load(f)
        step = metadata.get("step", 0)
        print(f"[Resume] Checkpoint was saved at step {step}")

        # Log previous training config for comparison
        prev_groups = metadata.get("param_groups", [])
        if prev_groups:
            print(f"[Resume] Previous param groups: {prev_groups}")

    return step


def _checkpoint_backbone_vocab_size(state_dict):
    value = state_dict.get("backbone.model.embed_tokens.weight")
    if value is not None and getattr(value, "ndim", 0) == 2:
        return int(value.shape[0])
    return None


def _model_tokenizer_len(model):
    tokenizer = getattr(model, "backbone_tokenizer", None)
    if tokenizer is None:
        backbone = getattr(model, "backbone", None)
        tokenizer = getattr(backbone, "tokenizer", None)
    if tokenizer is None or not hasattr(tokenizer, "__len__"):
        return None
    try:
        return int(len(tokenizer))
    except TypeError:
        return None


def _set_backbone_vocab_size(model, vocab_size):
    backbone = getattr(model, "backbone", None)
    if backbone is not None and hasattr(backbone, "config"):
        backbone.config.vocab_size = int(vocab_size)
    if hasattr(model, "config"):
        model.config.vocab_size = int(vocab_size)


def _resize_backbone_to_checkpoint_vocab(model, checkpoint_vocab):
    backbone = getattr(model, "backbone", None)
    if (
        backbone is None
        or not hasattr(backbone, "resize_token_embeddings")
        or not hasattr(backbone, "get_input_embeddings")
    ):
        return False

    embeddings = backbone.get_input_embeddings()
    if embeddings is None or not hasattr(embeddings, "weight"):
        return False
    current_vocab = int(embeddings.weight.shape[0])
    if current_vocab == checkpoint_vocab:
        _set_backbone_vocab_size(model, checkpoint_vocab)
        return True

    tokenizer_len = _model_tokenizer_len(model)
    if checkpoint_vocab < current_vocab:
        detail = (
            f" tokenizer length {tokenizer_len};" if tokenizer_len is not None else ""
        )
        print(
            "[Resume] Checkpoint vocab "
            f"{checkpoint_vocab} is smaller than current model vocab {current_vocab};"
            f"{detail} "
            "will pad checkpoint vocab tensors instead of shrinking the model"
        )
        return False

    print(
        f"[Resume] Resizing backbone token embeddings {current_vocab} -> "
        f"{checkpoint_vocab} to match checkpoint"
    )
    rng_state = torch.get_rng_state()
    try:
        torch.manual_seed(0)
        backbone.resize_token_embeddings(checkpoint_vocab)
    finally:
        torch.set_rng_state(rng_state)
    _set_backbone_vocab_size(model, checkpoint_vocab)
    return True


def _copy_vocab_rows_into_current_tensor(key, checkpoint_tensor, current_tensor):
    if checkpoint_tensor.ndim != 2 or current_tensor.ndim != 2:
        return None
    if checkpoint_tensor.shape[1] != current_tensor.shape[1]:
        return None

    merged = current_tensor.detach().clone()
    rows = min(int(checkpoint_tensor.shape[0]), int(current_tensor.shape[0]))
    merged[:rows].copy_(
        checkpoint_tensor[:rows].to(device=merged.device, dtype=merged.dtype)
    )
    print(
        f"[Resume] Adapted {key}: copied {rows} checkpoint vocab rows "
        f"into current shape {tuple(current_tensor.shape)}"
    )
    return merged


def _pad_mismatched_vocab_tensors(model, state_dict):
    current_state = model.state_dict()
    adapted = dict(state_dict)
    for key in ("backbone.model.embed_tokens.weight", "backbone.lm_head.weight"):
        checkpoint_tensor = adapted.get(key)
        current_tensor = current_state.get(key)
        if checkpoint_tensor is None or current_tensor is None:
            continue
        if tuple(checkpoint_tensor.shape) == tuple(current_tensor.shape):
            continue
        merged = _copy_vocab_rows_into_current_tensor(
            key, checkpoint_tensor, current_tensor
        )
        if merged is not None:
            adapted[key] = merged
    return adapted


def _align_checkpoint_vocab_for_resume(model, state_dict):
    checkpoint_vocab = _checkpoint_backbone_vocab_size(state_dict)
    if checkpoint_vocab is None:
        return state_dict

    resized = _resize_backbone_to_checkpoint_vocab(model, checkpoint_vocab)
    if resized:
        return state_dict
    return _pad_mismatched_vocab_tensors(model, state_dict)
