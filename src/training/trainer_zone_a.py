import os

import torch
import torch.optim as optim
from accelerate import (
    Accelerator,
    DataLoaderConfiguration,
    DistributedDataParallelKwargs,
)
from accelerate.utils import send_to_device
from tqdm import tqdm

from src.config import TrainingConfig
from src.model import UnifiedTransformer
from src.utils.perf_log import (
    _FlopCounter,
    batch_modality_counts,
    batch_token_count,
    count_parameters,
    log_perf_record,
    model_modalities,
    sequence_stats,
)

try:
    import wandb
except Exception:
    wandb = None


class _NullEvaluatorRegistry:
    @staticmethod
    def get(_registry_key):
        return None


_EVAL_IMPORT_ERROR = None
try:
    from src.eval import EvaluatorRegistry
except Exception as exc:
    EvaluatorRegistry = _NullEvaluatorRegistry()
    _EVAL_IMPORT_ERROR = exc

from transformers import get_cosine_schedule_with_warmup

_BATCH_VIZ_IMPORT_ERROR = None
try:
    from src.utils.batch_viz import BatchVisualizer
except Exception as exc:
    _BATCH_VIZ_IMPORT_ERROR = exc

    class BatchVisualizer:
        def __init__(self, tokenizer=None):
            self.tokenizer = tokenizer

        def visualize(self, _batch):
            return {}

        def visualize_predictions(self, _batch, _model, modality="image"):
            return None


import logging

# Configure logging to file
os.makedirs("log", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("log/model.log"),  # Logs to file
        logging.StreamHandler(),  # Also prints to console
    ],
)


class ModalityMonitor:
    def __init__(self, accelerator, threshold_warn=50, threshold_error=500):
        self.accelerator = accelerator
        self.limit_warn = threshold_warn
        self.limit_error = threshold_error
        self.consecutive_text_only = 0
        self.seen_modalities = set()

    def check(self, batch, step):
        # Identify present modalities (excluding 'text' which is usually present as labels)
        # Note: In PRISM, 'text' is the input for SFT. Multimodal inputs have other keys.
        present = [
            k
            for k in batch.keys()
            if k != "text" and k != "labels" and not k.startswith("_")
        ]

        if not present:
            self.consecutive_text_only += 1
        else:
            self.consecutive_text_only = 0
            for p in present:
                self.seen_modalities.add(p)

        # Logging
        if self.consecutive_text_only > 0 and self.consecutive_text_only % 10 == 0:
            if self.accelerator.is_main_process and wandb is not None:
                wandb.log(
                    {"monitor/consecutive_text_only": self.consecutive_text_only},
                    step=step,
                )

        # Warnings/Errors
        if self.consecutive_text_only == self.limit_warn:
            self.accelerator.print(
                f"\n[Warning] Data Starvation? {self.limit_warn} consecutive text-only batches."
            )

        if self.consecutive_text_only >= self.limit_error:
            msg = f"Data Starvation Error: {self.limit_error} consecutive text-only batches. Aborting to prevent ghost training."
            self.accelerator.print(msg)
            raise RuntimeError(msg)


class ZoneATrainer:
    def _unwrap_model(self):
        """Unwrap container layers without tripping Accelerate's compile-region bug.

        Accelerate's `extract_model_from_parallel` finds compiled submodules via a
        recursive walk but then assumes `_orig_mod` lives on the top-level container,
        which raises KeyError on DeepSpeedEngine. We strip DDP/DeepSpeed/FSDP wrappers
        manually instead — `.module` exposes the inner UnifiedTransformer directly.
        """
        if getattr(self, "_use_deepspeed", False) and getattr(self, "_ds_compiled_backbone", False):
            inner = self.model
            while hasattr(inner, "module"):
                inner = inner.module
            return inner
        return self.accelerator.unwrap_model(self.model)

    def __init__(self, model: UnifiedTransformer, config: TrainingConfig, train_loader):
        self.config = config
        self.monitor = None  # Init later

        # 1. Initialize Accelerator
        # Enables Mixed Precision, FSDP, and Multi-GPU automatically
        # FORCE BF16 for Aurora/XPU Stability
        dataloader_config = DataLoaderConfiguration(dispatch_batches=False)
        # DDP wrapping kwargs — only relevant when Accelerate is wrapping with DDP.
        # DeepSpeed manages its own grad sync and rejects DDP kwargs on the kwargs_handlers
        # list, so we drop them when DEEPSPEED_ZERO_STAGE is set to a non-empty value.
        use_deepspeed = os.environ.get("DEEPSPEED_ZERO_STAGE", "").strip() != ""
        self._use_deepspeed = use_deepspeed
        self._ds_compiled_backbone = False

        # Pin LOCAL_RANK to 0 BEFORE Accelerator() instantiation when DeepSpeed is set.
        # Accelerate calls torch.distributed.init_process_group(device_id=xpu:LOCAL_RANK)
        # during the Accelerator() constructor (BEFORE accelerator.prepare()). With
        # ZE_AFFINITY_MASK each rank sees only xpu:0, so LOCAL_RANK must be 0 here or
        # init_process_group raises "device_id xpu:N is out of range". Stash the real
        # value in PRISM_REAL_LOCAL_RANK so src/training/distributed.py can recover it.
        if use_deepspeed:
            real_local_rank = os.environ.get("LOCAL_RANK")
            if real_local_rank is not None and real_local_rank != "0":
                os.environ["PRISM_REAL_LOCAL_RANK"] = real_local_rank
                os.environ["LOCAL_RANK"] = "0"

        kwargs_handlers = []
        if not use_deepspeed:
            # For Zone A training, we have frozen encoders/backbone with trainable projectors
            # OPTIMIZATION: With frozen backbone+encoders, only projector params have gradients.
            # Setting find_unused_parameters=False avoids scanning all 7B+ params every forward.
            # Set PRISM_DDP_FIND_UNUSED=1 to revert to old behavior for debugging.
            #
            # Multi-modality projector-only auto-detect: mirror the logic in
            # src/training/distributed.py::_resolve_find_unused — heterogeneous
            # batches across ranks crash DDP's bucket rebuild with
            # `RuntimeError: Empty bucket specified` when projectors get no
            # gradient flow on some ranks. The native DDP wrap path computes
            # trainable_mb from the model; here we use len(model.config.modalities)
            # alone (Accelerate wraps before we've materialized the param count).
            from src.training.distributed import _resolve_find_unused

            num_modalities = len(getattr(model.config, "modalities", []) or [])
            is_composite = (
                os.environ.get("ZE_FLAT_DEVICE_HIERARCHY", "").upper() == "COMPOSITE"
            )
            is_interleaved = bool(getattr(model.config, "is_interleaved_qa", False))
            # Pass trainable_mb=0 to assume projector-only (the only regime
            # this Accelerate trainer is configured for — frozen encoders +
            # backbone). E2E Accelerate users would need a different code path.
            find_unused, _, _, _ = _resolve_find_unused(
                trainable_mb=0.0,
                num_modalities=num_modalities,
                is_composite=is_composite,
                is_interleaved=is_interleaved,
                env_override=os.environ.get("PRISM_DDP_FIND_UNUSED"),
            )
            ddp_kwargs = DistributedDataParallelKwargs(
                find_unused_parameters=find_unused,
                broadcast_buffers=False,  # Buffers are identical, skip broadcast
                static_graph=not find_unused,  # Enable graph optimization when not finding unused
            )
            kwargs_handlers.append(ddp_kwargs)
        self.accelerator = Accelerator(
            mixed_precision="bf16",
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            log_with="wandb" if config.wandb_project else None,
            dataloader_config=dataloader_config,
            kwargs_handlers=kwargs_handlers,
        )
        if self.accelerator.state.deepspeed_plugin is not None:
            ds_plugin = self.accelerator.state.deepspeed_plugin
            self.accelerator.print(
                f"[DeepSpeed] Active: ZeRO stage {ds_plugin.zero_stage}, "
                f"offload_optimizer={ds_plugin.offload_optimizer_device}, "
                f"offload_param={ds_plugin.offload_param_device}"
            )

        self.monitor = ModalityMonitor(self.accelerator)
        self.visualizer = BatchVisualizer(model.tokenizer if hasattr(model, "tokenizer") else None)
        if self.accelerator.is_main_process and _BATCH_VIZ_IMPORT_ERROR is not None:
            self.accelerator.print(
                f"BatchVisualizer dependencies unavailable; visualization disabled ({_BATCH_VIZ_IMPORT_ERROR})"
            )

        self.model = model
        self.train_loader = train_loader

        # 2. Freeze Logic & Dtype Unification (Must be done BEFORE prepare)

        # 2. Freeze/Unfreeze Logic and Parameter Groups
        # Determines target dtype from backbone
        target_dtype = torch.float32
        if self.model.backbone:
            target_dtype = next(self.model.backbone.parameters()).dtype
            if hasattr(torch, "xpu") and torch.xpu.is_available() and target_dtype == torch.float16:
                self.accelerator.print(
                    "[Warning] Backbone loaded as float16 on XPU. Forcing alignment to bfloat16."
                )
                target_dtype = torch.bfloat16
                self.model.backbone.to(dtype=target_dtype)

            self.accelerator.print(f"Aligning Dtypes to {target_dtype}")

        # --- Parameter Collections ---
        params_connector = []
        params_vit = []
        params_llm = []
        params_lora = []

        # A. Backbone (LLM)
        lora_enabled = getattr(config, "lora_enabled", False)
        if self.model.backbone:
            # Freeze check (Config overrides ModelConfig default if present)
            freeze_llm = getattr(config, "freeze_llm", True)
            # Note: ModelConfig has freeze_backbone, we should respect the toggle

            # freeze_llm and lora_enabled are independent, orthogonal knobs --
            # freeze_llm controls whether the base backbone is frozen,
            # lora_enabled controls whether LoRA adapters are added on top.
            # Checked separately, not coupled: BioReason SFT sets BOTH
            # freeze_llm=true (base frozen) AND lora_enabled=true (adapters
            # train) simultaneously. apply_lora_torchtune below re-freezes
            # base weights and unfreezes only lora_a/lora_b regardless of
            # what this block does, so the end state is the same either way
            # -- but freeze_llm's own value must still reflect the real
            # intent (and drives train.py's earlier freeze/log step, see
            # train.py:567-582), not be silently ignored whenever LoRA is on.
            if freeze_llm:
                self.accelerator.print("Freezing HF Backbone (LLM)...")
                for param in self.model.backbone.parameters():
                    param.requires_grad = False
            else:
                self.accelerator.print("Unfreezing HF Backbone (LLM)...")
                for param in self.model.backbone.parameters():
                    param.requires_grad = True

            if lora_enabled:
                # torchtune replaces every nn.Linear (except lm_head) with LoRALinear
                # in-place, copies pretrained weights, then freezes base weights via
                # set_trainable_params — only lora_a.weight / lora_b.weight keep
                # requires_grad=True afterward. Same mechanism trainer_grpo.py
                # uses for the GRPO stage, kept consistent here.
                from src.utils.lora_utils import apply_lora_torchtune

                self.model.backbone = apply_lora_torchtune(
                    self.model.backbone,
                    rank=config.lora_r,
                    alpha=config.lora_alpha,
                    dropout=config.lora_dropout,
                )
                self.accelerator.print("torchtune LoRA applied to backbone.")
            elif not freeze_llm:
                params_llm.extend(
                    [p for p in self.model.backbone.parameters() if p.requires_grad]
                )

        # B. Encoders (ViT / Others)
        # Cast to target dtype first
        self.model.encoders.to(dtype=target_dtype)

        # DNA/NT encoders produce NaN in bf16/fp16 due to attention overflow;
        # keep the NT backbone itself in float32 regardless of the blanket
        # dtype cast above (matches DNAEncoder._apply()'s own float32 pin in
        # src/encoders/dna.py — belt-and-suspenders). No-op for non-DNA models.
        if "dna" in self.model.encoders and hasattr(self.model.encoders["dna"], "model"):
            self.model.encoders["dna"].model.float()
            self.accelerator.print("DNA encoder backbone kept in float32 (NT bfloat16 instability guard)")

        # These are already frozen in model.py
        freeze_vit = getattr(config, "freeze_vit", True)
        if freeze_vit:
            self.accelerator.print("Freezing Encoders...")
            for param in self.model.encoders.parameters():
                param.requires_grad = False
        else:
            self.accelerator.print("Unfreezing Encoders...")
            for param in self.model.encoders.parameters():
                param.requires_grad = True
            params_vit.extend([p for p in self.model.encoders.parameters() if p.requires_grad])



        # PATRICK: Is commenting this out breaking? 
        #self.accelerator.print("Training Projectors Only (Freezing Non-Projector Params)...")
        # Letting the freeze logic handle parameters allows experimentation with unfreezing encoders/backbones.
        
        # Zone A is projector warmup: freeze everything except projectors.
        #for name, param in self.model.named_parameters():
        #    param.requires_grad = name.startswith("projectors.")

        # Cast projectors to target_dtype
        self.model.projectors.to(dtype=target_dtype)

        active_mods = getattr(self.train_loader.dataset, "active_modalities", None)
        # BioReason Stage 2 SFT loads a pretrained projector from Stage 1 and
        # keeps it frozen while only LoRA trains — freeze_connector overrides
        # active_mods-based training for every projector, not just inactive ones.
        freeze_connector = getattr(config, "freeze_connector", False)

        for name, submodule in self.model.projectors.items():
            if freeze_connector or (active_mods and name not in active_mods):
                for param in submodule.parameters():
                    param.requires_grad = False
                continue

            for param in submodule.parameters():
                param.requires_grad = True
                params_connector.append(param)

        if lora_enabled:
            params_lora = [
                p for n, p in self.model.named_parameters() if "lora_" in n and p.requires_grad
            ]
            self.accelerator.print(
                f"LoRA trainable params: {sum(p.numel() for p in params_lora):,}"
            )

        # --- Optimizer Construction with Groups ---
        # LRs
        lr_base = config.learning_rate
        lr_conn = getattr(config, "lr_connector", None) or lr_base
        lr_vit = getattr(config, "lr_vit", None) or lr_base
        lr_llm = getattr(config, "lr_llm", None) or lr_base
        lr_lora = getattr(config, "lr_llm", None) or lr_base

        param_groups = []
        # Group 0: Connector (Must be first for scheduler logic if relying on index)
        if params_connector:
            param_groups.append({"params": params_connector, "lr": lr_conn, "name": "connector"})

        # Group 1: ViT
        if params_vit:
            param_groups.append({"params": params_vit, "lr": lr_vit, "name": "vit"})


        # Group 2: LLM
        if params_llm:
            param_groups.append({"params": params_llm, "lr": lr_llm, "name": "llm"})

        # Group 3: LoRA
        if params_lora:
            param_groups.append({"params": params_lora, "lr": lr_lora, "name": "lora"})

        self.accelerator.print(
            f"Optimizer Groups: Connector LR={lr_conn}, ViT LR={lr_vit}, "
            f"LLM LR={lr_llm}, LoRA LR={lr_lora if params_lora else 'N/A'}"
        )
        self.accelerator.print(
            f"Trainable Counts: Connector={len(params_connector)}, ViT={len(params_vit)}, "
            f"LLM={len(params_llm)}, LoRA={len(params_lora)}"
        )

        self.optimizer = optim.AdamW(param_groups, weight_decay=config.weight_decay)

        # Scheduler Selection
        sched_type = getattr(config, "scheduler_type", "cosine")
        min_lr_ratio = getattr(config, "min_lr_ratio", 0.0)

        if sched_type == "molmo_layered":
            from src.utils.scheduler import get_molmo_scheduler

            self.accelerator.print("Using Molmo Layered Scheduler (Separate Warmups)")
            self.scheduler = get_molmo_scheduler(
                self.optimizer,
                config.max_steps,
                warmup_connector=getattr(config, "warmup_steps_connector", 200),
                warmup_main=getattr(config, "warmup_steps_main", 2000),
                min_lr_ratio=min_lr_ratio,
            )
        elif sched_type == "cosine_with_min_lr":
            from src.utils.scheduler import get_cosine_with_min_lr

            self.accelerator.print(f"Using Cosine Scheduler with Floor {min_lr_ratio}")
            self.scheduler = get_cosine_with_min_lr(
                self.optimizer,
                num_warmup_steps=config.warmup_steps,
                num_training_steps=config.max_steps,
                min_lr_ratio=min_lr_ratio,
            )
        elif sched_type == "wsd":
            from src.utils.scheduler import get_wsd_scheduler

            wsd_decay_steps = getattr(config, "wsd_decay_steps", None)
            self.accelerator.print(
                "Using WSD Scheduler "
                f"(warmup={config.warmup_steps}, "
                f"decay_ratio={getattr(config, 'wsd_decay_ratio', 0.1)}, "
                f"decay_steps={wsd_decay_steps}, floor={min_lr_ratio})"
            )
            self.scheduler = get_wsd_scheduler(
                self.optimizer,
                num_warmup_steps=config.warmup_steps,
                num_training_steps=config.max_steps,
                min_lr_ratio=min_lr_ratio,
                decay_ratio=getattr(config, "wsd_decay_ratio", 0.1),
                decay_steps=wsd_decay_steps,
            )
        else:
            # Default HF
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=config.warmup_steps,
                num_training_steps=config.max_steps,
            )

        # 3. Prepare via Accelerator
        # This handles device placement, DDP wrapping, FSDP wrapping, etc.
        # NOTE: We DO NOT prepare scheduler here yet. We must do it AFTER load_state
        # to allow resuming from checkpoints that lack scheduler.bin (partial loading).
        #
        # LOCAL_RANK already pinned to 0 above (before Accelerator()) when use_deepspeed.
        # Don't prepare the dataloader under DeepSpeed — Accelerate's IterableDatasetShard
        # wrapper re-shards data that WebDataset already sharded via split_by_node, causing
        # each rank to load num_processes * batch_size samples and discard all but
        # batch_size. This makes data loading ~12x slower on a 12-tile node.
        # DeepSpeed needs train_micro_batch_size_per_gpu set explicitly when no
        # dataloader is passed to prepare().
        if use_deepspeed:
            ds_plugin = self.accelerator.state.deepspeed_plugin
            ds_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = config.batch_size
            self.model, self.optimizer = self.accelerator.prepare(
                self.model, self.optimizer
            )
        else:
            self.model, self.optimizer, self.train_loader = self.accelerator.prepare(
                self.model, self.optimizer, self.train_loader
            )

        # Cache device for manual batch moves under DeepSpeed (where train_loader is
        # NOT prepared, so Accelerate doesn't auto-move tensors to xpu:0).
        self._device = self.accelerator.device

        # Compile the LLM backbone after DeepSpeed wraps the model in DeepSpeedEngine.
        # The non-DeepSpeed path compiles before wrapping (see src/training/distributed.py);
        # for ZeRO-3, params are sharded during prepare(), so compiling earlier would
        # trace unsharded params and conflict with the AllGather hooks added later.
        if use_deepspeed and os.environ.get("TORCH_COMPILE", "0") == "1":
            # NOTE: torch.compile + DeepSpeed is NOT validated on Aurora XPU.
            # Prior runs hit NaN losses (ZeRO-2) and AttributeError (ZeRO-3).
            # See ~/.claude/.../deepspeed_zero2_results.md. This path is left
            # in for future experimentation; do not enable in production.
            self.accelerator.print(
                "[WARN] TORCH_COMPILE=1 with DeepSpeed is experimental and not "
                "validated on Aurora XPU; prior runs produced NaN / AttributeError. "
                "Disable by unsetting TORCH_COMPILE if training diverges."
            )
            compile_backend = os.environ.get("TORCH_COMPILE_BACKEND", "inductor")
            use_dynamic = os.environ.get("TORCH_COMPILE_DYNAMIC", "1") == "1"
            # unwrap_model returns a reference to the inner UnifiedTransformer;
            # mutating .backbone here is observed by the DeepSpeedEngine wrapper
            # because both hold the same underlying module reference.
            inner_model = self.accelerator.unwrap_model(self.model)
            if hasattr(inner_model, "backbone") and inner_model.backbone is not None:
                inner_model.backbone = torch.compile(
                    inner_model.backbone,
                    backend=compile_backend,
                    dynamic=use_dynamic,
                )
                self._ds_compiled_backbone = True
                self.accelerator.print(
                    f"[torch.compile] DeepSpeed backbone compiled "
                    f"(backend={compile_backend}, dynamic={use_dynamic})"
                )

        # 4. Initialize Evaluators (Only on Main Process usually, or distributed?)
        # BaseEvaluator handles .to(device).
        # We should instantiate them here.
        # Ideally, we only run eval on main process to avoid duplication in logging.
        self.evaluators = []
        if self.accelerator.is_main_process and _EVAL_IMPORT_ERROR is None:
            self.accelerator.print("Initializing Evaluators...")
            # Unwrapped model for eval generation?
            # Accelerator wraps model.
            # We can pass self.model (wrapped) - generate() should work on FSDP/DDP wrapped models mostly.
            # But safer to unwrap if generation acts up. For now, pass wrapped.

            # Note: Evaluators load their own validation datasets.
            # Wrapper helper for robust init
            def add_evaluator(name, registry_key):
                try:
                    cls = EvaluatorRegistry.get(registry_key)
                    if cls:
                        # Unwrap model to access tokenizer and other attributes hidden by DDP/DeepSpeed/FSDP
                        unwrapped_model = self._unwrap_model()

                        self.evaluators.append(
                            cls(
                                unwrapped_model,
                                unwrapped_model.tokenizer,
                                self.accelerator.device,
                            )
                        )
                    else:
                        self.accelerator.print(
                            f"Warning: Evaluator {registry_key} not found in registry."
                        )
                except Exception as e:
                    self.accelerator.print(f"Warning: Failed to init evaluator {registry_key}: {e}")

            # Geometry
            add_evaluator("MatBench", "geometry_matbench")
            # Graph
            add_evaluator("ChEBI-20", "graph_chebi")
            # Time
            add_evaluator("Time-MMD", "ts_timemmd")
            # Table
            add_evaluator("Spider", "table_spider")

            # Expanded Suite
            add_evaluator("VQAv2", "vision_vqa")
            add_evaluator("Monash", "ts_monash")
            add_evaluator("MMLU-Pro", "text_mmlu_pro")
            add_evaluator("GPQA", "text_gpqa")
            add_evaluator("AIME", "text_aime2025")
            add_evaluator("IFEval", "text_ifeval")

            # Vision Expansion
            add_evaluator("MathVista", "vision_mathvista")
            add_evaluator("MathVision", "vision_mathvision")
            add_evaluator("MMStar", "vision_mmstar")
            add_evaluator("MMMU", "vision_mmmu")

            self.accelerator.print(f"Initialized {len(self.evaluators)} Evaluators.")
        elif self.accelerator.is_main_process and _EVAL_IMPORT_ERROR is not None:
            self.accelerator.print(
                f"Skipping evaluator initialization: optional dependencies unavailable ({_EVAL_IMPORT_ERROR})"
            )

        # WandB Init handled by Accelerator if 'log_with' is set,
        # or we config it manually. Accelerator tracking is cleaner.
        # Explicit WandB Init (Bypassing Accelerator for reliability with Native DDP)
        if config.wandb_project and self.accelerator.is_main_process:
            if wandb is None:
                self.accelerator.print(
                    "[Warning] wandb is not installed. Skipping explicit WandB initialization."
                )
            else:
                self.accelerator.print(
                    f"[DEBUG] Explicitly initializing WandB: {config.wandb_project} run={config.wandb_run_name}"
                )
                try:
                    wandb.init(
                        project=config.wandb_project,
                        name=config.wandb_run_name,
                        config=config.__dict__,
                        reinit=True,
                    )
                    self.accelerator.print("[DEBUG] WandB Explicit Init Successful!")
                except Exception as e:
                    self.accelerator.print(f"[ERROR] WandB Explicit Init Failed: {e}")
        else:
            if self.accelerator.is_main_process:
                self.accelerator.print("[DEBUG] WandB Project not set - Skipping Init.")

        # IsoFLOP FLOP counter. The launcher exports `CALIBRATION_JSON` per
        # cell (see `tools/isoflop_launch.py` in PR-2). When unset, the
        # counter is a no-op and per-step records get `flops_per_step=None`
        # — the collector treats that as missing rather than zero.
        self._flop_counter = _FlopCounter.from_calibration(
            os.environ.get("CALIBRATION_JSON")
        )

        # KEGG eval (run_kegg_eval): dataset/tokenizer are lazy-loaded on first call.
        self._kegg_eval_dataset = None
        self._kegg_eval_logger = None
        self._kegg_eval_nt_tokenizer = None

    def check_parameter_status(self, step):
        """Watchdog: Fail fast if projectors are seemingly frozen."""
        if not self.accelerator.is_main_process:
            return

        print(f"\n[Watchdog] Checking Parameter Status at Step {step}...")

        # 1. Check requires_grad
        # Unwrap model slightly to access projectors directly if needed, but .parameters() works on wrapped usually
        trainable_params = [n for n, p in self.model.named_parameters() if p.requires_grad]

        projector_active = any("projector" in n for n in trainable_params)
        encoder_active = any("encoder" in n for n in trainable_params)
        backbone_active = any("backbone" in n for n in trainable_params)
        lora_active = any("lora_" in n for n in trainable_params)

        print(
            f"  Trainable Scopes: Projectors={projector_active}, Encoders={encoder_active}, "
            f"Backbone={backbone_active}, LoRA={lora_active}"
        )

        # Ghost-training watchdog: projectors should be trainable when the
        # model has non-text modalities (image/ts/graph/table/geometry).
        # text_only cells don't have any active projector — backbone is the
        # only trainable scope and the watchdog should NOT abort in that case.
        # Iterate over the UNWRAPPED model so the `projectors.text` prefix
        # check works under FSDP/DDP wrapping. Under FSDP the param names
        # come back as `_fsdp_wrapped_module.projectors.text.*`, which would
        # make `startswith("projectors.text")` False — flagging the text
        # projector as a non-text modality and firing a false-positive
        # ghost-training abort on FSDP text_only runs.
        _unwrapped = self._unwrap_model()
        non_text_modality_present = any(
            "projectors." in n and not n.startswith("projectors.text")
            for n, _ in _unwrapped.named_parameters()
        )
        # freeze_connector (BioReason Stage 2 SFT): the projector is
        # intentionally frozen (loaded from Stage 1) — the projector-active
        # check above doesn't apply in this mode. lora_enabled and
        # freeze_connector are independent config knobs (bioreason_sft.yaml
        # happens to set both, but nothing requires that): the backbone
        # could instead be fully unfrozen (freeze_llm=false, lora_enabled=
        # false) rather than LoRA-adapted, which is an equally legitimate
        # "something besides the projector is training" state. Checking
        # lora_active alone would false-positive-abort that case.
        freeze_connector = getattr(self.config, "freeze_connector", False)
        if freeze_connector:
            if not (lora_active or backbone_active):
                msg = (
                    "CRITICAL: freeze_connector=True but neither LoRA nor the backbone "
                    "has any trainable params! Ghost Training detected. Aborting."
                )
                print(msg)
                raise RuntimeError(msg)
        elif not projector_active and non_text_modality_present:
            msg = (
                "CRITICAL: Projectors have requires_grad=False! Ghost Training detected. Aborting."
            )
            print(msg)
            raise RuntimeError(msg)
        elif not projector_active:
            print(
                "  Note: no projectors trainable, but model has no non-text "
                "projectors either (text_only cell) — skipping ghost-training check."
            )

        # 2. Check Gradients (Only if step > 0)
        # Sample from each component type
        if step > 0:
            components = [
                ("projector", "Projector"),
                ("lora_", "LoRA"),
                ("encoder", "Encoder/ViT"),
                ("backbone", "Backbone/LLM"),
            ]

            for key, label in components:
                found_grad = False
                max_norm = 0.0
                sample_name = None

                for n, p in self.model.named_parameters():
                    if key in n and p.requires_grad and p.grad is not None:
                        grad_norm = p.grad.norm().item()
                        if grad_norm > max_norm:
                            max_norm = grad_norm
                            sample_name = n
                            found_grad = True

                if found_grad and max_norm > 0:
                    print(
                        f"  ✅ {label}: {sample_name.split('.')[-2]}.{sample_name.split('.')[-1]} (Norm: {max_norm:.6f})"
                    )
                elif found_grad:
                    print(f"  ⚠️ {label}: Gradients exist but are zero!")
                else:
                    # Check if this component is supposed to be trainable
                    is_trainable = any(key in n for n in trainable_params)
                    if is_trainable:
                        print(f"  ⚠️ {label}: TRAINABLE but no gradients! (accum step?)")
                    else:
                        print(f"  ℹ️ {label}: Frozen (no gradients expected)")

    def train(self):
        self.model.train()
        step = 0
        self.model.global_step = 0

        # Only show progress bar on main process
        if self.accelerator.is_main_process:
            progress_bar = tqdm(range(self.config.max_steps), desc="Training Zone A")

        # --- Resumption Logic ---
        if self.config.resume_from_checkpoint:
            print(
                f"\n{'=' * 40}\n RESUMING TRAINING FROM:\n {self.config.resume_from_checkpoint}\n{'=' * 40}"
            )
            self.accelerator.print("[INFO] Loading checkpoint state...")
            self.accelerator.load_state(self.config.resume_from_checkpoint)

            # --- DEBUG: Verify Optimizer State Load ---
            if self.accelerator.is_main_process:
                # Inspect the underlying optimizer state
                validation_opt = self.optimizer
                # Unwrap if necessary (Accelerate/DeepSpeed wrappers)
                while hasattr(validation_opt, "optimizer"):
                    validation_opt = validation_opt.optimizer

                state_len = len(validation_opt.state)
                num_groups = len(validation_opt.param_groups)
                self.accelerator.print(
                    f"[RESUME CHECK] Optimizer Loaded! State Dict contains {state_len} parameter entries across {num_groups} groups."
                )
            # ------------------------------------------

            # Load Step Metadata
            import json

            state_file = f"{self.config.resume_from_checkpoint}/training_state.json"
            if os.path.exists(state_file):
                with open(state_file) as f:
                    state_data = json.load(f)
                    step = state_data.get("step", 0)
                    self.model.global_step = step
                    self.accelerator.print(f"Resumed at Step {step}")
            else:
                self.accelerator.print(
                    "Warning: training_state.json not found. Creating step from directory name?"
                )
                # Try to infer from dirname "step_X"
                try:
                    dirname = os.path.basename(self.config.resume_from_checkpoint.rstrip("/"))
                    if dirname.startswith("step_"):
                        step = int(dirname.split("_")[1])
                        self.accelerator.print(f"Inferred Step {step} from directory name.")
                except Exception:
                    self.accelerator.print(
                        "Could not infer step. Starting at 0 (Risk of LR schedule reset)."
                    )

            # --- CRITICAL: Prepare Scheduler AFTER Loading State ---
            # This avoids FileNotFoundError if resuming from a checkpoint without scheduler.bin.
            # Accelerate won't look for it because it wasn't prepared during load_state.
            self.scheduler = self.accelerator.prepare(self.scheduler)
            # -------------------------------------------------------

            # Fast-forward progress bar
            if self.accelerator.is_main_process and step > 0:
                progress_bar.update(step)

            # --- Scheduler Fast-Forward (Critical for Mid-Run Scheduler Introduction) ---
            # If we resumed from a checkpoint that DID NOT have a scheduler (like yours),
            # the new scheduler is at step 0. We must fast-forward it to 'step'
            # to avoid re-warming up.
            current_sched_step = (
                self.scheduler.scheduler.last_epoch
                if hasattr(self.scheduler, "scheduler")
                else self.scheduler.last_epoch
            )
            # Note: Accelerator wraps scheduler. Accessing underlying logic can be tricky.
            # Safest way: If using LambdaLR (Standard HF), just loop.

            if step > 0 and current_sched_step <= 0:
                self.accelerator.print(
                    f"[Scheduler] Fast-forwarding scheduler from 0 to {step} to match resume step..."
                )
                for _ in range(step):
                    self.scheduler.step()
            # ----------------------------------------------------------------------------
        # ------------------------

        # IsoFLOP startup snapshot — one perf record with per-component param
        # counts so the collector can fill `n_total_params / n_active_params`
        # without re-instantiating the model. Rank-0 only, after prepare().
        if self.accelerator.is_main_process:
            try:
                unwrapped_for_count = self._unwrap_model()
                log_perf_record(
                    getattr(self.config, "output_dir", None),
                    {
                        "event": "startup_param_count",
                        "site": "trainer_zone_a",
                        **count_parameters(unwrapped_for_count),
                        "sweep_id": getattr(self.config, "sweep_id", None),
                        "preset": getattr(self.config, "preset", None),
                        "seed": getattr(self.config, "seed", None),
                        "modalities": model_modalities(unwrapped_for_count),
                        "projector_hidden_mult": getattr(
                            unwrapped_for_count.config,
                            "projector_hidden_mult",
                            1,
                        ),
                        "projector_num_layers": getattr(
                            unwrapped_for_count.config,
                            "projector_num_layers",
                            2,
                        ),
                    },
                )
            except Exception as _e_startup:  # noqa: BLE001
                self.accelerator.print(
                    f"[perf] startup_param_count skipped: {_e_startup}"
                )

        data_iter = iter(self.train_loader)

        # --- Performance Timing Setup ---
        import time

        # Production-mode switch: when set (typically by --benchmark-mode in the
        # launcher), skip the per-microbatch torch.xpu.synchronize() and
        # dist.barrier() calls below. Those are useful for diagnosing where time
        # is spent (forward vs backward vs allreduce) but each adds ~ms of
        # overhead per microbatch. For fair throughput benchmarking against
        # other strategies (DeepSpeed, etc.), turn them off.
        _production_mode = os.environ.get("PRISM_PRODUCTION_MODE", "0") == "1"

        timing_data_fetch = 0.0
        timing_forward = 0.0
        timing_backward = 0.0
        timing_backward_compute = 0.0  # Pure gradient computation (no comm)
        timing_allreduce_wait = 0.0  # DDP AllReduce communication wait
        timing_optimizer = 0.0
        timing_sync_wait = 0.0  # Explicit sync overhead
        timing_steps = 0
        micro_batch_count = 0  # Track total micro-batches
        step_start_time = time.perf_counter()  # Wall clock for step

        # tokens_per_window: rank-0's non-pad text-token count across the
        # current 50-step window; reset alongside the timing accumulators below.
        # last_batch_modality_counts: per-step snapshot of which modalities
        # actually showed up in the batch.
        tokens_per_window = 0
        last_batch_modality_counts: dict[str, int] = {}
        # IsoFLOP: per-sample non-pad text length collected across the
        # current throughput window; flushed alongside the timing
        # accumulators below.
        # NOTE: rank-0 only (populate site gated on `is_main_process`).
        # The seq_p* / padding_ratio in the perf row reflect ONE rank's
        # batch composition, not a global aggregate — matches the
        # pre-existing tokens_per_window extrapolation. Safe for
        # homogeneous batches; misleading for heterogeneous multimodal
        # mixes. See PR feedback on PR #98.
        seq_lens_window: list[int] = []
        # Pull tokenizer.pad_token_id from the (unwrapped) model — the trainer
        # doesn't keep a direct tokenizer handle, but train.py attaches one to
        # the model via `setattr(model, "tokenizer", tokenizer)` before init.
        _tok_for_pad = None
        try:
            _tok_for_pad = getattr(self._unwrap_model(), "tokenizer", None)
        except Exception:  # noqa: BLE001
            pass
        _pad_id_for_tokens = (
            getattr(_tok_for_pad, "pad_token_id", None) if _tok_for_pad is not None else None
        )

        self.optimizer.zero_grad()

        while step < self.config.max_steps:
            # --- Data Fetch Timing ---
            t0 = time.perf_counter()
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                batch = next(data_iter)
            timing_data_fetch += time.perf_counter() - t0

            # Manual batch device move when DeepSpeed (train_loader was NOT prepared).
            # Non-DeepSpeed path: Accelerator's IterableDatasetShard auto-moves tensors.
            # send_to_device recurses into nested dicts/lists/tuples so non-image
            # modalities (graph adjacency lists, time-series covariates) reach xpu:0.
            if self._use_deepspeed:
                batch = send_to_device(batch, self._device)

            if "text" not in batch:
                self.accelerator.print("Error: 'text' missing in batch for Zone A training.")
                break

            # Rank-0-only counts; scaled by world_size in the perf record below.
            if self.accelerator.is_main_process:
                try:
                    tokens_per_window += batch_token_count(batch, pad_id=_pad_id_for_tokens)
                    last_batch_modality_counts = batch_modality_counts(batch)
                    # Per-sample non-pad lengths for IsoFLOP sequence-stat
                    # bookkeeping. Use the existing tokenizer pad id; when
                    # unknown, count every token (matches batch_token_count
                    # fallback).
                    text_tensor = batch.get("text")
                    if isinstance(text_tensor, torch.Tensor) and text_tensor.dim() == 2:
                        if _pad_id_for_tokens is None:
                            per_sample = [int(text_tensor.shape[1])] * int(
                                text_tensor.shape[0]
                            )
                        else:
                            per_sample = (
                                (text_tensor != _pad_id_for_tokens)
                                .sum(dim=1)
                                .tolist()
                            )
                        seq_lens_window.extend(int(n) for n in per_sample)
                except Exception as _e_perf:  # noqa: BLE001
                    self.accelerator.print(
                        f"[perf] modality/token count skipped: {_e_perf}"
                    )

            # Monitor Data Starvation
            self.monitor.check(batch, step)

            # Watchdog (Start and Intervals)
            if step == 0 or step == 1 or step % 100 == 0:
                self.check_parameter_status(step)
                # Dtype Check
                p = next(self.model.parameters())
                self.accelerator.print(
                    f"[Step {step}] Model First Parameter Dtype: {p.dtype}, Device: {p.device}"
                )

            did_sync = False
            with self.accelerator.accumulate(self.model):
                # --- Forward Timing ---
                t1 = time.perf_counter()
                with self.accelerator.autocast():
                    logits, loss = self.model(batch, labels=batch["text"])
                # Sync for accurate timing on XPU (skipped under PRISM_PRODUCTION_MODE)
                if not _production_mode and hasattr(torch, "xpu") and torch.xpu.is_available():
                    torch.xpu.synchronize()
                timing_forward += time.perf_counter() - t1

                # NaN TRAP (Data Root Cause)
                if torch.isnan(loss):
                    self.accelerator.print(f"\nCRITICAL: NaN Loss Detected at Step {step}")
                    if "_metadata" in batch:
                        # _metadata is a list of strings (collated)
                        self.accelerator.print(
                            f"Offending Batch Metadata: {batch['_metadata']}"
                        )
                    else:
                        self.accelerator.print(
                            "No metadata found in batch (Was it stripped by collator?)."
                        )
                        self.accelerator.print(
                            "No metadata found in batch (Was it stripped by collator?)."
                        )
                    # Dump batch keys?
                    # self.accelerator.print(f"Batch Keys: {batch.keys()}")
                    raise ValueError("Training Diverged (NaN) - See Log for Metadata")

                # --- Backward Timing (separating compute from AllReduce) ---
                # DEBUG_NO_SYNC: Run backward WITHOUT DDP AllReduce to isolate pure compute time
                debug_no_sync = os.environ.get("DEBUG_NO_SYNC", "0") == "1"

                t2 = time.perf_counter()

                if debug_no_sync:
                    # Measure PURE backward compute (no DDP AllReduce triggered)
                    # This helps diagnose if slowdown is in compute or communication
                    with self.accelerator.no_sync(self.model):
                        self.accelerator.backward(loss)
                    # Sync XPU to get accurate timing (skipped under PRISM_PRODUCTION_MODE)
                    if not _production_mode and hasattr(torch, "xpu") and torch.xpu.is_available():
                        torch.xpu.synchronize()
                    t_after_backward = time.perf_counter()
                    timing_backward_compute += t_after_backward - t2
                    timing_allreduce_wait += 0  # No AllReduce in debug mode

                    # Log bare backward time every 10 steps for debugging
                    if step % 10 == 0 and self.accelerator.is_main_process:
                        self.accelerator.print(
                            f"[DEBUG_NO_SYNC Step {step}] Bare Backward: {t_after_backward - t2:.3f}s"
                        )
                else:
                    # Normal mode: backward triggers DDP AllReduce
                    self.accelerator.backward(loss)

                    # Sync to complete backward compute before timing AllReduce
                    # (skipped under PRISM_PRODUCTION_MODE — overlapped backward+AR is the
                    # behavior we actually want to measure for fair benchmarking).
                    if not _production_mode and hasattr(torch, "xpu") and torch.xpu.is_available():
                        torch.xpu.synchronize()
                    t_after_backward = time.perf_counter()
                    timing_backward_compute += t_after_backward - t2

                    # Now wait for AllReduce to complete (DDP overlaps but we measure the wait).
                    # Skipped under PRISM_PRODUCTION_MODE.
                    if not _production_mode and hasattr(torch.distributed, "barrier") and torch.distributed.is_initialized():
                        torch.distributed.barrier()
                    t_after_allreduce = time.perf_counter()
                    timing_allreduce_wait += t_after_allreduce - t_after_backward

                # Gradient Clipping & Optimizer Step (only on sync steps)
                if self.accelerator.sync_gradients:
                    # Gradient Clipping (Molmo Separate + Global fallback)
                    is_molmo = getattr(self.config, "scheduler_type", "") == "molmo_layered"

                    # Unwrap model to access component attributes (DDP/DeepSpeed/FSDP wraps the model)
                    unwrapped_model = self._unwrap_model()
                    is_timeseries = (
                        unwrapped_model.config.is_timeseries
                        if hasattr(unwrapped_model.config, "is_timeseries")
                        else False
                    )

                    if is_molmo or is_timeseries:
                        # Separate Clipping
                        # 1. Projectors (Connectors)
                        if unwrapped_model.projectors:
                            self.accelerator.clip_grad_norm_(unwrapped_model.projectors.parameters(), 1.0)

                        # 2. Encoders (ViT) - Only if trainable
                        if not getattr(self.config, "freeze_vit", True) and unwrapped_model.encoders:
                            self.accelerator.clip_grad_norm_(unwrapped_model.encoders.parameters(), 1.0)

                        # 3. Backbone (LLM) - Only if trainable
                        if not getattr(self.config, "freeze_llm", True) and unwrapped_model.backbone:
                            self.accelerator.clip_grad_norm_(unwrapped_model.backbone.parameters(), 1.0)
                    else:
                        # Legacy Global Clipping
                        max_grad_norm = getattr(self.config, "max_grad_norm", 1.0)
                        if max_grad_norm > 0:
                            self.accelerator.clip_grad_norm_(self.model.parameters(), max_grad_norm)

                # Explicit sync to measure wait time (skipped under PRISM_PRODUCTION_MODE)
                t_sync_start = time.perf_counter()
                if not _production_mode and hasattr(torch, "xpu") and torch.xpu.is_available():
                    torch.xpu.synchronize()
                timing_sync_wait += time.perf_counter() - t_sync_start
                timing_backward += time.perf_counter() - t2

                micro_batch_count += 1

                # --- Optimizer Step Timing ---
                if self.accelerator.sync_gradients:
                    t3 = time.perf_counter()
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    # Skipped under PRISM_PRODUCTION_MODE
                    if not _production_mode and hasattr(torch, "xpu") and torch.xpu.is_available():
                        torch.xpu.synchronize()
                    timing_optimizer += time.perf_counter() - t3
                    did_sync = True

            if not did_sync:
                continue

            timing_steps += 1

            # --- Log Timing Every 50 Steps (or on the final step) ---
            # Also fire on `step + 1 == max_steps` so short runs (e.g. the
            # per-modality smoke at max_steps=50) emit at least one perf
            # record. Without this, `step % 50 == 0 and step > 0` never
            # fires when max_steps==50 because the loop exits at step 49.
            is_final_step = (step + 1) >= self.config.max_steps
            if (
                (step % 50 == 0 and step > 0) or is_final_step
            ) and self.accelerator.is_main_process:
                avg_data = timing_data_fetch / timing_steps
                avg_fwd = timing_forward / timing_steps
                avg_bwd = timing_backward / timing_steps
                avg_bwd_compute = timing_backward_compute / timing_steps
                avg_allreduce = timing_allreduce_wait / timing_steps
                avg_opt = timing_optimizer / timing_steps
                avg_sync = timing_sync_wait / timing_steps
                total = avg_data + avg_fwd + avg_bwd + avg_opt

                # Per micro-batch metrics
                micro_batches_per_step = micro_batch_count / timing_steps if timing_steps > 0 else 0
                time_per_micro = (
                    total / micro_batches_per_step if micro_batches_per_step > 0 else total
                )

                # Wall clock time for the 50 steps
                wall_time_50_steps = time.perf_counter() - step_start_time
                step_start_time = time.perf_counter()

                # Calculate throughput metrics
                iter_per_sec = 1.0 / total if total > 0 else 0
                world_size = self.accelerator.num_processes
                global_batch_size = self.config.batch_size * world_size
                samples_per_sec = global_batch_size / total if total > 0 else 0

                # XPU Memory Utilization
                mem_allocated_mb = 0.0
                mem_reserved_mb = 0.0
                if hasattr(torch, "xpu") and torch.xpu.is_available():
                    mem_allocated_mb = torch.xpu.memory_allocated() / 1024.0**2
                    mem_reserved_mb = torch.xpu.memory_reserved() / 1024.0**2
                elif torch.cuda.is_available():
                    mem_allocated_mb = torch.cuda.memory_allocated() / 1024.0**2
                    mem_reserved_mb = torch.cuda.memory_reserved() / 1024.0**2

                self.accelerator.print(
                    f"[TIMING Step {step}] Total: {total:.2f}s | "
                    f"Data: {avg_data:.2f}s ({100 * avg_data / total:.1f}%) | "
                    f"Fwd: {avg_fwd:.2f}s ({100 * avg_fwd / total:.1f}%) | "
                    f"Bwd: {avg_bwd:.2f}s ({100 * avg_bwd / total:.1f}%) | "
                    f"Sync: {avg_sync * 1000:.0f}ms | "
                    f"Throughput: {samples_per_sec:.1f} samp/s | "
                    f"Mem: {mem_allocated_mb:.0f}MB"
                )

                if self.accelerator.is_main_process:
                    # Scale rank-0 token count by world_size to mirror
                    # samples_per_sec aggregation.
                    global_tokens_window = tokens_per_window * world_size
                    tokens_per_sec = (
                        global_tokens_window / (total * timing_steps)
                        if total > 0 and timing_steps > 0
                        else 0.0
                    )
                    tokens_per_batch = (
                        global_tokens_window / timing_steps
                        if timing_steps > 0
                        else 0.0
                    )
                    # IsoFLOP per-step extensions: sequence-length distribution
                    # + analytic FLOP accounting + projector capacity knobs.
                    # The flop counter accumulates per *throughput-window flush*
                    # (one call per perf record); pass `timing_steps` so
                    # cumulative_flops tracks actual budget consumption — a
                    # bare `.step()` would undercount by the window size (see
                    # PR feedback on PR #98). When uncalibrated, both values
                    # come out None.
                    _seq_stats = sequence_stats(seq_lens_window)
                    _fps_window, _cum_flops = self._flop_counter.step(n_steps=timing_steps)
                    _unwrapped_cfg = getattr(self._unwrap_model(), "config", None)
                    log_perf_record(
                        getattr(self.config, "output_dir", None),
                        {
                            "site": "trainer_zone_a",
                            "step": step,
                            "samples_per_sec": samples_per_sec,
                            "tokens_per_sec": tokens_per_sec,
                            "tokens_per_batch": tokens_per_batch,
                            "batch_modality_counts": dict(last_batch_modality_counts),
                            "total_step_s": total,
                            "data_s": avg_data,
                            "fwd_s": avg_fwd,
                            "bwd_s": avg_bwd,
                            "world_size": world_size,
                            "batch_size": self.config.batch_size,
                            "global_batch_size": global_batch_size,
                            "dist_strategy": os.environ.get("DIST_STRATEGY", "ddp").lower(),
                            "mem_allocated_mb": mem_allocated_mb,
                            "mem_reserved_mb": mem_reserved_mb,
                            "sweep_id": getattr(self.config, "sweep_id", None),
                            "preset": getattr(self.config, "preset", None),
                            "modalities": model_modalities(self._unwrap_model()),
                            "seq_p50": _seq_stats["seq_p50"],
                            "seq_p95": _seq_stats["seq_p95"],
                            "seq_p99": _seq_stats["seq_p99"],
                            "seq_max": _seq_stats["seq_max"],
                            "padding_ratio": _seq_stats["padding_ratio"],
                            "flops_per_step": _fps_window,
                            "cumulative_flops": _cum_flops,
                            "projector_hidden_mult": (
                                getattr(_unwrapped_cfg, "projector_hidden_mult", 1)
                                if _unwrapped_cfg is not None
                                else 1
                            ),
                            "projector_num_layers": (
                                getattr(_unwrapped_cfg, "projector_num_layers", 2)
                                if _unwrapped_cfg is not None
                                else 2
                            ),
                        },
                    )
                self.accelerator.print(
                    f"       Bwd Breakdown: Compute={avg_bwd_compute:.2f}s ({100 * avg_bwd_compute / avg_bwd:.1f}%) | "
                    f"AllReduce={avg_allreduce:.2f}s ({100 * avg_allreduce / avg_bwd:.1f}%)"
                )
                self.accelerator.print(
                    f"       μBatch: {micro_batches_per_step:.1f}/step, {time_per_micro * 1000:.0f}ms/μbatch | "
                    f"Wall: {wall_time_50_steps:.1f}s/50steps"
                )

                # Log timing metrics to WandB
                if wandb is not None and self.accelerator.is_main_process:
                    wandb.log(
                        {
                            "perf/iter_per_sec": iter_per_sec,
                            "perf/samples_per_sec": samples_per_sec,
                            "perf/time_per_step_s": total,
                            "perf/time_per_micro_batch_ms": time_per_micro * 1000,
                            "perf/micro_batches_per_step": micro_batches_per_step,
                            "perf/data_time_s": avg_data,
                            "perf/forward_time_s": avg_fwd,
                            "perf/backward_time_s": avg_bwd,
                            "perf/backward_compute_s": avg_bwd_compute,
                            "perf/allreduce_wait_s": avg_allreduce,
                            "perf/sync_time_ms": avg_sync * 1000,
                            "perf/optimizer_time_s": avg_opt,
                            "perf/data_pct": 100 * avg_data / total,
                            "perf/forward_pct": 100 * avg_fwd / total,
                            "perf/backward_pct": 100 * avg_bwd / total,
                            "perf/optimizer_pct": 100 * avg_opt / total,
                            "perf/world_size": world_size,
                            "perf/global_batch_size": global_batch_size,
                            "perf/mem_allocated_mb": mem_allocated_mb,
                            "perf/mem_reserved_mb": mem_reserved_mb,
                            "perf/wall_time_50_steps_s": wall_time_50_steps,
                        },
                        step=step,
                    )

                # Reset accumulators
                timing_data_fetch = timing_forward = timing_backward = timing_optimizer = (
                    timing_sync_wait
                ) = 0.0
                timing_backward_compute = timing_allreduce_wait = 0.0
                timing_steps = 0
                micro_batch_count = 0
                tokens_per_window = 0
                seq_lens_window = []

            # Logging & Visualization Consolidation
            log_data = {}
            do_log = False

            # 1. Loss Logging (0 disables — guard against ZeroDivisionError).
            log_interval = self.config.log_every_n_steps
            if log_interval and step % log_interval == 0:
                avg_loss = loss.item()
                lr = self.scheduler.get_last_lr()[0]
                log_data.update(
                    {"loss": avg_loss, "lr": lr}
                )  # "step": step is handled by accelerator kwarg
                do_log = True

                if self.accelerator.is_main_process:
                    progress_bar.set_postfix({"loss": avg_loss})
                    print(f"Step {step}: Loss {avg_loss:.4f}", flush=True)

            # 2. Visualization Logging (0 disables — guard against ZeroDivisionError).
            viz_interval = self.config.viz_every_n_steps
            if viz_interval and step > 0 and step % viz_interval == 0:
                if self.accelerator.is_main_process:
                    try:
                        viz_results = self.visualizer.visualize(batch)

                        # Prediction Monitoring
                        try:
                            pred_table = self.visualizer.visualize_predictions(
                                batch,
                                self.model,
                                modality="time_series" if is_timeseries else "image",
                            )
                            if pred_table:
                                viz_results["viz/predictions"] = pred_table
                        except Exception as e_gen:
                            print(f"  [Viz] Error generating predictions: {e_gen}")

                        if viz_results:
                            log_data.update(viz_results)
                            do_log = True
                            # print(f"  [Viz] Generated batch visualization for step {step}")
                    except Exception as e:
                        print(f"  [Viz] Error visualizing batch: {e}")

            # 3. Commit Logs
            if do_log and log_data:
                self.accelerator.log(log_data, step=step)

            # Optional periodic evaluation. Default off — historical runs had
            # OOM problems and the block was disabled inline. IsoFLOP launches
            # opt-in via `training.eval_enabled=true training.eval_every_n_steps=N`.
            eval_enabled = getattr(self.config, "eval_enabled", False)
            eval_interval = getattr(self.config, "eval_every_n_steps", 0)
            if eval_enabled and eval_interval and step > 0 and step % eval_interval == 0:
                if getattr(self.config, "bioreason_dataset", None) == "wanglab/kegg":
                    self.run_kegg_eval(step)
                else:
                    self.run_evaluation(step)
                self.model.train()  # Switch back to train mode

            # Checkpointing (0 disables — guard against ZeroDivisionError).
            save_interval = getattr(self.config, "save_every_n_steps", 1000)
            if save_interval and step > 0 and step % save_interval == 0:
                self.save_checkpoint(step)

            step += 1
            unwrapped_model.global_step = step
            if self.accelerator.is_main_process:
                progress_bar.update(1)

        # Save Final Checkpoint
        self.save_checkpoint("final")
        self.accelerator.print("Training Complete.")
        self.accelerator.print(f"Checkpoints saved to: {self.config.output_dir}")
        self.accelerator.end_training()

    def run_evaluation(self, step):
        """Run all registered evaluators and log metrics."""
        if not self.accelerator.is_main_process:
            return

        self.accelerator.print(f"\n[Step {step}] Running Evaluation...")
        self.model.eval()

        total_metrics = {}
        for evaluator in self.evaluators:
            try:
                name = evaluator.__class__.__name__
                metrics = evaluator.evaluate(limit=10)  # Small batch for sanity
                # Prefix metrics
                for k, v in metrics.items():
                    total_metrics[f"eval/{name}/{k}"] = v
            except Exception as e:
                self.accelerator.print(f"Eval Error ({name}): {e}")

        self.accelerator.log(total_metrics, step=step)
        self.accelerator.print(f"Eval Metrics: {total_metrics}")

        # IsoFLOP per-family eval-loss records. The fitter keys off
        # `(family, last loss)`, so emit one perf record per active modality
        # whose name appears in a `loss` metric. Best-effort: if no
        # evaluator reports a per-family loss, the collector treats the
        # row's `loss_<family>` as missing.
        try:
            modalities = (
                model_modalities(self._unwrap_model()) or []
            )
            for family in modalities:
                # Pick the first metric whose lowercased key contains the
                # modality name AND the substring "loss" — accommodates
                # evaluators that prefix with class names (eval/VQAv2/loss).
                family_lc = family.lower()
                match_val: float | None = None
                for k, v in total_metrics.items():
                    k_lc = k.lower()
                    if family_lc in k_lc and "loss" in k_lc:
                        try:
                            match_val = float(v)
                            break
                        except (TypeError, ValueError):
                            continue
                if match_val is None:
                    continue
                log_perf_record(
                    getattr(self.config, "output_dir", None),
                    {
                        "event": "eval",
                        "site": "trainer_zone_a",
                        "step": step,
                        "family": family,
                        "loss": match_val,
                        # `held_out_eval` is the real evaluator suite (this
                        # trainer's run_evaluation calls registered evaluators
                        # against held-out data). Counterpart in trainer_native
                        # stamps "train_running_mean" because the native trainer
                        # has no evaluator wiring yet. Collector surfaces this
                        # into experiments.csv:loss_source so the fit can tell
                        # the two apart.
                        "loss_source": "held_out_eval",
                        "sweep_id": getattr(self.config, "sweep_id", None),
                        "preset": getattr(self.config, "preset", None),
                    },
                )
        except Exception as _e_eval_perf:  # noqa: BLE001
            self.accelerator.print(
                f"[perf] eval per-family loss record skipped: {_e_eval_perf}"
            )

    def run_kegg_eval(self, step):
        """
        Run inference over the wanglab/kegg 'val' split and log accuracy.

        Reuses eval_kegg.py's prefix-mode inference path (truncate_dna,
        run_inference, compute_metrics) so periodic in-training numbers match
        the offline `python eval_kegg.py` protocol exactly.
        """
        if not self.accelerator.is_main_process:
            return

        from eval_kegg import (  # noqa: F401 (truncate_dna used internally by run_inference)
            compute_metrics,
            run_inference,
        )

        unwrapped = self._unwrap_model()

        max_examples = getattr(self.config, "kegg_eval_max_examples", None)
        if self._kegg_eval_dataset is None:
            import logging as _logging

            from datasets import load_dataset
            from transformers import AutoTokenizer

            self.accelerator.print("[KEGG Eval] Loading wanglab/kegg 'val' split...")
            ds = load_dataset("wanglab/kegg", "default")["val"]
            if max_examples is not None:
                ds = ds.select(range(min(max_examples, len(ds))))
            self._kegg_eval_dataset = ds
            self._kegg_eval_logger = _logging.getLogger("kegg_eval")

            # Mirrors eval_kegg.py's load_model(): DNAEncoder has no `.tokenizer`
            # attribute in the current architecture (tokenization now happens in
            # the dataset's dna_tokenizer, see multimodal.py) — getattr(..., None)
            # keeps this safe and falls through to loading a fresh NT tokenizer,
            # same as BioReason's own defensive pattern already did.
            dna_encoder = unwrapped.encoders["dna"] if "dna" in unwrapped.encoders else None
            if dna_encoder is not None and getattr(dna_encoder, "tokenizer", None) is not None:
                self._kegg_eval_nt_tokenizer = dna_encoder.tokenizer
            else:
                self._kegg_eval_nt_tokenizer = AutoTokenizer.from_pretrained(
                    "InstaDeepAI/nucleotide-transformer-v2-250m-multi-species",
                    trust_remote_code=True,
                )

        self.accelerator.print(
            f"\n[Step {step}] Running KEGG Eval ({len(self._kegg_eval_dataset)} val examples)..."
        )
        self.model.eval()
        device = self.accelerator.device
        dna_truncation_per_side = getattr(self.config, "dna_truncation_per_side", 1024)
        max_dna_length = getattr(self.config, "max_dna_length", 1024)
        interleaved = bool(getattr(unwrapped.config, "is_interleaved_qa", False))

        results = []
        for ex in self._kegg_eval_dataset:
            try:
                # Training steps run inside accelerator.autocast(), which
                # transparently reconciles the fp32-kept DNA encoder .proj layer
                # against the bf16 ModalityProjector/backbone. generate() has no
                # such wrapper by default, so without this the DNA .proj -> fc1
                # matmul raises a dtype mismatch on every example.
                with self.accelerator.autocast():
                    result = run_inference(
                        ex,
                        unwrapped,
                        unwrapped.backbone_tokenizer,
                        self._kegg_eval_nt_tokenizer,
                        device,
                        dna_truncation_per_side,
                        max_dna_length,
                        max_new_tokens=256,
                        do_sample=False,
                        interleaved=interleaved,
                    )
            except Exception as e:
                self.accelerator.print(f"[KEGG Eval] Example failed: {e}")
                continue
            results.append(result)

        if not results:
            self.accelerator.print("[KEGG Eval] No examples produced a result; skipping metrics.")
            self.model.train()
            return

        import pandas as pd

        df = pd.DataFrame(results)
        metrics = compute_metrics(df, self._kegg_eval_logger)

        log_data = {"kegg_eval/accuracy": metrics["accuracy"], "kegg_eval/n": metrics["n"]}
        if "f1" in metrics:
            log_data.update(
                {
                    "kegg_eval/precision": metrics["precision"],
                    "kegg_eval/recall": metrics["recall"],
                    "kegg_eval/f1": metrics["f1"],
                }
            )
        if wandb is not None and wandb.run is not None and self.accelerator.is_main_process:
            wandb.log(log_data, step=step)
        self.accelerator.print(f"[KEGG Eval] {log_data}")
        self.model.train()

    def save_checkpoint(self, step):
        """Save model checkpoint via Accelerator, plus LoRA adapter weights when enabled."""
        output_dir = f"{self.config.output_dir}/step_{step}"
        self.accelerator.print(f"\n[Step {step}] Saving Checkpoint to {output_dir}...")
        self.accelerator.save_state(output_dir)

        # Save LoRA adapter weights separately (torchtune format: plain state-dict .pt).
        # These can be reloaded with load_lora_adapter() without needing the full checkpoint.
        if getattr(self.config, "lora_enabled", False) and self.accelerator.is_main_process:
            from src.utils.lora_utils import save_lora_adapter

            unwrapped = self.accelerator.unwrap_model(self.model)
            backbone = getattr(unwrapped, "backbone", None)
            if backbone is not None:
                lora_path = f"{output_dir}/lora_adapter.pt"
                save_lora_adapter(
                    backbone,
                    lora_path,
                    rank=self.config.lora_r,
                    alpha=self.config.lora_alpha,
                    target_modules=getattr(self.config, "lora_target_modules", None),
                )
                self.accelerator.print(f"torchtune LoRA adapter saved to {lora_path}")

        # Save Metadata for easy restart (Step, Config?)
        import json

        with open(f"{output_dir}/training_state.json", "w") as f:
            json.dump({"step": step, "config": self.config.__dict__}, f, default=str)

        self.accelerator.print("Checkpoint Saved.")

    # _to_device helper removed as Accelerator handles data prep
