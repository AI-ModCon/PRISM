import json
import os
import time

import torch
import torch.optim as optim
from accelerate import Accelerator, DataLoaderConfiguration, DistributedDataParallelKwargs
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from src.utils.perf_log import log_perf_record, model_modalities


class ZoneAVLATrainer:
    """Zone A trainer for CALVIN-style VLA regression."""

    def __init__(self, model, config, train_loader):
        self.model = model
        self.config = config
        self.train_loader = train_loader

        ddp_kwargs = DistributedDataParallelKwargs(
            find_unused_parameters=False,
            broadcast_buffers=False,
        )
        self.accelerator = Accelerator(
            mixed_precision="bf16",
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            log_with="wandb" if config.wandb_project else None,
            dataloader_config=DataLoaderConfiguration(dispatch_batches=False),
            kwargs_handlers=[ddp_kwargs],
        )

        if self.model.backbone is None:
            raise RuntimeError("VLA mode requires an HF LLM backbone")
        if not hasattr(self.model, "pose_embed") or not hasattr(self.model, "action_head"):
            raise RuntimeError("VLA modules are missing from model")

        target_dtype = next(self.model.backbone.parameters()).dtype
        self.model.encoders.to(dtype=target_dtype)
        self.model.projectors.to(dtype=target_dtype)
        self.model.pose_embed.to(dtype=target_dtype)
        self.model.action_head.to(dtype=target_dtype)

        for param in self.model.parameters():
            param.requires_grad = False

        self.trainable_params = []
        for name, param in self.model.named_parameters():
            if (
                name.startswith("projectors.")
                or name.startswith("pose_embed.")
                or name.startswith("action_head.")
                or name == "pose_modality_embedding"
            ):
                param.requires_grad = True
                self.trainable_params.append(param)

        if not self.trainable_params:
            raise RuntimeError("No trainable params found for VLA Zone A")

        self.optimizer = optim.AdamW(
            self.trainable_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=config.max_steps,
        )

        self.model, self.optimizer, self.train_loader = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.train_loader,
        )
        self.resume_step = 0
        if config.resume_from_checkpoint:
            self.accelerator.print(
                f"\n{'='*40}\n RESUMING VLA TRAINING FROM:\n {config.resume_from_checkpoint}\n{'='*40}"
            )
            self.accelerator.load_state(config.resume_from_checkpoint)
            state_file = os.path.join(config.resume_from_checkpoint, "training_state.json")
            if os.path.exists(state_file):
                with open(state_file) as f:
                    state_data = json.load(f)
                saved_step = state_data.get("step", 0)
                if isinstance(saved_step, int):
                    self.resume_step = saved_step
                elif isinstance(saved_step, str) and saved_step.isdigit():
                    self.resume_step = int(saved_step)
            if self.resume_step == 0:
                dirname = os.path.basename(config.resume_from_checkpoint.rstrip("/"))
                if dirname.startswith("step_"):
                    maybe_step = dirname.split("_", 1)[1]
                    if maybe_step.isdigit():
                        self.resume_step = int(maybe_step)
            self.accelerator.print(f"Resumed at step {self.resume_step}")
        self.scheduler = self.accelerator.prepare(self.scheduler)
        current_sched_step = (
            self.scheduler.scheduler.last_epoch
            if hasattr(self.scheduler, "scheduler")
            else self.scheduler.last_epoch
        )
        if self.resume_step > 0 and current_sched_step <= 0:
            self.accelerator.print(
                f"[Scheduler] Fast-forwarding scheduler from 0 to {self.resume_step}"
            )
            for _ in range(self.resume_step):
                self.scheduler.step()

        if config.wandb_project and self.accelerator.is_main_process:
            wandb_kwargs = {"name": config.wandb_run_name}
            if config.wandb_entity:
                wandb_kwargs["entity"] = config.wandb_entity
            if config.wandb_mode:
                wandb_kwargs["mode"] = config.wandb_mode
            self.accelerator.init_trackers(
                project_name=config.wandb_project,
                config=config.__dict__,
                init_kwargs={"wandb": wandb_kwargs},
            )

    def train(self):
        self.model.train()
        step = int(self.resume_step)
        data_iter = iter(self.train_loader)
        self.optimizer.zero_grad()

        progress = None
        if self.accelerator.is_main_process:
            progress = tqdm(range(self.config.max_steps), desc="Training Zone A VLA")
            if step > 0:
                progress.update(step)

        # Per-log-window timing accumulators (mirror trainer_native's rolling
        # window). Reset every log_every_n_steps so samples_per_sec reflects
        # recent throughput, not a since-start mean.
        window_start = time.perf_counter()
        window_steps = 0

        while step < self.config.max_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                batch = next(data_iter)

            did_sync = False
            with self.accelerator.accumulate(self.model):
                with self.accelerator.autocast():
                    pred_action, loss, per_dim_mse = self.model(batch)

                if torch.isnan(loss):
                    raise RuntimeError(
                        f"NaN VLA loss at step={step}, metadata={batch.get('_metadata')}"
                    )

                self.accelerator.backward(loss)
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.trainable_params, 1.0)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    did_sync = True

            if not did_sync:
                continue

            window_steps += 1

            if step % self.config.log_every_n_steps == 0:
                log_data = {
                    "loss": float(loss.item()),
                    "lr": float(self.scheduler.get_last_lr()[0]),
                }
                per_dim_mse_list = per_dim_mse.detach().float().tolist()
                for dim_idx, dim_mse in enumerate(per_dim_mse_list):
                    log_data[f"mse_dim/{dim_idx}"] = float(dim_mse)
                self.accelerator.log(log_data, step=step)
                if self.accelerator.is_main_process:
                    print(f"Step {step}: loss={loss.item():.6f}", flush=True)

                    # perf.jsonl row, rank-0 only (matches trainer_native's
                    # gate). `getattr` defaults keep this safe under the
                    # synthetic SimpleNamespace configs used by tests.
                    # `dispatch_batches=False` (above) means batch_size is
                    # per-device, so the global-sample multiplier is
                    # batch_size * grad_accum * world_size.
                    world_size = max(int(getattr(self.accelerator, "num_processes", 1)), 1)
                    batch_size = int(getattr(self.config, "batch_size", 1))
                    grad_accum = int(
                        getattr(self.config, "gradient_accumulation_steps", 1)
                    )
                    samples_per_step = batch_size * grad_accum * world_size
                    window_wall = time.perf_counter() - window_start
                    samples_per_sec = (
                        (samples_per_step * window_steps) / window_wall
                        if window_wall > 0
                        else 0.0
                    )
                    action_mse_mean = (
                        float(sum(per_dim_mse_list)) / len(per_dim_mse_list)
                        if per_dim_mse_list
                        else 0.0
                    )
                    perf_record: dict = {
                        "site": "trainer_zone_a_vla_per_log",
                        "step": int(step),
                        "loss": float(loss.item()),
                        "samples_per_sec": float(samples_per_sec),
                        "samples_per_step": int(samples_per_step),
                        "world_size": int(world_size),
                        "batch_size": batch_size,
                        "grad_accum": grad_accum,
                        "dist_strategy": "accelerate_ddp",  # VLA is Accelerate-only
                        "sweep_id": getattr(self.config, "sweep_id", None),
                        "preset": getattr(self.config, "preset", None),
                        "modalities": model_modalities(self.model),
                        "task": "vla_calvin",
                        # CALVIN action is a 7-dim regression head (see
                        # info.json features.action). Log per-dim so the
                        # aggregator doesn't collapse dimensionality.
                        "action_mse_per_dim": per_dim_mse_list,
                        "action_mse": action_mse_mean,
                        "calvin_loader": getattr(self.config, "calvin_loader", "map"),
                    }
                    if hasattr(torch, "xpu") and torch.xpu.is_available():
                        try:
                            perf_record["peak_gb"] = (
                                torch.xpu.max_memory_allocated() / 1e9
                            )
                            perf_record["current_gb"] = (
                                torch.xpu.memory_allocated() / 1e9
                            )
                        except Exception:  # noqa: BLE001
                            pass

                    log_perf_record(
                        getattr(self.config, "output_dir", None), perf_record
                    )

                # Reset the rolling window regardless of rank so all ranks
                # measure the same wall-clock interval next round.
                window_start = time.perf_counter()
                window_steps = 0

            save_interval = getattr(self.config, "save_every_n_steps", 1000)
            if step > 0 and step % save_interval == 0:
                self.save_checkpoint(step)

            step += 1
            if progress is not None:
                progress.update(1)

        self.save_checkpoint("final")
        if self.accelerator.is_main_process:
            print("Training Complete.", flush=True)
            print(f"Checkpoints saved to: {self.config.output_dir}", flush=True)
        self.accelerator.end_training()

    def save_checkpoint(self, step):
        output_dir = f"{self.config.output_dir}/step_{step}"
        if self.accelerator.is_main_process:
            print(f"\n[Step {step}] Saving Checkpoint to {output_dir}...", flush=True)
        self.accelerator.wait_for_everyone()
        self.accelerator.save_state(output_dir)
        if self.accelerator.is_main_process:
            with open(f"{output_dir}/training_state.json", "w") as f:
                json.dump({"step": step, "config": self.config.__dict__}, f, default=str)
            print("Checkpoint Saved.", flush=True)
        self.accelerator.wait_for_everyone()
