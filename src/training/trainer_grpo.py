"""
Zone D: GRPO (Group Relative Policy Optimization) Trainer

Implements reinforcement learning for biological reasoning using BioReason's
approach: generate multiple completions per prompt, score them with reward
functions, and optimize the policy using group-relative advantages.

Based on: https://arxiv.org/abs/2505.23579 (BioReason)
"""

import copy
import json
import logging
import os
import re

import torch
import torch.nn.functional as F
import torch.optim as optim

# GRPOSimpleLoss: validated GRPO loss/KL module from n-getty/torchtune-aurora
# (torchtune/dev/rl/loss.py), run at 32B/16-node scale on Aurora. Its k3 KL
# estimator matches this file's own (independently re-derived, same
# "BioReason 2048-gen step 4" NaN failure mode) almost line for line -- using
# the validated implementation directly here instead of maintaining a
# parallel copy that could silently diverge from upstream fixes. Only the
# loss/KL module is imported; rollout generation, advantage normalization
# (this file's is per-prompt-group; torchtune's dev.rl default is
# batch-level -- a different training semantic, not swapped in), and the
# vLLM/FSDP2/distributed machinery stay out of scope for this DNA-specific,
# single-process trainer.
from torchtune.dev.rl.loss import GRPOSimpleLoss
from tqdm import tqdm

try:
    import wandb
except Exception:
    wandb = None

from src.config import TrainingConfig
from src.model import UnifiedTransformer

logger = logging.getLogger(__name__)


# ============================================================
# Reward Functions (from BioReason)
# ============================================================

def xmlcount_reward(completion: str, **kwargs) -> float:
    """Reward for proper tag usage in reasoning output.

    Expected format emitted by _render_dna_prompt_text (multimodal.py):
        <think>\\n{reasoning}\\n</think>\\n\\n<answer>{text}</answer>
    """
    score = 0.0
    if "<think>" in completion:
        score += 0.125
    if "</think>" in completion:
        score += 0.125
    if "<answer>" in completion:
        score += 0.125
    if completion.count("<think>") == 1 and completion.count("</think>") == 1:
        score += 0.125
    if completion.count("<answer>") == 1 and completion.count("</answer>") == 1:
        score += 0.125
    # Bonus for answer block appearing after the closing think tag
    if re.search(r"</think>.*?<answer>", completion, re.DOTALL):
        score += 0.25
    return score


def soft_format_reward(completion: str, **kwargs) -> float:
    """Reward for loosely following the expected format.

    Matches: <think>...</think> followed by <answer>...</answer>
    """
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    match = re.search(pattern, completion, re.DOTALL)
    return 1.0 if match else 0.0


def strict_format_reward(completion: str, **kwargs) -> float:
    """Reward for strictly following the expected format (entire string).

    Matches: <think>\\n...\\n</think>\\n\\n<answer>text</answer>
    """
    pattern = r"^\s*<think>\n.*?\n</think>\n\n<answer>.+</answer>\s*$"
    match = re.match(pattern, completion, re.DOTALL)
    return 1.0 if match else 0.0


def concise_reward(completion: str, **kwargs) -> float:
    """Reward for concise answers (penalize overly long outputs)."""
    word_count = len(completion.split())
    if word_count < 10:
        return 0.0  # Too short
    elif word_count <= 300:
        return 1.0  # Good length
    elif word_count <= 600:
        return 0.5  # Acceptable
    else:
        return 0.0  # Too long


def correctness_reward(completion: str, answer: str = "", **kwargs) -> float:
    """Reward for containing the correct answer.

    Extracts the answer from <answer>...</answer> tags emitted by the model.
    Falls back to full completion if tags are absent.
    """
    if not answer:
        return 0.0

    # Extract text inside <answer>...</answer> — matches multimodal.py SFT format
    answer_match = re.search(r"<answer>(.*?)</answer>", completion, re.DOTALL)
    if answer_match:
        extracted = answer_match.group(1).strip().lower()
    else:
        extracted = completion.strip().lower()

    ground_truth = answer.strip().lower()

    # Exact match
    if extracted == ground_truth:
        return 1.0
    # Containment match
    if ground_truth in extracted:
        return 0.5
    return 0.0


REWARD_FUNCTIONS = {
    "xmlcount": xmlcount_reward,
    "soft_format": soft_format_reward,
    "strict_format": strict_format_reward,
    "concise": concise_reward,
    "correctness": correctness_reward,
}


# ============================================================
# GRPO Trainer
# ============================================================

class ZoneDTrainer:
    """
    Zone D: Group Relative Policy Optimization (GRPO) Trainer.

    For each prompt, generates multiple completions, scores them with
    reward functions, computes group-relative advantages, and updates
    the policy to maximize expected reward.
    """

    def __init__(
        self,
        model: UnifiedTransformer,
        config: TrainingConfig,
        train_loader,
        tokenizer=None,
        ref_model: UnifiedTransformer | None = None,
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.tokenizer = tokenizer
        _device_str = config.device
        if _device_str == "auto":
            if torch.cuda.is_available():
                _device_str = "cuda"
            elif hasattr(torch, "xpu") and torch.xpu.is_available():
                _device_str = "xpu"
            else:
                _device_str = "cpu"
        self.device = torch.device(_device_str)

        # GRPO requires interleaved mode so DNA embeddings are spliced inline.
        assert self.model.config.is_interleaved_qa, (
            "ZoneDTrainer requires is_interleaved_qa=True on the model config. "
            "DNA and language must be encoded separately and merged via the interleaved path."
        )

        # --- Freeze DNA encoder (never trained in GRPO) ---
        for param in self.model.encoders.parameters():
            param.requires_grad = False

        # --- Apply LoRA to backbone (mirrors Zone A) ---
        # Done before .to(device) so LoRALinear layers are created, then the single
        # .to(device+dtype) call below moves everything consistently.
        # Base weights are frozen by set_trainable_params inside apply_lora_torchtune;
        # only lora_a/lora_b remain requires_grad=True.
        lora_enabled = getattr(config, "lora_enabled", False)
        if lora_enabled and self.model.backbone is not None:
            from src.utils.lora_utils import apply_lora_torchtune, merge_lora_into_base

            # load_model_weights_only (called before ZoneDTrainer) only loads base
            # weights — the SFT lora_a/lora_b keys are flagged "unexpected" and
            # dropped, since LoRALinear layers don't exist yet at that point.
            # The SFT adapter was trained at a different rank than the GRPO adapter
            # (see bioreason_sft.yaml vs bioreason_grpo.yaml lora_r), so it can't be
            # loaded into the new rank's LoRALinear modules directly. Instead, fold
            # it into the frozen base nn.Linear weights now, before
            # apply_lora_torchtune creates the fresh GRPO-rank LoRA layers on top.
            if config.resume_weights_only:
                adapter_path = os.path.join(config.resume_weights_only, "lora_adapter.pt")
                if os.path.exists(adapter_path):
                    resume_lora_alpha = getattr(config, "resume_lora_alpha", config.lora_alpha)
                    merge_lora_into_base(self.model.backbone, adapter_path, alpha=resume_lora_alpha)
                    logger.info(f"[ZoneD] Merged SFT LoRA adapter from {adapter_path} into base weights (alpha={resume_lora_alpha})")
                elif getattr(config, "allow_missing_resume_adapter", False):
                    logger.warning(f"[ZoneD] no lora_adapter.pt found at {adapter_path}; starting from base weights only (allow_missing_resume_adapter=True)")
                else:
                    # Fail fast by default: resume_weights_only was set,
                    # meaning an SFT adapter was expected, so a missing
                    # adapter is very likely a typo'd path or an incomplete
                    # checkpoint -- silently starting GRPO from an
                    # untrained base is a costly failure to discover only
                    # after the fact, not something to warn-and-continue
                    # past. Set allow_missing_resume_adapter=True for the
                    # rare intentional case of starting GRPO from base
                    # weights with no prior SFT adapter.
                    raise FileNotFoundError(
                        f"resume_weights_only={config.resume_weights_only!r} was set but no "
                        f"lora_adapter.pt found at {adapter_path}. Refusing to silently start "
                        f"GRPO from an untrained base. Set training.allow_missing_resume_adapter="
                        f"true to override."
                    )

            apply_lora_torchtune(
                self.model.backbone,
                rank=config.lora_r,
                alpha=config.lora_alpha,
                dropout=config.lora_dropout,
            )

            # Verify the freshly created LoRA layers are actually rank config.lora_r
            # (catches stale checkpoints or a mis-merged base from silently producing
            # the wrong adapter shape).
            from torchtune.modules.peft import LoRALinear
            lora_linears = [m for m in self.model.backbone.modules() if isinstance(m, LoRALinear)]
            assert lora_linears, "apply_lora_torchtune produced no LoRALinear layers"
            bad_rank = [m for m in lora_linears if m.lora_a.weight.shape[0] != config.lora_r]
            assert not bad_rank, (
                f"Expected all {len(lora_linears)} LoRALinear layers at rank {config.lora_r}, "
                f"found {len(bad_rank)} at a different rank"
            )
            logger.info(f"[ZoneD] Verified {len(lora_linears)} LoRALinear layers at rank {config.lora_r}")

        # --- Move entire model to device, then align backbone dtype ---
        # LoRALinear initializes lora_a/lora_b in float32 regardless of backbone dtype.
        # Moving after LoRA replacement ensures new layers land on the correct device.
        # Then cast backbone to its own dtype so base weights and adapter weights match.
        self.model.to(self.device)
        if self.model.backbone is not None:
            backbone_dtype = self.model.backbone.dtype
            self.model.backbone.to(dtype=backbone_dtype)

        # NT encoder must stay in float32 to prevent bfloat16 NaN in attention.
        if "dna" in self.model.encoders and hasattr(self.model.encoders["dna"], "model"):
            self.model.encoders["dna"].model.float()

        # --- Freeze / unfreeze projector (controlled by freeze_connector config flag) ---
        # Default True for GRPO: projector is loaded from SFT and not updated.
        # Set freeze_connector=False in the config to unfreeze for future experiments.
        freeze_projector = getattr(config, "freeze_connector", True)
        for param in self.model.projectors.parameters():
            param.requires_grad = not freeze_projector

        # --- Reference model (frozen copy of policy) ---
        if ref_model is not None:
            self.ref_model = ref_model
        else:
            self.ref_model = copy.deepcopy(self.model)
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

        # GRPO hyperparameters
        self.num_generations = config.grpo_num_generations
        self.max_completion_length = config.grpo_max_completion_length
        self.temperature = config.grpo_temperature
        self.top_p = config.grpo_top_p
        self.top_k = config.grpo_top_k
        self.beta = config.grpo_beta

        # epsilon is unused by GRPOSimpleLoss's forward (single-gradient-step-
        # per-batch GRPO has no importance-sampling ratio to clip -- see its
        # docstring), kept at the module default for API completeness.
        self.grpo_loss = GRPOSimpleLoss(kl_coeff=self.beta)

        # Reward functions
        reward_names = config.grpo_reward_functions or [
            "xmlcount", "soft_format", "strict_format", "concise", "correctness"
        ]
        self.reward_fns = [REWARD_FUNCTIONS[name] for name in reward_names]
        self.reward_names = reward_names

        # Freeze audit: ZoneATrainer has check_parameter_status as a
        # per-step watchdog because its trainable set can differ by
        # stage/config; ZoneDTrainer's freeze decisions are all made once,
        # right above, and never change again during training, so a single
        # post-construction assertion (not a periodic re-check) is the
        # right shape here. Catches a freeze step silently not covering
        # some param group -- training with far more capacity than
        # intended, otherwise only caught by chance via a
        # trainable-param-count smoke print.
        self.check_parameter_status(lora_enabled=lora_enabled, freeze_projector=freeze_projector)

        # Optimizer — built after freezing so only trainable params are included.
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Gradient accumulation: config.max_steps counts real optimizer
        # steps (matching trainer_zone_a.py's convention, where `step` and
        # the scheduler are only advanced on Accelerate's sync_gradients
        # boundary) -- train() draws gradient_accumulation_steps
        # micro-batches per real step, only stepping the optimizer and
        # scheduler on the last one of each group.
        self.gradient_accumulation_steps = max(
            int(getattr(config, "gradient_accumulation_steps", 1) or 1), 1
        )

        # LR scheduler, same 3-way dispatch as trainer_zone_a.py (molmo_layered
        # doesn't apply here -- this optimizer has a single param group, not
        # separate connector/vit/llm groups -- but kept for config parity).
        sched_type = getattr(config, "scheduler_type", "cosine")
        min_lr_ratio = getattr(config, "min_lr_ratio", 0.0)
        if sched_type == "molmo_layered":
            from src.utils.scheduler import get_molmo_scheduler

            self.scheduler = get_molmo_scheduler(
                self.optimizer,
                config.max_steps,
                warmup_connector=getattr(config, "warmup_steps_connector", 200),
                warmup_main=getattr(config, "warmup_steps_main", 2000),
                min_lr_ratio=min_lr_ratio,
            )
        elif sched_type == "cosine_with_min_lr":
            from src.utils.scheduler import get_cosine_with_min_lr

            self.scheduler = get_cosine_with_min_lr(
                self.optimizer,
                num_warmup_steps=config.warmup_steps,
                num_training_steps=config.max_steps,
                min_lr_ratio=min_lr_ratio,
            )
        else:
            from transformers import get_cosine_schedule_with_warmup

            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=config.warmup_steps,
                num_training_steps=config.max_steps,
            )

        if config.wandb_project:
            wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name,
                config=config.__dict__,
            )

    def check_parameter_status(self, lora_enabled: bool, freeze_projector: bool) -> None:
        """Assert the trainable-parameter set is exactly what __init__ just
        configured -- fail fast if not, rather than silently training with
        the wrong capacity.

        Mirrors trainer_zone_a.py's check_parameter_status (name-substring
        matching against named_parameters()) but as a one-shot
        post-construction check, not a per-step watchdog: ZoneDTrainer's
        freeze/LoRA decisions are all made once in __init__ (encoder frozen
        unconditionally, LoRA applied, projector frozen/unfrozen per
        freeze_connector) and never change again during training, so there's
        nothing for a periodic re-check to catch that this one-time
        assertion wouldn't already have caught at construction time.
        """
        trainable = [n for n, p in self.model.named_parameters() if p.requires_grad]
        trainable_set = set(trainable)

        lora_active = any("lora_" in n for n in trainable)
        encoder_active = any(n.startswith("encoders.") or ".encoders." in n for n in trainable)
        projector_active = any(n.startswith("projectors.") or ".projectors." in n for n in trainable)
        # Non-LoRA backbone params -- i.e. the frozen base weights LoRA is
        # supposed to leave untouched.
        backbone_base_active = any(
            (n.startswith("backbone.") or ".backbone." in n) and "lora_" not in n
            for n in trainable
        )

        logger.info(
            f"[ZoneD Watchdog] Trainable scopes: LoRA={lora_active}, "
            f"Encoder={encoder_active}, Projector={projector_active}, "
            f"BackboneBase={backbone_base_active} ({len(trainable_set)} trainable tensors)"
        )

        # DNA encoder is unconditionally frozen a few lines above, in every
        # configuration -- never trained during GRPO.
        if encoder_active:
            raise RuntimeError(
                "CRITICAL: DNA encoder has requires_grad=True params after "
                "the unconditional freeze in __init__! Ghost training "
                "detected. Aborting."
            )

        if lora_enabled:
            if not lora_active:
                raise RuntimeError(
                    "CRITICAL: lora_enabled=True but no LoRA (lora_a/lora_b) "
                    "params are trainable! Ghost training detected. Aborting."
                )
            if backbone_base_active:
                raise RuntimeError(
                    "CRITICAL: lora_enabled=True but non-LoRA backbone base "
                    "weights are also trainable -- apply_lora_torchtune's "
                    "freeze via set_trainable_params should have left only "
                    "lora_a/lora_b trainable. Ghost training detected "
                    "(training with far more capacity than intended). "
                    "Aborting."
                )

        # freeze_connector controls whether the projector is trainable;
        # assert whichever state was actually configured, not just "some
        # scope is trainable somewhere" the way ZoneATrainer's broader
        # per-stage check does.
        if freeze_projector and projector_active:
            raise RuntimeError(
                "CRITICAL: freeze_connector=True but projector params have "
                "requires_grad=True! Ghost training detected. Aborting."
            )
        if not freeze_projector and not projector_active:
            raise RuntimeError(
                "CRITICAL: freeze_connector=False but no projector params "
                "are trainable! Ghost training detected. Aborting."
            )

    def _generate_completions(self, prompt_batch: dict) -> list[str]:
        """Generate G completions for a single prompt via multimodal interleaved sampling.

        Args:
            prompt_batch: Dict with keys "text" (1, P), "dna" (sub-dict with
                dna_reference/dna_variant), and "_metadata" (["P 0"]).
                DNA embeddings are encoded by the frozen NT encoder, projected, and
                spliced at the <dna_ref_start><dna_ref_end> / <dna_var_start><dna_var_end>
                placeholders before generation.

        Returns:
            List of G decoded completion strings (no BOS token, no prompt text).
        """
        completions = []
        self.model.eval()

        pad_token_id = (
            self.tokenizer.pad_token_id if self.tokenizer else None
        )
        eos_token_id = (
            self.tokenizer.eos_token_id if self.tokenizer else None
        )

        with torch.no_grad():
            for _ in range(self.num_generations):
                # model.generate() encodes DNA, projects, splices interleaved,
                # then calls backbone.generate(inputs_embeds=...).
                # When called with inputs_embeds only (no input_ids), HF prepends
                # a single BOS token to the output sequence; we strip it below.
                output_ids = self.model.generate(
                    prompt_batch,
                    max_new_tokens=self.max_completion_length,
                    do_sample=True,
                    temperature=self.temperature,
                    top_k=self.top_k,
                    top_p=self.top_p,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                )
                # output_ids: (1, 1 + new_tokens)  — leading 1 is the BOS seed token
                gen_ids = output_ids[0, 1:]  # strip BOS
                if self.tokenizer:
                    text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
                else:
                    text = str(gen_ids.tolist())
                completions.append(text)

        self.model.train()
        return completions

    def _compute_rewards(
        self, completions: list[str], answer: str = ""
    ) -> tuple[torch.Tensor, dict[str, list[float]]]:
        """Compute aggregate reward for each completion across all reward functions.

        Returns:
            rewards: aggregate (mean-across-functions) reward per completion,
                same as before -- used for advantage normalization.
            per_function: {reward_name: [score per completion]}, the raw
                per-function scores the aggregate was computed from. Exposed
                so train() can log reward_mean/std/min/max/successes broken
                out per reward function, not just the aggregate -- a
                function that's silently always 0 (dead reward plumbing)
                is invisible in the aggregate alone, especially once other
                functions are nonzero, but immediately visible per-function.
        """
        rewards = torch.zeros(len(completions))
        per_function: dict[str, list[float]] = {name: [] for name in self.reward_names}
        for completion_idx, completion in enumerate(completions):
            total = 0.0
            for name, fn in zip(self.reward_names, self.reward_fns, strict=True):
                score = fn(completion=completion, answer=answer)
                per_function[name].append(score)
                total += score
            rewards[completion_idx] = total / len(self.reward_fns)
        return rewards, per_function

    def _get_log_probs(
        self, model, batch: dict, prompt_len: int
    ) -> torch.Tensor:
        """Per-token log-prob over completion tokens only (not the prompt).

        Args:
            model: Policy or reference model.
            batch: Dict with "text" (1, P+C), "dna", "_metadata" (["P C"]).
                The merge function uses "P C" to expand DNA slots and align labels
                to the completion window.
            prompt_len: P — number of text token positions in the prompt
                (before DNA expansion). Used to index completion tokens in the
                original text sequence for label lookup.

        Returns:
            Tensor of shape (1, comp_len): per-token log-prob of each
            completion token. Left unreduced (not summed/meaned) so
            GRPOSimpleLoss.forward can apply its own masked-mean over the
            completion dimension, matching torchtune's validated GRPO loss
            exactly instead of pre-reducing to a scalar before combining
            with the KL term.
        """
        logits, _ = model(batch)
        # logits: (1, T_merged, V)  where T_merged = T_text + DNA_expansion
        # The completion occupies the last comp_len positions of the merged sequence.
        comp_len = int(batch["_metadata"][0].split()[1])
        T_merged = logits.shape[1]
        new_prompt_len = T_merged - comp_len

        # Shifted logits aligned to completion positions:
        # position new_prompt_len-1 predicts position new_prompt_len, etc.
        shift_logits = logits[:, new_prompt_len - 1 : new_prompt_len + comp_len - 1, :]

        # Ground-truth token IDs for the completion come from the original text
        # sequence (DNA tokens never appear in the completion, only in the prompt).
        shift_labels = batch["text"][:, prompt_len : prompt_len + comp_len]

        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = torch.gather(
            log_probs, -1, shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        return token_log_probs

    def _slice_dna_sample(self, dna_batch: dict, idx: int) -> dict:
        """Slice the i-th sample out of a collated DNA batch.

        Args:
            dna_batch: Collated dict {"dna_reference": {"input_ids": (B,L),
                "attention_mask": (B,L)}, "dna_variant": {...}}.
            idx: Batch index to extract.

        Returns:
            Single-sample dict {"dna_reference": {"input_ids": (1,L), ...},
            "dna_variant": {...}}.
        """
        return {
            sub: {k: v[idx].unsqueeze(0) for k, v in tensors.items()}
            for sub, tensors in dna_batch.items()
        }

    def train(self):
        """Main GRPO training loop."""
        self.model.train()
        step = 0
        progress_bar = tqdm(range(self.config.max_steps), desc="Training Zone D (GRPO)")
        data_iter = iter(self.train_loader)

        # gradient_accumulation_steps micro-batches are drawn per real
        # optimizer step; `step` (and logging/checkpointing/the scheduler)
        # only advance once every gradient_accumulation_steps iterations of
        # this inner counter -- config.max_steps counts real optimizer
        # steps, matching trainer_zone_a.py's convention.
        self.optimizer.zero_grad()
        micro_step = 0
        # Averaged over the accumulation window and reset at each sync
        # boundary, same reasoning as trainer_zone_a.py's identical
        # accum_window_losses: logging only the last micro-batch's loss
        # (instead of the window average) produces a misleading sawtooth
        # curve that doesn't reflect what the optimizer actually stepped on.
        accum_window_losses: list[float] = []
        # Raw per-reward-function scores across the accumulation window,
        # same reset-at-sync-boundary treatment as accum_window_losses.
        # Logged as mean/std/min/max/successes per function -- the
        # aggregate avg_reward alone can't distinguish "every function
        # contributes a little" from "one function is silently always 0
        # while the others carry the whole signal", which is exactly the
        # failure mode a dead/misconfigured reward function produces.
        accum_window_reward_breakdown: dict[str, list[float]] = {
            name: [] for name in self.reward_names
        }

        while step < self.config.max_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                batch = next(data_iter)

            batch = self._to_device(batch)

            # Process each sample in the batch individually (GRPO requires per-sample rollouts).
            total_loss = torch.tensor(0.0, device=self.device)
            batch_size = 0

            prompt_ids_batch = batch.get("prompt_ids", batch.get("text"))  # (B, P)
            answers = batch.get("answer", [""] * prompt_ids_batch.shape[0])
            metadata_list = batch.get("_metadata", [])
            dna_batch_full = batch.get("dna", {})

            for i in range(prompt_ids_batch.shape[0]):
                prompt_ids = prompt_ids_batch[i].unsqueeze(0)  # (1, P_max) -- right-padded to batch max
                answer = answers[i] if isinstance(answers, list) else ""

                # P = prompt token count in text-token space (before DNA expansion).
                # Taken from the "P T" _metadata string produced by the data pipeline.
                prompt_len = int(metadata_list[i].split()[0]) if metadata_list else prompt_ids.shape[-1]

                # Truncate off the batch's right-padding tail (collate.py's
                # pad_sequence right-pads to the longest prompt in the batch,
                # src/data/collate.py:92-94). Without this, any sample shorter
                # than the batch max carries trailing pad tokens into
                # `full_ids = cat([prompt_ids, comp_ids])` below, sitting
                # between the real prompt and the generated completion --
                # `shift_labels`/`shift_logits` in _get_log_probs then read
                # from the wrong offset, silently misaligning the label
                # window against the completion tokens under any batch_size>1
                # with prompts of differing length.
                prompt_ids = prompt_ids[:, :prompt_len]

                # Per-sample DNA tensors sliced from the collated batch.
                dna_i = self._slice_dna_sample(dna_batch_full, i)

                # Generation batch: T=0 because there are no target tokens yet.
                prompt_batch = {
                    "text": prompt_ids,
                    "dna": dna_i,
                    "_metadata": [f"{prompt_len} 0"],
                }

                # 1. Generate G completions via the full interleaved multimodal path.
                completions = self._generate_completions(prompt_batch)

                # 2. Compute rewards
                rewards, per_function_rewards = self._compute_rewards(completions, answer=answer)
                rewards = rewards.to(self.device)
                for name, scores in per_function_rewards.items():
                    accum_window_reward_breakdown[name].extend(scores)

                # 3. Compute group-relative advantages
                #
                # unbiased=False (population std, /N) is required, not
                # cosmetic: torch.Tensor.std()'s default (unbiased=True, /(N-1))
                # returns NaN for a 1-element group -- inert at the yaml
                # default grpo_num_generations=8, but a live NaN-loss trap
                # the moment num_generations=1 is used. The +1e-8 alone
                # does not rescue a NaN input.
                #
                # Note: when every rollout in this group scores identically
                # (e.g. all wrong, or all earning only a shared format
                # bonus), std_reward correctly collapses to ~0 and every
                # advantage goes to ~0 -- a zero policy-gradient contribution
                # for this prompt, by design (there's no relative signal to
                # learn from within an identical-reward group), not a bug.
                # With grpo_beta=0.0 (default) this means only prompts whose
                # completions vary in reward contribute gradient at all.
                mean_reward = rewards.mean()
                std_reward = rewards.std(unbiased=False) + 1e-8
                advantages = (rewards - mean_reward) / std_reward

                # 4. Compute policy gradient loss over completion tokens only.
                for comp_idx, completion in enumerate(completions):
                    if self.tokenizer:
                        # HF tokenizers return an empty tensor as float32 (not the
                        # usual int64) when encoding "" (e.g. a completion that
                        # decodes to nothing, such as immediate-EOS generations)
                        # since there are no token values to infer an integer
                        # dtype from. torch.cat then silently upcasts prompt_ids
                        # to float too, corrupting the embedding lookup downstream.
                        # Force long dtype explicitly to guard against this.
                        comp_ids = self.tokenizer.encode(
                            completion, return_tensors="pt"
                        ).to(self.device, dtype=torch.long)
                    else:
                        comp_ids = torch.empty(1, 0, dtype=torch.long, device=self.device)

                    comp_len = comp_ids.shape[-1]
                    full_ids = torch.cat([prompt_ids, comp_ids], dim=1)  # (1, P+C)

                    # Full batch for forward pass: DNA fixed, text extended with completion.
                    full_batch = {
                        "text": full_ids,
                        "dna": dna_i,
                        "_metadata": [f"{prompt_len} {comp_len}"],
                    }

                    # Policy log-prob (completion tokens only, gradients flow back).
                    # Per-token, shape (1, comp_len) -- unreduced, see _get_log_probs.
                    policy_logp = self._get_log_probs(self.model, full_batch, prompt_len)

                    # Reference log-prob (no gradient).
                    with torch.no_grad():
                        ref_logp = self._get_log_probs(self.ref_model, full_batch, prompt_len)

                    # GRPO loss + k3 KL estimator, via torchtune's validated
                    # GRPOSimpleLoss (see the import comment at the top of this
                    # file). One call per generation, matching this loop's
                    # existing per-sample-at-a-time structure: pi_old_logprobs
                    # is unused by GRPOSimpleLoss (no importance-sampling ratio
                    # in single-gradient-step GRPO), padding_masks is all-True
                    # since policy_logp/ref_logp are already sliced to exactly
                    # the completion window (no padding at this granularity).
                    # masked_mean's internal +1e-8 denominator (torchtune's
                    # own comp_len=0 guard) replaces this file's previous
                    # manual max(comp_len, 1e-9) length-normalization.
                    advantage = advantages[comp_idx].unsqueeze(0)  # (1,)
                    padding_mask = torch.ones_like(policy_logp, dtype=torch.bool)
                    loss, _policy_loss, _kl_loss, _ratios, _clipfrac = self.grpo_loss(
                        pi_old_logprobs=policy_logp,
                        pi_logprobs=policy_logp,
                        ref_logprobs=ref_logp,
                        advantages=advantage,
                        padding_masks=padding_mask,
                    )
                    total_loss = total_loss + loss

                batch_size += 1

            if batch_size > 0:
                total_loss = total_loss / (batch_size * self.num_generations)
            accum_window_losses.append(total_loss.item())

            # Backward. Scaled by gradient_accumulation_steps so the
            # effective gradient magnitude at the eventual optimizer.step()
            # matches a single accumulation_steps-times-larger batch,
            # rather than being accumulation_steps times too large.
            (total_loss / self.gradient_accumulation_steps).backward()
            micro_step += 1
            is_sync_step = micro_step % self.gradient_accumulation_steps == 0
            if not is_sync_step:
                # Mid-accumulation-window: skip step/scheduler/logging/
                # checkpointing/step-increment entirely, draw the next
                # micro-batch and keep accumulating gradients.
                continue

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            # Logging
            if step % self.config.log_every_n_steps == 0:
                avg_loss = sum(accum_window_losses) / len(accum_window_losses)
                # Aggregate: mean across every reward function AND every
                # completion in the whole accumulation window -- matches
                # what _compute_rewards' own per-completion aggregate
                # represents, just window-averaged instead of only the last
                # micro-batch's value.
                all_scores = [
                    s for scores in accum_window_reward_breakdown.values() for s in scores
                ]
                avg_reward = sum(all_scores) / len(all_scores) if all_scores else 0.0
                log_dict = {
                    "grpo_loss": avg_loss,
                    "avg_reward": avg_reward,
                    "lr": self.scheduler.get_last_lr()[0],
                    "step": step,
                }
                # Per-reward-function mean/std/min/max/successes. The
                # aggregate avg_reward alone can't distinguish "every
                # function contributes a little" from "one function is
                # silently always 0 while others carry the whole signal" --
                # a successes count (nonzero-reward completions) per
                # function surfaces a dead/misconfigured reward function on
                # step 1 instead of only after a wasted run.
                for name, scores in accum_window_reward_breakdown.items():
                    if not scores:
                        continue
                    scores_t = torch.tensor(scores)
                    log_dict[f"reward/{name}/mean"] = scores_t.mean().item()
                    log_dict[f"reward/{name}/std"] = scores_t.std(unbiased=False).item()
                    log_dict[f"reward/{name}/min"] = scores_t.min().item()
                    log_dict[f"reward/{name}/max"] = scores_t.max().item()
                    log_dict[f"reward/{name}/successes"] = sum(1 for s in scores if s > 0)
                if self.config.wandb_project:
                    wandb.log(log_dict)
                progress_bar.set_postfix({"loss": avg_loss, "reward": avg_reward})
                logger.info(f"[ZoneD] step={step} loss={avg_loss:.4f} avg_reward={avg_reward:.4f}")

            # Reset for the next accumulation window regardless of whether
            # this step logged -- every iteration reaching this point is a
            # sync (optimizer-step) boundary, since non-sync micro-batches
            # `continue` before this point.
            accum_window_losses = []
            accum_window_reward_breakdown = {name: [] for name in self.reward_names}

            # Checkpointing (0/None disables -- guard against ZeroDivisionError,
            # mirrors trainer_zone_a.py's save_every_n_steps gate).
            save_interval = getattr(self.config, "save_every_n_steps", 0)
            if save_interval and step > 0 and step % save_interval == 0:
                self.save_checkpoint(step)

            step += 1
            progress_bar.update(1)

        # Save Final Checkpoint. Previously missing entirely: train() had no
        # save_checkpoint call of any kind (periodic or final), so a complete
        # GRPO run -- however many steps, however much compute -- silently
        # discarded all trained weights. Confirmed live: a real 50-step
        # Aurora GRPO run completed ("Zone D (GRPO) Training Complete.") but
        # left no step_final checkpoint anywhere on disk. Mirrors
        # trainer_zone_a.py's "Save Final Checkpoint" call at the same point
        # in its own loop.
        self.save_checkpoint("final")
        logger.info("Zone D (GRPO) Training Complete.")
        if self.config.wandb_project:
            wandb.finish()

    def save_checkpoint(self, step):
        """Save a GRPO checkpoint: LoRA adapter weights + training state.

        Unlike trainer_zone_a.py's save_checkpoint, there is no
        self.accelerator here to call save_state() on -- ZoneDTrainer runs
        single-process by design (see module docstring / ea64b04). GRPO only
        ever trains the LoRA adapter (backbone base weights, DNA encoder, and
        projector are all frozen -- see __init__), so the adapter is the only
        state that needs to survive a run; saved in the same torchtune
        plain-state-dict .pt format trainer_zone_a.py's save_checkpoint
        uses, so resume_weights_only + resume_lora_alpha (this file's own
        merge_lora_into_base call in __init__) can load it back unchanged.
        """
        output_dir = os.path.join(self.config.output_dir, f"step_{step}")
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"[ZoneD] [Step {step}] Saving checkpoint to {output_dir}...")

        if getattr(self.config, "lora_enabled", False) and self.model.backbone is not None:
            from src.utils.lora_utils import save_lora_adapter

            lora_path = os.path.join(output_dir, "lora_adapter.pt")
            save_lora_adapter(
                self.model.backbone,
                lora_path,
                rank=self.config.lora_r,
                alpha=self.config.lora_alpha,
                target_modules=getattr(self.config, "lora_target_modules", None),
            )
            logger.info(f"[ZoneD] LoRA adapter saved to {lora_path}")

        with open(os.path.join(output_dir, "training_state.json"), "w") as f:
            json.dump({"step": step, "config": self.config.__dict__}, f, default=str)
        logger.info("[ZoneD] Checkpoint saved.")

    def _to_device(self, item):
        if isinstance(item, torch.Tensor):
            return item.to(self.device)
        elif isinstance(item, dict):
            return {k: self._to_device(v) for k, v in item.items()}
        elif isinstance(item, list):
            return [self._to_device(v) for v in item]
        return item
