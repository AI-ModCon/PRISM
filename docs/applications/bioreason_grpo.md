# BioReason GRPO — Implementation Reference

Full technical breakdown of the three-stage BioReason training pipeline with emphasis on
the Stage 3 GRPO reinforcement learning implementation.

---

## Table of Contents

1. [Training Stages Overview](#1-training-stages-overview)
2. [Model Architecture](#2-model-architecture)
3. [GRPO Training Framework](#3-grpo-training-framework)
4. [Reward Functions](#4-reward-functions)
5. [Data Pipeline](#5-data-pipeline)
6. [Key Implementation Details](#6-key-implementation-details)
7. [Hyperparameter Reference](#7-hyperparameter-reference)
8. [Suggested Upgrades](#8-suggested-upgrades)

---

## 1. Training Stages Overview

| Stage | Name | Trainer | What trains | Loss |
|---|---|---|---|---|
| 1 | Projector | `ZoneATrainer` | DNA projector only | Full sequence causal LM |
| 2 | SFT | `ZoneATrainer` | LoRA adapters on LLM | Answer tokens only (masked) |
| 3 | GRPO | `ZoneDTrainer` | LoRA adapters on LLM | GRPO reward signal |

Each stage loads weights from the previous via `training.resume_weights_only`.

Submit scripts:
- Stage 1: `scripts/submit_bioreason_projector.sh`
- Stage 2: `scripts/submit_bioreason_sft.sh`
- Stage 3: `scripts/submit_bioreason_grpo.sh`

---

## 2. Model Architecture

### Pipeline

```
DNA (reference sequence)  ──┐
                             ├──► Nucleotide Transformer v2 (250M, frozen)
DNA (variant sequence)    ──┘         ↓
                                 ModalityProjector (2-layer MLP)
                                      ↓
              spliced inline at <dna_ref_start><dna_ref_end> / <dna_var_start><dna_var_end>
                                      ↓
                              OLMo-1B backbone
                         (base weights frozen; LoRA adapters trainable)
                                      ↓
                                   LM head
```

### DNA Encoder

**Model**: `InstaDeepAI/nucleotide-transformer-v2-250m-multi-species`

- Reference and variant sequences tokenized **separately**, each up to `max_dna_length=1024` NT tokens
- Kept in **float32** via a custom `_apply()` override — prevents NaN from attention overflow when the rest of the model runs in bfloat16
- Token IDs clamped before embedding lookup to guard against out-of-vocabulary IDs leaking in from the text tokenizer
- Output shape: `(B, T_dna, 768)`

Source: `src/encoders/dna.py`

### DNA Projector

**Architecture**: `ModalityProjector` — two linear layers with LayerNorm and a learnable modality embedding

```
Input  (B, T_dna, 768)
  → fc1  Linear(768, d_model)
  → fc2  Linear(d_model, d_model)   [activation between fc1/fc2 currently commented out]
  → LayerNorm
  → + modality_embedding (1, 1, d_model)   [learnable, scale=0.02, added after norm]
Output (B, T_dna, 4096)
```

Both fc layers use Xavier uniform initialization. `d_model=4096` for OLMo-1B.

Source: `src/modules/projector.py`

### LLM Backbone

**Model**: `allenai/OLMo-1B-0724-hf`

- Base weights **frozen** throughout GRPO
- LoRA adapters applied via `apply_lora_torchtune()` to every `nn.Linear` layer except `lm_head`
- Input embeddings for text come from the backbone's native embedding layer
- DNA embeddings are injected directly into the residual stream, bypassing the embedding layer

Source: `src/utils/lora_utils.py`, `src/model.py`

### LoRA Configuration

| Parameter | Stage 2 (SFT) | Stage 3 (GRPO) |
|---|---|---|
| Rank (`lora_r`) | 32 | 16 |
| Alpha (`lora_alpha`) | 64 | 32 |
| Dropout | 0.05 | 0.0 |
| Target modules | All `nn.Linear` (excl. `lm_head`) | Same |

Trainable parameter count: ~20M / 1.3B total (~1.5%).

### What Is Trainable During GRPO

| Component | Status |
|---|---|
| DNA Encoder (NT-v2 250M) | Frozen |
| DNA Projector | Frozen (loaded from Stage 1) |
| OLMo-1B base weights | Frozen |
| LoRA adapters | **Trainable** |

### DNA Interleaving

Each DNA sequence is merged into the token sequence at its own placeholder:
`dna_reference` at `<dna_ref_start><dna_ref_end>` and `dna_variant` at
`<dna_var_start><dna_var_end>`. The prompt renders both, labeled and in fixed
order (reference before variant), so the merged sequence seen by OLMo is:

This is handled by `_merge_text_input_ids_with_modality_embeds()` in `src/model.py`,
where `dna_reference` and `dna_variant` are independent modality entries that go
through the same generic splice path as every other modality (image, time_series, ...).

```
[dna_ref (1024 tokens)] [dna_var (1024 tokens)] [text (question + role markers)]
```

---

## 3. GRPO Training Framework

Source: `src/training/trainer_grpo.py`

### Training Loop

For each step:

```
1. Sample batch of B=8 prompts from the KEGG dataloader

2. For each prompt i in [0, B):
   a. Generate G=8 independent completions via multinomial sampling
   b. Score each completion with 5 reward functions → rewards ∈ ℝ^G
   c. Compute group-relative advantages across the G completions
   d. For each completion j in [0, G):
        - Compute policy log-prob:    log π_θ(completion | prompt)
        - Compute reference log-prob: log π_ref(completion | prompt)   [no grad]
        - Accumulate loss term

3. total_loss /= B × G    (divide by 64)
4. loss.backward()
5. clip_grad_norm_(params, max_norm=1.0)
6. optimizer.step()
```

### Loss Formula

```
loss_ij = −advantage_ij × log π_θ(y_ij | x_i)
         + β × (log π_θ(y_ij | x_i) − log π_ref(y_ij | x_i))

total_loss = (1 / B×G) × Σ_i Σ_j loss_ij
```

With `β = 0.0` (current config) this reduces to pure reward maximization:

```
loss_ij = −advantage_ij × log π_θ(y_ij | x_i)
```

### Advantage Computation

Advantages are **normalized per-prompt group** (across the G=8 completions for one prompt),
not globally across the batch:

```python
mean_r = rewards.mean()        # over G=8 completions for prompt i
std_r  = rewards.std() + 1e-8
advantages = (rewards - mean_r) / std_r   # shape: (G,)
```

A completion is judged only relative to the other 7 for the same input. If all 8 completions
are correct, all advantages ≈ 0 and that prompt contributes no gradient.

### Log-Probability Computation

```python
logits, _ = model(inputs)
shift_logits = logits[..., :-1, :].contiguous()   # exclude last token
shift_labels = input_ids[..., 1:].contiguous()    # shift labels left
log_probs    = F.log_softmax(shift_logits, dim=-1)
token_lp     = gather(log_probs, shift_labels)    # per-token log-prob
return token_lp.sum(dim=-1)                        # sum over sequence
```

### Reference Model

- Initialized as `deepcopy(model)` at trainer init time
- Set to `eval()` mode; all parameters frozen
- **Never updated** during training — holds Stage 2 SFT weights for the full GRPO run
- Used only for the KL divergence term (which is currently zeroed out by `β=0.0`)

### Rollout Generation

```python
# Per prompt: G=8 independent samples
for g in range(num_generations):
    tokens = prompt_ids.clone()
    for _ in range(max_completion_length):   # max 800 tokens
        logits = model(tokens)[:, -1, :]     # next-token logits
        # Top-k filter (k=20), then top-p nucleus (p=0.95)
        filtered = top_k_top_p_filter(logits, k=20, p=0.95)
        next_tok = multinomial(softmax(filtered / temperature))
        tokens = cat([tokens, next_tok])
        if next_tok == eos_token_id:
            break
```

Both filters are applied sequentially: top-k first, then nucleus on the k-filtered distribution.

### Gradient Accumulation

- `gradient_accumulation_steps = 4`
- Effective prompts per optimizer step: `B × grad_accum = 8 × 4 = 32`
- Effective completions per optimizer step: `32 × G = 32 × 8 = 256`

### Distributed Training

```yaml
distribution_strategy: "ddp"
ddp:
  bucket_cap_mb: 25
  find_unused_parameters: false
  static_graph: false          # false because GRPO has variable-length computation graphs
  gradient_as_bucket_view: true
```

---

## 4. Reward Functions

Source: `src/training/trainer_grpo.py` (lines 50–131)

Five functions, each returning a scalar in `[0.0, 1.0]`, **averaged equally**:

```python
reward = mean([xmlcount, soft_format, strict_format, concise, correctness])
```

Maximum possible reward: **1.0**

### 4.1 `xmlcount_reward` — Structural tag counting

Rewards the presence and correct nesting of reasoning tags:

| Condition | Points |
|---|---|
| `<think>` present in completion | +0.125 |
| `</think>` present in completion | +0.125 |
| `Answer:` prefix present | +0.125 |
| Exactly one `<think>…</think>` pair | +0.250 |
| `</think>` appears **before** `Answer:` | +0.375 |
| **Maximum** | **1.000** |

### 4.2 `soft_format_reward` — Loose regex match

```python
pattern = r"<think>.*?</think>\s*Answer:"
score = 1.0 if re.search(pattern, completion, re.DOTALL) else 0.0
```

### 4.3 `strict_format_reward` — Exact format match

```python
pattern = r"^\s*<think>\n.*?\n</think>\n\nAnswer:.*$"
score = 1.0 if re.match(pattern, completion, re.DOTALL) else 0.0
```

Requires literal `\n` after `<think>` and `\n\n` before `Answer:` — significantly stricter
than `soft_format_reward`.

### 4.4 `concise_reward` — Length penalty

| Word count | Score |
|---|---|
| < 10 | 0.0 (too short) |
| 10–300 | 1.0 (ideal) |
| 300–600 | 0.5 (acceptable) |
| > 600 | 0.0 (too long) |

### 4.5 `correctness_reward` — Answer accuracy

Answer extraction:

```python
match = re.search(r"Answer:\s*(.*?)(?:<\|im_end\|>|$)", completion, re.DOTALL)
extracted = match.group(1).strip().lower() if match else completion.strip().lower()
ground_truth = answer.strip().lower()
```

Scoring:

| Condition | Score |
|---|---|
| `extracted == ground_truth` (exact) | 1.0 |
| `ground_truth in extracted` (containment) | 0.5 |
| No match | 0.0 |

### Expected Format

The model is trained to produce:

```
<think>
{chain-of-thought reasoning}
</think>

Answer: {label}
```

This format is established during Stage 2 SFT via `use_reasoning_traces: true`.

---

## 5. Data Pipeline

Source: `src/data/multimodal.py` — `_process_dna_bioreason()`, `_render_dna_prompt_text()`

### Dataset

**HuggingFace dataset**: `wanglab/kegg`

Raw example fields:

```python
{
    "reference_sequence": "ACGTACGT...",   # Reference DNA string
    "variant_sequence":   "ACGTACGT...",   # Variant DNA string
    "question":           "...",            # Natural language question
    "answer":             "...",            # Ground truth label (e.g. "pathogenic")
    "reasoning":          "...",            # CoT explanation (used in SFT only)
}
```

### Prompt Rendering (`_render_dna_prompt_text`)

**User turn** (DNA-LLM interleaved mode):

```
<|im_start|>user
Reference sequence: <dna_ref_start><dna_ref_end>
Variant sequence: <dna_var_start><dna_var_end>
{question}
<|im_end|>
```

Each `<dna_ref_start><dna_ref_end>` / `<dna_var_start><dna_var_end>` span is a two-token
placeholder that gets replaced by its respective projected DNA embeddings at forward time.

**Generation prompt** (GRPO — no assistant answer in the sequence):

```
<|im_start|>user
Reference sequence: <dna_ref_start><dna_ref_end>
Variant sequence: <dna_var_start><dna_var_end>
{question}
<|im_end|>
<|im_start|>assistant
```

The model generates starting from after `<|im_start|>assistant`.

**Full text** (SFT — answer included):

```
<|im_start|>user
Reference sequence: <dna_ref_start><dna_ref_end>
Variant sequence: <dna_var_start><dna_var_end>
{question}
<|im_end|>
<|im_start|>assistant
<think>
{reasoning}
</think>

{answer}<|im_end|>
```

### Metadata Format (`"P T"`)

Used for label masking in `_merge_text_input_ids_with_modality_embeds()`:

```python
prompt_ids = tokenizer(prompt_text, add_special_tokens=False)
joint_ids  = tokenizer(full_text,   add_special_tokens=False)
P = len(prompt_ids)
T = len(joint_ids) - P
example["_metadata"] = f"{P} {T}"
```

- `P` = number of prompt tokens (masked from loss)
- `T` = number of answer tokens (loss computed here only)
- During GRPO generation: `full_text == prompt_text`, so `T = 0`

### DNA Tokenization

```python
ref_enc = dna_tokenizer(
    ref_seq,
    padding="max_length",
    truncation=True,
    max_length=1024,      # max_dna_length from config
    return_tensors="pt",
)
# Output keys: input_ids (1, 1024), attention_mask (1, 1024)
```

Reference and variant encoded separately; stored as:

```python
example["dna"] = {
    "dna_reference": {"input_ids": ..., "attention_mask": ...},
    "dna_variant":   {"input_ids": ..., "attention_mask": ...},
}
```

### Batch Keys Used by GRPO Trainer

```python
batch["text"]           # tokenized prompt (token IDs), shape (B, T_text)
batch["dna"]["dna_reference"]  # NT-tokenized reference, shape (B, 1024)
batch["dna"]["dna_variant"]    # NT-tokenized variant,   shape (B, 1024)
batch["_metadata"]      # list of "P T" strings, length B
batch["answer"]         # list of ground-truth answer strings, length B
```

---

## 6. Key Implementation Details

### DNA Float32 Override

The NT encoder's `_apply()` is overridden to keep the backbone in **float32** regardless of
the surrounding mixed-precision context. The projection layer (the single linear crossing the
dtype boundary) runs in the lower precision of the rest of the model. This prevents attention
score overflow in bfloat16 for long DNA sequences.

Source: `src/encoders/dna.py`, lines 121–148

### DNA Merging

`dna_reference` and `dna_variant` are independent modality entries in
`modality_start_end_token_indices` (each with its own start/end token pair), so
`_merge_text_input_ids_with_modality_embeds` splices them through the same generic
path used for every other modality — no DNA-specific sentinel or branch. The prompt
renderer emits the reference placeholder before the variant placeholder (see
"Prompt Rendering" above), which is what fixes their relative order in the merged
sequence; the merge function itself is order-agnostic and just resolves each
modality's own tag pair wherever it appears.

Source: `src/model.py`, `_merge_text_input_ids_with_modality_embeds`

### Missing Activation in Projector

The activation function between `fc1` and `fc2` in `ModalityProjector` is currently commented out,
making the projector a linear map composed with itself (equivalent to a single linear layer in
terms of expressivity). Re-enabling it would add nonlinearity at the cost of one additional
elementwise operation.

Source: `src/modules/projector.py`, line 161

### Token ID Clamping

Before NT tokenizer output is passed to the DNA encoder's embedding lookup, token IDs are clamped
to `[0, vocab_size - 1]`:

```python
enc["input_ids"] = enc["input_ids"].clamp(max=max_valid_id)
```

This prevents CUDA device-side asserts that would otherwise fire if text tokenizer pad IDs
accidentally appear in the DNA input tensor.

### Reference Model Never Updated

The reference model is a `deepcopy` taken at trainer initialization and **never synced** again.
With `grpo_beta=0.0` this is inconsequential, but enabling KL regularization without also
scheduling reference model updates would cause the KL penalty to measure divergence from the
Stage 2 SFT policy throughout all 1000 steps — which may or may not be the intended behavior.

### Data Iterator Reset

There is no explicit epoch concept. When the dataloader iterator is exhausted it is simply
reset: `data_iter = iter(self.train_loader)`. Training continues until `max_steps` is reached.

### Gradient Norm Clipping

Applied after `loss.backward()` and before `optimizer.step()`:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### Special Token Registration

`modality_start_end_token_indices` is stored as string token names, not integer IDs:

```python
{
    "dna_reference": ("<dna_ref_start>", "<dna_ref_end>"),
    "dna_variant": ("<dna_var_start>", "<dna_var_end>"),
}
```

These are resolved to integer IDs at forward time via `backbone_tokenizer.convert_tokens_to_ids()`.
The tokens are registered in the PRISM custom tokenizer at `tokenizers/prism-olmo-1b-interleaved`.

---

## 7. Hyperparameter Reference

### Stage 3 — GRPO (`src/conf/training/bioreason_grpo.yaml`)

```yaml
task:                        bioreason_grpo
batch_size:                  8
max_steps:                   1000
learning_rate:               1e-5
weight_decay:                0.01
warmup_steps:                30
scheduler_type:              cosine
min_lr_ratio:                0.0
gradient_accumulation_steps: 4

# GRPO
grpo_num_generations:        8      # completions sampled per prompt
grpo_max_completion_length:  800    # max new tokens per completion
grpo_temperature:            1.0
grpo_top_p:                  0.95
grpo_top_k:                  20
grpo_beta:                   0.0    # KL coefficient (disabled)

# LoRA
lora_enabled:                true
lora_r:                      16
lora_alpha:                  32
lora_dropout:                0.0

# Freezing
freeze_llm:                  true   # base weights; LoRA adapters remain trainable
freeze_vit:                  true   # DNA encoder
freeze_connector:            true   # DNA projector

# Data
bioreason_dataset:           wanglab/kegg
max_dna_length:              1024
dna_truncation_per_side:     1024
```

### Stage 2 — SFT (`src/conf/training/bioreason_sft.yaml`)

```yaml
task:                        bioreason_sft
batch_size:                  1
max_steps:                   5000
learning_rate:               5e-5
gradient_accumulation_steps: 8
warmup_steps:                500
lora_r:                      32
lora_alpha:                  64
lora_dropout:                0.05
freeze_llm:                  false  # must be false so LoRA can wrap the backbone
freeze_vit:                  true
freeze_connector:            true
use_reasoning_traces:        true
```

### Stage 1 — Projector (`src/conf/training/bioreason_projector.yaml`)

```yaml
task:                        bioreason_projector
batch_size:                  1
max_steps:                   5000
learning_rate:               5e-5
freeze_llm:                  true
freeze_vit:                  true
freeze_connector:            false   # only the projector trains
lora_enabled:                false
```

### Model (`src/conf/model/prism_olmo_1b.yaml`)

```yaml
backbone_id:    allenai/OLMo-1B-0724-hf
tokenizer_id:   tokenizers/prism-olmo-1b-interleaved
d_dna:          768    # NT-v2 hidden size
d_text:         4096   # OLMo-1B hidden size
modalities:     [text, dna]
freeze_backbone: true
freeze_encoders: true
```

---

## 8. Suggested Upgrades

### High-Impact, Low-Effort

**1. Enable KL regularization (`grpo_beta > 0`)**

`β=0.0` means the policy can diverge arbitrarily from the SFT checkpoint. Setting `β=0.01–0.05`
would stabilize training and prevent reward hacking (e.g. collapsing to always outputting
"Answer: yes" to score 0.5 on `correctness_reward` consistently).

**2. Weight reward functions by importance**

All five rewards are equally weighted, but format compliance is a means to an end. Shifting
weight toward correctness:

```python
reward = (0.60 * correctness
        + 0.10 * xmlcount
        + 0.10 * soft_format
        + 0.10 * strict_format
        + 0.10 * concise)
```

This focuses the RL gradient on what actually matters for downstream task performance.

**3. Clip advantages**

The current implementation uses raw normalized advantages without clipping. Adding a clamp:

```python
advantages = advantages.clamp(-5.0, 5.0)
```

prevents extreme values when 7/8 completions are correct and one is catastrophically wrong,
or vice versa.

**4. Periodic reference model sync**

The reference model is frozen at Stage 2 weights for all 1000 steps. Syncing it every
100–200 steps (or using an EMA reference) would keep the KL baseline relevant as the policy
improves, making a non-zero `β` more principled.

---

### Medium Complexity

**5. DNA-grounded faithfulness reward**

None of the five current rewards verify that the reasoning actually uses the DNA input. A
faithfulness reward could check whether the completion references the variant position, allele
change, or gene name extracted from the KEGG metadata. Even a keyword-overlap heuristic
(`answer_keywords ∩ completion_words / |answer_keywords|`) would provide a signal the model
cannot game by ignoring the DNA modality entirely.

**6. Soft semantic correctness reward**

`correctness_reward` gives a binary cliff: 1.0 for exact match, 0.5 for containment, 0.0
otherwise. Replacing containment with token-level F1 (as used in SQuAD evaluation) or
normalized edit distance would provide a smoother gradient signal, especially for multi-word
pathway or disease names where partial credit is meaningful.

**7. Collapsed-group detection**

When all G=8 completions for a prompt score identically, `advantages ≈ 0` for all and the
prompt contributes no useful gradient. Detecting these "collapsed groups" (e.g.
`rewards.std() < 0.01`) and either skipping them or resampling at higher temperature improves
sample efficiency, particularly on easy KEGG examples the model already solves consistently.

---

### Architectural

**8. Entropy bonus (DAPO-style)**

With `β=0.0` and reward-driven training, the model can quickly collapse toward near-identical
completions across the G=8 samples, flattening advantages. Adding a small entropy bonus to
the loss:

```python
loss += -lambda_entropy * entropy(policy_logits)    # lambda ~0.001
```

counteracts this and maintains generation diversity throughout training.

**9. Re-enable the projector MLP activation**

The activation between `fc1` and `fc2` in `ModalityProjector` is currently commented out,
making the two-layer MLP equivalent to a single linear map. Re-enabling it (SiLU or GELU)
restores the nonlinearity and expressive capacity that the two-layer design was intended to
provide.

Source: `src/modules/projector.py`, line 161

**10. Trained reward model for correctness**

Replace `correctness_reward`'s string matching with a small reward model trained on
(DNA embeddings, completion, label) triples. This would enable partial credit for biologically
plausible but differently-phrased answers, and would generalize beyond exact label strings —
particularly important for free-form pathway descriptions where lexical matching is brittle.

---

*Generated from codebase analysis of `BaseMM_PRISM` as of 2026-06-26.*
