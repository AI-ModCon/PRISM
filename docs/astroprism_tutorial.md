# PRISM tutorial: align AION and Qwen3-4B on galaxy images and spectra

Train AstroPRISM on one Aurora node: a frozen AION-1 Base astronomy encoder
and a frozen Qwen3-4B language model, joined by two small projectors that
learn to describe calibrated galaxy images and DESI spectra in words.

```
 image  float32[4,160,160] ──► AION image codec ──► 576 tokens ×768 ──► image projector ───┐  (frozen AION)
 spectrum flux/ivar/mask/λ  ──► AION spectrum    ──► 273 tokens ×768 ──► spectrum projector ┤
          float32[7781]          codec                                                      ▼  spliced into the prompt
 "Describe the galaxy in this calibrated four-band image."  ────────────►  Qwen3-4B (frozen)  ──►  caption
```

Only the two projectors train — 17,059,840 parameters out of 4.45 B.

| Stage | Training config | Trains | Script |
|---|---|---|---|
| 1A. Image alignment | `astroprism_stage1a` | image projector | `experiments/astroprism_stage1a.yaml` |
| 1B. Spectrum alignment | `astroprism_stage1b_spectrum` | spectrum projector | `experiments/astroprism_stage1b_spectrum.yaml` |
| Joint | `astroprism_joint_aion` | both, alternating | `experiments/astroprism_joint_aion.yaml` |

A reference checkpoint from the joint stage is published at
[`Sand33p/AstroPRISM`](https://huggingface.co/Sand33p/AstroPRISM), trained with
exactly the config in this repo. Section 7 runs inference against it.

## Before you start

An Aurora allocation, and access to the two gated Hugging Face repos:
[`Sand33p/AstroPRISM`](https://huggingface.co/Sand33p/AstroPRISM) (weights) and
[`Sand33p/AstroPRISM`](https://huggingface.co/datasets/Sand33p/AstroPRISM)
(data). Request access on each page, then put a token in your environment:

```bash
export HF_TOKEN=hf_...        # from https://huggingface.co/settings/tokens
```

Use a **classic read** token. A fine-grained token scoped to specific repos
returns 401 on anything outside its list, which looks identical to not having
access at all.

## 1. Set up the environment

Build on a login node — it has direct internet access, while compute nodes
reach the network only through the ALCF proxy.

```bash
module use /soft/modulefiles && module load frameworks/2026.1.0
python -m venv --system-site-packages /flare/<project>/$USER/prism-envs/astroprism
source /flare/<project>/$USER/prism-envs/astroprism/bin/activate
pip install --no-deps -e .
pip install "hydra-core>=1.3" pyarrow huggingface_hub polymathic-aion==0.0.2
```

`--system-site-packages` plus `--no-deps` is the Aurora rule: the frameworks
module supplies the XPU-patched PyTorch, and a plain
`pip install -r requirements/base.txt` pulls a CUDA build from PyPI that
shadows it. Install `polymathic-aion` **without** its `torch` extra for the
same reason. See [platforms/aurora_operations.md](platforms/aurora_operations.md).

Point the caches at `/flare`; home quotas are too small for 4 B-parameter
checkpoints:

```bash
export HF_HOME=/flare/<project>/$USER/cache/model_weights
export PRISM_CKPT_ROOT=/flare/<project>/$USER/checkpoints
export ASTROPRISM_ROOT=/flare/<project>/$USER/astroprism
```

**`ASTROPRISM_ROOT` is required.** The model configs resolve the backbone from
it rather than from a Hub id, so that compute nodes load from a staged copy
instead of reaching the network:

```yaml
backbone_id: "${oc.env:ASTROPRISM_ROOT,_local/astroprism}/hf/models/qwen3-4b"
```

Stage Qwen3-4B there once, from a login node:

```bash
mkdir -p "$ASTROPRISM_ROOT/hf/models" "$ASTROPRISM_ROOT/hf/cache"
huggingface-cli download Qwen/Qwen3-4B \
    --revision 1cfa9a7208912126459214e8b04321603b3df60c \
    --local-dir "$ASTROPRISM_ROOT/hf/models/qwen3-4b"
```

Leave it unset and the configs fall back to a relative `_local/astroprism`,
which will not exist on a compute node.

## 2. Check the inputs

```bash
python -c "
import torch, src.modalities as m
print('XPU tiles  :', torch.xpu.device_count(), '(expect 12 on a compute node)')
print('modalities :', [x.value for x in m.Modality])
from src.encoders import AIONImageEncoder, AIONSpectrumEncoder, AIONMultimodalEncoder
print('AION encoders importable')
"
```

`spectrum` must appear in the modality list. **0 XPU tiles means you are on a
login node** — fine for this check and for downloads, but training needs a
compute node.

## 3. Get the data

The dataset holds 646 galaxies with captions for both modalities, split
**516 train / 65 validation / 65 test**, grouped by physical source so no
galaxy appears in two splits:

```bash
python -c "
from datasets import load_dataset
images = load_dataset('Sand33p/AstroPRISM', 'image_caption')
print({k: len(v) for k, v in images.items()})
"
```

Model inputs are the raw arrays, not the preview columns:
`assets/raw/images/image_array.npy` is `float32[N,4,160,160]` in DES-G/R/I/Z
band order, and the spectra are `float32[7781]` flux, ivar, wavelength plus a
bool mask. `candidate_index` indexes axis 0 of every array directly.

Two limits the dataset card states plainly, and they bound what any result
here means: the image captions are **synthetic weak supervision** that can
contain unsupported interpretation, and the spectrum captions are generated
from measured facts but remain **unreviewed**. The
`spectrum_caption_inspection` config is held-out and marked
`never_train` — do not train on it.

## 4. Stage 1A: image-caption alignment

```bash
python tools/launch_aurora.py \
  --file experiments/astroprism_stage1a.yaml \
  --design ASTROPRISM-STAGE1A \
  --id ASTROPRISM-STAGE1A \
  --nodes 1 --batch
```

This trains the image projector only; AION and Qwen stay frozen. `--batch`
submits with `qsub` and prints a job id. Drop `--batch` to run interactively
on a node you already hold.

Watch for the merged sequence length in the log: 576 image tokens plus the
prompt must fit `max_merged_seq_length` (2048 in the shipped config), or
examples are silently dropped.

## 5. Stage 1B: spectrum-caption alignment

```bash
python tools/launch_aurora.py \
  --file experiments/astroprism_stage1b_spectrum.yaml \
  --design ASTROPRISM-STAGE1B-SPECTRUM \
  --id ASTROPRISM-STAGE1B-SPECTRUM \
  --nodes 1 --batch
```

Same shape, 273 spectrum tokens, training the spectrum projector. Note the two
projectors are configured differently and deliberately so: the image projector
uses `norm_mode: none`, the spectrum projector `text_scaled_layernorm` with
`target_norm: 1.04`. Spectra and images do not arrive at the backbone with
comparable scales.

## 6. Joint alignment

```bash
python tools/launch_aurora.py \
  --file experiments/astroprism_joint_aion.yaml \
  --design ASTROPRISM-JOINT-AION \
  --id ASTROPRISM-JOINT-AION \
  --nodes 1 --batch
```

One AION core serves both routes, and a rank-synchronous batch sampler
alternates modalities so each projector gets the same number of updates it
would have had alone. Each example carries **one** non-text modality;
simultaneous image-plus-spectrum prompting was never trained.

The published checkpoint is this stage at 200 steps, batch 2 per rank on 12
XPU ranks, BF16, cosine LR 1e-3 with 10 warmup steps, seed 42. Its held-out
teacher-forced losses:

| Stream | Test loss |
|---|---:|
| Combined | 1.4708 |
| Image-caption | 1.5210 |
| Spectrum-caption | 1.3006 |

These measure target-token prediction, not scientific correctness.

## 7. Generate from the published checkpoint

The checkpoint is a PRISM `UnifiedTransformer`, **not** loadable with
`AutoModelForCausalLM.from_pretrained`. The runner used for the published
examples ships with the weights:

```bash
huggingface-cli download Sand33p/AstroPRISM \
  inference/generate_astroprism_joint_examples.py model.safetensors \
  --local-dir $HF_HOME/astroprism
```

It builds the pinned Qwen3-4B (`1cfa9a72…`) and AION Base (`40541618…`),
then loads `model.safetensors` through PRISM's checkpoint loader. Greedy
decoding on a held-out spectrum produces:

> At the catalog redshift z ≈ 0.113, the spectrum shows a red, rising
> continuum and a red, rising [O III] emission line.

**Read that output next to its reference before trusting the model.** The
published examples disagree with their references in both cases: the image
example calls a galaxy lenticular where the reference says elliptical, and
this spectrum asserts `[O III]` where the reference emphasizes a possible
4000 Å break. A 200-step projector alignment over 516 weakly-supervised
captions produces fluent astronomy-shaped text, and fluency is not evidence.

## 8. Troubleshooting

**`GatedRepoError` / 401.** Request access on both HF repo pages, and check
your token is classic-read rather than fine-grained:
`curl -s -H "Authorization: Bearer $HF_TOKEN" https://huggingface.co/api/whoami-v2`.

**Downloads hang or raise `LocalEntryNotFoundError`.** On a **compute node**
export `HTTPS_PROXY=http://proxy.alcf.anl.gov:3128` (and `HTTP_PROXY`). On a
**login node** do the opposite — it reaches Hugging Face directly, and setting
the proxy there makes requests time out. Check with `env | grep -i proxy`.

**`ValueError: The pinned AION image codec emits exactly 576 tokens`.** The
config validates the AION contract at construction. Changing
`aion_num_image_tokens`, `d_img`, the band order or the 160×160 cutout size
fails loudly rather than silently feeding the encoder something it was not
trained on.

**`Disk quota exceeded`.** Something is writing to `$HOME`. `HF_HOME` and
`PRISM_CKPT_ROOT` both belong on `/flare`.

**Identifiers ending in `000`.** Astronomical ids exceed JavaScript's exact
integer range; read them as strings end to end.

## Where to read further

- [applications/astroprism_training_plan.md](applications/astroprism_training_plan.md) — the full architecture and stage rationale
- [applications/astroprism_data.md](applications/astroprism_data.md) — roster provenance, pinned revisions, recovery from upstream sources
- [applications/astroprism_benchmark_landscape.md](applications/astroprism_benchmark_landscape.md) — what a real astronomy evaluation would require
- [applications/astroprism_synthetic_data_prompts.md](applications/astroprism_synthetic_data_prompts.md) — how the captions were generated
