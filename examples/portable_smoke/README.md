# Portable smoke check

A PRISM training step you can run on a laptop. No HPC, no scheduler, no DAOS, no
institutional filesystem, no HuggingFace account, and no network access at all.

```bash
python examples/portable_smoke/train_tiny.py
```

It takes a few seconds and prints a loss curve.

## What it proves, and what it does not

It trains a **~90K-parameter** model on **random noise** for 40 steps. The loss
drops because the model memorises 4 fixed nonsense sequences. That is the point:
memorising noise is the cheapest possible proof that the plumbing is connected
end to end.

**It proves** that the pieces fit together on your machine and are wired
correctly — the multimodal batch assembles, the time-series encoder and its
projector produce tokens, those tokens interleave with text in the backbone, the
loss is finite, gradients reach every trainable parameter, and the optimiser
moves them.

**It does not prove** anything about model quality, and it is not a benchmark.
Nothing here is pretrained; the backbone is 53K randomly-initialised weights
synthesised on the spot. A real PRISM run is described in
[docs/training/cli.md](../../docs/training/cli.md).

## Why it can run offline

Every PRISM preset points `backbone_id` at a real HuggingFace model — a 7B OLMo
by default — and `src/model.py` loads it with `local_files_only=True`. On a
fresh clone with a cold cache that fails, which is why none of the documented
training commands work on a laptop.

This example sidesteps that by **synthesising its own backbone**: it builds a
1-layer, 64-hidden Qwen3 with random weights plus a matching word-level
tokenizer, saves them to a temporary directory (~220 KB), and points
`llm_backbone_id` at that directory. To PRISM this is an ordinary local model
path. Nothing in `src/` is stubbed, patched, or monkeypatched — the example is
just a user with a small model on disk.

The script then **blocks the network at the socket layer** and **redirects the
HuggingFace cache** to an empty temp directory, which it asserts is still empty
when the run finishes — so "it ran offline" is verified rather than asserted.

The redirect has to move more than the environment variables. `huggingface_hub`
resolves `HF_HUB_CACHE` and `HF_HOME` once, at import time, and both it and
`transformers` then hold their own copies of the result; on this host that is
eight bindings across five modules. Setting the environment after those imports
moves none of them, so an environment-only redirect would leave a hub id
resolving happily against whatever the developer has cached locally — and the
end-of-run file count would still read zero, because a cache *read* writes
nothing. `hf_cache_redirected()` therefore rebinds the module globals too, and
restores them on exit. `test_a_hub_id_cannot_resolve_from_the_ambient_cache`
holds that line: it fails if the redirect ever weakens back to environment-only.

The block itself is **probed, not trusted**. A block that silently failed to
arm — a renamed attribute, a `socket` import that resolves somewhere else —
would let the run print "network blocked" and pass anyway, carried by a warm
cache or a live connection, which would make the offline claim self-certifying.
So before training starts the script calls its own patched entry points and
requires them to raise. It costs about a microsecond when armed, because the
patch raises before any lookup happens. The name it probes is under RFC 6761's
`.invalid`, which can never resolve to a real host, and resolution is checked
before any connection is attempted — so even a completely unarmed run stops at
`NXDOMAIN` without opening a socket to anything.

## Making it fail on purpose

A check nobody has watched fail is not a check. Two flags sabotage the run in
different ways, and both must exit 1:

| Flag | What breaks | Loss drop |
|---|---|---|
| *(none)* | nothing — the real run | **0.999** ✅ |
| `--break-training` | learning rate zeroed; nothing can move | 0.000 ❌ |
| `--break-wiring` | backbone frozen; gradients reach only PRISM's own layers | 0.039 ❌ |

The threshold is **0.30**, and the second control is the one that justifies it.
A dead model scoring zero only shows the comparison runs. The realistic
regression is a model that trains but is not fully connected — it still looks
alive, its loss still falls. Measured, it falls 3.9%, an order of magnitude
short of the bar, while a healthy run clears it by a factor of three. So the
threshold separates *plausibly broken* from *working*, not merely *dead* from
*alive*.

## Two portability defects this example had to work around

Both are real, both are documented here rather than hidden, and both are worth
knowing if you run PRISM outside a cluster.

1. **The backbone loads in `float16` on CPU, and `float16` training NaNs
   immediately.** `src/model.py:88-99` selects `bfloat16` only for CUDA-with-bf16
   or Intel XPU, and falls back to `float16` everywhere else — including plain
   CPU, where the comment's rationale ("fit a 30B model in RAM") does not apply
   to training. Inference is fine: measured drift against `float32` is 4e-4. But
   the first backward pass produces a non-finite gradient and the loss is `nan`
   by step 1. The example calls `.float()` after construction and prints a note
   when it does.

2. **`freeze_backbone` and `freeze_encoders` both default to `True`**
   (`src/config.py:181-182`), so a default-constructed model has **zero**
   trainable parameters and the optimiser raises
   `ValueError: optimizer got an empty parameter list`. The example sets both to
   `False` explicitly and asserts the trainable count is non-zero before
   training, so this failure mode can never masquerade as a pass.

## Running it in CI

`tests/test_portable_smoke.py` imports this script and runs it in-process, so it
is covered by the standard `unit` job. It is not a subprocess: subprocess
execution records no coverage.

Ruff lints this directory, but nothing else in CI reads it — mypy reads
`files = ["src"]`, pytest collects `testpaths = ["tests"]`, and coverage
measures `--cov=src`. So lint aside, that test file is the only thing keeping
this example honest, which is why it asserts on measurements rather than on
exit codes.

It also tests the network probe from both sides: silent when the block is
armed, and raising when the block lets traffic through or fails in some other
way. Those two failure branches are simulated by patching the socket entry
points to succeed, not by calling the probe unpatched — an unpatched call would
issue a real DNS query from a required CI job, and a slow resolver would turn a
gate into a flake.

Three of its assertions are regression guards with measured margins:

- **under 1M parameters** — a default-dimension model is ~3.7B and has OOM-ed
  this runner before;
- **under 16 MB of weights and buffers** — a parameter count cannot see buffers
  or dtype. On the backbone-less path each block registers a `max_seq_len ** 2`
  float32 causal mask (268 MB per layer at the 8192 default) while the parameter
  count stays *identical*: measured 91,664 at both 512 and 8192. The two bounds
  therefore fail on different regressions.
- **over 60K parameters** — the synthesised Qwen3 alone is 53,456, so a
  regression to training a bare HuggingFace model with no PRISM layers in the
  path cannot pass while still looking green.

One thing worth knowing if you copy `TINY_DIMS` elsewhere: `max_seq_len=512` is
defensive here rather than load-bearing. With a backbone present PRISM delegates
attention to it and never builds its own blocks — measured, the only buffer in
this model is the backbone's 4-element `rotary_emb.inv_freq` — so the 8192
default would cost nothing *on this path*. On the backbone-less path it would
cost 268 MB per layer.
