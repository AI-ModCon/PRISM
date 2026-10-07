---
name: Bug report
about: Report a defect in PRISM
title: "[BUG] "
labels: bug
assignees: ''

---

## Describe the Bug

A clear and concise description of what the bug is.

## Steps to Reproduce

1. Command or script run (include the full `prism ...` invocation or launcher call)
2. Config used (Hydra config name plus any command-line overrides)
3. What you did next
4. See error

## Expected Behavior

What you expected to happen.

## Actual Behavior

What actually happens instead. Include the traceback or log excerpt, with any
institutional paths, hostnames, or credentials redacted.

## Environment

- Platform: [Aurora / Polaris / Perlmutter / CUDA workstation / CPU-only / other]
- Accelerator and count: [e.g. 12x Intel Max 1550, 1x A100, none]
- OS: [e.g. SUSE 15.4, Ubuntu 22.04, macOS 14]
- Python version: [e.g. 3.12]
- PyTorch version and build: [e.g. 2.8.0+xpu from frameworks/2025.3.1]
- PRISM revision: [git SHA or tag]
- Install method: [`tools/build_aurora_env.sh`, `tools/setup_deepspeed_env.sh`,
  `tools/setup_polaris_env.sh`, `requirements/base.txt` + `pip install -e . --no-deps`,
  other]

## Modality and Component

Which parts are involved, if known: [text / image / time series / geometry /
graph / table], [encoder / model / training / data / CLI / launcher / docs].

## Additional Context

Anything else that would help — scale of the run, whether it reproduces on a
single rank, whether it is a regression from a known-good revision.
