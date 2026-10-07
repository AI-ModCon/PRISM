---
name: prism-env-build
description: >
  Build and pack PRISM's Python environment on Aurora (the venv tarball shipped
  to compute nodes). Use when creating/rebuilding the venv, updating the lockfile,
  debugging a torch/import shadow, or understanding the do-NOT install rules.
  Triggers: "build the env", "setup_deepspeed_env", "build_aurora_env",
  "deepspeed_env.tar.gz", "packed venv", "torch shadowing / undefined symbol",
  "uv pip compile", "pyg / torch_scatter", "module load frameworks order".
compatibility: Aurora login node (UAN). Build once, ship the tarball to nodes.
metadata:
  version: "1.0"
  project: prism
  facility: alcf
---

# PRISM — Environment Build (Aurora)

The launchers run training inside a **packed venv extracted to `/tmp` on each
compute node**. You build it once on the UAN and tar it. The venv inherits
Aurora's XPU-patched torch/IPEX from the `frameworks` module via
`--system-site-packages` — the whole build is about *not shadowing* that torch.
For generic import-precedence debugging, see the installed
**`python-env-shadowing-hpc`** skill.

## Build once, then pack

```bash
bash tools/setup_deepspeed_env.sh          # → .venv-deepspeed (delegates to build_aurora_env.sh)
tar -czf deepspeed_env.tar.gz .venv-deepspeed
```

`build_aurora_env.sh` (the real builder): loads `frameworks/2025.3.1`, creates the
venv with `--system-site-packages -p $SYSTEM_PYTHON`, installs from the lockfile
+ a `--no-deps` tail, installs `walrus` editable (`--no-deps`), verifies every
modality import, and writes `PRISM_BUILD_INFO` + `PRISM_BUILD_PIP_FREEZE.txt`.
It **refuses to overwrite** an existing venv unless `--rebuild`.

## Module load ORDER matters

```
module load frameworks/2025.3.1   →   source .venv-deepspeed/bin/activate
```

Reversing this breaks `torch_geometric`. Always frameworks first, activate second.

## The do-NOT rules (each shadows or breaks XPU torch)

- **No PyG C++ wheels** (`pyg_lib`, `torch_scatter`, `torch_sparse`, …) — they
  fail to link against Aurora's XPU torch. Install pure-Python `torch_geometric`
  only.
- **No bundled Triton** in the packed venv — it shadows the system
  `pytorch_triton_xpu`.
- **`transformers` only ever with `--no-deps`** (pip must not re-resolve torch).
  The default build installs none at all — frameworks/2025.3.1's system 4.57.6
  already clears OLMo-3's ≥4.57.0 floor. Intern-S2 Preview needs ≥5.2.0, which
  breaks the system vLLM (`transformers<5`), so it is an opt-in overlay:
  `bash tools/build_aurora_env.sh --intern-s2`. The two are mutually exclusive;
  build them on separate `VENV_PATH`s.
- **`walrus` + `the_well` with `--no-deps`** (avoid transitive torch wheels).
- **`--system-site-packages` is required** — the venv must see the framework torch.

## The lockfile trap

**`uv pip compile` ignores `--system-site-packages`** and will resolve torch 2.12
+ nvidia/cu12 wheels that shadow XPU torch. Generate the lockfile by *pip install
+ freeze delta* instead, not `uv pip compile`. The build's verification block
asserts `torch` resolves from **system** site-packages (not the venv) — that
check is what catches a shadow.

## Local sanity (login node, no XPU)

For quick pytest/import checks on the login node, the system `python3` is 3.6 —
use the project venv python (3.12), e.g.
`/flare/ModCon/ngetty/venvs/torchtune-pt-nightly-xpu/bin/python`.

## Don't delete these during cleanup

`deepspeed_env.tar.gz` is actively shipped to compute nodes by the launcher —
not a stale artifact.

## See also

- CLAUDE.md "Build & Setup". Requirements: `requirements/{aurora-py3.12.lock,
  aurora-py3.12.nodeps,aurora-py3.12.intern-s2.nodeps,deepspeed,base}.txt`.
- [prism-platforms](../prism-platforms/SKILL.md) (per-platform env),
  [prism-launching-jobs](../prism-launching-jobs/SKILL.md) (`--packed-env`).
- Generic installed skill: `python-env-shadowing-hpc`.
