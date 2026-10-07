#!/usr/bin/env python3
"""One-command ablation sweep over (modality preset × experiment design).

For each (preset, design) pair, this tool generates one launcher invocation
that produces a qsub script. The launchers themselves are unmodified — this
is orchestration only.

Examples:
  # Print 4 sweep cmds without submitting (review before launch)
  python tools/run_sweep.py --preset text_image_ts --designs PRISM-IMAGE-ONLY-2N --print

  # Submit 2 designs × 2 presets = 4 jobs
  python tools/run_sweep.py \\
      --preset text_image,all6 \\
      --designs PRISM-IMAGE-ONLY-2N,PRISM-OLMO3-E2E-PROD \\
      --storage daos --nodes 2

A `--sweep-id` is auto-generated (or passed) and propagated to each run's
`exp.id` and the underlying perf.jsonl, so `tools/perf_aggregate.py
outputs/ --filter sweep_id=<id>` groups the runs.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import shlex
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
PRESETS_PATH = REPO_ROOT / "experiments" / "modality_presets.yaml"

_LAUNCHERS: dict[str, str] = {
    "daos": "tools/launch_aurora_daos.py",
    "lustre": "tools/launch_aurora.py",
    "webdataset-staged": "tools/launch_aurora_web.py",
}


def load_presets() -> dict[str, dict[str, Any]]:
    with open(PRESETS_PATH) as f:
        data = yaml.safe_load(f)
    return data.get("presets", {})


def split_csv(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def build_launcher_cmd(
    launcher: str,
    design: str,
    preset_name: str,
    modalities: list[str],
    sweep_id: str,
    extra_args: list[str],
    dry_run: bool,
) -> tuple[list[str], dict[str, str]]:
    """Construct the launcher invocation for one (design, preset) combination.

    The launcher itself handles design lookup, qsub, etc. We just supply
    the Hydra overrides that pin the modalities for this sweep cell.

    Returns:
        (cmd, extra_env): The argv list and any env vars the caller should
        merge into the subprocess environment for THIS cell only. Callers
        must NOT merge extra_env into os.environ — that would leak the
        per-cell env (e.g. DL_NUM_WORKERS=0) into later cells in the
        same sweep. Pass via subprocess.run(env={**os.environ, **extra_env}).
    """
    run_id = f"{design}-{preset_name}-{sweep_id}"
    cmd: list[str] = [
        sys.executable,
        str(REPO_ROOT / launcher),
        "--id",
        run_id,
        "--design",
        design,
    ]
    # Hydra modality override and a sweep_id label exp uses.
    modalities_str = ",".join(modalities)
    cmd.append(f"model.modalities=[{modalities_str}]")
    cmd.append(f"exp.sweep_id={sweep_id}")
    cmd.append(f"exp.preset={preset_name}")
    # Wire the per-cell dataset_overrides yaml so the dataloader actually emits
    # the requested modality. Without this, every cell loaded the global
    # datasets_config.json with all-skip=true except ts_qa, leaving only
    # time_series active no matter what model.modalities asked for. Phase 2
    # masked the bug because the WEBDATASET_LOCAL_PATH leak fed image shards
    # to every cell; PR #73 + PR #90 closed the leak and exposed it.
    smoke_preset_path = REPO_ROOT / "src" / "conf" / "data" / "per_modality_smoke" / f"{preset_name}.yaml"
    if smoke_preset_path.exists():
        # `+data=...` (append) — there's no `data:` entry in the root defaults,
        # so the bare form `data=...` fails with "No match in the defaults list".
        cmd.append(f"+data=per_modality_smoke/{preset_name}")

    # text_ts needs model.modality_start_end_token_indices for ts_qa's 2D series.
    # 50280/50281 are the <ts>/<ts/> token indices used by the
    # prism_olmo*_linear_interleaved_ts model configs. Setting per-cell so the
    # default OLMo-1B model config (which lacks them) still works for this cell.
    # Use dotted-path append form — the dict-literal form fails Hydra parsing
    # ("mismatched input '['"). The leading `+` adds the key since the default
    # model config doesn't declare it.
    if preset_name == "text_ts":
        cmd.append(
            "+model.modality_start_end_token_indices.time_series=[50280,50281]"
        )
        # Point at the local in-repo tokenizer that defines <ts>/<ts/> at
        # 50280/50281. The default OLMo-1B tokenizer doesn't have these
        # tokens, so tokenizer.decode([50280]) returns '' and
        # _process_ts_qa's `prompt.split('')` raises "empty separator".
        local_tok = REPO_ROOT / "tokenizers" / "prism-olmo-1b-interleaved"
        if local_tok.exists():
            cmd.append(f"+model.tokenizer_id={local_tok}")
        # ts_qa stores 16 vars × 256 steps per sample; at BS=8 the model
        # OOMs on XPU (UR_RESULT_ERROR_OUT_OF_RESOURCES). Drop to BS=2 for
        # this cell so the smoke fits in memory.
        cmd.append("training.batch_size=2")

    # Non-image cells use HF IterableDatasets that report n_shards <
    # world_size (ts_qa=1, ts_instruction=3, graph_captioning=1,
    # table_reasoning=4, geo_pde_synthetic=1). With data_num_workers > 0,
    # HF stops all but one dataloader worker ("Too many dataloader workers:
    # 4 (max is dataset.num_shards=1). Stopping 3 dataloader workers.")
    # and the silenced workers leave the buffer below batch_size, which
    # surfaces as "Stream X is persistently empty after 5 restarts". Force
    # single-process dataloading for these cells via DL_NUM_WORKERS=0 —
    # the bucketed-collator path at src/train.py:1009 hard-codes 4 workers
    # which overrides training.data_num_workers, so the env var is the only
    # reliable knob. We return it in extra_env so the caller can pass it
    # *only* to this cell's subprocess; mutating os.environ here would
    # leak DL_NUM_WORKERS=0 into every subsequent image cell in the sweep.
    extra_env: dict[str, str] = {}
    if preset_name in ("text_ts", "text_graph", "text_table", "text_geometry", "text_only"):
        # The Hydra override path remains for documentation / non-bucketed
        # consumers; the env var is what the bucketed-collator path reads.
        cmd.append("training.data_num_workers=0")
        # Per-cell env: launcher reads DL_NUM_WORKERS from its own os.environ
        # and exports it into the mpiexec heredoc. Scoping it to this
        # subprocess keeps later image cells parallel.
        extra_env["DL_NUM_WORKERS"] = "0"
    if dry_run:
        cmd.append("--dry-run")
    cmd.extend(extra_args)
    return cmd, extra_env


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument(
        "--preset",
        required=True,
        help="Comma-separated preset name(s) from experiments/modality_presets.yaml",
    )
    parser.add_argument(
        "--designs",
        required=True,
        help="Comma-separated design ID(s) from experiments/prism_designs.yaml",
    )
    parser.add_argument(
        "--storage",
        choices=tuple(_LAUNCHERS),
        default="daos",
        help="Which Aurora launcher to invoke",
    )
    parser.add_argument(
        "--sweep-id",
        default=None,
        help="Group label propagated to exp.sweep_id (auto-generated if omitted)",
    )
    parser.add_argument(
        "--print",
        action="store_true",
        help="Print the launcher commands without invoking them",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Pass --dry-run to each launcher (generates qsub script, doesn't submit)",
    )
    args, extra = parser.parse_known_args(argv)

    presets = load_presets()
    sweep_id = args.sweep_id or _dt.datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    launcher = _LAUNCHERS[args.storage]

    preset_names = split_csv(args.preset)
    for p in preset_names:
        if p not in presets:
            known = ", ".join(sorted(presets))
            print(f"ERROR: unknown preset {p!r}. Known: {known}", file=sys.stderr)
            return 2

    design_names = split_csv(args.designs)
    if not design_names:
        print("ERROR: --designs must list at least one design ID", file=sys.stderr)
        return 2

    print(f"# sweep_id={sweep_id}  storage={args.storage}  launcher={launcher}", file=sys.stderr)
    print(f"# {len(preset_names) * len(design_names)} runs total", file=sys.stderr)

    failures = 0
    for design in design_names:
        for preset_name in preset_names:
            # PRISM-MODALITY-SMOKE freezes backbone+encoders → text_only has
            # zero trainable params → DDP refuses. Route `text_only` only to
            # the TEXTONLY variant (unfreezes backbone) and route the TEXTONLY
            # variant only to `text_only` (parent variant is for non-text cells).
            is_textonly_design = "MODALITY-SMOKE-TEXTONLY" in design
            is_textonly_preset = preset_name == "text_only"
            if is_textonly_preset and (
                "MODALITY-SMOKE" in design and not is_textonly_design
            ):
                print(
                    f"WARN: skipping {design}/{preset_name} — "
                    f"frozen-backbone design has no trainable params; "
                    f"use PRISM-MODALITY-SMOKE-TEXTONLY-1N instead",
                    file=sys.stderr,
                )
                continue
            if is_textonly_design and not is_textonly_preset:
                print(
                    f"WARN: skipping {design}/{preset_name} — "
                    f"TEXTONLY design is only meaningful with the text_only preset",
                    file=sys.stderr,
                )
                continue
            # The `vla` preset routes through ZoneAVLATrainer which expects
            # CALVIN-style batches; non-VLA designs use a different trainer.
            # Constrain bidirectionally: `vla` runs only on VLA designs, and
            # VLA designs only run when `vla` is selected.
            is_vla_design = "VLA-CALVIN" in design or "ZONE-A-VLA" in design
            is_vla_preset = preset_name == "vla"
            if is_vla_preset and not is_vla_design:
                print(
                    f"WARN: skipping {design}/{preset_name} — "
                    f"vla preset is only valid against VLA designs",
                    file=sys.stderr,
                )
                continue
            if is_vla_design and not is_vla_preset:
                print(
                    f"WARN: skipping {design}/{preset_name} — "
                    f"VLA design requires the vla preset (routes through ZoneAVLATrainer)",
                    file=sys.stderr,
                )
                continue
            modalities = presets[preset_name]["modalities"]
            cmd, extra_env = build_launcher_cmd(
                launcher=launcher,
                design=design,
                preset_name=preset_name,
                modalities=modalities,
                sweep_id=sweep_id,
                extra_args=extra,
                dry_run=args.dry_run,
            )
            # Show env-prefix in --print mode so users see exactly what runs.
            env_prefix = " ".join(f"{k}={v}" for k, v in extra_env.items())
            if env_prefix:
                print(env_prefix + " " + " ".join(shlex.quote(c) for c in cmd))
            else:
                print(" ".join(shlex.quote(c) for c in cmd))
            if args.print:
                continue
            try:
                # Per-cell env merge: extra_env is scoped to this subprocess
                # only — do NOT mutate os.environ (would leak into later cells).
                cell_env = {**os.environ, **extra_env} if extra_env else None
                subprocess.run(cmd, check=True, env=cell_env)
            except subprocess.CalledProcessError as e:
                print(f"FAILED ({e.returncode}): {design}/{preset_name}", file=sys.stderr)
                failures += 1
    if failures:
        print(f"# {failures} of {len(design_names) * len(preset_names)} runs failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
