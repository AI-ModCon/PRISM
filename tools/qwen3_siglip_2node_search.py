#!/usr/bin/env python3
"""Generate two-node Qwen3/SigLIP throughput-search launcher commands.

The default mode only prints commands. Use ``--mode dry-run`` to ask the
Aurora DAOS launcher to generate job scripts, or ``--mode submit`` to qsub.
"""

from __future__ import annotations

import argparse
import math
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT_FILE = "experiments/qwen3_siglip_2node_search.yaml"
DEFAULT_DESIGN = "PRISM-QWEN3-0P6B-SIGLIP2-BASE-2N"
# sandeep/hub is searched last because it currently contains a stale
# .no_exist/tapas/.../model.safetensors marker that crashes offline mode
# with "'NoneType' has no attribute 'endswith'". Not a problem for SigLIP2/
# Qwen3, but keeping it last avoids surprising other models that might land
# here later.
DEFAULT_HF_FALLBACK_DIRS = (
    "/flare/ModCon/sww/huggingface,"
    "/flare/ModCon/sww/huggingface/hub,"
    "/flare/ModCon/ngetty/huggingface/hub,"
    "/flare/ModCon/sandeep/hub"
)


@dataclass(frozen=True)
class SearchRun:
    name: str
    design: str
    strategy: str
    batch_size: int
    max_seq_length: int
    extra_overrides: tuple[str, ...] = ()


STRATEGY_FLAGS: dict[str, tuple[str, ...]] = {
    "ddp": ("--dist-strategy", "ddp"),
    # ZeRO-1/2 use AllReduce/ReduceScatter for gradients, not AllGather, so
    # the DAOS-17499 libpil4dfs hang (AllGather-specific) doesn't apply.
    # ZeRO-3 does AllGather and the launcher auto-enables --no-pil4dfs for it.
    "zero1": ("--deepspeed", "1"),
    "zero2": ("--deepspeed", "2"),
    "fsdp_shard_grad": (
        "--dist-strategy",
        "fsdp",
        "--fsdp-sharding",
        "shard_grad_op",
        "--fsdp-production-mode",
        "--grad-ckpt-freq",
        "2",
        "--no-pil4dfs",
    ),
    "fsdp_full": (
        "--dist-strategy",
        "fsdp",
        "--fsdp-sharding",
        "full_shard",
        "--fsdp-production-mode",
        "--grad-ckpt-freq",
        "2",
        "--no-pil4dfs",
    ),
    "hsdp_shard_grad": (
        "--dist-strategy",
        "hsdp",
        "--fsdp-sharding",
        "shard_grad_op",
        "--fsdp-production-mode",
        "--grad-ckpt-freq",
        "2",
        "--no-pil4dfs",
    ),
}


def _run_name(design: str, strategy: str, batch_size: int, seq: int) -> str:
    short = design.replace("PRISM-", "").replace("-SIGLIP2-", "-S2-")
    return f"Q3S-2N-{short}-{strategy}-bs{batch_size}-s{seq}".replace("_", "-")


def build_phase(phase: str, design: str) -> list[SearchRun]:
    """Return the ordered search matrix for a named phase."""
    runs: list[SearchRun] = []

    if phase in ("ddp-batch", "all"):
        # First pass: establish the DDP ceiling for the smallest model.
        # 512 is the practical caption/pointing default; 1024 checks longer
        # samples without pulling in the full long-doc scientific mix.
        for seq, batches in ((512, (4, 8, 12, 16, 24)), (1024, (4, 8, 12, 16))):
            for bs in batches:
                runs.append(
                    SearchRun(
                        name=_run_name(design, "ddp", bs, seq),
                        design=design,
                        strategy="ddp",
                        batch_size=bs,
                        max_seq_length=seq,
                    )
                )

    if phase in ("parallelism", "all"):
        # Second pass: fixed conservative shape, compare communication/sharding
        # overhead. Do not use torch.compile here; multi-node compile is known
        # to hang on Aurora XPU collectives.
        for strategy in (
            "ddp",
            "zero1",
            "zero2",
            "fsdp_shard_grad",
            "fsdp_full",
            "hsdp_shard_grad",
        ):
            runs.append(
                SearchRun(
                    name=_run_name(design, strategy, 8, 512),
                    design=design,
                    strategy=strategy,
                    batch_size=8,
                    max_seq_length=512,
                )
            )

    if phase in ("scale-up", "all"):
        # Third pass: take the likely-good DDP shape to larger towers. These
        # are intentionally separate from the initial 0.6B search so failures
        # do not hide the baseline result.
        for scale_design, bs in (
            ("PRISM-QWEN3-0P6B-SIGLIP2-SO400M-2N", 8),
            ("PRISM-QWEN3-1P7B-SIGLIP2-BASE-2N", 8),
            ("PRISM-QWEN3-4B-SIGLIP2-BASE-2N", 4),
        ):
            for strategy in ("ddp", "hsdp_shard_grad"):
                runs.append(
                    SearchRun(
                        name=_run_name(scale_design, strategy, bs, 512),
                        design=scale_design,
                        strategy=strategy,
                        batch_size=bs,
                        max_seq_length=512,
                    )
                )

    return runs


def build_command(
    run: SearchRun,
    *,
    experiment_file: str,
    nodes: int,
    dataset_groups: str,
    max_steps: int,
    target_effective_tokens: int,
    tokens_per_sample: int,
    sweep_id: str,
    queue: str,
    project: str,
    walltime: str,
    hf_fallback_dirs: str,
    mode: str,
) -> list[str]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "launch_aurora_daos.py"),
        "--file",
        experiment_file,
        "--design",
        run.design,
        "--id",
        run.name,
        "--nodes",
        str(nodes),
        "--dataset-groups",
        dataset_groups,
        "--max-steps",
        str(
            steps_for_run(
                run,
                nodes=nodes,
                max_steps=max_steps,
                target_effective_tokens=target_effective_tokens,
                tokens_per_sample=tokens_per_sample,
            )
        ),
        "--max-seq-length",
        str(run.max_seq_length),
        "--queue",
        queue,
        "--project",
        project,
        "--walltime",
        walltime,
        "--hf-fallback-dirs",
        hf_fallback_dirs,
        "--use-bucketing",
        "--bucket-buffer-size",
        "5000",
        "--benchmark-mode",
        *STRATEGY_FLAGS[run.strategy],
        f"training.batch_size={run.batch_size}",
        f"exp.sweep_id={sweep_id}",
        f"exp.preset={run.strategy}",
        *run.extra_overrides,
    ]
    if mode == "dry-run":
        cmd.append("--dry-run")
    elif mode == "submit":
        cmd.append("--batch")
    return cmd


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=["ddp-batch", "parallelism", "scale-up", "all"],
        default="ddp-batch",
        help="Search phase to generate.",
    )
    parser.add_argument("--design", default=DEFAULT_DESIGN)
    parser.add_argument("--file", default=DEFAULT_EXPERIMENT_FILE)
    parser.add_argument("--nodes", type=int, default=2)
    parser.add_argument("--dataset-groups", default="pixmo")
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument(
        "--target-effective-tokens",
        type=int,
        default=0,
        help=(
            "If >0, compute max_steps per run so nodes*12*batch_size*"
            "tokens_per_sample*steps approximates this token budget. "
            "Use for fair time-to-N-token comparisons."
        ),
    )
    parser.add_argument(
        "--tokens-per-sample",
        type=int,
        default=260,
        help=(
            "Estimated effective multimodal tokens/sample. For SigLIP2-base "
            "PixMo, 260 ~= 196 image patch tokens + ~64 text tokens."
        ),
    )
    parser.add_argument(
        "--sweep-id",
        default=None,
        help="Label written into perf.jsonl for aggregation. Defaults to qwen3_siglip_2n_<phase>.",
    )
    parser.add_argument("--queue", default="debug")
    parser.add_argument("--project", default="AuroraGPT")
    parser.add_argument("--walltime", default="01:00:00")
    parser.add_argument(
        "--hf-fallback-dirs",
        default=DEFAULT_HF_FALLBACK_DIRS,
        help="Comma-separated HF cache roots searched if DAOS model staging lacks a model.",
    )
    parser.add_argument(
        "--mode",
        choices=["print", "dry-run", "submit"],
        default="print",
        help="print commands, generate scripts, or submit batch jobs.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Run only the first N commands after matrix construction.",
    )
    args = parser.parse_args(argv)

    runs = build_phase(args.phase, args.design)
    if args.limit:
        runs = runs[: args.limit]
    sweep_id = args.sweep_id or f"qwen3_siglip_2n_{args.phase}"

    if not runs:
        print("No runs selected", file=sys.stderr)
        return 2

    print(
        f"# phase={args.phase} runs={len(runs)} mode={args.mode} sweep_id={sweep_id}",
        file=sys.stderr,
    )
    failures = 0
    for run in runs:
        cmd = build_command(
            run,
            experiment_file=args.file,
            nodes=args.nodes,
            dataset_groups=args.dataset_groups,
            max_steps=args.max_steps,
            target_effective_tokens=args.target_effective_tokens,
            tokens_per_sample=args.tokens_per_sample,
            sweep_id=sweep_id,
            queue=args.queue,
            project=args.project,
            walltime=args.walltime,
            hf_fallback_dirs=args.hf_fallback_dirs,
            mode=args.mode,
        )
        print(" ".join(shlex.quote(part) for part in cmd))
        if args.mode == "print":
            continue
        try:
            subprocess.run(cmd, cwd=REPO_ROOT, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"FAILED {run.name}: exit={exc.returncode}", file=sys.stderr)
            failures += 1
    return 1 if failures else 0


# Image patch tokens per design's vision tower. SigLIP2-base-patch16-224
# emits 14x14=196 patches; SigLIP2-so400m-patch14-384 emits 27x27=729.
# Used by steps_for_run() to size effective-token budgets per variant.
_PATCH_TOKENS_BY_DESIGN: dict[str, int] = {
    "PRISM-QWEN3-0P6B-SIGLIP2-BASE-2N": 196,
    "PRISM-QWEN3-0P6B-SIGLIP2-SO400M-2N": 729,
    "PRISM-QWEN3-1P7B-SIGLIP2-BASE-2N": 196,
    "PRISM-QWEN3-4B-SIGLIP2-BASE-2N": 196,
}
_DEFAULT_TEXT_TOKENS_PER_SAMPLE = 64


def effective_tokens_per_sample(design: str, fallback: int) -> int:
    """Return image-patch + text tokens/sample for a design.

    Falls back to ``fallback`` (the CLI default 260 ~= 196 patch + 64 text)
    when the design isn't in the table, preserving prior behavior.
    """
    patches = _PATCH_TOKENS_BY_DESIGN.get(design)
    if patches is None:
        return fallback
    return patches + _DEFAULT_TEXT_TOKENS_PER_SAMPLE


def steps_for_run(
    run: SearchRun,
    *,
    nodes: int,
    max_steps: int,
    target_effective_tokens: int,
    tokens_per_sample: int,
) -> int:
    if target_effective_tokens <= 0:
        return max_steps
    if tokens_per_sample <= 0:
        raise ValueError("tokens_per_sample must be positive")
    per_sample = effective_tokens_per_sample(run.design, tokens_per_sample)
    world_size = nodes * 12
    tokens_per_step = world_size * run.batch_size * per_sample
    return max(1, math.ceil(target_effective_tokens / tokens_per_step))


if __name__ == "__main__":
    raise SystemExit(main())
