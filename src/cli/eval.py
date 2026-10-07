"""prism eval — Run model evaluation and benchmarks.

Wraps tools/universal_evaluator.py and tools/benchmark_throughput.py.

Examples:
    prism eval run --checkpoint checkpoints/step_10000
    prism eval run --checkpoint ckpt --mode inspect_train --limit 50
    prism eval throughput --model 1b --seq-len 256 --num-steps 10
"""

import os
import subprocess
import sys

import typer

from src._repo_paths import PROJECT_ROOT, require_repo_checkout

app = typer.Typer()


@app.command("run")
def run_eval(
    checkpoint: str = typer.Option(..., "--checkpoint", "-c", help="Path to model checkpoint"),
    mode: str = typer.Option(
        "run_eval",
        "--mode",
        "-m",
        help=(
            "Eval mode: run_eval, inspect_train, inspect_eval, verify_image, "
            "verify_timeseries_scits, verify_timeseries_interleave"
        ),
    ),
    limit: int = typer.Option(0, "--limit", "-l", help="Limit number of samples (0 = default)"),
    save_dir: str = typer.Option(None, "--save-dir", help="Directory to save outputs"),
    visualize: bool = typer.Option(False, "--visualize", help="Enable visualization plotting"),
    viz_dir: str = typer.Option(None, "--viz-dir", help="Directory to save visualizations"),
    exhaustive: bool = typer.Option(False, "--exhaustive", help="Iterate ALL datasets"),
    data_only: bool = typer.Option(False, "--data-only", help="Skip model loading"),
    modality: str = typer.Option(
        None, "--modality", help="Filter by modality name (e.g. 'Vision')"
    ),
    backbone: str = typer.Option("allenai/OLMo-7B-0724-hf", "--backbone", help="Backbone model ID"),
    validation: bool = typer.Option(False, "--validation", help="Use validation data"),
):
    """Run model evaluation using the universal evaluator."""
    require_repo_checkout("prism eval run")
    script = os.path.join(PROJECT_ROOT, "tools/universal_evaluator.py")
    if not os.path.exists(script):
        typer.echo(f"Error: Evaluator not found: {script}", err=True)
        raise typer.Exit(1)

    cmd = [sys.executable, script, "--checkpoint", checkpoint, "--mode", mode]
    if limit > 0:
        cmd.extend(["--limit", str(limit)])
    if save_dir:
        cmd.extend(["--save_dir", save_dir])
    if visualize:
        cmd.append("--visualize")
    if viz_dir:
        cmd.extend(["--viz_dir", viz_dir])
    if exhaustive:
        cmd.append("--exhaustive")
    if data_only:
        cmd.append("--data-only")
    if modality:
        cmd.extend(["--modality", modality])
    if backbone != "allenai/OLMo-7B-0724-hf":
        cmd.extend(["--backbone", backbone])
    if validation:
        cmd.append("--validation")

    typer.echo(f"Running evaluation: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    raise typer.Exit(result.returncode)


@app.command()
def throughput(
    test: str = typer.Option("all", "--test", help="Comma-separated tests or 'all'"),
    model: str = typer.Option(None, "--model", help="Model size: 1b or 7b"),
    seq_len: int = typer.Option(256, "--seq-len", help="Sequence length"),
    num_steps: int = typer.Option(10, "--num-steps", help="Number of benchmark steps"),
    output: str = typer.Option("benchmark_results.json", "--output", "-o", help="Output JSON file"),
):
    """Run throughput benchmarks."""
    require_repo_checkout("prism eval throughput")
    script = os.path.join(PROJECT_ROOT, "tools/benchmark_throughput.py")
    if not os.path.exists(script):
        typer.echo(f"Error: Benchmark script not found: {script}", err=True)
        raise typer.Exit(1)

    cmd = [
        sys.executable,
        script,
        "--test",
        test,
        "--seq-len",
        str(seq_len),
        "--num-steps",
        str(num_steps),
        "--output",
        output,
    ]
    if model:
        cmd.extend(["--model", model])

    typer.echo(f"Running throughput benchmark: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    raise typer.Exit(result.returncode)
