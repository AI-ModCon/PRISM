"""prism analyze — Analyze models and datasets.

Examples:
    prism analyze model --preset prism-olmo3-7b
    prism analyze model --preset prism-auroragpt-2b
    prism analyze dataset --daos-mount /tmp/data --samples 200
"""

import os
import subprocess
import sys

import typer

from src._repo_paths import PROJECT_ROOT, require_repo_checkout

app = typer.Typer()


@app.command()
def model(
    preset: str = typer.Option(
        None, "--preset", "-p", help="Model preset name (e.g. prism-olmo3-7b)"
    ),
    list_presets: bool = typer.Option(False, "--list", "-l", help="List all available presets"),
):
    """Analyze a model configuration: parameter counts, projector dims, encoder sizes."""
    sys.path.insert(0, PROJECT_ROOT)
    from src.config import PRISM_CONFIGS, ModelConfig

    if list_presets or preset is None:
        typer.echo("\nAvailable model presets:")
        typer.echo("-" * 60)
        for name, cfg in PRISM_CONFIGS.items():
            backbone = cfg.get("llm_backbone_id", cfg.get("hf_model_id", "custom"))
            typer.echo(f"  {name:<35} {backbone}")
        typer.echo()
        raise typer.Exit()

    if preset not in PRISM_CONFIGS:
        typer.echo(
            f"Error: Unknown preset '{preset}'. Use --list to see available presets.", err=True
        )
        raise typer.Exit(1)

    config = ModelConfig.from_preset(preset)
    typer.echo(f"\n{'=' * 50}")
    typer.echo(f"  Model Analysis: {preset}")
    typer.echo(f"{'=' * 50}")
    typer.echo(f"  Backbone        : {config.llm_backbone_id or config.hf_model_id or 'custom'}")
    typer.echo(f"  d_model         : {config.d_model}")
    typer.echo(f"  Layers          : {config.num_layers}")
    typer.echo(f"  Heads           : {config.num_heads}")
    typer.echo(f"  Vocab Size      : {config.vocab_size:,}")
    typer.echo(f"  Max Seq Len     : {config.max_seq_len}")
    typer.echo(f"  Modalities      : {[str(m) for m in config.modalities]}")
    typer.echo(f"  Freeze Backbone : {config.freeze_backbone}")
    typer.echo(f"  Freeze Encoders : {config.freeze_encoders}")

    typer.echo("\n  --- Encoder Dimensions ---")
    typer.echo(f"  d_text  : {config.d_text}")
    typer.echo(f"  d_img   : {config.d_img}")
    typer.echo(f"  d_table : {config.d_table}")
    typer.echo(f"  d_ts    : {config.d_ts}")
    typer.echo(f"  d_geo   : {config.d_geo}")
    typer.echo(f"  d_graph : {config.d_graph}")

    if config.is_timeseries:
        typer.echo("\n  --- Timeseries Settings ---")
        typer.echo(f"  ts_variates          : {config.ts_variates}")
        typer.echo(f"  max_ts_length        : {config.max_ts_length}")
        typer.echo(f"  ts_projector         : {config.ts_projector}")

    if config.is_interleaved_qa:
        typer.echo("\n  --- Interleaving ---")
        typer.echo(f"  Interleaved QA : {config.is_interleaved_qa}")
        typer.echo(f"  Token Indices  : {config.modality_start_end_token_indices}")

    typer.echo(f"{'=' * 50}\n")


@app.command()
def dataset(
    daos_mount: str = typer.Option(None, "--daos-mount", help="DAOS mount path for dataset"),
    dataset_groups: str = typer.Option(
        "all", "--dataset-groups", "-g", help="Comma-separated dataset groups or 'all'"
    ),
    samples: int = typer.Option(100, "--samples", "-n", help="Samples per dataset to analyze"),
    tokenizer: str = typer.Option(
        "allenai/OLMo-2-1124-7B-Instruct", "--tokenizer", "-t", help="Tokenizer for length analysis"
    ),
):
    """Analyze dataset length distributions and statistics."""
    require_repo_checkout("prism analyze dataset")
    script = os.path.join(PROJECT_ROOT, "tools/analyze_dataset_lengths.py")
    if not os.path.exists(script):
        typer.echo(f"Error: Analysis script not found: {script}", err=True)
        raise typer.Exit(1)

    cmd = [
        sys.executable,
        script,
        "--dataset-groups",
        dataset_groups,
        "--samples-per-dataset",
        str(samples),
        "--tokenizer",
        tokenizer,
    ]
    if daos_mount:
        cmd.extend(["--daos-mount", daos_mount])

    typer.echo(f"Analyzing datasets: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    raise typer.Exit(result.returncode)
