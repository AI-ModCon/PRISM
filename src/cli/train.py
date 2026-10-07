"""prism train — Hydra-based training with config composition.

Forwards all arguments to the Hydra-powered train.py entry point.
Supports full Hydra override syntax and multirun sweeps.

Examples:
    prism train model=prism_olmo3_7b training=zone_a
    prism train model=prism_olmo3_7b training.batch_size=8
    prism train --multirun training.learning_rate=1e-4,1e-5,1e-6
"""

import os
import subprocess
import sys
from typing import Annotated

import typer

from src._repo_paths import PROJECT_ROOT, require_repo_checkout

app = typer.Typer(invoke_without_command=True)


@app.callback()
def train(
    # Declared as a variadic argument rather than read off `ctx.args`: a Typer group
    # leaves `ctx.args` empty and tries to resolve the first leftover token as a
    # subcommand, so every invocation in the docstring above used to die with
    # "No such command 'model=prism_olmo3_7b'".
    overrides: Annotated[
        list[str] | None,
        typer.Argument(help="Hydra overrides, e.g. model=prism_olmo3_7b training.batch_size=8"),
    ] = None,
    multirun: bool = typer.Option(False, "--multirun", "-m", help="Enable Hydra multirun sweep"),
    config_path: str = typer.Option(
        "src/conf", help="Path to Hydra config directory (relative to project root)"
    ),
    config_name: str = typer.Option("config", help="Name of the Hydra config file"),
    list_presets: bool = typer.Option(False, "--list-presets", help="List available model presets"),
):
    """Run PRISM training with Hydra config composition.

    All positional arguments are passed as Hydra overrides.
    """
    if list_presets:
        _list_presets()
        raise typer.Exit()

    require_repo_checkout("prism train")
    train_script = os.path.join(PROJECT_ROOT, "src/train.py")
    if not os.path.exists(train_script):
        typer.echo(f"Error: src/train.py not found at {train_script}", err=True)
        raise typer.Exit(1)

    cmd = [sys.executable, train_script]

    if multirun:
        cmd.append("--multirun")
    if config_path != "src/conf":
        cmd.extend(["--config-path", config_path])
    if config_name != "config":
        cmd.extend(["--config-name", config_name])

    # Forward all extra args as Hydra overrides
    cmd.extend(overrides or [])

    typer.echo(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    raise typer.Exit(result.returncode)


def _list_presets():
    """Print available model presets from src/config.py."""
    sys.path.insert(0, PROJECT_ROOT)
    from src.config import PRISM_CONFIGS

    typer.echo("\nAvailable model presets:")
    typer.echo("-" * 60)
    for name, cfg in PRISM_CONFIGS.items():
        backbone = cfg.get("llm_backbone_id", cfg.get("hf_model_id", "custom"))
        mods = cfg.get("modalities", ["text", "table", "time_series", "image", "geometry", "graph"])
        typer.echo(f"  {name:<35} backbone={backbone}")
        typer.echo(f"  {'':35} modalities={mods}")
    typer.echo()
