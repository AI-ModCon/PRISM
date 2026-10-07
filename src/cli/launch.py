"""prism launch — Generate and submit HPC jobs.

Wraps tools/launch_perlmutter.py, tools/launch_aurora.py, and
tools/launch_baremetal.py behind a unified interface.

Examples:
    prism launch --platform perlmutter --id my-experiment --nodes 2 --dry-run
    prism launch --platform aurora --id zone-a-run --nodes 1 --queue debug
    prism launch --platform aurora-daos --id PRISM-AURORA-ZONE-A-1B --nodes 2
    prism launch --platform aurora-web --id PRISM-IMAGE-ONLY --webdataset-dir /path/to/shards --nodes 2
    prism launch --platform baremetal --id local-run
    prism launch --platform perlmutter --id my-exp --submit -- model=prism_olmo3_7b
"""

import os
import subprocess
import sys
from typing import Annotated

import typer

from src._repo_paths import PROJECT_ROOT, require_repo_checkout

app = typer.Typer(invoke_without_command=True)


def _load_env() -> dict[str, str]:
    """Load key=value pairs from .env file in project root."""
    env_config = {}
    env_path = os.path.join(PROJECT_ROOT, ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    k, v = line.split("=", 1)
                    env_config[k.strip()] = v.strip().strip("'").strip('"')
    return env_config


_ENV = _load_env()

_LAUNCHER_MAP = {
    "perlmutter": os.path.join(PROJECT_ROOT, "tools/launch_perlmutter.py"),
    "aurora": os.path.join(PROJECT_ROOT, "tools/launch_aurora.py"),
    "aurora-daos": os.path.join(PROJECT_ROOT, "tools/launch_aurora_daos.py"),
    "aurora-web": os.path.join(PROJECT_ROOT, "tools/launch_aurora_web.py"),
    "baremetal": os.path.join(PROJECT_ROOT, "tools/launch_baremetal.py"),
}


@app.callback()
def launch(
    # Same contract as `prism train`: a Typer group leaves `ctx.args` empty and tries
    # to resolve the first leftover token as a subcommand, so the `-- model=...` form
    # this module's docstring documents needs a variadic argument to land in.
    overrides: Annotated[
        list[str] | None,
        typer.Argument(help="Hydra overrides forwarded to src/train.py, e.g. model=prism_olmo3_7b"),
    ] = None,
    platform: str = typer.Option(
        ...,
        "--platform",
        "-p",
        help="Target platform: perlmutter, aurora, aurora-daos, aurora-web, baremetal",
    ),
    id: str = typer.Option(..., "--id", help="Experiment / job ID"),
    file: str = typer.Option(
        "experiments/prism_designs.yaml", "--file", "-f", help="Experiment design YAML"
    ),
    design: str = typer.Option(None, "--design", help="Design ID in YAML (defaults to --id)"),
    nodes: int = typer.Option(1, "--nodes", "-n", help="Number of nodes"),
    # Accepted but not forwarded: no tools/launch_*.py parses `--gpus`, and the
    # callback never adds it to `cmd`. Kept so existing invocations do not start
    # erroring, with the help text no longer claiming an effect it does not have.
    gpus: int = typer.Option(
        None, "--gpus", "-g", help="Accepted for compatibility; currently ignored"
    ),
    partition: str = typer.Option(
        _ENV.get("QUEUE_NAME", "debug"), "--partition", "--queue", "-q", help="Queue/partition name"
    ),
    account: str = typer.Option(
        _ENV.get("PROJECT_ALLOCATION"), "--account", "-A", help="SLURM account / PBS project"
    ),
    time: str = typer.Option("00:30:00", "--time", "-t", help="Walltime"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print script without executing/submitting"
    ),
    submit: bool = typer.Option(False, "--submit", help="Submit the job (sbatch/qsub)"),
    native_ddp: bool = typer.Option(False, "--native-ddp", help="Use native PyTorch DDP"),
    native_fsdp: bool = typer.Option(False, "--native-fsdp", help="Use native PyTorch FSDP"),
    # DAOS-specific options (aurora-daos platform)
    daos_pool: str = typer.Option("AuroraGPT", "--daos-pool", help="DAOS pool name"),
    daos_container: str = typer.Option(
        "prism_training_data", "--daos-container", help="DAOS container for training data"
    ),
    daos_models_container: str = typer.Option(
        "prism_models", "--daos-models-container", help="DAOS container for model weights"
    ),
    dataset_groups: str = typer.Option(
        "all",
        "--dataset-groups",
        help="Dataset groups: all, pixmo, s1mmalign, nemotron, cosyn, or comma-separated",
    ),
    dataset_config: str = typer.Option(
        "src/conf/data/daos_datasets.yaml", "--dataset-config", help="Dataset configuration YAML"
    ),
    dataset_proportions: str = typer.Option(
        None,
        "--dataset-proportions",
        help="Override dataset proportions, e.g. 'dataset1:0.1,dataset2:0.2'",
    ),
    # Aurora-web specific options
    webdataset_dir: str = typer.Option(
        None, "--webdataset-dir", help="WebDataset directory (aurora-web platform)"
    ),
    packed_env: str = typer.Option(
        _ENV.get("ENV_TARBALL", "deepspeed_env.tar.gz"),
        "--packed-env",
        help="Packed env tarball path",
    ),
):
    """Generate and optionally submit HPC launch scripts.

    Any arguments after -- are forwarded as Hydra overrides to src/train.py.
    """
    require_repo_checkout("prism launch")
    platform_lower = platform.lower()
    if platform_lower not in _LAUNCHER_MAP:
        typer.echo(
            f"Error: Unknown platform '{platform}'. Available: {', '.join(_LAUNCHER_MAP.keys())}",
            err=True,
        )
        raise typer.Exit(1)

    launcher_script = _LAUNCHER_MAP[platform_lower]
    if not os.path.exists(launcher_script):
        typer.echo(f"Error: Launcher script not found: {launcher_script}", err=True)
        raise typer.Exit(1)

    cmd = [sys.executable, launcher_script, "--id", id, "--file", file, "--nodes", str(nodes)]

    if design:
        cmd.extend(["--design", design])
    if dry_run:
        cmd.append("--dry-run")
    if submit:
        # Aurora launchers use --batch for PBS script generation + qsub submission
        if platform_lower in ("aurora", "aurora-daos", "aurora-web"):
            cmd.append("--batch")
        else:
            cmd.append("--submit")

    # Platform-specific flags
    if platform_lower == "perlmutter":
        cmd.extend(["--partition", partition, "--time", time])
        if account:
            cmd.extend(["--account", account])
    elif platform_lower in ("aurora", "aurora-daos", "aurora-web"):
        cmd.extend(["--queue", partition])
        if time != "00:30:00":
            cmd.extend(["--walltime", time])
        if account:
            cmd.extend(["--project", account])
        if native_ddp:
            cmd.append("--native-ddp")
        if native_fsdp:
            cmd.append("--native-fsdp")
        # DAOS-specific flags
        if platform_lower == "aurora-daos":
            cmd.extend(["--daos-pool", daos_pool])
            cmd.extend(["--daos-container", daos_container])
            cmd.extend(["--daos-models-container", daos_models_container])
            cmd.extend(["--dataset-groups", dataset_groups])
            cmd.extend(["--dataset-config", dataset_config])
            if dataset_proportions:
                cmd.extend(["--dataset-proportions", dataset_proportions])
        # Aurora-web specific flags
        if platform_lower == "aurora-web":
            if webdataset_dir:
                cmd.extend(["--webdataset-dir", webdataset_dir])
            else:
                typer.echo("Error: --webdataset-dir is required for aurora-web platform", err=True)
                raise typer.Exit(1)
            cmd.extend(["--packed-env", packed_env])

    # Forward extra args (Hydra overrides after --)
    if overrides:
        cmd.extend(overrides)

    # Propagate .env values into subprocess environment
    env = os.environ.copy()
    for k, v in _ENV.items():
        if k not in env:
            env[k] = v

    typer.echo(f"Launching on {platform}: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env)
    raise typer.Exit(result.returncode)
