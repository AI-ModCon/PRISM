"""prism data — Data download, conversion, validation, and staging.

Examples:
    prism data download --zone a --output-dir data/zone_a
    prism data convert --source-format parquet --output-dir data/wds --input-dir data/parquet
    prism data validate --images-dir data/images --workers 16
    prism data stage --manifest manifest.json --shards-dir /shared --local-dir /scratch
"""

import os
import subprocess
import sys

import typer

from src._repo_paths import PROJECT_ROOT, require_repo_checkout

app = typer.Typer()


@app.command()
def download(
    zone: str = typer.Option(..., "--zone", "-z", help="Training zone: a, b, or c"),
    output_dir: str = typer.Option("data", "--output-dir", "-o", help="Output directory"),
    num_samples: int = typer.Option(
        500, "--num-samples", "-n", help="Number of samples to download"
    ),
):
    """Download datasets for a specific training zone."""
    zone_lower = zone.lower()
    if zone_lower not in ("a", "b", "c"):
        typer.echo(f"Error: Unknown zone '{zone}'. Available: a, b, c", err=True)
        raise typer.Exit(1)

    sys.path.insert(0, PROJECT_ROOT)
    typer.echo(f"Downloading Zone {zone.upper()} data to {output_dir}...")

    if zone_lower == "a":
        from src.data.download_zone_a_data import download_cc3m_subset

        download_cc3m_subset(output_dir, num_samples=num_samples)
    elif zone_lower == "b":
        from src.data.download_zone_b_data import download_scienceqa_subset

        download_scienceqa_subset(output_dir, num_samples=num_samples)
    elif zone_lower == "c":
        from src.data.download_zone_c_data import download_preference_data

        download_preference_data(output_dir, num_samples=num_samples)

    typer.echo("Download complete.")


@app.command("convert")
def convert(
    source_format: str = typer.Option(
        "arrow",
        "--source-format",
        "-s",
        help="Source format: arrow, parquet, s1mmalign, nemotron, arxiv",
    ),
    output_dir: str = typer.Option(..., "--output-dir", "-o", help="Output directory"),
    arrow_dir: str = typer.Option(None, "--arrow-dir", help="Directory with Arrow files"),
    images_dir: str = typer.Option(None, "--images-dir", help="Directory with images"),
    input_dir: str = typer.Option(None, "--input-dir", "-i", help="Input directory"),
    dataset_name: str = typer.Option(None, "--dataset-name", help="Name prefix for shards"),
    images_per_shard: int = typer.Option(1000, "--images-per-shard", help="Images per TAR shard"),
    workers: int = typer.Option(8, "--workers", "-w", help="Parallel workers"),
    val_split: float = typer.Option(0.01, "--val-split", help="Validation split ratio"),
):
    """Convert datasets to WebDataset format."""
    require_repo_checkout("prism data convert")
    script_map = {
        "arrow": "scripts/convert_to_webdataset.py",
        "parquet": "scripts/convert_parquet_to_webdataset.py",
        "s1mmalign": "scripts/convert_s1mmalign_to_webdataset.py",
        "nemotron": "scripts/convert_nemotron_to_webdataset.py",
        "arxiv": "scripts/convert_arxiv_streaming.py",
    }

    if source_format not in script_map:
        typer.echo(
            f"Error: Unknown format '{source_format}'. Available: {', '.join(script_map.keys())}",
            err=True,
        )
        raise typer.Exit(1)

    script_path = os.path.join(PROJECT_ROOT, script_map[source_format])
    if not os.path.exists(script_path):
        typer.echo(f"Error: Conversion script not found: {script_path}", err=True)
        raise typer.Exit(1)

    cmd = [sys.executable, script_path, "--output-dir", output_dir]
    if arrow_dir:
        cmd.extend(["--arrow-dir", arrow_dir])
    if images_dir:
        cmd.extend(["--images-dir", images_dir])
    if input_dir:
        cmd.extend(["--input-dir", input_dir])
    if dataset_name:
        cmd.extend(["--dataset-name", dataset_name])
    cmd.extend(
        [
            "--images-per-shard",
            str(images_per_shard),
            "--workers",
            str(workers),
        ]
    )
    if source_format == "parquet":
        cmd.extend(["--val-split", str(val_split)])

    typer.echo(f"Converting ({source_format} -> webdataset): {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    raise typer.Exit(result.returncode)


@app.command()
def validate(
    images_dir: str = typer.Option(..., "--images-dir", help="Directory containing images"),
    workers: int = typer.Option(32, "--workers", "-w", help="Number of parallel workers"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be removed"),
    remove: bool = typer.Option(False, "--remove", help="Actually remove corrupt files"),
):
    """Validate images and optionally remove corrupt files."""
    require_repo_checkout("prism data validate")
    script = os.path.join(PROJECT_ROOT, "scripts/validate_images.py")
    cmd = [sys.executable, script, "--images-dir", images_dir, "--workers", str(workers)]
    if dry_run:
        cmd.append("--dry-run")
    if remove:
        cmd.append("--remove")

    typer.echo(f"Validating images: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    raise typer.Exit(result.returncode)


@app.command()
def stage(
    manifest: str = typer.Option(..., "--manifest", help="Path to manifest.json"),
    shards_dir: str = typer.Option(..., "--shards-dir", help="Directory containing shards"),
    local_dir: str = typer.Option(..., "--local-dir", help="Local destination directory"),
    node_rank: int = typer.Option(..., "--node-rank", help="This node's rank (0-indexed)"),
    num_nodes: int = typer.Option(..., "--num-nodes", help="Total number of nodes"),
):
    """Stage shards to local storage for distributed training."""
    require_repo_checkout("prism data stage")
    script = os.path.join(PROJECT_ROOT, "scripts/stage_shards.py")
    cmd = [
        sys.executable,
        script,
        "--manifest",
        manifest,
        "--shards-dir",
        shards_dir,
        "--local-dir",
        local_dir,
        "--node-rank",
        str(node_rank),
        "--num-nodes",
        str(num_nodes),
    ]
    typer.echo(f"Staging shards: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    raise typer.Exit(result.returncode)
