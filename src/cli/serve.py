"""prism serve — Serve models via Gradio UI or FastAPI.

Examples:
    prism serve ui --checkpoint checkpoints/step_10000 --port 7860
    prism serve api --port 8000
"""

import os
import subprocess
import sys

import typer

from src._repo_paths import PROJECT_ROOT, require_repo_checkout

app = typer.Typer()


@app.command()
def ui(
    checkpoint: str = typer.Option(None, "--checkpoint", "-c", help="Path to model checkpoint"),
    port: int = typer.Option(7860, "--port", "-p", help="Server port"),
    device: str = typer.Option("auto", "--device", "-d", help="Device: auto, cuda, xpu, mps, cpu"),
    share: bool = typer.Option(False, "--share", help="Create a public Gradio share link"),
):
    """Launch the Gradio Chat UI for interactive inference."""
    require_repo_checkout("prism serve ui")
    env = os.environ.copy()
    if checkpoint:
        env["PRISM_CHECKPOINT"] = checkpoint
    if device != "auto":
        env["PRISM_DEVICE"] = device

    script = os.path.join(PROJECT_ROOT, "src/ui/app.py")
    cmd = [sys.executable, script]

    typer.echo(f"Starting Gradio UI on port {port}...")
    env["GRADIO_SERVER_PORT"] = str(port)
    if share:
        env["GRADIO_SHARE"] = "1"

    result = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env)
    raise typer.Exit(result.returncode)


@app.command()
def api(
    port: int = typer.Option(8000, "--port", "-p", help="Server port"),
    host: str = typer.Option("0.0.0.0", "--host", help="Server host"),
):
    """Launch the FastAPI server (OpenAI-compatible)."""
    sys.path.insert(0, PROJECT_ROOT)

    typer.echo(f"Starting PRISM API server on {host}:{port}...")
    import uvicorn

    from src.api.server import app as fastapi_app

    uvicorn.run(fastapi_app, host=host, port=port)
