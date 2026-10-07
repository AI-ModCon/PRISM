"""PRISM Unified CLI — one command to rule them all.

Usage:
    prism train model=prism_olmo3_7b training=zone_a
    prism launch --platform perlmutter --id my-experiment --nodes 4
    prism data download --zone a
    prism eval --checkpoint checkpoints/step_10000
    prism analyze model --preset prism-olmo3-7b
    prism serve ui --checkpoint checkpoints/step_10000
"""

import typer

app = typer.Typer(
    name="prism",
    help="PRISM: Foundational Multimodal Training Framework",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)


def _register_subcommands():
    from src.cli.analyze import app as analyze_app
    from src.cli.data import app as data_app
    from src.cli.eval import app as eval_app
    from src.cli.launch import app as launch_app
    from src.cli.serve import app as serve_app
    from src.cli.train import app as train_app

    app.add_typer(train_app, name="train", help="Run training (Hydra config composition)")
    app.add_typer(launch_app, name="launch", help="Generate and submit HPC jobs")
    app.add_typer(data_app, name="data", help="Data download, conversion, and validation")
    app.add_typer(eval_app, name="eval", help="Run model evaluation and benchmarks")
    app.add_typer(analyze_app, name="analyze", help="Analyze models and datasets")
    app.add_typer(serve_app, name="serve", help="Serve models via Gradio UI or API")


_register_subcommands()
