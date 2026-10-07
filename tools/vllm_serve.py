"""Wrapper around vLLM's OpenAI server that pre-registers PRISM and
attaches our custom /v1/prism/ts route.

Two responsibilities:

  1. Register the PRISM model class with vLLM's ModelRegistry. Spawn
     workers re-import this module before instantiating the engine, so
     they pick up the registration via `src.vllm_plugin.register()`'s
     module-level side effect plus the `vllm.general_plugins` entry-point
     installed by `tools/install_prism_entry_point.sh`.

  2. Monkey-patch `vllm.entrypoints.openai.api_server.build_app` to
     attach `attach_prism_routes(app)` after vLLM builds the FastAPI
     app. This adds `/v1/prism/ts` for PRISM-modality payloads that
     don't fit the OpenAI schema (raw tensors, etc.) — see
     `src/vllm_plugin/openai_schema.py`.

Usage mirrors `vllm serve`:
    python tools/vllm_serve.py --model <exported-dir> --port 8000 \\
        --trust-remote-code --enforce-eager [--no-prism-routes]
"""

from __future__ import annotations

import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

# Spawn workers inherit PYTHONPATH so they can import src.* even before the
# setuptools entry point is registered (which it should be — see
# tools/install_prism_entry_point.sh). Belt and suspenders.
_pp = os.environ.get("PYTHONPATH", "")
os.environ["PYTHONPATH"] = (
    _PROJECT_ROOT if not _pp else f"{_PROJECT_ROOT}{os.pathsep}{_pp}"
)

import src.vllm_plugin  # noqa: E402

src.vllm_plugin.register()


def _install_prism_route_patch() -> None:
    """Wrap vllm's build_app so app gets `/v1/prism/ts` after init.

    Done before run_server consumes argv. Idempotent — guards against
    double-wrapping if vllm_serve.main() is invoked more than once.
    """
    import vllm.entrypoints.openai.api_server as srv

    if getattr(srv.build_app, "_prism_patched", False):
        return

    from src.vllm_plugin.openai_schema import attach_prism_routes

    original_build_app = srv.build_app

    def patched_build_app(args):
        app = original_build_app(args)
        attach_prism_routes(app)
        return app

    patched_build_app._prism_patched = True  # type: ignore[attr-defined]
    srv.build_app = patched_build_app


def main() -> None:
    # --no-prism-routes opts out of the custom route attachment, leaving
    # the server identical to upstream `vllm serve`. Useful for A/B
    # debugging if our route ever breaks startup.
    enable_prism_routes = True
    argv = list(sys.argv[1:])
    if "--no-prism-routes" in argv:
        enable_prism_routes = False
        argv.remove("--no-prism-routes")

    if enable_prism_routes:
        _install_prism_route_patch()

    # Import after the patch so any references in api_server's module
    # globals already resolve to the patched build_app.
    # Requires vLLM >= 0.15 (frameworks/2025.3.1).
    import uvloop
    from vllm.entrypoints.openai.api_server import run_server
    from vllm.entrypoints.openai.cli_args import (
        make_arg_parser,
        validate_parsed_serve_args,
    )
    from vllm.entrypoints.utils import cli_env_setup
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    # Match upstream `python -m vllm.entrypoints.openai.api_server` pre-flight:
    # cli_env_setup() sets VLLM_WORKER_MULTIPROC_METHOD=spawn (required for XPU
    # spawn workers), and validate_parsed_serve_args() rejects flag
    # combinations like --enable-auto-tool-choice without --tool-call-parser
    # before the engine boots.
    cli_env_setup()
    parser = FlexibleArgumentParser(description="vLLM OpenAI server + PRISM routes")
    parser = make_arg_parser(parser)
    args = parser.parse_args(argv)
    validate_parsed_serve_args(args)

    try:
        uvloop.run(run_server(args))
    except KeyboardInterrupt:
        pass
    except SystemExit:
        raise
    except BaseException as exc:
        print(f"vllm_serve failed: {exc!r}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
