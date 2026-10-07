import os
from pathlib import Path
from typing import Any


def resolve_hf_hub_cache() -> str | None:
    """Return the configured Hugging Face Hub cache, if any."""
    return (
        os.environ.get("HF_HUB_CACHE")
        or os.environ.get("HUGGINGFACE_HUB_CACHE")
        or os.environ.get("TRANSFORMERS_CACHE")
    )


def load_cached_tokenizer(tokenizer_id: str, **kwargs):
    """Load a tokenizer from configured caches without network access."""
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    tokenizer_path = Path(tokenizer_id).expanduser()
    if not tokenizer_path.is_dir():
        # Annotated so mypy does not infer `dict[str, str | bool | None]` from the
        # literal and then reject the heterogeneous `**` splat below.
        snapshot_kwargs: dict[str, Any] = {
            "repo_id": tokenizer_id,
            "cache_dir": resolve_hf_hub_cache(),
            "local_files_only": True,
        }
        revision = kwargs.pop("revision", None)
        if revision is not None:
            snapshot_kwargs["revision"] = revision
        tokenizer_path = Path(snapshot_download(**snapshot_kwargs))

    return AutoTokenizer.from_pretrained(
        str(tokenizer_path),
        **kwargs,
        local_files_only=True,
    )
