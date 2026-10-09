from unittest.mock import patch

import pytest
from src.hf_cache import load_cached_tokenizer, resolve_hf_hub_cache

pytestmark = pytest.mark.unit


def test_resolve_hf_hub_cache_prefers_hub_cache(monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", "/shared/huggingface/hub")
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", "/other/huggingface/hub")
    monkeypatch.setenv("TRANSFORMERS_CACHE", "/tmp/huggingface/hub")

    assert resolve_hf_hub_cache() == "/shared/huggingface/hub"


def test_load_cached_tokenizer_uses_local_snapshot_path(monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", "/shared/huggingface/hub")

    with (
        patch(
            "huggingface_hub.snapshot_download",
            return_value="/shared/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/abc",
        ) as snapshot_download,
        patch("transformers.AutoTokenizer.from_pretrained") as from_pretrained,
    ):
        load_cached_tokenizer("Qwen/Qwen3-0.6B", trust_remote_code=True)

    snapshot_download.assert_called_once_with(
        repo_id="Qwen/Qwen3-0.6B",
        cache_dir="/shared/huggingface/hub",
        local_files_only=True,
    )
    from_pretrained.assert_called_once_with(
        "/shared/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/abc",
        trust_remote_code=True,
        local_files_only=True,
    )


def test_load_cached_tokenizer_preserves_local_path(tmp_path):
    with (
        patch("huggingface_hub.snapshot_download") as snapshot_download,
        patch("transformers.AutoTokenizer.from_pretrained") as from_pretrained,
    ):
        load_cached_tokenizer(str(tmp_path), trust_remote_code=True)

    snapshot_download.assert_not_called()
    from_pretrained.assert_called_once_with(
        str(tmp_path),
        trust_remote_code=True,
        local_files_only=True,
    )