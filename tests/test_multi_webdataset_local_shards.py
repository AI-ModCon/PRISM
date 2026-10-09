"""Tests for MultiWebDataset's local_shards_dir mode.

This mode replaces the inline LocalShardDataset that lived in src/train.py
prior to the consolidation. It reads launcher-staged tmpfs shards
(scripts/stage_shards.py output: flat .tar files + local_manifest.json)
and partitions across local ranks instead of global ranks — required because
each Aurora node has its own node-private shard subset on /tmp.
"""

import io
import json
import os
import tarfile

import pytest

webdataset = pytest.importorskip("webdataset")
PIL = pytest.importorskip("PIL.Image")
import torch  # noqa: E402
from src.data.multi_webdataset import MultiWebDataset, MultiWebDatasetWrapper


def _make_shard(path: str, n_samples: int = 4) -> None:
    """Build a WDS-compatible .tar with .jpg / .txt / .json triples
    matching the Pixmo caption shard layout."""
    with tarfile.open(path, "w") as tar:
        for i in range(n_samples):
            key = f"{i:08d}"
            img = PIL.new("RGB", (32, 32), (i * 50 % 255, 0, 0))
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            data = buf.getvalue()
            info = tarfile.TarInfo(name=f"{key}.jpg")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))

            cap = f"sample {i} caption text".encode()
            info = tarfile.TarInfo(name=f"{key}.txt")
            info.size = len(cap)
            tar.addfile(info, io.BytesIO(cap))

            meta = json.dumps({"id": i, "source": "fixture"}).encode("utf-8")
            info = tarfile.TarInfo(name=f"{key}.json")
            info.size = len(meta)
            tar.addfile(info, io.BytesIO(meta))


@pytest.fixture
def staged_shards(tmp_path):
    """Three shards × 4 samples + local_manifest.json — mimics
    scripts/stage_shards.py output for a Pixmo caption subset."""
    for i in range(3):
        _make_shard(str(tmp_path / f"shard-{i:06d}.tar"), n_samples=4)
    (tmp_path / "local_manifest.json").write_text(
        json.dumps({
            "node_rank": 0,
            "num_nodes": 1,
            "local_dir": str(tmp_path),
            "shards": [f"shard-{i:06d}.tar" for i in range(3)],
            "total_samples_estimate": 12,
        })
    )
    return str(tmp_path)


@pytest.fixture
def fake_tokenizer():
    class _Tok:
        pad_token_id = 0

        def __call__(self, text, **kw):
            ids = torch.tensor(
                [hash(w) % 1000 for w in text.split()][: kw.get("max_length", 32)]
            )

            class _R:
                pass

            r = _R()
            r.input_ids = ids.unsqueeze(0)
            return r

    return _Tok()


@pytest.fixture
def fake_model_config():
    class _MC:
        image_processor_id = None
        d_img = 768

    return _MC()


# --- MultiWebDataset (raw) ----------------------------------------------


def test_local_shards_reads_manifest(staged_shards):
    """local_manifest.json drives shard discovery; samples have the
    standard {image, caption, metadata} shape from _process_sample."""
    ds = MultiWebDataset(
        local_shards_dir=staged_shards, world_size=1, rank=0, modality="image"
    )
    stats = ds.get_stats()
    assert stats["total_shards"] == 3
    assert stats["total_samples"] == 12

    samples = []
    it = iter(ds)
    for _ in range(3):
        samples.append(next(it))
    assert all("image" in s for s in samples)
    assert all("caption" in s for s in samples)
    assert all(s["caption"].startswith("sample ") for s in samples)
    assert all(isinstance(s["metadata"], dict) for s in samples)


def test_local_shards_falls_back_to_glob_when_manifest_missing(staged_shards):
    """When local_manifest.json is absent, glob *.tar still finds shards.
    Matches the historical LocalShardDataset behavior pre-stage_shards.py."""
    os.remove(os.path.join(staged_shards, "local_manifest.json"))
    ds = MultiWebDataset(
        local_shards_dir=staged_shards, world_size=1, rank=0, modality="image"
    )
    assert ds.get_stats()["total_shards"] == 3


def test_local_shards_forces_partition_by_local(staged_shards):
    """local_shards_dir → partition_by="local" automatically. Operators
    can't accidentally pick global partition for a per-node tmpfs layout."""
    ds = MultiWebDataset(
        local_shards_dir=staged_shards,
        world_size=4,  # try to simulate multi-rank
        rank=2,
        modality="image",
        partition_by="global",  # request global; should be overridden to local
    )
    assert ds.partition_by == "local"


def test_no_config_required_in_local_shards_mode(staged_shards):
    """The DAOS config path was always required pre-consolidation, even
    for local-only smokes that didn't use DAOS at all. Local-shards mode
    bypasses that."""
    ds = MultiWebDataset(local_shards_dir=staged_shards, world_size=1, rank=0)
    assert ds.config is None


def test_normal_mode_still_requires_config():
    """Defensive: don't accidentally let normal users skip config."""
    with pytest.raises(ValueError, match="config is required"):
        MultiWebDataset(world_size=1, rank=0)


# --- MultiWebDatasetWrapper (training-pipeline view) --------------------


def test_wrapper_yields_training_sample_shape(
    staged_shards, fake_tokenizer, fake_model_config
):
    """The wrapper must produce dicts with the exact keys the trainer's
    collator + model forward expect, matching the old LocalShardDataset:
        {image: tensor, text: tokens (unpadded), _metadata: str}
    """
    w = MultiWebDatasetWrapper(
        tokenizer=fake_tokenizer,
        local_shards_dir=staged_shards,
        world_size=1,
        rank=0,
        max_length=32,
        modalities=("image",),
        model_config=fake_model_config,
    )
    assert w.active_modalities == {"image"}

    it = iter(w)
    samples = [next(it) for _ in range(3)]
    for s in samples:
        assert set(s) == {"image", "text", "_metadata"}
        assert isinstance(s["image"], torch.Tensor)
        assert s["image"].shape == (3, 224, 224)
        assert isinstance(s["text"], torch.Tensor)
        assert s["text"].ndim == 1  # unpadded; collator pads to batch max
        assert isinstance(s["_metadata"], str)


def test_wrapper_skips_daos_config_load_when_local(
    staged_shards, fake_tokenizer, fake_model_config
):
    """Passing local_shards_dir should NOT call load_daos_config — the
    file may not exist on local-only smokes."""
    # If the wrapper tried to load src/conf/data/daos_datasets.yaml and that
    # path were unreadable, the constructor would raise. Just verify success.
    w = MultiWebDatasetWrapper(
        tokenizer=fake_tokenizer,
        local_shards_dir=staged_shards,
        world_size=1,
        rank=0,
        modalities=("image",),
        model_config=fake_model_config,
    )
    assert w.multi_ds.config is None
