from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from src.data.astroprism_stage1a_collate import AstroPRISMStage1ACollator
from src.data.astroprism_stage1a_dataset import (
    ASTROPRISM_IMAGE_TOKENS,
    IGNORE_INDEX,
    AstroPRISMStage1ADataset,
    AstroPRISMStage1ADatasetConfig,
    AstroPRISMStage1ADatasetError,
)

pytestmark = [pytest.mark.unit, pytest.mark.multimodal]


class _FakeQwenTokenizer:
    pad_token_id = 0

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    @staticmethod
    def _content_ids(content: str) -> list[int]:
        return [100 + (sum(word.encode("utf-8")) % 500) for word in content.split()]

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> list[int]:
        self.calls.append(
            {
                "messages": messages,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
                "enable_thinking": enable_thinking,
            }
        )
        assert tokenize is True
        role_ids = {"system": 10, "user": 11, "assistant": 12}
        ids = [1]
        for message in messages:
            ids.append(role_ids[message["role"]])
            ids.extend(self._content_ids(message["content"]))
            ids.append(2)
        if add_generation_prompt:
            ids.append(role_ids["assistant"])
        return ids


def _candidate(
    index: int,
    *,
    status: str = "accepted",
    caption_status: str = "released_reused",
    split: str | None = "train",
    paired_targetid: str | None = None,
    caption: str | None = None,
    revised: str | None = None,
) -> dict[str, Any]:
    paired_id = paired_targetid or f"paired-{index}"
    caption_payload = {
        "original": caption or f"Galaxy {index} has a smooth central bulge.",
        "summary_original": f"Summary {index}.",
        "status": caption_status,
    }
    if revised is not None:
        caption_payload["revised"] = revised
    return {
        "schema_version": 1,
        "candidate_index": index,
        "paired_targetid": paired_id,
        "caption_object_id": f"caption-{index}",
        "source_group_id": f"source-{index}",
        "acceptance_status": status,
        "split": split,
        "caption": caption_payload,
        "caption_source": {"repo_id": "captions", "row": index},
        "paired_source": {"repo_id": "paired", "row": index},
    }


def _write_case(
    tmp_path: Path,
    records: list[dict[str, Any]] | None = None,
) -> AstroPRISMStage1ADatasetConfig:
    records = records or [
        _candidate(0, caption="Compact galaxy."),
        _candidate(1, caption="Extended galaxy with a smooth central bulge and diffuse disk."),
    ]
    raw_root = tmp_path / "raw"
    candidate_root = tmp_path / "candidates"
    raw_root.mkdir()
    candidate_root.mkdir()

    images = np.empty((len(records), 4, 160, 160), dtype=np.float32)
    for index in range(len(records)):
        images[index].fill(index + 0.25)
    np.save(raw_root / "image_array.npy", images, allow_pickle=False)

    raw_candidates = []
    for index, record in enumerate(records):
        raw_candidates.append(
            {
                "candidate_index": index,
                "paired_targetid": record["paired_targetid"],
                "caption_object_id": record["caption_object_id"],
                "acceptance_status": "pending_review",
                "split": None,
                "raw_row": 1000 + index,
                "bands": {
                    "source": ["des-g", "des-r", "des-i", "des-z"],
                    "aion": ["DES-G", "DES-R", "DES-I", "DES-Z"],
                },
                "coordinates_deg": {"provabgs": {"ra": 10.0 + index, "dec": -2.0}},
            }
        )
    raw_manifest = {
        "artifact_type": "astroprism_raw_images",
        "schema_version": 1,
        "roster": {"sha256": "same-roster"},
        "source": {"location": "fixture.hdf5"},
        "outputs": {
            "image_array": {
                "path": "image_array.npy",
                "dtype": "float32",
                "shape": list(images.shape),
            }
        },
        "candidates": raw_candidates,
    }
    (raw_root / "manifest.json").write_text(json.dumps(raw_manifest), encoding="utf-8")

    candidate_path = candidate_root / "candidates.jsonl"
    candidate_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    candidate_manifest = {
        "artifact_type": "astroprism_stage1a_candidates",
        "schema_version": 1,
        "pins": {"roster_sha256": "same-roster"},
        "outputs": {
            "candidates": {
                "path": "candidates.jsonl",
                "row_count": len(records),
            }
        },
    }
    (candidate_root / "manifest.json").write_text(json.dumps(candidate_manifest), encoding="utf-8")

    return AstroPRISMStage1ADatasetConfig(
        raw_manifest_path=raw_root / "manifest.json",
        candidates_manifest_path=candidate_root / "manifest.json",
        split="train",
        max_merged_seq_length=1024,
    )


def test_emits_raw_image_qwen_chat_assistant_labels_and_provenance(tmp_path: Path) -> None:
    tokenizer = _FakeQwenTokenizer()
    dataset = AstroPRISMStage1ADataset(_write_case(tmp_path), tokenizer)

    sample = dataset[0]

    assert len(dataset) == 2
    assert dataset.active_modalities == {"image"}
    assert sample["image"].shape == (4, 160, 160)
    assert sample["image"].dtype == torch.float32
    assert torch.all(sample["image"] == torch.tensor(0.25))
    assert sample["text"].dtype == torch.long
    assert sample["labels"].shape == sample["text"].shape
    assert torch.equal(sample["text_attention_mask"], torch.ones_like(sample["text"]))
    first_target = int(torch.nonzero(sample["labels"] != IGNORE_INDEX)[0].item())
    assert torch.all(sample["labels"][:first_target] == IGNORE_INDEX)
    assert torch.equal(sample["labels"][first_target:], sample["text"][first_target:])
    assert sample["image_token_count"] == ASTROPRISM_IMAGE_TOKENS
    assert sample["merged_length"] == ASTROPRISM_IMAGE_TOKENS + sample["text"].numel()
    assert sample["provenance"]["paired_targetid"] == "paired-0"
    assert sample["provenance"]["raw_row"] == 1000
    assert sample["provenance"]["bands"]["aion"] == [
        "DES-G",
        "DES-R",
        "DES-I",
        "DES-Z",
    ]
    assert sample["provenance"]["caption_field"] == "original"
    assert sample["provenance"]["caption_field_config"] == "auto"
    assert sample["provenance"]["caption_status"] == "released_reused"
    assert len(tokenizer.calls) == 2
    assert all(call["enable_thinking"] is False for call in tokenizer.calls)
    assert tokenizer.calls[0]["add_generation_prompt"] is True
    assert tokenizer.calls[1]["add_generation_prompt"] is False


def test_auto_selects_revised_text_for_released_revised_caption(tmp_path: Path) -> None:
    tokenizer = _FakeQwenTokenizer()
    record = _candidate(
        0,
        caption_status="released_revised",
        caption="Original text not approved for this crop.",
        revised="Reviewed text supported by the exact crop.",
    )
    dataset = AstroPRISMStage1ADataset(_write_case(tmp_path, [record]), tokenizer)

    sample = dataset[0]

    assert tokenizer.calls[-1]["messages"][-1]["content"] == record["caption"]["revised"]
    assert sample["provenance"]["caption_field"] == "revised"
    assert sample["provenance"]["caption_status"] == "released_revised"


@pytest.mark.parametrize(
    ("config_override", "message"),
    [
        ({}, "no non-empty caption.revised"),
        ({"caption_field": "original"}, "requires caption.revised"),
        (
            {"caption_field": "original", "allow_unsafe_candidates": True},
            "requires caption.revised",
        ),
        ({"allow_unsafe_candidates": True}, "no non-empty caption.revised"),
    ],
)
def test_released_revised_never_silently_uses_original(
    tmp_path: Path,
    config_override: dict[str, Any],
    message: str,
) -> None:
    record = _candidate(
        0,
        caption_status="released_revised",
        caption="Original text is not crop-reviewed.",
    )
    config = replace(_write_case(tmp_path, [record]), **config_override)

    with pytest.raises(AstroPRISMStage1ADatasetError, match=message):
        AstroPRISMStage1ADataset(config, _FakeQwenTokenizer())


@pytest.mark.parametrize(
    ("record", "message"),
    [
        (_candidate(0, status="pending_review", split="train"), "acceptance_status"),
        (_candidate(0, status="accepted", split=None), "split=null"),
        (_candidate(0, caption_status="unreviewed", split="train"), "caption.status"),
        (_candidate(0, caption_status="rejected", split="train"), "caption.status"),
    ],
)
def test_refuses_unreviewed_or_unsplit_candidates_by_default(
    tmp_path: Path, record: dict[str, Any], message: str
) -> None:
    config = replace(_write_case(tmp_path, [record]), split=None)

    with pytest.raises(AstroPRISMStage1ADatasetError, match=message):
        AstroPRISMStage1ADataset(config, _FakeQwenTokenizer())


def test_explicit_unsafe_override_loads_pending_unsplit_fixture(tmp_path: Path) -> None:
    config = replace(
        _write_case(tmp_path, [_candidate(0, status="pending_review", split=None)]),
        split=None,
        allow_unsafe_candidates=True,
    )

    tokenizer = _FakeQwenTokenizer()
    dataset = AstroPRISMStage1ADataset(config, tokenizer)
    sample = dataset[0]

    assert len(dataset) == 1
    assert sample["provenance"]["acceptance_status"] == "pending_review"
    assert sample["provenance"]["split"] is None
    assert sample["provenance"]["caption_field"] == "original"
    assert tokenizer.calls[-1]["messages"][-1]["content"] == (
        "Galaxy 0 has a smooth central bulge."
    )


def test_split_view_ignores_explicitly_excluded_rows(tmp_path: Path) -> None:
    records = [
        _candidate(
            0,
            split="train",
            caption_status="released_revised",
            revised="Reviewed revision.",
        ),
        _candidate(1, status="rejected", split=None, caption_status="rejected"),
    ]
    dataset = AstroPRISMStage1ADataset(_write_case(tmp_path, records), _FakeQwenTokenizer())

    assert len(dataset) == 1
    assert dataset[0]["provenance"]["candidate_index"] == 0


def test_rejects_source_identity_across_splits(tmp_path: Path) -> None:
    first = _candidate(0, split="train", paired_targetid="same-object")
    second = _candidate(1, split="validation", paired_targetid="same-object")
    config = _write_case(tmp_path, [first, second])

    with pytest.raises(AstroPRISMStage1ADatasetError, match="source split leakage"):
        AstroPRISMStage1ADataset(config, _FakeQwenTokenizer())


def test_rejects_raw_caption_identity_mismatch(tmp_path: Path) -> None:
    config = _write_case(tmp_path)
    raw_manifest = json.loads(config.raw_manifest_path.read_text(encoding="utf-8"))
    raw_manifest["candidates"][1]["caption_object_id"] = "wrong-object"
    config.raw_manifest_path.write_text(json.dumps(raw_manifest), encoding="utf-8")

    with pytest.raises(AstroPRISMStage1ADatasetError, match="caption_object_id mismatch"):
        AstroPRISMStage1ADataset(config, _FakeQwenTokenizer())


def test_rejects_unverified_raw_band_order(tmp_path: Path) -> None:
    config = _write_case(tmp_path)
    raw_manifest = json.loads(config.raw_manifest_path.read_text(encoding="utf-8"))
    raw_manifest["candidates"][0]["bands"]["source"] = [
        "des-r",
        "des-g",
        "des-i",
        "des-z",
    ]
    config.raw_manifest_path.write_text(json.dumps(raw_manifest), encoding="utf-8")

    with pytest.raises(AstroPRISMStage1ADatasetError, match="unsupported band order"):
        AstroPRISMStage1ADataset(config, _FakeQwenTokenizer())


def test_merged_length_guard_raises_instead_of_truncating(tmp_path: Path) -> None:
    tokenizer = _FakeQwenTokenizer()
    config = replace(_write_case(tmp_path), max_merged_seq_length=577)
    dataset = AstroPRISMStage1ADataset(config, tokenizer)

    with pytest.raises(AstroPRISMStage1ADatasetError, match="never truncates"):
        dataset[0]

    assert all("truncation" not in call for call in tokenizer.calls)


def test_collator_pads_text_mask_and_labels_in_lockstep(tmp_path: Path) -> None:
    tokenizer = _FakeQwenTokenizer()
    config = _write_case(tmp_path)
    dataset = AstroPRISMStage1ADataset(config, tokenizer)
    samples = [dataset[0], dataset[1]]
    original_lengths = [sample["text"].numel() for sample in samples]
    collator = AstroPRISMStage1ACollator(
        tokenizer, max_merged_seq_length=config.max_merged_seq_length
    )

    batch = collator(samples)

    longest = max(original_lengths)
    shortest_index = original_lengths.index(min(original_lengths))
    shortest = original_lengths[shortest_index]
    assert batch["image"].shape == (2, 4, 160, 160)
    assert batch["text"].shape == batch["labels"].shape == (2, longest)
    assert batch["text_attention_mask"].shape == (2, longest)
    assert torch.all(batch["text"][shortest_index, shortest:] == tokenizer.pad_token_id)
    assert torch.all(batch["text_attention_mask"][shortest_index, shortest:] == 0)
    assert torch.all(batch["labels"][shortest_index, shortest:] == IGNORE_INDEX)
    assert [sample["text"].numel() for sample in samples] == original_lengths
    assert batch["image_token_count"] == ASTROPRISM_IMAGE_TOKENS
    assert batch["provenance"][1]["candidate_index"] == 1


def test_collator_refuses_oversized_sample_without_truncating(tmp_path: Path) -> None:
    tokenizer = _FakeQwenTokenizer()
    dataset = AstroPRISMStage1ADataset(_write_case(tmp_path), tokenizer)
    sample = dataset[0]
    collator = AstroPRISMStage1ACollator(
        tokenizer,
        max_merged_seq_length=sample["merged_length"] - 1,
    )

    with pytest.raises(ValueError, match="refusing to truncate"):
        collator([sample])

    assert sample["text"].numel() == sample["merged_length"] - ASTROPRISM_IMAGE_TOKENS
