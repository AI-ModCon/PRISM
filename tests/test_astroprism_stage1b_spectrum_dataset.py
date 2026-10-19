from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from src.data.astroprism_spectrum_contract import ASTROPRISM_SPECTRUM_LENGTH
from src.data.astroprism_stage1a_dataset import (
    IGNORE_INDEX,
    AstroPRISMStage1ADatasetError,
)
from src.data.astroprism_stage1b_spectrum_collate import (
    AstroPRISMStage1BSpectrumCollator,
)
from src.data.astroprism_stage1b_spectrum_dataset import (
    ASTROPRISM_SPECTRUM_TOKENS,
    AstroPRISMStage1BSpectrumDataset,
    AstroPRISMStage1BSpectrumDatasetConfig,
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


def _caption_record(index: int, *, caption: str, split: str) -> dict[str, Any]:
    is_train = split == "train"
    return {
        "schema_version": 1,
        "artifact_type": "astroprism_spectrum_caption_candidate",
        "candidate_index": index,
        "paired_targetid": f"paired-{index}",
        "caption_object_id": f"caption-{index}",
        "source_group_id": f"source-{index}",
        "split": split,
        "spectrum_caption": {
            "original": caption,
            "revised": None,
            "status": "accepted",
        },
        "review": {
            "status": "accepted",
            "decision_source": "user_instruction",
        },
        "training_policy": {
            "dataset_ready": True,
            "gradient_training_allowed": is_train,
            "eligible_for_training": is_train,
            "eligible_for_evaluation": not is_train,
            "role": f"alignment_{'training' if is_train else split}",
        },
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_case(tmp_path: Path) -> AstroPRISMStage1BSpectrumDatasetConfig:
    records = [
        _caption_record(0, caption="Spectrum caption zero.", split="train"),
        _caption_record(
            1,
            caption="A longer reviewed spectrum caption for candidate one.",
            split="train",
        ),
        _caption_record(2, caption="Validation spectrum caption.", split="validation"),
        _caption_record(3, caption="Test spectrum caption.", split="test"),
    ]
    raw_root = tmp_path / "raw"
    caption_root = tmp_path / "captions"
    raw_root.mkdir()
    caption_root.mkdir()

    pixels = np.arange(ASTROPRISM_SPECTRUM_LENGTH, dtype=np.float32)
    arrays = {
        "spectrum_flux": np.stack([pixels / 100.0 + 0.25 + index for index in range(len(records))]),
        "spectrum_ivar": np.stack(
            [
                np.full(
                    ASTROPRISM_SPECTRUM_LENGTH,
                    2.0 + index,
                    dtype=np.float32,
                )
                for index in range(len(records))
            ]
        ),
        "spectrum_mask": np.zeros((len(records), ASTROPRISM_SPECTRUM_LENGTH), dtype=np.bool_),
        "spectrum_wavelength": np.broadcast_to(
            3500.0 + pixels,
            (len(records), ASTROPRISM_SPECTRUM_LENGTH),
        ).copy(),
    }
    for index in range(len(records)):
        arrays["spectrum_mask"][index, :: 257 + index] = True
    outputs: dict[str, dict[str, Any]] = {}
    for key, array in arrays.items():
        array_path = raw_root / f"{key}.npy"
        np.save(array_path, array, allow_pickle=False)
        outputs[key] = {
            "path": array_path.name,
            "dtype": str(array.dtype),
            "shape": list(array.shape),
        }

    raw_candidates = [
        {
            "candidate_index": index,
            "paired_targetid": record["paired_targetid"],
            "caption_object_id": record["caption_object_id"],
            "acceptance_status": "pending_review",
            "split": None,
            "raw_row": 1000 + index,
            "coordinates_deg": {
                "desi": {"ra": 10.0 + index, "dec": -2.0},
            },
        }
        for index, record in enumerate(records)
    ]
    raw_manifest = {
        "artifact_type": "astroprism_raw_spectra",
        "schema_version": 1,
        "roster": {"sha256": "same-roster", "row_count": len(records)},
        "source": {"kind": "local", "location": "fixture.hdf5"},
        "spectrum": {
            "length": ASTROPRISM_SPECTRUM_LENGTH,
            "aion_token_count": ASTROPRISM_SPECTRUM_TOKENS,
            "observed_frame": True,
            "mask_true_means_excluded": True,
            "aion_modality": "DESISpectrum",
        },
        "outputs": outputs,
        "candidates": raw_candidates,
    }
    raw_manifest_path = raw_root / "manifest.json"
    raw_manifest_path.write_text(json.dumps(raw_manifest), encoding="utf-8")

    captions_path = caption_root / "captions.jsonl"
    captions_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    split_counts = {"train": 2, "validation": 1, "test": 1}
    captions_manifest = {
        "artifact_type": "astroprism_spectrum_captions_accepted",
        "schema_version": 1,
        "role": "reviewed_alignment_dataset",
        "row_count": len(records),
        "split_counts": split_counts,
        "candidate_indices": list(range(len(records))),
        "roster": {"sha256": "same-roster", "row_count": len(records)},
        "review": {
            "status": "accepted",
            "decision_source": "user_instruction",
        },
        "training_policy": {
            "dataset_ready": True,
            "gradient_training_splits": ["train"],
            "evaluation_splits": ["validation", "test"],
            "roles": {
                "train": "alignment_training",
                "validation": "alignment_validation",
                "test": "alignment_test",
            },
        },
        "splits": {"row_counts": split_counts},
        "outputs": {
            "captions.jsonl": {
                "size_bytes": captions_path.stat().st_size,
                "sha256": _sha256(captions_path),
            }
        },
    }
    captions_manifest_path = caption_root / "manifest.json"
    captions_manifest_path.write_text(json.dumps(captions_manifest), encoding="utf-8")

    return AstroPRISMStage1BSpectrumDatasetConfig(
        raw_manifest_path=raw_manifest_path,
        captions_manifest_path=captions_manifest_path,
        max_merged_seq_length=1024,
        verify_file_hashes=True,
    )


def _rewrite_caption_records(
    config: AstroPRISMStage1BSpectrumDatasetConfig,
    mutation,
    *,
    refresh_manifest: bool = True,
) -> None:
    captions_path = config.captions_manifest_path.parent / "captions.jsonl"
    records = [json.loads(line) for line in captions_path.read_text().splitlines()]
    mutation(records)
    captions_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    if refresh_manifest:
        manifest = json.loads(config.captions_manifest_path.read_text(encoding="utf-8"))
        output = manifest["outputs"]["captions.jsonl"]
        output["size_bytes"] = captions_path.stat().st_size
        output["sha256"] = _sha256(captions_path)
        config.captions_manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def _rewrite_caption_manifest(config, mutation) -> None:
    path = config.captions_manifest_path
    manifest = json.loads(path.read_text(encoding="utf-8"))
    mutation(manifest)
    path.write_text(json.dumps(manifest), encoding="utf-8")


def _set_nested(record: dict[str, Any], keys: tuple[str, ...], value: Any) -> None:
    current = record
    for key in keys[:-1]:
        current = current[key]
    current[keys[-1]] = value


def test_emits_reviewed_spectrum_caption_labels_and_grounded_provenance(
    tmp_path: Path,
) -> None:
    tokenizer = _FakeQwenTokenizer()
    dataset = AstroPRISMStage1BSpectrumDataset(_write_case(tmp_path), tokenizer)

    sample = dataset[0]

    assert len(dataset) == 2
    assert dataset.active_modalities == {"spectrum"}
    assert set(sample["spectrum"]) == {"flux", "ivar", "mask", "wavelength"}
    assert {key: value.shape for key, value in sample["spectrum"].items()} == {
        "flux": (ASTROPRISM_SPECTRUM_LENGTH,),
        "ivar": (ASTROPRISM_SPECTRUM_LENGTH,),
        "mask": (ASTROPRISM_SPECTRUM_LENGTH,),
        "wavelength": (ASTROPRISM_SPECTRUM_LENGTH,),
    }
    assert {key: value.dtype for key, value in sample["spectrum"].items()} == {
        "flux": torch.float32,
        "ivar": torch.float32,
        "mask": torch.bool,
        "wavelength": torch.float32,
    }
    assert sample["spectrum"]["flux"][0].item() == pytest.approx(0.25)
    assert sample["spectrum"]["ivar"][0].item() == pytest.approx(2.0)
    assert sample["spectrum"]["mask"][0].item() is True
    assert sample["spectrum"]["wavelength"][0].item() == pytest.approx(3500.0)

    assert sample["text"].dtype == torch.long
    assert sample["labels"].shape == sample["text"].shape
    assert torch.equal(sample["text_attention_mask"], torch.ones_like(sample["text"]))
    first_target = int(torch.nonzero(sample["labels"] != IGNORE_INDEX)[0].item())
    assert torch.all(sample["labels"][:first_target] == IGNORE_INDEX)
    assert torch.equal(sample["labels"][first_target:], sample["text"][first_target:])

    assert ASTROPRISM_SPECTRUM_TOKENS == 273
    assert sample["spectrum_token_count"] == 273
    assert sample["merged_length"] == 273 + sample["text"].numel()
    assert sample["provenance"]["raw_row"] == 1000
    assert sample["provenance"]["raw_source"] == {
        "kind": "local",
        "location": "fixture.hdf5",
    }
    assert sample["provenance"]["roster_sha256"] == "same-roster"
    assert sample["provenance"]["caption_field"] == "original"
    assert sample["provenance"]["caption_status"] == "accepted"
    assert sample["provenance"]["caption_review_status"] == "accepted"
    assert sample["provenance"]["caption_review_decision_source"] == "user_instruction"
    assert sample["provenance"]["caption_training_role"] == "alignment_training"
    assert sample["provenance"]["caption_supervision"] == "reviewed_spectrum_caption"
    assert sample["provenance"]["spectral_grounding_claimed"] is True
    assert sample["provenance"]["chat_template"] == "qwen_non_thinking"
    assert len(tokenizer.calls) == 2
    assert tokenizer.calls[1]["messages"][-1] == {
        "role": "assistant",
        "content": "Spectrum caption zero.",
    }
    assert all(call["enable_thinking"] is False for call in tokenizer.calls)
    assert tokenizer.calls[0]["add_generation_prompt"] is True
    assert tokenizer.calls[1]["add_generation_prompt"] is False


def test_current_accepted_schema_uses_original_not_revised(tmp_path: Path) -> None:
    config = _write_case(tmp_path)
    _rewrite_caption_records(
        config,
        lambda rows: rows[0]["spectrum_caption"].update(
            {"revised": "A later, unsupported revision."}
        ),
    )
    tokenizer = _FakeQwenTokenizer()

    AstroPRISMStage1BSpectrumDataset(config, tokenizer)[0]

    assert tokenizer.calls[1]["messages"][-1]["content"] == "Spectrum caption zero."


def test_merged_length_guard_counts_273_spectrum_tokens_without_truncation(
    tmp_path: Path,
) -> None:
    config = _write_case(tmp_path)
    baseline = AstroPRISMStage1BSpectrumDataset(config, _FakeQwenTokenizer())[0]
    constrained = replace(config, max_merged_seq_length=baseline["merged_length"] - 1)
    tokenizer = _FakeQwenTokenizer()
    dataset = AstroPRISMStage1BSpectrumDataset(constrained, tokenizer)

    with pytest.raises(
        AstroPRISMStage1ADatasetError,
        match=r"273 spectrum tokens.*refusing silent truncation",
    ):
        dataset[0]

    assert all("truncation" not in call for call in tokenizer.calls)


def test_collator_stacks_nested_dtypes_and_pads_labels_in_lockstep(
    tmp_path: Path,
) -> None:
    tokenizer = _FakeQwenTokenizer()
    config = _write_case(tmp_path)
    dataset = AstroPRISMStage1BSpectrumDataset(config, tokenizer)
    samples = [dataset[0], dataset[1]]
    original_lengths = [sample["text"].numel() for sample in samples]
    collator = AstroPRISMStage1BSpectrumCollator(
        tokenizer,
        max_merged_seq_length=config.max_merged_seq_length,
    )

    batch = collator(samples)

    longest = max(original_lengths)
    shortest_index = original_lengths.index(min(original_lengths))
    shortest = original_lengths[shortest_index]
    assert {key: value.shape for key, value in batch["spectrum"].items()} == {
        "flux": (2, ASTROPRISM_SPECTRUM_LENGTH),
        "ivar": (2, ASTROPRISM_SPECTRUM_LENGTH),
        "mask": (2, ASTROPRISM_SPECTRUM_LENGTH),
        "wavelength": (2, ASTROPRISM_SPECTRUM_LENGTH),
    }
    assert batch["text"].shape == batch["labels"].shape == (2, longest)
    assert batch["text_attention_mask"].shape == (2, longest)
    assert torch.all(batch["text"][shortest_index, shortest:] == tokenizer.pad_token_id)
    assert torch.all(batch["text_attention_mask"][shortest_index, shortest:] == 0)
    assert torch.all(batch["labels"][shortest_index, shortest:] == IGNORE_INDEX)
    assert batch["spectrum_token_count"] == 273
    assert torch.equal(
        batch["merged_length"],
        torch.tensor([273 + length for length in original_lengths], dtype=torch.long),
    )
    assert batch["provenance"][1]["caption_supervision"] == "reviewed_spectrum_caption"


def test_collator_rejects_wrong_nested_dtype(tmp_path: Path) -> None:
    tokenizer = _FakeQwenTokenizer()
    config = _write_case(tmp_path)
    sample = AstroPRISMStage1BSpectrumDataset(config, tokenizer)[0]
    sample["spectrum"]["mask"] = sample["spectrum"]["mask"].to(torch.float32)
    collator = AstroPRISMStage1BSpectrumCollator(
        tokenizer,
        max_merged_seq_length=config.max_merged_seq_length,
    )

    with pytest.raises(ValueError, match=r"spectrum\.mask must have dtype torch\.bool"):
        collator([sample])


def test_rejects_non_promoted_caption_manifest(tmp_path: Path) -> None:
    config = _write_case(tmp_path)
    _rewrite_caption_manifest(
        config,
        lambda manifest: manifest.update({"artifact_type": "astroprism_spectrum_caption_pilot"}),
    )

    with pytest.raises(AstroPRISMStage1ADatasetError, match="artifact_type"):
        AstroPRISMStage1BSpectrumDataset(config, _FakeQwenTokenizer())


def test_rejects_raw_caption_roster_mismatch(tmp_path: Path) -> None:
    config = _write_case(tmp_path)
    _rewrite_caption_manifest(
        config,
        lambda manifest: manifest["roster"].update({"sha256": "other-roster"}),
    )

    with pytest.raises(AstroPRISMStage1ADatasetError, match="roster SHA-256 mismatch"):
        AstroPRISMStage1BSpectrumDataset(config, _FakeQwenTokenizer())


def test_checks_source_split_integrity_across_unselected_rows(tmp_path: Path) -> None:
    config = _write_case(tmp_path)
    _rewrite_caption_records(
        config,
        lambda rows: rows[2].update({"source_group_id": rows[0]["source_group_id"]}),
    )

    with pytest.raises(AstroPRISMStage1ADatasetError, match="source split leakage"):
        AstroPRISMStage1BSpectrumDataset(config, _FakeQwenTokenizer())


@pytest.mark.parametrize(
    "keys,value,message",
    [
        (("spectrum_caption", "status"), "pending_review", "spectrum_caption.status"),
        (("review", "status"), "pending_review", "review.status"),
        (("review", "decision_source"), "heuristic", "review.decision_source"),
        (("training_policy", "dataset_ready"), False, "dataset_ready"),
        (("training_policy", "gradient_training_allowed"), False, "gradient_training_allowed"),
        (("training_policy", "eligible_for_training"), False, "eligible_for_training"),
        (("training_policy", "eligible_for_evaluation"), True, "eligible_for_evaluation"),
        (("training_policy", "role"), "alignment_validation", "training_policy.role"),
    ],
)
def test_train_view_rejects_unsafe_selected_rows(
    tmp_path: Path,
    keys: tuple[str, ...],
    value: Any,
    message: str,
) -> None:
    config = _write_case(tmp_path)
    _rewrite_caption_records(
        config,
        lambda rows: _set_nested(rows[0], keys, value),
    )

    with pytest.raises(AstroPRISMStage1ADatasetError, match=message):
        AstroPRISMStage1BSpectrumDataset(config, _FakeQwenTokenizer())


def test_train_view_does_not_apply_review_gate_to_unselected_rows(tmp_path: Path) -> None:
    config = _write_case(tmp_path)
    _rewrite_caption_records(
        config,
        lambda rows: rows[2]["spectrum_caption"].update({"status": "pending_review"}),
    )

    dataset = AstroPRISMStage1BSpectrumDataset(config, _FakeQwenTokenizer())

    assert len(dataset) == 2


def test_validation_view_requires_evaluation_policy(tmp_path: Path) -> None:
    config = replace(_write_case(tmp_path), split="validation")
    dataset = AstroPRISMStage1BSpectrumDataset(config, _FakeQwenTokenizer())

    assert len(dataset) == 1
    assert dataset[0]["provenance"]["caption_training_role"] == "alignment_validation"

    _rewrite_caption_records(
        config,
        lambda rows: rows[2]["training_policy"].update({"eligible_for_evaluation": False}),
    )
    with pytest.raises(AstroPRISMStage1ADatasetError, match="eligible_for_evaluation"):
        AstroPRISMStage1BSpectrumDataset(config, _FakeQwenTokenizer())


def test_caption_hash_verification_detects_tampering(tmp_path: Path) -> None:
    config = _write_case(tmp_path)
    _rewrite_caption_records(
        config,
        lambda rows: rows[0]["spectrum_caption"].update({"original": "Spectrum caption hero."}),
        refresh_manifest=False,
    )

    with pytest.raises(AstroPRISMStage1ADatasetError, match="SHA-256 mismatch"):
        AstroPRISMStage1BSpectrumDataset(config, _FakeQwenTokenizer())
