import json
from pathlib import Path

import pytest
import torch
from PIL import Image
from src.data.image_generation import (
    ImageGenerationCollator,
    ImageGenerationDataset,
    load_image_generation_manifest,
    move_image_batch_to_device,
    validate_manifest_splits,
)


def _image(root: Path, name: str, color: tuple[int, int, int]) -> str:
    Image.new("RGB", (6, 4), color).save(root / name)
    return name


def _manifest(root: Path, rows, name="data.jsonl") -> Path:
    path = root / name
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def _row(identifier="example", **updates):
    result = {
        "id": identifier,
        "task": "t2i",
        "prompt": "draw a blue circle",
        "split": "train",
        "group_ids": [f"subject:{identifier}"],
        "source_images": [],
        "target_image": "target.png",
    }
    result.update(updates)
    return result


def _source_transform(image):
    color = image.getpixel((0, 0))
    return torch.tensor(color, dtype=torch.float32).reshape(3, 1, 1).expand(3, 2, 2)


class TinyTokenizer:
    def __call__(self, texts, *, padding, truncation, return_tensors):
        assert padding and not truncation and return_tensors == "pt"
        lengths = [len(text.split()) for text in texts]
        ids = torch.zeros((len(texts), max(lengths)), dtype=torch.long)
        masks = torch.zeros_like(ids)
        for index, length in enumerate(lengths):
            ids[index, :length] = torch.arange(1, length + 1)
            masks[index, :length] = 1
        return {"input_ids": ids, "attention_mask": masks}


def test_source_processing_is_separate_from_vae_target_normalization(tmp_path):
    _image(tmp_path, "source.png", (255, 0, 0))
    _image(tmp_path, "target.png", (0, 0, 255))
    manifest = _manifest(tmp_path, [_row(task="edit", source_images=["source.png"])])
    dataset = ImageGenerationDataset(
        manifest, source_transform=_source_transform, target_size=(8, 10)
    )
    sample = dataset[0]
    assert sample["encoder_source_images"][0].shape == (3, 2, 2)
    assert sample["encoder_source_images"][0][0].eq(255).all()
    assert sample["reference_images"][0].getpixel((0, 0)) == (255, 0, 0)
    assert sample["reference_images"][0].size == (6, 4)
    assert sample["target_image"].shape == (3, 8, 10)
    assert sample["target_image"][2].eq(1).all()
    assert sample["target_image"][:2].eq(-1).all()
    batch = ImageGenerationCollator(TinyTokenizer())([sample])
    assert batch["targets"]["image"].shape == (1, 3, 8, 10)
    assert "target_image" not in batch["inputs"]
    assert "labels" not in batch["inputs"]
    assert set(batch["native_context"]["image"]) == {
        "reference_images",
        "source_ids",
        "source_mask",
    }


def test_mixed_tasks_keep_reference_order_and_explicit_padding_masks(tmp_path):
    for name, color in [
        ("red.png", (255, 0, 0)),
        ("green.png", (0, 255, 0)),
        ("target.png", (0, 0, 255)),
    ]:
        _image(tmp_path, name, color)
    rows = [
        _row("text", prompt="blue circle"),
        _row("edit", task="edit", source_images=["red.png"]),
        _row(
            "context",
            task="in_context",
            source_images=["green.png", "red.png"],
            source_ids=["background", "subject"],
            prompt="draw it",
        ),
    ]
    dataset = ImageGenerationDataset(_manifest(tmp_path, rows), source_transform=_source_transform)
    batch = ImageGenerationCollator(TinyTokenizer())([dataset[i] for i in range(3)])
    assert batch["inputs"]["image"].shape == (3, 2, 3, 2, 2)
    assert batch["inputs"]["image_mask"].tolist() == [[False, False], [True, False], [True, True]]
    assert batch["inputs"]["image"][0].eq(0).all()
    assert batch["inputs"]["image"][2, 0, 1].eq(255).all()
    assert batch["inputs"]["image"][2, 1, 0].eq(255).all()
    native = batch["native_context"]["image"]
    assert [len(refs) for refs in native["reference_images"]] == [0, 1, 2]
    assert native["source_ids"][2] == ["background", "subject"]
    assert batch["inputs"]["text_attention_mask"][0].tolist() == [1, 1, 0, 0]


def test_target_free_eval_has_no_implicit_sources_or_supervision(tmp_path):
    row = _row(task="text_to_image", target_image=None)
    dataset = ImageGenerationDataset(
        _manifest(tmp_path, [row]), source_transform=None, require_targets=False
    )
    batch = ImageGenerationCollator(TinyTokenizer())([dataset[0]])
    assert "image" not in batch["inputs"]
    assert batch["targets"] == {}
    assert batch["native_context"]["image"]["reference_images"] == [[]]
    assert batch["native_context"]["image"]["source_mask"].shape == (1, 0)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"task": "edit"}, "exactly one source"),
        ({"task": "in_context", "source_images": ["source.png"]}, "at least two"),
        ({"source_images": ["source.png"]}, "t2i examples"),
        ({"task": "edit", "source_images": ["target.png"]}, "also be a source"),
        ({"group_ids": []}, "at least one split group"),
        ({"target_image": None}, "required for training"),
    ],
)
def test_invalid_manifest_never_falls_back_to_target_sources(tmp_path, updates, message):
    _image(tmp_path, "source.png", (255, 0, 0))
    _image(tmp_path, "target.png", (0, 0, 255))
    manifest = _manifest(tmp_path, [_row(**updates)])
    with pytest.raises(ValueError, match=message):
        load_image_generation_manifest(manifest)


def test_group_split_leakage_is_checked_before_train_selection(tmp_path):
    _image(tmp_path, "target.png", (0, 0, 255))
    _image(tmp_path, "other.png", (0, 255, 255))
    rows = [
        _row("train", group_ids=["video:shared"]),
        _row(
            "validation", split="validation", target_image="other.png", group_ids=["video:shared"]
        ),
    ]
    with pytest.raises(ValueError, match="split leakage for group"):
        ImageGenerationDataset(_manifest(tmp_path, rows), source_transform=None)


def test_cross_manifest_decoded_duplicate_images_are_rejected(tmp_path):
    _image(tmp_path, "target.png", (0, 0, 255))
    # Different file encoding, identical decoded pixels.
    _image(tmp_path, "copy.bmp", (0, 0, 255))
    train = load_image_generation_manifest(_manifest(tmp_path, [_row("train")]))
    validation = load_image_generation_manifest(
        _manifest(
            tmp_path,
            [_row("validation", split="validation", target_image="copy.bmp")],
            "validation.jsonl",
        )
    )
    with pytest.raises(ValueError, match="split leakage for image_content"):
        validate_manifest_splits([*train, *validation])


def test_duplicate_target_pixels_cannot_hide_as_different_source_path(tmp_path):
    _image(tmp_path, "target.png", (0, 0, 255))
    _image(tmp_path, "copy.bmp", (0, 0, 255))
    with pytest.raises(ValueError, match="target pixels are identical"):
        ImageGenerationDataset(
            _manifest(tmp_path, [_row(task="edit", source_images=["copy.bmp"])]),
            source_transform=_source_transform,
        )


def test_evaluation_aliases_preserve_source_ids_and_group(tmp_path):
    _image(tmp_path, "source.png", (255, 0, 0))
    row = {
        "case_id": "case1",
        "task": "editing",
        "prompt": "make it blue",
        "reference_paths": ["source.png"],
        "group": "subject1",
        "split": "test",
    }
    records = load_image_generation_manifest(_manifest(tmp_path, [row]), require_targets=False)
    assert records[0].group_ids == ("subject1",)
    assert records[0].source_ids == ("source.png",)
    assert records[0].target_path is None


def test_requires_explicit_encoder_transform_and_rejects_silent_truncation(tmp_path):
    _image(tmp_path, "source.png", (255, 0, 0))
    _image(tmp_path, "target.png", (0, 0, 255))
    manifest = _manifest(tmp_path, [_row(task="edit", source_images=["source.png"])])
    with pytest.raises(ValueError, match="source_transform is required"):
        ImageGenerationDataset(manifest, source_transform=None)
    dataset = ImageGenerationDataset(manifest, source_transform=_source_transform)
    with pytest.raises(ValueError, match="truncation is not allowed"):
        ImageGenerationCollator(TinyTokenizer(), max_text_length=2)([dataset[0]])


def test_recursive_device_transfer_preserves_pil_references():
    reference = Image.new("RGB", (2, 2))
    value = {"nested": [{"tensor": torch.ones(2), "image": reference}]}
    moved = move_image_batch_to_device(value, "cpu")
    assert moved["nested"][0]["tensor"].device.type == "cpu"
    assert moved["nested"][0]["image"] is reference


def test_collator_rejects_missing_encoder_copy_of_native_reference(tmp_path):
    _image(tmp_path, "source.png", (255, 0, 0))
    _image(tmp_path, "target.png", (0, 0, 255))
    dataset = ImageGenerationDataset(
        _manifest(tmp_path, [_row(task="edit", source_images=["source.png"])]),
        source_transform=_source_transform,
    )
    sample = dataset[0]
    sample["encoder_source_images"] = []
    with pytest.raises(ValueError, match="native references, and source IDs must align"):
        ImageGenerationCollator(TinyTokenizer())([sample])


def test_data_provenance_changes_when_pixels_change_without_manifest_edit(tmp_path):
    _image(tmp_path, "target.png", (0, 0, 255))
    manifest = _manifest(tmp_path, [_row()])
    before = ImageGenerationDataset(manifest, source_transform=None)
    _image(tmp_path, "target.png", (0, 255, 0))
    after = ImageGenerationDataset(manifest, source_transform=None)
    assert before.data_fingerprint != after.data_fingerprint
    assert len(before.image_fingerprints) == 1
