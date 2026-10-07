import hashlib
import io
import json
import tarfile

import pytest
import torch
from PIL import Image
from src.data.image_generation import ImageGenerationCollator
from src.data.image_generation_webdataset import ImageGenerationWebDataset
from tools.prepare_docci_webdataset import convert_docci


def _jpeg(color):
    buffer = io.BytesIO()
    Image.new("RGB", (12, 8), color).save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


def _fixture(tmp_path, *, duplicate=False, omit=False, extra_member=None):
    caption = "A full caption with “quotes”, a newline\n" + "red blue green " * 100
    rows, images = [], {}
    for number, split in enumerate(("train", "train", "qual_dev", "test", "qual_test")):
        identifier = f"{split}_{number:05d}"
        rows.append(
            {
                "example_id": identifier,
                "image_file": identifier + ".jpg",
                "split": split,
                "description": caption + str(number),
            }
        )
        color = (20 + 30 * number, 50, 180)
        if duplicate and split == "qual_dev":
            color = (20, 50, 180)
        images[identifier + ".jpg"] = _jpeg(color)
    descriptions = tmp_path / "descriptions.jsonlines"
    descriptions.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    archive = tmp_path / "images.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        for name, content in list(images.items())[int(omit) :]:
            info = tarfile.TarInfo("images/" + name)
            info.size = len(content)
            handle.addfile(info, io.BytesIO(content))
        if extra_member is not None:
            info = tarfile.TarInfo(extra_member)
            info.size = 1
            handle.addfile(info, io.BytesIO(b"x"))
    return descriptions, archive, rows, images


class _Tokenizer:
    def __call__(self, texts, *, padding, truncation, return_tensors):
        assert not truncation
        ids = torch.ones(len(texts), max(len(t.split()) for t in texts), dtype=torch.long)
        return {"input_ids": ids, "attention_mask": ids.clone()}


def test_full_roundtrip_preserves_bytes_captions_and_official_splits(tmp_path):
    descriptions, archive, original, images = _fixture(tmp_path)
    output = tmp_path / "out"
    result = convert_docci(descriptions, archive, output, expected_counts=None, shard_records=1)
    assert result["official_split_counts"] == {"train": 2, "qual_dev": 1, "test": 1, "qual_test": 1}
    assert result["images_archive_sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
    assert result["validation"]["all_images_decoded"] == 5
    assert len(result["shards"]) == 5
    train = ImageGenerationWebDataset(output / "train.jsonl", target_size=(16, 20))
    validation = ImageGenerationWebDataset(output / "validation.jsonl", split="validation")
    assert len(train) == 2 and len(validation) == 1
    assert train.data_fingerprint == validation.data_fingerprint == result["data_fingerprint"]
    assert train.records[0].prompt == original[0]["description"]
    assert {record.id for record in train.records} == {r["example_id"] for r in original[:2]}
    record = train.records[0]
    with tarfile.open(record.shard) as handle:
        assert handle.extractfile(record.member).read() == images[record.member]
        assert handle.extractfile(record.id + ".txt").read().decode() == original[0]["description"]
        assert json.load(handle.extractfile(record.id + ".json"))["source_images"] == []
    sample = train[0]
    assert sample["target_image"].shape == (3, 16, 20)
    assert sample["target_image"].min() >= -1 and sample["target_image"].max() <= 1
    assert sample["encoder_source_images"] == sample["reference_images"] == []
    batch = ImageGenerationCollator(_Tokenizer())([sample, train[1]])
    assert "target_image" not in batch["inputs"] and "image" not in batch["inputs"]
    assert batch["targets"]["image"].shape == (2, 3, 16, 20)


@pytest.mark.parametrize(
    "options,match",
    [
        ({"duplicate": True}, "split leakage"),
        ({"omit": True}, "missing 1 described"),
        ({"extra_member": "../escape.jpg"}, "unsafe archive"),
        ({"extra_member": "images/extra.jpg"}, "no unique description"),
    ],
)
def test_failed_conversion_never_publishes_partial_output(tmp_path, options, match):
    descriptions, archive, _, _ = _fixture(tmp_path, **options)
    output = tmp_path / "out"
    with pytest.raises(ValueError, match=match):
        convert_docci(descriptions, archive, output, expected_counts=None)
    assert not output.exists()
    assert not list(tmp_path.glob(".out.building-*"))
    assert not (tmp_path.parent / "escape.jpg").exists()


def test_official_split_count_enforced_before_writing(tmp_path):
    descriptions, archive, _, _ = _fixture(tmp_path)
    with pytest.raises(ValueError, match="split counts"):
        convert_docci(descriptions, archive, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_related_cluster_overlap_is_reported_without_changing_official_splits(tmp_path):
    descriptions, archive, rows, _ = _fixture(tmp_path)
    metadata = tmp_path / "metadata.jsonlines"
    metadata.write_text(
        "".join(
            json.dumps(
                {
                    "example_id": row["example_id"],
                    "cluster_id": "130",
                    "entity_tags": ["red car"],
                    "image_width": 12,
                    "image_height": 8,
                    "cloud_vision_api_responses": {"large_unused": "metadata"},
                }
            )
            + "\n"
            for row in rows
        )
    )
    output = tmp_path / "out"
    summary = convert_docci(descriptions, archive, output, expected_counts=None, metadata=metadata)
    assert summary["metadata_sha256"] == hashlib.sha256(metadata.read_bytes()).hexdigest()
    overlaps = summary["validation"]["related_clusters_crossing_official_splits"]
    assert overlaps == [
        {"group": "docci-cluster:130", "splits": ["qual_test", "test", "train", "validation"]}
    ]
    assert summary["validation"]["entity_tags_crossing_official_splits"][0]["entity"] == "red car"
    dataset = ImageGenerationWebDataset(output / "train.jsonl")
    assert len(dataset) == 2
    row = json.loads((output / "train.jsonl").read_text().splitlines()[0])
    assert row["related_group_ids"] == ["docci-cluster:130"]
    assert "cloud_vision_api_responses" not in row["source_metadata"]
    assert "docci-cluster:130" not in row["group_ids"]


def test_metadata_must_join_every_description_once(tmp_path):
    descriptions, archive, _, _ = _fixture(tmp_path)
    metadata = tmp_path / "metadata.jsonlines"
    metadata.write_text(json.dumps({"example_id": "train_00000", "cluster_id": "1"}) + "\n")
    with pytest.raises(ValueError, match="metadata missing"):
        convert_docci(
            descriptions, archive, tmp_path / "out", expected_counts=None, metadata=metadata
        )


def test_loader_detects_changed_index_and_changed_image_bytes(tmp_path):
    descriptions, archive, _, _ = _fixture(tmp_path)
    output = tmp_path / "out"
    convert_docci(descriptions, archive, output, expected_counts=None)
    train = ImageGenerationWebDataset(output / "train.jsonl")
    record = train.records[0]
    with record.shard.open("r+b") as handle:
        handle.seek(record.data_offset + 20)
        handle.write(b"CORRUPTED")
    with pytest.raises(ValueError, match="checksum mismatch"):
        train[0]
    with (output / "test.jsonl").open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="index hash mismatch"):
        ImageGenerationWebDataset(output / "train.jsonl")


def _resign(output):
    path = output / "conversion.json"
    summary = json.loads(path.read_text())
    for detail in summary["indexes"].values():
        detail["sha256"] = hashlib.sha256((output / detail["path"]).read_bytes()).hexdigest()
    del summary["data_fingerprint"]
    canonical = (json.dumps(summary, sort_keys=True, ensure_ascii=False) + "\n").encode()
    summary["data_fingerprint"] = hashlib.sha256(canonical).hexdigest()
    path.write_text(json.dumps(summary))


@pytest.mark.parametrize(
    "change,match",
    [
        ({"shard": "../outside.tar"}, "unsafe dataset path"),
        ({"header_offset": -512, "data_offset": 0}, "out-of-bounds"),
        ({"data_offset": 42}, "out-of-bounds"),
        ({"size": 2**50}, "out-of-bounds"),
        ({"source_images": ["target.jpg"]}, "target-separated"),
        ({"member": "../image.jpg"}, "member name"),
    ],
)
def test_loader_rejects_unsafe_or_invalid_locators(tmp_path, change, match):
    descriptions, archive, _, _ = _fixture(tmp_path)
    output = tmp_path / "out"
    convert_docci(descriptions, archive, output, expected_counts=None)
    index = output / "train.jsonl"
    rows = [json.loads(line) for line in index.read_text().splitlines()]
    rows[0].update(change)
    index.write_text("".join(json.dumps(row) + "\n" for row in rows))
    _resign(output)
    with pytest.raises(ValueError, match=match):
        ImageGenerationWebDataset(index)


def test_loader_checks_actual_tar_header_before_pixels(tmp_path):
    descriptions, archive, _, _ = _fixture(tmp_path)
    output = tmp_path / "out"
    convert_docci(descriptions, archive, output, expected_counts=None)
    train = ImageGenerationWebDataset(output / "train.jsonl")
    record = train.records[0]
    wrong = tarfile.TarInfo("wrong.jpg")
    wrong.size = record.size
    with record.shard.open("r+b") as handle:
        handle.seek(record.header_offset)
        handle.write(wrong.tobuf())
    with pytest.raises(ValueError, match="header mismatch"):
        train[0]
