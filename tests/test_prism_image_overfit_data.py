import hashlib
import io
import json
import tarfile

import pytest
from PIL import Image
from tools.prepare_prism_image_overfit_data import prepare_pack, source_groups


def _image(color):
    image = Image.new("RGB", (16, 16), color)
    data = io.BytesIO()
    image.save(data, format="PNG")
    return data.getvalue()


def _shard(tmp_path, directory, rows):
    folder = tmp_path / directory
    folder.mkdir(exist_ok=True)
    path = folder / "pixmo.tar"
    with tarfile.open(path, "w") as archive:
        for key, color, caption, url in rows:
            for suffix, data in (
                ("png", _image(color)),
                ("txt", caption.encode()),
                ("json", json.dumps({"image_url": url}).encode()),
            ):
                member = tarfile.TarInfo(f"{key}.{suffix}")
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
    return path


def _row(
    key="0001",
    color="red",
    caption="A red object on a flat background.\n",
    url="https://example.org/a.jpg",
):
    return key, color, caption, url


def test_original_caption_bytes_and_manifest_paths_are_preserved(tmp_path):
    caption = "  A red object on a flat background.\n"
    train = _shard(tmp_path, "shards", [_row(caption=caption)])
    validation = _shard(
        tmp_path, "val_shards", [_row("0002", "blue", url="https://example.org/b.jpg")]
    )
    output = tmp_path / "pack"
    report = prepare_pack(train, validation, output, train_count=1, validation_count=1)
    rows = [json.loads(line) for line in (output / "manifest.jsonl").read_text().splitlines()]
    assert rows[0]["prompt"] == caption
    assert (output / "source_sidecars" / "pixmo-train-0001.txt").read_bytes() == caption.encode()
    assert report["counts"] == {"train": 1, "validation": 1}
    assert all((output / row["target_image"]).is_file() for row in rows)
    assert report["examples"][0]["caption_sha256"] == hashlib.sha256(caption.encode()).hexdigest()
    assert "parent pretraining overlap unknown" in report["held_out_scope"]
    # Same source bytes yield byte-identical manifests in a second output directory.
    report2 = prepare_pack(
        train, validation, tmp_path / "second", train_count=1, validation_count=1
    )
    assert report2["manifest_sha256"] == report["manifest_sha256"]
    with pytest.raises(ValueError, match="never overwritten"):
        prepare_pack(train, validation, output, train_count=1, validation_count=1)


def test_cross_split_duplicate_pixels_and_source_urls_are_rejected(tmp_path):
    train = _shard(tmp_path, "shards", [_row()])
    validation = _shard(
        tmp_path,
        "val_shards",
        [
            _row("0002", "blue"),  # same source URL
            _row("0003", "red", url="https://example.org/c.jpg"),  # same RGB pixels
            _row("0004", "green", url="https://example.org/d.jpg"),
        ],
    )
    report = prepare_pack(train, validation, tmp_path / "pack", train_count=1, validation_count=1)
    assert report["examples"][1]["source_key"] == "0004"
    assert report["exclusions"]["source_or_subject_group_duplicate"] == 1
    assert report["exclusions"]["decoded_pixel_duplicate"] == 1


def test_short_caption_filter_never_truncates_long_caption(tmp_path):
    train = _shard(
        tmp_path,
        "shards",
        [_row(caption="word " * 81), _row("0002", "green", url="https://example.org/b.jpg")],
    )
    validation = _shard(
        tmp_path, "val_shards", [_row("0003", "blue", url="https://example.org/c.jpg")]
    )
    report = prepare_pack(train, validation, tmp_path / "pack", train_count=1, validation_count=1)
    assert report["examples"][0]["source_key"] == "0002"
    assert report["exclusions"]["caption_word_budget"] == 1


def test_real_video_and_flickr_grouping():
    a = source_groups({"image_url": "https://i.ytimg.com/vi/ABCD/maxresdefault.jpg"})
    b = source_groups({"image_url": "https://img.youtube.com/vi/ABCD/1.jpg"})
    assert "youtube-video:ABCD" in set(a).intersection(b)
    a = source_groups({"image_url": "https://live.staticflickr.com/12/999_abc_b.jpg"})
    b = source_groups({"image_url": "https://live.staticflickr.com/12/999_abc_c.jpg"})
    assert "flickr-photo:999" in set(a).intersection(b)


def test_missing_provenance_and_budget_exhaustion_fail_without_partial_pack(tmp_path, monkeypatch):
    import tools.prepare_prism_image_overfit_data as prepare

    train = _shard(tmp_path, "shards", [_row()])
    validation = _shard(tmp_path, "val_shards", [_row("0002", "blue", url="")])
    output = tmp_path / "pack"
    with pytest.raises(ValueError, match="insufficient eligible"):
        prepare_pack(train, validation, output, train_count=1, validation_count=1)
    assert not output.exists()
    monkeypatch.setattr(prepare, "MAX_IMAGE_BYTES", 1)
    with pytest.raises(ValueError, match="image read budget"):
        prepare_pack(train, validation, output, train_count=1, validation_count=1)
    assert not output.exists()


def test_rejects_unofficial_split_directories_and_escaping_member_names(tmp_path):
    train = _shard(tmp_path, "shards", [_row("../../escape")])
    validation = _shard(
        tmp_path, "val_shards", [_row("0002", "blue", url="https://example.org/b.jpg")]
    )
    output = tmp_path / "pack"
    prepare_pack(train, validation, output, train_count=1, validation_count=1)
    assert (output / "assets" / "pixmo-train-escape.png").is_file()
    assert not (tmp_path / "escape.png").exists()
    with pytest.raises(ValueError, match="distinct"):
        prepare_pack(train, train, tmp_path / "other", train_count=1, validation_count=1)
    renamed = validation.rename(tmp_path / "other.tar")
    with pytest.raises(ValueError, match="explicit"):
        prepare_pack(train, renamed, tmp_path / "other", train_count=1, validation_count=1)


def test_selection_continues_across_train_shards_in_declared_order(tmp_path):
    first = _shard(tmp_path, "shards", [_row("0001")])
    first = first.rename(first.with_name("first.tar"))
    second = _shard(tmp_path, "shards", [_row("0002", "green", url="https://example.org/b.jpg")])
    validation = _shard(
        tmp_path, "val_shards", [_row("0003", "blue", url="https://example.org/c.jpg")]
    )
    report = prepare_pack(
        [first, second], validation, tmp_path / "pack", train_count=2, validation_count=1
    )
    assert [example["source_key"] for example in report["examples"]] == ["0001", "0002", "0003"]
    assert [source["path"] for source in report["sources"]] == [
        str(first),
        str(second),
        str(validation),
    ]


@pytest.mark.parametrize("second_key", ["0001", "different-directory/0001"])
def test_duplicate_selected_ids_fail_before_creating_pack(tmp_path, second_key):
    first = _shard(tmp_path, "shards", [_row("0001")])
    first = first.rename(first.with_name("first.tar"))
    second = _shard(
        tmp_path,
        "shards",
        [_row(second_key, "green", url="https://example.org/b.jpg")],
    )
    validation = _shard(
        tmp_path, "val_shards", [_row("0003", "blue", url="https://example.org/c.jpg")]
    )
    source_bytes = {path: path.read_bytes() for path in (first, second, validation)}
    output = tmp_path / "pack"
    with pytest.raises(ValueError, match="duplicate selected sample ID: pixmo-train-0001"):
        prepare_pack([first, second], validation, output, train_count=2, validation_count=1)
    assert not output.exists()
    assert all(path.read_bytes() == content for path, content in source_bytes.items())


def test_reviewed_rejection_and_parent_manifest_keep_labels_out_of_input(tmp_path):
    train = _shard(
        tmp_path, "shards", [_row("0001"), _row("0002", "green", url="https://example.org/b.jpg")]
    )
    validation = _shard(
        tmp_path, "val_shards", [_row("0003", "blue", url="https://example.org/c.jpg")]
    )
    output = tmp_path / "pack"
    report = prepare_pack(
        train,
        validation,
        output,
        train_count=1,
        validation_count=1,
        excluded_keys={"0001": "caption names the wrong object"},
    )
    assert report["examples"][0]["source_key"] == "0002"
    assert report["quality_review_exclusions"] == {"0001": "caption names the wrong object"}
    parent = json.loads((output / "parent-cases.jsonl").read_text())
    assert set(parent) == {"id", "prompt", "source_images"}
    assert parent["prompt"] == "Describe the main objects in this image."
    assert parent["source_images"] == ["assets/pixmo-validation-0003.png"]
    assert len((output / "train.jsonl").read_text().splitlines()) == 1
    assert len((output / "validation.jsonl").read_text().splitlines()) == 1
    for name, expected in report["auxiliary_manifest_sha256"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == expected
