from unittest.mock import patch

import pytest
import torch
from src.data.image_transforms import build_image_transform


class _Image:
    def convert(self, mode):
        assert mode == "RGB"
        return self


class _Processor:
    def __call__(self, images, return_tensors):
        assert return_tensors == "pt"
        return {"pixel_values": torch.ones(1, 3, 4, 4)}


class _AutoImageProcessor:
    from_pretrained_calls = []

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        cls.from_pretrained_calls.append((args, kwargs))
        return _Processor()


class _FailingAutoImageProcessor:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        raise OSError("missing")


def test_build_image_transform_uses_configured_processor():
    _AutoImageProcessor.from_pretrained_calls.clear()
    with patch(
        "src.data.image_transforms._load_auto_image_processor",
        return_value=_AutoImageProcessor,
    ):
        transform = build_image_transform(
            processor_id="google/siglip2-so400m-patch14-384",
            strict_processor=True,
        )

    out = transform(_Image())
    assert out.shape == (3, 4, 4)
    assert _AutoImageProcessor.from_pretrained_calls == [
        (
            ("google/siglip2-so400m-patch14-384",),
            {"trust_remote_code": True, "local_files_only": True},
        )
    ]


def test_build_image_transform_strict_processor_failure_is_fatal():
    with (
        patch(
            "src.data.image_transforms._load_auto_image_processor",
            return_value=_FailingAutoImageProcessor,
        ),
        pytest.raises(RuntimeError, match="Could not load image processor"),
    ):
        build_image_transform(
            processor_id="google/siglip2-so400m-patch14-384",
            strict_processor=True,
        )
