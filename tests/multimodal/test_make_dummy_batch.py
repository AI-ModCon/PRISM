"""Tests for make_dummy_batch (src/modalities.py)."""
import pytest
import torch
from src.modalities import ALL_MODALITIES, Modality, make_dummy_batch


def test_image_dummy_shape():
    t = make_dummy_batch(Modality.IMAGE)
    assert isinstance(t, torch.Tensor)
    assert t.shape == (3, 224, 224)


def test_table_dummy_is_long():
    t = make_dummy_batch(Modality.TABLE)
    assert t.dtype == torch.long
    assert t.shape == (128,)


def test_time_series_dummy_shape():
    t = make_dummy_batch(Modality.TIME_SERIES)
    assert t.shape == (64, 1)


def test_geometry_dummy_shape():
    t = make_dummy_batch(Modality.GEOMETRY)
    assert t.shape == (1, 10)


def test_graph_dummy_is_dict():
    g = make_dummy_batch(Modality.GRAPH)
    assert isinstance(g, dict)
    assert g["x"].shape == (128, 32)
    assert g["edge_index"].shape == (2, 0)
    assert g["edge_index"].dtype == torch.long


def test_text_dummy_is_none():
    """Text dummy is handled by the tokenizer downstream, not here."""
    assert make_dummy_batch(Modality.TEXT) is None


def test_accepts_string_input():
    t = make_dummy_batch("image")
    assert t.shape == (3, 224, 224)


def test_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown modality"):
        make_dummy_batch("audio")


@pytest.mark.parametrize("m", ALL_MODALITIES, ids=[m.value for m in ALL_MODALITIES])
def test_all_modalities_handled(m):
    """Every Modality enum value must have a branch in make_dummy_batch.

    Adding a new modality without a corresponding branch would hit the
    `raise AssertionError(...)` exhaustiveness guard — this test forces
    that contract at test time instead of runtime.
    """
    make_dummy_batch(m)
