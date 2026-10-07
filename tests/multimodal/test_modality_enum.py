"""Tests for the central Modality enum (src/modalities.py)."""
import pytest
from src.modalities import ALL_MODALITIES, Modality, parse_modality


def test_modality_str_equality():
    """Modality.X == 'x' so existing string-keyed dicts and configs work."""
    assert Modality.IMAGE == "image"
    assert Modality.TEXT == "text"
    assert Modality.TIME_SERIES == "time_series"
    assert "image" == Modality.IMAGE


def test_modality_membership_works_with_string_lists():
    """`if Modality.X in modalities:` works when modalities is a list[str]."""
    modalities = ["text", "image"]
    assert Modality.TEXT in modalities
    assert Modality.IMAGE in modalities
    assert Modality.GRAPH not in modalities


def test_dict_lookup_round_trip():
    """encoders[Modality.X] retrieves the same value as encoders['x']."""
    d = {"image": "encoder_obj"}
    assert d[Modality.IMAGE] == "encoder_obj"
    d2: dict = {}
    d2[Modality.GRAPH] = "graph_enc"
    assert d2["graph"] == "graph_enc"


def test_all_modalities_has_seven():
    assert len(ALL_MODALITIES) == 7
    assert set(ALL_MODALITIES) == {
        Modality.TEXT,
        Modality.IMAGE,
        Modality.TABLE,
        Modality.TIME_SERIES,
        Modality.GEOMETRY,
        Modality.GRAPH,
        Modality.DNA,
    }


def test_parse_modality_coerces_string():
    assert parse_modality("image") is Modality.IMAGE
    assert parse_modality(Modality.GRAPH) is Modality.GRAPH


def test_parse_modality_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown modality"):
        parse_modality("audio")


def test_str_returns_value():
    assert str(Modality.IMAGE) == "image"
    assert f"{Modality.IMAGE}" == "image"
