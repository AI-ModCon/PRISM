import pytest

pytestmark = [pytest.mark.unit]


def _get_env_int(env, keys, default):
    """Mirror train.py env-int parsing contract for rank/world-size fallbacks."""
    for key in keys:
        value = env.get(key, "")
        if value and value.strip():
            try:
                return int(value)
            except ValueError:
                continue
    return default


def test_env_rank_parsing_prefers_primary_keys():
    env = {
        "RANK": "5",
        "PMI_RANK": "2",
        "PALS_RANKID": "1",
    }
    assert _get_env_int(env, ["RANK", "PMI_RANK", "PALS_RANKID"], 0) == 5


def test_env_rank_parsing_skips_empty_and_invalid_values():
    env = {
        "RANK": "",
        "PMI_RANK": "not-a-number",
        "PALS_RANKID": "7",
    }
    assert _get_env_int(env, ["RANK", "PMI_RANK", "PALS_RANKID"], 0) == 7


def test_env_rank_parsing_uses_default_when_missing():
    env = {}
    assert _get_env_int(env, ["RANK", "PMI_RANK", "PALS_RANKID"], 3) == 3
