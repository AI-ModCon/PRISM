"""Unit tests for src/train.py helpers.

Currently tests _coerce_str_or_none, the defensive coerce that prevents
empty OmegaConf containers from polluting perf.jsonl with "sweep_id": "{}".
"""

import pytest
from src.train import _coerce_str_or_none


def test_none_passes_through():
    assert _coerce_str_or_none(None) is None


def test_plain_string_passes_through():
    assert _coerce_str_or_none("my-sweep") == "my-sweep"


def test_empty_string_passes_through_as_string():
    """Empty string is a valid (if degenerate) string; don't munge to None."""
    assert _coerce_str_or_none("") == ""


def test_empty_dict_becomes_none():
    """The pathological case: an empty container masquerading as None."""
    assert _coerce_str_or_none({}) is None


def test_empty_list_becomes_none():
    assert _coerce_str_or_none([]) is None


def test_omegaconf_empty_dictconfig_becomes_none():
    """Specifically the case observed in smoke runs."""
    omegaconf = pytest.importorskip("omegaconf")
    empty = omegaconf.OmegaConf.create({})
    assert _coerce_str_or_none(empty) is None


def test_omegaconf_empty_listconfig_becomes_none():
    omegaconf = pytest.importorskip("omegaconf")
    empty = omegaconf.OmegaConf.create([])
    assert _coerce_str_or_none(empty) is None


def test_nonempty_dict_stringifies():
    """If a real dict ever shows up, str-coerce it rather than dropping data."""
    assert _coerce_str_or_none({"a": 1}) == "{'a': 1}"


def test_integer_stringifies():
    assert _coerce_str_or_none(42) == "42"


# === Hydra wire-through regression tests ===
#
# PR-98 (commit 3ef83a6) added `eval_enabled` to TrainingConfig and
# `projector_hidden_mult` / `projector_num_layers` to ModelConfig + every
# src/conf/{training,model}/*.yaml. It missed the explicit
# `cfg.X.field` → `Config(field=...)` plumbing in src/train.py, so Hydra
# silently dropped the overrides and every run used the dataclass defaults.
#
# Without these wire-throughs, the IsoFLOP smoke runs with rotating
# variants all execute as BASE (hm=1, nl=2) and `training.eval_enabled=true`
# is ignored. Discovered during Smoke 4 replay on 2026-05-27/28.
#
# These tests are source-level (grep the file rather than invoke Hydra)
# because the alternative — spinning up an OmegaConf cfg and calling
# train.main() — drags in torch + the full repo and turns a 1-second
# unit test into a 30-second integration test.

import re
from pathlib import Path

_TRAIN_PY = Path(__file__).resolve().parents[1] / "src" / "train.py"


def _train_py_text() -> str:
    return _TRAIN_PY.read_text()


def test_train_py_wires_eval_enabled():
    """`cfg.training.eval_enabled` must reach `train_config.eval_enabled`.
    Without this line, Hydra accepts `training.eval_enabled=true` but the
    trainer sees TrainingConfig's default (False) and the eval gate never
    fires — observed during Smoke 4 replay 2026-05-27 / 2026-05-28.
    """
    text = _train_py_text()
    # The fix sets `train_config.eval_enabled = getattr(cfg.training, ...)`.
    # Match liberally on the assignment + the getattr that pulls from cfg.training.
    pattern = re.compile(
        r"train_config\.eval_enabled\s*=\s*getattr\(\s*cfg\.training\s*,\s*['\"]eval_enabled['\"]",
        re.MULTILINE,
    )
    assert pattern.search(text), (
        "src/train.py missing `train_config.eval_enabled = getattr(cfg.training, 'eval_enabled', ...)` "
        "— Hydra wire-through gap for the IsoFLOP eval gate"
    )


def test_train_py_wires_projector_hidden_mult():
    """`cfg.model.projector_hidden_mult` must reach `ModelConfig(...)`.
    Without this, the IsoFLOP variant ladder (W2X/W4X/D2X/D4X) collapses
    silently to BASE (hm=1, nl=2) since every cell runs with the
    dataclass default. Observed during Smoke 4 replay 2026-05-27 / 28.
    """
    text = _train_py_text()
    pattern = re.compile(
        r"projector_hidden_mult\s*=\s*cfg\.model\.get\(\s*['\"]projector_hidden_mult['\"]",
        re.MULTILINE,
    )
    assert pattern.search(text), (
        "src/train.py missing `projector_hidden_mult=cfg.model.get('projector_hidden_mult', ...)` "
        "in ModelConfig(...) — Hydra wire-through gap for the IsoFLOP capacity ladder"
    )


def test_train_py_wires_projector_num_layers():
    """Counterpart to projector_hidden_mult — depth knob (D2X/D4X)."""
    text = _train_py_text()
    pattern = re.compile(
        r"projector_num_layers\s*=\s*cfg\.model\.get\(\s*['\"]projector_num_layers['\"]",
        re.MULTILINE,
    )
    assert pattern.search(text), (
        "src/train.py missing `projector_num_layers=cfg.model.get('projector_num_layers', ...)` "
        "in ModelConfig(...) — Hydra wire-through gap for the IsoFLOP depth ladder"
    )


def test_train_py_sets_system_seed():
    """`cfg.system.seed` must be read AND fed into every RNG (random/np/torch/xpu).
    Stage A IsoFLOP's variance-floor measurement runs BASE@C with seeds {0,1,2,3}
    and needs the seed-replicas to actually diverge. Before pre-A the seed
    field existed in src/conf/config.yaml but no code read it, so the four
    replicas would have been bit-identical (modulo dataloader nondeterminism).
    """
    text = _train_py_text()
    # The fix sets seeds on random, numpy, and torch. Match each separately.
    assert re.search(r"\brandom\.seed\(\s*seed\s*\)", text), (
        "src/train.py missing `random.seed(seed)` — Python RNG not seeded"
    )
    assert re.search(r"\bnp\.random\.seed\(\s*seed\s*\)", text), (
        "src/train.py missing `np.random.seed(seed)` — NumPy RNG not seeded"
    )
    assert re.search(r"\btorch\.manual_seed\(\s*seed\s*\)", text), (
        "src/train.py missing `torch.manual_seed(seed)` — PyTorch RNG not seeded"
    )
    # XPU is the production device for PRISM; CUDA is supported but optional.
    assert re.search(r"torch\.xpu\.manual_seed_all\(\s*seed\s*\)", text), (
        "src/train.py missing `torch.xpu.manual_seed_all(seed)` — XPU RNG not seeded"
    )
    # The seed itself must come from cfg.system.seed (not a hardcoded literal).
    assert re.search(
        r"_seed_cfg\s*=\s*cfg\.get\(\s*['\"]system['\"]", text
    ) or re.search(
        r"cfg\.system\.get\(\s*['\"]seed['\"]", text
    ), (
        "src/train.py seed not read from cfg.system.seed — Hydra wire-through gap"
    )


def test_train_py_threads_seed_into_training_config():
    """The resolved seed must reach TrainingConfig so the trainer's
    startup_param_count perf.jsonl row carries `seed`. IsoFLOP collector
    needs it to group variance-floor replicas of BASE@C.
    """
    text = _train_py_text()
    assert re.search(r"seed\s*=\s*seed\s*,", text), (
        "src/train.py missing `seed=seed,` in TrainingConfig(...) — perf.jsonl "
        "startup_param_count event won't carry the resolved seed"
    )
