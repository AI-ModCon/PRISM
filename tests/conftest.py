import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def offline_hf(monkeypatch):
    """Neutralise the HuggingFace downloads `UnifiedTransformer.__init__` reaches.

    Covers the encoder and tokenizer sites. It deliberately does **not** touch
    `AutoModelForCausalLM.from_pretrained`: most tests here supply their own
    backbone through that symbol, and stubbing it would override them. A test
    that needs a backbone and patches nothing should request `offline_backbone`
    as well.

    Opt-in rather than autouse — a global stub would hide a regression in the
    loader itself, and a test that wants to exercise real loading should be able
    to.

    Patch targets are the *importing* modules, not `transformers`: every call
    site does `from transformers import AutoModel` at module scope, so the name
    to rebind lives in the consumer. `src/model.py:20` likewise does
    `from .hf_cache import load_cached_tokenizer`, so patching
    `src.hf_cache.load_cached_tokenizer` would rebind a name nothing reads. See
    `tests/offline_hf.py`.
    """
    # 1. Backbone tokenizer. `load_cached_tokenizer` reaches `snapshot_download`
    #    before `AutoTokenizer.from_pretrained`, so the usual patch never fires.
    #    The replacement *delegates* to `AutoTokenizer.from_pretrained`, skipping
    #    only the download — so a test that patches that symbol (several do, to
    #    supply a tokenizer of a specific length) still gets its own object, and
    #    only a test that patches nothing falls through to the stub.
    import src.model
    from tests.offline_hf import AutoStub, StubTokenizer

    def _load_cached_tokenizer(tokenizer_id, **kwargs):
        from transformers import AutoTokenizer

        try:
            return AutoTokenizer.from_pretrained(tokenizer_id, **kwargs)
        except Exception:
            return StubTokenizer()

    monkeypatch.setattr(src.model, "load_cached_tokenizer", _load_cached_tokenizer)

    # 2. Image encoder (SigLIP vision tower) and 3. external text encoder.
    import src.encoders.image
    import src.encoders.text

    monkeypatch.setattr(src.encoders.image, "AutoModel", AutoStub())
    monkeypatch.setattr(src.encoders.text, "AutoModel", AutoStub())
    monkeypatch.setattr(src.encoders.text, "AutoTokenizer", AutoStub(tokenizer=True))

    return None


@pytest.fixture
def offline_backbone(monkeypatch):
    """Supply a tiny real causal-LM backbone instead of downloading one.

    For tests that build a `UnifiedTransformer` against a real `llm_backbone_id`
    and then run a forward pass — a MagicMock will not do, the model has to
    compute. Follows the pattern already used in
    `tests/test_time_series_forecast_config.py`: construct a minimal
    `Qwen3ForCausalLM` in memory, no weights fetched.

    Compose with `offline_hf`, which covers the encoder-side downloads.
    """
    import transformers

    if not hasattr(transformers, "Qwen3ForCausalLM"):
        pytest.skip("Qwen3 requires a recent Transformers release")

    from tests.offline_hf import build_tiny_causal_lm

    backbone = build_tiny_causal_lm(transformers)
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM, "from_pretrained", lambda *a, **kw: backbone
    )
    return backbone
