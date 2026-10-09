"""Keep `examples/portable_smoke/train_tiny.py` honest.

The example is the one thing in this repo a newcomer can run on a laptop: no
scheduler, no DAOS, no institutional path, no HuggingFace account, no network.
That claim rots silently unless something checks it, and `examples/` is barely
checked at all -- ruff lints it, but mypy reads `files = ["src"]`, pytest's
`testpaths = ["tests"]`, and coverage measures `--cov=src`. So the example's
CI hook has to live here.

It runs in-process rather than as a subprocess on purpose. A subprocess would
be a more faithful imitation of the user's command line, but coverage does not
follow subprocesses without a `COVERAGE_PROCESS_START` hook this repo does not
install, so the example's code would count for nothing against the coverage
gate. Running `run()` directly also lets these tests assert on the measurements
rather than scraping stdout.

No marker beyond `unit`: `*_contract` jobs select by file path, and the `unit`
and `coverage_gate` jobs select by marker, so anything from the deselect list
(`slow` is the tempting one for a training loop) would drop this file out of
CI entirely while still looking authored.
"""

from __future__ import annotations

import importlib.util
import os
import socket
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = REPO_ROOT / "examples" / "portable_smoke" / "train_tiny.py"


def _load_example():
    """Import the example by path -- `examples/` is not an importable package."""
    spec = importlib.util.spec_from_file_location("prism_portable_smoke", EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def example():
    return _load_example()


@pytest.fixture(scope="module")
def result(example):
    """Run the example once; every assertion below reads this one run."""
    return example.run(quiet=True)


def test_example_file_exists():
    # A broken path here would make every other test in this file vacuous.
    assert EXAMPLE.is_file(), f"the portable smoke example is missing: {EXAMPLE}"


def test_trains_offline_end_to_end(result):
    assert result["cache_files"] == 0, "the run touched the HuggingFace cache"
    assert result["trainable_parameters"] > 0
    assert result["final_loss"] < result["first_loss"]


def test_loss_drop_clears_the_threshold(example, result):
    # Measured >= 99.4% across five seeds, so the 30% floor has wide headroom.
    assert result["relative_drop"] >= example.MIN_RELATIVE_DROP


def test_negative_control_dead_model_fails(example):
    """With the learning rate zeroed the check must fail, or it proves nothing."""
    with pytest.raises(example.SmokeCheckFailed, match="did not learn"):
        example.run(break_training=True, quiet=True)


def test_negative_control_half_wired_model_fails(example):
    """The control that actually calibrates the threshold.

    A dead model failing is cheap -- 0% drop against a 30% bar proves only that
    the comparison runs. The regression worth catching is a model that trains
    but is not fully connected, which still looks alive from a distance. With
    the backbone frozen, gradients reach only PRISM's own layers and the loss
    falls 3.9%: far enough to be visibly "working", nowhere near the bar. That
    gap is what makes the threshold a discriminator rather than a liveness bit.
    """
    with pytest.raises(example.SmokeCheckFailed, match="did not learn"):
        example.run(break_wiring=True, quiet=True)


def test_model_is_small_enough_for_a_ci_runner(result):
    # A default-dimension UnifiedTransformer is ~3.7B parameters and has OOM-ed
    # this suite's runner before. Guard the order of magnitude, not the digits.
    assert result["total_parameters"] < 1_000_000


def test_model_footprint_is_small_in_bytes_too(result):
    """Bytes, because a parameter count cannot see the thing that OOM-ed us.

    Buffers are not parameters. On the backbone-less path each block registers
    a `max_seq_len ** 2` float32 causal mask -- 268 MB per layer at the 8192
    default -- and the parameter count is identical either way (measured: 91,664
    at both 512 and 8192). A wider dtype is likewise invisible to a count. So
    the count above and this bound fail on different regressions; keep both.
    """
    assert result["model_bytes"] < 16 * 1024 * 1024


def test_exercises_prism_layers_not_just_the_backbone(result):
    """The synthesised Qwen3 is 53,456 parameters; the rest is PRISM's own.

    Without this the example could regress to training a bare HF model and
    still look green, which would prove nothing about PRISM's plumbing.
    """
    assert result["total_parameters"] > 60_000


def test_cli_entry_point_reports_success(example):
    assert example.main([]) == 0


@pytest.mark.parametrize("flag", ["--break-training", "--break-wiring"])
def test_cli_entry_point_reports_each_negative_control(example, flag):
    assert example.main([flag]) == 1


def test_network_block_is_not_left_armed(example):
    """The block is process-global; leaving it on would break the rest of CI."""
    before = socket.getaddrinfo
    with example.network_blocked():
        with pytest.raises(example.NetworkBlocked):
            socket.getaddrinfo("huggingface.co", 443)
    assert socket.getaddrinfo is before


def test_the_network_probe_passes_when_armed(example):
    """Armed, the probe must be silent -- the example's happy path runs through it."""
    with example.network_blocked():
        example.assert_network_is_blocked()


def test_the_network_probe_catches_a_block_that_lets_traffic_through(example, monkeypatch):
    """The probe is what makes "offline" a measurement, so test the probe too.

    The regression it exists to catch is a block that looks armed but is not --
    a renamed attribute, a `socket` import resolving elsewhere. Then the run
    prints "network blocked", passes, and the offline claim is self-certifying
    because a warm cache or a live connection carried it.

    Simulated by making the entry points *succeed*, which is both the branch
    worth covering and hermetic. Deliberately not tested by calling the probe
    unpatched: that does issue a real DNS query (measured at 29 ms to NXDOMAIN
    here, against ~1 us armed), and a slow or captive resolver would turn a
    required job into a flake.
    """
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [])
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: None)
    with pytest.raises(example.SmokeCheckFailed, match="not armed"):
        example.assert_network_is_blocked()


def test_the_network_probe_rejects_an_unexpected_error(example, monkeypatch):
    """A probe that raised something else was never blocked -- it just broke.

    Distinct from the branch above: there the block let traffic through, here
    it failed in some other way. Both mean "not armed", and neither may pass.
    """

    def boom(*_args, **_kwargs):
        raise OSError("resolver is down")

    monkeypatch.setattr(socket, "getaddrinfo", boom)
    with pytest.raises(example.SmokeCheckFailed, match="not armed"):
        example.assert_network_is_blocked()


def test_hf_cache_env_is_restored(example, tmp_path, monkeypatch):
    """Every variable must come back exactly as it was, set or unset."""
    monkeypatch.setenv("HF_HOME", "/sentinel/hf-home")
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    with example.hf_cache_redirected(tmp_path):
        assert os.environ["HF_HOME"] == str(tmp_path)
        assert os.environ["HF_HUB_CACHE"] == str(tmp_path)
    assert os.environ["HF_HOME"] == "/sentinel/hf-home"
    assert "HF_HUB_CACHE" not in os.environ


def test_hf_cache_redirect_moves_the_frozen_module_globals(example, tmp_path):
    """The redirect has to move what the libraries actually read.

    `huggingface_hub.constants` resolves the cache environment variables once,
    at import time, and several modules rebind the result under their own
    names. The example imports transformers before it redirects anything, so
    an environment-only redirect leaves every one of those copies pointing at
    the ambient cache -- which is exactly the state this test pins against.
    """
    huggingface_hub = pytest.importorskip("huggingface_hub")

    frozen = example._frozen_cache_bindings()
    assert frozen, "no frozen cache bindings found -- the attribute names moved"

    with example.hf_cache_redirected(tmp_path):
        assert huggingface_hub.constants.HF_HUB_CACHE == str(tmp_path)
        for module, attr, _previous in frozen:
            assert getattr(module, attr) == str(tmp_path), f"{module.__name__}.{attr} not moved"

    for module, attr, previous in frozen:
        assert getattr(module, attr) == previous, f"{module.__name__}.{attr} not restored"


def test_a_hub_id_cannot_resolve_from_the_ambient_cache(example, tmp_path):
    """The regression the offline claim actually rests on.

    If someone replaces the synthesised local backbone with a hub id, this run
    must fail rather than quietly succeed off whatever the developer happens to
    have cached. The end-of-run file count cannot catch that on its own -- a
    cache *read* writes nothing -- so what has to hold is that the libraries are
    looking at this empty directory. Asserted by trying a load that only
    succeeds if they are not.

    Uses a repo id that is not fetched even on a miss: `local_files_only=True`
    plus the socket block means no outbound request either way.

    Honest about its own reach: on a runner whose ambient cache is cold this
    passes trivially, because the id would fail to resolve either way. It has
    real discriminating power exactly where the bug is dangerous -- a developer
    machine with a warm cache, which is where an offline claim gets certified by
    accident. It was written against that case and watched fail there.
    """
    transformers = pytest.importorskip("transformers")

    with example.hf_cache_redirected(tmp_path), example.network_blocked():
        with pytest.raises(Exception) as caught:
            transformers.AutoConfig.from_pretrained(
                "HuggingFaceTB/SmolLM2-360M-Instruct", local_files_only=True
            )

    # Any failure is acceptable except reaching the network: the point is that
    # the id did not silently resolve. A NetworkBlocked here would mean the
    # redirect worked but the load tried to go out anyway, which is also a pass
    # for this test's purpose -- but it must not have *succeeded*.
    assert caught.value is not None
    assert not list(tmp_path.rglob("*")), "the redirected cache should stay empty"
