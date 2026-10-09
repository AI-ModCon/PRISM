"""Login-node tests for VLLM-6: tools/vllm_eval_time_series.py.

No engine boot — covers the request-building contract and CLI surface.
The actual throughput benchmark runs on a compute node; this test just
checks that the eval driver's deterministic helpers behave correctly so
a regression doesn't go silently to a PBS job.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

vllm = pytest.importorskip("vllm")


def _load_eval_module():
    """Load tools/vllm_eval_time_series.py by path (tools/ isn't a package)."""
    project_root = Path(__file__).resolve().parent.parent
    eval_path = project_root / "tools" / "vllm_eval_time_series.py"
    spec = importlib.util.spec_from_file_location(
        "_vllm_eval_ts_inline", str(eval_path)
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_build_requests_n_and_shape():
    mod = _load_eval_module()
    requests, sample_ts = mod._build_requests(
        n=4, max_ts_length=32, num_vars=2, prompt="<time_series>p", seed=42
    )
    assert len(requests) == 4
    for req in requests:
        assert req["prompt"] == "<time_series>p"
        ts = req["multi_modal_data"]["time_series"]
        assert tuple(ts.shape) == (32, 2)
        assert ts.dtype == torch.float32
    assert tuple(sample_ts.shape) == (32, 2)


def test_build_requests_deterministic():
    """Same seed -> identical tensors across calls."""
    mod = _load_eval_module()
    r1, ts1 = mod._build_requests(
        n=1, max_ts_length=16, num_vars=1, prompt="<time_series>", seed=7
    )
    r2, ts2 = mod._build_requests(
        n=1, max_ts_length=16, num_vars=1, prompt="<time_series>", seed=7
    )
    torch.testing.assert_close(ts1, ts2)
    torch.testing.assert_close(
        r1[0]["multi_modal_data"]["time_series"],
        r2[0]["multi_modal_data"]["time_series"],
    )


def test_build_requests_independent_tensor_per_item():
    """N requests share the same DATA but each gets its own tensor object
    (so mutations don't leak across the engine's batch)."""
    mod = _load_eval_module()
    requests, _ = mod._build_requests(
        n=3, max_ts_length=8, num_vars=1, prompt="<time_series>", seed=0
    )
    ts0 = requests[0]["multi_modal_data"]["time_series"]
    ts1 = requests[1]["multi_modal_data"]["time_series"]
    assert ts0 is not ts1
    torch.testing.assert_close(ts0, ts1)  # equal values from .clone()
    ts0.fill_(0.0)
    assert not torch.equal(ts0, ts1)  # mutation doesn't leak
