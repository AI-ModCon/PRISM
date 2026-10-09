"""Test torch.compile on Intel XPU with Triton 3.4.0 + PyTorch 2.8.0.

Run on a compute node (not login node):
    ZE_AFFINITY_MASK=0 python tests/test_xpu_compile.py
"""
import os
import time

import pytest
import torch

pytestmark = [pytest.mark.aurora, pytest.mark.gpu, pytest.mark.integration]

# NOTE: Diagnostic test — returns True/False instead of asserting, so a "fail"
# result will still report PASS under pytest. Read the stdout when running
# manually on a compute node.

device = "xpu:0"
dtype = torch.bfloat16

def test_simple_compile():
    """Test 1: Simple linear model compile on XPU."""
    print("=" * 60)
    print("TEST 1: Simple model compile on XPU")
    print("=" * 60)

    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = torch.nn.Linear(1024, 1024)
            self.norm = torch.nn.LayerNorm(1024)
            self.linear2 = torch.nn.Linear(1024, 1024)

        def forward(self, x):
            return self.linear2(self.norm(torch.nn.functional.silu(self.linear1(x))))

    model = SimpleModel().to(device, dtype=dtype)
    x = torch.randn(4, 128, 1024, device=device, dtype=dtype)

    # Eager baseline
    torch.xpu.synchronize()
    t0 = time.time()
    for _ in range(10):
        _ = model(x)
    torch.xpu.synchronize()
    eager_time = (time.time() - t0) / 10
    print(f"  Eager:    {eager_time*1000:.2f} ms/step")

    # Compile
    t0 = time.time()
    compiled = torch.compile(model, backend="inductor", dynamic=True)
    compile_setup_time = time.time() - t0
    print(f"  Compile setup: {compile_setup_time:.2f}s")

    # First forward (triggers actual compilation)
    torch.xpu.synchronize()
    t0 = time.time()
    _ = compiled(x)
    torch.xpu.synchronize()
    first_fwd_time = time.time() - t0
    print(f"  First compiled forward: {first_fwd_time:.2f}s")

    # Subsequent forwards
    torch.xpu.synchronize()
    t0 = time.time()
    for _ in range(10):
        _ = compiled(x)
    torch.xpu.synchronize()
    compiled_time = (time.time() - t0) / 10
    speedup = eager_time / compiled_time if compiled_time > 0 else float('inf')
    print(f"  Compiled: {compiled_time*1000:.2f} ms/step ({speedup:.2f}x speedup)")
    print(f"  RESULT: {'PASS' if first_fwd_time < 300 else 'TOO SLOW'}")
    return first_fwd_time < 300


def test_dynamic_shapes_xpu():
    """Test 2: Dynamic shapes on XPU (the previous failure mode)."""
    print("\n" + "=" * 60)
    print("TEST 2: Dynamic shapes on XPU")
    print("=" * 60)

    class DynModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(512, 512)
            self.norm = torch.nn.LayerNorm(512)

        def forward(self, x):
            return self.norm(self.linear(x))

    model = DynModel().to(device, dtype=dtype)
    compiled = torch.compile(model, backend="inductor", dynamic=True)

    seq_lengths = [32, 64, 48, 128, 24, 96]
    for seq_len in seq_lengths:
        x = torch.randn(2, seq_len, 512, device=device, dtype=dtype)
        torch.xpu.synchronize()
        t0 = time.time()
        out = compiled(x)
        torch.xpu.synchronize()
        elapsed = time.time() - t0
        print(f"  seq_len={seq_len:4d}: output={out.shape}, time={elapsed:.3f}s")

    print("  RESULT: PASS (no recompilation crash)")
    return True


def test_olmo_backbone_compile():
    """Test 3: Compile OLMo backbone (the real target)."""
    print("\n" + "=" * 60)
    print("TEST 3: OLMo backbone compile on XPU")
    print("=" * 60)

    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(
        "allenai/OLMo-1B-0724-hf", trust_remote_code=True
    )
    # Use 2 layers for faster testing
    config.num_hidden_layers = 2

    print(f"  Loading tiny OLMo ({config.num_hidden_layers} layers, hidden={config.hidden_size})...")
    model = AutoModelForCausalLM.from_config(config).to(device, dtype=dtype).train()
    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {params:,}")

    # Eager baseline
    input_ids = torch.randint(0, 1000, (2, 64), device=device)
    labels = input_ids.clone()

    torch.xpu.synchronize()
    t0 = time.time()
    for _ in range(5):
        out = model(input_ids=input_ids, labels=labels)
        out.loss.backward()
    torch.xpu.synchronize()
    eager_time = (time.time() - t0) / 5
    print(f"  Eager fwd+bwd: {eager_time*1000:.1f} ms/step")

    # Compile backbone only
    print("  Compiling backbone...")
    model.model = torch.compile(model.model, backend="inductor", dynamic=True)

    # First compiled forward (compilation time)
    torch.xpu.synchronize()
    t0 = time.time()
    out = model(input_ids=input_ids, labels=labels)
    out.loss.backward()
    torch.xpu.synchronize()
    first_time = time.time() - t0
    print(f"  First compiled fwd+bwd: {first_time:.1f}s (includes compilation)")

    # Subsequent steps
    times = []
    for seq_len in [64, 96, 48, 128]:
        input_ids = torch.randint(0, 1000, (2, seq_len), device=device)
        labels = input_ids.clone()
        model.zero_grad()
        torch.xpu.synchronize()
        t0 = time.time()
        out = model(input_ids=input_ids, labels=labels)
        out.loss.backward()
        torch.xpu.synchronize()
        elapsed = time.time() - t0
        times.append(elapsed)
        print(f"  seq_len={seq_len}: loss={out.loss.item():.4f}, time={elapsed*1000:.1f}ms")

    avg_compiled = sum(times) / len(times)
    speedup = eager_time / avg_compiled if avg_compiled > 0 else float('inf')
    print(f"  Avg compiled: {avg_compiled*1000:.1f} ms/step ({speedup:.2f}x vs eager)")
    print(f"  Compilation time: {first_time:.1f}s")
    print(f"  RESULT: {'PASS' if first_time < 300 else 'COMPILATION TOO SLOW'}")
    return first_time < 300


def test_olmo_full_layers():
    """Test 4: Full OLMo-1B backbone compile (all layers)."""
    print("\n" + "=" * 60)
    print("TEST 4: Full OLMo-1B backbone compile (all layers)")
    print("=" * 60)

    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(
        "allenai/OLMo-1B-0724-hf", trust_remote_code=True
    )
    # Full model
    print(f"  Loading OLMo-1B ({config.num_hidden_layers} layers, hidden={config.hidden_size})...")
    model = AutoModelForCausalLM.from_config(config).to(device, dtype=dtype).train()
    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {params:,}")

    input_ids = torch.randint(0, 1000, (2, 64), device=device)
    labels = input_ids.clone()

    # Eager baseline
    torch.xpu.synchronize()
    t0 = time.time()
    for _ in range(3):
        out = model(input_ids=input_ids, labels=labels)
        out.loss.backward()
        model.zero_grad()
    torch.xpu.synchronize()
    eager_time = (time.time() - t0) / 3
    print(f"  Eager fwd+bwd: {eager_time*1000:.1f} ms/step")

    # Compile backbone only
    print("  Compiling full backbone (this may take a while)...")
    model.model = torch.compile(model.model, backend="inductor", dynamic=True)

    torch.xpu.synchronize()
    t0 = time.time()
    out = model(input_ids=input_ids, labels=labels)
    out.loss.backward()
    torch.xpu.synchronize()
    first_time = time.time() - t0
    print(f"  First compiled fwd+bwd: {first_time:.1f}s (includes compilation)")

    # Subsequent steps
    model.zero_grad()
    times = []
    for step in range(5):
        input_ids = torch.randint(0, 1000, (2, 64), device=device)
        labels = input_ids.clone()
        model.zero_grad()
        torch.xpu.synchronize()
        t0 = time.time()
        out = model(input_ids=input_ids, labels=labels)
        out.loss.backward()
        torch.xpu.synchronize()
        elapsed = time.time() - t0
        times.append(elapsed)
        print(f"  Step {step}: loss={out.loss.item():.4f}, time={elapsed*1000:.1f}ms")

    avg_compiled = sum(times) / len(times)
    speedup = eager_time / avg_compiled if avg_compiled > 0 else float('inf')
    print(f"  Avg compiled: {avg_compiled*1000:.1f} ms/step ({speedup:.2f}x vs eager)")
    print(f"  Compilation time: {first_time:.1f}s")

    mem = torch.xpu.memory_allocated() / 1e9
    mem_reserved = torch.xpu.memory_reserved() / 1e9
    print(f"  Memory: {mem:.1f}/{mem_reserved:.1f} GB (allocated/reserved)")
    print(f"  RESULT: {'PASS' if first_time < 600 else 'COMPILATION TOO SLOW'}")
    return first_time < 600


if __name__ == "__main__":
    # Proxy for HuggingFace access on compute nodes
    os.environ.setdefault("HTTP_PROXY", "http://proxy.alcf.anl.gov:3128")
    os.environ.setdefault("HTTPS_PROXY", "http://proxy.alcf.anl.gov:3128")
    os.environ.setdefault("http_proxy", "http://proxy.alcf.anl.gov:3128")
    os.environ.setdefault("https_proxy", "http://proxy.alcf.anl.gov:3128")

    print(f"PyTorch {torch.__version__}")
    print(f"Device: {torch.xpu.get_device_name(0)}")
    print("Triton: ", end="")
    import triton

    print(triton.__version__)
    print()

    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/tmp/torchinductor_cache")

    results = {}
    results["simple_compile"] = test_simple_compile()
    results["dynamic_shapes"] = test_dynamic_shapes_xpu()
    results["olmo_2layer"] = test_olmo_backbone_compile()

    # Only run full model if 2-layer test passed quickly
    if results["olmo_2layer"]:
        results["olmo_full"] = test_olmo_full_layers()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
