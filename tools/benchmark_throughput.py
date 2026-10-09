#!/usr/bin/env python3
"""Throughput Benchmark: 1B vs 7B Model Comparison

This script isolates throughput bottlenecks by testing:
1. Pure forward pass speed (no DDP, no data loading)
2. Forward + backward pass speed
3. DDP overhead (find_unused_parameters impact)
4. Batch size scaling (how throughput scales with batch size)
5. Gradient checkpointing impact
6. Full training loop comparison

Usage (on Aurora interactive node):
    python tools/benchmark_throughput.py --test all
    python tools/benchmark_throughput.py --test forward_only
    python tools/benchmark_throughput.py --test ddp_overhead
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
from dataclasses import dataclass

import torch

# Path safety
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    test_name: str
    model_name: str
    batch_size: int
    seq_len: int
    num_steps: int
    avg_time_per_step: float
    samples_per_sec: float
    memory_allocated_mb: float
    memory_reserved_mb: float
    extra: dict = None

    def __str__(self):
        extra_str = ""
        if self.extra:
            extra_str = " | " + " | ".join(f"{k}={v}" for k, v in self.extra.items())
        return (
            f"[{self.test_name}] {self.model_name} | "
            f"BS={self.batch_size} | SeqLen={self.seq_len} | "
            f"Time/Step={self.avg_time_per_step:.4f}s | "
            f"Throughput={self.samples_per_sec:.1f} samp/s | "
            f"Mem={self.memory_allocated_mb:.0f}MB alloc / {self.memory_reserved_mb:.0f}MB reserved"
            f"{extra_str}"
        )


def get_device():
    """Get the best available device."""
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu:0")
    elif torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def get_memory_stats(device):
    """Get memory stats for the current device."""
    if device.type == "xpu":
        return (
            torch.xpu.memory_allocated(device) / 1024**2,
            torch.xpu.memory_reserved(device) / 1024**2,
        )
    elif device.type == "cuda":
        return (
            torch.cuda.memory_allocated(device) / 1024**2,
            torch.cuda.memory_reserved(device) / 1024**2,
        )
    return 0.0, 0.0


def sync_device(device):
    """Synchronize device for accurate timing."""
    if device.type == "xpu":
        torch.xpu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def clear_cache(device):
    """Clear device memory cache."""
    gc.collect()
    if device.type == "xpu":
        torch.xpu.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def load_model(
    model_id: str, device: torch.device, gradient_checkpointing: bool = False
):
    """Load a HuggingFace model for benchmarking.

    Returns the model and its config.
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    print(f"\n  Loading model: {model_id}...", flush=True)
    t0 = time.time()

    config = AutoConfig.from_pretrained(model_id, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )

    # Freeze all parameters (simulating projector-only training)
    for param in model.parameters():
        param.requires_grad = False

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    model = model.to(device).to(torch.bfloat16)
    sync_device(device)

    num_params = sum(p.numel() for p in model.parameters())
    elapsed = time.time() - t0
    print(f"  Loaded {num_params / 1e9:.2f}B params in {elapsed:.1f}s")

    return model, config


def create_dummy_inputs(
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    vocab_size: int,
    device: torch.device,
):
    """Create dummy inputs_embeds and labels for benchmarking."""
    inputs_embeds = torch.randn(
        batch_size, seq_len, hidden_dim, dtype=torch.bfloat16, device=device
    )
    labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    return inputs_embeds, labels


def benchmark_forward_only(
    model,
    hidden_dim,
    vocab_size,
    device,
    batch_sizes=(1, 2, 4, 8, 16),
    seq_len=256,  # ~196 image patches + ~60 text tokens
    num_warmup=3,
    num_steps=10,
    model_name="",
) -> list[BenchmarkResult]:
    """Benchmark pure forward pass speed across batch sizes."""
    results = []
    model.eval()

    for bs in batch_sizes:
        clear_cache(device)

        try:
            inputs_embeds, labels = create_dummy_inputs(
                bs, seq_len, hidden_dim, vocab_size, device
            )
        except RuntimeError as e:
            print(f"  BS={bs}: OOM during input creation - {e}")
            break

        # Warmup
        try:
            for _ in range(num_warmup):
                with torch.no_grad():
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                        outputs = model(
                            inputs_embeds=inputs_embeds, labels=labels, return_dict=True
                        )
                sync_device(device)
        except RuntimeError as e:
            print(f"  BS={bs}: OOM during warmup - {e}")
            clear_cache(device)
            break

        # Timed runs
        times = []
        for _ in range(num_steps):
            sync_device(device)
            t0 = time.perf_counter()
            with torch.no_grad():
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    outputs = model(  # noqa: F841
                        inputs_embeds=inputs_embeds, labels=labels, return_dict=True
                    )
            sync_device(device)
            times.append(time.perf_counter() - t0)

        avg_time = sum(times) / len(times)
        mem_alloc, mem_reserved = get_memory_stats(device)

        result = BenchmarkResult(
            test_name="forward_only",
            model_name=model_name,
            batch_size=bs,
            seq_len=seq_len,
            num_steps=num_steps,
            avg_time_per_step=avg_time,
            samples_per_sec=bs / avg_time,
            memory_allocated_mb=mem_alloc,
            memory_reserved_mb=mem_reserved,
            extra={"min_time": f"{min(times):.4f}s", "max_time": f"{max(times):.4f}s"},
        )
        results.append(result)
        print(f"  {result}")

        del inputs_embeds, labels
        clear_cache(device)

    return results


def benchmark_forward_backward(
    model,
    hidden_dim,
    vocab_size,
    device,
    batch_sizes=(1, 2, 4, 8),
    seq_len=256,
    num_warmup=3,
    num_steps=10,
    model_name="",
    grad_checkpointing=False,
) -> list[BenchmarkResult]:
    """Benchmark forward + backward pass with a trainable projector layer.

    Simulates projector-only training: backbone is frozen, only a small
    projector layer has gradients.
    """
    results = []
    model.train()

    # Add a small trainable projector to simulate real training
    projector = torch.nn.Linear(hidden_dim, hidden_dim, bias=False).to(
        device=device, dtype=torch.bfloat16
    )
    projector_optimizer = torch.optim.AdamW(projector.parameters(), lr=1e-4)

    for bs in batch_sizes:
        clear_cache(device)

        try:
            inputs_embeds, labels = create_dummy_inputs(
                bs, seq_len, hidden_dim, vocab_size, device
            )
        except RuntimeError as e:
            print(f"  BS={bs}: OOM during input creation - {e}")
            break

        # Warmup
        try:
            for _ in range(num_warmup):
                projected = projector(inputs_embeds)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    outputs = model(
                        inputs_embeds=projected, labels=labels, return_dict=True
                    )
                    loss = outputs.loss
                loss.backward()
                projector_optimizer.step()
                projector_optimizer.zero_grad()
                sync_device(device)
        except RuntimeError as e:
            print(f"  BS={bs}: OOM during warmup - {e}")
            clear_cache(device)
            break

        # Timed runs
        fwd_times = []
        bwd_times = []
        total_times = []
        for _ in range(num_steps):
            projector_optimizer.zero_grad()
            sync_device(device)

            # Forward
            t0 = time.perf_counter()
            projected = projector(inputs_embeds)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                outputs = model(
                    inputs_embeds=projected, labels=labels, return_dict=True
                )
                loss = outputs.loss
            sync_device(device)
            t_fwd = time.perf_counter() - t0

            # Backward
            t1 = time.perf_counter()
            loss.backward()
            sync_device(device)
            t_bwd = time.perf_counter() - t1

            projector_optimizer.step()
            sync_device(device)
            t_total = time.perf_counter() - t0

            fwd_times.append(t_fwd)
            bwd_times.append(t_bwd)
            total_times.append(t_total)

        avg_total = sum(total_times) / len(total_times)
        avg_fwd = sum(fwd_times) / len(fwd_times)
        avg_bwd = sum(bwd_times) / len(bwd_times)
        mem_alloc, mem_reserved = get_memory_stats(device)

        gc_label = "grad_ckpt" if grad_checkpointing else "no_ckpt"
        result = BenchmarkResult(
            test_name=f"fwd_bwd_{gc_label}",
            model_name=model_name,
            batch_size=bs,
            seq_len=seq_len,
            num_steps=num_steps,
            avg_time_per_step=avg_total,
            samples_per_sec=bs / avg_total,
            memory_allocated_mb=mem_alloc,
            memory_reserved_mb=mem_reserved,
            extra={
                "fwd": f"{avg_fwd:.4f}s",
                "bwd": f"{avg_bwd:.4f}s",
                "bwd/fwd_ratio": f"{avg_bwd / avg_fwd:.2f}x",
            },
        )
        results.append(result)
        print(f"  {result}")

        del inputs_embeds, labels, projected
        clear_cache(device)

    del projector, projector_optimizer
    clear_cache(device)
    return results


def benchmark_ddp_overhead(
    model,
    hidden_dim,
    vocab_size,
    device,
    batch_size=2,
    seq_len=256,
    num_warmup=3,
    num_steps=10,
    model_name="",
) -> list[BenchmarkResult]:
    """Compare DDP with find_unused_parameters=True vs False.

    This is the PRIMARY suspect for 7B slowdown -- scanning 7.3B params
    every forward pass to find the ~8M trainable ones.

    Must be run with torchrun or mpiexec (world_size >= 2).
    """
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    if not dist.is_initialized():
        print("  [SKIP] DDP overhead test requires distributed init (world_size >= 2)")
        return []

    results = []

    # Create a wrapper model with a trainable projector
    class ProjectorWrapper(torch.nn.Module):
        def __init__(self, backbone, hidden_dim):
            super().__init__()
            self.backbone = backbone
            self.projector = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
            # Freeze backbone
            for p in self.backbone.parameters():
                p.requires_grad = False

        def forward(self, inputs_embeds, labels=None):
            x = self.projector(inputs_embeds)
            return self.backbone(inputs_embeds=x, labels=labels, return_dict=True)

    wrapper = ProjectorWrapper(model, hidden_dim).to(
        device=device, dtype=torch.bfloat16
    )

    for find_unused in [True, False]:
        clear_cache(device)

        ddp_device_id = (
            0
            if "ZE_AFFINITY_MASK" in os.environ
            else int(os.environ.get("LOCAL_RANK", 0))
        )

        try:
            ddp_model = DDP(
                wrapper,
                device_ids=[ddp_device_id],
                find_unused_parameters=find_unused,
                broadcast_buffers=False,
            )
        except Exception as e:
            print(f"  DDP wrap failed (find_unused={find_unused}): {e}")
            continue

        optimizer = torch.optim.AdamW(
            [p for p in ddp_model.parameters() if p.requires_grad], lr=1e-4
        )

        inputs_embeds, labels = create_dummy_inputs(
            batch_size, seq_len, hidden_dim, vocab_size, device
        )

        # Warmup
        for _ in range(num_warmup):
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                outputs = ddp_model(inputs_embeds=inputs_embeds, labels=labels)
                loss = outputs.loss
            loss.backward()
            optimizer.step()
            sync_device(device)

        # Timed
        times = []
        for _ in range(num_steps):
            optimizer.zero_grad()
            sync_device(device)
            t0 = time.perf_counter()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                outputs = ddp_model(inputs_embeds=inputs_embeds, labels=labels)
                loss = outputs.loss
            loss.backward()
            optimizer.step()
            sync_device(device)
            times.append(time.perf_counter() - t0)

        avg_time = sum(times) / len(times)
        mem_alloc, mem_reserved = get_memory_stats(device)

        result = BenchmarkResult(
            test_name=f"ddp_find_unused={find_unused}",
            model_name=model_name,
            batch_size=batch_size,
            seq_len=seq_len,
            num_steps=num_steps,
            avg_time_per_step=avg_time,
            samples_per_sec=batch_size / avg_time,
            memory_allocated_mb=mem_alloc,
            memory_reserved_mb=mem_reserved,
        )
        results.append(result)
        print(f"  {result}")

        del ddp_model, optimizer
        clear_cache(device)

    del wrapper
    clear_cache(device)
    return results


def benchmark_sync_overhead(
    model,
    hidden_dim,
    vocab_size,
    device,
    batch_size=4,
    seq_len=256,
    grad_accum_steps=(1, 4, 8, 16),
    num_steps=5,
    model_name="",
) -> list[BenchmarkResult]:
    """Measure the overhead of torch.xpu.synchronize() calls.

    Tests with/without sync after each phase to quantify pipeline stall cost.
    With grad_accum=16, the current code does 64+ syncs per optimizer step.
    """
    results = []
    model.train()

    projector = torch.nn.Linear(hidden_dim, hidden_dim, bias=False).to(
        device=device, dtype=torch.bfloat16
    )
    optimizer = torch.optim.AdamW(projector.parameters(), lr=1e-4)

    inputs_embeds, labels = create_dummy_inputs(
        batch_size, seq_len, hidden_dim, vocab_size, device
    )

    for accum in grad_accum_steps:
        for do_sync in [True, False]:
            clear_cache(device)

            # Warmup
            for _ in range(2):
                optimizer.zero_grad()
                for _ in range(accum):
                    projected = projector(inputs_embeds)
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                        outputs = model(
                            inputs_embeds=projected, labels=labels, return_dict=True
                        )
                        loss = outputs.loss / accum
                    loss.backward()
                optimizer.step()
                sync_device(device)

            # Timed
            times = []
            sync_device(device)
            for _ in range(num_steps):
                t0 = time.perf_counter()
                optimizer.zero_grad()
                for _ in range(accum):
                    projected = projector(inputs_embeds)
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                        outputs = model(
                            inputs_embeds=projected, labels=labels, return_dict=True
                        )
                        loss = outputs.loss / accum
                    loss.backward()
                    if do_sync:
                        sync_device(device)  # This is the overhead we're measuring
                optimizer.step()
                sync_device(device)  # Always sync at end to get accurate time
                times.append(time.perf_counter() - t0)

            avg_time = sum(times) / len(times)
            effective_samples = batch_size * accum
            mem_alloc, mem_reserved = get_memory_stats(device)

            sync_label = "with_sync" if do_sync else "no_mid_sync"
            result = BenchmarkResult(
                test_name=f"sync_overhead_accum{accum}_{sync_label}",
                model_name=model_name,
                batch_size=batch_size,
                seq_len=seq_len,
                num_steps=num_steps,
                avg_time_per_step=avg_time,
                samples_per_sec=effective_samples / avg_time,
                memory_allocated_mb=mem_alloc,
                memory_reserved_mb=mem_reserved,
                extra={
                    "grad_accum": accum,
                    "sync_per_step": accum * 4 if do_sync else 1,
                    "effective_batch": effective_samples,
                },
            )
            results.append(result)
            print(f"  {result}")

    del projector, optimizer, inputs_embeds, labels
    clear_cache(device)
    return results


def benchmark_batch_size_scaling(
    model,
    hidden_dim,
    vocab_size,
    device,
    batch_sizes=(1, 2, 4, 8, 16, 32),
    seq_len=256,
    num_warmup=3,
    num_steps=10,
    model_name="",
    with_backward=True,
) -> list[BenchmarkResult]:
    """Find optimal batch size for GPU utilization.

    Measures how throughput scales with batch size to find the sweet spot
    where GPU utilization saturates.
    """
    results = []

    projector = (
        torch.nn.Linear(hidden_dim, hidden_dim, bias=False).to(
            device=device, dtype=torch.bfloat16
        )
        if with_backward
        else None
    )

    if with_backward:
        model.train()
        optimizer = torch.optim.AdamW(projector.parameters(), lr=1e-4)
    else:
        model.eval()

    for bs in batch_sizes:
        clear_cache(device)

        try:
            inputs_embeds, labels = create_dummy_inputs(
                bs, seq_len, hidden_dim, vocab_size, device
            )
        except RuntimeError:
            print(f"  BS={bs}: OOM")
            break

        # Warmup
        try:
            for _ in range(num_warmup):
                if with_backward:
                    optimizer.zero_grad()
                    projected = projector(inputs_embeds)
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                        outputs = model(
                            inputs_embeds=projected, labels=labels, return_dict=True
                        )
                        loss = outputs.loss
                    loss.backward()
                    optimizer.step()
                else:
                    with torch.no_grad():
                        with torch.autocast(
                            device_type=device.type, dtype=torch.bfloat16
                        ):
                            model(
                                inputs_embeds=inputs_embeds,
                                labels=labels,
                                return_dict=True,
                            )
                sync_device(device)
        except RuntimeError:
            print(f"  BS={bs}: OOM during warmup")
            clear_cache(device)
            break

        # Timed
        times = []
        for _ in range(num_steps):
            sync_device(device)
            t0 = time.perf_counter()
            if with_backward:
                optimizer.zero_grad()
                projected = projector(inputs_embeds)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    outputs = model(
                        inputs_embeds=projected, labels=labels, return_dict=True
                    )
                    loss = outputs.loss
                loss.backward()
                optimizer.step()
            else:
                with torch.no_grad():
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                        model(
                            inputs_embeds=inputs_embeds, labels=labels, return_dict=True
                        )
            sync_device(device)
            times.append(time.perf_counter() - t0)

        avg_time = sum(times) / len(times)
        mem_alloc, mem_reserved = get_memory_stats(device)

        # Compute GPU utilization proxy: samples/sec normalized by BS
        samp_per_sec = bs / avg_time
        efficiency = samp_per_sec / bs  # Higher = better utilization per sample

        result = BenchmarkResult(
            test_name="batch_scaling" + ("_train" if with_backward else "_infer"),
            model_name=model_name,
            batch_size=bs,
            seq_len=seq_len,
            num_steps=num_steps,
            avg_time_per_step=avg_time,
            samples_per_sec=samp_per_sec,
            memory_allocated_mb=mem_alloc,
            memory_reserved_mb=mem_reserved,
            extra={
                "efficiency": f"{efficiency:.2f}",
                "mem_util_pct": f"{mem_alloc / 64000 * 100:.1f}%",
            },
        )
        results.append(result)
        print(f"  {result}")

        del inputs_embeds, labels
        clear_cache(device)

    if projector:
        del projector, optimizer
    clear_cache(device)
    return results


def run_comparison(args):
    """Run the full 1B vs 7B comparison."""
    device = get_device()
    print(f"\n{'=' * 80}")
    print("PRISM Throughput Benchmark")
    print(f"Device: {device}")
    print(f"Tests: {args.test}")
    print(f"{'=' * 80}")

    # Model paths
    models = {
        "OLMo-1B": {
            "id": "allenai/OLMo-1B-0724-hf",
            "hidden_dim": 2048,
            "vocab_size": 50304,  # OLMo-1B vocab
            "batch_sizes_fwd": [1, 2, 4, 8, 16, 32],
            "batch_sizes_train": [1, 2, 4, 8, 16],
            "grad_checkpointing": False,
        },
        "OLMo-3-7B": {
            "id": "allenai/Olmo-3-1025-7B",
            "hidden_dim": 4096,
            "vocab_size": 100278,  # OLMo-3 vocab
            "batch_sizes_fwd": [1, 2, 4, 8, 16],
            "batch_sizes_train": [1, 2, 4, 8],
            "grad_checkpointing": True,
        },
    }

    # Allow running only one model
    if args.model:
        if args.model == "1b":
            models = {k: v for k, v in models.items() if "1B" in k}
        elif args.model == "7b":
            models = {k: v for k, v in models.items() if "7B" in k}

    all_results = []

    for model_name, model_info in models.items():
        print(f"\n{'=' * 80}")
        print(f"Model: {model_name}")
        print(f"{'=' * 80}")

        model, config = load_model(
            model_info["id"],
            device,
            gradient_checkpointing=model_info["grad_checkpointing"],
        )
        hidden_dim = model_info["hidden_dim"]
        vocab_size = model_info["vocab_size"]

        tests = (
            args.test.split(",")
            if args.test != "all"
            else ["forward_only", "batch_scaling", "fwd_bwd", "sync_overhead"]
        )

        for test in tests:
            test = test.strip()
            print(f"\n--- Test: {test} ({model_name}) ---")

            if test == "forward_only":
                results = benchmark_forward_only(
                    model,
                    hidden_dim,
                    vocab_size,
                    device,
                    batch_sizes=model_info["batch_sizes_fwd"],
                    seq_len=args.seq_len,
                    num_steps=args.num_steps,
                    model_name=model_name,
                )
                all_results.extend(results)

            elif test == "batch_scaling":
                # Inference scaling
                results = benchmark_batch_size_scaling(
                    model,
                    hidden_dim,
                    vocab_size,
                    device,
                    batch_sizes=model_info["batch_sizes_fwd"],
                    seq_len=args.seq_len,
                    num_steps=args.num_steps,
                    model_name=model_name,
                    with_backward=False,
                )
                all_results.extend(results)

                # Training scaling
                results = benchmark_batch_size_scaling(
                    model,
                    hidden_dim,
                    vocab_size,
                    device,
                    batch_sizes=model_info["batch_sizes_train"],
                    seq_len=args.seq_len,
                    num_steps=args.num_steps,
                    model_name=model_name,
                    with_backward=True,
                )
                all_results.extend(results)

            elif test == "fwd_bwd":
                # Without gradient checkpointing
                results = benchmark_forward_backward(
                    model,
                    hidden_dim,
                    vocab_size,
                    device,
                    batch_sizes=model_info["batch_sizes_train"],
                    seq_len=args.seq_len,
                    num_steps=args.num_steps,
                    model_name=model_name,
                    grad_checkpointing=False,
                )
                all_results.extend(results)

                # With gradient checkpointing (for 7B comparison)
                if model_info["grad_checkpointing"]:
                    print("\n  (Re-testing with gradient checkpointing enabled)")
                    results = benchmark_forward_backward(
                        model,
                        hidden_dim,
                        vocab_size,
                        device,
                        batch_sizes=model_info["batch_sizes_train"],
                        seq_len=args.seq_len,
                        num_steps=args.num_steps,
                        model_name=model_name,
                        grad_checkpointing=True,
                    )
                    all_results.extend(results)

            elif test == "sync_overhead":
                results = benchmark_sync_overhead(
                    model,
                    hidden_dim,
                    vocab_size,
                    device,
                    batch_size=2,  # Match 7B config
                    seq_len=args.seq_len,
                    grad_accum_steps=[1, 4, 16],
                    num_steps=args.num_steps,
                    model_name=model_name,
                )
                all_results.extend(results)

            elif test == "ddp_overhead":
                results = benchmark_ddp_overhead(
                    model,
                    hidden_dim,
                    vocab_size,
                    device,
                    batch_size=2,
                    seq_len=args.seq_len,
                    num_steps=args.num_steps,
                    model_name=model_name,
                )
                all_results.extend(results)

        # Cleanup model before loading next one
        del model
        clear_cache(device)

    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")

    # Group results by test for comparison
    tests_seen = set(r.test_name for r in all_results)
    for test_name in sorted(tests_seen):
        print(f"\n--- {test_name} ---")
        test_results = [r for r in all_results if r.test_name == test_name]
        for r in test_results:
            print(f"  {r}")

        # Compare 1B vs 7B at matching batch sizes
        results_1b = [r for r in test_results if "1B" in r.model_name]
        results_7b = [r for r in test_results if "7B" in r.model_name]

        for r1b in results_1b:
            for r7b in results_7b:
                if r1b.batch_size == r7b.batch_size:
                    ratio = (
                        r1b.samples_per_sec / r7b.samples_per_sec
                        if r7b.samples_per_sec > 0
                        else float("inf")
                    )
                    print(
                        f"  >> BS={r1b.batch_size}: 1B is {ratio:.1f}x faster than 7B "
                        f"({r1b.samples_per_sec:.1f} vs {r7b.samples_per_sec:.1f} samp/s)"
                    )

    # Save results
    if args.output:
        results_data = [
            {
                "test": r.test_name,
                "model": r.model_name,
                "batch_size": r.batch_size,
                "seq_len": r.seq_len,
                "time_per_step": r.avg_time_per_step,
                "samples_per_sec": r.samples_per_sec,
                "mem_alloc_mb": r.memory_allocated_mb,
                "mem_reserved_mb": r.memory_reserved_mb,
                **(r.extra or {}),
            }
            for r in all_results
        ]
        with open(args.output, "w") as f:
            json.dump(results_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")


def main():
    parser = argparse.ArgumentParser(description="PRISM Throughput Benchmark")
    parser.add_argument(
        "--test",
        default="all",
        help="Comma-separated tests: forward_only,batch_scaling,fwd_bwd,sync_overhead,ddp_overhead (or 'all')",
    )
    parser.add_argument(
        "--model", choices=["1b", "7b"], default=None, help="Run only one model"
    )
    parser.add_argument(
        "--seq-len", type=int, default=256, help="Sequence length (default: 256)"
    )
    parser.add_argument(
        "--num-steps", type=int, default=10, help="Number of timed steps"
    )
    parser.add_argument(
        "--output", default="benchmark_results.json", help="Output JSON file"
    )
    args = parser.parse_args()

    run_comparison(args)


if __name__ == "__main__":
    main()
