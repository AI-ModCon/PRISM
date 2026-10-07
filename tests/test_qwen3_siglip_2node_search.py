from tools.qwen3_siglip_2node_search import (
    DEFAULT_DESIGN,
    STRATEGY_FLAGS,
    build_command,
    build_phase,
    steps_for_run,
)


def test_ddp_batch_phase_starts_with_smallest_qwen_siglip_base():
    runs = build_phase("ddp-batch", DEFAULT_DESIGN)

    assert runs[0].design == "PRISM-QWEN3-0P6B-SIGLIP2-BASE-2N"
    assert runs[0].strategy == "ddp"
    assert runs[0].batch_size == 4
    assert runs[0].max_seq_length == 512
    assert len(runs) == 9


def test_parallelism_phase_includes_requested_strategies():
    strategies = [run.strategy for run in build_phase("parallelism", DEFAULT_DESIGN)]

    assert strategies == [
        "ddp",
        "zero1",
        "zero2",
        "fsdp_shard_grad",
        "fsdp_full",
        "hsdp_shard_grad",
    ]
    # ZeRO-1/2 don't do AllGather, so --no-pil4dfs (DAOS-17499 workaround)
    # isn't needed. ZeRO-3's AllGather case is handled by the launcher.
    assert STRATEGY_FLAGS["zero1"] == ("--deepspeed", "1")
    assert STRATEGY_FLAGS["zero2"] == ("--deepspeed", "2")


def test_build_command_uses_two_nodes_and_batch_override():
    run = build_phase("parallelism", DEFAULT_DESIGN)[1]
    cmd = build_command(
        run,
        experiment_file="experiments/qwen3_siglip_2node_search.yaml",
        nodes=2,
        dataset_groups="pixmo",
        max_steps=80,
        target_effective_tokens=0,
        tokens_per_sample=260,
        sweep_id="qwen3_siglip_2n_test",
        queue="debug",
        project="AuroraGPT",
        walltime="01:00:00",
        hf_fallback_dirs="/flare/ModCon/sww/huggingface,/flare/ModCon/sww/huggingface/hub",
        mode="dry-run",
    )

    assert "--nodes" in cmd
    assert cmd[cmd.index("--nodes") + 1] == "2"
    assert "--deepspeed" in cmd
    assert cmd[cmd.index("--deepspeed") + 1] == "1"
    assert "--hf-fallback-dirs" in cmd
    assert (
        cmd[cmd.index("--hf-fallback-dirs") + 1]
        == "/flare/ModCon/sww/huggingface,/flare/ModCon/sww/huggingface/hub"
    )
    assert "training.batch_size=8" in cmd
    assert "--dry-run" in cmd
    assert "exp.sweep_id=qwen3_siglip_2n_test" in cmd


def test_target_effective_tokens_sets_per_run_steps():
    run = build_phase("ddp-batch", DEFAULT_DESIGN)[1]  # bs8, seq512

    assert (
        steps_for_run(
            run,
            nodes=2,
            max_steps=80,
            target_effective_tokens=10_000_000,
            tokens_per_sample=260,
        )
        == 201
    )


def test_scale_up_phase_covers_three_designs_and_two_strategies():
    runs = build_phase("scale-up", DEFAULT_DESIGN)

    assert len(runs) == 6
    designs = [run.design for run in runs]
    assert designs == [
        "PRISM-QWEN3-0P6B-SIGLIP2-SO400M-2N",
        "PRISM-QWEN3-0P6B-SIGLIP2-SO400M-2N",
        "PRISM-QWEN3-1P7B-SIGLIP2-BASE-2N",
        "PRISM-QWEN3-1P7B-SIGLIP2-BASE-2N",
        "PRISM-QWEN3-4B-SIGLIP2-BASE-2N",
        "PRISM-QWEN3-4B-SIGLIP2-BASE-2N",
    ]
    strategies = [run.strategy for run in runs]
    assert strategies == [
        "ddp",
        "hsdp_shard_grad",
        "ddp",
        "hsdp_shard_grad",
        "ddp",
        "hsdp_shard_grad",
    ]
    # 4B model gets bs=4 (memory-bound), others get bs=8.
    assert [run.batch_size for run in runs] == [8, 8, 8, 8, 4, 4]


def test_target_effective_tokens_scales_with_so400m_patch_count():
    # SO400M emits 729 patch tokens vs base's 196; per-sample budget is
    # ~3x larger, so the same token budget should need ~1/3 the steps at
    # the same batch size.
    base_run = build_phase("scale-up", DEFAULT_DESIGN)[2]  # 1.7B base, ddp, bs=8
    so400m_run = build_phase("scale-up", DEFAULT_DESIGN)[0]  # SO400M, ddp, bs=8

    base_steps = steps_for_run(
        base_run,
        nodes=2,
        max_steps=80,
        target_effective_tokens=10_000_000,
        tokens_per_sample=260,
    )
    so400m_steps = steps_for_run(
        so400m_run,
        nodes=2,
        max_steps=80,
        target_effective_tokens=10_000_000,
        tokens_per_sample=260,
    )

    # 196+64=260 tokens/sample for base; 729+64=793 for SO400M.
    # base: ceil(10M / (24*8*260))   = ceil(10M / 49920)   = 201
    # so400m: ceil(10M / (24*8*793)) = ceil(10M / 152256) = 66
    assert base_steps == 201
    assert so400m_steps == 66


def test_build_command_uses_target_effective_tokens_when_set():
    run = build_phase("ddp-batch", DEFAULT_DESIGN)[1]  # bs8, seq512
    cmd = build_command(
        run,
        experiment_file="experiments/qwen3_siglip_2node_search.yaml",
        nodes=2,
        dataset_groups="pixmo",
        max_steps=80,
        target_effective_tokens=10_000_000,
        tokens_per_sample=260,
        sweep_id="qwen3_siglip_2n_10m",
        queue="debug",
        project="AuroraGPT",
        walltime="01:00:00",
        hf_fallback_dirs="/flare/ModCon/sww/huggingface",
        mode="print",
    )

    assert cmd[cmd.index("--max-steps") + 1] == "201"
