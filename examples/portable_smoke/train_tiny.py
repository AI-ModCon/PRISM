#!/usr/bin/env python3
"""Train a tiny PRISM model on random noise, on a laptop, with the network off.

    python examples/portable_smoke/train_tiny.py

THIS TRAINS ON RANDOM NOISE AND PROVES PLUMBING, NOT LEARNING. The model is
~90K parameters and the "dataset" is four fixed nonsense sequences; the loss
falls because the model memorises them. What that demonstrates is that the
multimodal batch assembles, the time-series encoder and projector emit tokens,
those tokens interleave with text through the backbone, the loss is finite,
gradients reach every trainable parameter, and the optimiser moves them. It
says nothing whatsoever about model quality. See README.md in this directory.

Why this runs where the documented training commands do not: every shipped
preset points `backbone_id` at a real HuggingFace repo (a 7B OLMo by default)
and `src/model.py` loads it with `local_files_only=True`, so a fresh clone with
a cold cache cannot train at all. Here we *synthesise* a ~220 KB random-weight
Qwen3 plus a matching tokenizer into a temp directory and point
`llm_backbone_id` at it. To PRISM that is an ordinary local model path -- no
part of `src/` is stubbed, patched, or monkeypatched.

The network is then blocked at the socket layer -- and the block is *probed*
rather than trusted, since a block that silently failed to arm would let a warm
cache carry the run -- while the HuggingFace cache is redirected to an empty
temp directory that is asserted still empty at the end. So "it ran offline" is
verified rather than claimed. And because a check nobody has seen fail is not a
check, two flags sabotage the run on purpose -- `--break-training` and
`--break-wiring` -- and both must make it exit 1.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import socket
import sys
import tempfile
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import cast

# Run from anywhere: resolve the repo root off this file, not off cwd.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Tiny on every axis. `max_seq_len=512` is defensive rather than load-bearing
# here: with a backbone present PRISM delegates attention to it and never builds
# its own blocks (measured -- the only buffer in this model is the backbone's
# 4-element `rotary_emb.inv_freq`), so the 8192 default costs nothing on this
# path. It would matter on the backbone-less path, where each block registers a
# `max_seq_len ** 2` float32 causal mask as a non-persistent buffer -- 268 MB
# per layer, and invisible to any parameter count. Kept small so that copying
# these dims to that path does not quietly OOM.
VOCAB_SIZE = 128
TINY_DIMS: dict[str, int] = dict(
    vocab_size=VOCAB_SIZE,
    d_model=64,
    d_text=64,
    num_layers=2,
    num_heads=2,
    num_experts=2,
    max_seq_len=512,
)
BATCH = 4
SEQ_LEN = 16
TS_LEN = 32
STEPS = 40
LEARNING_RATE = 1e-2
SEED = 0

# The loss must fall by at least this fraction of its starting value. A relative
# drop rather than an absolute floor keeps the check meaningful on a host with
# different BLAS or threading.
#
# The threshold is calibrated against a HALF-WIRED model, not a dead one, since
# that is the regression worth catching. Measured on this config:
#
#   everything trains (the real run)   0.9993   passes
#   learning rate zeroed               0.0000   fails   (--break-training)
#   backbone frozen, PRISM layers only 0.0391   fails   (--break-wiring)
#
# So a model where gradients reach only part of the stack still falls short by
# most of the margin: the check discriminates plausibly-broken from working, not
# merely dead from alive.
MIN_RELATIVE_DROP = 0.30


class NetworkBlocked(RuntimeError):
    """Raised instead of opening any socket once the block is armed."""


class SmokeCheckFailed(RuntimeError):
    """The run completed but did not prove what it is supposed to prove."""


@contextlib.contextmanager
def network_blocked() -> Iterator[None]:
    """Make every outbound connection raise, while keeping ssl and torch usable.

    Patch the connect/resolve entry points rather than replacing `socket.socket`
    itself: `ssl` -- and therefore `torch` -- needs the class to stay
    constructible at import time. Enter this only after torch is imported.

    A context manager rather than a one-way switch because CI runs this module
    in-process alongside the rest of the suite; a block left armed would break
    every test that follows.
    """

    def refuse(*_args: object, **_kwargs: object) -> None:
        raise NetworkBlocked(
            "This example must run with no network access; something tried to "
            "open a connection. If this fires inside huggingface_hub, a model "
            "or tokenizer id is being resolved against the hub instead of the "
            "local directory this script creates."
        )

    saved = (
        socket.socket.connect,
        socket.socket.connect_ex,
        socket.create_connection,
        socket.getaddrinfo,
    )
    # A deny-everything stub has a different signature from all four by
    # design, so each replacement needs an ignore. The restores do not: they
    # put back the values mypy already knows the right types for.
    socket.socket.connect = refuse  # type: ignore[method-assign]
    socket.socket.connect_ex = refuse  # type: ignore[method-assign,assignment]
    socket.create_connection = refuse  # type: ignore[assignment]
    socket.getaddrinfo = refuse  # type: ignore[assignment]
    try:
        yield
    finally:
        socket.socket.connect = saved[0]  # type: ignore[method-assign]
        socket.socket.connect_ex = saved[1]  # type: ignore[method-assign]
        socket.create_connection = saved[2]
        socket.getaddrinfo = saved[3]


_CACHE_NAMES = ("HF_HUB_CACHE", "HF_HOME", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE")


def _frozen_cache_bindings() -> list[tuple[object, str, str]]:
    """Find every already-imported module holding its own copy of a cache path.

    Setting the environment variables is not enough on its own.
    `huggingface_hub.constants` resolves them once, at import time, into module
    globals, and several modules then bind those values again under their own
    names. By the time this example redirects anything, `build_local_backbone`
    has already imported transformers, so all of those copies point at the
    developer's ambient cache and no later `os.environ` write can move them.
    Measured on this host: eight bindings across five modules
    (`huggingface_hub.constants`, `.file_download`, `.utils._cache_manager`,
    `transformers`, `transformers.utils`, `transformers.utils.hub`).

    Discovered by walking `sys.modules` rather than hard-coded, because the set
    differs by library version and a name that moved would silently reduce this
    to the environment-only behaviour it exists to replace.
    """
    found: list[tuple[object, str, str]] = []
    for name, module in list(sys.modules.items()):
        if not (name == "transformers" or name.startswith(("transformers.", "huggingface_hub"))):
            continue
        if module is None:
            continue
        for attr in _CACHE_NAMES:
            value = getattr(module, attr, None)
            if isinstance(value, str) and value:
                found.append((module, attr, value))
    return found


@contextlib.contextmanager
def hf_cache_redirected(cache_dir: Path) -> Iterator[None]:
    """Point every HuggingFace cache path at `cache_dir`, then restore them all.

    Both the environment and the already-resolved module globals, because the
    two layers read different things. PRISM's own `resolve_hf_hub_cache()` reads
    the environment on each call, so that half matters; `huggingface_hub` and
    `transformers` read the globals they froze at import, so without the second
    half a hub id still resolves against the ambient cache. That is the failure
    this whole block exists to prevent, and it is verified in
    `test_a_hub_id_cannot_resolve_from_the_ambient_cache`.
    """
    saved_env = {name: os.environ.get(name) for name in _CACHE_NAMES}
    for name in _CACHE_NAMES:
        os.environ[name] = str(cache_dir)
    frozen = _frozen_cache_bindings()
    for module, attr, _previous in frozen:
        setattr(module, attr, str(cache_dir))
    try:
        yield
    finally:
        for module, attr, previous in frozen:
            setattr(module, attr, previous)
        for name, previous_value in saved_env.items():
            if previous_value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous_value


def build_local_backbone(directory: Path) -> Path:
    """Write a random-weight Qwen3 and a matching tokenizer into `directory`.

    `save_pretrained` on the model alone is not enough: it writes config and
    weights but no tokenizer, and PRISM resolves a tokenizer for the backbone
    id, so loading would fail on a directory that has only weights.
    """
    import torch
    import transformers

    # The shipped `tokenizers` stubs re-export these through a star-import
    # mypy does not follow, so all three read as missing attributes.
    from tokenizers import (  # type: ignore[attr-defined]
        Tokenizer,
        models,
        pre_tokenizers,
    )

    torch.manual_seed(937)
    config = transformers.Qwen3Config(
        vocab_size=VOCAB_SIZE,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=8,
        max_position_embeddings=256,
        attention_dropout=0.0,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    config._attn_implementation = "eager"
    transformers.Qwen3ForCausalLM(config).save_pretrained(directory)

    vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
    vocab.update({f"tok{i}": i for i in range(len(vocab), VOCAB_SIZE)})
    backing = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    backing.pre_tokenizer = pre_tokenizers.Whitespace()
    transformers.PreTrainedTokenizerFast(
        tokenizer_object=backing,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
    ).save_pretrained(directory)
    return directory


def count_cache_files(cache_dir: Path) -> int:
    """Count files written under the redirected cache.

    This catches a *write* -- something fetched or materialised a file. It
    cannot catch a *read* of a pre-existing cache, since reading writes
    nothing; what makes reads impossible is `hf_cache_redirected` pointing the
    library at this empty directory in the first place, and the socket block
    stopping the download that would follow the miss.
    """
    return sum(len(files) for _root, _dirs, files in os.walk(cache_dir))


def assert_network_is_blocked() -> None:
    """Confirm the block actually armed, instead of printing that it did.

    Everything else here is measured, so this line should be too. If
    `network_blocked()` ever regressed to a no-op -- a renamed attribute, a
    `socket` import that resolves elsewhere -- the run would still print
    "network blocked", still pass, and the offline claim would be worth
    nothing, because a warm cache or a live network would quietly carry it.

    Costs ~1 us when armed, because the patched entry point raises before any
    lookup happens (measured; a real resolution of the same name takes ~29 ms
    on this host). The name is under RFC 6761's `.invalid`, which can never
    resolve to a real host, and resolution is probed before connection -- so
    even a fully unarmed run gets an `NXDOMAIN` and stops here, rather than
    opening a socket to anything.
    """
    for probe in (
        lambda: socket.getaddrinfo("prism-portable-smoke.invalid", 443),
        lambda: socket.create_connection(("prism-portable-smoke.invalid", 443), timeout=0.01),
    ):
        try:
            probe()
        except NetworkBlocked:
            continue
        except Exception as exc:  # broad on purpose: anything else means unarmed
            raise SmokeCheckFailed(
                "the network block is not armed: a probe raised "
                f"{type(exc).__name__} instead of NetworkBlocked, so this run "
                "could have reached the network without anyone noticing"
            ) from exc
        raise SmokeCheckFailed(
            "the network block is not armed: a probe to a .invalid host "
            "succeeded, so this run could have reached the network"
        )


def run(
    break_training: bool = False,
    break_wiring: bool = False,
    quiet: bool = False,
) -> dict[str, object]:
    """Build the model, train it on noise, and return the measurements.

    Returns a dict rather than only printing, so the CI test can assert on the
    same numbers a human reads off the terminal. Every global it touches -- the
    socket module, the HuggingFace cache variables, its scratch directory -- is
    restored or removed on the way out.

    The two `break_*` flags are negative controls: each sabotages the run in a
    different way and the loss-drop check must then fail. Without them a green
    run would only mean the code did not crash.

    `break_training` zeroes the learning rate -- the blunt control, a model that
    cannot move at all. `break_wiring` is the subtle and more valuable one: it
    leaves `freeze_backbone` at its library default so gradients reach only
    PRISM's own layers. That model trains, and its loss does fall, just not far
    (0.039 measured, against a 0.30 threshold and 0.999 when healthy) -- which
    is what makes the threshold a real discriminator rather than a liveness bit.
    """

    def say(message: str = "") -> None:
        if not quiet:
            print(message)

    with tempfile.TemporaryDirectory(prefix="prism-portable-smoke-") as tmp:
        scratch = Path(tmp)
        cache_dir = scratch / "hf-cache"
        cache_dir.mkdir()
        backbone_dir = build_local_backbone(scratch / "backbone")
        on_disk = sum(p.stat().st_size for p in backbone_dir.iterdir())
        say(f"synthesised a local backbone in {backbone_dir} ({on_disk / 1024:.0f} KB)")

        with hf_cache_redirected(cache_dir), network_blocked():
            assert_network_is_blocked()
            say("network blocked at the socket layer (verified by probe)")
            result = _train(backbone_dir, break_training, break_wiring, say)

        leaked = count_cache_files(cache_dir)
        say(f"HuggingFace cache files written: {leaked}")
        if leaked:
            raise SmokeCheckFailed(
                f"{leaked} files landed in the HuggingFace cache; this run was not offline"
            )

    # `result` is deliberately heterogeneous -- counts, losses, a dtype
    # string -- so its values type as `object` and this one needs narrowing.
    drop = cast(float, result["relative_drop"])
    if drop < MIN_RELATIVE_DROP:
        raise SmokeCheckFailed(
            f"loss fell only {100 * drop:.1f}%, below the "
            f"{100 * MIN_RELATIVE_DROP:.0f}% this check requires -- the model ran "
            "but did not learn, so something is connected but not training"
        )

    say()
    say("OK -- PRISM trains end to end on this machine, offline.")
    say("Reminder: this trained on random noise. It proves plumbing, not quality.")
    result["backbone_bytes"] = on_disk
    result["cache_files"] = leaked
    return result


def _train(
    backbone_dir: Path,
    break_training: bool,
    break_wiring: bool,
    say: Callable[..., None],
) -> dict[str, object]:
    import torch
    from src.config import ModelConfig
    from src.model import UnifiedTransformer

    torch.manual_seed(SEED)
    config = ModelConfig(
        # Strings, not Modality members: ModelConfig.__post_init__ runs them
        # through `parse_modality`, and every shipped preset writes them this
        # way too (src/config.py:319 and the preset table below it).
        modalities=["text", "time_series"],  # type: ignore[list-item]
        llm_backbone_id=str(backbone_dir),
        ts_projector="linear",
        output_decoders=["text"],
        # Both default to True, which leaves a model with zero trainable
        # parameters and makes the optimiser raise on an empty parameter list.
        # `break_wiring` restores the backbone default on purpose: see run().
        freeze_backbone=break_wiring,
        freeze_encoders=False,
        # A `dict[str, int]` unpacked into a signature that also takes
        # strs, bools and tuples: mypy checks the unpack against every
        # parameter, not just the seven this dict actually fills. The dict
        # itself stays typed, so the dims are still checked.
        **TINY_DIMS,  # type: ignore[arg-type]
    )
    model = UnifiedTransformer(config)

    backbone = model.backbone
    if backbone is None:  # pragma: no cover - llm_backbone_id is always set here
        raise SmokeCheckFailed(
            "no backbone was built, so this would exercise the backbone-less "
            "path instead of the one the example is documenting"
        )

    # The backbone arrives in float16 on any device that is not CUDA-with-bf16
    # or XPU -- including plain CPU, where float16 training produces a
    # non-finite gradient on the very first backward pass. Inference in float16
    # is fine (drift vs float32 measured at 4e-4); training is not.
    loaded_dtype = next(backbone.parameters()).dtype
    if loaded_dtype != torch.float32:
        say(f"backbone loaded as {loaded_dtype}; casting to float32 to train on CPU")
        model = model.float()

    trainable = [p for p in model.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in trainable)
    if not trainable_count:
        raise SmokeCheckFailed(
            "the model has zero trainable parameters -- check freeze_backbone "
            "and freeze_encoders, which both default to True"
        )
    total = sum(p.numel() for p in model.parameters())
    # Bytes, not just a count: buffers do not appear in a parameter count and a
    # causal mask or a wider dtype can cost orders of magnitude more than the
    # weights. This is the number that decides whether a CI runner survives.
    resident = sum(p.numel() * p.element_size() for p in model.parameters()) + sum(
        b.numel() * b.element_size() for b in model.buffers()
    )
    say(
        f"model built: {total:,} parameters, {trainable_count:,} trainable "
        f"({resident / 1024:.0f} KB of weights and buffers)"
    )

    generator = torch.Generator().manual_seed(1234)
    tokens = torch.randint(4, VOCAB_SIZE, (BATCH, SEQ_LEN), generator=generator)
    series = torch.randn(BATCH, TS_LEN, 1, generator=generator)
    batch = {"text": tokens, "time_series": series}

    learning_rate = 0.0 if break_training else LEARNING_RATE
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate)

    say(f"training {STEPS} steps on random noise (lr={learning_rate})")
    losses: list[float] = []
    for step in range(STEPS):
        _logits, loss = model(batch, labels=tokens)
        if not torch.isfinite(loss):
            raise SmokeCheckFailed(f"loss became non-finite at step {step}: {loss}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().item())
        if step % 10 == 0 or step == STEPS - 1:
            say(f"  step {step:>3}  loss {losses[-1]:.4f}")

    starved = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    if starved:
        raise SmokeCheckFailed(
            f"{len(starved)} trainable parameters never received a gradient, "
            f"starting with {starved[0]}"
        )

    relative_drop = 1.0 - losses[-1] / losses[0]
    say()
    say(f"loss {losses[0]:.4f} -> {losses[-1]:.4f}  ({100 * relative_drop:.1f}% drop)")
    return {
        "total_parameters": total,
        "trainable_parameters": trainable_count,
        "model_bytes": resident,
        "first_loss": losses[0],
        "final_loss": losses[-1],
        "relative_drop": relative_drop,
        "loaded_dtype": str(loaded_dtype),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").splitlines()[0],
        epilog="The two --break-* flags are negative controls: each should make "
        "this exit 1. Use them to confirm the check can actually fail.",
    )
    parser.add_argument(
        "--break-training",
        action="store_true",
        help="zero the learning rate, so nothing can move at all (expect exit 1)",
    )
    parser.add_argument(
        "--break-wiring",
        action="store_true",
        help="freeze the backbone, so gradients reach only part of the stack; "
        "the loss still falls, just nowhere near far enough (expect exit 1)",
    )
    args = parser.parse_args(argv)
    try:
        run(break_training=args.break_training, break_wiring=args.break_wiring)
    except (SmokeCheckFailed, NetworkBlocked) as exc:
        print(f"\nFAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
