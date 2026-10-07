import json
import logging
import os

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _meta_path(adapter_path: str) -> str:
    return adapter_path + ".meta.json"


def _load_adapter_meta(adapter_path: str) -> dict | None:
    """Read the {rank, alpha, target_modules} sidecar next to adapter_path,
    if one was written by save_lora_adapter. Returns None for an adapter
    saved before this sidecar existed -- callers must treat that as "no
    verification available", not an error.
    """
    meta_path = _meta_path(adapter_path)
    if not os.path.exists(meta_path):
        return None
    with open(meta_path) as f:
        return json.load(f)


def apply_lora_torchtune(
    backbone: nn.Module,
    rank: int = 32,
    alpha: float = 64.0,
    dropout: float = 0.05,
    exclude: tuple[str, ...] = ("lm_head",),
) -> nn.Module:
    """Replace every nn.Linear in backbone (except exclusions) with torchtune LoRALinear.

    Base weights are copied from the original layer, then frozen via set_trainable_params.
    Only lora_a.weight and lora_b.weight remain trainable.

    Args:
        backbone: The HuggingFace backbone nn.Module to modify in-place.
        rank: LoRA rank (r).
        alpha: LoRA alpha scaling factor.
        dropout: Dropout applied to the LoRA path.
        exclude: Leaf module names to skip (default: lm_head).

    Returns:
        The modified backbone with LoRA layers.
    """
    from torchtune.modules.peft import LoRALinear, get_adapter_params, set_trainable_params

    def _get_parent(model: nn.Module, dotted_name: str):
        parts = dotted_name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        return parent, parts[-1]

    # Collect replacements first to avoid mutating the module tree mid-iteration.
    to_replace = []
    for name, module in backbone.named_modules():
        if isinstance(module, nn.Linear):
            leaf = name.split(".")[-1]
            if leaf not in exclude:
                to_replace.append((name, module))

    replaced = 0
    for name, module in to_replace:
        parent, leaf = _get_parent(backbone, name)
        lora_layer = LoRALinear(
            in_dim=module.in_features,
            out_dim=module.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            use_bias=module.bias is not None,
        )
        # Copy pre-trained weights into the LoRA layer.
        lora_layer.weight.data.copy_(module.weight.data)
        if module.bias is not None:
            lora_layer.bias.data.copy_(module.bias.data)
        setattr(parent, leaf, lora_layer)
        replaced += 1

    # Freeze all params, then unfreeze only the LoRA adapter params.
    adapter_params = get_adapter_params(backbone)
    set_trainable_params(backbone, adapter_params)

    n_trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in backbone.parameters())
    logger.info(
        f"torchtune LoRA: replaced {replaced} linear layers | "
        f"trainable {n_trainable:,} / {n_total:,} params "
        f"({100.0 * n_trainable / n_total:.3f}%)"
    )

    return backbone


def save_lora_adapter(
    backbone: nn.Module,
    path: str,
    *,
    rank: int | None = None,
    alpha: float | None = None,
    target_modules: list[str] | None = None,
) -> None:
    """Save only the LoRA adapter weights to a .pt file.

    When rank/alpha/target_modules are supplied, also writes a
    {rank, alpha, target_modules} sidecar JSON next to path
    (<path>.meta.json). load_lora_adapter and merge_lora_into_base use it
    to verify a caller-supplied rank/alpha actually matches what this
    adapter was saved with -- a mismatched resume_lora_alpha otherwise
    silently mis-scales every delta with no error, since alpha itself
    never appears in the saved tensors.
    """
    from torchtune.modules.peft import get_adapter_params

    adapter_state = dict(get_adapter_params(backbone))
    torch.save(adapter_state, path)
    n = sum(v.numel() for v in adapter_state.values())
    logger.info(f"LoRA adapter saved to {path} ({len(adapter_state)} tensors, {n:,} params)")

    if rank is not None or alpha is not None or target_modules is not None:
        meta = {"rank": rank, "alpha": alpha, "target_modules": target_modules}
        meta_path = _meta_path(path)
        with open(meta_path, "w") as f:
            json.dump(meta, f)
        logger.info(f"LoRA adapter metadata saved to {meta_path}: {meta}")


def load_lora_adapter(backbone: nn.Module, path: str, *, expected_rank: int | None = None) -> None:
    """Load LoRA adapter weights back into a backbone that already has LoRALinear layers.

    If expected_rank is given and a {rank, ...} sidecar exists next to path
    (written by save_lora_adapter), raises on a mismatch rather than
    silently loading a wrong-shaped adapter.
    """
    meta = _load_adapter_meta(path)
    if expected_rank is not None and meta is not None and meta.get("rank") is not None:
        if meta["rank"] != expected_rank:
            raise ValueError(
                f"LoRA adapter at {path} was saved with rank={meta['rank']}, but "
                f"expected_rank={expected_rank} was requested. Refusing to load a "
                f"mismatched adapter."
            )

    state = torch.load(path, map_location="cpu", weights_only=True)
    missing, unexpected = backbone.load_state_dict(state, strict=False)
    unexpected_lora = [k for k in unexpected if "lora_" in k]
    if unexpected_lora:
        logger.warning(f"Unexpected LoRA keys during load: {unexpected_lora}")

    # Assert at least one lora_ key was actually consumed -- strict=False
    # relaxes key matching but not shapes, so a load matching zero adapter
    # keys (e.g. the backbone has no LoRALinear layers yet, or is a
    # completely different architecture) would otherwise succeed silently
    # as a no-op.
    matched_keys = [k for k in state if k not in unexpected]
    loaded_lora_keys = [k for k in matched_keys if "lora_" in k]
    if not loaded_lora_keys:
        raise ValueError(
            f"LoRA adapter at {path} matched zero lora_ keys into the backbone "
            f"(0 of {len(state)} saved keys were consumed). Refusing a no-op load."
        )

    logger.info(f"LoRA adapter loaded from {path} ({len(loaded_lora_keys)} lora_ tensors matched)")


def merge_lora_into_base(backbone: nn.Module, path: str, alpha: float) -> None:
    """Fold a saved LoRA adapter (plain lora_a/lora_b tensors) into the backbone's
    base nn.Linear weights, in place.

    Used when a new adapter of a different rank needs to be stacked on top of a
    previously trained one (e.g. SFT rank 32 -> GRPO rank 16): the old adapter
    can't be loaded into differently-shaped LoRALinear modules, so instead it is
    merged into the frozen base weights before the new LoRALinear layers are
    created by apply_lora_torchtune.

    Args:
        backbone: The HuggingFace backbone with plain nn.Linear layers (must be
            called before apply_lora_torchtune has replaced them with LoRALinear).
        path: Path to a lora_adapter.pt saved by save_lora_adapter.
        alpha: The lora_alpha used to train the saved adapter. The merge scale is
            alpha / rank, where rank is inferred from the saved tensor shapes.
            If a {alpha, ...} sidecar exists next to path (written by
            save_lora_adapter) and disagrees with this value, raises rather
            than silently merging at the wrong scale -- alpha never appears
            in the saved tensors themselves, so a wrong value here produces
            no shape error, just a subtly corrupted merge.
    """
    meta = _load_adapter_meta(path)
    if meta is not None and meta.get("alpha") is not None and float(meta["alpha"]) != float(alpha):
        raise ValueError(
            f"LoRA adapter at {path} was saved with alpha={meta['alpha']}, but the "
            f"caller supplied alpha={alpha} to merge with. Refusing to merge at a "
            f"mismatched scale -- this would silently corrupt the merged base weights."
        )

    state = torch.load(path, map_location="cpu", weights_only=True)

    # Group lora_a/lora_b pairs by their owning linear layer's dotted name.
    layer_names = {k[: -len(".lora_a.weight")] for k in state if k.endswith(".lora_a.weight")}
    if not layer_names:
        raise ValueError(f"No lora_a.weight tensors found in {path}; nothing to merge")

    linears = dict(backbone.named_modules())
    merged = 0
    for layer_name in layer_names:
        a = state[f"{layer_name}.lora_a.weight"]
        b = state[f"{layer_name}.lora_b.weight"]
        rank = a.shape[0]
        scale = alpha / rank

        module = linears.get(layer_name)
        if module is None or not isinstance(module, nn.Linear):
            logger.warning(f"Skipping merge for {layer_name}: not found or not nn.Linear")
            continue

        with torch.no_grad():
            module.weight.data += scale * (b.to(module.weight.dtype) @ a.to(module.weight.dtype))
        merged += 1

    logger.info(f"Merged {merged} LoRA adapter layers from {path} into base weights (alpha={alpha})")
