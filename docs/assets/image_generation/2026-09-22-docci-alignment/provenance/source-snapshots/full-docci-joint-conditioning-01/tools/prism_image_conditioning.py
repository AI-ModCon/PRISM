"""Explicit text-conditioning ablations shared by image experiments.

These helpers do not change the default PRISM prompt contract. Chat formatting
and EOS-anchored empty conditioning are experimental alternatives; neither is
claimed to be a trained unconditional representation.
"""

from __future__ import annotations

from contextlib import contextmanager

IMAGE_SYSTEM_PROMPT = (
    "You are a helpful assistant that generates high-quality images based on user instructions."
)


def format_prompt(prompt, tokenizer, mode="raw"):
    if not isinstance(prompt, str):
        raise TypeError("Prompt must be a string")
    if mode == "raw":
        return prompt
    if mode != "chat":
        raise ValueError("Prompt format must be raw or chat")
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": IMAGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )


def format_items(items, tokenizer, mode="raw"):
    """Copy records while preserving targets and source ordering exactly."""
    return [dict(item, prompt=format_prompt(item["prompt"], tokenizer, mode)) for item in items]


def tokenize_prompt(tokenizer, prompt, *, device, mode="raw", max_text_length=1024):
    """Tokenize without truncation, using an explicit EOS anchor for empty raw input."""
    import torch

    formatted = format_prompt(prompt, tokenizer, mode)
    values = tokenizer([formatted], padding=True, truncation=False, return_tensors="pt")
    ids, mask = values["input_ids"], values["attention_mask"]
    if ids.ndim != 2 or mask.shape != ids.shape or ids.shape[0] != 1:
        raise ValueError("Expected one prompt and aligned two-dimensional mask")
    empty_anchor = None
    if ids.shape[1] == 0 or not mask.bool().any():
        if prompt != "" or mode != "raw":
            raise ValueError("Only empty raw prompts may use the explicit EOS anchor")
        token_id = getattr(tokenizer, "eos_token_id", None)
        if not isinstance(token_id, int) or token_id < 0:
            raise ValueError("Empty raw conditioning requires a valid tokenizer EOS ID")
        ids = torch.tensor([[token_id]], dtype=torch.long)
        mask = torch.ones_like(ids)
        empty_anchor = "eos"
    if ids.shape[1] > max_text_length:
        raise ValueError("Prompt exceeds max_text_length; diagnostic truncation is forbidden")
    if not torch.all((mask == 0) | (mask == 1)):
        raise ValueError("Prompt attention mask must be binary")
    return {
        "input_ids": ids.to(device),
        "input_attention_mask": mask.to(device),
        "formatted_prompt": formatted,
        "format": mode,
        "empty_anchor": empty_anchor,
    }


def encode_prism_prompt(model, tokenizer, prompt, *, device, mode="raw", max_text_length=1024):
    """Run a target-free text condition; caller owns grad and evaluation context."""
    values = tokenize_prompt(
        tokenizer, prompt, device=device, mode=mode, max_text_length=max_text_length
    )
    condition, _, _, _ = model._output_condition(
        {"text": values["input_ids"], "text_attention_mask": values["input_attention_mask"]}
    )
    embeds, mask = model.decoders["image"].connect(condition)
    return dict(values, hidden_states=condition.hidden_states, embeds=embeds, attention_mask=mask)


@contextmanager
def preserved_model_state(model):
    """Restore RNG, per-module modes and parameter grad flags, including on errors."""
    from tools.train_prism_image_connector import preserved_rng

    modes = {module: module.training for module in model.modules()}
    requires_grad = {parameter: parameter.requires_grad for parameter in model.parameters()}
    try:
        with preserved_rng():
            model.requires_grad_(False)
            model.eval()
            yield
    finally:
        for module, mode in modes.items():
            module.training = mode
        for parameter, enabled in requires_grad.items():
            parameter.requires_grad_(enabled)


def feature_statistics(embeds, mask):
    """Small finite summaries over valid tokens only, with no feature dumps."""
    import torch

    if embeds.ndim != 3 or mask.shape != embeds.shape[:2]:
        raise ValueError("Feature tensor and mask shapes must be BLD and BL")
    if not torch.all((mask == 0) | (mask == 1)):
        raise ValueError("Feature attention mask must be binary")
    valid = mask.bool()
    if not valid.any(dim=1).all():
        raise ValueError("Every condition must have at least one valid token")
    values = embeds[valid].detach().float()
    finite = torch.isfinite(values)
    result = {
        "shape": list(embeds.shape),
        "dtype": str(embeds.dtype),
        "valid_token_counts": valid.sum(1).cpu().tolist(),
        "valid_prefix": bool(not (valid[:, 1:] & ~valid[:, :-1]).any()),
        "finite": bool(finite.all()),
        "nonfinite_values": int((~finite).sum()),
        "padding_token_count": int((~valid).sum()),
    }
    if result["finite"]:
        norms = values.norm(dim=-1)
        result.update(
            mean=float(values.mean()),
            std=float(values.std(unbiased=False)),
            rms=float(values.square().mean().sqrt()),
            abs_max=float(values.abs().max()),
            token_norm_mean=float(norms.mean()),
            token_norm_min=float(norms.min()),
            token_norm_max=float(norms.max()),
            token_norm_quantiles={
                str(q): float(torch.quantile(norms, q)) for q in (0.0, 0.05, 0.5, 0.95, 1.0)
            },
            token_rms_quantiles={
                str(q): float(torch.quantile(values.square().mean(-1).sqrt(), q))
                for q in (0.0, 0.05, 0.5, 0.95, 1.0)
            },
        )
    return result


def summarize_caption_controls(rows):
    """Aggregate paired flow-loss controls; not an image quality metric."""
    import math
    from collections import defaultdict

    buckets = defaultdict(list)
    for row in rows:
        for route, losses in row["routes"].items():
            matched, wrong = losses["matched"], losses["wrong"]
            if not all(math.isfinite(value) for value in (matched, wrong)):
                raise ValueError("Cannot summarize nonfinite flow losses")
            buckets[(row["split"], route)].append((matched, wrong))
    result = {}
    for (split, route), losses in sorted(buckets.items()):
        count = len(losses)
        result.setdefault(split, {})[route] = {
            "paired_evaluations": count,
            "matched_loss": sum(pair[0] for pair in losses) / count,
            "wrong_loss": sum(pair[1] for pair in losses) / count,
            "wrong_minus_matched": sum(pair[1] - pair[0] for pair in losses) / count,
            "matched_wins": sum(pair[0] < pair[1] for pair in losses),
        }
    return result
