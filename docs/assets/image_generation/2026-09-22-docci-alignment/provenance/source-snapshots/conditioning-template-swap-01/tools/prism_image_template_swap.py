"""Caption-only teacher-assisted template swaps; never a deployable PRISM route.

Suffix teacher states can contain causal information about the whole caption.
An improvement therefore implicates conditioning outside the trained content span,
but does not by itself prove that inert template formatting caused the failure.
"""

from __future__ import annotations

import hashlib

ROUTE_SOURCES = {
    "native_pretrained": {"prefix": "native", "content": "native", "suffix": "native"},
    "prism_aligned": {"prefix": "aligned", "content": "aligned", "suffix": "aligned"},
    "prism_native_prefix": {"prefix": "native", "content": "aligned", "suffix": "aligned"},
    "prism_native_suffix": {"prefix": "aligned", "content": "aligned", "suffix": "native"},
    "prism_native_template": {"prefix": "native", "content": "aligned", "suffix": "native"},
}


def protocol():
    return {
        "routes": list(ROUTE_SOURCES),
        "text_guidance_scale": 1.0,
        "swap_stage": "unnormalized_before_dit_rmsnorm",
        "teacher_assisted": True,
        "teacher_cache_persisted": False,
        "target_pixels_used_for_conditioning": False,
        "additional_reference": "native_pretrained CFG5 before alignment restore",
        "interpretation": (
            "Native suffix states may encode the full caption. Hybrid improvements localize "
            "conditioning sensitivity but do not establish an inert-formatting defect or "
            "standalone PRISM generation quality."
        ),
    }


def compose_conditions(prism, native, raw_ids, norm):
    """Clone pre-normalization states, proving common positions and unchanged content."""
    import torch
    from tools.align_prism_image_conditioning import validate_pair
    from tools.prism_image_conditioning import feature_statistics
    from tools.train_image_decoder import _tensor_hash
    from torch.nn import functional as F

    content, _, span = validate_pair(prism, native, raw_ids)
    mask = native["attention_mask"]
    valid = mask.bool()
    if (valid[:, 1:] & ~valid[:, :-1]).any():
        raise ValueError("Template swaps require right-padded valid prefixes")
    native_states = native["embeds"]
    aligned_states = prism["embeds"]
    if (
        aligned_states.shape != native_states.shape
        or not aligned_states.is_floating_point()
        or not native_states.is_floating_point()
        or not torch.isfinite(aligned_states).all()
        or not torch.isfinite(native_states).all()
    ):
        raise ValueError("Connected and native features must have identical finite BLD shapes")
    if any(parameter.requires_grad for parameter in norm.parameters()):
        raise ValueError("Template-swap caption normalization must remain frozen")
    native_states = native_states.detach().clone()
    aligned_states = aligned_states.detach().to(native_states).clone()
    if not torch.isfinite(aligned_states).all():
        raise ValueError("Connected features overflowed the native conditioning dtype")
    positions = torch.arange(mask.shape[1], device=mask.device).unsqueeze(0)
    partitions = {
        "prefix": valid & (positions < span[0]),
        "content": content.to(mask.device),
        "suffix": valid & (positions >= span[1]),
    }
    if any(not bool(part.any()) for part in partitions.values()):
        raise ValueError("Template swaps require nonempty prefix, content and suffix partitions")
    if not torch.equal(sum(part.int() for part in partitions.values()), valid.int()):
        raise ValueError("Template partitions must cover valid tokens exactly once")
    states = {"native": native_states, "aligned": aligned_states}
    teacher = norm(native_states).float()
    if teacher.shape != native_states.shape or not torch.isfinite(teacher).all():
        raise ValueError("Actual caption normalization returned invalid native features")
    routes, statistics = {}, {}
    for route, sources in ROUTE_SOURCES.items():
        value = native_states.clone() if route == "native_pretrained" else aligned_states.clone()
        for region, source in sources.items():
            part = partitions[region].to(value.device)
            value[part] = states[source][part]
        if route != "native_pretrained" and not torch.equal(
            value[partitions["content"]], aligned_states[partitions["content"]]
        ):
            raise RuntimeError("Template swap changed aligned caption-content states")
        normalized = norm(value).float()
        if normalized.shape != value.shape or not torch.isfinite(normalized).all():
            raise ValueError("Actual caption normalization returned invalid hybrid features")
        base = native if route == "native_pretrained" else prism
        routes[route] = {
            key: item for key, item in base.items() if key not in ("embeds", "hidden_states")
        } | {"embeds": value, "attention_mask": mask.clone()}
        region_stats = {}
        for region, part in partitions.items():
            part = part.to(value.device)
            selected = value[part].unsqueeze(0)
            region_stats[region] = {
                "unnormalized_sha256": _tensor_hash(selected),
                "unnormalized_features": feature_statistics(
                    selected, torch.ones(selected.shape[:2], device=selected.device)
                ),
                "normalized_mse_to_native": float(
                    (normalized[part] - teacher[part]).square().mean()
                ),
                "normalized_cosine_to_native": float(
                    F.cosine_similarity(normalized[part], teacher[part], dim=-1, eps=1e-8).mean()
                ),
            }
        statistics[route] = {
            "embeds_sha256": _tensor_hash(value),
            "attention_mask_sha256": _tensor_hash(mask),
            "region_sources": dict(sources),
            "partitions": region_stats,
        }
    count = int(valid.sum())
    audit = {
        "content_span": span,
        "partition_spans": {"prefix": [0, span[0]], "content": span, "suffix": [span[1], count]},
        "partition_token_counts": {key: int(part.sum()) for key, part in partitions.items()},
        "total_tokens": mask.shape[1],
        "valid_tokens": count,
        "input_ids_sha256": _tensor_hash(native["input_ids"]),
        "input_mask_sha256": _tensor_hash(native["input_attention_mask"]),
        "formatted_prompt_sha256": hashlib.sha256(native["formatted_prompt"].encode()).hexdigest(),
        "conditioning_dtype": str(native_states.dtype),
        "exact_formatted_input_match": True,
        "exact_input_ids_match": True,
        "exact_input_masks_match": True,
        "swap_stage": "unnormalized_before_dit_rmsnorm",
        "target_pixels_read": False,
        "routes": statistics,
    }
    return routes, audit


def encode_template_swap_conditions(model, tokenizer, backend, prompt, *, device, max_text_length):
    """Audit the real native teacher's IDs/masks, then assemble five target-free routes."""
    import torch
    from tools.align_prism_image_conditioning import caption_norm, encode_observed_native
    from tools.prism_image_conditioning import encode_prism_prompt

    if any(module.training for module in model.modules()) or any(
        parameter.requires_grad for parameter in model.parameters()
    ):
        raise ValueError("Template controls require every model component frozen and eval")
    with torch.no_grad():
        prism = encode_prism_prompt(
            model, tokenizer, prompt, device=device, mode="chat", max_text_length=max_text_length
        )
        native = encode_observed_native(backend, prompt, max_text_length=max_text_length)
        if native["embeds"].dtype != next(backend.transformer.parameters()).dtype:
            raise ValueError(
                "Native teacher and DiT conditioning dtypes must agree for exact swaps"
            )
        raw = tokenizer(
            [prompt], padding=False, truncation=False, add_special_tokens=False, return_tensors="pt"
        )
        if not raw["attention_mask"].bool().all():
            raise ValueError("Raw caption must not contain padding")
        routes, audit = compose_conditions(prism, native, raw["input_ids"], caption_norm(backend))
    audit["actual_native_forward_inputs_verified"] = True
    audit["prompt_sha256"] = hashlib.sha256(prompt.encode()).hexdigest()
    return routes, audit


def verify_condition_trace(trace, value):
    """Prove sampling used the intended pre-RMSNorm positive states and one CFG1 branch."""
    import torch
    from tools.train_image_decoder import _tensor_hash

    expected, mask = value["embeds"].detach().cpu(), value["attention_mask"].detach().cpu()
    for features_key, mask_key in (
        ("condition.positive", "mask.positive"),
        ("condition.branch0", "mask.branch0"),
    ):
        actual, actual_mask = trace.get(features_key), trace.get(mask_key)
        if not isinstance(actual, torch.Tensor) or not isinstance(actual_mask, torch.Tensor):
            raise RuntimeError("Sampling did not capture the actual positive condition and mask")
        if _tensor_hash(actual) != _tensor_hash(expected) or _tensor_hash(
            actual_mask
        ) != _tensor_hash(mask):
            raise RuntimeError("Sampling changed the intended template-swap condition or mask")
    if any(key.startswith("condition.branch") and key != "condition.branch0" for key in trace):
        raise RuntimeError("CFG1 template sampling unexpectedly used another conditioning branch")
    if trace.get("condition.negative") is not None or trace.get("mask.negative") is not None:
        raise RuntimeError("CFG1 template sampling unexpectedly encoded negative conditioning")
    return {"embeds_sha256": _tensor_hash(expected), "attention_mask_sha256": _tensor_hash(mask)}


def sample_template_swaps(
    backend, prompt, positive, *, seed, args, output_dir, case_id, saved_latent
):
    """Generate five CFG1 routes with audited conditions and replayed initial latents."""
    import torch
    from src.decoders.loading import file_sha256
    from tools.train_image_decoder import _seed_everything, _tensor_hash
    from tools.train_prism_image_connector import preserved_rng

    if list(positive) != list(ROUTE_SOURCES) or not isinstance(saved_latent, torch.Tensor):
        raise ValueError(
            "Template sampling requires exactly five ordered routes and baseline noise"
        )
    records = []
    with preserved_rng(), torch.no_grad():
        for route, value in positive.items():
            _seed_everything(seed)
            images = backend.generate_conditioned(
                value["embeds"],
                value["attention_mask"],
                height=args.height,
                width=args.width,
                num_inference_steps=args.sampling_steps,
                text_guidance_scale=1.0,
                image_guidance_scale=1.0,
                max_sequence_length=args.max_text_length,
                generator=torch.Generator(device=args.device).manual_seed(seed),
                latents=saved_latent.to(
                    device=args.device, dtype=getattr(torch, args.dtype)
                ).clone(),
                trace=True,
            )
            actual = backend.last_trace.get("latents.initial")
            if not isinstance(actual, torch.Tensor) or _tensor_hash(actual) != _tensor_hash(
                saved_latent
            ):
                raise RuntimeError("Template sampling changed initial noise")
            condition_audit = verify_condition_trace(backend.last_trace, value)
            images = images.images if hasattr(images, "images") else images
            if not isinstance(images, (list, tuple)) or len(images) != 1:
                raise RuntimeError("Template sampling requires exactly one returned image")
            path = output_dir / f"sample-{case_id}-{route}_cfg1.png"
            images[0].save(path)
            records.append(
                {
                    "case_id": case_id,
                    "route": route + "_cfg1",
                    "path": str(path),
                    "sha256": file_sha256(path),
                    "seed": seed,
                    "initial_latent_sha256": _tensor_hash(actual),
                    "prompt": prompt,
                    "text_guidance_scale": 1.0,
                    "sampling_steps": args.sampling_steps,
                    "target_free": True,
                    "quality_claim": False,
                    "teacher_assisted": route not in ("native_pretrained", "prism_aligned"),
                    "actual_condition_verified": True,
                    "template_swap": True,
                    "condition": condition_audit,
                }
            )
            backend.last_trace = {}
    return records
