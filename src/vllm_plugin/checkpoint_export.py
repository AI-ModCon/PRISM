"""Export a PRISM training checkpoint into a vLLM-loadable directory.

Input:  PRISM checkpoint (model.safetensors, training_state.json) + a hydra
        run dir or explicit backbone_id / d_img.
Output: <out_dir>/
            config.json          # HF config of the backbone, with extra
                                 # `prism_config` and `architectures` fields
            model.safetensors    # weights, key-renamed for the vLLM model class
            tokenizer files      # copied from the HF backbone

Usage:
    python -m src.vllm_plugin.checkpoint_export \\
        --checkpoint outputs/SMOKE-TEST/.../checkpoints/step_500/model.safetensors \\
        --backbone allenai/OLMo-1B-0724-hf \\
        --image-encoder google/siglip2-base-patch16-224 \\
        --out exported/prism-olmo1b-image

    # Multi-modality (VLLM-4+):
    python -m src.vllm_plugin.checkpoint_export \\
        --checkpoint outputs/.../step_N/model.safetensors \\
        --backbone allenai/OLMo-1B-0724-hf \\
        --image-encoder google/siglip2-base-patch16-224 \\
        --active-modalities image,time_series \\
        --ts-max-length 512 --ts-num-vars 1 --ts-patch-size 16 \\
        --out exported/prism-olmo1b-image-ts
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoTokenizer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# Modalities the exporter knows how to emit. `dna` is reserved for the
# incoming BioReason work — it has no Modality enum entry yet, but allowing
# it here means BioReason exports won't trip the strict-unknown check.
# All other names must match `src/modalities.py::Modality`.
KNOWN_MODALITIES: tuple[str, ...] = (
    "text",
    "image",
    "table",
    "time_series",
    "geometry",
    "graph",
    "dna",
)

# Default placeholder tokens per modality. Override via the per-modality
# `--*-placeholder-token` flag if a different special token is wanted.
DEFAULT_PLACEHOLDERS: dict[str, str] = {
    m: f"<{m}>" for m in KNOWN_MODALITIES if m != "text"
}


def build_key_prefix_map(active_modalities: list[str]) -> list[tuple[str, str]]:
    """PRISM checkpoint prefix -> vLLM model attribute prefix.

    Order matters: most specific first. Image keeps the historical
    `vision_tower.` / `multi_modal_projector.` aliases so PR #41 exports stay
    structurally identical when only image is active. Other active modalities
    map under `encoders.<m>.` -> `encoders.<m>.` (model class uses a
    ModuleDict) and `projectors.<m>.` -> `multi_modal_projectors.<m>.`.
    """
    pairs: list[tuple[str, str]] = [("backbone.", "language_model.")]

    # Image-only checkpoints are the PR #41 baseline; preserve the legacy
    # rename so existing exports round-trip identically. Use set equality so
    # `["image", "image"]` or a sorted/canonicalized list still matches.
    image_only = set(active_modalities) == {"image"}

    for m in active_modalities:
        if m == "image" and image_only:
            pairs.append(("encoders.image.", "vision_tower."))
            pairs.append(("image_encoder.", "vision_tower."))
            pairs.append(("projectors.image.", "multi_modal_projector."))
        else:
            # Per-modality ModuleDict layout (VLLM-2 owns the attribute).
            pairs.append((f"encoders.{m}.", f"encoders.{m}."))
            pairs.append((f"projectors.{m}.", f"multi_modal_projectors.{m}."))
            if m == "image":
                # Also accept the legacy image_encoder.* prefix that some
                # PR #41 callers used.
                pairs.append(("image_encoder.", "encoders.image."))

    return pairs


def build_drop_prefixes(
    active_modalities: list[str],
    *,
    rpc_modalities: list[str] | None = None,
) -> tuple[str, ...]:
    """Drop every encoder/projector that is NOT in the active list.

    Text is always dropped regardless of `active_modalities` — it has no
    standalone encoder/projector on the vLLM side (the LM's embed_tokens
    handles it), so the trainer's text-encoder/projector weights have no
    target to load into.

    `rpc_modalities` additionally drops the *encoder* prefix (not the
    projector) for any active modality that runs out-of-process (e.g.
    Intern-S2 in its sidecar — see tools/intern_s2_sidecar.py). Those
    weights no longer live in this process's nn.Module tree, so
    AutoWeightsLoader must never see them; the small trained connector
    (`projectors.<m>.`) still loads normally, since only the encoder moved.
    """
    inactive = [m for m in KNOWN_MODALITIES if m not in active_modalities]
    # Drop both prefixes for text unconditionally — `_validate_active_modalities`
    # already rejects text from active, so the inactive loop would always cover
    # it, but the explicit drop also guards the older code paths and any future
    # refactor that loosens the validation.
    drops: list[str] = ["encoders.text.", "projectors.text."]
    for m in inactive:
        if m == "text":
            continue  # already covered unconditionally above
        drops.append(f"encoders.{m}.")
        drops.append(f"projectors.{m}.")
    for m in rpc_modalities or ():
        drops.append(f"encoders.{m}.")
    return tuple(drops)


def _validate_active_modalities(active_modalities: list[str]) -> list[str]:
    """Return a canonical (deduped, original-order) active list or raise.

    Rejects unknown names against `KNOWN_MODALITIES` and rejects `text`
    (text has no encoder/projector and is implicit via the LM head).
    """
    if not active_modalities:
        raise ValueError("active_modalities is empty")
    seen: set[str] = set()
    canonical: list[str] = []
    for m in active_modalities:
        if m in seen:
            continue
        if m == "text":
            raise ValueError(
                "'text' is not a separately-exportable modality; it is handled "
                "by the language model's embed_tokens."
            )
        if m not in KNOWN_MODALITIES:
            known = ", ".join(sorted(KNOWN_MODALITIES))
            raise ValueError(
                f"Unknown modality {m!r}. Known: {known}. Add it to "
                "KNOWN_MODALITIES (and DEFAULT_PLACEHOLDERS) in "
                "checkpoint_export.py if this is a new modality."
            )
        seen.add(m)
        canonical.append(m)
    return canonical


def remap_state_dict(
    sd: dict[str, torch.Tensor],
    active_modalities: list[str],
    *,
    rpc_modalities: list[str] | None = None,
) -> dict[str, torch.Tensor]:
    """Strip distributed prefixes, drop unused modalities, remap to vLLM names.

    `rpc_modalities` (e.g. `["time_series"]` when `--ts-intern-s2-rpc` is
    set) excludes that modality's encoder weights from the export — see
    `build_drop_prefixes`.
    """
    key_map = build_key_prefix_map(active_modalities)
    drop_prefixes = build_drop_prefixes(active_modalities, rpc_modalities=rpc_modalities)

    out: dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if k.startswith("module."):
            k = k[len("module.") :]
        if k.startswith("_orig_mod."):
            k = k[len("_orig_mod.") :]

        if any(k.startswith(p) for p in drop_prefixes):
            continue

        new_k = k
        for src, dst in key_map:
            if k.startswith(src):
                new_k = dst + k[len(src) :]
                break
        out[new_k] = v
    return out


def detect_d_model(sd: dict[str, torch.Tensor]) -> int:
    for candidate in (
        "language_model.model.embed_tokens.weight",
        "backbone.model.embed_tokens.weight",
    ):
        if candidate in sd:
            return int(sd[candidate].shape[1])
    raise RuntimeError("Could not detect d_model from checkpoint")


def detect_lm_vocab_size(sd: dict[str, torch.Tensor]) -> int:
    """Return embed_tokens.weight.shape[0] — the row count the LM can index."""
    for candidate in (
        "language_model.model.embed_tokens.weight",
        "backbone.model.embed_tokens.weight",
    ):
        if candidate in sd:
            return int(sd[candidate].shape[0])
    raise RuntimeError("Could not detect LM vocab size from checkpoint")


def detect_modality_token_id(
    tokenizer,
    modality: str,
    requested: str | None = None,
) -> tuple[str, int]:
    """Find or add a placeholder special token for `modality`.

    Adds a new id via `add_special_tokens` if the token isn't present in the
    vocab. Asserts the resulting id is real (not the tokenizer's unk_token).
    Generalization of PR #41's `detect_image_token_id`.
    """
    token = requested or DEFAULT_PLACEHOLDERS.get(modality) or f"<{modality}>"
    tok_id = tokenizer.convert_tokens_to_ids(token)
    unk_id = getattr(tokenizer, "unk_token_id", None)
    if tok_id is None or tok_id == unk_id:
        added = tokenizer.add_special_tokens(
            {"additional_special_tokens": [token]}
        )
        if added > 0:
            logger.info("Added new special token %r to tokenizer", token)
        tok_id = tokenizer.convert_tokens_to_ids(token)
    if tok_id is None or tok_id == unk_id:
        raise RuntimeError(
            f"Tokenizer could not resolve {modality} placeholder {token!r}; "
            f"check that the tokenizer supports add_special_tokens."
        )
    return token, int(tok_id)


# ---------------------------------------------------------------------------
# Per-modality config block builders


def _image_block(
    *,
    placeholder_token: str,
    placeholder_token_id: int,
    encoder_model: str,
    d_img: int,
    num_image_tokens: int,
    image_size: int,
    projector_kwargs: dict,
) -> dict:
    return {
        "encoder_model": encoder_model,
        "placeholder_token": placeholder_token,
        "placeholder_token_id": placeholder_token_id,
        "d_img": d_img,
        "num_image_tokens": num_image_tokens,
        "image_size": image_size,
        "projector": projector_kwargs,
    }


def _time_series_block(
    *,
    placeholder_token: str,
    placeholder_token_id: int,
    encoder_model: str,
    max_ts_length: int,
    num_vars: int,
    patch_size: int,
    encoder_type: str,
    projector_kwargs: dict,
    d_ts: int | None = None,
    ts_start_id: int | None = None,
    ts_end_id: int | None = None,
    ts_tokens_per_instance: int | None = None,
    intern_s2_rpc: bool = False,
    intern_s2_rpc_hidden_dim: int | None = None,
) -> dict:
    block: dict = {
        "encoder_model": encoder_model,
        "encoder_type": encoder_type,
        "placeholder_token": placeholder_token,
        "placeholder_token_id": placeholder_token_id,
        "max_ts_length": max_ts_length,
        "num_vars": num_vars,
        "patch_size": patch_size,
        "projector": projector_kwargs,
    }
    if d_ts is not None:
        block["d_ts"] = int(d_ts)
    if intern_s2_rpc:
        if encoder_type not in ("intern_s2", "intern_s2_397b"):
            raise ValueError(
                "intern_s2_rpc=True is only valid for encoder_type in "
                f"('intern_s2', 'intern_s2_397b'); got encoder_type={encoder_type!r}."
            )
        if intern_s2_rpc_hidden_dim is None:
            raise ValueError(
                "intern_s2_rpc=True requires --ts-intern-s2-rpc-hidden-dim "
                "(the encoder no longer runs in this process, so its hidden "
                "dim can't be probed from a live TimeSeriesEncoder instance)."
            )
        block["intern_s2_rpc"] = True
        block["intern_s2_rpc_hidden_dim"] = int(intern_s2_rpc_hidden_dim)
    # intern_s2_397b's Q-former subsampling has no closed-form token-count
    # formula (unlike the other encoder types), so the vLLM processor can't
    # derive num_tokens() on its own; the caller must supply the value
    # measured via TimeSeriesEncoder.tokens_per_instance() at training time.
    if encoder_type == "intern_s2_397b" and ts_tokens_per_instance is None:
        raise ValueError(
            "encoder_type='intern_s2_397b' requires --ts-tokens-per-instance "
            "(compute via TimeSeriesEncoder(...).tokens_per_instance() on the "
            "trained config before exporting)."
        )
    if ts_tokens_per_instance is not None:
        block["ts_tokens_per_instance"] = int(ts_tokens_per_instance)
    # ts_start_id / ts_end_id come from training's
    # model_config.modality_start_end_token_indices["time_series"]. Both or
    # neither — the processor refuses a half-configured pair.
    if (ts_start_id is None) != (ts_end_id is None):
        raise ValueError(
            "ts_start_id and ts_end_id must both be set or both be None; "
            f"got start={ts_start_id!r} end={ts_end_id!r}"
        )
    # Both-or-neither is enforced by the check above, so the `ts_end_id`
    # half is redundant at runtime; it is spelled out so the narrowing is
    # visible to readers and to the type checker.
    if ts_start_id is not None and ts_end_id is not None:
        block["ts_start_id"] = int(ts_start_id)
        block["ts_end_id"] = int(ts_end_id)
    return block


def export(
    checkpoint: Path,
    backbone_id: str,
    image_encoder: str,
    out_dir: Path,
    *,
    active_modalities: list[str] | None = None,
    image_token: str | None = None,
    num_image_tokens: int = 196,
    image_size: int = 224,
    d_img: int = 768,
    # Time-series knobs (used when "time_series" is active).
    ts_encoder_model: str = "Salesforce/moirai-2.0-R-small",
    ts_encoder_type: str = "moirai",
    ts_max_length: int = 512,
    ts_num_vars: int = 1,
    ts_patch_size: int = 16,
    ts_placeholder_token: str | None = None,
    ts_d_ts: int | None = None,
    ts_start_id: int | None = None,
    ts_end_id: int | None = None,
    ts_tokens_per_instance: int | None = None,
    ts_intern_s2_rpc: bool = False,
    ts_intern_s2_rpc_hidden_dim: int | None = None,
    # Generic knobs.
    language_model_arch_override: str | None = None,
    projector_kwargs: dict | None = None,
) -> None:
    active = _validate_active_modalities(active_modalities or ["image"])
    out_dir.mkdir(parents=True, exist_ok=True)

    if checkpoint.is_dir():
        ckpt_file = checkpoint / "model.safetensors"
    else:
        ckpt_file = checkpoint
    if not ckpt_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_file}")

    logger.info("Loading checkpoint: %s", ckpt_file)
    sd = load_file(str(ckpt_file))
    rpc_modalities = ["time_series"] if ts_intern_s2_rpc and "time_series" in active else []
    sd = remap_state_dict(sd, active, rpc_modalities=rpc_modalities)
    d_model = detect_d_model(sd)
    logger.info(
        "Detected d_model=%d, %d weight tensors after remap (active=%s)",
        d_model,
        len(sd),
        active,
    )

    save_file(sd, str(out_dir / "model.safetensors"))

    # Tokenizer: one placeholder per active modality.
    tokenizer = AutoTokenizer.from_pretrained(backbone_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    per_modality: dict[str, dict] = {}
    default_projector = projector_kwargs or {
        "norm_mode": "layernorm",
        "modality_embed_pos": "after_norm",
        "modality_embed_scale": 0.02,
    }

    if "image" in active:
        img_tok, img_tok_id = detect_modality_token_id(
            tokenizer, "image", image_token
        )
        per_modality["image"] = _image_block(
            placeholder_token=img_tok,
            placeholder_token_id=img_tok_id,
            encoder_model=image_encoder,
            d_img=d_img,
            num_image_tokens=num_image_tokens,
            image_size=image_size,
            projector_kwargs=default_projector,
        )

    if "time_series" in active:
        ts_tok, ts_tok_id = detect_modality_token_id(
            tokenizer, "time_series", ts_placeholder_token
        )
        # ts_start_id / ts_end_id come from training's
        # modality_start_end_token_indices["time_series"] (see
        # src/conf/model/prism_olmo1b_linear_interleaved_ts.yaml:17).
        # Training picks IDs that already exist in the custom tokenizer AND
        # have a row in embed_tokens.weight (OLMo's padded vocab leaves
        # unused IDs between 50256 and 50304). vLLM's text path runs
        # embed_tokens on every non-feature input_id, so any envelope ID we
        # write must be < vocab_size — otherwise inference OOB-faults.
        #
        # Refuse to auto-add: a freshly-added special token gets ID
        # `vocab_size`, which has no embedding row. The user must pass the
        # IDs from training config explicitly. The data plane (no envelope)
        # remains valid for callers who only need to send `<time_series>`.
        if ts_start_id is not None and ts_end_id is not None:
            vocab_size = detect_lm_vocab_size(sd)
            for name, val in (("ts_start_id", ts_start_id),
                              ("ts_end_id", ts_end_id)):
                if not 0 <= int(val) < vocab_size:
                    raise ValueError(
                        f"{name}={val} is out of range for the LM's embed "
                        f"matrix (vocab_size={vocab_size}). Pick IDs that "
                        "exist in the trained tokenizer AND fall within the "
                        "padded vocab; see training's "
                        "modality_start_end_token_indices['time_series']."
                    )
        per_modality["time_series"] = _time_series_block(
            placeholder_token=ts_tok,
            placeholder_token_id=ts_tok_id,
            encoder_model=ts_encoder_model,
            encoder_type=ts_encoder_type,
            max_ts_length=ts_max_length,
            num_vars=ts_num_vars,
            patch_size=ts_patch_size,
            projector_kwargs=default_projector,
            d_ts=ts_d_ts,
            ts_start_id=ts_start_id,
            ts_end_id=ts_end_id,
            ts_tokens_per_instance=ts_tokens_per_instance,
            intern_s2_rpc=ts_intern_s2_rpc,
            intern_s2_rpc_hidden_dim=ts_intern_s2_rpc_hidden_dim,
        )

    # Every active modality must have produced a block above. Anything
    # missing means a new modality was added to KNOWN_MODALITIES without a
    # matching builder branch — and the load-side processor registry
    # (src/vllm_plugin/processors/registry.py::MODALITY_PROCESSORS) would
    # KeyError anyway. Fail loud at export time.
    missing = [m for m in active if m not in per_modality]
    if missing:
        raise NotImplementedError(
            f"No exporter branch for active modalities {missing}. Add a "
            "per-modality block builder (see _image_block / _time_series_block) "
            "and register a matching ModalityProcessor before exporting these."
        )

    tokenizer.save_pretrained(str(out_dir))

    backbone_config = AutoConfig.from_pretrained(backbone_id)
    backbone_config_dict = backbone_config.to_dict()
    backbone_config_dict.setdefault("torch_dtype", "bfloat16")
    backbone_config_dict["architectures"] = ["PrismForConditionalGeneration"]

    prism_block: dict = {
        "active_modalities": list(active),
        "d_model": d_model,
        "language_model_arch_override": (
            [language_model_arch_override] if language_model_arch_override else None
        ),
        **per_modality,
    }

    # Back-compat mirror: when image is active, also publish the flat keys
    # PR #41 callers may still look at (image_token, image_token_id,
    # num_image_tokens, image_size, d_img, image_encoder_model, projector).
    # _normalize_prism_config() (in prism_for_conditional_generation.py)
    # makes this purely additive; nothing reads BOTH paths.
    if "image" in active:
        img = per_modality["image"]
        prism_block.update({
            "image_encoder_model": img["encoder_model"],
            "image_token": img["placeholder_token"],
            "image_token_id": img["placeholder_token_id"],
            "num_image_tokens": img["num_image_tokens"],
            "image_size": img["image_size"],
            "d_img": img["d_img"],
            "projector": img["projector"],
        })

    backbone_config_dict["prism_config"] = prism_block

    with open(out_dir / "config.json", "w") as f:
        json.dump(backbone_config_dict, f, indent=2)

    logger.info("Wrote vLLM-compatible export to: %s", out_dir)
    for m in active:
        block = per_modality[m]
        logger.info(
            "  %s: token=%r id=%d",
            m,
            block["placeholder_token"],
            block["placeholder_token_id"],
        )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument(
        "--backbone",
        required=True,
        help="HF id of the LLM backbone, e.g. allenai/OLMo-1B-0724-hf",
    )
    p.add_argument("--image-encoder", default="google/siglip2-base-patch16-224")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument(
        "--active-modalities",
        default="image",
        help="Comma-separated active modality list (default: image).",
    )
    # Image
    p.add_argument("--image-token", default="<image>")
    p.add_argument("--num-image-tokens", type=int, default=196)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--d-img", type=int, default=768)
    # Time-series
    p.add_argument(
        "--ts-encoder-model", default="Salesforce/moirai-2.0-R-small"
    )
    p.add_argument(
        "--ts-encoder-type",
        default="moirai",
        choices=["moirai", "linear", "intern_s2", "intern_s2_397b", "timeomni"],
    )
    p.add_argument("--ts-max-length", type=int, default=512)
    p.add_argument("--ts-num-vars", type=int, default=1)
    p.add_argument("--ts-patch-size", type=int, default=16)
    p.add_argument("--ts-placeholder-token", default=None)
    p.add_argument(
        "--ts-d-ts",
        type=int,
        default=None,
        help="Override TimeSeriesEncoder d_ts (defaults to the backbone "
        "d_model). Must match training's d_ts for the linear encoder.",
    )
    p.add_argument(
        "--ts-start-id",
        type=int,
        default=None,
        help="Token id that opens the time-series envelope in the post-update "
        "sequence. Must come from training's "
        "modality_start_end_token_indices['time_series'][0]. Required together "
        "with --ts-end-id; both default to None (no envelope, plain placeholder "
        "replacement — use only when training didn't use start/end tokens).",
    )
    p.add_argument(
        "--ts-end-id",
        type=int,
        default=None,
        help="Token id that closes the time-series envelope. Pair with --ts-start-id.",
    )
    p.add_argument(
        "--ts-tokens-per-instance",
        type=int,
        default=None,
        help="Required for --ts-encoder-type=intern_s2_397b: the fixed "
        "placeholder token count, measured via "
        "TimeSeriesEncoder(...).tokens_per_instance() on the trained config "
        "(no closed-form formula exists for this encoder's subsampling).",
    )
    p.add_argument(
        "--ts-intern-s2-rpc",
        action="store_true",
        help="Run intern_s2/intern_s2_397b in a separate sidecar process "
        "(tools/intern_s2_sidecar.py) instead of importing it into vLLM's "
        "process. Required because Intern-S2's vendored config needs "
        "transformers>=5.2.0, incompatible with vLLM serving's "
        "transformers<5 floor. Excludes the encoder's weights from this "
        "export (the sidecar loads them separately); the small trained "
        "connector still exports normally. Requires "
        "--ts-intern-s2-rpc-hidden-dim.",
    )
    p.add_argument(
        "--ts-intern-s2-rpc-hidden-dim",
        type=int,
        default=None,
        help="Required with --ts-intern-s2-rpc: the encoder's hidden dim. "
        "Can't be derived without instantiating TimeSeriesEncoder (the "
        "thing RPC mode avoids importing), so it must be supplied "
        "explicitly — same reasoning as --ts-tokens-per-instance.",
    )
    # Generic
    p.add_argument(
        "--language-model-arch",
        default=None,
        help="Override the architecture used to build the language model.",
    )
    args = p.parse_args()

    active = [s.strip() for s in args.active_modalities.split(",") if s.strip()]
    export(
        checkpoint=args.checkpoint,
        backbone_id=args.backbone,
        image_encoder=args.image_encoder,
        out_dir=args.out,
        active_modalities=active,
        image_token=args.image_token,
        num_image_tokens=args.num_image_tokens,
        image_size=args.image_size,
        d_img=args.d_img,
        ts_encoder_model=args.ts_encoder_model,
        ts_encoder_type=args.ts_encoder_type,
        ts_max_length=args.ts_max_length,
        ts_num_vars=args.ts_num_vars,
        ts_patch_size=args.ts_patch_size,
        ts_placeholder_token=args.ts_placeholder_token,
        ts_d_ts=args.ts_d_ts,
        ts_start_id=args.ts_start_id,
        ts_end_id=args.ts_end_id,
        ts_tokens_per_instance=args.ts_tokens_per_instance,
        ts_intern_s2_rpc=args.ts_intern_s2_rpc,
        ts_intern_s2_rpc_hidden_dim=args.ts_intern_s2_rpc_hidden_dim,
        language_model_arch_override=args.language_model_arch,
    )


if __name__ == "__main__":
    main()
