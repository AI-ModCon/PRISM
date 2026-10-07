"""Local, explicit checkpoint loading for the image-connector experiment."""

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch


def file_sha256(path):
    """Hash a file's bytes for provenance.

    Args:
        path: file to read, streamed in 1 MiB blocks.

    Returns:
        The hex-encoded SHA-256 digest of the file's contents.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _local_snapshot(identifier):
    from ..site_paths import require_resolved

    # An unconfigured site variable would otherwise reach the hub as a repo id
    # and fail as a confusing "repository not found".
    require_resolved(identifier, f"Model asset {identifier!r}")
    if Path(identifier).is_dir():
        return str(Path(identifier).resolve())
    from huggingface_hub import snapshot_download

    return snapshot_download(identifier, local_files_only=True)


def preprocessing_sha256(path):
    """Content identity for staged tokenizer/processor assets, excluding weights."""
    root = Path(path).resolve()
    excluded = {".safetensors", ".bin", ".pt", ".pth", ".h5", ".msgpack"}
    files = {
        p.relative_to(root).as_posix(): file_sha256(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.suffix not in excluded and ".git" not in p.parts
    }
    if not files:
        raise ValueError(f"No preprocessing assets found in {root}")
    return hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _clean_parent_state(state):
    """Remove distributed/compiled wrappers without silently replacing tensors."""
    from . import remap_legacy_decoder_keys

    cleaned = {}
    renamed = {}
    for key, value in state.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise ValueError("Parent checkpoint must contain named tensors only")
        parts = key.split(".")
        while parts and parts[0] in {"module", "_fsdp_wrapped_module", "_orig_mod"}:
            parts.pop(0)
        # torch.compile/FSDP can also wrap an individual backbone block.
        clean_key = ".".join(
            part for part in parts if part not in {"_orig_mod", "_fsdp_wrapped_module"}
        )
        clean_key = next(iter(remap_legacy_decoder_keys({clean_key: value})))
        if not clean_key or clean_key in cleaned:
            raise ValueError(f"Parent checkpoint keys collide after normalization: {clean_key}")
        cleaned[clean_key] = value
        if clean_key != key:
            renamed[key] = clean_key
    return cleaned, renamed


def _align_parent_vocab(model, state, tokenizer):
    """Restore a saved HF vocabulary, including unpadded PRISM-Harness exports.

    Discover the embedding through the HF interface, not a Qwen-specific name.
    Resizing is safe only when every possible tokenizer ID still has a row;
    strict restoration below then replaces every retained embedding/head tensor.
    """
    backbone = getattr(model, "backbone", None)
    get_embeddings = getattr(backbone, "get_input_embeddings", None)
    embeddings = get_embeddings() if callable(get_embeddings) else None
    if embeddings is None or not hasattr(embeddings, "weight"):
        return None
    names = [
        f"{name}.weight"
        for name, module in model.named_modules(remove_duplicate=False)
        if module is embeddings and f"{name}.weight" in state
    ]
    if not names:
        return None  # The strict missing-parent check supplies the error.
    saved = state[names[0]]
    if saved.ndim != 2 or saved.shape[1:] != embeddings.weight.shape[1:]:
        raise ValueError("Parent checkpoint input embedding dimensions are incompatible")
    rows = int(saved.shape[0])
    required_rows = len(tokenizer)
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if callable(get_vocab):
        ids = list(get_vocab().values())
        if ids:
            required_rows = max(required_rows, max(ids) + 1)
    special_ids = getattr(tokenizer, "all_special_ids", ())
    if special_ids:
        required_rows = max(required_rows, max(special_ids) + 1)
    if rows < required_rows:
        raise ValueError(
            f"Parent checkpoint vocabulary {rows} cannot represent tokenizer IDs "
            f"requiring {required_rows} rows"
        )
    old_rows = int(embeddings.weight.shape[0])
    if rows == old_rows:
        return None
    resize_token_embeddings = getattr(backbone, "resize_token_embeddings", None)
    if not callable(resize_token_embeddings):
        raise ValueError("Parent backbone cannot resize its vocabulary to the checkpoint")
    # The output connector's initialization must not depend on temporary rows.
    with torch.random.fork_rng(devices=[]):
        resize_token_embeddings(rows)
    return {"embedding_key": names[0], "old_rows": old_rows, "checkpoint_rows": rows}


def restore_prism_parent(model, state, tokenizer):
    """Strictly restore the full trained parent while retaining checkpoint dtypes.

    ``assign=True`` avoids silently rounding a BF16/FP32 trained LM through the
    temporary FP16 CPU backbone constructed by UnifiedTransformer. No optimizer
    may exist yet. Validate and restore any parameter/buffer aliases explicitly,
    because assigning checkpoint tensors can otherwise break tied weights.
    """
    state, renamed = _clean_parent_state(state)
    vocab_change = _align_parent_vocab(model, state, tokenizer)
    expected = model.state_dict()
    missing = set(expected) - set(state)
    unexpected = set(state) - set(expected)
    if unexpected or any(not key.startswith("decoders.image.connector.") for key in missing):
        raise ValueError(
            f"Parent checkpoint incompatible: missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    wrong_shapes = {
        key: (tuple(state[key].shape), tuple(expected[key].shape))
        for key in state
        if state[key].shape != expected[key].shape
    }
    if wrong_shapes:
        raise ValueError(f"Parent checkpoint tensor shapes are incompatible: {wrong_shapes}")
    complete = {**{key: expected[key] for key in missing}, **state}
    groups = defaultdict(list)
    for name, value in (
        *model.named_parameters(remove_duplicate=False),
        *model.named_buffers(remove_duplicate=False),
    ):
        if name in complete:
            groups[id(value)].append(name)
    aliases = [names for names in groups.values() if len(names) > 1]
    for names in aliases:
        first = complete[names[0]]
        if any(
            complete[name].dtype != first.dtype or not torch.equal(complete[name], first)
            for name in names[1:]
        ):
            raise ValueError(f"Parent checkpoint has inconsistent tied tensors: {names}")
    model.load_state_dict(complete, strict=True, assign=True)
    for names in aliases:
        module_name, _, attribute = names[0].rpartition(".")
        value = getattr(model.get_submodule(module_name), attribute)
        for name in names[1:]:
            module_name, _, attribute = name.rpartition(".")
            setattr(model.get_submodule(module_name), attribute, value)
    return {
        "strict_parent": True,
        "missing_parent_keys": [],
        "unexpected_keys": [],
        "new_connector_keys": sorted(missing),
        "loaded_key_count": len(state),
        "loaded_keys_by_component": dict(Counter(key.split(".")[0] for key in state)),
        "checkpoint_tensor_dtypes": dict(Counter(str(value.dtype) for value in state.values())),
        "dtype_policy": "preserve_checkpoint_tensors",
        "normalized_keys": renamed,
        "backbone_vocab_resize": vocab_change,
        "restored_alias_groups": aliases,
    }


def load_image_training_bundle(model_config, checkpoint, tokenizer, source_processor):
    """Load an aligned PRISM parent plus a new output connector, without downloads.

    The only permitted missing weights are the new image connector. Every
    input encoder, projector, and backbone parameter must be present. External
    frozen OmniGen2 weights load lazily from their separately pinned checkpoint.
    """
    from transformers import AutoImageProcessor, AutoTokenizer

    from ..config import ModelConfig
    from ..model import UnifiedTransformer
    from ..site_paths import expand_tree

    config_path, checkpoint_path = Path(model_config), Path(checkpoint)
    # Shipped configs name site roots as ${PRISM_*} rather than one user's
    # directories; resolve them before the values reach ModelConfig.
    config_values = expand_tree(json.loads(config_path.read_text()))
    config = ModelConfig(**config_values)
    if "image" not in config.output_decoders or not config.llm_backbone_id:
        raise ValueError("Image training needs an HF backbone and configured image output decoder")
    config.llm_backbone_id = _local_snapshot(config.llm_backbone_id)
    if "image" in config.modalities:
        config.image_encoder_id = _local_snapshot(config.image_encoder_id)
    config.llm_tokenizer_id = _local_snapshot(str(tokenizer))
    processor_path = _local_snapshot(str(source_processor))
    tok = AutoTokenizer.from_pretrained(config.llm_tokenizer_id, local_files_only=True)
    if tok.pad_token_id is None:
        raise ValueError("Provide a staged PRISM tokenizer with an explicit pad token")
    processor = AutoImageProcessor.from_pretrained(processor_path, local_files_only=True)
    model = UnifiedTransformer(config)
    model.tokenizer = tok
    if checkpoint_path.suffix == ".safetensors":
        from safetensors.torch import load_file

        payload = load_file(str(checkpoint_path), device="cpu")
    else:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(
            "Parent checkpoint must be a state dict or contain model_state_dict/state_dict"
        )
    state = payload.get("model_state_dict", payload.get("state_dict", payload))
    if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError("Unsupported checkpoint format; export a tensor state dict first")
    restoration = restore_prism_parent(model, state, tok)

    def source_transform(image):
        return processor(images=image, return_tensors="pt")["pixel_values"][0]

    image_config = config.decoder_configs.get("image", {})
    # The explicit connector API nests generator options. Keep the old flat
    # fallback so historical configurations retain their provenance semantics.
    generator_config = image_config.get("generator") or image_config
    return {
        "model": model,
        "tokenizer": tok,
        "source_transform": source_transform,
        "provenance": {
            "parent_checkpoint_sha256": file_sha256(checkpoint_path),
            "model_config_sha256": file_sha256(config_path),
            "parent_checkpoint": str(checkpoint_path.resolve()),
            "tokenizer": config.llm_tokenizer_id,
            "source_processor": processor_path,
            "tokenizer_sha256": preprocessing_sha256(config.llm_tokenizer_id),
            "source_processor_sha256": preprocessing_sha256(processor_path),
            "reference_model_id": generator_config.get("model_id", "OmniGen2/OmniGen2"),
            "reference_revision": generator_config.get("revision"),
            "new_connector_keys": restoration["new_connector_keys"],
            "restoration": restoration,
        },
    }


def load_image_connector(
    model, checkpoint, *, parent_checkpoint_sha256, reference_checkpoint_sha256, allow_fixture=False
):
    """Restore just the output connector after verifying its two parent identities."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    allowed_kind = {"real_checkpoint_training"}
    if allow_fixture:
        allowed_kind.add("fixture_only")
    if payload.get("schema_version") != 1 or payload.get("evidence_kind") not in allowed_kind:
        raise ValueError("Connector checkpoint is not an accepted training artifact")
    provenance = payload.get("provenance", {})
    for key, expected in (
        ("parent_checkpoint_sha256", parent_checkpoint_sha256),
        ("reference_checkpoint_sha256", reference_checkpoint_sha256),
    ):
        if not expected or provenance.get(key) != expected:
            raise ValueError(f"Connector parent identity mismatch: {key}")
    prefix = "decoders.image.connector."
    state = payload.get("connector_state_dict", {})
    expected_keys = {key for key in model.state_dict() if key.startswith(prefix)}
    if set(state) != expected_keys or not expected_keys:
        raise ValueError("Connector checkpoint must contain exactly the configured image connector")
    model.decoders["image"].connector.load_state_dict(
        {key[len(prefix) :]: value for key, value in state.items()},
        strict=True,
    )
    return {"step": payload.get("step"), "provenance": provenance}
