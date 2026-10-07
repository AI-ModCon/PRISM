"""Build a synthetic time-series checkpoint for VLLM-5 smoke testing.

A real time-series training checkpoint doesn't exist yet, but VLLM-5 needs
to exercise the full plugin path: model class loads `encoders.time_series.*`
+ `multi_modal_projectors.time_series.*` weights, instantiates a
TimeSeriesEncoder, runs forward on a tensor, splices into the LM stream.

This script composes:
  - A real OLMo backbone (weights from HF cache, key prefix `backbone.`)
  - A randomly-initialized TimeSeriesEncoder (linear encoder type by
    default — Moirai requires uni2ts and a network download to a checkpoint
    that may not match d_ts perfectly)
  - A randomly-initialized ModalityProjector for time_series

Saves to <out>/model.safetensors with the same key layout the trainer
produces, ready to feed into `python -m src.vllm_plugin.checkpoint_export
--checkpoint <out>/model.safetensors --active-modalities image,time_series
--ts-encoder-type linear --ts-d-ts <same d_ts used here>`. Add
`--ts-start-id/--ts-end-id` only when the LM tokenizer already has
embedding rows at those IDs (training stamps these from
`modality_start_end_token_indices`).

Random init produces meaningless outputs — this checkpoint validates the
PLUMBING (load + forward + splice + generate without error), not behavior.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _maybe_seed(seed: int | None) -> None:
    if seed is None:
        return
    torch.manual_seed(seed)


def build_state_dict(
    backbone_id: str,
    *,
    encoder_type: str,
    num_vars: int,
    d_ts: int,
    max_ts_length: int,
    moirai_model: str,
    include_image: bool,
) -> dict[str, torch.Tensor]:
    """Compose a checkpoint dict with backbone + ts encoder + ts projector.

    Optionally also include random-init image encoder + projector weights
    so the export can produce a dual-modality checkpoint (image trivially
    passes through PR #41's loader; the ts side is the new code path).
    """
    from src.encoders.time_series import TimeSeriesEncoder
    from src.modules.projector import ModalityProjector

    logger.info("Loading backbone weights: %s", backbone_id)
    backbone = AutoModelForCausalLM.from_pretrained(
        backbone_id, torch_dtype=torch.float32
    )

    sd: dict[str, torch.Tensor] = {}
    for name, tensor in backbone.state_dict().items():
        sd[f"backbone.{name}"] = tensor.detach().contiguous()
    logger.info("Backbone weights: %d tensors", len(sd))
    d_model = int(backbone.config.hidden_size)
    logger.info("Detected d_model=%d", d_model)

    # TimeSeriesEncoder (random init; linear is local-only, moirai pulls uni2ts).
    logger.info(
        "Building TimeSeriesEncoder(type=%s, num_vars=%d, d_ts=%d, "
        "max_ts_length=%d)",
        encoder_type, num_vars, d_ts, max_ts_length,
    )
    ts_encoder = TimeSeriesEncoder(
        encoder_type=encoder_type,
        num_vars=num_vars,
        d_ts=d_ts,
        model_name=moirai_model,
        max_ts_length=max_ts_length,
    )
    ts_encoder_sd = ts_encoder.state_dict()
    for name, tensor in ts_encoder_sd.items():
        sd[f"encoders.time_series.{name}"] = tensor.detach().contiguous()
    logger.info(
        "TimeSeriesEncoder weights: %d tensors (hidden_dim=%d)",
        len(ts_encoder_sd),
        getattr(ts_encoder, "hidden_dim", d_ts),
    )

    # Projector for time_series.
    ts_proj_input = int(getattr(ts_encoder, "hidden_dim", d_ts))
    logger.info(
        "Building ModalityProjector(input_dim=%d, d_model=%d) for time_series",
        ts_proj_input,
        d_model,
    )
    ts_projector = ModalityProjector(input_dim=ts_proj_input, d_model=d_model)
    for name, tensor in ts_projector.state_dict().items():
        sd[f"projectors.time_series.{name}"] = tensor.detach().contiguous()

    # Optional image side — random-init, just to exercise the multi-modality
    # export path. For real image parity, use the real PR #41 export.
    if include_image:
        from transformers import AutoConfig, AutoModel

        logger.info("Building random-init image encoder + projector")
        siglip_cfg = AutoConfig.from_pretrained(
            "google/siglip2-base-patch16-224"
        )
        siglip = AutoModel.from_config(siglip_cfg)
        if hasattr(siglip, "vision_model"):
            siglip = siglip.vision_model
        for name, tensor in siglip.state_dict().items():
            sd[f"encoders.image.model.{name}"] = tensor.detach().contiguous()

        img_d_img = int(getattr(siglip.config, "hidden_size", 768))
        img_projector = ModalityProjector(input_dim=img_d_img, d_model=d_model)
        for name, tensor in img_projector.state_dict().items():
            sd[f"projectors.image.{name}"] = tensor.detach().contiguous()

    return sd


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--backbone",
        default="allenai/OLMo-1B-0724-hf",
        help="HF id of the LLM backbone to pull weights from.",
    )
    p.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output directory. Writes model.safetensors here.",
    )
    p.add_argument(
        "--ts-encoder-type",
        default="linear",
        choices=["linear", "moirai"],
        help="linear (default) builds locally without network; moirai pulls "
        "uni2ts and downloads the Salesforce/moirai model.",
    )
    p.add_argument(
        "--ts-encoder-model",
        default="Salesforce/moirai-2.0-R-small",
        help="Moirai HF id (only used when --ts-encoder-type=moirai).",
    )
    p.add_argument("--ts-num-vars", type=int, default=1)
    p.add_argument(
        "--ts-d-ts",
        type=int,
        default=None,
        help="TimeSeriesEncoder.d_ts. Defaults to the backbone's d_model "
        "(so the projector has a square input).",
    )
    p.add_argument("--ts-max-length", type=int, default=512)
    p.add_argument(
        "--include-image",
        action="store_true",
        help="Also pack a random-init SigLIP2 image encoder + projector.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Manual seed for reproducible random init.",
    )
    args = p.parse_args()

    _maybe_seed(args.seed)

    # If d_ts not given, default to backbone d_model — projector input == d_ts
    # then becomes a "square" linear that's easy to size.
    d_ts = args.ts_d_ts
    if d_ts is None:
        # Cheap peek at the backbone config to learn d_model without loading
        # weights.
        from transformers import AutoConfig

        bcfg = AutoConfig.from_pretrained(args.backbone)
        d_ts = int(bcfg.hidden_size)
        logger.info("--ts-d-ts not set; defaulting to backbone d_model=%d", d_ts)

    sd = build_state_dict(
        args.backbone,
        encoder_type=args.ts_encoder_type,
        num_vars=args.ts_num_vars,
        d_ts=d_ts,
        max_ts_length=args.ts_max_length,
        moirai_model=args.ts_encoder_model,
        include_image=args.include_image,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    out_file = args.out / "model.safetensors"
    save_file(sd, str(out_file))
    logger.info("Wrote synthetic checkpoint: %s (%d tensors)", out_file, len(sd))


if __name__ == "__main__":
    main()
