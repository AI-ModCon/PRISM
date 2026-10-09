"""Train material-property regression through PRISM's LLM fusion path."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.config import ModelConfig
from src.model import UnifiedTransformer

from .data import (
    HuggingFaceTokenizer,
    MaterialsDataset,
    build_manifest,
    collate_materials,
    split_records,
)
from .prism_model import PRISMMaterialRegressor


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


def regression_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    error = prediction - target
    denominator = ((target - target.mean()) ** 2).sum()
    return {
        "mae": error.abs().mean().item(),
        "rmse": error.square().mean().sqrt().item(),
        "r2": (
            (1 - error.square().sum() / denominator).item()
            if denominator > 0
            else float("nan")
        ),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train text + periodic crystal graphs with PRISM's LLM fusion path."
    )
    parser.add_argument("--materials-dir", type=Path, default=Path("Materials"))
    parser.add_argument("--target", default="band_gap")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/materials-prism"))
    parser.add_argument("--mode", choices=("joint", "text", "graph"), default="joint")
    parser.add_argument("--text-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--train-text-backbone", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--text-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--graph-layers", type=int, default=3)
    parser.add_argument("--graph-tokens", type=int, default=8)
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--max-neighbors", type=int, default=16)
    parser.add_argument("--max-text-length", type=int, default=192)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--graph-cache-dir", type=Path, default=None)
    parser.add_argument("--no-graph-cache", action="store_true")
    return parser.parse_args(argv)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def make_loader(records, tokenizer, args, cache_dir, shuffle: bool) -> DataLoader:
    dataset = MaterialsDataset(
        records,
        tokenizer,
        max_text_length=args.max_text_length,
        cutoff=args.cutoff,
        max_neighbors=args.max_neighbors,
        graph_cache_dir=cache_dir,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_materials,
        pin_memory=False,
    )


@torch.no_grad()
def evaluate(model, loader, device, mode, target_mean, target_std):
    model.eval()
    predictions = []
    targets = []
    material_ids = []
    for batch in loader:
        material_ids.extend(batch["material_id"])
        batch = move_batch(batch, device)
        normalized, _ = model(batch, mode=mode)
        predictions.append(normalized.cpu() * target_std + target_mean)
        targets.append(batch["target"].cpu())
    prediction = torch.cat(predictions)
    target = torch.cat(targets)
    rows = list(zip(material_ids, target.tolist(), prediction.tolist(), strict=True))
    return regression_metrics(prediction, target), rows


def _trainable_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    return {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if name in trainable
    }


def run(args: argparse.Namespace) -> dict[str, float]:
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    if args.max_samples is not None and args.max_samples < 3:
        raise ValueError("--max-samples must be at least 3")
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = build_manifest(
        args.materials_dir, args.target, max_records=args.max_samples, seed=args.seed
    )
    train_records, val_records, test_records = split_records(
        records, args.val_fraction, args.test_fraction, args.seed
    )
    tokenizer = HuggingFaceTokenizer(args.text_model)
    targets = torch.tensor([record.target for record in train_records])
    target_mean = targets.mean().item()
    target_std = targets.std(unbiased=False).clamp_min(1e-8).item()
    cache_dir = None
    if not args.no_graph_cache:
        cache_dir = args.graph_cache_dir or args.output_dir / "graph_cache"
    train_loader = make_loader(train_records, tokenizer, args, cache_dir, True)
    val_loader = make_loader(val_records, tokenizer, args, cache_dir, False)
    test_loader = make_loader(test_records, tokenizer, args, cache_dir, False)

    device = resolve_device(args.device)
    prism_config = ModelConfig(
        llm_backbone_id=args.text_model,
        llm_tokenizer_id=args.text_model,
        freeze_backbone=not args.train_text_backbone,
        freeze_encoders=False,
        modalities=["text"],
        attn_implementation="sdpa",
    )
    prism = UnifiedTransformer(prism_config)
    prism.tokenizer = tokenizer.tokenizer
    model = PRISMMaterialRegressor(
        prism,
        graph_hidden_dim=args.hidden_dim,
        graph_layers=args.graph_layers,
        cutoff=args.cutoff,
        graph_tokens=args.graph_tokens,
    ).to(device)

    if args.mode == "text":
        for parameter in model.prism.encoders["graph"].parameters():
            parameter.requires_grad = False
        for parameter in model.prism.projectors["graph"].parameters():
            parameter.requires_grad = False

    backbone_parameters = [
        parameter for parameter in model.prism.backbone.parameters() if parameter.requires_grad
    ]
    backbone_ids = {id(parameter) for parameter in backbone_parameters}
    task_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in backbone_ids
    ]
    parameter_groups = [{"params": task_parameters, "lr": args.learning_rate}]
    if backbone_parameters:
        parameter_groups.append(
            {"params": backbone_parameters, "lr": args.text_learning_rate}
        )
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs)
    )

    print(
        f"PRISM paired samples: {len(records):,} "
        f"(train={len(train_records):,}, val={len(val_records):,}, test={len(test_records):,})"
    )
    print(
        f"Target={args.target} mode={args.mode} device={device} "
        f"backbone={args.text_model} ({'trainable' if args.train_text_backbone else 'frozen'}) "
        f"graph_tokens={args.graph_tokens}"
    )

    best_mae = math.inf
    best_state = None
    checkpoint_path = args.output_dir / "best.pt"
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for batch in train_loader:
            batch = move_batch(batch, device)
            normalized_target = (batch["target"] - target_mean) / target_std
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(batch, mode=args.mode, targets=normalized_target)
            assert loss is not None
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad], 5.0
            )
            optimizer.step()
            running_loss += loss.item() * normalized_target.numel()
            seen += normalized_target.numel()
        scheduler.step()
        val_metrics, _ = evaluate(
            model, val_loader, device, args.mode, target_mean, target_std
        )
        result = {
            "epoch": epoch,
            "train_loss": running_loss / seen,
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(result)
        print(
            f"epoch={epoch:03d} loss={result['train_loss']:.5f} "
            f"val_mae={val_metrics['mae']:.5f} val_rmse={val_metrics['rmse']:.5f} "
            f"val_r2={val_metrics['r2']:.5f}"
        )
        if val_metrics["mae"] < best_mae:
            best_mae = val_metrics["mae"]
            best_state = _trainable_state_dict(model)

    torch.save(best_state, checkpoint_path)
    model.load_state_dict(best_state, strict=False)
    test_metrics, prediction_rows = evaluate(
        model, test_loader, device, args.mode, target_mean, target_std
    )
    print(
        f"test_mae={test_metrics['mae']:.5f} test_rmse={test_metrics['rmse']:.5f} "
        f"test_r2={test_metrics['r2']:.5f}"
    )

    metadata = {
        "implementation": "prism",
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "split_sizes": {
            "train": len(train_records),
            "validation": len(val_records),
            "test": len(test_records),
        },
        "split_ids": {
            "train": [record.material_id for record in train_records],
            "validation": [record.material_id for record in val_records],
            "test": [record.material_id for record in test_records],
        },
        "target_mean": target_mean,
        "target_std": target_std,
        "history": history,
        "test_metrics": test_metrics,
    }
    (args.output_dir / "run.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    with (args.output_dir / "test_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(("material_id", f"true_{args.target}", f"predicted_{args.target}"))
        writer.writerows(prediction_rows)
    return test_metrics


def main(argv: list[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
