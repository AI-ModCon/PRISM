from types import SimpleNamespace

import torch
from src.materials import prism_train
from src.materials.data import MaterialRecord, collate_materials
from src.materials.prism_model import PRISMMaterialRegressor


class TinyDecoder(torch.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.layer = torch.nn.Linear(hidden_dim, hidden_dim)

    def forward(self, inputs_embeds, attention_mask, return_dict, use_cache):
        assert return_dict and not use_cache
        assert attention_mask.shape == inputs_embeds.shape[:2]
        return SimpleNamespace(last_hidden_state=self.layer(inputs_embeds))


class TinyBackbone(torch.nn.Module):
    def __init__(self, hidden_dim=12):
        super().__init__()
        self.embedding = torch.nn.Embedding(32, hidden_dim)
        self.model = TinyDecoder(hidden_dim)

    def get_input_embeddings(self):
        return self.embedding


class TinyPRISM(torch.nn.Module):
    def __init__(self, hidden_dim=12):
        super().__init__()
        self.backbone = TinyBackbone(hidden_dim)
        self.backbone_dim = hidden_dim
        self.encoders = torch.nn.ModuleDict()
        self.projectors = torch.nn.ModuleDict()


def make_batch():
    return collate_materials(
        [
            {
                "material_id": "mp-1",
                "text_ids": torch.tensor([2, 3, 4]),
                "graph": {
                    "atomic_numbers": torch.tensor([6, 8]),
                    "edge_index": torch.tensor([[0, 1], [1, 0]]),
                    "edge_distance": torch.tensor([1.2, 1.2]),
                },
                "target": torch.tensor(1.0),
            },
            {
                "material_id": "mp-2",
                "text_ids": torch.tensor([5, 6]),
                "graph": {
                    "atomic_numbers": torch.tensor([14, 14, 8]),
                    "edge_index": torch.tensor([[0, 1, 2], [1, 2, 0]]),
                    "edge_distance": torch.tensor([2.0, 2.1, 1.8]),
                },
                "target": torch.tensor(2.0),
            },
        ]
    )


def test_prism_material_regressor_supports_all_ablation_modes():
    model = PRISMMaterialRegressor(
        TinyPRISM(), graph_hidden_dim=16, graph_layers=2, graph_tokens=3
    )
    batch = make_batch()

    for mode in ("joint", "text", "graph"):
        prediction, loss = model(batch, mode=mode, targets=batch["target"])
        assert prediction.shape == (2,)
        assert loss is not None and loss.ndim == 0


def test_joint_training_reaches_graph_projector_and_head_with_frozen_backbone():
    prism = TinyPRISM()
    for parameter in prism.backbone.parameters():
        parameter.requires_grad = False
    model = PRISMMaterialRegressor(
        prism, graph_hidden_dim=16, graph_layers=1, graph_tokens=2
    )
    model.train()
    batch = make_batch()

    _, loss = model(batch, mode="joint", targets=batch["target"])
    loss.backward()

    assert not prism.backbone.training
    assert all(parameter.grad is None for parameter in prism.backbone.parameters())
    assert model.prism.encoders["graph"].atom_embedding.weight.grad is not None
    assert model.prism.projectors["graph"].fc1.weight.grad is not None
    assert model.regression_head[-1].weight.grad is not None


def test_run_saves_checkpoint_exactly_once_across_multiple_improving_epochs(tmp_path, monkeypatch):
    """torch.save must not be called on every val-MAE improvement.

    Saving the full trainable state (including the text backbone when
    --train-text-backbone is set) on every improving epoch means repeated
    multi-GB blocking writes to a shared filesystem. The best state should be
    held in memory and written once after training ends.
    """
    fake_records = [
        MaterialRecord(f"mp-{i}", tmp_path / f"mp-{i}.txt", tmp_path / f"mp-{i}.cif", float(i))
        for i in range(6)
    ]
    monkeypatch.setattr(prism_train, "build_manifest", lambda *a, **k: fake_records)
    monkeypatch.setattr(
        prism_train,
        "split_records",
        lambda records, val_fraction, test_fraction, seed: (
            records[:4],
            records[4:5],
            records[5:6],
        ),
    )

    class FakeTokenizer:
        def __init__(self, model_id):
            self.tokenizer = None

    monkeypatch.setattr(prism_train, "HuggingFaceTokenizer", FakeTokenizer)
    monkeypatch.setattr(
        prism_train,
        "make_loader",
        lambda records, tokenizer, args, cache_dir, shuffle: [make_batch()],
    )
    monkeypatch.setattr(prism_train, "UnifiedTransformer", lambda config: TinyPRISM())

    # 3 val calls (one per epoch, each an improvement) + 1 final test call.
    mae_sequence = iter([1.0, 0.5, 0.25, 0.25])
    monkeypatch.setattr(
        prism_train,
        "evaluate",
        lambda model, loader, device, mode, target_mean, target_std: (
            {"mae": next(mae_sequence), "rmse": 0.0, "r2": 0.0},
            [],
        ),
    )

    save_calls = []
    monkeypatch.setattr(prism_train.torch, "save", lambda obj, path: save_calls.append(path))

    args = prism_train.parse_args([])
    args.output_dir = tmp_path / "out"
    args.epochs = 3
    args.device = "cpu"
    args.mode = "joint"
    args.no_graph_cache = True

    prism_train.run(args)

    assert save_calls == [args.output_dir / "best.pt"]
