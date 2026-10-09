"""CPU-only trainability contracts; no released weights or accelerator acceptance."""

from types import SimpleNamespace

import pytest
import torch
from src.decoders.image import ImageDecoder
from src.decoders.omnigen2_backend import OmniGen2Backend
from src.decoders.types import DecoderCondition
from torch import nn


class _Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(5, 1)
        self.dropout = nn.Dropout(0.0)
        self.gradient_checkpointing = False
        self.checkpoint_calls = []
        self.config = SimpleNamespace(text_feat_dim=5, patch_size=1)

    @property
    def dtype(self):
        return self.proj.weight.dtype

    @property
    def device(self):
        return self.proj.weight.device

    def enable_gradient_checkpointing(self):
        self.checkpoint_calls.append("enable")
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        self.checkpoint_calls.append("disable")
        self.gradient_checkpointing = False

    def forward(self, hidden_states, timestep, text_hidden_states, text_attention_mask, **kwargs):
        def project(values):
            values = self.dropout(self.proj(values)).squeeze(-1)
            return (values * text_attention_mask).sum(1)

        if self.training and self.gradient_checkpointing:
            from torch.utils.checkpoint import checkpoint

            context = checkpoint(project, text_hidden_states, use_reentrant=False)
        else:
            context = project(text_hidden_states)
        return hidden_states + context[:, None, None, None]


class _VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.config = SimpleNamespace(shift_factor=None, scaling_factor=None)

    @property
    def dtype(self):
        return self.scale.dtype

    def encode(self, target):
        latent = target[:, :1] * self.scale
        return SimpleNamespace(latent_dist=SimpleNamespace(sample=lambda generator=None: latent))


class _Pipeline:
    def __init__(self, transformer, vae, mllm, **kwargs):
        self.transformer, self.vae, self.mllm = transformer, vae, mllm
        self.vae_scale_factor = 1
        self.raise_sampling = False

    def to(self, **kwargs):
        return self

    def __call__(self, **kwargs):
        assert not torch.is_grad_enabled()
        assert all(
            not module.training
            for component in (self.transformer, self.vae, self.mllm)
            for module in component.modules()
        )
        if self.raise_sampling:
            raise RuntimeError("fixture sampling failure")
        return SimpleNamespace(images=torch.zeros(1, 3, 4, 4))


def _backend(monkeypatch):
    backend = OmniGen2Backend(conditioning_dim=5)
    backend.transformer, backend.vae, backend.mllm = _Transformer(), _VAE(), nn.Linear(2, 2)
    backend._pipeline = _Pipeline(backend.transformer, backend.vae, backend.mllm)
    backend.configure_training()
    monkeypatch.setattr(backend, "_rotary_embeddings", lambda: torch.empty(0))
    return backend


def _condition():
    return DecoderCondition(torch.randn(2, 3, 8), torch.ones(2, 3, dtype=torch.bool))


@pytest.mark.parametrize("gradient_checkpointing", [False, True])
def test_explicit_diffusion_training_backpropagates_to_all_transformer_weights_only(
    monkeypatch, gradient_checkpointing
):
    backend = _backend(monkeypatch)
    decoder = ImageDecoder(8, backend=backend)
    decoder.requires_grad_(False)
    decoder.configure_training(train_diffusion=True, gradient_checkpointing=gradient_checkpointing)
    assert decoder.training and backend.training and backend.transformer.training
    assert not backend.vae.training and not backend.mllm.training
    _, loss = decoder.forward_condition(
        _condition(),
        torch.full((2, 3, 4, 4), 0.5),
        noise=torch.zeros(2, 1, 4, 4),
        timesteps=torch.tensor([0.25, 0.75]),
    )
    loss.backward()
    for module in (decoder.connector, backend.transformer):
        assert all(
            p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0
            for p in module.parameters()
        )
    for module in (backend.vae, backend.mllm):
        assert all(not p.requires_grad and p.grad is None for p in module.parameters())
    assert backend.transformer.gradient_checkpointing is gradient_checkpointing


def test_default_connector_only_and_explicit_return_to_default(monkeypatch):
    backend = _backend(monkeypatch)
    decoder = ImageDecoder(8, backend=backend).train()
    assert not backend.training and not backend.transformer.training
    assert all(not p.requires_grad for p in backend.parameters())
    decoder.configure_training(train_diffusion=True, gradient_checkpointing=True)
    decoder.configure_training()
    assert not backend.train_diffusion and not backend.transformer.gradient_checkpointing
    assert backend.transformer.checkpoint_calls == ["enable", "disable"]
    assert all(not p.requires_grad for p in backend.parameters())
    assert all(p.requires_grad for p in decoder.connector.parameters())
    assert not backend.training and not backend.transformer.training


def test_parent_modes_preserve_opt_in_and_keep_native_components_eval(monkeypatch):
    backend = _backend(monkeypatch)
    decoder = ImageDecoder(8, backend=backend).configure_training(train_diffusion=True)
    parent = nn.Sequential(decoder)
    for mode in (False, True, False, True):
        parent.train(mode)
        assert backend.transformer.training is mode
        assert all(p.requires_grad for p in backend.transformer.parameters())
        assert not backend.vae.training and not backend.mllm.training
        assert all(
            not p.requires_grad
            for component in (backend.vae, backend.mllm)
            for p in component.parameters()
        )
        backend.ensure_loaded()
        assert backend.transformer.training is mode
        assert all(p.requires_grad for p in backend.transformer.parameters())


def test_preconfigured_backend_is_not_refrozen_by_decoder_constructor(monkeypatch):
    backend = _backend(monkeypatch).configure_training(train_diffusion=True)
    decoder = ImageDecoder(8, backend=backend)
    assert all(p.requires_grad for p in decoder.backend.transformer.parameters())
    assert decoder.backend.transformer.training


@pytest.mark.parametrize("raises", [False, True])
@pytest.mark.parametrize("reference", [False, True])
def test_sampling_temporarily_uses_eval_and_restores_all_modes_even_on_failure(
    monkeypatch, raises, reference
):
    backend = _backend(monkeypatch).configure_training(train_diffusion=True)
    backend.transformer.dropout.eval()  # Deliberately mixed submodule mode.
    modes = [(module, module.training) for module in backend.modules()]
    backend._pipeline.raise_sampling = raises

    def sample():
        if reference:
            return backend.generate_reference({"prompt": "fixture"})
        return backend.generate_conditioned(torch.zeros(1, 2, 5), torch.ones(1, 2))

    if raises:
        with pytest.raises(RuntimeError, match="fixture sampling failure"):
            sample()
    else:
        assert sample().shape == (1, 3, 4, 4)
    assert all(module.training is mode for module, mode in modes)
    assert all(p.requires_grad for p in backend.transformer.parameters())


def test_lazy_configuration_never_loads_or_imports_checkpoint(monkeypatch):
    backend = OmniGen2Backend(conditioning_dim=5)
    monkeypatch.setattr(backend, "_load", lambda: pytest.fail("configure must stay lazy"))
    decoder = ImageDecoder(8, backend=backend)
    decoder.configure_training(train_diffusion=True, gradient_checkpointing=True)
    assert backend._pipeline is None and backend.train_diffusion
    assert backend.provenance()["training_scope"] == "connector_and_full_diffusion"


def test_invalid_and_unsupported_requests_fail_without_silently_changing_scope(monkeypatch):
    backend = _backend(monkeypatch)
    with pytest.raises(ValueError, match="requires train_diffusion"):
        backend.configure_training(gradient_checkpointing=True)
    with pytest.raises(TypeError, match="must be bool"):
        backend.configure_training(train_diffusion="true")
    monkeypatch.setattr(backend.transformer, "enable_gradient_checkpointing", None)
    with pytest.raises(RuntimeError, match="does not support"):
        backend.configure_training(train_diffusion=True, gradient_checkpointing=True)
    assert not backend.train_diffusion
    assert all(not p.requires_grad for p in backend.transformer.parameters())


def _fake_pretrained_imports(monkeypatch, backend):
    import src.decoders.omnigen2_backend as adapter

    def factory(value):
        return SimpleNamespace(from_pretrained=lambda *a, **kw: value)

    transformer, vae, conditioner = _Transformer(), _VAE(), nn.Linear(2, 2)
    monkeypatch.setattr(backend, "preflight", lambda: {"errors": []})
    monkeypatch.setattr(
        adapter, "configure_omnigen2_kernels", lambda _: {"policy": "upstream_torch_fallback"}
    )
    modules = {
        "diffusers": SimpleNamespace(AutoencoderKL=factory(vae)),
        "omnigen2.models.transformers.transformer_omnigen2": SimpleNamespace(
            OmniGen2Transformer2DModel=factory(transformer)
        ),
        "omnigen2.pipelines.omnigen2.pipeline_omnigen2": SimpleNamespace(
            OmniGen2Pipeline=_Pipeline
        ),
        "omnigen2.schedulers.scheduling_flow_match_euler_discrete": SimpleNamespace(
            FlowMatchEulerDiscreteScheduler=factory(object())
        ),
        "transformers": SimpleNamespace(
            Qwen2_5_VLForConditionalGeneration=factory(conditioner),
            Qwen2_5_VLProcessor=factory(object()),
        ),
    }
    for name, module in modules.items():
        monkeypatch.setitem(adapter.sys.modules, name, module)
    return transformer


@pytest.mark.parametrize("opt_in", [False, True])
def test_actual_lazy_load_applies_requested_policy_without_real_weights(monkeypatch, opt_in):
    backend = OmniGen2Backend(conditioning_dim=5)
    decoder = ImageDecoder(8, backend=backend)
    if opt_in:
        decoder.configure_training(train_diffusion=True, gradient_checkpointing=True)
    _fake_pretrained_imports(monkeypatch, backend)
    backend.ensure_loaded()
    assert backend.transformer.training is opt_in
    assert all(p.requires_grad is opt_in for p in backend.transformer.parameters())
    assert backend.transformer.gradient_checkpointing is opt_in
    assert not backend.vae.training and not backend.mllm.training
    assert all(
        not p.requires_grad
        for component in (backend.vae, backend.mllm)
        for p in component.parameters()
    )


def test_unsupported_checkpointing_during_load_does_not_publish_loaded_pipeline(monkeypatch):
    backend = OmniGen2Backend(conditioning_dim=5)
    backend.configure_training(train_diffusion=True, gradient_checkpointing=True)
    transformer = _fake_pretrained_imports(monkeypatch, backend)
    monkeypatch.setattr(transformer, "enable_gradient_checkpointing", None)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="does not support gradient checkpointing"):
            backend.ensure_loaded()
        assert backend._pipeline is None
