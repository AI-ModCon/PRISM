"""CPU contracts using deliberately small fixtures, never checkpoint acceptance.

The fixtures verify routing, target separation, masks and autograd. They do not
establish native preprocessing parity, image quality or hardware feasibility.
"""

from types import SimpleNamespace

import pytest
import torch
from src.decoders.image import ImageDecoder
from src.decoders.omnigen2_backend import (
    DEFAULT_REVISION,
    OmniGen2Backend,
    configure_omnigen2_kernels,
)
from src.decoders.types import DecoderCondition
from torch import nn

pytestmark = pytest.mark.unit


class FixtureBackend(nn.Module):
    """An interface fixture; intentionally not a production image generator."""

    conditioning_dim = 5

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(5, 3))
        self.last = None

    def generate_reference(self, native_context, **kwargs):
        self.last = (native_context, kwargs)
        return "reference-image-fixture"

    def generate_conditioned(self, embeds, attention_mask, native_context, **kwargs):
        self.last = (embeds, attention_mask, native_context, kwargs)
        return embeds @ self.weight

    def training_step(self, embeds, attention_mask, targets, native_context, **kwargs):
        self.last = (embeds, attention_mask, native_context, kwargs)
        pred = (embeds @ self.weight).sum(1)
        return pred, (pred - targets).square().mean()


def condition(hidden=None, mask=None, **kwargs):
    hidden = torch.randn(2, 4, 8) if hidden is None else hidden
    mask = torch.ones(hidden.shape[:2], dtype=torch.bool) if mask is None else mask
    return DecoderCondition(hidden, mask, **kwargs)


def test_default_backend_is_lazy_and_pinned(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("constructor attempted pretrained loading")

    monkeypatch.setattr(OmniGen2Backend, "_load", forbidden)
    dec = ImageDecoder(d_model=8)
    assert dec.backend._pipeline is None
    assert dec.backend.revision == DEFAULT_REVISION
    assert dec.backend.local_files_only is True
    assert dec.connector[1].out_features == 2048


@pytest.mark.parametrize("device", ["cpu", "xpu", "mps"])
def test_noncuda_policy_selects_existing_torch_fallbacks_without_cuda_calls(monkeypatch, device):
    import src.decoders.omnigen2_backend as backend_module

    availability = SimpleNamespace(_triton_available=True, _flash_attn_available=True)
    imports = []

    def import_module(name):
        imports.append(name)
        assert name == "omnigen2.utils.import_utils"
        return availability

    monkeypatch.setattr(backend_module.importlib, "import_module", import_module)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: pytest.fail("CUDA API invoked"))
    report = configure_omnigen2_kernels(device)
    assert imports == ["omnigen2.utils.import_utils"]
    assert report["policy"] == "upstream_torch_fallback"
    assert report["detected_packages"] == {"triton": True, "flash_attn": True}
    assert not availability._triton_available and not availability._flash_attn_available
    assert report["rms_norm"] == "torch.nn.RMSNorm"
    assert report["attention"] == "torch_sdpa"
    assert report["conditioner_attention"] == "sdpa"
    assert report["cross_kernel_parity"] == "not_run"
    # Repeated selection preserves original package-detection provenance.
    assert configure_omnigen2_kernels(device) == report


def test_cuda_policy_preserves_upstream_detection_and_rejects_cross_policy_cache(monkeypatch):
    import src.decoders.omnigen2_backend as backend_module

    availability = SimpleNamespace(_triton_available=True, _flash_attn_available=False)
    monkeypatch.setattr(backend_module.importlib, "import_module", lambda _: availability)
    report = configure_omnigen2_kernels("cuda")
    assert report["policy"] == "upstream_package_detection"
    assert availability._triton_available and not availability._flash_attn_available
    with pytest.raises(RuntimeError, match="fresh process"):
        configure_omnigen2_kernels("xpu")


def test_kernel_policy_rejects_unmanaged_early_model_import(monkeypatch):
    import src.decoders.omnigen2_backend as backend_module

    availability = SimpleNamespace(_triton_available=True, _flash_attn_available=False)
    monkeypatch.setattr(backend_module.importlib, "import_module", lambda _: availability)
    monkeypatch.setitem(
        backend_module.sys.modules, "omnigen2.models.transformers.block_lumina2", object()
    )
    with pytest.raises(RuntimeError, match="before kernel policy selection"):
        configure_omnigen2_kernels("xpu")
    assert availability._triton_available  # Fail before changing imported class semantics.


def test_preload_provenance_states_kernel_policy_without_model_import():
    backend = OmniGen2Backend()
    provenance = backend.provenance()
    assert provenance["device_type"] == "cpu"
    assert provenance["kernel_policy"]["policy"] == "upstream_torch_fallback"
    assert provenance["kernel_policy"]["selection"] == "not_imported"


def test_prism_inference_is_target_free_and_preserves_masked_order():
    backend = FixtureBackend()
    dec = ImageDecoder(8, backend=backend)
    hidden = torch.randn(2, 4, 8)
    mask = torch.tensor([[0, 1, 0, 1], [1, 0, 0, 0]])
    cond = condition(hidden, mask, native_context={"input_images": ["source"]})
    result = dec.generate_condition(cond)
    connected, actual_mask, native, _ = backend.last
    expected = dec.connector(hidden[0, [1, 3]])
    assert result.shape == (2, 2, 3)
    torch.testing.assert_close(connected[0], expected)
    assert actual_mask.tolist() == [[True, True], [True, False]]
    assert connected[1, 1].eq(0).all()
    assert native == {"input_images": ["source"]}
    assert not result.requires_grad


def test_padding_perturbation_and_nan_do_not_affect_valid_outputs():
    dec = ImageDecoder(8, backend=FixtureBackend())
    hidden = torch.randn(2, 4, 8)
    mask = torch.tensor([[1, 1, 0, 0], [0, 1, 0, 1]])
    baseline = dec.generate_condition(condition(hidden, mask))
    perturbed = hidden.clone().masked_fill(~mask.bool().unsqueeze(-1), float("nan"))
    actual = dec.generate_condition(condition(perturbed, mask))
    torch.testing.assert_close(actual, baseline, rtol=0, atol=0)


def test_connector_gradients_cross_frozen_generator_and_target_stays_separate():
    backend = FixtureBackend()
    dec = ImageDecoder(8, backend=backend).train()
    hidden = torch.randn(2, 4, 8, requires_grad=True)
    pred, loss = dec.forward_condition(condition(hidden), torch.randn(2, 3))
    loss.backward()
    assert pred.requires_grad and torch.isfinite(loss)
    assert dec.connector[1].weight.grad.norm() > 0
    assert hidden.grad.norm() > 0
    assert backend.weight.grad is None and not backend.weight.requires_grad
    assert not backend.training
    assert backend.last[2:] == ({}, {})


def test_reference_bypasses_connector_and_keeps_native_arguments():
    backend = FixtureBackend()
    dec = ImageDecoder(8, backend=backend)
    cond = condition(
        native_context={"prompt": "a red cube"}, output_spec={"mode": "reference", "height": 32}
    )
    handle = dec.connector.register_forward_hook(
        lambda *_: pytest.fail("connector used in reference mode")
    )
    assert dec.generate_condition(cond, num_inference_steps=2) == "reference-image-fixture"
    handle.remove()
    assert backend.last == ({"prompt": "a red cube"}, {"height": 32, "num_inference_steps": 2})
    with pytest.raises(ValueError, match="requires mode"):
        dec.forward_condition(cond, targets=torch.zeros(2, 3))


def test_connector_save_reload_preserves_output():
    torch.manual_seed(9)
    dec = ImageDecoder(8, backend=FixtureBackend())
    copied = ImageDecoder(8, backend=FixtureBackend())
    copied.load_state_dict(dec.state_dict())
    cond = condition()
    torch.testing.assert_close(
        dec.generate_condition(cond), copied.generate_condition(cond), rtol=0, atol=0
    )


def test_invalid_backend_configuration_and_unavailable_failure(monkeypatch):
    with pytest.raises(ValueError, match="does not match"):
        ImageDecoder(8, backend=FixtureBackend(), conditioning_dim=6)
    with pytest.raises(ValueError, match="immutable"):
        OmniGen2Backend(revision="main")
    backend = OmniGen2Backend()
    monkeypatch.setattr(backend, "preflight", lambda: {"errors": ["fixture missing dependency"]})
    with pytest.raises(RuntimeError, match="backend unavailable.*missing dependency"):
        backend.generate_reference({"prompt": "cube"})


class FixtureVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.config = SimpleNamespace(shift_factor=0.25, scaling_factor=2.0)

    @property
    def dtype(self):
        return self.scale.dtype

    def encode(self, images):
        latent = images[:, :1] * self.scale
        return SimpleNamespace(latent_dist=SimpleNamespace(sample=lambda generator=None: latent))


class FixtureFlowTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(5, 1, bias=False)
        self.config = SimpleNamespace(patch_size=1)
        self.seen = None

    @property
    def dtype(self):
        return self.proj.weight.dtype

    @property
    def device(self):
        return self.proj.weight.device

    def forward(self, hidden_states, timestep, text_hidden_states, text_attention_mask, **kwargs):
        self.seen = (hidden_states.detach(), timestep.detach(), kwargs)
        value = self.proj(text_hidden_states).squeeze(-1)
        value = (value * text_attention_mask).sum(1)
        return hidden_states + value[:, None, None, None]


def fixture_flow_backend(monkeypatch):
    """Exercise real backend objective math around fixture VAE/transformer."""
    backend = OmniGen2Backend(conditioning_dim=5)
    backend.transformer, backend.vae = FixtureFlowTransformer(), FixtureVAE()
    backend.requires_grad_(False)
    backend._pipeline = SimpleNamespace(
        transformer=backend.transformer, vae=backend.vae, vae_scale_factor=1
    )
    monkeypatch.setattr(backend, "_rotary_embeddings", lambda: torch.empty(0))
    return backend


def test_real_backend_flow_equations_vae_normalization_and_gradient_path(monkeypatch):
    backend = fixture_flow_backend(monkeypatch)
    dec = ImageDecoder(8, backend=backend)
    targets = torch.full((2, 3, 4, 4), 0.5, requires_grad=True)
    noise = torch.full((2, 1, 4, 4), -1.0)
    timesteps = torch.tensor([0.0, 1.0])
    pred, loss = dec.forward_condition(condition(), targets, noise=noise, timesteps=timesteps)
    noisy, actual_t, kwargs = backend.transformer.seen
    # clean = (0.5 - shift 0.25) * scale 2 = 0.5; velocity = clean-noise=1.5.
    torch.testing.assert_close(noisy[0], noise[0])
    torch.testing.assert_close(noisy[1], torch.full_like(noisy[1], 0.5))
    torch.testing.assert_close(actual_t, timesteps)
    torch.testing.assert_close(loss, (pred.float() - 1.5).square().mean())
    assert kwargs["ref_image_hidden_states"] == [None, None]
    loss.backward()
    assert dec.connector[1].weight.grad.norm() > 0
    assert backend.transformer.proj.weight.grad is None
    assert backend.vae.scale.grad is None and targets.grad is None


def test_real_backend_flow_seeded_sampling_reproducible(monkeypatch):
    backend = fixture_flow_backend(monkeypatch)
    dec = ImageDecoder(8, backend=backend)
    cond = condition()
    targets = torch.zeros(2, 3, 4, 4)
    first, _ = dec.forward_condition(cond, targets, generator=torch.Generator().manual_seed(42))
    second, _ = dec.forward_condition(cond, targets, generator=torch.Generator().manual_seed(42))
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_real_backend_rejects_target_leakage_bad_masks_and_unsupported_batch():
    backend = OmniGen2Backend(conditioning_dim=5)
    with pytest.raises(ValueError, match="native_context keys"):
        backend.generate_reference({"prompt": "cube", "target_image": torch.zeros(1)})
    with pytest.raises(ValueError, match="batch_size=1"):
        backend.generate_conditioned(torch.zeros(2, 3, 5), torch.ones(2, 3))
    with pytest.raises(ValueError, match="right-padded"):
        backend.generate_conditioned(torch.zeros(1, 3, 5), torch.tensor([[0, 1, 1]]))
    with pytest.raises(ValueError, match="normalized"):
        backend.training_step(torch.zeros(1, 3, 5), torch.ones(1, 3), torch.full((1, 3, 4, 4), 2.0))


@pytest.mark.parametrize("source", ["native_context", "kwargs"])
def test_conditioned_cached_negatives_move_to_generator_device_and_dtype(monkeypatch, source):
    backend = OmniGen2Backend(conditioning_dim=5)
    # Meta placement verifies transfer logic without allocating an accelerator.
    backend._pipeline = SimpleNamespace(
        transformer=SimpleNamespace(device=torch.device("meta"), dtype=torch.bfloat16)
    )
    monkeypatch.setattr(backend, "_call", lambda options, trace: options)
    negatives = {
        "negative_prompt_embeds": torch.randn(1, 3, 5, dtype=torch.float32),
        "negative_prompt_attention_mask": torch.tensor([[1, 1, 0]], dtype=torch.long),
    }
    arguments = {"native_context": negatives} if source == "native_context" else negatives
    options = backend.generate_conditioned(
        torch.zeros(1, 2, 5), torch.ones(1, 2, dtype=torch.long), **arguments
    )
    assert options["prompt_embeds"].device.type == "meta"
    assert options["negative_prompt_embeds"].device.type == "meta"
    assert options["negative_prompt_embeds"].dtype == torch.bfloat16
    assert options["negative_prompt_attention_mask"].device.type == "meta"
    assert options["negative_prompt_attention_mask"].dtype == torch.long
    assert negatives["negative_prompt_embeds"].device.type == "cpu"
    assert negatives["negative_prompt_embeds"].dtype == torch.float32


@pytest.mark.parametrize(
    "negative_embeds,negative_mask,error",
    [
        (torch.zeros(1, 3, 5), None, "supplied together"),
        (None, torch.ones(1, 3), "supplied together"),
        (torch.zeros(1, 3, 6), torch.ones(1, 3), "negative prompt conditioning"),
        (torch.zeros(1, 3, 5), torch.ones(1, 2), "negative prompt conditioning"),
        (torch.zeros(2, 3, 5), torch.ones(2, 3), "batch size"),
        (torch.zeros(1, 3, 5), torch.tensor([[1, 0, 1]]), "negative prompt conditioning"),
        (torch.zeros(1, 3, 5), [[1, 1, 1]], "must be tensors"),
    ],
)
def test_invalid_negative_conditioning_fails_before_pretrained_loading(
    monkeypatch, negative_embeds, negative_mask, error
):
    backend = OmniGen2Backend(conditioning_dim=5)
    monkeypatch.setattr(backend, "_load", lambda: pytest.fail("invalid condition loaded weights"))
    with pytest.raises(ValueError, match=error):
        backend.generate_conditioned(
            torch.zeros(1, 2, 5),
            torch.ones(1, 2),
            native_context={
                "negative_prompt_embeds": negative_embeds,
                "negative_prompt_attention_mask": negative_mask,
            },
        )


def test_reference_cached_negative_arguments_remain_native_pass_through(monkeypatch):
    backend = OmniGen2Backend(conditioning_dim=5)
    monkeypatch.setattr(backend, "_call", lambda options, trace: options)
    negatives = torch.randn(1, 3, 5)
    mask = torch.ones(1, 3, dtype=torch.long)
    options = backend.generate_reference(
        {
            "prompt": "cube",
            "negative_prompt_embeds": negatives,
            "negative_prompt_attention_mask": mask,
        }
    )
    assert options["negative_prompt_embeds"] is negatives
    assert options["negative_prompt_attention_mask"] is mask


def test_checkpoint_manifest_requires_loaded_backend():
    with pytest.raises(RuntimeError, match="before hashing"):
        OmniGen2Backend().checkpoint_manifest()


def test_checkpoint_manifest_digest_ignores_cache_location(tmp_path):
    manifests = []
    for cache in ("first_host", "second_host"):
        root = tmp_path / cache
        for component in ("transformer", "vae", "mllm", "processor", "scheduler"):
            (root / component).mkdir(parents=True)
            (root / component / "fixture.json").write_text('{"fixture": true}')
        backend = OmniGen2Backend(model_id=str(root))
        backend._pipeline = object()  # Manifest test uses small local fixture files only.
        manifests.append(backend.checkpoint_manifest())
    assert manifests[0]["model_id"] != manifests[1]["model_id"]
    assert manifests[0]["manifest_sha256"] == manifests[1]["manifest_sha256"]


def test_collator_reference_metadata_is_validated_and_flattened_only_for_single_batch():
    source_a, source_b = object(), object()
    native = {
        "reference_images": [[source_a, source_b]],
        "source_ids": [["a", "b"]],
        "source_mask": torch.tensor([[1, 1, 0]], dtype=torch.bool),
    }
    assert OmniGen2Backend._native(native, batch_size=1) == {"input_images": [source_a, source_b]}
    assert "reference_images" in native  # Caller metadata is not mutated.
    multi = {"reference_images": [[source_a], []], "source_mask": torch.tensor([[1], [0]])}
    assert OmniGen2Backend._native(multi, batch_size=2) == {"input_images": [[source_a], []]}
    with pytest.raises(ValueError, match="valid reference-image order"):
        OmniGen2Backend._native({**native, "source_mask": torch.tensor([[1, 0, 1]])}, batch_size=1)
    with pytest.raises(ValueError, match="source_ids"):
        OmniGen2Backend._native({**native, "source_ids": [["b"]]}, batch_size=1)


@pytest.mark.parametrize(
    "native,batch_size",
    [
        ({"reference_images": [[object()] * 6]}, 1),
        ({"reference_images": [[], [object()] * 6]}, 2),
        ({"input_images": [object()] * 6}, 1),
        ({"input_images": [None, [object()] * 6]}, 2),
    ],
)
def test_native_reference_count_respects_pretrained_five_entry_index_embedding(native, batch_size):
    with pytest.raises(ValueError, match="at most 5 reference images per example"):
        OmniGen2Backend._native(native, batch_size=batch_size)


def test_five_references_and_mixed_empty_rows_are_supported():
    images = [object() for _ in range(5)]
    assert OmniGen2Backend._native({"input_images": images}, batch_size=1)["input_images"] == images
    assert OmniGen2Backend._native({"input_images": [None, images]}, batch_size=2)[
        "input_images"
    ] == [None, images]
    assert OmniGen2Backend._native({"reference_images": [[]]}, batch_size=1)["input_images"] == []


class FixtureTracePipeline:
    """Small native-call fixture; no approximate diffusion implementation."""

    def __init__(self):
        self.mllm = nn.Identity()
        self.vae = SimpleNamespace(decode=lambda latent, **kwargs: (latent * 2,))
        self.scheduler = SimpleNamespace(
            step=lambda pred, t, latent, **kwargs: (latent + pred,),
            timesteps=torch.tensor([0.0]),
        )
        self.options = None

    def encode_prompt(self, prompt, **kwargs):
        values = self.mllm(torch.ones(1, 2, 5))
        return values, torch.ones(1, 2), None, None

    def prepare_latents(self, latents):
        return latents

    def prepare_image(self):
        return [[torch.ones(1, 2, 2)]]

    def predict(self, **kwargs):
        return kwargs["latents"] + 1

    def __call__(self, **kwargs):
        self.options = kwargs
        embeds, mask, _, _ = self.encode_prompt(kwargs["prompt"])
        self.prepare_image()
        latent = self.prepare_latents(kwargs["latents"])
        pred = self.predict(latents=latent, prompt_embeds=embeds, prompt_attention_mask=mask)
        latent = self.scheduler.step(pred, 0.0, latent)[0]
        pixels = self.vae.decode(latent)[0]
        return SimpleNamespace(images=pixels)


def test_reference_trace_captures_actual_boundaries_and_restores_native_methods():
    backend = OmniGen2Backend(conditioning_dim=5)
    pipe = FixtureTracePipeline()
    backend._pipeline = pipe
    original = pipe.encode_prompt
    initial = torch.zeros(1, 1, 2, 2)
    out = backend.generate_reference({"prompt": "cube"}, latents=initial, trace=True)
    assert pipe.options["latents"] is initial
    assert pipe.encode_prompt == original
    assert not pipe.mllm._forward_pre_hooks
    for name in (
        "condition.positive",
        "mask.positive",
        "reference.0.0",
        "token_ids.positive",
        "prediction.step0",
        "prediction.branch0",
        "latents.step0",
        "latents.final",
        "latents.vae_input",
        "pixels.vae_output",
        "pixels",
        "schedule.timesteps",
    ):
        assert name in backend.last_trace
    torch.testing.assert_close(backend.last_trace["latents.initial"], initial)
    torch.testing.assert_close(backend.last_trace["latents.final"], torch.ones_like(initial))
    torch.testing.assert_close(backend.last_trace["pixels"], out)
