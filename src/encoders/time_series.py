"""Time-series modality encoders.
"""

import importlib
import logging
import math
from pathlib import Path

import torch
import torch.nn as nn
from einops import rearrange, repeat
from safetensors.torch import load_file as load_safetensors
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from .base import ModalityEncoder

logger = logging.getLogger(__name__)


def _load_intern_s2_model(
    checkpoint_path: str | Path,
    package: str = "src.encoders.intern_s2_preview",
    load_pretrained: bool = True,
) -> nn.Module:
    """Construct the vendored Intern-S2 encoder, optionally loading extracted weights.

    `package` selects which vendored variant to load (e.g. the 35B
    `intern_s2_preview` package or the 397B `intern_s2_preview_397b` package),
    since each ships its own config.json / modeling code extracted from a
    different source repo.

    If `load_pretrained` is False, the model is built from the vendored config
    only (random init) and no checkpoint file is required.
    """
    configuration = importlib.import_module(f"{package}.configuration_interns2_preview")
    modeling = importlib.import_module(f"{package}.modeling_interns2_preview")
    package_dir = package.rsplit(".", 1)[-1]
    config_path = Path(__file__).with_name(package_dir) / "config.json"
    config = configuration.InternS2PreviewTimeSeriesConfig.from_json_file(config_path)
    model = modeling.InternS2PreviewTimeSeriesModel(config)

    if load_pretrained:
        checkpoint = Path(checkpoint_path).expanduser()
        if checkpoint.is_dir():
            checkpoint = checkpoint / "model.safetensors"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Intern-S2 checkpoint not found: {checkpoint}")
        model.load_state_dict(load_safetensors(str(checkpoint), device="cpu"), strict=True)

    # encoder_embed.forward_encoder hardcodes a cast to bfloat16 before its
    # transformer encoder, so that submodule's weights must be bf16 too or the
    # matmul dtypes mismatch (float32 vs bfloat16). Cast the submodule and
    # wrap encoder_embed's forward to cast the output back so the rest of the
    # (float32) model stays dtype-consistent. Done here rather than editing
    # modeling_interns2_preview.py, which is a downloaded HF file. Only the
    # 35B preview variant's encoder_embed has this bf16 cast; the 397B variant's
    # transformer_encoder is None when it uses mrqformer instead, and its
    # forward_encoder doesn't hardcode a bf16 cast, so it needs no patching.
    if getattr(model.encoder_embed, "transformer_encoder", None) is not None:
        model_dtype = next(model.parameters()).dtype
        model.encoder_embed.transformer_encoder = model.encoder_embed.transformer_encoder.to(torch.bfloat16)
        _orig_encoder_embed_forward = model.encoder_embed.forward

        def _encoder_embed_forward_cast(*args, **kwargs):
            outputs, output_lens = _orig_encoder_embed_forward(*args, **kwargs)
            return outputs.to(model_dtype), output_lens

        model.encoder_embed.forward = _encoder_embed_forward_cast

    return model


def _load_moirai_model(model_name: str, load_pretrained: bool = True, map_location: str = "cpu"):
    """Construct Moirai2Module, optionally skipping the pretrained weight download/load.

    If `load_pretrained` is False, only the HF repo's `config.json` is downloaded
    (via huggingface_hub) and the module is built from those config kwargs directly
    (random init), mirroring what PyTorchModelHubMixin.from_pretrained does for
    __init__ but skipping the weight file download/load.
    """
    from uni2ts.model.moirai2 import Moirai2Module

    if load_pretrained:
        return Moirai2Module.from_pretrained(model_name, map_location=map_location)

    import json

    from huggingface_hub import hf_hub_download

    config_path = hf_hub_download(repo_id=model_name, filename="config.json")
    with open(config_path) as f:
        config = json.load(f)
    return Moirai2Module(**config)


# =============================================================================
# TimeOmni helper classes (self-contained, no external layer dependencies)
# Adapted from: https://github.com/OpenTSLab/TimeOmni
# =============================================================================


class _TimeOmniPositionalEmbedding(nn.Module):
    """Sinusoidal positional embedding."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        return self.pe[:, : x.size(1)]


class _TimeOmniTokenEmbedding(nn.Module):
    """1-D conv token embedding."""

    def __init__(self, c_in: int, d_model: int):
        super().__init__()
        self.tokenConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
            padding_mode="circular",
            bias=False,
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="leaky_relu")

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, L, patch_len)
        return self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)


class _TimeOmniReplicationPad1d(nn.Module):
    """Right-side replication padding for 1-D signals."""

    def __init__(self, padding: tuple):
        super().__init__()
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        replicate_padding = x[:, :, -1].unsqueeze(-1).repeat(1, 1, self.padding[-1])
        return torch.cat([x, replicate_padding], dim=-1)


class _TimeOmniPatchEmbedding(nn.Module):
    """
    Patch embedding: unfolds a 1-D signal into patches and projects them.

    Input:  (B, C, T)
    Output: (B*C, num_patches, d_model), n_vars
    """

    def __init__(self, d_model: int, patch_len: int, stride: int, dropout: float):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = _TimeOmniReplicationPad1d((0, stride))
        self.value_embedding = _TimeOmniTokenEmbedding(patch_len, d_model)
        self.position_embedding = _TimeOmniPositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor):
        # x: (B, C, T)
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)  # (B, C, T + stride)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # (B, C, L, patch_len)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))  # (B*C, L, patch_len)
        x = self.value_embedding(x)  # (B*C, L, d_model)
        x = x + self.position_embedding(x)
        return self.dropout(x), n_vars


class _TimeOmniReprogrammingLayer(nn.Module):
    """
    Cross-attention reprogramming layer that maps time-series patch embeddings
    into the LLM token embedding space using vocabulary embeddings as keys/values.

    Modes:
      - attention (default): cross-attention with learned Q/K/V projections
      - mlp: simple MLP projection (ablation baseline)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_keys: int | None = None,
        d_llm: int | None = None,
        attention_dropout: float = 0.1,
        use_mlp: bool = False,
    ):
        super().__init__()
        self.use_mlp = use_mlp

        if use_mlp:
            hidden_dim = d_model * 2
            self.mlp = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.GELU(),
                nn.Dropout(attention_dropout),
                nn.Linear(hidden_dim, d_llm),
                nn.Dropout(attention_dropout),
            )
        else:
            d_keys = d_keys or (d_model // n_heads)
            self.query_projection = nn.Linear(d_model, d_keys * n_heads)
            self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
            self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
            self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
            self.n_heads = n_heads
            self.dropout = nn.Dropout(attention_dropout)

    def forward(
        self,
        target_embedding: Tensor,
        source_embedding: Tensor,
        value_embedding: Tensor,
    ) -> Tensor:
        """
        Args:
            target_embedding: (B, L, d_model) - time series patch embeddings
            source_embedding: (S, d_llm) - projected vocabulary embeddings
            value_embedding: (S, d_llm) - projected vocabulary embeddings
        Returns:
            (B, L, d_llm)
        """
        if self.use_mlp:
            return self.mlp(target_embedding)

        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self._reprogramming(target_embedding, source_embedding, value_embedding)
        out = out.reshape(B, L, -1)
        return self.out_projection(out)

    def _reprogramming(
        self,
        target_embedding: Tensor,
        source_embedding: Tensor,
        value_embedding: Tensor,
    ) -> Tensor:
        B, L, H, E = target_embedding.shape
        scale = 1.0 / math.sqrt(E)
        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        return torch.einsum("bhls,she->blhe", A, value_embedding)


# =============================================================================
# The core Time Series Encoder class for PRISM
# =============================================================================
   

class TimeSeriesEncoder(ModalityEncoder):
    """
    A single scientific timeseries input may have:
    - Multiple timeseries
    - Each timeseries may be multivariate

    Thus, this encoder 

    Warning - make sure that the patch size for variable-length
     timeseries divides the max length to avoid patches that span multiple
     instances in interleaved mode (or pad the max length).
    Otherwise, the model may learn artifacts from patch boundaries.

    Encoder type: linear
        Simple linear projection of each input feature variate to d_ts.
        Input: Time series data (B, T_ts, Num_Vars)
        Output: (B, T_ts, d_ts) projected features

    Encoder type: moirai
        Moirai (Salesforce) via Hugging Face for time series encoding.
        Input: Time Series Data (B, T_ts, Num_Vars)
        Output: Features (B, Num_Patches, d_model)

    Encoder type: timeomni
        TimeOmni-style dynamic patch embedding.
        Input: Time Series Data as either:
            - Tensor(B, T_ts, Num_Vars), or
            - list of Tensor(T_i, Num_Vars) with variable T_i.
        Output: Features (B, Num_Vars * Num_Patches, d_model), padded across
            batch when per-sample token counts differ.
    """

    # Process-wide one-shot flags: warn the first time timeomni hits a known
    # multivariate/non-interleaved limitation (see _forward_timeomni), not
    # once per sample.
    _warned_multivariate_timeomni = False
    _warned_left_pad_unmasked = False

    def __init__(
        self,
        encoder_type: str = "moirai",
        num_vars: int = 1,
        d_ts: int = 512,
        model_name: str = "Salesforce/moirai-2.0-R-small",
        max_ts_length: int = 512,
        is_interleaved: bool = False,
        # TimeOmni-specific params
        timeomni_patch_len: int | list[int] = 16,
        timeomni_stride: int | list[int] | None = None,
        timeomni_d_model: int = 512,
        timeomni_dropout: float = 0.1,
        timeomni_ts_tokens: int = 100,
        timeomni_max_patches: int = 100,
        intern_s2_sampling_rate: float = 1.0,
        load_pretrained: bool = True,
    ):
        # We don't call super with d_ts because Moirai's output dim is fixed by the model
        super().__init__(d_ts)
        self.model_name = model_name
        self.num_vars = num_vars
        self.max_ts_length = max_ts_length
        self.model = None
        self.encoder_type = encoder_type
        self.is_interleaved = is_interleaved
        
        if self.encoder_type == "linear":
            self.model = torch.nn.Linear(num_vars, d_ts)
        elif self.encoder_type == "moirai":
            try:
                logger.info(f"Loading Moirai (Time Series Encoder): {model_name}...")

                device = "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else (
                    "cuda" if torch.cuda.is_available() else "cpu"
                )

                # Load model directly using the specific class
                self.model = _load_moirai_model(model_name, load_pretrained=load_pretrained, map_location=device)


                self.hidden_dim = self.model.d_model
                self.patch_size = self.model.patch_size
                logger.info(
                    f"Successfully loaded Moirai. Hidden Dim: {self.hidden_dim}, Patch Size: {self.patch_size}"
                )

            except Exception as e:
                logger.error(f"TimeSeriesEncoder Init Error: {e}")
                logger.error("TimeSeriesEncoder will be unavailable - install uni2ts package to enable")
                self.model = None
                self.hidden_dim = d_ts
                self.patch_size = 16  # Default
        elif self.encoder_type == "intern_s2":
            import transformers

            try:
                self.model = _load_intern_s2_model(model_name, load_pretrained=load_pretrained)
            except ImportError as error:
                raise RuntimeError(
                    "Intern-S2 Preview custom code requires transformers>=5.2.0; "
                    f"found transformers=={transformers.__version__}"
                ) from error
            self.hidden_dim = int(self.model.config.out_hidden_size)
            self.output_dim = self.hidden_dim
            self.intern_s2_sampling_rate = float(intern_s2_sampling_rate)
            if self.intern_s2_sampling_rate <= 0:
                raise ValueError("intern_s2_sampling_rate must be positive")
        elif self.encoder_type == "intern_s2_397b":
            import transformers

            try:
                self.model = _load_intern_s2_model(
                    model_name,
                    package="src.encoders.intern_s2_preview_397b",
                    load_pretrained=load_pretrained,
                )
            except ImportError as error:
                raise RuntimeError(
                    "Intern-S2 Preview custom code requires transformers>=5.2.0; "
                    f"found transformers=={transformers.__version__}"
                ) from error
            self.hidden_dim = int(self.model.config.out_hidden_size)
            self.output_dim = self.hidden_dim
            # 397B's subsampling has no closed-form token count; derive it once via a dummy forward pass.
            self._intern_s2_397b_tokens_per_instance = self._probe_intern_s2_397b_tokens()
        elif self.encoder_type == "timeomni":
            # Resolve patch_len / stride lists
            if isinstance(timeomni_patch_len, int):
                self.timeomni_patch_lens = [timeomni_patch_len]
            else:
                self.timeomni_patch_lens = list(timeomni_patch_len)

            if timeomni_stride is None:
                self.timeomni_strides = list(self.timeomni_patch_lens)  # default stride = patch_len
            elif isinstance(timeomni_stride, int):
                self.timeomni_strides = [timeomni_stride] * len(self.timeomni_patch_lens)
            else:
                self.timeomni_strides = list(timeomni_stride)
                if len(self.timeomni_strides) == 1 and len(self.timeomni_patch_lens) > 1:
                    self.timeomni_strides = self.timeomni_strides * len(self.timeomni_patch_lens)

            if len(self.timeomni_strides) != len(self.timeomni_patch_lens):
                raise ValueError(
                    f"timeomni_stride (len={len(self.timeomni_strides)}) must match "
                    f"timeomni_patch_len (len={len(self.timeomni_patch_lens)}) or be a single value"
                )

            self.timeomni_d_model = timeomni_d_model

            # Build multi-scale patch embeddings
            self.timeomni_patch_embeddings = nn.ModuleDict()
            for pl, st in zip(self.timeomni_patch_lens, self.timeomni_strides, strict=False):
                self.timeomni_patch_embeddings[str(pl)] = _TimeOmniPatchEmbedding(
                    timeomni_d_model, pl, st, timeomni_dropout
                )

            # Target number of tokens per patch embedding selection
            self.timeomni_ts_tokens = int(timeomni_ts_tokens)
            self.timeomni_max_patches = int(timeomni_max_patches)

            # TimeOmni features are projected to LLM space by PRISM's
            # ModalityProjector (not by TimeOmni reprogramming in this path).
            self.hidden_dim = timeomni_d_model
            self.output_dim = timeomni_d_model

            logger.info(
                f"TimeOmni encoder initialized: patch_lens={self.timeomni_patch_lens}, "
                f"strides={self.timeomni_strides}, d_model={timeomni_d_model}"
            )
        else:
            raise ValueError(f"Unsupported encoder type: {self.encoder_type}")

    def tokens_per_instance(self):
        """Return the fixed token count one ``<ts>`` span contributes, per encoder type.

        Interleaving reserves this many slots per time-series instance, so the
        count must be independent of the runtime sample. It is derived per
        encoder type: ``linear`` reserves ``max_ts_length * num_vars`` — note
        that the ``linear`` branch of ``forward`` projects ``(B, T, V)`` to
        ``(B, T, d_ts)``, i.e. one token per timestep regardless of ``V``, so
        the reservation only matches the emitted count when ``num_vars == 1``;
        ``moirai`` reserves one token per patch per variate, matching its
        ``b (t p) v -> b (v t) p`` patchify; ``intern_s2`` follows the
        stride-based subsampling formula below; ``intern_s2_397b`` reuses the
        count measured once at construction by ``_probe_intern_s2_397b_tokens``;
        and ``timeomni`` uses its explicit ``timeomni_max_patches`` budget,
        since ``_forward_timeomni`` flattens multivariate samples to a single
        variate sequence.

        Returns:
            Number of feature tokens reserved for one instance of this modality.

        Raises:
            ValueError: If ``self.encoder_type`` is unsupported. The
                ``intern_s2`` branch also guards against a stride below 1, but
                ``__init__`` already rejects a non-positive sampling rate and
                every positive rate yields a stride in ``[2, 160]``, so that
                guard only fires if ``intern_s2_sampling_rate`` is reassigned
                to a large negative value after construction.
        """
        if self.encoder_type == "linear":
            return self.max_ts_length
        elif self.encoder_type == "moirai":
            return (self.max_ts_length // self.patch_size) * self.num_vars
        elif self.encoder_type == "intern_s2":
            sampling_rate = self.intern_s2_sampling_rate
            stride = math.floor(160 / ((1 + math.exp(-sampling_rate / 100)) ** 6))
            if stride < 1:
                raise ValueError(
                    f"intern_s2_sampling_rate={sampling_rate} produces invalid stride={stride}"
                )
            patch_count = math.ceil((self.max_ts_length - 2 * stride) / stride) + 1
            concatenated_count = patch_count // 2
            return (concatenated_count + 1) // 2
        elif self.encoder_type == "intern_s2_397b":
            return self._intern_s2_397b_tokens_per_instance
        elif self.encoder_type == "timeomni":
            # Interleaved merge uses a fixed token budget per <ts> span.
            # For timeomni this budget is explicit and independent of both
            # runtime patch size and per-sample variate count. Multivariate
            # samples are flattened to a single variate sequence in
            # _forward_timeomni, so one span maps to at most
            # `timeomni_max_patches` feature tokens.
            return self.timeomni_max_patches
        else:
            raise ValueError(f"Unsupported encoder type: {self.encoder_type}")
        
    def forward(self, inputs: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
        """
        Currently we assume that either the timeseries is a 
            multivariate series with inputs.shape[2] variates and length Time,
            or, if in interleaved mode, T = num_instances * max_ts_length_per_instance, V = 1, and B = batch_size.

        TODO: We can investigate (Batch, Instances, Time, Vars) for interleaved series, 
            padded to some max number of instances?

        Args:
            inputs: (Batch, Time, Vars) values
        Returns:
            embeddings: (Batch, Patches, Hidden)
        """
        self._last_pad_mask = None
        if self.encoder_type == "timeomni":
            return self._forward_timeomni(inputs)

        if self.encoder_type in ("intern_s2", "intern_s2_397b"):
            # Accepts Tensor(B,T,V) or list[Tensor(T_i,V_i)] with heterogeneous
            # lengths/variates, mirroring the timeomni contract.
            return self._forward_intern_s2_any(inputs)

        if not isinstance(inputs, torch.Tensor):
            raise TypeError(
                f"encoder_type='{self.encoder_type}' expects Tensor(B,T,V), "
                f"got {type(inputs).__name__}"
            )

        B, T, V = inputs.shape
        device = inputs.device

        if self.encoder_type == "linear":
            # Simple linear projection baseline
            model_dtype = next(self.model.parameters()).dtype
            if inputs.dtype != model_dtype:
                inputs = inputs.to(dtype=model_dtype)
            projected = self.model(inputs)  # (B, T, d_ts)
            return projected

        elif self.encoder_type == "moirai":

            # Handle case where model failed to load
            if self.model is None:
                logger.warning("TimeSeriesEncoder: model not loaded, returning zeros")
                num_patches = T // self.patch_size
                return torch.zeros(B, max(1, num_patches), self.hidden_dim, device=device)

            # Ensure inputs match model dtype
            model_dtype = next(self.model.parameters()).dtype
            if inputs.dtype != model_dtype:
                inputs = inputs.to(dtype=model_dtype)


            # 1. Pad inputs to be divisible by patch_size
            patch_size = self.patch_size
            pad_len = (patch_size - (T % patch_size)) % patch_size
            if pad_len > 0:
                # Pad the time dimension (dim=1)
                inputs = torch.nn.functional.pad(inputs, (0, 0, 0, pad_len))
                # New Time length
                T = T + pad_len


            # 2. Patchify and Reshape to (Batch, Seq_Len_Patches, Patch_Size)
            # Moirai flattens variates into the sequence dimension: (B, T_patches * V, Patch_Size)


            # Reshape (B, T, V) -> (B, V, T) -> (B, V, T_patches, Patch_Size)
            # Then flatten V and T_patches

            # Patrick - should we be using a sliding window for patching instead of rearrange?
            # Without overlap, the model may learn artifacts caused by artificial patch boundaries.
            # one way to address is to add a random starting offset of zeros
            inputs_patched = rearrange(inputs, "b (t p) v -> b (v t) p", p=patch_size)

            # Sequence length in tokens (patches)
            seq_len = inputs_patched.shape[1]


            # 3. Construct Masks and IDs required by Moirai
            # observed_mask: (Batch, Seq_Len, Patch) - All 1s (observed)
            observed_mask = torch.ones_like(inputs_patched, dtype=torch.bool)

            # sample_id: (Batch, Seq_Len) - Identify samples in batch
            sample_id = torch.zeros((B, seq_len), dtype=torch.long, device=device)

            
            # time_id: (Batch, Seq_Len) - Identify time steps of patches
            # For each variate, time IDs are 0, 1, ..., T_patches-1
            num_patches_per_var = T // patch_size
            time_id_single = torch.arange(num_patches_per_var, device=device)
            time_id = repeat(time_id_single, "t -> b (v t)", b=B, v=V)

            # variate_id: (Batch, Seq_Len) - Identify variate index - in interleaved mode, this is an instance id. 
            # 000... 111... etc
            variate_id_single = torch.arange(V, device=device)
            variate_id = repeat(variate_id_single, "v -> b (v t)", b=B, t=num_patches_per_var)

            # prediction_mask: (Batch, Seq_Len) - All 0s (we are encoding context, not masking for prediction)
            prediction_mask = torch.zeros((B, seq_len), dtype=torch.bool, device=device)

            try:
                # Manually run Moirai Encoder pipeline
                # Based on Moirai2Module.forward logic

                # A. Scaler
                loc, scale = self.model.scaler(
                    inputs_patched,
                    observed_mask * ~prediction_mask.unsqueeze(-1),
                    sample_id,
                    variate_id,
                )

                # HARDENED: Sanitize Scale to prevent Div-by-Zero
                scale = torch.where(scale == 0, torch.ones_like(scale), scale)
                scale = torch.clamp(scale, min=1e-6)  # Ensure no tiny values explode

                scaled_target = (inputs_patched - loc) / scale

                # HARDENED: Check for NaNs after scaling
                if torch.isnan(scaled_target).any():
                    logger.warning(
                        f"NaN detected in Time Series Scaled Target (B={B}). Replaced with 0s."
                    )
                    scaled_target = torch.nan_to_num(scaled_target, nan=0.0, posinf=0.0, neginf=0.0)

                # B. Input Projection
                input_tokens = torch.cat(
                    [scaled_target, observed_mask.to(dtype=scaled_target.dtype)], dim=-1
                )
                # Force cast to model dtype (scaler might return float32)
                input_tokens = input_tokens.to(model_dtype)
                reprs = self.model.in_proj(input_tokens)

                # C. Transformers Encoder
                # We need packed_causal_attention_mask from uni2ts
                from uni2ts.common.torch_util import packed_causal_attention_mask

                attn_mask = packed_causal_attention_mask(sample_id, time_id)

                reprs = self.model.encoder(
                    reprs,
                    attn_mask,
                    time_id=time_id,
                    var_id=variate_id,
                )

                # reprs is (Batch, Seq_Tokens, Hidden)
                return reprs

            except Exception as e:
                logger.error(f"Moirai Forward Error: {e}")
                raise e

        else:
            raise ValueError(f"Unsupported encoder type: {self.encoder_type}")

    def _probe_intern_s2_397b_tokens(self) -> int:
        """Run a dummy forward pass to determine the fixed token count for max_ts_length.

        The 397B Q-former subsampling frontend has no simple closed-form
        token-count formula (unlike the 35B stride-based encoder), so we
        measure it once at construction time instead of reimplementing the
        chunk/patch arithmetic. A full-length, single-variate probe gives the
        upper-bound token count reused by `_pad_intern_s2_tokens` for shorter,
        heterogeneous-length batches.
        """
        model_dtype = next(self.model.parameters()).dtype
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                dummy = torch.zeros(1, self.max_ts_length, self.num_vars, dtype=model_dtype)
                ts_lens = torch.full((1,), self.max_ts_length, dtype=torch.long)
                channels = torch.full((1,), self.num_vars, dtype=torch.long)
                sr = torch.full((1,), 1.0, dtype=torch.float32)
                _, pad_mask, _ = self.model(
                    time_series_signals=dummy,
                    ts_lens=ts_lens,
                    sr=sr,
                    channels=channels,
                )
            return int((~pad_mask).sum(dim=1)[0])
        finally:
            self.model.train(was_training)

    def _prepare_intern_s2_batch(
        self, inputs: torch.Tensor | list[torch.Tensor], device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build the fixed-size (B, max_ts_length, V_max) batch intern_s2 expects.

        Accepts either a pre-batched Tensor(B,T,V) (legacy, assumed fully
        real/no padding) or a list[Tensor(T_i,V_i)] with heterogeneous
        per-sample lengths/variate counts (mirrors `_forward_timeomni`'s
        input contract). Real per-sample lengths/channel counts are tracked
        in `ts_lens`/`channels` so the underlying model slices out only the
        real content (`inputs[i, :ts_lens[i], :channels[i]]`) internally.
        """
        if isinstance(inputs, torch.Tensor):
            if inputs.dim() != 3:
                raise ValueError(
                    f"encoder_type='{self.encoder_type}' expects Tensor(B,T,V) or "
                    f"list[Tensor(T,V)], got shape={tuple(inputs.shape)}"
                )
            B, T, V = inputs.shape
            ts_lens = torch.full((B,), T, dtype=torch.long, device=device)
            channels = torch.full((B,), V, dtype=torch.long, device=device)
            return inputs.to(device=device), ts_lens, channels

        if not isinstance(inputs, list):
            raise TypeError(
                f"encoder_type='{self.encoder_type}' expects Tensor(B,T,V) or "
                f"list[Tensor(T,V)], got {type(inputs).__name__}"
            )
        if not inputs:
            raise ValueError(f"{self.encoder_type} forward received an empty input list")

        items = []
        lens = []
        channels_list = []
        for idx, ts in enumerate(inputs):
            if not isinstance(ts, torch.Tensor):
                ts = torch.as_tensor(ts)
            if ts.dim() == 1:
                ts = ts.unsqueeze(-1)
            if ts.dim() != 2:
                raise ValueError(
                    f"Each {self.encoder_type} sample must be Tensor(T,V), "
                    f"got sample[{idx}] shape={tuple(ts.shape)}"
                )
            t_i, v_i = ts.shape
            if t_i > self.max_ts_length:
                ts = ts[: self.max_ts_length]
                t_i = self.max_ts_length
            items.append(ts)
            lens.append(t_i)
            channels_list.append(v_i)

        v_max = max(channels_list)
        padded = [
            torch.nn.functional.pad(ts, (0, v_max - ts.shape[1], 0, self.max_ts_length - ts.shape[0]))
            for ts in items
        ]
        batch = torch.stack(padded, dim=0).to(device=device)
        ts_lens = torch.tensor(lens, dtype=torch.long, device=device)
        channels = torch.tensor(channels_list, dtype=torch.long, device=device)
        return batch, ts_lens, channels

    def _pad_intern_s2_tokens(
        self, embeddings: torch.Tensor, pad_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Right-pad (or truncate) intern_s2 output to the fixed per-instance token budget.
        """
        if pad_mask.shape != embeddings.shape[:2]:
            raise ValueError(
                "Intern-S2 pad mask must match embedding batch/token dimensions: "
                f"embeddings={tuple(embeddings.shape)}, pad_mask={tuple(pad_mask.shape)}"
            )
        target = self.tokens_per_instance()
        n = embeddings.shape[1]
        if n < target:
            pad = embeddings.new_zeros(embeddings.shape[0], target - n, embeddings.shape[2])
            embeddings = torch.cat([embeddings, pad], dim=1)
            pad_mask = torch.cat(
                [pad_mask, torch.ones(embeddings.shape[0], target - n, dtype=torch.bool, device=pad_mask.device)],
                dim=1,
            )
        elif n > target:
            embeddings = embeddings[:, :target, :]
            pad_mask = pad_mask[:, :target]
        return embeddings, pad_mask.to(dtype=torch.bool)

    def _forward_intern_s2_any(self, inputs: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
        device = next(self.model.parameters()).device
        model_dtype = next(self.model.parameters()).dtype
        signals, ts_lens, channels = self._prepare_intern_s2_batch(inputs, device)
        signals = signals.to(dtype=model_dtype)
        B = signals.shape[0]

        if self.encoder_type == "intern_s2":
            sampling_rates = torch.full(
                (B,), self.intern_s2_sampling_rate, dtype=torch.float32, device=device
            )
            embeddings, pad_mask = self.model(
                time_series_signals=signals, ts_lens=ts_lens, sr=sampling_rates, channels=channels
            )
        else:
            # `sr` is accepted for API parity but unused internally by the 397B model.
            sampling_rates = torch.full((B,), 1.0, dtype=torch.float32, device=device)
            embeddings, pad_mask, _ = self.model(
                time_series_signals=signals, ts_lens=ts_lens, sr=sampling_rates, channels=channels
            )

        embeddings, pad_mask = self._pad_intern_s2_tokens(embeddings, pad_mask)
        self._last_pad_mask = pad_mask.detach()
        return embeddings

    def _forward_timeomni(self, inputs: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
        if isinstance(inputs, list):
            if not inputs:
                raise ValueError("timeomni forward received an empty input list")
            raw_items = inputs
        elif isinstance(inputs, torch.Tensor):
            if inputs.dim() == 3:
                raw_items = [inputs[i] for i in range(inputs.shape[0])]
            elif inputs.dim() == 2:
                raw_items = [inputs]
            else:
                raise ValueError(
                    "timeomni forward expects Tensor(B,T,V) or Tensor(T,V), "
                    f"got shape={tuple(inputs.shape)}"
                )
        else:
            raise TypeError(
                "timeomni forward expects Tensor or list[Tensor], "
                f"got {type(inputs).__name__}"
            )

        model_dtype = next(self.timeomni_patch_embeddings.parameters()).dtype
        encoded_sequences: list[torch.Tensor] = []

        for idx, ts in enumerate(raw_items):
            if not isinstance(ts, torch.Tensor):
                ts = torch.as_tensor(ts)
            if ts.dim() == 1:
                ts = ts.unsqueeze(-1)
            if ts.dim() != 2:
                raise ValueError(
                    "Each timeomni sample must be Tensor(T,V), "
                    f"got sample[{idx}] shape={tuple(ts.shape)}"
                )

            # Following the SciTS paper here.

            # Apply variate-wise RevIn before flattening
            t_raw, v_raw = ts.shape
            ts = ts.to(dtype=model_dtype).permute(1, 0).contiguous()  # (V,T)
            # RevIN-style instance normalization: normalize each variate over time.
            revin_mean = ts.mean(dim=-1, keepdim=True)          # (V, 1)
            revin_std = ts.std(dim=-1, keepdim=True).clamp(min=1e-6)  # (V, 1)
            ts = (ts - revin_mean) / revin_std  # (V, T)
            ts = ts.permute(1, 0).contiguous()  # (T, V)

            # Budget is enforced on the *flattened* length (V*T), because the
            # multivariate reshape below interleaves variates into one
            # sequence. Checking ts.shape[0] alone lets a multivariate sample
            # clear the guard and then fail _select_patch_embedding.
            if self.max_ts_length > 0:
                flat_len = t_raw * v_raw
                if flat_len > self.max_ts_length:
                    # (Jared: we can revisit if another way to handle is preferred) 
                    # Decimate along time rather than truncate. Truncation keeps
                    # full resolution over a prefix and discards the rest of the
                    # series; for a 10x-over-budget recording that deletes 90% of
                    # the signal, which is fatal for the anomaly-detection and
                    # event-localisation tasks in SciTS (the event may lie past
                    # the cut). The series is compressed to <= timeomni_max_patches
                    # tokens regardless, so resolution is already being discarded —
                    # decimation trades resolution for extent instead.
                    #
                    # Block-mean (rather than strided subsampling) acts as a crude
                    # anti-aliasing filter, so a narrow spike is attenuated rather
                    # than dropped outright.
                    factor = math.ceil(flat_len / self.max_ts_length)
                    factor = min(factor, t_raw)
                    if factor > 1:
                        keep = (t_raw // factor) * factor
                        ts = ts[:keep].reshape(t_raw // factor, factor, v_raw)
                        # Accumulate the block mean in fp32 for numerical
                        # stability (bf16 has ~3 decimal digits, and `factor`
                        # can be large), then restore `model_dtype` — the
                        # patch-embedding Conv1d requires its input dtype to
                        # match its weights. Under torch.autocast the cast
                        # back is redundant but harmless; without autocast
                        # (e.g. tools/universal_evaluator.py, which does
                        # `model.to(device, dtype=torch.bfloat16)` and never
                        # enters an autocast region) omitting it raises
                        # "Input type (torch.FloatTensor) and weight type
                        # (CPUBFloat16Type) should be the same".
                        ts = ts.float().mean(dim=1).to(model_dtype)
                        logger.warning(
                            f"timeomni sample[{idx}] flattened length "
                            f"V*T={flat_len} exceeds max_ts_length="
                            f"{self.max_ts_length}; decimated time axis by "
                            f"{factor}x to T={ts.shape[0]} "
                            f"(V*T={ts.shape[0] * v_raw})"
                        )

            # ts is (T, V) where T is possibly reduced above
            # recompute
            T, V = ts.shape
            ts = ts.permute(1, 0).contiguous()  # (V, T)
            # Deliberately serialize time-major values into one stream. E.g.,
            # a0 a1 a2 / b0 b1 b2 becomes a0 b0 a1 b1 a2 b2. Patch boundaries
            # are value-stream windows, not necessarily whole time windows.
            ts = torch.stack(torch.tensor_split(ts, V, dim=0), dim=-1).view(1, 1, V*T)

            # ts is now (1, 1, V*T) ready for patch embedding
            selected_patch_len = self._select_patch_embedding(V*T)
            patch_emb = self.timeomni_patch_embeddings[str(selected_patch_len)]
            enc_out, n_vars = patch_emb(ts)  # (V, num_patches, d_model) with B=1
            num_patches = enc_out.shape[1]
            enc_out = enc_out.view(1, n_vars, num_patches, -1)
            enc_out = enc_out.reshape(1, n_vars * num_patches, -1)  # (1, tokens, d_model)
            encoded_sequences.append(enc_out.squeeze(0))

        if self.is_interleaved:
            target_tokens = int(self.tokens_per_instance())
            if target_tokens <= 0:
                raise ValueError(
                    f"Invalid timeomni interleaved token budget: {target_tokens}"
                )
            fixed_sequences: list[torch.Tensor] = []
            for seq in encoded_sequences:
                seq_len = seq.shape[0]
                if seq_len > target_tokens:
                    raise RuntimeError(
                        "timeomni interleaved token overflow: "
                        f"encoded tokens={seq_len} exceeds merge budget={target_tokens}. "
                        "Check patch selection and max patch budget config."
                    )
                elif seq_len < target_tokens:
                    pad = torch.zeros(
                        target_tokens - seq_len,
                        seq.shape[1],
                        dtype=seq.dtype,
                        device=seq.device,
                    )
                    seq = torch.cat([seq, pad], dim=0)
                fixed_sequences.append(seq)
            return torch.stack(fixed_sequences, dim=0)

        if len(encoded_sequences) == 1:
            return encoded_sequences[0].unsqueeze(0)

        # KNOWN LIMITATION (non-interleaved batches only, latent while SciTS
        # stays interleaved-QA — flagged, not redesigned, per PR review):
        # this path only runs when `is_interleaved` is False (the interleaved
        # branch above always returns a fixed-length, right-padded batch
        # instead). Samples here can have different encoded token counts, so
        # they're left-padded with zeros below. The caller
        # (model.py's non-interleaved "prefix" forward) builds its attention
        # mask as all-ones over the full sequence length — it has no way to
        # see which positions here are real vs. left-pad — so the backbone
        # attends to the zero-padded slots as if they were real content, and
        # position ids computed from that all-ones mask are wrong for
        # left-padded sequences (they should start counting from the first
        # real token, not position 0). Do not batch non-interleaved timeomni
        # samples of unequal encoded length without also threading a
        # timeomni-aware attention mask through to the backbone call.
        if not TimeSeriesEncoder._warned_left_pad_unmasked:
            logger.warning(
                "timeomni non-interleaved batch has samples with differing "
                "encoded token counts and is left-padding with zeros; the "
                "caller's attention mask does not account for this padding. "
                "See src/encoders/time_series.py:_forward_timeomni."
            )
            TimeSeriesEncoder._warned_left_pad_unmasked = True

        # Left-pad to the max token length in the current batch (TimeOmni-style).
        reversed_sequences = [torch.flip(seq, [0]) for seq in encoded_sequences]
        padded_reversed = pad_sequence(reversed_sequences, batch_first=True, padding_value=0.0)
        return torch.flip(padded_reversed, [1])

    @staticmethod
    def _num_timeomni_patches(seq_length: int, patch_len: int, stride: int) -> int:
        # Matches _TimeOmniPatchEmbedding:
        # 1) right-pad by `stride`
        # 2) unfold(size=patch_len, step=stride)
        # L = floor((T + stride - patch_len) / stride) + 1
        return ((seq_length + stride - patch_len) // stride) + 1

    def _select_patch_embedding(self, seq_length: int) -> int:
        """Select patch size while enforcing max patch budget."""
        if seq_length <= 0:
            raise ValueError(f"seq_length must be positive, got {seq_length}")

        candidates: list[tuple[int, int]] = []
        for pl, st in zip(self.timeomni_patch_lens, self.timeomni_strides, strict=False):
            n_patches = self._num_timeomni_patches(seq_length, pl, st)
            if 1 <= n_patches <= self.timeomni_max_patches:
                candidates.append((pl, n_patches))

        if not candidates:
            raise RuntimeError(
                "No configured TimeOmni patch setting satisfies max patch budget: "
                f"T={seq_length}, max_patches={self.timeomni_max_patches}, "
                f"patch_lens={self.timeomni_patch_lens}, strides={self.timeomni_strides}"
            )

        target_patch_len = seq_length / self.timeomni_ts_tokens
        valid = [pl for pl, _ in candidates if pl <= target_patch_len]
        if valid:
            return max(valid)
        return min(pl for pl, _ in candidates)
