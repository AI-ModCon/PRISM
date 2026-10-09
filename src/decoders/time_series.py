"""Time-series forecast decoder.

The inverse of ``TimeSeriesEncoder`` (``src/encoders/time_series.py``): the
encoder turns a ``(B, T, V)`` series into backbone tokens; this decoder reads
backbone hidden states and emits a multi-horizon forecast ``(B, H, V)``.

Design:
- **Direct multi-horizon**, not autoregressive — one forward predicts all H
  steps (Chronos-Bolt style). This keeps the decoder a single-forward head
  and preserves the AR serving path for text.
- **Quantile head** trained with the pinball (quantile) loss, giving a
  probabilistic forecast. The median quantile is the point forecast used by
  ``generate``.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn

from ..connectors import BackboneFeatures, DecoderContext, build_bridge, build_readout
from .base import OutputDecoder
from .types import DecoderCondition

DEFAULT_QUANTILES: tuple[float, ...] = (0.1, 0.5, 0.9)


def _generator_options(generator, *, horizon, num_vars, quantiles):
    legacy = {"horizon": horizon, "num_vars": num_vars, "quantiles": quantiles}
    if generator is None:
        options = {key: value for key, value in legacy.items() if value is not None}
    else:
        if not isinstance(generator, Mapping):
            raise TypeError("generator configuration must be a mapping")
        extra = set(generator) - {"type", *legacy}
        if extra:
            raise ValueError(f"Unknown generator configuration fields: {sorted(extra)}")
        if generator.get("type", "linear_quantile") != "linear_quantile":
            raise ValueError("Unsupported generator type; only 'linear_quantile' is implemented")
        if any(value is not None for value in legacy.values()):
            raise ValueError("Cannot mix nested generator configuration with flat generator options")
        options = {key: value for key, value in generator.items() if key != "type"}
    options.setdefault("num_vars", 1)
    options.setdefault("quantiles", DEFAULT_QUANTILES)
    horizon = options.get("horizon")
    num_vars = options["num_vars"]
    if type(horizon) is not int or horizon < 1:
        raise ValueError(f"horizon must be a positive integer, got {horizon}")
    if type(num_vars) is not int or num_vars < 1:
        raise ValueError(f"num_vars must be a positive integer, got {num_vars}")
    return options


class TimeSeriesDecoder(OutputDecoder):
    """Direct multi-horizon quantile forecaster.

    Reads a pooled backbone hidden state and projects it to a forecast of
    shape ``(B, H, V, Q)`` (horizon × variates × quantiles). Trained with the
    pinball loss; ``generate`` returns the median-quantile point forecast
    ``(B, H, V)``.
    """

    output_kind = "tensor"
    loss_kind = "pinball"
    response_encoding = "tensor_b64"
    _q: torch.Tensor  # registered buffer; annotate for `register_buffer` typing

    def __init__(
        self,
        d_model: int,
        horizon: int | None = None,
        num_vars: int | None = None,
        quantiles: tuple[float, ...] | list[float] | None = None,
        pool: str | None = None,
        readout: Mapping | None = None,
        bridge: Mapping | None = None,
        generator: Mapping | None = None,
    ):
        super().__init__()
        options = _generator_options(
            generator, horizon=horizon, num_vars=num_vars, quantiles=quantiles
        )
        quantiles = tuple(float(q) for q in options["quantiles"])
        if not quantiles:
            raise ValueError("quantiles must be non-empty")
        if any(not (0.0 < q < 1.0) for q in quantiles):
            raise ValueError(f"quantiles must be in (0, 1), got {quantiles}")
        if readout is not None and pool is not None:
            raise ValueError("Cannot mix readout configuration with flat pool option")
        if readout is None:
            readout = {"type": "pool", "layers": "final", "pool": "last" if pool is None else pool}
        if not isinstance(readout, Mapping):
            raise TypeError("readout configuration must be a mapping")
        readout = {"type": "pool", "layers": "final", "pool": "last", **readout}
        if readout["type"] != "pool":
            raise ValueError("Time-series decoder requires a pooled readout")
        if bridge is not None and not isinstance(bridge, Mapping):
            raise TypeError("bridge configuration must be a mapping")
        bridge = {"type": "identity", **(bridge or {})}
        if bridge["type"] != "identity":
            raise ValueError("Time-series decoder currently requires an identity bridge")
        # Parameter-free stages preserve initialization RNG and old head.* keys.
        self.readout = build_readout(readout, d_model=d_model)
        self.connector = build_bridge(bridge, input_dim=d_model, output_dim=d_model)

        self.d_model = d_model
        # _generator_options validates both as positive ints before returning.
        self.horizon: int = options["horizon"]
        self.num_vars: int = options["num_vars"]
        self.quantiles = quantiles
        self.num_quantiles = len(quantiles)
        self.pool = readout["pool"]
        # Index of the median quantile (closest to 0.5) — the point forecast.
        self._median_idx = min(
            range(self.num_quantiles),
            key=lambda i: abs(quantiles[i] - 0.5),
        )
        # Quantile levels as a buffer so loss math moves with the module.
        self.register_buffer(
            "_q", torch.tensor(quantiles, dtype=torch.float32), persistent=False
        )

        self.head = nn.Linear(d_model, self.horizon * self.num_vars * self.num_quantiles)

    # -- helpers -----------------------------------------------------------

    def conditioning_contract(self) -> dict:
        """Describe this decoder's readout, bridge, layout, and generator.

        Returns:
            A JSON-serializable record of the implemented conditioning route:
            the pooled final-layer readout (with its ``pool`` mode), the
            identity bridge and its widths, the ``single_token`` layout, and
            the ``linear_quantile`` generator with its horizon, variate count,
            and quantile levels.
        """
        return {
            "schema_version": 1,
            "readout": {"type": "pool", "layers": "final", "pool": self.pool},
            "bridge": {
                "type": "identity", "input_dim": self.d_model, "output_dim": self.d_model,
            },
            "layout": {"type": "single_token"},
            "generator": {
                "type": "linear_quantile", "conditioning_dim": self.d_model,
                "horizon": self.horizon, "num_vars": self.num_vars,
                "quantiles": list(self.quantiles),
            },
        }

    def prepare_condition(self, condition: DecoderCondition) -> DecoderContext:
        """Pool the conditioning states into the single token the head reads.

        Args:
            condition: backbone states plus their attention mask, spans, and
                provenance.

        Returns:
            A ``DecoderContext`` whose ``tokens`` are ``(B, 1, d_model)`` and
            whose ``provenance`` carries ``conditioning_contract()`` under the
            ``"conditioning_contract"`` key.
        """
        features = BackboneFeatures(
            hidden_states=condition.hidden_states,
            attention_mask=condition.attention_mask,
            modality_spans=condition.modality_spans,
            provenance=condition.provenance,
        )
        context = self.connector.connect(self.readout(features))
        context.provenance["conditioning_contract"] = self.conditioning_contract()
        return context

    def connect(self, condition: DecoderCondition) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the prepared conditioning as a plain tensor pair.

        Args:
            condition: backbone states plus their attention mask.

        Returns:
            ``(tokens, attention_mask)`` from ``prepare_condition`` — the
            pooled ``(B, 1, d_model)`` tokens and their ``(B, 1)`` mask.
        """
        context = self.prepare_condition(condition)
        return context.tokens, context.attention_mask

    @staticmethod
    def _tensor_condition(hidden_states, attention_mask=None):
        if hidden_states.ndim == 2:
            # Historically an already-pooled vector ignores the sequence mask.
            hidden_states = hidden_states.unsqueeze(1)
            attention_mask = None
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must be (B, T, d_model) or (B, d_model)")
        if attention_mask is None:
            attention_mask = torch.ones(
                hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device
            )
        else:
            # Legacy tensor calls accepted host masks for accelerator states.
            attention_mask = attention_mask.to(device=hidden_states.device)
        return DecoderCondition(hidden_states, attention_mask)

    def _predict_condition(self, condition):
        pooled = self.prepare_condition(condition).tokens[:, 0]
        pooled = pooled.to(dtype=self.head.weight.dtype)
        return self.head(pooled).view(
            pooled.shape[0], self.horizon, self.num_vars, self.num_quantiles
        )

    def predict(self, hidden_states: torch.Tensor, attention_mask=None) -> torch.Tensor:
        """Forecast quantiles ``(B, H, V, Q)`` from hidden states."""
        if attention_mask is None and hidden_states.ndim == 3 and self.pool == "mean":
            # Preserve the legacy no-mask reduction (including half-precision
            # accumulation). Replacing mean with sum/count can overflow FP16.
            # The tensor API has no source-position metadata; typed conditions
            # retain their explicit mask and always pool through the readout.
            hidden_states = hidden_states.mean(dim=1)
        return self._predict_condition(self._tensor_condition(hidden_states, attention_mask))

    def pinball_loss(
        self, pred_quantiles: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Mean pinball (quantile) loss.

        Args:
            pred_quantiles: ``(B, H, V, Q)``.
            target: ``(B, H, V)`` ground-truth future values.
        """
        if target.shape != pred_quantiles.shape[:-1]:
            raise RuntimeError(
                f"target shape {tuple(target.shape)} != forecast shape "
                f"{tuple(pred_quantiles.shape[:-1])} (B, H, V)"
            )
        target = target.to(dtype=pred_quantiles.dtype)
        q = self._q.to(device=pred_quantiles.device, dtype=pred_quantiles.dtype)
        # error: (B, H, V, Q)
        error = target.unsqueeze(-1) - pred_quantiles
        loss = torch.maximum(q * error, (q - 1.0) * error)
        return loss.mean()

    # -- OutputDecoder API -------------------------------------------------

    def forward_condition(self, condition: DecoderCondition, targets=None, **kwargs):
        """Forecast quantiles from a typed condition and optionally score them.

        Always pools through the readout using the condition's explicit
        attention mask. The tensor ``forward`` path instead falls back to a
        legacy unmasked mean on its one branch that has no mask to honor
        (3-D states, no ``attention_mask``, ``pool="mean"``).

        Args:
            condition: backbone states plus their attention mask.
            targets: ground-truth future values ``(B, H, V)``, or ``None`` for
                inference. Default: ``None``.
            **kwargs: accepted and ignored.

        Returns:
            ``(pred_quantiles, loss)`` where ``pred_quantiles`` is
            ``(B, H, V, Q)`` and ``loss`` is the mean pinball loss, or ``None``
            when ``targets`` is ``None``.
        """
        prediction = self._predict_condition(condition)
        loss = None if targets is None else self.pinball_loss(prediction, targets)
        return prediction, loss

    @torch.no_grad()
    def generate_condition(self, condition: DecoderCondition, **kwargs) -> torch.Tensor:
        """Point-forecast from a typed condition.

        Args:
            condition: backbone states plus their attention mask.
            **kwargs: accepted and ignored.

        Returns:
            The median-quantile point forecast ``(B, H, V)``.
        """
        return self._predict_condition(condition)[..., self._median_idx]

    def forward(
        self,
        hidden_states: torch.Tensor,
        targets: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Predict quantiles and (optionally) compute the pinball loss.

        Returns ``(pred_quantiles (B,H,V,Q), loss)``.
        """
        pred_quantiles = self.predict(hidden_states, kwargs.get("attention_mask"))
        if targets is None:
            return pred_quantiles, None
        loss = self.pinball_loss(pred_quantiles, targets)
        return pred_quantiles, loss

    @torch.no_grad()
    def generate(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        """Point forecast ``(B, H, V)`` — the median quantile."""
        pred_quantiles = self.predict(hidden_states, kwargs.get("attention_mask"))
        return pred_quantiles[..., self._median_idx]
