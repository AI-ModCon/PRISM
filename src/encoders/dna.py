"""DNA modality encoder built on Nucleotide Transformer or Evo2."""

import logging

import torch
import torch.nn as nn
from transformers import AutoModelForMaskedLM

from .base import ModalityEncoder

logger = logging.getLogger(__name__)


class DNAEncoder(ModalityEncoder):
    """
    DNA sequence encoder using Nucleotide Transformer (NT) or Evo2.

    Supports:
      - InstaDeepAI/nucleotide-transformer-v2-500m-multi-species (default)
      - togethercomputer/evo2_1b_base (set dna_is_evo2=True)

    Input: Tokenized DNA sequences (B, T) as input_ids
    Output: Sequence features (B, T, d_dna)
    """

    def __init__(
        self,
        d_dna: int = 1024,
        model_name: str = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",
        dna_is_evo2: bool = False,
        max_dna_length: int | None = None,
    ):
        super().__init__(d_dna)
        self.model_name = model_name
        self._max_dna_length_override = max_dna_length

        logger.info(f"Loading DNA Encoder: {model_name}...")
        self.model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)

        # Probe the actual hidden state dimension by requesting output_hidden_states.
        # AutoModelForMaskedLM.outputs[0] = LM-head logits (vocab_size ~4096), NOT hidden
        # states. Using hidden_states[-1] gives the last encoder layer's representation,
        # which is what downstream tasks want.
        with torch.no_grad():
            dummy_ids = torch.zeros(1, 6, dtype=torch.long)
            try:
                dummy_out = self.model(dummy_ids, output_hidden_states=True)
                if (
                    hasattr(dummy_out, "hidden_states")
                    and dummy_out.hidden_states is not None
                    and len(dummy_out.hidden_states) > 0
                ):
                    model_hidden_size = dummy_out.hidden_states[-1].shape[-1]
                    logger.info(f"DNA model hidden_states[-1] dim (probed): {model_hidden_size}")
                elif hasattr(dummy_out, "last_hidden_state") and dummy_out.last_hidden_state is not None:
                    model_hidden_size = dummy_out.last_hidden_state.shape[-1]
                    logger.info(f"DNA model last_hidden_state dim (probed): {model_hidden_size}")
                else:
                    # Fallback: first output is typically LM-head logits
                    model_hidden_size = dummy_out[0].shape[-1]
                    logger.warning(
                        f"Could not find hidden_states; using outputs[0] dim={model_hidden_size} "
                        "(may be LM-head logits, not hidden states)"
                    )
            except TypeError:
                # Model doesn't accept output_hidden_states kwarg
                try:
                    dummy_out = self.model(dummy_ids)
                    if hasattr(dummy_out, "last_hidden_state") and dummy_out.last_hidden_state is not None:
                        model_hidden_size = dummy_out.last_hidden_state.shape[-1]
                    else:
                        model_hidden_size = dummy_out[0].shape[-1]
                except Exception as e:
                    logger.warning(f"Dummy forward probe failed ({e}), falling back to config attributes")
                    model_hidden_size = None
                    for attr in ("hidden_size", "d_model", "embed_dim", "hidden_dim"):
                        val = getattr(self.model.config, attr, None)
                        if val is not None and isinstance(val, int):
                            model_hidden_size = val
                            break
                    if model_hidden_size is None:
                        model_hidden_size = 768
                        logger.warning(f"Could not determine hidden size, falling back to {model_hidden_size}")
            except Exception as e:
                logger.warning(f"Dummy forward probe failed ({e}), falling back to config attributes")
                model_hidden_size = None
                for attr in ("hidden_size", "d_model", "embed_dim", "hidden_dim"):
                    val = getattr(self.model.config, attr, None)
                    if val is not None and isinstance(val, int):
                        model_hidden_size = val
                        break
                if model_hidden_size is None:
                    model_hidden_size = 768
                    logger.warning(f"Could not determine hidden size, falling back to {model_hidden_size}")

        self.model_hidden_dim = model_hidden_size
        self.hidden_dim = d_dna  # Output dim expected by the main model

        if model_hidden_size != d_dna:
            logger.info(f"DNA Encoder projection: {model_hidden_size} -> {d_dna}")
            self.proj = nn.Linear(model_hidden_size, d_dna)
        else:
            self.proj = nn.Identity()

        # tokens_per_instance must match the DNA tokenizer's actual output length
        # (src/data/multimodal.py pads every DNA span to model_config.max_dna_length
        # exactly) — the interleaved merge path (_merge_text_input_ids_with_modality_embeds)
        # sizes each DNA slot from this value and hard-errors on any mismatch. Inferring
        # it from the NT model's own max_position_embeddings/etc. was wrong: that's the
        # model's *capacity*, unrelated to how long the tokenizer actually pads spans to.
        if max_dna_length is not None:
            self._tokens_per_instance = max_dna_length
        else:
            self._tokens_per_instance = None
            for attr in (
                "max_position_embeddings",
                "max_seq_len",
                "n_positions",
                "max_len",
            ):
                val = getattr(self.model.config, attr, None)
                if isinstance(val, int):
                    self._tokens_per_instance = val
                    break
            if self._tokens_per_instance is None:
                self._tokens_per_instance = getattr(self.model.config, "hidden_size", d_dna)

        logger.info(
            f"DNA Encoder loaded: {model_name} "
            f"(model_hidden={model_hidden_size}, output={d_dna}, "
            f"tokens_per_instance={self._tokens_per_instance})"
        )

    def _apply(self, fn, recurse=True):
        """Override _apply() to keep the NT backbone in float32.

        nn.Module.to() calls _apply() on child modules directly, bypassing any
        child-level to() override. This override intercepts dtype conversions at
        the _apply level and prevents the NT backbone from being cast to reduced
        precision (bfloat16/float16), which causes NaN from attention overflow.
        Device moves are applied normally to both components.
        """
        is_reduced_precision = False
        try:
            # Test fn on a CPU float32 dummy to detect reduced-precision dtype casts.
            # Device-specific fns (e.g. moving to XPU) may fail here — that's fine,
            # the except clause lets super()._apply() handle them normally.
            _dummy = torch.zeros(1, dtype=torch.float32)
            _result = fn(_dummy)
            if isinstance(_result, torch.Tensor) and _result.dtype not in (torch.float32, torch.float64):
                is_reduced_precision = True
        except Exception:
            pass

        if is_reduced_precision:
            # Cast only the projection layer; keep NT backbone in float32
            self.proj._apply(fn, recurse=recurse)
            return self

        # Device moves, float32/float64 casts, and anything else: apply normally
        return super()._apply(fn, recurse=recurse)

    def forward(self, inputs: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass.
        Args:
            inputs: Token IDs (B, T) from the DNA tokenizer, or a BatchEncoding dict.
            attention_mask: Optional attention mask (B, T).
        Returns:
            features: (B, T, d_dna)
        """
        # Handle BatchEncoding or dict-like inputs from tokenizer
        if isinstance(inputs, dict) or hasattr(inputs, 'keys'):
            kwargs = {}
            if "input_ids" in inputs:
                input_ids = inputs["input_ids"]
                if not isinstance(input_ids, torch.Tensor):
                    input_ids = torch.tensor(input_ids) if not isinstance(input_ids, list) else torch.stack(input_ids)
                if input_ids.dim() == 3 and input_ids.shape[1] == 1:
                    input_ids = input_ids.squeeze(1)
                kwargs["input_ids"] = input_ids

            if "attention_mask" in inputs:
                attention_mask = inputs["attention_mask"]
                if not isinstance(attention_mask, torch.Tensor):
                    attention_mask = torch.tensor(attention_mask) if not isinstance(attention_mask, list) else torch.stack(attention_mask)
                if attention_mask.dim() == 3 and attention_mask.shape[1] == 1:
                    attention_mask = attention_mask.squeeze(1)
                kwargs["attention_mask"] = attention_mask
        else:
            kwargs = {"input_ids": inputs}
            if attention_mask is not None:
                kwargs["attention_mask"] = attention_mask

        # Guard: clamp input_ids to the model's vocab range.  Out-of-range indices
        # (e.g. from the LLM pad_token_id leaking into DNA padding) produce a silent
        # CUDA device-side assert in ESM's embedding lookup — very hard to debug.
        if "input_ids" in kwargs:
            _ids = kwargs["input_ids"]
            _vocab = self.model.config.vocab_size
            _bad = (_ids >= _vocab) | (_ids < 0)
            if _bad.any():
                logger.warning(
                    f"DNA input_ids contain {_bad.sum().item()} out-of-range values "
                    f"(vocab_size={_vocab}, min={_ids.min().item()}, max={_ids.max().item()}). "
                    "Likely caused by the collator padding DNA sequences with the LLM pad_token_id. "
                    "Clamping to [0, vocab_size-1]."
                )
                kwargs["input_ids"] = _ids.clamp(0, _vocab - 1)

        # NT backbone is kept in float32 (see to() override); ensure integer inputs
        # are on the correct device regardless of dtype handling.
        try:
            outputs = self.model(**kwargs, output_hidden_states=True)
            if (
                hasattr(outputs, "hidden_states")
                and outputs.hidden_states is not None
                and len(outputs.hidden_states) > 0
            ):
                x = outputs.hidden_states[-1]  # Last encoder layer — actual semantic features
            elif hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
                x = outputs.last_hidden_state
            else:
                x = outputs[0]
        except TypeError:
            # Model doesn't accept output_hidden_states
            outputs = self.model(**kwargs)
            if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
                x = outputs.last_hidden_state
            else:
                x = outputs[0]

        return self.proj(x)

    def tokens_per_instance(self) -> int:
        """Returns the number of tokens the DNA modality produces per instance."""
        return int(self._tokens_per_instance)
