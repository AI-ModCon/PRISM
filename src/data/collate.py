import logging

import torch
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)

# Track sequence length statistics for debugging
_seq_len_stats = {"count": 0, "total_len": 0, "max_len": 0, "min_len": float("inf")}


class MultimodalCollator:
    """
    Collate function for multimodal batches of dicts.

    Expects a list of samples where each sample is a dict of modality keys
    (for example: "image", "text", "audio", "graph"). The collator builds a
    batch dict by key, applying dynamic padding for variable-length tensors
    and stacking fixed-shape tensors. Text sequences are always padded with
    the tokenizer pad id (or ``padding_value`` fallback). Nested dicts are
    collated recursively to support structured modalities.

    Behavior by value type:
    - ``torch.Tensor``: stack when shapes match; otherwise pad along dim 0 for
      1D or general 2D+ tensors. Special-case edge indices with shape (2, E)
      by returning the list unchanged to avoid incorrect padding.
    - ``dict``: recursively collated with the same rules.
    - other types (lists/strings/metadata): returned as a list of items.

    Args:
        tokenizer: Tokenizer providing ``pad_token_id`` for text padding.
        padding_value: Fallback padding value used when tokenizer has no pad id.
        max_seq_length: Optional hard cap on text sequence length. When set,
            1D text tensors longer than this are truncated (tail-dropped) BEFORE
            padding, mirroring ``BucketedCollator``. Leave ``None`` on the
            interleaved-QA path: tail-truncating an interleaved sample would
            desync the ``<ts>``/``<ts/>`` token balance and the prompt/target
            metadata the model relies on (that path is capped at the variate
            level in ``StreamingMultimodalDataset._process_ts_qa`` instead).
        passthrough_time_series: When True, the ``time_series`` key is kept as
            a list of per-sample tensors (no pad/stack). Used by dynamic-length
            time-series encoders (timeomni, intern_s2, intern_s2_397b) whose
            batching/padding happens in the encoder forward pass.
    """

    def __init__(
        self,
        tokenizer,
        padding_value=0,
        max_seq_length: int | None = None,
        passthrough_time_series: bool = False,
    ):
        self.tokenizer = tokenizer
        self.padding_value = padding_value
        if tokenizer.pad_token_id is not None:
            self.padding_value = tokenizer.pad_token_id
        self.max_seq_length = max_seq_length
        self.passthrough_time_series = passthrough_time_series

    def __call__(self, batch):
        global _seq_len_stats
        # Batch is a list of dicts.
        # batch[0] = {"image": tensor, "text": tensor, ...}

        # separate keys
        keys = batch[0].keys()
        collated = {}

        for k in keys:
            # Grab every item for this modality across the batch
            # items is a list of tensors/dicts/lists for modality k: [sample1[k], sample2[k], ...]
            items = [d[k] for d in batch]

            # In interleaved mode, sample1[k] may itself be a list of N timeseries instances of different lengths.

            # DEBUG
            if isinstance(items[0], torch.Tensor) and items[0].numel() > 0:
                logger.debug(
                    f"COLLATOR {k}: Type={items[0].dtype}, Shape={items[0].shape}, Range=[{items[0].float().min():.3f}, {items[0].float().max():.3f}]"
                )

            if k == "text":
                # Handle both tokenized tensors AND raw strings
                if isinstance(items[0], torch.Tensor):
                    # Hard sequence-length cap: truncate BEFORE padding so an
                    # outlier-length sample can't blow up the padded batch width
                    # (issue #120). Truncate a local copy — never the caller's
                    # sample tensor, which may be reused across epochs. Only
                    # applied on non-interleaved paths (callers pass None for
                    # interleaved; see the __init__ docstring).
                    if self.max_seq_length is not None:
                        items = [
                            t[: self.max_seq_length]
                            if t.shape[0] > self.max_seq_length
                            else t
                            for t in items
                        ]
                    # Log sequence length statistics
                    seq_lens = [t.shape[0] for t in items]
                    max_len = max(seq_lens)
                    min_len = min(seq_lens)
                    avg_len = sum(seq_lens) / len(seq_lens)

                    # Update global stats
                    _seq_len_stats["count"] += len(seq_lens)
                    _seq_len_stats["total_len"] += sum(seq_lens)
                    _seq_len_stats["max_len"] = max(_seq_len_stats["max_len"], max_len)
                    _seq_len_stats["min_len"] = min(_seq_len_stats["min_len"], min_len)

                    # Log every 100 batches (to avoid spam)
                    if _seq_len_stats["count"] % 100 < len(seq_lens):
                        global_avg = (
                            _seq_len_stats["total_len"] / _seq_len_stats["count"]
                            if _seq_len_stats["count"] > 0
                            else 0
                        )
                        logger.info(
                            f"[SEQ_LEN] Batch: min={min_len}, max={max_len}, avg={avg_len:.1f} | "
                            f"Global: avg={global_avg:.1f}, max={_seq_len_stats['max_len']}, samples={_seq_len_stats['count']}"
                        )

                    # Dynamic Padding for Text Tensors
                    # items is list of 1D tensors (L,)
                    collated[k] = pad_sequence(
                        items, batch_first=True, padding_value=self.padding_value
                    )
                else:
                    # Raw strings from some dataloaders (e.g., val_loader with WebDataset)
                    # Keep as list - let downstream code (batch_viz) handle tokenization
                    collated[k] = items

            elif (
                k == "time_series"
                and self.passthrough_time_series
                and isinstance(items[0], torch.Tensor)
            ):
                collated[k] = items

            elif isinstance(items[0], torch.Tensor):
                # Check for variable size in any dim (except 0 for stack, but stack requires all dims match)
                shapes = [t.shape for t in items]
                is_variable = False
                if len(shapes) > 0:
                    first = shapes[0]
                    for s in shapes:
                        if s != first:
                            is_variable = True
                            break

                if is_variable:
                    # If it's 1D (Text/Audio), pad
                    if items[0].dim() == 1:
                        collated[k] = pad_sequence(
                            items, batch_first=True, padding_value=self.padding_value
                        )
                    elif items[0].dim() == 2 and items[0].shape[0] == 2:
                        # Edge Index (2, E). Transpose -> Pad -> Transpose
                        # items_T = [t.t() for t in items]
                        # padded = pad_sequence(items_T, batch_first=True, padding_value=-1) # -1 for edges?
                        # padded_T = padded.permute(0, 2, 1)
                        # Complex. Return LIST for now to avoid crashes.
                        collated[k] = items
                    else:
                        # General 2D+ Variable (Graph X, etc)
                        # Pad dim 0
                        collated[k] = pad_sequence(
                            items, batch_first=True, padding_value=0
                        )
                else:
                    collated[k] = torch.stack(items)

            elif isinstance(items[0], dict):
                # Recursive Collation for Table/Graph Dicts
                # items is list of dicts: [{'input_ids': ...}, {'input_ids': ...}]
                # Reuse the same collator logic to stack inner keys
                collated[k] = self.__call__(items)

            else:
                # Lists/Strings (e.g. captions for debut)
                collated[k] = items

        return collated


class BucketedCollator:
    """
    Collator that groups samples by sequence length to minimize padding waste.

    Instead of batching samples in arrival order (which may have wildly different
    lengths), this collator buffers samples and groups them by similar length
    before padding.

    This significantly improves throughput when training on mixed datasets with
    variable sequence lengths (e.g., pixmo short captions vs arxiv long captions).

    Usage:
        collator = BucketedCollator(tokenizer, bucket_size=100, num_buckets=8)
        dataloader = DataLoader(dataset, batch_size=8, collate_fn=collator)

    How it works:
        1. Samples arrive in mini-batches from DataLoader
        2. We sort samples within each batch by text length
        3. This ensures samples in the same forward pass have similar lengths
        4. Reduces padding from O(max_len) to O(bucket_max_len)

    For more aggressive bucketing across batches, use BucketedBatchSampler instead.
    """

    def __init__(
        self,
        tokenizer,
        padding_value=0,
        sort_within_batch: bool = True,
        log_efficiency: bool = True,
        max_seq_length: int | None = None,
    ):
        """
        Args:
            tokenizer: HuggingFace tokenizer for padding value
            padding_value: Value to use for padding (default: tokenizer.pad_token_id)
            sort_within_batch: If True, sort samples by length within each batch
            log_efficiency: If True, log padding efficiency statistics
            max_seq_length: Hard cap on sequence length. Sequences longer than this
                are truncated BEFORE padding. This prevents OOM and backward pass
                spikes from outlier-length sequences at dataset epoch boundaries.
        """
        self.tokenizer = tokenizer
        self.padding_value = padding_value
        if tokenizer.pad_token_id is not None:
            self.padding_value = tokenizer.pad_token_id

        self.sort_within_batch = sort_within_batch
        self.log_efficiency = log_efficiency
        self.max_seq_length = max_seq_length

        # Track efficiency stats
        self._stats = {
            "batches": 0,
            "total_tokens": 0,
            "padded_tokens": 0,
            "max_efficiency": 0.0,
            "min_efficiency": 1.0,
            "truncated_samples": 0,
            "total_samples": 0,
        }

    def __call__(self, batch):
        """
        Collate a batch of samples, optionally sorting by sequence length.
        """
        if not batch:
            return {}

        # Sort batch by text length if enabled
        if self.sort_within_batch and "text" in batch[0]:
            # Get text lengths
            text_key = "text"
            if isinstance(batch[0][text_key], torch.Tensor):
                lengths = [
                    (i, sample[text_key].shape[0]) for i, sample in enumerate(batch)
                ]
            else:
                # Raw strings - estimate by character length
                lengths = [(i, len(sample[text_key])) for i, sample in enumerate(batch)]

            # Sort by length (longest first for better GPU utilization)
            lengths.sort(key=lambda x: x[1], reverse=True)
            sorted_indices = [i for i, _ in lengths]
            batch = [batch[i] for i in sorted_indices]

        # --- Hard sequence length cap ---
        # Truncate sequences exceeding max_seq_length BEFORE padding.
        # This prevents backward pass spikes from outlier-length sequences
        # (e.g., at dataset epoch boundaries where the bucket buffer drains
        # and emits poorly-sorted batches with extreme length variance).
        if self.max_seq_length is not None:
            for sample in batch:
                if "text" in sample and isinstance(sample["text"], torch.Tensor):
                    self._stats["total_samples"] += 1
                    if sample["text"].shape[0] > self.max_seq_length:
                        sample["text"] = sample["text"][: self.max_seq_length]
                        self._stats["truncated_samples"] += 1

            # Log truncation stats periodically
            if (
                self.log_efficiency
                and self._stats["truncated_samples"] > 0
                and self._stats["batches"] % 100 == 0
            ):
                trunc_rate = (
                    self._stats["truncated_samples"] / self._stats["total_samples"]
                    if self._stats["total_samples"] > 0
                    else 0.0
                )
                logger.info(
                    f"[TRUNCATION] {self._stats['truncated_samples']} of "
                    f"{self._stats['total_samples']} samples truncated to "
                    f"{self.max_seq_length} tokens ({trunc_rate:.1%})"
                )

        # Now collate using standard logic
        keys = batch[0].keys()
        collated = {}

        for k in keys:
            items = [d[k] for d in batch]

            if k == "text":
                if isinstance(items[0], torch.Tensor):
                    # Compute efficiency stats
                    seq_lens = [t.shape[0] for t in items]
                    max_len = max(seq_lens)
                    total_actual = sum(seq_lens)
                    total_padded = max_len * len(seq_lens)
                    efficiency = (
                        total_actual / total_padded if total_padded > 0 else 1.0
                    )

                    # Update stats
                    self._stats["batches"] += 1
                    self._stats["total_tokens"] += total_actual
                    self._stats["padded_tokens"] += total_padded
                    self._stats["max_efficiency"] = max(
                        self._stats["max_efficiency"], efficiency
                    )
                    self._stats["min_efficiency"] = min(
                        self._stats["min_efficiency"], efficiency
                    )

                    # Log periodically
                    if self.log_efficiency and self._stats["batches"] % 100 == 0:
                        overall_eff = (
                            self._stats["total_tokens"] / self._stats["padded_tokens"]
                        )
                        logger.info(
                            f"[BUCKET] Batch {self._stats['batches']}: "
                            f"len_range=[{min(seq_lens)}, {max_len}], eff={efficiency:.1%}, "
                            f"overall_eff={overall_eff:.1%}"
                        )

                    # Pad sequences
                    collated[k] = pad_sequence(
                        items, batch_first=True, padding_value=self.padding_value
                    )
                else:
                    collated[k] = items

            elif isinstance(items[0], torch.Tensor):
                shapes = [t.shape for t in items]
                is_variable = len(set(shapes)) > 1

                if is_variable:
                    if items[0].dim() == 1:
                        collated[k] = pad_sequence(
                            items, batch_first=True, padding_value=self.padding_value
                        )
                    elif items[0].dim() == 2 and items[0].shape[0] == 2:
                        # Edge Index (2, E): pad_sequence on dim 0 would try to
                        # align E (dim 1) and crash with "size of tensor a must
                        # match size of tensor b at non-singleton dimension 1".
                        # Match MultimodalCollator's behavior: return the list
                        # and let the encoder's per-graph batching handle it.
                        # NB: the only (2, E) 2D tensors in this codebase are
                        # graph edge_index payloads (see src/data/multimodal.py).
                        # If a future modality introduces a different (2, F)
                        # variable-shape tensor that should be padded/stacked,
                        # gate this branch on the key name (e.g. k == "edge_index").
                        collated[k] = items
                    else:
                        collated[k] = pad_sequence(
                            items, batch_first=True, padding_value=0
                        )
                else:
                    collated[k] = torch.stack(items)

            elif isinstance(items[0], dict):
                collated[k] = self.__call__(items)

            else:
                collated[k] = items

        return collated

    def get_efficiency_stats(self):
        """Return padding efficiency statistics."""
        if self._stats["padded_tokens"] > 0:
            overall = self._stats["total_tokens"] / self._stats["padded_tokens"]
        else:
            overall = 1.0

        return {
            "batches": self._stats["batches"],
            "overall_efficiency": overall,
            "min_efficiency": self._stats["min_efficiency"],
            "max_efficiency": self._stats["max_efficiency"],
            "wasted_tokens": self._stats["padded_tokens"] - self._stats["total_tokens"],
        }
