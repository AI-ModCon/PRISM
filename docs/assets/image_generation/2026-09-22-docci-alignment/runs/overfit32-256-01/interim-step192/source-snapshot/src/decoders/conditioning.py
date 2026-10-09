"""Prompt-only input compilation; supervision never enters the input sequence."""

import torch
from torch.nn.utils.rnn import pad_sequence


def compile_inputs(model, inputs, embeddings):
    """Return padded embeddings, validity, spans, and original text positions.

    Unlike the QA loss compiler this needs no prompt/answer metadata. In
    interleaved mode each adjacent modality marker pair consumes one reference.
    Prefix mode orders modalities as configured, then places text last.
    """
    by_name = dict(embeddings)
    if "text" not in by_name or "text" not in inputs:
        raise ValueError("Structured output decoding requires text instructions")
    text = by_name["text"]
    ids = inputs["text"]
    batch, text_len = ids.shape
    if text.shape[:2] != ids.shape:
        raise ValueError("Text embedding and token shapes differ")
    pad_id = model._resolve_pad_id()
    text_mask = inputs.get("text_attention_mask")
    if text_mask is None:
        for attr in ("backbone_tokenizer", "tokenizer"):
            tokenizer = getattr(model, attr, None)
            if pad_id is not None and getattr(tokenizer, "eos_token_id", None) == pad_id:
                raise ValueError(
                    "Explicit text_attention_mask is required when PAD and EOS share an ID"
                )
        text_mask = ids.ne(pad_id) if pad_id is not None else torch.ones_like(ids, dtype=torch.bool)
    if text_mask.shape != ids.shape:
        raise ValueError("text_attention_mask must match text token IDs")
    if not torch.all((text_mask == 0) | (text_mask == 1)):
        raise ValueError("text_attention_mask must be binary")
    text_mask = text_mask.to(device=text.device, dtype=torch.bool)
    chunks = {}
    for name, embedding in embeddings:
        if name == "text":
            continue
        if embedding.ndim != 3 or embedding.shape[0] != batch:
            raise ValueError(f"Invalid {name} embedding batch")
        # Multiple source images remain individually addressable and ordered.
        if name == "image" and inputs[name].ndim == 5:
            count = inputs[name].shape[1]
            valid = inputs.get(
                "image_mask", torch.ones(batch, count, device=text.device, dtype=torch.bool)
            )
            if valid.shape != (batch, count):
                raise ValueError("image_mask must have shape (B, N)")
            if not torch.all((valid == 0) | (valid == 1)):
                raise ValueError("image_mask must be binary")
            valid = valid.to(device=text.device, dtype=torch.bool)
            if torch.any(valid[:, 1:] & ~valid[:, :-1]):
                raise ValueError("Valid references must form a prefix in image_mask")
            if count == 0 or embedding.shape[1] % count:
                raise ValueError("Image tokens must divide evenly across references")
            per = embedding.shape[1] // count
            chunks[name] = [
                [embedding[b, n * per : (n + 1) * per] for n in range(count) if valid[b, n]]
                for b in range(batch)
            ]
        else:
            tokens_per = (
                int(model.encoders[name].tokens_per_instance())
                if model.config.is_interleaved_qa and name != "image"
                else embedding.shape[1]
            )
            if tokens_per < 1 or embedding.shape[1] % tokens_per:
                raise ValueError(f"Invalid token count for {name}")
            chunks[name] = [list(embedding[b].split(tokens_per)) for b in range(batch)]
    marker_pairs = {
        name: pair
        for name, pair in (model.config.modality_start_end_token_indices or {}).items()
        if name != "text"
    }
    start_map = {pair[0]: name for name, pair in marker_pairs.items() if name != "text"}
    end_ids = {pair[1] for name, pair in marker_pairs.items() if name != "text"}
    if (
        len(start_map) != len(marker_pairs)
        or set(start_map) & end_ids
        or len(end_ids) != len(marker_pairs)
    ):
        if marker_pairs:
            raise ValueError("Modality marker token IDs must be unique")
    rows, spans = [], {name: [[] for _ in range(batch)] for name in by_name}
    text_positions = torch.full_like(ids, -1)
    for b in range(batch):
        row = []
        offset = 0

        def append(name, value, row=row, b=b):
            nonlocal offset
            value = value.to(device=text.device, dtype=text.dtype)
            row.append(value)
            spans.setdefault(name, [[] for _ in range(batch)])[b].append(
                (offset, offset + len(value))
            )
            offset += len(value)

        if not model.config.is_interleaved_qa:
            for name in chunks:
                for chunk in chunks[name][b]:
                    append(name, chunk)
            for t in range(text_len):
                if text_mask[b, t]:
                    text_positions[b, t] = offset
                    append("text", text[b, t : t + 1])
        else:
            cursors = {name: 0 for name in chunks}
            t = 0
            while t < text_len:
                if not text_mask[b, t]:
                    t += 1
                    continue
                token = int(ids[b, t])
                if token in start_map:
                    name = start_map[token]
                    if (
                        t + 1 >= text_len
                        or not text_mask[b, t + 1]
                        or int(ids[b, t + 1]) != marker_pairs[name][1]
                    ):
                        raise ValueError(f"Start/end tokens for {name} must be adjacent")
                    if name not in chunks or cursors[name] >= len(chunks[name][b]):
                        raise ValueError(f"Missing reference for {name} marker")
                    append(name, chunks[name][b][cursors[name]])
                    cursors[name] += 1
                    t += 2
                elif token in end_ids:
                    raise ValueError("Unpaired modality end token")
                else:
                    text_positions[b, t] = offset
                    append("text", text[b, t : t + 1])
                    t += 1
            for name in chunks:
                if cursors[name] != len(chunks[name][b]):
                    raise ValueError(
                        f"Unconsumed {name} references: every source needs a marker pair"
                    )
        if not row:
            raise ValueError("Empty conditioning sequence")
        limit = model.config.max_merged_seq_length
        if limit is not None and offset > limit:
            # Structured generation always fails before an oversized attention allocation.
            raise ValueError(
                f"Merged sequence length {offset} exceeds max_merged_seq_length={limit}"
            )
        rows.append(torch.cat(row, dim=0))
    x = pad_sequence(rows, batch_first=True)
    lengths = torch.tensor([len(row) for row in rows], device=x.device)
    mask = torch.arange(x.shape[1], device=x.device)[None, :] < lengths[:, None]
    return x, mask, spans, text_positions


def align_text_targets(target, text_positions, mask):
    if target.shape != text_positions.shape:
        raise ValueError("Text targets must align with input text IDs; use -100 for instructions")
    labels = torch.full(mask.shape, -100, dtype=torch.long, device=mask.device)
    valid = text_positions >= 0
    batch_indices = torch.arange(len(target), device=mask.device)[:, None].expand_as(text_positions)
    labels[batch_indices[valid], text_positions[valid]] = target.to(mask.device)[valid]
    if not labels[:, 1:].ne(-100).any():
        raise ValueError("Text supervision has no next-token targets")
    return labels
