import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb

logger = logging.getLogger(__name__)


def denormalize_image(tensor):
    """Denormalize ImageNet tensors to [0, 1] for visualization."""
    # (C, H, W)
    if tensor.max() > 10.0:  # Heuristic for already 0-255?
        return tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(tensor.device)
    img = tensor * std + mean
    img = torch.clamp(img, 0, 1)
    return img.permute(1, 2, 0).cpu().numpy()


class BatchVisualizer:
    def __init__(self, tokenizer=None, use_gt_prefix_for_captions: bool = False):
        """
        Args:
            tokenizer: Tokenizer for encoding/decoding text.
            use_gt_prefix_for_captions: If True, use first few words from GT as prompt
                for caption-only data. If False (default), use "The image" to avoid
                hallucination from GT leakage.
        """
        self.tokenizer = tokenizer
        self.use_gt_prefix_for_captions = use_gt_prefix_for_captions

    def visualize(self, batch, active_modalities=None):
        """
        Visualizes a multimodal batch.
        Returns a dictionary of wandb objects ready for logging.
        Only logs modalities that are actually active (have data).
        """
        viz_outputs = {}
        logger.debug(f"VIEWER Inputs: {list(batch.keys())}")

        # Helper to get first N non-empty samples
        def get_samples(key, limit=4):
            samples = []
            if key not in batch:
                return samples

            data = batch[key]

            if isinstance(data, torch.Tensor):
                for i in range(data.shape[0]):
                    if len(samples) >= limit:
                        break
                    if data[i].sum() != 0:
                        samples.append(data[i])
            elif isinstance(data, dict) and "x" in data:
                x = data["x"]
                edge_index = data.get("edge_index")
                for i in range(x.shape[0]):
                    if len(samples) >= limit:
                        break
                    if x[i].sum() != 0:
                        e = edge_index[i] if edge_index is not None else None
                        samples.append({"x": x[i], "edge_index": e})
            elif isinstance(data, dict) and "input_ids" in data:
                ids = data["input_ids"]
                for i in range(ids.shape[0]):
                    if len(samples) >= limit:
                        break
                    samples.append(ids[i])
            elif key == "table" and isinstance(data, torch.Tensor):
                for i in range(data.shape[0]):
                    if len(samples) >= limit:
                        break
                    samples.append(data[i])
            return samples

        # Only visualize modalities that have actual data (skip empty placeholders)

        # 1. Images (only if present)
        images = get_samples("image")
        if images:
            pass

        # 2. Time Series (only if present)
        ts_data = get_samples("time_series")
        if ts_data:
            fig, axes = plt.subplots(1, len(ts_data), figsize=(4 * len(ts_data), 4))
            if len(ts_data) == 1:
                axes = [axes]
            for ax, ts in zip(axes, ts_data, strict=False):
                ax.plot(ts.view(-1).cpu().numpy())
                ax.grid(True)
                ax.set_title(f"TS (L={ts.numel()})")
            plt.tight_layout()
            viz_outputs["viz/time_series"] = wandb.Image(fig)
            plt.close(fig)

        # 3. Graph
        graph_data = get_samples("graph")
        if graph_data:
            fig, axes = plt.subplots(1, len(graph_data), figsize=(4 * len(graph_data), 4))
            if len(graph_data) == 1:
                axes = [axes]

            for ax, g in zip(axes, graph_data, strict=False):
                edges = g.get("edge_index")
                if edges is not None and edges.numel() > 0:
                    # edges is (2, E)
                    u = edges[0].cpu().numpy()
                    v = edges[1].cpu().numpy()
                    ax.scatter(u, v, s=1, alpha=0.5)
                    ax.set_title(f"Edges (E={edges.shape[1]})")

                    # Set Limits
                    max_node = max(u.max(), v.max(), 32)
                    ax.set_xlim(0, max_node)
                    ax.set_ylim(0, max_node)
                    ax.invert_yaxis()
                else:
                    ax.text(0.5, 0.5, "No Edges", ha="center")
                    ax.set_title(f"Nodes Only (N={g['x'].shape[0]})")

            plt.tight_layout()
            viz_outputs["viz/graph"] = wandb.Image(fig)
            plt.close(fig)
        # (No placeholder for empty graphs)

        # 4. Geometry
        geo_data = get_samples("geometry")
        if geo_data:
            fig, axes = plt.subplots(1, len(geo_data), figsize=(4 * len(geo_data), 4))
            if len(geo_data) == 1:
                axes = [axes]

            for ax, vol in zip(axes, geo_data, strict=False):
                vol = vol.cpu()
                if vol.dim() == 5:  # (T, X, Y, Z, C)
                    # Slice mid-time, mid-depth
                    t = vol.shape[0] // 2
                    d = vol.shape[3] // 2
                    slc = vol[t, :, :, d, :]  # (X, Y, C)
                    if slc.shape[-1] == 3:  # RGB/Vector-3
                        mag = torch.norm(slc, dim=-1)
                    else:
                        mag = slc.mean(dim=-1)

                    ax.imshow(mag.numpy(), cmap="magma")
                    ax.set_title("5D Slice (Mag)")

                elif vol.dim() == 4 and vol.shape[-1] == 3:  # (X, Y, Z, C)
                    d = vol.shape[2] // 2
                    slc = vol[:, :, d, :]
                    mag = torch.norm(slc, dim=-1)
                    ax.imshow(mag.numpy(), cmap="magma")
                    ax.set_title("4D Vec Slice")

                elif vol.dim() == 3:  # (D, H, W) or (C, H, W)?
                    # GeometryEncoder usually inputs (D, H, W) or (C, D, H, W)
                    # PRISM Geometry is usually (D, H, W) ?
                    # Let's assume D is first dim
                    d = vol.shape[0] // 2
                    if d < vol.shape[0]:
                        ax.imshow(vol[d].numpy(), cmap="viridis")
                        ax.set_title("3D Scalar Slice")
                    else:
                        ax.text(0.5, 0.5, "Small Vol", ha="center")
                if vol.dim() == 5:  # (T, X, Y, Z, C)
                    # Slice mid-time, mid-depth
                    t = vol.shape[0] // 2
                    d = vol.shape[3] // 2
                    slc = vol[t, :, :, d, :]  # (X, Y, C)
                    if slc.shape[-1] == 3:  # RGB/Vector-3
                        mag = torch.norm(slc, dim=-1)
                    else:
                        mag = slc.mean(dim=-1)

                    ax.imshow(mag.numpy(), cmap="magma")
                    ax.set_title("5D Slice (Mag)")

                elif vol.dim() == 4 and vol.shape[-1] == 3:  # (X, Y, Z, C)
                    d = vol.shape[2] // 2
                    slc = vol[:, :, d, :]
                    mag = torch.norm(slc, dim=-1)
                    ax.imshow(mag.numpy(), cmap="magma")
                    ax.set_title("4D Vec Slice")

                elif vol.dim() == 3:  # (D, H, W) or (C, H, W)?
                    # GeometryEncoder usually inputs (D, H, W) or (C, D, H, W)
                    # PRISM Geometry is usually (D, H, W) ?
                    # Let's assume D is first dim
                    d = vol.shape[0] // 2
                    if d < vol.shape[0]:
                        ax.imshow(vol[d].numpy(), cmap="viridis")
                        ax.set_title("3D Scalar Slice")
                    else:
                        ax.text(0.5, 0.5, "Small Vol", ha="center")
                else:
                    ax.text(0.5, 0.5, f"Dim {vol.shape}", ha="center")

                ax.axis("off")

            plt.tight_layout()
            viz_outputs["viz/geometry"] = wandb.Image(fig)
            plt.close(fig)
        # (No placeholder for empty geometry)

        # 5. Table
        table_data = get_samples("table")
        if table_data:
            fig, axes = plt.subplots(1, len(table_data), figsize=(6 * len(table_data), 4))
            if len(table_data) == 1:
                axes = [axes]

            for ax, ids in zip(axes, table_data, strict=False):
                # Decode if tokenizer available
                decoded_text = "No Tokenizer"
                if self.tokenizer:
                    try:
                        # TAPAS/Table decoding
                        # Just decode straight IDs
                        decoded_text = self.tokenizer.decode(
                            ids, skip_special_tokens=True
                        )
                        # Wrap text
                        import textwrap

                        decoded_text = "\n".join(textwrap.wrap(decoded_text, width=40))
                    except Exception:
                        decoded_text = "Decode Error"

                ax.text(
                    0.1,
                    0.5,
                    decoded_text,
                    fontsize=9,
                    va="center",
                    fontfamily="monospace",
                )
                ax.set_title(f"Table (Tokens={len(ids)})")
                ax.axis("off")

            plt.tight_layout()
            viz_outputs["viz/table"] = wandb.Image(fig)
            plt.close(fig)
        # (No placeholder for empty tables)

        # 6. Text samples visualization removed per user request
        # Ground truth is shown in viz/predictions table instead

        # Predictions require model argument — see visualize_predictions()

        return viz_outputs

    def visualize_predictions(self, batch, model, limit=4, modality="image"):
        """
        Generates predictions and returns a consolidated WandB Table.
        Includes Image/Timeseries (if available), Ground Truth, and Prediction in one view.
        Set modality="time_series" to render timeseries instead of images.
        """
        if not model or not self.tokenizer:
            return None

        # Some training callers pass the modality flag inconsistently; the data
        # itself is the reliable signal. Detect the effective modality from the
        # actual batch contents so we do not send a time-series batch through the
        # image-generation path.
        effective_modality = modality
        if "time_series" in batch:
            effective_modality = "time_series"
        elif modality not in {"image", "time_series"}:
            effective_modality = "image"

        if effective_modality == "time_series":
            table = wandb.Table(columns=["Timeseries", "Ground Truth", "Prediction"])
        else:
            table = wandb.Table(columns=["Image", "Ground Truth", "Prediction"])

        # We need to construct a mini-batch for generation
        # Slicing is hard. Let's trying generating for the WHOLE batch (if small) or just first item
        # Training batch size is 16. Generating for 16 is fine.

        try:
            # Generate
            # We need to handle DDP wrapper if present
            unwrapped = model.module if hasattr(model, "module") else model

            # Determine Model Dtype
            target_dtype = next(unwrapped.parameters()).dtype
            # logger.info(f"Viz Gen Target Dtype: {target_dtype}")

            # Cast Batch Tensors to Model Dtype (if float)
            # Recursively handle dicts
            def cast_inputs(item):
                if isinstance(item, torch.Tensor):
                    if item.is_floating_point():
                        return item.to(dtype=target_dtype)
                    return item
                elif isinstance(item, dict):
                    return {k: cast_inputs(v) for k, v in item.items()}
                elif isinstance(item, list):
                    return [cast_inputs(x) for x in item]
                return item

            gen_batch = cast_inputs(batch)

            # --- Debug: Log input batch info ---
            print(f"[VIZ DEBUG] Input batch keys: {list(gen_batch.keys())}")
            if "text" in gen_batch and isinstance(gen_batch["text"], torch.Tensor):
                print(f"[VIZ DEBUG] Input text shape: {gen_batch['text'].shape}")
                first_decoded = self.tokenizer.decode(
                    gen_batch["text"][0], skip_special_tokens=False
                )
                print(
                    f"[VIZ DEBUG] First GT caption (first 150 chars): {repr(first_decoded[:150])}"
                )

            # --- Fix: Create a proper generation prompt ---
            # For caption-based training (PixMo), the batch['text'] IS the caption (ground truth)
            # For generation, we need to give the model a PROMPT, not the answer
            #
            # Strategy: Replace text with a short prompt like "Describe this image:"
            # The model should then generate the description

            if (
                "text" in gen_batch
                and isinstance(gen_batch["text"], torch.Tensor)
                and self.tokenizer
            ):
                if effective_modality == "time_series":
                    # Timeseries prompt
                    try:
                        orig_ids = gen_batch["text"]
                        batch_size = orig_ids.shape[0]
                        device = orig_ids.device

                        if self.tokenizer.bos_token:
                            prompt = [
                                self.tokenizer.bos_token + "This timeseries has "
                            ] * batch_size
                        else:
                            prompt = ["This timeseries has "] * batch_size
                        start_ids = self.tokenizer(
                            prompt, padding=True, truncation=True, return_tensors="pt"
                        ).input_ids.to(device)

                        gen_batch["text"] = start_ids
                        print(
                            f"[VIZ DEBUG] TS prompt tokenized, shape: {gen_batch['text'].shape}"
                        )
                    except Exception as e_prompt:
                        print(f"[VIZ DEBUG] Prompt creation error: {e_prompt}")
                        import traceback

                        traceback.print_exc()
                else:
                    try:
                        is_tensor = isinstance(gen_batch["text"], torch.Tensor)
                        is_list = isinstance(gen_batch["text"], list)

                        print(f"[VIZ DEBUG] gen_batch['text'] type: {type(gen_batch['text'])}")
                        if is_list and len(gen_batch["text"]) > 0:
                            print(
                                f"[VIZ DEBUG] First element type: {type(gen_batch['text'][0])}, value: {repr(gen_batch['text'][0])}"
                            )

                        if is_list and len(gen_batch["text"]) == 0:
                            # Empty list causes crash in model.generate
                            print("[VIZ DEBUG] Removing empty text list from batch")
                            del gen_batch["text"]

                        elif is_tensor or (is_list and len(gen_batch["text"]) > 0):
                            orig_text_data = gen_batch["text"]
                            batch_size = len(orig_text_data)
                            device = (
                                orig_text_data.device
                                if is_tensor
                                else next(unwrapped.parameters()).device
                            )

                            # Get Decoded Text
                            if is_tensor:
                                decoded = self.tokenizer.batch_decode(
                                    orig_text_data, skip_special_tokens=True
                                )
                            else:
                                # Robustly convert to string (handles integers, numpy str, etc)
                                decoded = [str(x) for x in orig_text_data]

                            first_sample = decoded[0] if decoded else ""

                            # Check if this is instruction-formatted data
                            has_instruction_format = any(
                                marker in first_sample
                                for marker in [
                                    "Assistant:",
                                    "Response:",
                                    "Output:",
                                    "User:",
                                ]
                            )

                            if has_instruction_format:
                                # Instruction-tuned data (VQA, pointing, etc.): truncate at response marker
                                print(
                                    "[VIZ DEBUG] Detected instruction format, truncating at response marker"
                                )
                                prompts = []
                                for txt in decoded:
                                    for marker in ["Assistant:", "Response:", "Output:"]:
                                        if marker in txt:
                                            txt = txt.split(marker)[0] + marker
                                            break
                                    prompts.append(txt)
                            else:
                                # Caption-only data (pretraining)
                                if self.use_gt_prefix_for_captions:
                                    print(
                                        "[VIZ DEBUG] Caption-only format detected, using GT prefix as prompt"
                                    )
                                    prompts = []
                                    for txt in decoded:
                                        words = txt.split()
                                        if len(words) >= 3:
                                            num_words = min(4, max(2, len(words) // 4))
                                            prompt = " ".join(words[:num_words])
                                        else:
                                            prompt = "The image"
                                        prompts.append(prompt)
                                else:
                                    print(
                                        "[VIZ DEBUG] Caption-only format detected, using 'The image' prompt"
                                    )
                                    prompts = ["The image"] * batch_size

                            # Re-tokenize the prompts
                            encoded = self.tokenizer(
                                prompts, padding=True, truncation=True, return_tensors="pt"
                            )
                            gen_batch["text"] = encoded["input_ids"].to(device)
                            print(f"[VIZ DEBUG] Prompt tokenized, shape: {gen_batch['text'].shape}")
                            print(f"[VIZ DEBUG] First prompt: {repr(prompts[0][:100])}")

                    except Exception as e_prompt:
                        print(f"[VIZ DEBUG] Prompt creation error: {e_prompt}")
                        import traceback

                        traceback.print_exc()
            # ------------------------------------------

            # Ensure model is in eval mode for generation (affects Dropout/Norm)
            was_training = unwrapped.training
            unwrapped.eval()

            # Resolve pad_token_id from tokenizer (model may not have it attached)
            pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
            if pad_token_id is None:
                pad_token_id = getattr(self.tokenizer, "eos_token_id", 0)

            try:
                with torch.no_grad():
                    if effective_modality == "time_series":
                        gen_inputs = {
                            "time_series": batch["time_series"],
                            "text": gen_batch["text"],
                        }
                        gen_ids = unwrapped.generate(
                            gen_inputs,
                            max_new_tokens=50,
                            top_p=0.9,
                            temperature=0.8,
                            repetition_penalty=1.2,
                        )
                    else:
                        # Limit generation to first few tokens for speed
                        # Use Greedy Decoding (do_sample=False) to match evaluator stability
                        gen_ids = unwrapped.generate(
                            gen_batch,
                            max_new_tokens=50,
                            do_sample=False,
                        )
            except Exception as e:
                print(f"[VIZ DEBUG] model.generate failed for effective_modality={effective_modality}: {e}")
                return None
            finally:
                # Restore training state
                if was_training:
                    unwrapped.train()

            # --- DEBUG: Log raw generation output for ALL samples ---
            print(
                f"[VIZ DEBUG] gen_ids type: {type(gen_ids)}, shape: {gen_ids.shape if hasattr(gen_ids, 'shape') else 'N/A'}"
            )
            if hasattr(gen_ids, "shape") and len(gen_ids.shape) >= 2:
                print(f"[VIZ DEBUG] First sequence raw IDs: {gen_ids[0].tolist()[:30]}")
            raw_decoded = self.tokenizer.batch_decode(
                gen_ids, skip_special_tokens=False
            )
            print(f"[VIZ DEBUG] Raw decoded (first): {repr(raw_decoded[0][:100])}")
            # Log ALL predictions for leakage diagnosis
            for _dbg_i, _dbg_txt in enumerate(raw_decoded[:limit]):
                _clean = _dbg_txt.replace("<pad>", "").strip()
                print(f"[VIZ DEBUG] Pred[{_dbg_i}]: {repr(_clean[:150])}")
            # --- Leakage check: re-generate sample 0 alone and compare ---
            # NOTE: XPU (Intel Max Series GPU) exhibits inherent floating-point
            # non-determinism — even the same single sample generated twice produces
            # different outputs. Exact string match is therefore too strict.
            # We use common-prefix-ratio as a soft similarity metric instead.
            # A high ratio (>0.5) indicates the model is behaving consistently
            # despite hardware-level non-determinism; a very low ratio (<0.2) with
            # completely unrelated content would suggest a real problem.
            try:
                single_batch = {
                    k: v[0:1] if isinstance(v, torch.Tensor) else [v[0]]
                    for k, v in gen_batch.items()
                }
                # Re-tokenize prompt for single sample (no padding artifact)
                if "text" in single_batch and isinstance(
                    single_batch["text"], torch.Tensor
                ):
                    single_prompt = self.tokenizer.decode(
                        single_batch["text"][0], skip_special_tokens=True
                    )
                    single_enc = self.tokenizer(single_prompt, return_tensors="pt")
                    single_batch["text"] = single_enc["input_ids"].to(
                        single_batch["text"].device
                    )
                with torch.no_grad():
                    single_ids = unwrapped.generate(
                        single_batch,
                        max_new_tokens=50,
                        do_sample=False,
                        pad_token_id=pad_token_id,
                    )
                single_pred = self.tokenizer.decode(
                    single_ids[0], skip_special_tokens=True
                )
                batched_pred = self.tokenizer.decode(
                    gen_ids[0], skip_special_tokens=True
                )
                exact_match = single_pred == batched_pred
                # Compute common prefix length as soft similarity
                single_toks = single_ids[0].tolist()
                batched_toks = gen_ids[0].tolist()
                common = 0
                for s, b in zip(single_toks, batched_toks, strict=False):
                    if s == b:
                        common += 1
                    else:
                        break
                total = min(len(single_toks), len(batched_toks))
                ratio = common / total if total > 0 else 0.0
                print(
                    f"[VIZ LEAKAGE CHECK] Sample 0 single vs batched: "
                    f"exact={exact_match}, common_prefix={common}/{total} ({ratio:.0%})"
                )
                if not exact_match:
                    print(f"[VIZ LEAKAGE CHECK] Single:  {repr(single_pred[:120])}")
                    print(f"[VIZ LEAKAGE CHECK] Batched: {repr(batched_pred[:120])}")
                    if ratio < 0.2:
                        print(
                            "[VIZ LEAKAGE CHECK] WARNING: Very low similarity — "
                            "investigate potential cross-sample leakage or padding bug"
                        )
            except Exception as e_lk:
                print(f"[VIZ LEAKAGE CHECK] Error: {e_lk}")
            # ----------------------------------------

            # Decode
            # gen_ids shape: (B, Seq)
            preds = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

            # Collect Rows
            meta = batch.get("_metadata", [""] * len(preds))  # noqa: F841

            # Ground Truth - handle both Tensor and list of strings
            if "text" in batch:
                if isinstance(batch["text"], torch.Tensor):
                    gts = self.tokenizer.batch_decode(
                        batch["text"], skip_special_tokens=True
                    )
                elif isinstance(batch["text"], list):
                    # Raw strings from val_loader (collator keeps them as list)
                    gts = [str(x) for x in batch["text"]]
                else:
                    gts = ["N/A"] * len(preds)
            else:
                gts = ["N/A"] * len(preds)

            for i in range(min(limit, len(preds))):
                if modality == "time_series":
                    ts_data = batch["time_series"][i]
                    fig, axes = plt.subplots(1, len(ts_data), figsize=(4 * len(ts_data), 4))
                    if len(ts_data) == 1:
                        axes = [axes]
                    for ax, ts in zip(axes, ts_data, strict=False):
                        ax.plot(ts.view(-1).cpu().numpy())
                        ax.grid(True)
                        ax.set_title(f"TS (L={ts.numel()})")
                    plt.tight_layout()
                    viz_item = wandb.Image(fig)
                    plt.close(fig)
                else:
                    # Get image for this sample (if available)
                    viz_item = None
                    if "image" in batch and batch["image"][i].sum() != 0:
                        try:
                            img_np = denormalize_image(batch["image"][i])
                            viz_item = wandb.Image(img_np)
                        except Exception:
                            viz_item = None

                # Truncate long texts for readability
                gt_text = gts[i][:500] if len(gts[i]) > 500 else gts[i]
                pred_text = preds[i][:500] if len(preds[i]) > 500 else preds[i]

                table.add_data(viz_item, gt_text, pred_text)

            return table

        except Exception as e:
            print(f"Viz Gen Error: {e}")
            return None
