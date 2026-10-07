
import os
import sys

# Set HF cache BEFORE any transformers/huggingface_hub imports
# models_cache/ contains models--*/ directly (not under hub/ subdirectory)
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_models_cache = os.path.join(_project_root, "models_cache")
if os.path.isdir(_models_cache):
    os.environ["HF_HUB_CACHE"] = _models_cache
    os.environ["HUGGINGFACE_HUB_CACHE"] = _models_cache
    os.environ["TRANSFORMERS_CACHE"] = _models_cache

import argparse
import datetime
import difflib
import hashlib
import json
import logging
import os
import re

import numpy as np
import torch
from accelerate import Accelerator
from safetensors.torch import load_file

# BLEU score (optional, graceful fallback)
try:
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False


def compute_word_overlap(gt: str, pred: str) -> float:
    """Compute Jaccard word overlap (intersection / union)."""
    gt_words = set(gt.lower().split())
    pred_words = set(pred.lower().split())
    if not gt_words or not pred_words:
        return 0.0
    intersection = gt_words & pred_words
    union = gt_words | pred_words
    return len(intersection) / len(union) if union else 0.0


def compute_bleu(gt: str, pred: str) -> float:
    """Compute BLEU-4 score with smoothing."""
    if not HAS_NLTK:
        return 0.0
    reference = [gt.lower().split()]
    hypothesis = pred.lower().split()
    smoothie = SmoothingFunction().method1
    try:
        return sentence_bleu(reference, hypothesis, smoothing_function=smoothie)
    except Exception:
        return 0.0


def compute_rouge_l(gt: str, pred: str) -> float:
    """Compute ROUGE-L F1 score (longest common subsequence)."""
    gt_tokens = gt.lower().split()
    pred_tokens = pred.lower().split()

    if not gt_tokens or not pred_tokens:
        return 0.0

    # LCS using SequenceMatcher on word tokens
    matcher = difflib.SequenceMatcher(None, gt_tokens, pred_tokens)
    lcs_length = sum(block.size for block in matcher.get_matching_blocks())

    precision = lcs_length / len(pred_tokens) if pred_tokens else 0
    recall = lcs_length / len(gt_tokens) if gt_tokens else 0

    if precision + recall == 0:
        return 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return f1


try:
    import matplotlib.pyplot as plt
    from PIL import Image
except ImportError:
    plt = None
    Image = None

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from src.config import ModelConfig
from src.data.multimodal import StreamingMultimodalDataset, _get_modality
from src.decoders import remap_legacy_decoder_keys
from src.eval import EvaluatorRegistry
from src.eval.tasks.geometry import MatBenchEvaluator
from src.eval.tasks.modality_tasks import (
    GraphEvaluator,
    SciTSEvaluator,
    TableEvaluator,
)
from src.eval.tasks.vision import VQAv2Evaluator
from src.hf_cache import load_cached_tokenizer
from src.model import UnifiedTransformer
from src.utils.banner import print_prism_banner

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _default_viz_dir(prefix: str, checkpoint_path: str | None) -> str:
    """Build a collision-resistant visualization directory from a checkpoint."""
    checkpoint = os.path.basename(os.path.normpath(checkpoint_path or "checkpoint"))
    checkpoint_slug = re.sub(r"[^A-Za-z0-9._-]+", "-", checkpoint).strip("-") or "checkpoint"
    checkpoint_hash = hashlib.sha1(
        os.path.abspath(checkpoint_path or checkpoint).encode("utf-8")
    ).hexdigest()[:10]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"visualization/{prefix}_{checkpoint_slug}_{checkpoint_hash}_{timestamp}"


# --- Visualization Helpers (Refactored for Subplots) ---


def denormalize_image(tensor):
    """Denormalize ImageNet tensors for visualization."""
    # (C, H, W) -> (H, W, C)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = tensor * std + mean
    img = torch.clamp(img, 0, 1)
    return img.permute(1, 2, 0).cpu().numpy()


def _plot_geometry_sample(tensor, text, name, viz_dir, ax=None):
    if plt is None:
        return
    try:
        data = tensor
        if hasattr(tensor, "cpu"):
            data = tensor.cpu().numpy()
        elif isinstance(tensor, list | tuple):
            data = np.array(tensor)

        # Setup Axis
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
            own_fig = True
        else:
            own_fig = False

        if len(data.shape) == 4:  # (C, D, H, W)
            d, h, w = data.shape[1:]
            ax.imshow(data[0, d // 2, :, :])
            ax.set_title("Geo (Mid-Slice 4D)")
        elif len(data.shape) == 3:  # (D, H, W)
            d, h, w = data.shape
            ax.imshow(data[d // 2, :, :])
            ax.set_title("Geo (Mid-Slice 3D)")
        elif len(data.shape) == 2:  # (H, W) or (N, D)
            if data.shape[1] == 6:  # Point Cloud (N, 6)
                # 3D scatter on 2D ax? Just project XY
                ax.scatter(data[:, 0], data[:, 1], c=data[:, 2], cmap="Greens", s=10)
                ax.set_title("Geo (Point Cloud XY)")
            else:
                ax.imshow(data)
                ax.set_title("Geo (2D)")
        else:
            ax.plot(data.flatten())
            ax.set_title("Geo (1D)")

        ax.axis("off")

        if own_fig:
            plt.tight_layout()
            plt.savefig(f"{viz_dir}/{name}.png")
            plt.close()
    except Exception as e:
        print(f"[PLOT ERROR] {name}: {e}")


def _plot_table_sample(tensor, text, name, viz_dir, ax=None):
    if plt is None:
        return
    try:
        if isinstance(tensor, dict):
            tensor = tensor.get("input_ids", tensor.get("ids", None))
        if tensor is None:
            return
        if hasattr(tensor, "cpu"):
            tensor = tensor.cpu()
        if hasattr(tensor, "tolist"):
            arr = np.array(tensor.tolist())
        else:
            arr = np.array(tensor)

        if len(arr.shape) == 1:
            arr = arr.reshape(1, -1)

        # Setup Axis
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 4))
            own_fig = True
        else:
            own_fig = False

        ax.imshow(arr, aspect="auto", cmap="viridis")
        # ax.colorbar(label="Token ID") # Colorbar tricky on subplot
        ax.set_title(f"Table (Tokens)\nL: {arr.shape[1]}")

        if own_fig:
            plt.tight_layout()
            plt.savefig(f"{viz_dir}/{name}.png")
            plt.close()
    except Exception as e:
        print(f"[PLOT ERROR] {name}: {e}")


def _plot_graph_sample(tensor, text, name, viz_dir, ax=None):
    if plt is None:
        return
    try:
        # Setup Axis
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
            own_fig = True
        else:
            own_fig = False

        # Usually dict for graph
        if isinstance(tensor, dict):
            edges = tensor.get("edge_index", [])
            x = tensor.get("x", [])
            if hasattr(edges, "cpu"):
                edges = edges.cpu().numpy()
            if hasattr(x, "cpu"):
                x = x.cpu().numpy()

            num_nodes = x.shape[0] if hasattr(x, "shape") else 10
            theta = np.linspace(0, 2 * np.pi, num_nodes, endpoint=False)
            radius = 1.0
            x_coords = radius * np.cos(theta)
            y_coords = radius * np.sin(theta)

            ax.scatter(x_coords, y_coords, s=20)
            if hasattr(edges, "shape") and len(edges.shape) == 2 and edges.shape[0] == 2:
                for i in range(min(edges.shape[1], 100)):  # Limit edges for speed
                    u, v = edges[0, i], edges[1, i]
                    if u < num_nodes and v < num_nodes:
                        ax.plot(
                            [x_coords[u], x_coords[v]], [y_coords[u], y_coords[v]], "k-", alpha=0.3
                        )
            ax.set_title(f"Graph (N={num_nodes})")
        else:
            ax.text(0.5, 0.5, "Unknown Format", ha="center")

        ax.axis("off")

        if own_fig:
            plt.tight_layout()
            plt.savefig(f"{viz_dir}/{name}.png")
            plt.close()
    except Exception as e:
        print(f"[PLOT ERROR] {name}: {e}")


def _plot_ts_sample(tensor, text, name, viz_dir, ax=None):
    if plt is None:
        return
    try:
        # Setup Axis
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))
            own_fig = True
        else:
            own_fig = False

        data = tensor
        if hasattr(tensor, "cpu"):
            data = tensor.cpu().numpy()
        ax.plot(data.flatten())
        ax.set_title("Time Series")
        ax.grid(True)

        if own_fig:
            plt.tight_layout()
            plt.savefig(f"{viz_dir}/{name}.png")
            plt.close()
    except Exception as e:
        print(f"[PLOT ERROR] {name}: {e}")


def _plot_image_sample(tensor, text, name, viz_dir, ax=None):
    if plt is None:
        return
    try:
        # Setup Axis
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
            own_fig = True
        else:
            own_fig = False

        if hasattr(tensor, "cpu"):
            tensor = tensor.cpu()
        img = denormalize_image(tensor)
        ax.imshow(img)
        t_short = text[:20].replace("\n", " ")
        ax.set_title(f"Image\n{t_short}...")
        ax.axis("off")

        if own_fig:
            plt.tight_layout()
            plt.savefig(f"{viz_dir}/{name}.png")
            plt.close()
    except Exception as e:
        print(f"[PLOT ERROR] {name}: {e}")


def _save_verification_image(batch, name, viz_dir):
    """Save raw image for verification."""
    if plt is None:
        return

    if "image" in batch and batch["image"].max() > 0:
        if hasattr(batch["image"], "cpu"):
            tensor = batch["image"][0].cpu()
        else:
            tensor = batch["image"][0]

        img = denormalize_image(tensor)
        plt.imsave(f"{viz_dir}/{name}_image.png", img)


def _plot_collated_sample(batch, text, name, viz_dir):
    """Generates a single 1x5 composite plot for the sample."""
    if plt is None:
        return

    # Create Figure
    # Create Figure
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    if text:
        fig.suptitle(text, wrap=True, fontsize=10)

    # 1. Image
    if "image" in batch and batch["image"].max() > 0:
        _plot_image_sample(batch["image"][0], text, name, viz_dir, ax=axes[0])
    else:
        axes[0].text(0.5, 0.5, "BLANK (Image)", ha="center", va="center")
        axes[0].axis("off")
        axes[0].set_title("Image (Inactive)")

    # 2. Graph
    has_graph = False
    if "graph" in batch:
        g = batch["graph"]
        if isinstance(g, dict):
            edges = g.get("edge_index", None)
            if edges is not None and edges.numel() > 0 and edges.sum() > 0:
                has_graph = True

    if has_graph:
        _plot_graph_sample(batch["graph"], text, name, viz_dir, ax=axes[1])
    else:
        axes[1].text(0.5, 0.5, "BLANK (Graph)", ha="center", va="center")
        axes[1].axis("off")
        axes[1].set_title("Graph (Inactive)")

    # 3. Table
    if "table" in batch and batch["table"].sum() > 0:
        _plot_table_sample(batch["table"], text, name, viz_dir, ax=axes[2])
    else:
        axes[2].text(0.5, 0.5, "BLANK (Table)", ha="center", va="center")
        axes[2].axis("off")
        axes[2].set_title("Table (Inactive)")

    # 4. Time Series
    if "time_series" in batch and batch["time_series"].abs().sum() > 1e-6:
        _plot_ts_sample(batch["time_series"][0], text, name, viz_dir, ax=axes[3])
    else:
        axes[3].text(0.5, 0.5, "BLANK (Time Series)", ha="center", va="center")
        axes[3].axis("off")
        axes[3].set_title("Time Series (Inactive)")

    # 5. Geometry
    if "geometry" in batch and batch["geometry"].abs().sum() > 1e-6:
        _plot_geometry_sample(batch["geometry"][0], text, name, viz_dir, ax=axes[4])
    else:
        axes[4].text(0.5, 0.5, "BLANK (Geometry)", ha="center", va="center")
        axes[4].axis("off")
        axes[4].set_title("Geometry (Inactive)")

    plt.suptitle(f"Sample: {name} | Text: {text[:60]}...", fontsize=10)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{viz_dir}/{name}_composite.png")
    plt.close()


# --- Modality Tasks Imports (Already imported in top of file, safe to proceed) ---


def _load_resolved_model_config(config_path: str) -> dict:
    """Load and resolve only the model subtree of a Hydra or model YAML."""
    from omegaconf import OmegaConf

    raw_config = OmegaConf.load(config_path)
    model_config = raw_config.get("model", raw_config)
    resolved = OmegaConf.to_container(model_config, resolve=True)
    if not isinstance(resolved, dict):
        raise ValueError(f"Unsupported config format in {config_path}")
    return resolved

def _checkpoint_backbone_vocab_size(state_dict):
    value = state_dict.get("backbone.model.embed_tokens.weight")
    if value is not None and getattr(value, "ndim", 0) == 2:
        return int(value.shape[0])
    return None


def _model_tokenizer_len(model):
    tokenizer = getattr(model, "backbone_tokenizer", None)
    if tokenizer is None:
        tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None or not hasattr(tokenizer, "__len__"):
        return None
    try:
        return int(len(tokenizer))
    except TypeError:
        return None


def _set_backbone_vocab_size(model, vocab_size):
    backbone = getattr(model, "backbone", None)
    if backbone is not None and hasattr(backbone, "config"):
        backbone.config.vocab_size = int(vocab_size)
    if hasattr(model, "config"):
        model.config.vocab_size = int(vocab_size)


def _copy_vocab_rows_into_current_tensor(key, checkpoint_tensor, current_tensor):
    if checkpoint_tensor.ndim != 2 or current_tensor.ndim != 2:
        return None
    if checkpoint_tensor.shape[1] != current_tensor.shape[1]:
        return None

    merged = current_tensor.detach().clone()
    rows = min(int(checkpoint_tensor.shape[0]), int(current_tensor.shape[0]))
    merged[:rows].copy_(
        checkpoint_tensor[:rows].to(device=merged.device, dtype=merged.dtype)
    )
    logger.info(
        "Adapted %s: copied %d checkpoint vocab rows into current shape %s",
        key,
        rows,
        tuple(current_tensor.shape),
    )
    return merged


def _pad_mismatched_vocab_tensors(model, state_dict):
    current_state = model.state_dict()
    adapted = dict(state_dict)
    for key in ("backbone.model.embed_tokens.weight", "backbone.lm_head.weight"):
        checkpoint_tensor = adapted.get(key)
        current_tensor = current_state.get(key)
        if checkpoint_tensor is None or current_tensor is None:
            continue
        if tuple(checkpoint_tensor.shape) == tuple(current_tensor.shape):
            continue
        merged = _copy_vocab_rows_into_current_tensor(
            key, checkpoint_tensor, current_tensor
        )
        if merged is not None:
            adapted[key] = merged
    return adapted


def _align_checkpoint_backbone_vocab(model, state_dict):
    """Align vocab-sized tensors before load_state_dict enforces shapes."""
    target = _checkpoint_backbone_vocab_size(state_dict)
    backbone = getattr(model, "backbone", None)
    if (
        target is None
        or backbone is None
        or not hasattr(backbone, "resize_token_embeddings")
        or not hasattr(backbone, "get_input_embeddings")
    ):
        return state_dict

    input_embeddings = backbone.get_input_embeddings()
    current = int(input_embeddings.weight.shape[0])
    if current == target:
        _set_backbone_vocab_size(model, target)
        return state_dict

    tokenizer_len = _model_tokenizer_len(model)
    if target < current:
        logger.info(
            "Checkpoint vocab %d is smaller than eval model vocab %d"
            "%s; padding checkpoint vocab tensors instead of shrinking",
            target,
            current,
            f" (tokenizer length {tokenizer_len})" if tokenizer_len is not None else "",
        )
        return _pad_mismatched_vocab_tensors(model, state_dict)

    logger.info(
        "Resizing eval backbone token embeddings %d -> %d to match checkpoint",
        current,
        target,
    )
    rng_state = torch.get_rng_state()
    try:
        torch.manual_seed(0)
        backbone.resize_token_embeddings(target)
    finally:
        torch.set_rng_state(rng_state)

    _set_backbone_vocab_size(model, target)
    return state_dict


def setup_model(checkpoint_path, model_config=None, backbone_id=None):
    """Unified Model Loading Logic."""
    print(f"Using HF_HOME={os.environ.get('HF_HOME', 'default')}, offline={os.environ.get('HF_HUB_OFFLINE', '0')}")

    def _load_model_config_from_yaml(config_path: str, detected_d_model: int, backbone_override: str | None):
        """Build ModelConfig from either a model YAML or a Hydra run config.yaml."""
        model_cfg = _load_resolved_model_config(config_path)

        ts_projector = str(model_cfg.get("ts_projector", "linear"))
        patch_lens = model_cfg.get("timeomni_patch_len", 16)
        if isinstance(patch_lens, int):
            patch_lens = [patch_lens]
        elif isinstance(patch_lens, tuple):
            patch_lens = list(patch_lens)
        elif not isinstance(patch_lens, list):
            patch_lens = [16]

        strides = model_cfg.get("timeomni_stride", None)
        if strides is None:
            strides = patch_lens
        elif isinstance(strides, int):
            strides = [strides] * len(patch_lens)
        elif isinstance(strides, tuple):
            strides = list(strides)
        elif isinstance(strides, list):
            if len(strides) == 1 and len(patch_lens) > 1:
                strides = strides * len(patch_lens)
            elif len(strides) != len(patch_lens):
                strides = (strides + [strides[-1]])[: len(patch_lens)]
        else:
            strides = patch_lens

        timeomni_max_patches = int(model_cfg.get("timeomni_max_patches", 100))
        ts_max_length = int(model_cfg.get("max_ts_length", 512))
        if ts_projector == "timeomni":
            largest_idx = max(range(len(patch_lens)), key=lambda i: int(patch_lens[i]))
            budget_stride = int(strides[largest_idx])
            ts_max_length = budget_stride * max(1, timeomni_max_patches - 1)

        llm_backbone_id = model_cfg.get("backbone_id", backbone_override)
        if backbone_override:
            llm_backbone_id = backbone_override
        if not llm_backbone_id:
            raise ValueError(
                f"Model config {config_path} did not provide model.backbone_id and no --backbone was supplied"
            )

        llm_tokenizer_id = model_cfg.get("tokenizer_id", llm_backbone_id)

        return ModelConfig(
            llm_backbone_id=llm_backbone_id,
            llm_tokenizer_id=llm_tokenizer_id,
            freeze_backbone=bool(model_cfg.get("freeze_backbone", True)),
            freeze_encoders=bool(model_cfg.get("freeze_encoders", True)),
            modalities=list(model_cfg.get("modalities", ["text", "image"])),
            d_model=int(model_cfg.get("d_model", detected_d_model)),
            d_text=int(model_cfg.get("d_text", detected_d_model)),
            d_img=int(model_cfg.get("d_img", 1152)),
            d_table=int(model_cfg.get("d_table", 768)),
            d_ts=int(model_cfg.get("d_ts", 512)),
            d_geo=int(model_cfg.get("d_geo", 512)),
            d_graph=int(model_cfg.get("d_graph", 768)),
            image_encoder_id=model_cfg.get("image_encoder_id", "google/siglip2-base-patch16-224"),
            image_processor_id=model_cfg.get("image_processor_id", None),
            image_processor_strict=bool(model_cfg.get("image_processor_strict", False)),
            image_size=int(model_cfg.get("image_size", 224)),
            image_mean=tuple(model_cfg.get("image_mean", (0.5, 0.5, 0.5))),
            image_std=tuple(model_cfg.get("image_std", (0.5, 0.5, 0.5))),
            is_timeseries=bool(model_cfg.get("is_timeseries", False)),
            ts_projector=ts_projector,
            ts_encoder_id=model_cfg.get("ts_encoder_id", "Salesforce/moirai-2.0-R-small"),
            intern_s2_sampling_rate=float(model_cfg.get("intern_s2_sampling_rate", 1.0)),
            ts_variates=int(model_cfg.get("ts_variates", 1)),
            max_ts_length=ts_max_length,
            normalize_ts_in_encoder=bool(model_cfg.get("normalize_ts_in_encoder", True)),
            timeomni_patch_len=model_cfg.get("timeomni_patch_len", 16),
            timeomni_stride=model_cfg.get("timeomni_stride", None),
            timeomni_d_model=int(model_cfg.get("timeomni_d_model", 512)),
            timeomni_dropout=float(model_cfg.get("timeomni_dropout", 0.1)),
            timeomni_ts_tokens=int(model_cfg.get("timeomni_ts_tokens", 100)),
            timeomni_max_patches=timeomni_max_patches,
            is_interleaved_qa=bool(model_cfg.get("is_interleaved_qa", False)),
            modality_start_end_token_indices=model_cfg.get("modality_start_end_token_indices", {}),
            projector_norm_mode=model_cfg.get("projector_norm_mode", "layernorm"),
            projector_target_norm=float(model_cfg.get("projector_target_norm", 0.25)),
            projector_modality_embed_pos=model_cfg.get("projector_modality_embed_pos", "after_norm"),
            projector_modality_embed_scale=float(model_cfg.get("projector_modality_embed_scale", 0.02)),
            projector_text_norm_mean=float(model_cfg.get("projector_text_norm_mean", 0.25)),
            projector_text_norm_std=float(model_cfg.get("projector_text_norm_std", 0.05)),
            projector_text_elem_mean=float(model_cfg.get("projector_text_elem_mean", 0.0)),
            projector_text_elem_std=float(model_cfg.get("projector_text_elem_std", 0.006)),
            projector_norm_clip_min=float(model_cfg.get("projector_norm_clip_min", 0.1)),
            projector_norm_clip_max=float(model_cfg.get("projector_norm_clip_max", 0.5)),
            projector_hidden_mult=int(model_cfg.get("projector_hidden_mult", 1)),
            projector_num_layers=int(model_cfg.get("projector_num_layers", 2)),
            attn_implementation=model_cfg.get("attn_implementation", "sdpa"),
        )

    # Auto-detect d_model from checkpoint backbone weights
    ckpt_file = checkpoint_path
    if os.path.isdir(ckpt_file):
        ckpt_file = os.path.join(ckpt_file, "model.safetensors")
    if not os.path.exists(ckpt_file):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_file}")

    print("Probing checkpoint for model dimensions...")
    state_dict = load_file(ckpt_file)
    # Detect d_model from backbone embed_tokens or projector output dim
    d_model = None
    for k, v in state_dict.items():
        clean_k = k[7:] if k.startswith("module.") else k
        if clean_k == "backbone.model.embed_tokens.weight":
            d_model = v.shape[1]
            break
    if d_model is None:
        # Fallback: infer from projector output dim
        for k, v in state_dict.items():
            clean_k = k[7:] if k.startswith("module.") else k
            if clean_k == "projectors.image.fc1.bias":
                d_model = v.shape[0]
                break
    if d_model is None:
        d_model = 1280  # absolute fallback
    print(f"Detected d_model={d_model} from checkpoint")



    if model_config is None:
        ckpt_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(ckpt_file))))
        hydra_config_path = os.path.join(ckpt_root, ".hydra", "config.yaml")
        if os.path.isfile(hydra_config_path):
            print(f"No --model_config provided; falling back to Hydra config: {hydra_config_path}")
            model_config = hydra_config_path

    if model_config is None and backbone_id is None:
        raise ValueError("Must provide at least one of model_config or backbone_id to setup_model")
    
    if model_config:
        print(f"Loading Model Config ({model_config})...")
        if os.path.isfile(model_config):
            model_config = _load_model_config_from_yaml(model_config, d_model, backbone_id)
        else:
            model_config = ModelConfig.from_preset(model_config)
    else:
        print(f"No model config provided, using default backbone_id={backbone_id} with minimal config.")
        model_config = ModelConfig(
            llm_backbone_id=backbone_id,
            freeze_backbone=True,
            d_model=d_model,
            d_text=d_model,
            d_img=768,
            modalities=["text", "image"]
        )
    model = UnifiedTransformer(model_config)

    print(f"Loading Weights from {ckpt_file}...")
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    new_state_dict = _align_checkpoint_backbone_vocab(model, new_state_dict)

    # Remap pre-decoder-refactor keys (VLA action_head.* -> action_head.head.*)
    # so old checkpoints evaluate losslessly.
    new_state_dict = remap_legacy_decoder_keys(new_state_dict)

    # Strict=False to allow missing backbone/buffers
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)

    proj_missing = [k for k in missing if "projector" in k]
    if proj_missing:
        logger.warning(f"Projector weights missing: {len(proj_missing)} keys.")
    else:
        logger.info("Projector weights loaded successfully.")

    accelerator = Accelerator(mixed_precision="bf16")
    device = accelerator.device
    model.to(device, dtype=torch.bfloat16)
    model.eval()

    tokenizer_id = model_config.llm_tokenizer_id if model_config.llm_tokenizer_id else backbone_id
    print(f"Loading Tokenizer: {tokenizer_id}...")
    tokenizer = load_cached_tokenizer(tokenizer_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.tokenizer = tokenizer

    return model, tokenizer, device


def inspect_dataset_exhaustively(dataset, tokenizer, limit=1, visualize=False, viz_dir=None):
    """
    Iterates through EVERY active dataset individually (bypassing random sampling).
    Ports logic from verify_and_visualize_real_data.py to manually process items.
    """
    print("\n=== Exhaustive Dataset Inspection (Iterating all sources) ===")

    dataset_map = dataset.datasets_map
    sorted_keys = sorted(dataset_map.keys())

    for key in sorted_keys:
        info = dataset_map[key]
        if info.get("skip", False):
            print(f"[SKIP Config] {key}")
            continue

        name = info["name"]
        handler = info["handler"]
        print(f"\n>>> Verifying Dataset: {name} (Key: {key}, Handler: {handler})")

        # 1. Fetch Raw Item
        raw_item = None
        # Try Stream
        if key in dataset.streams:
            try:
                raw_item = next(dataset.streams[key])
            except StopIteration:
                print(f"FAILURE: Stream {name} is EMPTY.")
                continue
        # Try Object
        elif key in dataset.datasets_objs:
            dset = dataset.datasets_objs[key]
            print(f"DEBUG: Dataset Object Type: {type(dset)}")
            try:
                if hasattr(dset, "__iter__") and not hasattr(dset, "__getitem__"):  # Generator-like
                    raw_item = next(iter(dset))
                else:  # Map-style
                    # Check if it supports integer indexing
                    try:
                        raw_item = dset[0]
                    except Exception:
                        # Fallback to iteration just in case
                        raw_item = next(iter(dset))
            except Exception as e:
                print(f"FAILURE: Could not fetch from Dataset Object {name}: {e}")
                continue
        else:
            print(f"ERROR: No Stream OR Dataset Object found for {key}.")
            continue

        # 2. Dispatch Processing
        try:
            # Identify Modality (for visualization routing)
            # We map specific handlers to dataset methods, mirroring StreamingMultimodalDataset.__iter__
            # This is critical because some handlers (like graph_circuit) are NOT handled by generic _process_graph

            processor = None
            modality = "unknown"  # For visualization selection

            # --- Dispatch Logic ---
            if handler == "ts_aurora":
                processor = dataset._process_ts_aurora
                modality = "time_series"
            elif handler == "ts_quants":
                processor = dataset._process_ts_quants
                modality = "time_series"
            elif handler == "ts_mmd":
                processor = dataset._process_ts_mmd
                modality = "time_series"
            elif handler == "ts_time_mmd":
                processor = dataset._process_ts_time_mmd
                modality = "time_series"

            elif handler == "table_reasoning":
                processor = dataset._process_table_reasoning
                modality = "table"
            elif handler == "table_instruction":
                processor = dataset._process_table_instruction
                modality = "table"
            elif handler == "table_structure":
                processor = dataset._process_table_structure
                modality = "table"

            elif handler == "graph_captioning":
                processor = dataset._process_graph_captioning
                modality = "graph"
            elif handler == "graph_grounding":
                processor = dataset._process_graph_grounding
                modality = "graph"
            elif handler == "graph_instruction":
                processor = dataset._process_graph_instruction
                modality = "graph"
            elif handler == "graph_crystal":
                processor = dataset._process_graph_crystal
                modality = "graph"
            elif handler == "graph_circuit":
                processor = dataset._process_graph_circuit
                modality = "graph"

            elif handler.startswith("geo_physics"):
                processor = dataset._process_geo_physics
                modality = "geometry"
            elif handler == "geo_mat_tomo":
                processor = dataset._process_geo_mat_tomo
                modality = "geometry"
            elif handler == "geo_pde":
                processor = dataset._process_geo_pde
                modality = "geometry"

            elif handler == "image_pixmo":
                processor = dataset._process_image_pixmo
                modality = "image"
            elif handler == "image_points":
                processor = dataset._process_image_points
                modality = "image"
            elif handler == "text_sft":
                processor = dataset._process_text_sft
                modality = "text"

            # Generic fallbacks: route by the explicit `modality` field
            # in datasets_config.json (was substring-matched on handler — issue #23).
            else:
                modality = _get_modality(info, name)
                if modality == "image":
                    processor = dataset._process_image
                elif modality == "geometry":
                    processor = dataset._process_geo
                elif modality == "time_series":
                    processor = dataset._process_ts
                elif modality == "table":
                    processor = dataset._process_table
                elif modality == "graph":
                    processor = dataset._process_graph
                elif modality == "text":
                    # SFT usually handled by _process_text_sft, but fallback
                    def processor(x):
                        return None, str(x), "Generic Text"

            if processor is None:
                print(f"WARNING: No processor found for handler {handler}. Using generic str.")
                processed_tensor = None
                processed_text = str(raw_item)
            else:
                # execute
                res = processor(raw_item)

                # Robust Unpacking (2 or 3 values)
                metadata_str = ""
                if isinstance(res, tuple):
                    if len(res) == 3:
                        processed_tensor, processed_text, metadata_str = res
                    elif len(res) == 2:
                        processed_tensor, processed_text = res
                    else:
                        print(
                            f"WARNING: Processor returned tuple of length {len(res)}. Expecting 2 or 3."
                        )
                        processed_tensor, processed_text = res[0], str(res[1:])
                else:
                    processed_tensor = None
                    processed_text = str(res)

            print(f"  Processed Text: {repr(processed_text[:100])}...")
            if metadata_str:
                print(f"  Metadata: {metadata_str}")
            if processed_tensor is not None:
                if isinstance(processed_tensor, torch.Tensor):
                    print(f"  Processed Tensor: {processed_tensor.shape}")

                # Visualization
                if visualize:
                    safe_name = key.replace("/", "_")
                    if modality == "image":
                        _plot_image_sample(
                            processed_tensor, processed_text, f"{safe_name}_image", viz_dir
                        )
                    elif modality == "geometry":
                        _plot_geometry_sample(
                            processed_tensor, processed_text, f"{safe_name}_geo", viz_dir
                        )
                    elif modality == "time_series":
                        _plot_ts_sample(
                            processed_tensor, processed_text, f"{safe_name}_ts", viz_dir
                        )
                    elif modality == "table":
                        _plot_table_sample(
                            processed_tensor, processed_text, f"{safe_name}_table", viz_dir
                        )
                    elif modality == "graph":
                        _plot_graph_sample(
                            processed_tensor, processed_text, f"{safe_name}_graph", viz_dir
                        )
        except Exception as e:
            print(f"FAILURE Processing {name}: {e}")


def inspect_training_data(
    model,
    tokenizer,
    device,
    limit=5,
    visualize=False,
    viz_dir=None,
    exhaustive=False,
    modality_filter=None,
    checkpoint_path=None,
):
    """Inspects what the model sees during training by iterating the StreamingDataset.
    If model is provided, runs inference to check for overfit/memorization.
    """
    print(
        f"\n=== Inspecting Training Data (Streaming, Limit={limit}, Filter={modality_filter}) ==="
    )

    if visualize:
        if viz_dir is None:
            viz_dir = _default_viz_dir("verification", checkpoint_path)
        os.makedirs(viz_dir, exist_ok=True)
        print(f"Visualization Enabled. Saving plots to: {viz_dir}")

    # Load Dataset Config
    config_path = os.path.join(os.path.dirname(__file__), "../src/data/datasets_config.json")
    if not os.path.exists(config_path):
        print(f"Error: Config not found at {config_path}")
        return

    with open(config_path) as f:
        json.load(f)

    # Initialize Dataset (Zone A + B + C active)
    # We use a dummy transform/tokenizer handling since StreamingMultimodalDataset handles it internally if we pass tokenizer?
    # Actually StreamingMultimodalDataset logic is complex.
    # Simpler: Just instantiate it with split='train' and default config.

    # NOTE: We need to ensure we use the same logic as `train.py`.
    # `train.py` calls `StreamingMultimodalDataset(datasets_config, tokenizer=tokenizer, ...)`

    # Note: Scanning src/data/multimodal.py, init takes:
    # (self, tokenizer, batch_size=1, max_steps=1000, zone="zone_a", hf_token=None, ...)
    # It does NOT take config dict directly. It seems to manage datasets internally via DatasetManager.
    # We will assume zone="zone_a" covers the training set.

    dataset = StreamingMultimodalDataset(
        tokenizer=tokenizer,
        batch_size=1,
        zone="zone_a",  # Default training zone
    )

    # Branch for Exhaustive Mode
    if exhaustive:
        inspect_dataset_exhaustively(
            dataset, tokenizer, limit=limit, visualize=visualize, viz_dir=viz_dir
        )
        return

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)

    print(f"Iterating first {limit} samples...")
    for i, batch in enumerate(dataloader):
        if i >= limit:
            break

        print(f"\n--- Sample {i+1} ---")
        print(f"Batch Keys: {list(batch.keys())}")

        # Decode text (Ground Truth)
        gt_text = "N/A"
        if "text" in batch:
            # New format: 'text' key holds input_ids
            input_ids = batch["text"][0]
            if isinstance(input_ids, torch.Tensor):
                gt_text = tokenizer.decode(input_ids, skip_special_tokens=True)
                print(f"Text Content (GT): {repr(gt_text[:200])}...")
        elif "input_ids" in batch:
            input_ids = batch["input_ids"][0]
            gt_text = tokenizer.decode(input_ids, skip_special_tokens=True)
            print(f"Text Content (GT): {repr(gt_text[:200])}...")
        else:
            print("WARNING: 'text'/'input_ids' missing in batch!")

        # Check active modalities
        if "_metadata" in batch:
            print(f"  Metadata: {batch['_metadata']}")

        # Check Time Series stats if present
        if "time_series" in batch:
            ts = batch["time_series"]
            print(
                f"  TS Stats: Mean={ts.float().mean():.4f}, Std={ts.float().std():.4f}, Max={ts.max():.4f}"
            )

        modalities = [
            k
            for k in batch.keys()
            if k not in ["input_ids", "attention_mask", "labels", "token_type_ids"]
        ]
        print(f"Active Modalities: {modalities}")
        # Check shapes
        for m in modalities:
            val = batch[m]
            if isinstance(val, torch.Tensor):
                print(f"  {m}: shape={val.shape}, dtype={val.dtype}")

        # Filter check
        if modality_filter:
            if modality_filter.lower() not in modalities and modality_filter.lower() not in str(
                batch.keys()
            ):
                continue

        # Inference / Accuracy Check
        if model is not None:
            print("  [Inference Check]: Generating...")
            try:
                # Construct Inference Inputs
                # For Captioning (Image): Input = Image, Target = Text.
                # We feed image and prompt for start.
                gen_inputs = {}

                if "image" in batch:
                    gen_inputs["image"] = batch["image"].to(device)
                    # If the model is pure captioner, maybe empty string?
                    # Let's try standard VLM formatting if available, otherwise raw.
                    # For LLaVA-1.5 style pretraining, it often just predicts.

                if "input_ids" in batch:
                    # Ground Truth text
                    gt_text = tokenizer.decode(batch["input_ids"][0], skip_special_tokens=True)
                    print(f"  Ground Truth: {repr(gt_text[:100])}...")

                if gen_inputs:
                    # Tokenize prompt if needed
                    # For now, let's assume the model can handle 'image' kwarg and we can pass input_ids for start
                    # Check model generate signature... usually takes pixel_values etc.
                    # We'll rely on the model's prepare_inputs_for_generation or similar if it exists
                    # OR just pass raw if UnifiedTransformer handles it.
                    # UnifiedTransformer forward() handles dict. generate() usually wraps .

                    # Simple approach: Encode a start token/prompt
                    # tokenizer.bos_token might be handy

                    # Create dummy input_ids for start
                    start_prompt = tokenizer.bos_token if tokenizer.bos_token else ""
                    prompt_ids = tokenizer(start_prompt, return_tensors="pt").input_ids.to(device)

                    gen_inputs["input_ids"] = prompt_ids

                    # Move all tensors to device
                    gen_inputs = {
                        k: v.to(device) if hasattr(v, "to") else v for k, v in gen_inputs.items()
                    }

                    with torch.no_grad():
                        # Generate
                        # UnifiedTransformer.generate requires 'inputs' dict as first argument.
                        # We construct inputs dict with 'image' and 'text' (start prompt)

                        # Note: UnifiedTransformer.generate(inputs, max_new_tokens...)
                        # inputs = {'image': ..., 'text': ...}

                        inference_inputs = {
                            "text": prompt_ids,
                            **{k: v for k, v in gen_inputs.items() if k != "input_ids"},
                        }

                        out = model.generate(inference_inputs, max_new_tokens=50)

                    # Unwrap if ModelOutput
                    if hasattr(out, "sequences"):
                        out_seq = out.sequences[0]
                    elif isinstance(out, torch.Tensor):
                        out_seq = out[0]
                    else:
                        out_seq = out[0]

                    pred_text = tokenizer.decode(out_seq, skip_special_tokens=True)
                    print(f"  Prediction:   {repr(pred_text)}")
            except Exception as e:
                print(f"  [Inference Error]: {e}")
                pred_text = f"Error: {e}"

        # Visualization
        if visualize:
            name = f"sample_{i+1}"
            full_caption = f"GT: {gt_text}\nPred: {pred_text}"
            _plot_collated_sample(batch, full_caption, name, viz_dir)


def inspect_eval_examples(model, tokenizer, device, limit=5, modality_filter=None):
    """Qualitative Inspection of Examples per Modality."""
    print(f"\n=== Inspecting Evaluation Examples (Limit={limit}, Filter={modality_filter}) ===")

    all_evaluators = [
        ("Graph (ChEBI-20)", GraphEvaluator),
        ("Time (SciTS)", SciTSEvaluator),
        ("Table (Spider)", TableEvaluator),
        ("Vision (VQAv2)", VQAv2Evaluator),
        ("Geometry (MatBench)", MatBenchEvaluator),
    ]

    evaluators = []
    for name, cls in all_evaluators:
        if modality_filter:
            if modality_filter.lower() not in name.lower():
                continue
        evaluators.append((name, cls))

    for name, cls in evaluators:
        print(f"\n\n>>> Modality: {name} <<<")
        try:
            evaluator = cls(model, tokenizer, device=str(device))
            # Get iterator
            iterable = iter(evaluator.dataset)

            for i in range(limit):
                print(f"\n--- Sample {i+1}/{limit} ---")
                try:
                    item = next(iterable)
                except StopIteration:
                    print("End of dataset.")
                    break

                # --- Logic from inspect_modality_examples.py ---
                inputs = None
                prompt = "N/A"
                ground_truth = "N/A"

                if "Graph" in name:
                    smiles = item.get("SMILES", item.get("smiles", ""))
                    ground_truth = item.get("description", item.get("caption", ""))
                    prompt = f"Describe the following molecule/graph: {smiles}\nDescription:"
                    tok_inputs = tokenizer(prompt, return_tensors="pt")
                    inputs = {
                        "text": tok_inputs.input_ids,
                        "graph": evaluator._featurize_graph(smiles),
                    }

                elif "Time" in name:
                    q = item.get("question", item.get("question_text", ""))
                    ground_truth = item.get("answer", item.get("description", item.get("characteristics", "")))
                    if q:
                        prompt = f"Question: {q}\nAnswer:"
                    else:
                        prompt = "Question: Describe this time series.\nAnswer:"
                    tok_inputs = tokenizer(prompt, return_tensors="pt")
                    inputs = {
                        "text": tok_inputs.input_ids,
                        "time_series": evaluator._featurize_ts(item),
                    }

                elif "Table" in name:
                    q = item["question"]
                    db_id = item["db_id"]
                    prompt = f"Generate SQL for Database {db_id}: {q}\nSQL:"
                    ground_truth = item["query"]
                    tok_inputs = tokenizer(prompt, return_tensors="pt")
                    inputs = {
                        "text": tok_inputs.input_ids,
                        "table": evaluator._featurize_table(item),
                    }

                elif "Vision" in name:
                    image = item["image"]
                    if image.mode != "RGB":
                        image = image.convert("RGB")
                    inputs_proc = evaluator.processor(images=image, return_tensors="pt")
                    pixel_values = inputs_proc["pixel_values"].to(device)
                    q = item["question"]
                    prompt = f"Question: {q}\nAnswer:"
                    ground_truth = item["multiple_choice_answer"]
                    tok_inputs = tokenizer(prompt, return_tensors="pt")
                    inputs = {"image": pixel_values, "text": tok_inputs.input_ids}

                elif "Geometry" in name:
                    atoms = item["atomic_numbers"]
                    prompt = f"Predict property (Band Gap) for Crystal Structure:\nCrystal containing {len(atoms)} atoms.\nTarget:"
                    ground_truth = str(item.get("y", "N/A"))
                    tok_inputs = tokenizer(prompt, return_tensors="pt")
                    geo_feat = evaluator._featurize_geometry(item)
                    inputs = {"text": tok_inputs.input_ids, "geometry": geo_feat}

                print(f"Prompt: {repr(prompt)}")
                print(f"Ground Truth: {ground_truth}")

                if inputs:
                    output = evaluator.generate(inputs, max_new_tokens=64)
                    generated = output.replace(prompt, "").strip()
                    print(f"Prediction: {generated}")

                    # Basic Scoring Display
                    match = "NO"
                    if (
                        str(ground_truth).lower() in generated.lower()
                        or generated.lower() in str(ground_truth).lower()
                    ):
                        match = "YES (Partial)"
                    print(f"Status: {match}")
                else:
                    print("Error: Inputs could not be constructed.")

        except Exception as e:
            print(f"Failed to inspect {name}: {e}")


def run_full_eval(model, tokenizer, device, limit=100):
    """Run quantitative evaluation on all modalities."""
    print(f"\n=== Running Full Evaluation (Limit={limit}) ===")

    evaluators = [
        ("Graph (ChEBI-20)", "graph_chebi"),
        ("Time (SciTS)", "ts_scits"),
        ("Table (Spider)", "table_spider"),
        ("Geometry (MatBench)", "geometry_matbench"),
        ("Vision (VQAv2)", "vision_vqa"),
    ]

    # Explicit key -> headline metric map. Substring checks like `"time" in
    # key` silently fall through to N/A for keys like "ts_scits" that don't
    # contain the literal substring "time" — this maps the registered key
    # itself, so a mismatch is a KeyError-free "N/A" only when truly absent.
    _HEADLINE_METRIC = {
        "vision_vqa": "accuracy",
        "graph_chebi": "bleu",
        "ts_scits": "accuracy",
        "table_spider": "exact_match",
        "geometry_matbench": "mae",
    }

    results = {}

    for name, key in evaluators:
        print(f"-- Initializing {name} --")
        try:
            cls = EvaluatorRegistry.get(key)
            if not cls:
                print(f"Skipping {name}: Not found in registry.")
                continue

            evaluator = cls(model, tokenizer, device=str(device))
            print(f"   Evaluating {limit} samples...")
            metrics = evaluator.evaluate(limit=limit)

            # Extract key metric
            score = "N/A"
            metric_name = _HEADLINE_METRIC.get(key)
            if metric_name is not None:
                score = metrics.get(metric_name, 0)

            print(f"   Score: {score} (Full Metrics: {metrics})")
            results[name] = score

        except Exception as e:
            print(f"   Failed to evaluate {name}: {e}")
            results[name] = "Error"

    print("\n\n=== FINAL RESULTS ===")
    print(f"{'Task':<25} | {'Score':<10}")
    print("-" * 40)
    for k, v in results.items():
        print(f"{k:<25} | {str(v):<10}")


def verify_image_modality(
    model,
    tokenizer,
    device,
    limit=10,
    visualize=False,
    viz_dir=None,
    use_validation=False,
    checkpoint_path=None,
):
    """
    Dedicated verification for Image Modality (Captioning).
    Filters for 'pixmo_cap' style data, generates captions, and calculates similarity.

    Args:
        use_validation: If True, load from held-out validation shards instead of training data.
    """
    split_name = "VALIDATION" if use_validation else "TRAINING"
    print(f"\n=== Verifying Image Modality (Limit={limit}, Split={split_name}) ===")

    if use_validation:
        # Load validation shards directly via WebDataset
        try:
            import webdataset as wds
        except ImportError as err:
            raise ImportError(
                "webdataset required for validation mode. pip install webdataset"
            ) from err

        val_shards_dir = os.environ.get("PRISM_VAL_SHARDS_DIR")
        if not val_shards_dir:
            raise RuntimeError(
                "use_validation=True requires PRISM_VAL_SHARDS_DIR to be set "
                "to a directory containing *.tar WebDataset shards "
                "(e.g. /flare/<project>/<user>/.../pixmo_cap_webdataset/val_shards). "
                "Previously this defaulted to a hardcoded ngetty path; that "
                "fallback was removed because it silently failed for any other user."
            )
        # glob.glob() can hang on dfuse/DAOS mounts; os.listdir() + filter
        # is the safe pattern for shard directories that may live on DAOS.
        tar_files = sorted(
            os.path.join(val_shards_dir, f)
            for f in os.listdir(val_shards_dir)
            if f.endswith(".tar")
        )

        if not tar_files:
            raise ValueError(f"No validation shards found in {val_shards_dir}")

        print(f"Loading {len(tar_files)} validation shard(s) from {val_shards_dir}")

        # WebDataset pipeline
        from PIL import Image as PILImage
        from torchvision import transforms

        image_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        def process_sample(sample):
            """Convert webdataset sample to batch format."""
            img = sample.get("jpg") or sample.get("png") or sample.get("jpeg") or sample.get("webp")
            txt = (
                sample.get("txt", b"").decode("utf-8")
                if isinstance(sample.get("txt"), bytes)
                else sample.get("txt", "")
            )

            if img is not None and isinstance(img, PILImage.Image):
                img = img.convert("RGB")
                img_tensor = image_transform(img).unsqueeze(0)  # Add batch dim
            else:
                img_tensor = torch.zeros(1, 3, 224, 224)

            return {"image": img_tensor, "text": [txt]}

        val_dataset = (
            wds.WebDataset(tar_files, shardshuffle=False).decode("pil").map(process_sample)
        )
        dataloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=None
        )  # batch_size=None since already batched
    else:
        # Original training data path
        config_path = os.path.join(os.path.dirname(__file__), "../src/data/datasets_config.json")
        with open(config_path) as f:
            json.load(f)

        dataset = StreamingMultimodalDataset(tokenizer=tokenizer, batch_size=1, zone="zone_a")
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)

    if visualize:
        if viz_dir is None:
            viz_dir = _default_viz_dir("verify_image", checkpoint_path)
        os.makedirs(viz_dir, exist_ok=True)
        print(f"Visualization Enabled. Saving to: {viz_dir}")

    total_overlap = 0
    total_rouge = 0
    total_bleu = 0
    total_char = 0
    count = 0

    print(
        f"{'ID':<5} | {'Ground Truth (Trunc)':<40} | {'Prediction (Trunc)':<40} | {'Metrics':<20}"
    )
    print("-" * 115)

    for _i, batch in enumerate(dataloader):
        if count >= limit:
            break

        # 1. Filter: Must have Image AND Text (GT)
        if "image" not in batch or ("text" not in batch and "input_ids" not in batch):
            continue

        # 2. Extract GT
        gt_text = ""
        if "text" in batch:
            val = batch["text"][0]
            if isinstance(val, torch.Tensor):
                gt_text = tokenizer.decode(val, skip_special_tokens=True)
            elif isinstance(val, str):
                gt_text = val
        elif "input_ids" in batch:
            gt_text = tokenizer.decode(batch["input_ids"][0], skip_special_tokens=True)

        if not gt_text or len(gt_text) < 5:
            continue  # Skip empty/short GT (likely dummy)

        # 3. Predict
        try:
            # Use a better prompt than just BOS
            # prompt = "Describe this image."
            # Or simply use BOS if trained that way, but adding sampling helps repetition.
            # Prompt: "The image shows" usually works well for captioning models not heavily instruction tuned.
            prompt = "The image shows"

            start_ids = None
            if tokenizer.bos_token:
                start_ids = tokenizer(
                    tokenizer.bos_token + prompt, return_tensors="pt"
                ).input_ids.to(device)
            else:
                start_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

            gen_inputs = {"image": batch["image"].to(device), "text": start_ids}

            with torch.no_grad():
                # Enable sampling to reduce repetition loops
                out = model.generate(
                    gen_inputs,
                    max_new_tokens=60,
                    do_sample=True,
                    top_p=0.9,
                    temperature=1.0,  # Higher temp for more diversity
                    repetition_penalty=1.2,  # Penalize repetition
                )

            # Unwrap
            if hasattr(out, "sequences"):
                out = out.sequences[0]
            elif isinstance(out, torch.Tensor):
                out = out[0]
            else:
                out = out[0]

            # Decode (skip prompt in output if it's there? decode all for now and see)
            pred_text = tokenizer.decode(out, skip_special_tokens=True)

            # Post-process: Remove input prompt if the model echo's it (HF generate returns full seq)
            # OLMo generate usually returns full sequence?
            # Let's clean it up visually
            if pred_text.startswith(prompt):
                pass  # keep it for readability? Or remove?
                # pred_text = pred_text[len(prompt):].strip()

        except Exception as e:
            pred_text = f"Error: {e}"

        # 4. Score - Multiple Metrics
        # Word Overlap (Jaccard) - simple but effective for semantic similarity
        overlap_score = compute_word_overlap(gt_text, pred_text)
        # ROUGE-L (F1 of longest common subsequence) - standard for summarization
        rouge_score = compute_rouge_l(gt_text, pred_text)
        # BLEU-4 (n-gram precision) - standard for MT/captioning
        bleu_score = compute_bleu(gt_text, pred_text)
        # Legacy character ratio for comparison
        char_score = difflib.SequenceMatcher(None, gt_text, pred_text).ratio()

        total_overlap += overlap_score
        total_rouge += rouge_score
        total_bleu += bleu_score
        total_char += char_score
        count += 1

        # 5. Report
        gt_trunc = (gt_text[:37] + "...") if len(gt_text) > 37 else gt_text
        pred_trunc = (pred_text[:37] + "...") if len(pred_text) > 37 else pred_text
        # Clean newlines for table
        gt_trunc = gt_trunc.replace("\n", " ")
        pred_trunc = pred_trunc.replace("\n", " ")

        print(
            f"{count:<5} | {gt_trunc:<40} | {pred_trunc:<40} | Ovlp:{overlap_score:.2f} R-L:{rouge_score:.2f} B:{bleu_score:.2f}"
        )

        # 6. Viz
        if visualize:
            name = f"verify_{count}"
            _save_verification_image(batch, name, viz_dir)
            print(f"\n[VIZ_DATA] ID: {count}")
            print(f"[VIZ_GT] {gt_text}")
            print(f"[VIZ_PRED] {pred_text}\n")

            # Save GT/Pred to text file for easy comparison
            with open(f"{viz_dir}/{name}_comparison.txt", "w") as f:
                f.write(f"=== Sample {count} ===\n")
                f.write(
                    f"Scores: WordOverlap={overlap_score:.4f} | ROUGE-L={rouge_score:.4f} | BLEU={bleu_score:.4f} | Char={char_score:.4f}\n\n"
                )
                f.write(f"--- GROUND TRUTH ---\n{gt_text}\n\n")
                f.write(f"--- PREDICTION ---\n{pred_text}\n")

    # Compute averages
    avg_overlap = total_overlap / count if count > 0 else 0
    avg_rouge = total_rouge / count if count > 0 else 0
    avg_bleu = total_bleu / count if count > 0 else 0
    avg_char = total_char / count if count > 0 else 0

    print("-" * 115)
    print(f"Average Scores (n={count}):")
    print(f"  Word Overlap (Jaccard): {avg_overlap:.4f}")
    print(f"  ROUGE-L (F1):           {avg_rouge:.4f}")
    print(f"  BLEU-4:                 {avg_bleu:.4f}")
    print(f"  Char Similarity:        {avg_char:.4f}")

    # Save summary results
    if visualize and viz_dir:
        with open(f"{viz_dir}/results_summary.txt", "w") as f:
            f.write("PRISM Verification Results\n")
            f.write("=" * 40 + "\n")
            f.write(f"Samples: {count}\n\n")
            f.write("Average Scores:\n")
            f.write(f"  Word Overlap (Jaccard): {avg_overlap:.4f}\n")
            f.write(f"  ROUGE-L (F1):           {avg_rouge:.4f}\n")
            f.write(f"  BLEU-4:                 {avg_bleu:.4f}\n")
            f.write(f"  Char Similarity:        {avg_char:.4f}\n")



def verify_timeseries_scits(
    model,
    tokenizer,
    device,
    limit=10,
    visualize=False,
    viz_dir=None,
    ascii_art=True,
    checkpoint_path=None,
):
    """Verification for SciTS held-out shards (question/answer style)."""
    print(f"\n=== Verifying Time Series (SciTS Validation Mode, Limit={limit}) ===")

    # os.environ.get(var, default) only substitutes the default when the var
    # is UNSET — an explicitly-empty PRISM_VAL_SHARDS_DIR="" passes through,
    # and os.path.join("", "*.tar") silently globs the current working
    # directory instead of raising or falling back.
    val_shards_dir = (
        os.environ.get("PRISM_VAL_SHARDS_DIR")
        or "/flare/ModCon/pemami/data/SciTS-processed/val_shards"
    )
    # glob.glob() can hang on dfuse/DAOS mounts; os.listdir() + filter
    # is the safe pattern for shard directories that may live on DAOS.
    tar_files = sorted(
        os.path.join(val_shards_dir, f)
        for f in os.listdir(val_shards_dir)
        if f.endswith(".tar")
    )
    if not tar_files:
        raise ValueError(f"No *.tar shards found in {val_shards_dir}")
    print(f"Loading {len(tar_files)} SciTS validation shard(s) from {val_shards_dir}")

    max_ts_len = getattr(getattr(model, "config", None), "max_ts_length", None) or 512

    def _scits_iter():
        import io as _io
        import tarfile as _tarfile
        for tar_path in tar_files:
            with _tarfile.open(tar_path) as tf:
                members = {m.name: m for m in tf.getmembers()}
                stems: dict = {}
                for name in members:
                    stem, _, ext = name.partition(".")
                    stems.setdefault(stem, {})[ext] = name
                for _stem, exts in stems.items():
                    if "ts.npy" not in exts or "text" not in exts:
                        continue
                    npy_f = tf.extractfile(members[exts["ts.npy"]])
                    arr = np.load(_io.BytesIO(npy_f.read()))
                    text_f = tf.extractfile(members[exts["text"]])
                    text = text_f.read().decode("utf-8")
                    if "Question:" in text and "Answer:" in text:
                        parts = text.split("Answer:", 1)
                        q = parts[0].replace("Question:", "", 1).strip()
                        a = parts[1].strip()
                    else:
                        q = text.strip()
                        a = ""
                    ts = torch.from_numpy(
                        arr[:max_ts_len] if arr.shape[0] > max_ts_len else arr
                    ).float()
                    if ts.dim() == 1:
                        ts = ts.view(-1, 1)
                    yield {
                        "time_series": ts.unsqueeze(0),
                        "_scits_qa": {"question": q, "answer": a},
                    }

    dataloader = _scits_iter()

    if viz_dir is None:
        viz_dir = _default_viz_dir("verify_timeseries_scits", checkpoint_path)
    os.makedirs(viz_dir, exist_ok=True)

    total_overlap = total_rouge = total_bleu = total_char = 0
    count = 0

    for _i, batch in enumerate(dataloader):
        if limit and count >= limit:
            break

        if "time_series" not in batch:
            continue
        if batch["time_series"].abs().sum() < 1e-6:
            continue

        try:
            qa = batch["_scits_qa"]
            gt_text = qa["answer"]
            prompt = f"Question: {qa['question']}\nAnswer:"
            start_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            gen_inputs = {"time_series": batch["time_series"].to(device), "text": start_ids}
            ts_np = batch["time_series"][0].cpu().numpy().flatten()  # noqa: F841 — used by commented-out viz below

            with torch.no_grad():
                # out = model.generate(
                #     gen_inputs,
                #     max_new_tokens=256,
                #     do_sample=True,
                #     top_p=0.9,
                #     temperature=0.8,
                #     repetition_penalty=1.2,
                    
                # )
                out = model.generate(
                    gen_inputs,
                    max_new_tokens=64,
                    do_sample=False,
                    eos_token_id=tokenizer.eos_token_id,
                )

            if hasattr(out, "sequences"):
                out = out.sequences[0]
            elif isinstance(out, torch.Tensor):
                out = out[0]
            else:
                out = out[0]

            pred_text = tokenizer.decode(out, skip_special_tokens=False)
            # split by newline, keep the first line after the prompt
            pred_text = pred_text.split("\n", 1)[0].strip()
            
        except Exception as e:
            raise RuntimeError(f"Error: {e}") from e

        overlap_score = compute_word_overlap(gt_text, pred_text)
        rouge_score = compute_rouge_l(gt_text, pred_text)
        bleu_score = compute_bleu(gt_text, pred_text)
        char_score = difflib.SequenceMatcher(None, gt_text, pred_text).ratio()

        total_overlap += overlap_score
        total_rouge += rouge_score
        total_bleu += bleu_score
        total_char += char_score
        count += 1

        name = f"verify_ts_{count}"

        # if plt is not None:
        #     fig, ax = plt.subplots(figsize=(6, 2))
        #     ax.plot(ts_np.flatten(), linewidth=4)
        #     ax.set_xticks([])
        #     ax.set_yticks([])
        #     ax.set_xlabel("")
        #     ax.set_ylabel("")
        #     plt.tight_layout()
        #     plt.savefig(f"{viz_dir}/{name}.png")
        #     plt.close()

        def _ascii_plot(series, width=60, height=10):
            s = series.flatten()
            if len(s) == 0:
                return "  (empty series)"
            lo, hi = float(s.min()), float(s.max())
            span = hi - lo if hi != lo else 1.0
            indices = np.linspace(0, len(s) - 1, width).astype(int)
            sampled = s[indices]
            rows = []
            for r in range(height - 1, -1, -1):
                threshold = lo + span * r / (height - 1)
                rows.append(f"  |{''.join('*' if v >= threshold else ' ' for v in sampled)}|")
            rows.append(f"  +{'-' * width}+")
            rows.append(f"   min={lo:.3g}  max={hi:.3g}  len={len(s)}")
            return "\n".join(rows)

        # print()
        # print()
        # print("=" * 80)
        # print()
        # print("SCITS PROMPT >>")
        # print(tokenizer.decode(gen_inputs["text"][0], skip_special_tokens=False))
        # if ascii_art:
        #     print(f"\n[Time Series]\n{_ascii_plot(ts_np)}\n")
        # print()
        # print("=" * 80)
        # print()
        # print(f"ANSWER >> {pred_text}\n")

        with open(f"{viz_dir}/{name}_comparison.txt", "w") as f:
            f.write(f"=== Time Series Sample {count} ===\n")
            f.write(
                f"Scores: WordOverlap={overlap_score:.4f} | ROUGE-L={rouge_score:.4f} | BLEU={bleu_score:.4f} | Char={char_score:.4f}\n\n"
            )
            f.write(f"--- PROMPT ---\n{tokenizer.decode(gen_inputs['text'][0], skip_special_tokens=False)}\n\n")
            f.write(f"--- GROUND TRUTH ---\n{gt_text}\n\n")
            f.write(f"--- PREDICTION ---\n{pred_text}\n")

    avg_overlap = total_overlap / count if count > 0 else 0
    avg_rouge = total_rouge / count if count > 0 else 0
    avg_bleu = total_bleu / count if count > 0 else 0
    avg_char = total_char / count if count > 0 else 0

    with open(f"{viz_dir}/results_summary.txt", "w") as f:
        f.write("PRISM Time Series SciTS Verification Results\n")
        f.write("=" * 40 + "\n")
        f.write(f"Samples: {count}\n\n")
        f.write("Average Scores:\n")
        f.write(f"  Word Overlap (Jaccard): {avg_overlap:.4f}\n")
        f.write(f"  ROUGE-L (F1):           {avg_rouge:.4f}\n")
        f.write(f"  BLEU-4:                 {avg_bleu:.4f}\n")
        f.write(f"  Char Similarity:        {avg_char:.4f}\n")

    return {
        "count": count,
        "avg_overlap": avg_overlap,
        "avg_rouge": avg_rouge,
        "avg_bleu": avg_bleu,
        "avg_char": avg_char,
    }


def verify_timeseries_interleave(
    model,
    tokenizer,
    device,
    limit=10,
    visualize=False,
    viz_dir=None,
    ascii_art=True,
    checkpoint_path=None,
):
    """
    Verification for Time Series Modality — interleaved QA training data mode.
    Splits prompt and target from batch metadata and scores completions.
    """
    print(f"\n=== Verifying Time Series (Interleave Mode, Limit={limit}, Split=TRAINING) ===")

    dataset = StreamingMultimodalDataset(
        tokenizer=tokenizer,
        batch_size=1,
        zone="zone_a",
        model_config=model.config,
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)

    if visualize:
        if viz_dir is None:
            viz_dir = _default_viz_dir("verify_timeseries_qa", checkpoint_path)
        os.makedirs(viz_dir, exist_ok=True)

    total_overlap = total_rouge = total_bleu = total_char = 0
    count = 0

    for _i, batch in enumerate(dataloader):
        if count >= limit:
            break

        if "time_series" not in batch:
            continue
        if batch["time_series"].abs().sum() < 1e-6:
            continue

        try:
            # Training interleaved path: split prompt and target from metadata
            prompt_len, target_len = batch["_metadata"][0].split(" ")
            prompt_len = int(prompt_len)
            target_len = int(target_len)
            prompt_target_tokens = batch["text"][0]
            prompt_tokens = prompt_target_tokens[:prompt_len]
            gt_text = tokenizer.decode(
                prompt_target_tokens[prompt_len:prompt_len + target_len],
                skip_special_tokens=True,
            )
            gen_inputs = {
                "time_series": batch["time_series"].to(device),
                "text": prompt_tokens.unsqueeze(0).to(device),
                "_metadata": [f"{prompt_len} 0"],
            }
            ts_np = batch["time_series"][0].cpu().numpy().flatten()
            # Each <ts><ts/> span is a fixed-length segment of max_ts_length
            # steps (see _process_ts_qa's linear/moirai pad/truncate path).
            # This used to hardcode 256, which only matched the default
            # config — any model config with a different max_ts_length
            # (e.g. max_ts_length=512) would raise a cryptic reshape error
            # or silently misalign segments if the size happened to still
            # divide evenly.
            _seg_len = int(getattr(model.config, "max_ts_length", 256) or 256)
            if _seg_len <= 0 or ts_np.size % _seg_len != 0:
                raise RuntimeError(
                    f"[verify_timeseries_interleave] time_series length "
                    f"{ts_np.size} is not a multiple of model.config."
                    f"max_ts_length={_seg_len}; cannot split into <ts> segments"
                )
            ts_np = ts_np.reshape(-1, _seg_len)
            num_ts = ts_np.shape[0]

            with torch.no_grad():
                out = model.generate(
                    gen_inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    top_p=0.9,
                    temperature=0.8,
                    repetition_penalty=1.2,
                )

            if hasattr(out, "sequences"):
                out = out.sequences[0]
            elif isinstance(out, torch.Tensor):
                out = out[0]
            else:
                out = out[0]

            pred_text = tokenizer.decode(out, skip_special_tokens=True)

        except Exception as e:
            raise RuntimeError(f"Error: {e}") from e

        overlap_score = compute_word_overlap(gt_text, pred_text)
        rouge_score = compute_rouge_l(gt_text, pred_text)
        bleu_score = compute_bleu(gt_text, pred_text)
        char_score = difflib.SequenceMatcher(None, gt_text, pred_text).ratio()

        total_overlap += overlap_score
        total_rouge += rouge_score
        total_bleu += bleu_score
        total_char += char_score
        count += 1

        if visualize:
            name = f"verify_ts_{count}"

            if plt is not None:
                for i in range(num_ts):
                    fig, ax = plt.subplots(figsize=(6, 2))
                    ax.plot(ts_np[i], linewidth=4)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_xlabel("")
                    ax.set_ylabel("")
                    plt.tight_layout()
                    plt.savefig(f"{viz_dir}/{name}_{i}_plot.png")
                    plt.close()

            def _ascii_plot(series, width=60, height=10):
                """Render a 1-D array as a small ASCII spark-line plot."""
                s = series.flatten()
                if len(s) == 0:
                    return "  (empty series)"
                lo, hi = float(s.min()), float(s.max())
                span = hi - lo if hi != lo else 1.0
                indices = np.linspace(0, len(s) - 1, width).astype(int)
                sampled = s[indices]
                rows = []
                for r in range(height - 1, -1, -1):
                    threshold = lo + span * r / (height - 1)
                    rows.append(f"  |{''.join('*' if v >= threshold else ' ' for v in sampled)}|")
                rows.append(f"  +{'-' * width}+")
                rows.append(f"   min={lo:.3g}  max={hi:.3g}  len={len(s)}")
                return "\n".join(rows)

            ts_segments = [ts_np[i] for i in range(num_ts)]
            num_ts_display = num_ts

            decoded_prompt = tokenizer.decode(gen_inputs["text"][0], skip_special_tokens=False)
            ts_idx = 0
            output_parts = []
            remaining = decoded_prompt
            while "<ts>" in remaining and ts_idx < num_ts_display:
                before, _, after = remaining.partition("<ts>")
                _, _, after = after.partition("<ts/>")
                output_parts.append(before)
                if ascii_art:
                    output_parts.append(
                        f"\n[Time Series {ts_idx}]\n{_ascii_plot(ts_segments[ts_idx])}\n"
                    )
                else:
                    output_parts.append(f"\n[Time Series {ts_idx}]\n")
                remaining = after
                ts_idx += 1
            output_parts.append(remaining)

            import time
            print()
            print()
            print("=" * 80)
            print()
            print("INTERLEAVED TEXT & TIMESERIES PROMPT >>")
            for o in output_parts:
                print(o)
                if ascii_art:
                    time.sleep(1)
            print()
            print("=" * 80)
            print()
            print(f"ANSWER >> {pred_text}\n")

            with open(f"{viz_dir}/{name}_comparison.txt", "w") as f:
                f.write(f"=== Time Series Sample {count} ===\n")
                f.write(
                    f"Scores: WordOverlap={overlap_score:.4f} | ROUGE-L={rouge_score:.4f} | BLEU={bleu_score:.4f} | Char={char_score:.4f}\n\n"
                )
                f.write(f"--- PROMPT ---\n{tokenizer.decode(gen_inputs['text'][0], skip_special_tokens=False)}\n\n")
                f.write(f"--- GROUND TRUTH ---\n{gt_text}\n\n")
                f.write(f"--- PREDICTION ---\n{pred_text}\n")

    avg_overlap = total_overlap / count if count > 0 else 0
    avg_rouge = total_rouge / count if count > 0 else 0
    avg_bleu = total_bleu / count if count > 0 else 0
    avg_char = total_char / count if count > 0 else 0

    if visualize and viz_dir:
        with open(f"{viz_dir}/results_summary.txt", "w") as f:
            f.write("PRISM Time Series Interleaved QA Verification Results\n")
            f.write("=" * 40 + "\n")
            f.write(f"Samples: {count}\n\n")
            f.write("Average Scores:\n")
            f.write(f"  Word Overlap (Jaccard): {avg_overlap:.4f}\n")
            f.write(f"  ROUGE-L (F1):           {avg_rouge:.4f}\n")
            f.write(f"  BLEU-4:                 {avg_bleu:.4f}\n")
            f.write(f"  Char Similarity:        {avg_char:.4f}\n")

    return {
        "count": count,
        "avg_overlap": avg_overlap,
        "avg_rouge": avg_rouge,
        "avg_bleu": avg_bleu,
        "avg_char": avg_char,
    }


def main():
    parser = argparse.ArgumentParser(description="PRISM Universal Evaluator")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["inspect_train", "inspect_eval", "run_eval", "verify_image", "verify_timeseries_scits", "verify_timeseries_interleave"],
        help="Operation mode",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Limit samples (default depends on mode)"
    )
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save outputs")
    parser.add_argument("--visualize", action="store_true", help="Enable visualization plotting")
    parser.add_argument(
        "--viz_dir", type=str, default=None, help="Directory to save visualizations"
    )
    parser.add_argument(
        "--exhaustive",
        action="store_true",
        help="Iterate ALL datasets individually (requires inspect_train)",
    )
    parser.add_argument(
        "--data-only",
        action="store_true",
        help="Skip model loading for faster data inspection (inspect_train only)",
    )
    parser.add_argument(
        "--modality", type=str, default=None, help="Filter by modality name (e.g. 'Vision')"
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default=None,
        help="Path to model configuration file (optional)",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="allenai/OLMo-7B-0724-hf",
        help="HuggingFace backbone model ID",
    )
    parser.add_argument(
        "--validation",
        action="store_true",
        help="Use held-out validation data instead of training data",
    )
    parser.add_argument(
        "--no-ascii",
        action="store_true",
        help="Disable ASCII art rendering of time series in verify_timeseries mode",
    )

    args = parser.parse_args()

    # Defaults
    if args.limit == 0:
        if args.mode == "inspect_train":
            args.limit = 5
        elif args.mode == "inspect_eval":
            args.limit = 1
        elif args.mode == "run_eval":
            args.limit = 100

    print_prism_banner("Universal Evaluator")

    # Setup Model
    if args.data_only and args.mode == "inspect_train":
        print("Running in DATA-ONLY mode. Skipping model load.")
        # Load only tokenizer (assuming backbone ID logic is consistent)
        # We'll just load the backbone tokenizer
        tokenizer = load_cached_tokenizer("allenai/OLMo-7B-0724-hf")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = None
        device = "cpu"  # or "cuda:0" if needed for collator
        if torch.cuda.is_available():
            device = "cuda"
    else:

        model, tokenizer, device = setup_model(args.checkpoint, 
                                               model_config=args.model_config, 
                                               backbone_id=args.backbone)

    # Dispatch
    if args.mode == "inspect_train":
        inspect_training_data(
            model,
            tokenizer,
            device,
            limit=args.limit,
            visualize=args.visualize,
            viz_dir=args.viz_dir,
            exhaustive=args.exhaustive,
            modality_filter=args.modality,
            checkpoint_path=args.checkpoint,
        )
    elif args.mode == "verify_image":
        if model is None:
            raise ValueError("verify_image requires checkpoint")
        verify_image_modality(
            model,
            tokenizer,
            device,
            limit=args.limit,
            visualize=args.visualize,
            viz_dir=args.viz_dir,
            use_validation=args.validation,
            checkpoint_path=args.checkpoint,
        )
    elif args.mode == "verify_timeseries_scits":
        if model is None:
            raise ValueError("verify_timeseries_scits requires checkpoint")
        verify_timeseries_scits(
            model,
            tokenizer,
            device,
            limit=args.limit,
            viz_dir=args.viz_dir,
            ascii_art=not args.no_ascii,
            checkpoint_path=args.checkpoint,
        )
    elif args.mode == "verify_timeseries_interleave":
            if model is None:
                raise ValueError("verify_timeseries_interleave requires checkpoint")
            verify_timeseries_interleave(
                model,
                tokenizer,
                device,
                limit=args.limit,
                viz_dir=args.viz_dir,
                ascii_art=not args.no_ascii,
                checkpoint_path=args.checkpoint,
            )
    elif args.mode == "inspect_eval":
        if model is None:
            raise ValueError("Cannot run inspect_eval without model.")
        inspect_eval_examples(
            model, tokenizer, device, limit=args.limit, modality_filter=args.modality
        )
    elif args.mode == "run_eval":
        if model is None:
            raise ValueError("Cannot run run_eval without model.")
        run_full_eval(model, tokenizer, device, limit=args.limit)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

if __name__ == "__main__":
    main()
