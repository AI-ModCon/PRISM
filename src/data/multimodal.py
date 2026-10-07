import csv
import io
import logging
import os
import random
import re
import time
import warnings

import datasets
import h5py
import numpy as np
import requests
import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer, TapasTokenizer

# Suppress Pandas/Transformers FutureWarnings (Tapas legacy code)
warnings.simplefilter(action="ignore", category=FutureWarning)
# Suppress PIL Palette transparency warnings (spamming logs)
warnings.filterwarnings("ignore", message="Palette images with Transparency")
# Suppress TorchGeometric import warnings if not used
warnings.filterwarnings("ignore", message="An issue occurred while importing 'torch-cluster'")
warnings.filterwarnings("ignore", message="An issue occurred while importing 'torch-sparse'")
import ast

from ..config import DYNAMIC_LENGTH_TS_PROJECTORS
from ..modalities import Modality, make_dummy_batch, parse_modality
from .dataset_manager import DatasetManager
from .image_transforms import build_image_transform

# Configure Logger
logger = logging.getLogger(__name__)

# Fraction of max_seq_length reserved for prompt/target text when capping the
# number of interleaved time-series variates (issue #120). The TS spans get
# `max_seq_length - max_seq_length // DIVISOR` of the budget; the rest is
# headroom for text so total merged length stays within max_seq_length. 4 =>
# reserve 25%. Conservative: realistic ts_qa text is a few hundred tokens.
_TS_QA_TEXT_RESERVE_DIVISOR = 4

# WebDataset for TAR shard streaming
try:
    import webdataset as wds

    HAS_WEBDATASET = True
except ImportError:
    wds = None
    HAS_WEBDATASET = False
    logger.warning("webdataset not installed. TAR shard loading disabled.")

# Multi-dataset WebDataset loader
try:
    from .multi_webdataset import MultiWebDataset, load_daos_config

    HAS_MULTI_WEBDATASET = True
except ImportError:
    MultiWebDataset = None
    load_daos_config = None
    HAS_MULTI_WEBDATASET = False

try:

    # Fix: truncated images
    from PIL import Image, ImageFile

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    from torchvision import transforms
except ImportError:
    Image = None
    transforms = None

try:
    from rdkit import Chem
except ImportError:
    Chem = None


def _get_modality(dataset_info: dict, dataset_name: str = "<unknown>") -> Modality:
    """Read explicit `modality` field from a dataset_info dict and coerce to enum."""
    raw = dataset_info.get("modality")
    if not raw:
        raise KeyError(
            f"Dataset {dataset_name!r} (handler={dataset_info.get('handler')!r}) "
            f"has no 'modality' field. Add 'modality': '<image|text|table|"
            f"time_series|geometry|graph>' to its entry in "
            f"src/data/datasets_config.json."
        )
    return parse_modality(raw)


def clean_kegg_answer(answer: str) -> str:
    """Clean a wanglab/kegg answer string.

    Extracted so _process_dna_bioreason and _build_class_weights can't silently
    drift apart (both previously inlined this exact regex separately). Matches
    eval_kegg.py's strip_punct(ground_truth.lower()): removes punctuation
    (including apostrophes), lowercases, strips whitespace.
    e.g. "Parkinson's disease" -> "parkinsons disease"
    """
    return re.sub(r"[^\w\s]", "", answer.strip().lower()).strip()


def clean_variant_effect_coding_answer(answer: str) -> str:
    """Clean a wanglab/variant_effect_coding answer string.

    Mirrors BioReason/bioreason/dataset/variant_effect.py:clean_variant_effect_example,
    fixing a bug in BioReason's own training path where this cleaning was only
    ever applied to a throwaway copy used to build the label vocabulary — the
    real training answer there was left raw/uncleaned (see train_dna_qwen.py:415-416).
    Here it is applied to the actual answer used for training.

    Answers are always "Pathogenicity; free-text description" (confirmed against
    1000 real rows of wanglab/variant_effect_coding — 0 rows lacked a ';'), so we
    keep only the pathogenicity clause and normalize case/whitespace.
    e.g. "Pathogenic; Renal tubular epithelial cell apoptosis" -> "pathogenic"
    """
    return answer.split(";")[0].strip().lower()


def clean_variant_effect_non_snv_answer(answer: str) -> str:
    """Clean a wanglab/variant_effect_non_snv answer string.

    Mirrors BioReason/bioreason/dataset/variant_effect.py:clean_variant_effect_non_snv_example
    exactly (this cleaning *is* correctly applied to the real training answer in
    BioReason's own code, train_dna_qwen.py:446).

    Answers are either a bare pathogenicity ("benign"/"pathogenic", 228/1000 real
    rows sampled) or "pathogenicity; ['term_one', 'term_two']" (772/1000 rows;
    0 rows had a ';' without a matching '[...]' list) — this function is a
    no-op on the bare-pathogenicity case since there is nothing to strip.
    e.g. "pathogenic; ['Congenital_myasthenic_syndrome_8']"
         -> "pathogenic; Congenital myasthenic syndrome 8"
    e.g. "benign" -> "benign"
    """
    return answer.replace("[", "").replace("]", "").replace("'", "").replace("_", " ").strip()


class StreamingMultimodalDataset(IterableDataset):
    def __init__(
        self,
        tokenizer,
        batch_size=1,
        max_steps=1000,
        zone="zone_a",
        hf_token=None,
        allow_dummy_data=False,
        force_streaming=False,
        verbosity=logging.INFO,
        model_config=None,
        dataset_overrides: dict | None = None,
        max_seq_length: int | None = None,
        use_reasoning_traces: bool = True,
        model_name: str = "dna-llm",
        is_sft: bool = True,
        task: str = "vlm",
        use_class_weights: bool = False,
        class_weight_max: float = 10.0,
        is_projector_only: bool = False,
    ):
        """
        dataset_overrides: optional per-dataset field overrides applied AFTER
        the JSON config is loaded but BEFORE the strict skip=False validator
        runs. Plumbed in by Phase 2 plan §2.3 so the per-modality smoke yamls
        (`src/conf/data/per_modality_smoke/*.yaml`) can flip `skip: false`
        for a single dataset per run without touching `datasets_config.json`.
        Schema: {dataset_name: {field_name: value, ...}}. Unknown dataset
        names are logged and ignored.
        """
        # Set Logger Verbosity
        logger.setLevel(verbosity)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("[%(levelname)s] %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)

            handler = logging.StreamHandler()
            formatter = logging.Formatter("[%(levelname)s] %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_steps = max_steps
        # Merged-sequence budget for the interleaved ts_qa variate cap (issue
        # #120). None disables the cap (all variates kept). Set from the
        # launcher's MAX_SEQ_LENGTH via train.py.
        self.max_seq_length = max_seq_length
        self.hf_token = hf_token
        self.allow_dummy_data = allow_dummy_data
        self.force_streaming = force_streaming
        self.use_reasoning_traces = use_reasoning_traces
        self.model_name = model_name.lower()
        self.is_sft = is_sft
        self.task = task
        self.use_class_weights = use_class_weights
        self.class_weight_max = class_weight_max
        # When True, DNA items emit the "[dna_bioreason]" metadata sentinel
        # (model.py: full causal-LM loss over the merged text, no prompt/target
        # split) for the projector-alignment stage. When False, is_sft selects
        # between the normal SFT promptlen/targetlen split and the RL/GRPO
        # informational-only metadata (see _process_dna_bioreason).
        self.is_projector_only = is_projector_only
        # class_weights_map: {answer_label_lower: scalar_weight} — populated below
        # for task="bioreason_sft" when use_class_weights=True, empty otherwise.
        self.class_weights_map: dict = {}

        # NT tokenizer for DNA-LLM mode — loaded once here, not per item.
        # LLM mode inlines sequences as plain text and never needs this.
        self.dna_tokenizer = None
        if self.model_name == "dna-llm":
            try:
                self.dna_tokenizer = AutoTokenizer.from_pretrained(
                    "InstaDeepAI/nucleotide-transformer-v2-250m-multi-species",
                    trust_remote_code=True,
                )
                logger.info("DNA tokenizer loaded: nucleotide-transformer-v2-250m-multi-species")
            except Exception as e:
                logger.warning(f"Could not load DNA tokenizer: {e}")

        # Load Config
        self.manager = DatasetManager()
        self.datasets_map = self.manager.get_zone_config(zone).get("datasets", {})

        # Apply per-run dataset overrides (Phase 2 plan §2.3).
        # Per-modality smoke yamls flip `skip: false` for one dataset per run
        # without modifying the global datasets_config.json. Applied here so
        # the downstream activation loop, strict validator, and active-modality
        # computation all see the override.
        if dataset_overrides:
            unknown = []
            for name, fields in dataset_overrides.items():
                if name not in self.datasets_map:
                    unknown.append(name)
                    continue
                if not isinstance(fields, dict):
                    logger.warning(
                        f"dataset_overrides[{name!r}]: expected dict, got {type(fields).__name__}; skipping"
                    )
                    continue
                self.datasets_map[name].update(fields)
                logger.info(f"dataset_overrides applied: {name} <- {fields}")
            if unknown:
                logger.warning(
                    f"dataset_overrides referenced unknown datasets (ignored): {unknown}"
                )

        # Computed here (after dataset_overrides so a smoke-test override that
        # flips a dna_bioreason* dataset's skip flag is reflected in which
        # datasets get counted) so _build_class_weights can check which
        # dna_bioreason* datasets are actually active/not skipped instead of
        # hardcoding dataset ids blindly.
        if use_class_weights and task in ("bioreason_sft",):
            self._build_class_weights(class_weight_max)

        # Model dependent modality-specific constants
        # for data processing (e.g., max modality seq lengths).
        # Initialized here (before the activation block) so we can scope
        # activation to the model's configured modalities.
        from src.config import ModelConfig

        self.model_config = model_config if model_config is not None else ModelConfig()

        # When allow_dummy_data=True (set via ENABLE_ALL_MODALITIES=1), activate every
        # skip=true dataset whose modality is in `model.modalities`, so the model's
        # full modality set has at least one stream producing data — real where
        # present, dummy-fallback where not. This makes
        # `model.modalities=[text,image,table,time_series,geometry,graph]` work without
        # needing every dataset's local shards to exist. Datasets whose modality is
        # NOT in `model.modalities` are left skipped — flipping them would waste
        # init time loading streams the model can't consume.
        if self.allow_dummy_data:
            try:
                wanted = {parse_modality(m) for m in self.model_config.modalities}
            except (ValueError, TypeError) as e:
                logger.warning(
                    f"allow_dummy_data=True but could not parse model.modalities "
                    f"({self.model_config.modalities!r}): {e}. Activating all skipped datasets."
                )
                wanted = None
            activated = []
            skipped_irrelevant = []
            for name, info in self.datasets_map.items():
                if not info.get("skip", False):
                    continue
                if wanted is not None:
                    try:
                        m = _get_modality(info, name)
                    except KeyError:
                        # Missing modality field — can't decide, leave skipped.
                        continue
                    if m not in wanted:
                        skipped_irrelevant.append(name)
                        continue
                info["skip"] = False
                info["fallback_dummy"] = True
                activated.append(name)
            if activated:
                logger.info(
                    f"allow_dummy_data=True: activated {len(activated)} skipped datasets "
                    f"with fallback_dummy=true: {activated[:5]}{'...' if len(activated) > 5 else ''}"
                )
            if skipped_irrelevant:
                logger.info(
                    f"allow_dummy_data=True: left {len(skipped_irrelevant)} skipped "
                    f"datasets untouched (modality not in model.modalities): "
                    f"{skipped_irrelevant[:5]}{'...' if len(skipped_irrelevant) > 5 else ''}"
                )

        self.streams = {}
        self.datasets_objs = {}  # Store IterableDataset objects for restarting
        self.load_status = {}

        self.image_transform = build_image_transform(self.model_config)

        # Tapas Tokenizer for Tables
        try:
            self.tapas_tokenizer = TapasTokenizer.from_pretrained("google/tapas-base")
        except Exception:
            self.tapas_tokenizer = None
            logger.warning(
                "Warning: Could not load Tapas Tokenizer. Tables will fallback to dummy."
            )

        self.handlers_map = {
            "image_webdataset": self._process_image,
            "image_generic": self._process_image,
            "image_pixmo": self._process_image_pixmo,
            "image_points": self._process_image_points,
            "text_sft": self._process_text_sft,
            "table_pubtables": self._process_table,
            "table_generic": self._process_table,
            "geo_objaverse": self._process_geo,
            "geometry_generic": self._process_geo,
            "graph_moltextnet": self._process_graph,
            "graph_generic": self._process_graph,
            "graph_captioning": self._process_graph_captioning,
            "graph_grounding": self._process_graph_grounding,
            "graph_instruction": self._process_graph_instruction,
            "ts_qa": self._process_ts_qa,
            "ts_caption": self._process_ts_caption,
            "ts_instruction": self._process_ts_instruction,
            # "ts_weak": self._process_ts
        }

        logger.info(f"Initializing Streaming Datasets for {zone}...")

        for modality, info in self.datasets_map.items():
            name = info["name"]

            # Check Skip
            if info.get("skip", False):
                logger.warning(
                    f"Skipping {name} ({info.get('skip_reason', 'Configured skip')})"
                )
                # Do NOT add to self.streams if skipped
                self.load_status[name] = f"Skipped ({info.get('skip_reason')})"
                continue

            try:
                required_keys = info.get("required_keys", [])
                hf_id = info.get("hf_id", "")

                if "h5_direct" in required_keys:
                    logger.info(f"Initializing Physics Stream: {hf_id}")
                    stream = self._create_physics_stream(
                        hf_id, local_path=info.get("local_path")
                    )
                    if stream:
                        self.datasets_objs[modality] = stream
                        self.load_status[name] = "Active"
                    else:
                        logger.warning("Skipping Physics: Initialized stream was None.")
                        self.load_status[name] = "Skipped (Stream Init Failed)"

                elif "real_microct" in required_keys:
                    logger.info("Initializing Real Micro-CT Stream (Berea)...")
                    stream = self._create_real_microct_stream(local_path=info.get("local_path"))
                    # Strict Check: If stream is empty/None, do not register it
                    # We can't easily check 'empty' on a generator without consuming,
                    # but _create_real_microct_stream should return None if checks fail.
                    if stream:
                        self.datasets_objs[modality] = stream
                        self.load_status[name] = "Active"
                    else:
                        logger.warning(
                            "Skipping Berea: Stream creation failed (likely missing local file)."
                        )
                        self.load_status[name] = "Skipped (Missing Local NPZ)"


                elif "PDEBench" in name:
                    # PDEBench Special (Config-based)
                    lp = info.get("local_path")
                    if lp and os.path.exists(lp) and len(os.listdir(lp)) > 0:
                        logger.info(f"Loading PDEBench Local: {lp}")
                        try:
                            dataset = datasets.load_from_disk(lp)
                            if hasattr(dataset, "to_iterable_dataset"):
                                dataset = dataset.to_iterable_dataset()
                        except Exception as e:
                            logger.warning(f"Failed to load PDEBench local: {e}")
                            dataset = None
                    else:
                        if self.force_streaming:
                            dataset = self._load_dataset_safe(
                                hf_id,
                                "Advection_Sols_beta0.1",
                                split="train",
                                streaming=True,
                                token=self.hf_token,
                            )
                        else:
                            logger.warning(
                                "Skipping PDEBench: Local path empty and streaming not forced."
                            )
                            dataset = None

                    if dataset:
                        # shuffled = dataset.shuffle(buffer_size=10000, seed=random.randint(0, 10000))
                        # self.datasets_objs[modality] = shuffled
                        # self.streams[modality] = iter(shuffled)
                        self.datasets_objs[modality] = dataset  # Store raw
                    else:
                        logger.warning("Skipping PDEBench: Not loaded (Strict Local or Empty).")
                        self.load_status[name] = "Skipped (Missing Local Arrow)"

                else:
                    # 1. Try Generic Local Loading First (Overrides Custom Handlers like ChEBI)
                    local_path = info.get("local_path")
                    preferred_source = info.get(
                        "preferred_source", "local"
                    ).lower()  # Configurable Priority
                    logger.info(
                        f"[DATA CONFIG] Dataset: {name} | Priority: {preferred_source.upper()}"
                    )

                    # Force Remote if preferred
                    if self.force_streaming or preferred_source == "remote":
                        local_path = None

                    debug_log_path = "debug_data_load.txt"

                    def log_debug(msg, path=debug_log_path):
                        with open(path, "a") as f:
                            f.write(msg + "\n")
                        print(msg, flush=True)

                    loaded_local = False
                    if local_path and os.path.exists(local_path):
                        logger.debug(
                            f"Checking Local Path for {name}: {local_path} (Exists: {os.path.exists(local_path)})"
                        )

                        if os.path.isdir(local_path):
                            import glob

                            # CHECK 0: Multi-dataset mode from DAOS
                            # launch_aurora_daos.py sets USE_MULTI_DATASET=1 for multi-dataset mode
                            use_multi_dataset = (
                                os.environ.get("USE_MULTI_DATASET", "0") == "1"
                            )
                            daos_mount = os.environ.get("DAOS_MOUNT")
                            dataset_groups = os.environ.get("DATASET_GROUPS", "all")
                            dataset_config = os.environ.get("DATASET_CONFIG")

                            if (
                                use_multi_dataset
                                and daos_mount
                                and HAS_MULTI_WEBDATASET
                                and HAS_WEBDATASET
                            ):
                                # Multi-dataset mode: load from multiple DAOS directories
                                logger.info(
                                    "[DATA LOAD] Using Multi-Dataset Mode from DAOS"
                                )
                                logger.info(f"[DATA LOAD] DAOS Mount: {daos_mount}")
                                logger.info(
                                    f"[DATA LOAD] Dataset Groups: {dataset_groups}"
                                )
                                log_debug(
                                    f"DEBUG: Loading multi-dataset from DAOS with groups: {dataset_groups}"
                                )

                                try:
                                    config = load_daos_config(dataset_config)
                                    groups = (
                                        dataset_groups.split(",")
                                        if "," in dataset_groups
                                        else dataset_groups
                                    )

                                    # Parse proportion overrides from environment
                                    # Format: "dataset1:0.1,dataset2:0.2"
                                    proportion_overrides = {}
                                    proportions_str = os.environ.get(
                                        "DATASET_PROPORTIONS", ""
                                    )
                                    if proportions_str:
                                        for pair in proportions_str.split(","):
                                            if ":" in pair:
                                                ds_name, prop = pair.split(":", 1)
                                                try:
                                                    proportion_overrides[
                                                        ds_name.strip()
                                                    ] = float(prop.strip())
                                                except ValueError:
                                                    logger.warning(
                                                        f"Invalid proportion value: {pair}"
                                                    )
                                        if proportion_overrides:
                                            logger.info(
                                                f"[DATA LOAD] Proportion overrides: {proportion_overrides}"
                                            )

                                    # Get distributed training info for proper shard splitting
                                    import torch.distributed as dist

                                    if dist.is_initialized():
                                        world_size = dist.get_world_size()
                                        rank = dist.get_rank()
                                    else:
                                        world_size = 1
                                        rank = 0

                                    multi_ds = MultiWebDataset(
                                        config=config,
                                        groups=groups,
                                        daos_mount=daos_mount,
                                        proportion_overrides=proportion_overrides
                                        if proportion_overrides
                                        else None,
                                        world_size=world_size,
                                        rank=rank,
                                    )

                                    # Log stats
                                    stats = multi_ds.get_stats()
                                    logger.info(
                                        f"[DATA LOAD] Loaded {stats['num_datasets']} datasets, {stats['total_samples']} total samples"
                                    )

                                    dataset = multi_ds._dataset
                                    loaded_local = True
                                    self.datasets_objs[modality] = dataset
                                    self.load_status[name] = (
                                        f"Active (Multi-WebDataset: {stats['num_datasets']} datasets)"
                                    )
                                except Exception as e:
                                    logger.error(
                                        f"[DATA LOAD] Multi-dataset load failed: {e}"
                                    )
                                    log_debug(f"DEBUG: Multi-dataset error: {e}")
                                    # Fall through to single-dataset loading

                            # CHECK 0.5: Environment variable override for staged shards (legacy single-dataset)
                            # launch_aurora_web.py sets this to point to /tmp/webdataset.
                            #
                            # Phase 3 leak fix: gate the override on modality. The launcher stages
                            # ONE webdataset directory (typically image shards via --webdataset-dir),
                            # but earlier this block applied the override to ANY dataset reaching
                            # this code path — so a per-modality sweep cell that loaded ts_qa or
                            # graph_captioning would silently train on image shards. The PRISM-MODALITY-SMOKE
                            # validation in PR #71 surfaced exactly this (3 of 4 non-image cells
                            # reported `mismatch=True` via the Phase 1 startup check).
                            #
                            # Only honor the override for image datasets unless the operator
                            # explicitly set WEBDATASET_LOCAL_MODALITY to broaden it (escape hatch
                            # for hand-built staging directories that genuinely hold non-image data).
                            if not loaded_local:
                                webdataset_local_override = os.environ.get("WEBDATASET_LOCAL_PATH")
                                ds_modality = _get_modality(info, modality)
                                allowed_override_modality = os.environ.get(
                                    "WEBDATASET_LOCAL_MODALITY", "image"
                                )
                                if (
                                    webdataset_local_override
                                    and os.path.isdir(webdataset_local_override)
                                    and str(ds_modality) == allowed_override_modality
                                ):
                                    base_path = webdataset_local_override
                                    logger.info(
                                        f"[DATA LOAD] Using WEBDATASET_LOCAL_PATH override: {base_path} "
                                        f"(modality={ds_modality})"
                                    )
                                elif webdataset_local_override and str(ds_modality) != allowed_override_modality:
                                    logger.info(
                                        f"[DATA LOAD] Ignoring WEBDATASET_LOCAL_PATH override for {name}: "
                                        f"dataset modality={ds_modality} != staged modality={allowed_override_modality}. "
                                        f"Falling back to local_path={local_path}"
                                    )
                                    base_path = local_path
                                else:
                                    base_path = local_path

                                # Try manifest.json first (fast path)
                                # PREFER local_manifest.json if it exists (created by stage_shards.py)
                                # This contains only the shards staged to this node
                                local_manifest_path = os.path.join(base_path, "local_manifest.json")
                                manifest_path = os.path.join(base_path, "manifest.json")
                                shards_dir = os.path.join(base_path, "shards")
                                tar_files = []

                                # Try local_manifest.json first (node-specific staged shards)
                                if os.path.exists(local_manifest_path):
                                    try:
                                        import json as json_module

                                        with open(local_manifest_path) as mf:
                                            local_manifest = json_module.load(mf)
                                        # local_manifest has shards as plain strings (not dicts)
                                        if "shards" in local_manifest and isinstance(
                                            local_manifest["shards"], list
                                        ):
                                            shard_names = local_manifest["shards"]
                                            # Shards are in base_path directly (not shards/ subdir)
                                            tar_files = [
                                                os.path.join(base_path, name)
                                                for name in shard_names
                                            ]
                                            logger.info(
                                                f"[DATA LOAD] Loaded {len(tar_files)} shards from local_manifest.json (node-staged)"
                                            )
                                    except Exception as e:
                                        logger.warning(
                                            f"[DATA LOAD] Failed to read local_manifest: {e}, trying manifest.json"
                                        )
                                        tar_files = []

                                # Fall back to full manifest.json
                                if not tar_files and os.path.exists(manifest_path):
                                    try:
                                        import json as json_module

                                        with open(manifest_path) as mf:
                                            manifest = json_module.load(mf)
                                        # Extract shard names from manifest
                                        if "shards" in manifest and isinstance(
                                            manifest["shards"], list
                                        ):
                                            shard_names = [s["name"] for s in manifest["shards"]]
                                            if os.path.isdir(shards_dir):
                                                tar_files = [
                                                    os.path.join(shards_dir, name)
                                                    for name in shard_names
                                                ]
                                            else:
                                                tar_files = [
                                                    os.path.join(base_path, name)
                                                    for name in shard_names
                                                ]
                                            logger.info(
                                                f"[DATA LOAD] Loaded {len(tar_files)} shards from manifest (fast path)"
                                            )
                                    except Exception as e:
                                        logger.warning(
                                            f"[DATA LOAD] Failed to read manifest: {e}, falling back to glob"
                                        )
                                        tar_files = []

                                # Fallback to glob if manifest didn't work
                                if not tar_files:
                                    if os.path.isdir(shards_dir):
                                        tar_files = sorted(
                                            glob.glob(os.path.join(shards_dir, "*.tar"))
                                        )
                                    else:
                                        tar_files = sorted(
                                            glob.glob(os.path.join(base_path, "*.tar"))
                                        )
                                    if tar_files:
                                        logger.warning(
                                            f"[DATA LOAD] Using glob (slow path): {len(tar_files)} shards"
                                        )

                                # Lazy retry: webdataset may have been pip-installed after Python started
                                # Use globals() dict to avoid SyntaxError from Python 3.10's
                                # stricter "global must precede use" rule (HAS_WEBDATASET is read earlier in __init__)
                                if tar_files and not HAS_WEBDATASET:
                                    try:
                                        import webdataset as _wds_retry

                                        globals()["wds"] = _wds_retry
                                        globals()["HAS_WEBDATASET"] = True
                                        logger.info(
                                            "[DATA LOAD] webdataset imported on retry (installed at runtime)"
                                        )
                                    except ImportError:
                                        pass

                                if tar_files and HAS_WEBDATASET:
                                    logger.info(
                                        f"[DATA LOAD] Source: WebDataset TAR | Path: {local_path} ({len(tar_files)} shards)"
                                    )
                                    log_debug(
                                        f"DEBUG: Loading {len(tar_files)} TAR shards via WebDataset..."
                                    )

                                    # Create WebDataset pipeline
                                    # Split shards for this rank BEFORE creating WebDataset
                                    # This avoids issues with nodesplitter + DataLoader workers
                                    import torch.distributed as dist

                                    if dist.is_initialized():
                                        wds_rank = dist.get_rank()
                                        wds_world_size = dist.get_world_size()
                                    else:
                                        wds_rank = 0
                                        wds_world_size = 1

                                    # Manually split shards for this rank
                                    if wds_world_size > 1:
                                        tar_files_for_rank = tar_files[wds_rank::wds_world_size]
                                    else:
                                        tar_files_for_rank = tar_files

                                    if tar_files_for_rank:
                                        logger.info(
                                            f"[DATA LOAD] Rank {wds_rank}: Using {len(tar_files_for_rank)}/{len(tar_files)} shards"
                                        )
                                    else:
                                        logger.warning(
                                            f"[DATA LOAD] Rank {wds_rank}: No shards assigned! Total shards: {len(tar_files)}, world_size: {wds_world_size}"
                                        )

                                    # Create WebDataset with identity nodesplitter (we handle splitting above)
                                    # We pass a lambda that returns shards unchanged to satisfy WebDataset's
                                    # multi-node detection while avoiding double-splitting
                                    dataset = (
                                        wds.WebDataset(
                                            tar_files_for_rank
                                            if tar_files_for_rank
                                            else tar_files[
                                                :1
                                            ],  # Fallback to 1 shard to avoid empty
                                            shardshuffle=True,
                                            empty_check=False,  # Don't error on empty - let training handle it
                                            nodesplitter=lambda src: src,  # Identity function - shards already split
                                        )
                                        .shuffle(1000)
                                        .decode("pil")  # Decode images as PIL
                                        .to_tuple(
                                            "jpg;png;jpeg;webp;gif", "txt", "json"
                                        )  # Extract image, caption, metadata (including GIF)
                                        .map(
                                            lambda x: {
                                                "image": x[0],  # PIL Image
                                                "caption": x[1]
                                                if isinstance(x[1], str)
                                                else x[1].decode("utf-8"),
                                                "metadata": x[2] if isinstance(x[2], dict) else {},
                                            }
                                        )
                                    )
                                    loaded_local = True
                                    self.datasets_objs[modality] = dataset
                                    self.load_status[name] = "Active (WebDataset TAR)"

                            # CHECK 2: Arrow files (fallback) - only if WebDataset didn't load
                            if not loaded_local:
                                arrow_files = glob.glob(
                                    os.path.join(local_path, "*.arrow")
                                )
                                if arrow_files:
                                    logger.info(
                                        f"[DATA LOAD] Source: LOCAL (Raw Arrow) | Path: {local_path} ({len(arrow_files)} shards)"
                                    )
                                    log_debug(
                                        f"DEBUG: Loading {len(arrow_files)} Arrow files directly..."
                                    )
                                    dataset = load_dataset(
                                        "arrow",
                                        data_files=arrow_files,
                                        split="train",
                                        streaming=True,
                                    )

                                    if hasattr(dataset, "to_iterable_dataset"):
                                        dataset = dataset.to_iterable_dataset()

                                    self.datasets_objs[modality] = dataset
                                    self.load_status[name] = "Active (Local Arrow)"
                                    loaded_local = True

                            # CHECK 2 (cont.): Raw CSV/JSONL files inside the directory.
                            # This block used to live under a structurally-unreachable
                            # `elif os.path.isdir(local_path):` after this `if` exited
                            # without break/continue/return — so every dataset whose
                            # local_path was a directory full of CSV/JSONL files fell
                            # through to the HF remote loader and died on offline nodes.
                            # See plan: per-modality sweep dispatch fix.
                            if not loaded_local:
                                files = os.listdir(local_path)
                                csv_files = [
                                    os.path.join(local_path, f) for f in files if f.endswith(".csv")
                                ]
                                # Skip the WebDataset manifest files — they look like .json
                                # but describe shard layout, not training data. Without this
                                # filter, a WebDataset directory whose tar loader was disabled
                                # (e.g. `webdataset` not installed) silently falls into the
                                # JSONL branch and loads the manifest as data, which produces
                                # IndexError at iteration time.
                                _MANIFEST_NAMES = {"manifest.json", "local_manifest.json"}
                                json_files = [
                                    os.path.join(local_path, f)
                                    for f in files
                                    if (f.endswith(".jsonl") or f.endswith(".json"))
                                    and f not in _MANIFEST_NAMES
                                ]

                                if csv_files:
                                    logger.info(f"Loading Local CSVs: {csv_files}")
                                    load_kwargs = {}
                                    if (
                                        "ChEBI-20 (Global)" in name
                                        or "chebi" in name.lower()
                                    ):
                                        load_kwargs["sep"] = "\t"

                                    dataset = load_dataset(
                                        "csv",
                                        data_files=csv_files,
                                        split="train",
                                        streaming=True,
                                        **load_kwargs,
                                    )
                                    dataset.shuffle(buffer_size=10000, seed=random.randint(0, 10000))
                                    self.datasets_objs[modality] = dataset

                                    self.load_status[name] = "Active (Local CSV)"
                                    loaded_local = True

                                elif json_files:
                                    logger.info(f"Loading local JSONL: {json_files}")
                                    dataset = load_dataset(
                                        "json",
                                        data_files=json_files,
                                        split="train",
                                        streaming=True,
                                    )
                                    dataset.shuffle(buffer_size=10000, seed=random.randint(0, 10000))
                                    self.datasets_objs[modality] = dataset

                                    self.load_status[name] = "Active (Local JSONL)"
                                    loaded_local = True

                                else:
                                    # Recursive JSONL Search (e.g. TableInstruct/data_v3/*.json)
                                    rec_json_files = []
                                    for root, _, filenames in os.walk(local_path):
                                        if "eval_data" in root or "test" in root:
                                            continue

                                        for filename in filenames:
                                            if (
                                                filename.endswith(".jsonl") or filename.endswith(".json")
                                            ) and filename not in _MANIFEST_NAMES:
                                                rec_json_files.append(os.path.join(root, filename))

                                    if rec_json_files:
                                        logger.info(
                                            f"Loading local JSONL (Recursive): Found {len(rec_json_files)} files."
                                        )
                                        dataset = load_dataset(
                                            "json",
                                            data_files=rec_json_files,
                                            split="train",
                                            streaming=True,
                                        )
                                        dataset.shuffle(
                                            buffer_size=10000, seed=random.randint(0, 10000)
                                        )
                                        self.datasets_objs[modality] = dataset

                                        self.load_status[name] = "Active (Local JSONL Recursive)"
                                        loaded_local = True
                                    else:
                                        log_debug(
                                            f"DEBUG: No .tar, .arrow, .csv, or .json files in {local_path}"
                                        )
                                        dataset = None

                    if loaded_local:
                        continue

                    # 2. Fallbacks / Specific Handlers if Local Failed
                    # Short-circuit: if allow_dummy_data is set and this dataset
                    # is marked fallback_dummy (was originally skipped), use the
                    # dummy generator directly rather than attempting a remote HF
                    # load that will succeed lazily but fail (or stall) on iteration.
                    if self.allow_dummy_data and info.get("fallback_dummy", False):
                        self.datasets_objs[modality] = (
                            self._dummy_generator()
                            if modality != "time_series"
                            else self._ts_generator()
                        )
                        self.load_status[name] = "Active (Dummy - local unavailable)"
                        continue

                    logger.info(f"[DATA LOAD] Source: REMOTE (HF) | ID: {hf_id}")
                    if "chebi_20" in hf_id or "ChEBI-20" in name:
                        # Robust CSV loading for ChEBI fallback
                        self.streams[modality] = self._create_chebi_stream(hf_id)
                    else:
                        # Generic Remote Helper
                        # User Update: Enable lazy loading (streaming) globally for efficiency.
                        # preferred_source controls PRIORITY, but loading is always STREAMED.
                        kwargs = info.get("kwargs", {})
                        dataset = self._load_dataset_safe(
                            hf_id,
                            split=info.get("split", "train"),
                            streaming=True,
                            token=self.hf_token,
                            **kwargs
                        )
                        # shuffled = dataset.shuffle(buffer_size=10000, seed=random.randint(0, 10000))
                        self.datasets_objs[modality] = dataset  # Store raw
                        # self.streams[modality] = iter(shuffled)

                self.load_status[name] = "Active"

            except Exception as e:
                logger.error(f"Error loading {name}: {e}")
                print(
                    f"\n[CRITICAL ERROR] Failed to load dataset {name}: {e}\n",
                    flush=True,
                )
                if self.allow_dummy_data and info.get("fallback_dummy", False):
                    self.datasets_objs[modality] = (
                        self._dummy_generator()
                        if modality != "time_series"
                        else self._ts_generator()
                    )
                    # self.streams[modality] = ...
                    self.load_status[name] = f"Failed ({str(e)[:50]}...)"
                else:
                    self.load_status[name] = "Failed (Fatal)"

        # --- Pre-Calculate Active Modalities for External Access (Trainer) ---
        self.active_modalities = set()
        for dataset_name in self.datasets_objs.keys():
            if dataset_name not in self.datasets_map:
                continue
            self.active_modalities.add(
                _get_modality(self.datasets_map[dataset_name], dataset_name)
            )

        logger.info(f"[Init] Pre-Calculated Active Modalities: {list(self.active_modalities)}")

    def _load_dataset_safe(self, *args, **kwargs):
        """Wrapper for load_dataset with retry logic for 429 Rate Limits."""
        max_retries = 3
        base_wait = 5

        for i in range(max_retries + 1):
            try:
                return load_dataset(*args, **kwargs)
            except Exception as e:
                # Check for 429 or connection errors
                error_str = str(e)
                if "429" in error_str or "connection" in error_str.lower():
                    if i < max_retries:
                        wait = base_wait * (2**i) + random.uniform(0, 1)
                        logger.warning(
                            f"Dataset Load 429/Connection Error: {e}. Retrying in {wait:.1f}s..."
                        )
                        time.sleep(wait)
                        continue
                raise e

    # --- Generators ---
    def _ts_generator(self):
        while True:
            vals = torch.randn(64, 1)
            mean = vals.mean().item()
            std = vals.std().item()
            caption = f"Time series data with mean {mean:.2f} and standard deviation {std:.2f}."
            yield {"ts": vals, "caption": caption}

    def _dummy_generator(self, choice=None):
        while True:
            yield None

    # --- Processors ---
    def _process_image(self, item):
        # Image Processing (Real)
        default_tensor = torch.zeros(3, 224, 224)

        if item is None:
            if not self.allow_dummy_data:
                raise RuntimeError("Item is None in Image Stream")
            return default_tensor, "Dummy Image"

        # DEBUG KEYS
        # print(f"DEBUG Image Item Keys: {list(item.keys())}")
        # if random.random() < 0.05:
        #      print(f"DEBUG Image Item Keys: {list(item.keys())}", flush=True)
        # print(f"DEBUG Image Item Content (Partial): {str(item)[:200]}", flush=True)

        text = ""
        image_tensor = default_tensor

        try:
            # 1. Extract Text
            if "text" in item:
                text = item["text"]
            elif "caption" in item:
                text = item["caption"]
            elif "txt" in item:  # WebDataset
                text = (
                    item["txt"].decode("utf-8") if isinstance(item["txt"], bytes) else item["txt"]
                )

            # CRITICAL: Ensure text is never empty (causes NaN loss when all labels are -100)
            if not text or (isinstance(text, str) and not text.strip()):
                text = "Image."

            # 2. Extract Image
            img_obj = None
            if "image" in item and item["image"] is not None:
                img_obj = item["image"]

                # Case 1: Already a PIL Image (from WebDataset .decode("pil"))
                if isinstance(img_obj, Image.Image):
                    img_obj = img_obj.convert("RGB")  # Ensure RGB

                # Case 2: Dict format (HF format sometimes)
                elif isinstance(img_obj, dict):
                    if "bytes" in img_obj and img_obj["bytes"]:
                        img_obj = Image.open(io.BytesIO(img_obj["bytes"])).convert("RGB")
                    elif "path" in img_obj and img_obj["path"]:
                        if os.path.exists(img_obj["path"]):
                            img_obj = Image.open(img_obj["path"]).convert("RGB")
                        else:
                            img_obj = None  # Mark as not found

                # Case 3: Bytes
                elif isinstance(img_obj, bytes):
                    img_obj = Image.open(io.BytesIO(img_obj)).convert("RGB")

            elif "jpg" in item:  # WebDataset
                img_obj = Image.open(io.BytesIO(item["jpg"])).convert("RGB")
            elif "image_url" in item or "local_path" in item:
                try:
                    # 1. Prioritize 'local_path'
                    local_p = item.get("local_path")

                    # Validate local_path
                    if local_p and os.path.exists(local_p):
                        url_or_path = local_p
                    else:
                        # Fallback to URL only if local_path missing or doesn't exist
                        url_or_path = item.get("image_url")

                    # 2. Check Existence
                    if url_or_path and os.path.exists(url_or_path):
                        img_obj = Image.open(url_or_path).convert("RGB")

                    # 3. Error - Strict Mode
                    else:
                        if not self.allow_dummy_data:
                            raise RuntimeError(
                                f"Strict Mode: Local Image Not Found: {url_or_path}. Download Disabled."
                            )

                except Exception as e:
                    logger.warning(f"Image Load Failed: {e} - Skipping")
                    return None, "Skipped Item"

            if img_obj is None:
                if not self.allow_dummy_data:
                    raise RuntimeError(f"No image found in item. Keys: {list(item.keys())}")
            # 3. Transform
            if img_obj and self.image_transform:
                try:
                    if not isinstance(img_obj, Image.Image):
                        pass
                    image_tensor = self.image_transform(img_obj)
                except Exception as e:
                    logger.debug(f"Image Transform Error: {e}")
                    if not self.allow_dummy_data:
                        raise RuntimeError(f"Image Transform Error: {e}") from e
                    # Fallback if transform fails (e.g. truncated)
                    pass

        except Exception as e:
            if not self.allow_dummy_data:
                raise RuntimeError(
                    f"Image Process Error ({item.get('image_url', 'No URL')}): {e}"
                ) from e
            logger.warning(f"Image Process Error (Fallback Used): {e}")
            return default_tensor, text

        logger.debug(f"Image Tensor Stats: Min={image_tensor.min()}, Max={image_tensor.max()}")
        if image_tensor is default_tensor:
            logger.warning(
                f"Warning: Returning Default Image Tensor for item {item.get('id', 'unknown')}"
            )
        return image_tensor, text

    def _process_image_pixmo(self, item):
        # Dense Captioning
        tensor, _ = self._process_image(item)  # Reuse generic image loader
        if item is None:
            return tensor, "Dummy PixMo Caption", "[pixmo_cap] None"

        # PixMo usually has 'image' and 'caption'
        text = item.get("caption", item.get("text", "No caption"))

        # CRITICAL: Ensure text is never empty (causes NaN loss)
        if not text or (isinstance(text, str) and not text.strip()):
            logger.warning("Empty caption in pixmo_cap sample, using fallback")
            text = "An image."

        # Metadata
        meta = f"[pixmo_cap] {item.get('image_id', 'Unknown')}"

        return tensor, text, meta

    def _process_image_points(self, item):
        # Pointing Data
        if item is None:
            raise RuntimeError("Item is None in Image Handler (Strict)")

        # Points often come as list of [x, y] or string.
        if item is None:
            raise RuntimeError("Item is None in Image Handler (Strict)")

        # Points often come as list of [x, y] or string.
        # For projector alignment, we treat them as text tokens "point at (x,y)"
        points = item.get("points", [])
        label = item.get("label", "point")

        if isinstance(points, list):
            # Handle both [x, y] lists an {'x': x, 'y': y} dicts
            def fmt_p(p):
                if isinstance(p, list | tuple):
                    return f"({p[0]},{p[1]})"
                elif isinstance(p, dict):
                    return f"({p.get('x', 0)},{p.get('y', 0)})"
                return str(p)

            points_str = " ".join([fmt_p(p) for p in points])
        else:
            points_str = str(points)

        text = f"{label} {points_str}"

        # Handle Video Frames (Molmo2-VideoPoint)
        if "raw_frames" in item and len(item["raw_frames"]) > 0:
            try:
                # Take middle frame or first frame
                frames = item["raw_frames"]
                frame = frames[len(frames) // 2]
                if self.image_transform:
                    tensor = self.image_transform(frame)
                else:
                    tensor = transforms.ToTensor()(frame)
            except Exception as e:
                logger.debug(f"Video Frame Process Error: {e}")
                # Fallback: Create an image with the Video ID
                from PIL import ImageDraw

                img = Image.new("RGB", (224, 224), color=(73, 109, 137))
                d = ImageDraw.Draw(img)
                d.text(
                    (10, 10),
                    f"Video: {item.get('video_id', 'Unk')}",
                    fill=(255, 255, 0),
                )
                d.text((10, 30), "Frame Load Failed", fill=(255, 0, 0))

                if self.image_transform:
                    tensor = self.image_transform(img)
                else:
                    tensor = transforms.ToTensor()(img)
        else:
            # Fallback to generic image lookups (URL/JPG)
            tensor, _ = self._process_image(item)

        return tensor, text

    def _process_text_sft(self, item):
        # Language Preservation (Pure Text)
        # Returns Dummy Image (Zeros) + Text
        if item is None:
            return torch.zeros(3, 224, 224), "Dummy SFT Instruction"

        # Tulu structure: 'messages': [{'role': 'user', ...}, {'role': 'assistant', ...}]
        messages = item.get("messages", [])
        text = ""
        for m in messages:
            text += f"{m['role']}: {m['content']}\n"

        meta = f"[tulu_sft] {item.get('id', 'Unknown')}"
        return torch.zeros(3, 224, 224), text, meta

    def _process_ts_caption(self, item):
        """Process TSQA instruction data from ChengsenWang/TSQA, converting into (timeseries, text caption) pairs.

        Supports datasets with (Task, Question, Label, Series) fields, where Question/Label are text and Series is the time series data.

        Returns timeseries, caption, meta
        """
        if item is None:
            raise RuntimeError("Item is None in TS Caption (Strict)")

        # Definition mappings for each task type
        TREND_DEFINITIONS = {
            "constant trend": "a constant trend, where the time series does not show any significant increase or decrease over time",
            "upward trend": "an upward trend, where the time series consistently increases over time.",
            "downward trend": "a downward trend, where the time series consistently decreases over time.",
        }

        VOLATILITY_DEFINITIONS = {
            "constant volatility": "constant volatility, where the time series shows relatively consistent fluctuation magnitude throughout the period.",
            "increased volatility": "increased volatility, where the time series shows a rise in the magnitude of fluctuation over time.",
            "decreased volatility": "decreased volatility, where the time series shows a reduction in the magnitude of fluctuations over time.",
        }

        SEASON_DEFINITIONS = {
            "no seasonal": "no seasonal pattern, where the time series shows a repetitive and predictable fluctuation throughout the period.",
            "fixed seasonal": "a fixed seasonal pattern, where the timing and magnitude of the seasonal fluctuation remain constant over time.",
            "shifting seasonal": "a shifting seasonal pattern, where the timing or magnitude of the seasonal fluctuation changes over time.",
        }

        OUTLIER_DEFINITIONS = {
            "no outlier": "no data point that significantly differs from other observations in the time series.",
            "sudden spike": "a sudden spike, which is the rapid and significant increase in the value of a variable over a short period, followed by a return to the original baseline.",
            "level shift": "a level shift, which is the significant and sustained change in the average level of a time series.",
        }
        try:
            # Extract fields from item
            task = item.get("Task", item.get("task", "")).strip().lower()
            label = item.get(
                "Label", item.get("label", item.get("Answer", item.get("answer", "")))
            ).strip()
            series = item.get(
                "Series", item.get("series", item.get("ts", item.get("time_series", None)))
            )

            # Clean label
            label_clean = label.lower().strip()

            # Generate caption based on task type
            caption = ""
            if "trend" in task:
                definition = TREND_DEFINITIONS.get(label_clean, f"it exhibits a {label_clean}")
                caption = f"This timeseries has {definition}."

            elif "volatil" in task:
                definition = VOLATILITY_DEFINITIONS.get(
                    label_clean, f"the values show {label_clean} levels of variation"
                )
                caption = f"This timeseries has {definition}."

            elif "season" in task:
                definition = SEASON_DEFINITIONS.get(
                    label_clean, f"it follows a {label_clean} cycle"
                )
                caption = f"This timeseries has {definition}."

            elif "outlier" in task:
                definition = OUTLIER_DEFINITIONS.get(
                    label_clean, f"the data contains {label_clean} anomalous points"
                )
                caption = f"This timeseries has {definition}."

            else:
                # Fallback for unknown task types
                caption = f"This timeseries exhibits the following characteristic: {label}."
            caption += self.tokenizer.eos_token if self.tokenizer else ""

            # Process series data into tensor
            vals = []
            if series is not None:
                if isinstance(series, torch.Tensor):
                    vals = series.view(-1).tolist()
                elif isinstance(series, list | tuple):
                    vals = list(series)
                elif isinstance(series, str):
                    # parse a string [1.0, 2.0, 3.0]

                    # Try parsing string representation of list
                    try:
                        parsed = ast.literal_eval(series)
                        if isinstance(parsed, list | tuple):
                            vals = list(parsed)
                    except (ValueError, SyntaxError):
                        # Try splitting by common delimiters
                        for delimiter in [",", " ", "\t", ";"]:
                            try:
                                vals = [
                                    float(x.strip()) for x in series.split(delimiter) if x.strip()
                                ]
                                if len(vals) > 1:
                                    break
                            except ValueError:
                                continue
                elif isinstance(series, np.ndarray):
                    vals = series.flatten().tolist()

            if not vals:
                raise RuntimeError(
                    f"Empty or invalid Series in TS Caption. Keys: {list(item.keys())}"
                )

            # Convert to float tensor
            vals = [float(v) for v in vals]
            tensor = torch.tensor(vals, dtype=torch.float).view(-1, 1)

            is_dynamic_length_ts = (
                getattr(self.model_config, "ts_projector", "linear")
                in DYNAMIC_LENGTH_TS_PROJECTORS
            )
            # Truncate only (never pad) for dynamic-length encoders; they
            # track each sample's real length themselves.
            if tensor.shape[0] > self.model_config.max_ts_length:
                tensor = tensor[: self.model_config.max_ts_length, :]
            elif not is_dynamic_length_ts and tensor.shape[0] < self.model_config.max_ts_length:
                pad = torch.zeros(self.model_config.max_ts_length - tensor.shape[0], 1)
                tensor = torch.cat([tensor, pad], dim=0)

            # Metadata
            meta = f"[ts_caption] Task={task} Label={label_clean}"

            return tensor, caption, meta

        except Exception as e:
            logger.error(f"TS Caption Error: {e}")
            raise e

    def _process_ts_qa(self, item):

        # Time Series
        if item is None:
            raise RuntimeError("Item is None in TS QA (Strict)")
        try:
            # Robust Text Extraction for Instruction Tuning
            # Try specific instruction/response pairs first
            prompt = item.get("instruction", item.get("question", item.get("input", "")))
            target = item.get("output", item.get("answer", item.get("response", "")))

            # TS Reasoning Specifics
            if not (prompt or target):
                desc = item.get("description", "")
                chars = item.get("characteristics", "")
                if desc:
                    target = f"Description: {desc}\nCharacteristics: {chars}".strip()
                    prompt = "Describe this time series."

            if prompt or target:
                caption = f"{prompt}\nTarget: {target}".strip()  # noqa: F841
            else:
                # Fallback: parse WebDataset-format "text" field emitted by
                # convert_scits_to_webdataset._compose_text (and similar
                # converters) which writes "Question: X\nAnswer: Y" or plain
                # text into the "text" shard key.
                raw_text = item.get("text", item.get("caption", "")) or ""
                if raw_text:
                    if "Question:" in raw_text and "Answer:" in raw_text:
                        parts = raw_text.split("Answer:", 1)
                        prompt = parts[0].replace("Question:", "", 1).strip()
                        target = parts[1].strip()
                    else:
                        prompt = raw_text.strip()

            # Extract Series from dict/list
            vals = []

            # 1. Direct Keys
            for k in ["ts", "series", "timeseries", "time_series", "preds", "history"]:
                if k in item:
                    vals = item[k]
                    break

            # 2. Heuristic Search (List of Numbers)
            if not vals or vals is None:
                for _k, v in item.items():
                    if isinstance(v, list) and len(v) > 10 and isinstance(v[0], int | float):
                        vals = v
                        break

            if vals is None:
                vals = []

            # 3. Handle Types (str, Tensor, List)
            if isinstance(vals, torch.Tensor):
                vals = vals.view(-1).tolist()
            elif isinstance(vals, str):
                # Try parsing string list "[1.0, 2.0]"
                import ast

                try:
                    parsed = ast.literal_eval(vals)
                    if isinstance(parsed, list):
                        vals = parsed
                except Exception:
                    # Check for ### delimited (ChatTime)
                    if "###" in vals:
                        # It's likely a full text prompt. Parse numbers if possible or just use dummy numbers?
                        # ChatTime is TimeSeries-Text dataset.
                        # We need to extract the SERIES data if hidden in there.
                        # BUT usually ChatTime CSV has a separate series column??
                        # Debug script showed keys: ['text'].
                        # So series is NOT present as a separate column?
                        # Let's assume Dummy Series for ChatTime if no series column?
                        # Wait, if vals is the 'text' column, attempting to parse it as numbers will fail.
                        pass
                    vals = []

                # IMPORTANT: DO NOT OVERWRITE CAPTION IF IT EXISTS
                if "Parsed from Text" not in prompt:
                    # Only update if we actually parsed something useful or if caption was default?
                    pass
                else:
                    if "###" in vals:
                        try:
                            vals = [float(x) for x in vals.replace("###", " ").split() if x.strip()]
                        except Exception:
                            vals = []
                    else:
                        vals = []

            # Check if 'text' column contained the series (fallback)
            if not vals and isinstance(item.get("text"), str) and "###" in item.get("text"):
                raw_txt = item["text"]
                try:
                    vals = [float(x) for x in raw_txt.replace("###", " ").split() if x.strip()]
                    # Don't use the series string as caption
                    if len(vals) > 0:
                        pass
                        # caption = "Time Series Data (Parsed from Text)"
                except Exception:
                    pass

            if not vals:
                raise RuntimeError("Empty Series (Strict Mode)")
                # Fallback to random if empty (shouldn't happen with real data)
                # return torch.randn(512, 1), "Empty Series"

            # Tensorize — handle jagged multivariate series by padding to max length
            if isinstance(vals, list) and len(vals) > 0 and isinstance(vals[0], list):
                max_len = max(len(v) for v in vals)
                vals = [v + [0.0] * (max_len - len(v)) for v in vals]
            tensor = torch.tensor(vals, dtype=torch.float)

            # If completely random/dummy fallback in loop, make it a sine wave for viz
            if tensor.std() == 0 and "Dummy" in prompt:
                t_steps = torch.linspace(0, 4 * 3.14159, self.model_config.max_ts_length)
                tensor = torch.sin(t_steps).view(-1, 1)

            is_dynamic_length_ts = (
                getattr(self.model_config, "ts_projector", "linear")
                in DYNAMIC_LENGTH_TS_PROJECTORS
            )
            if is_dynamic_length_ts:
                # Dynamic patching path (SciTS / TimeOmni / intern_s2*):
                # each encoder handles its own per-sample lengths, so skip
                # dataset-level pad/truncate-to-max_ts_length entirely.

                if tensor.dim() == 1:
                    # Single-variate: reshape to (T, 1) — no normalize or pad/truncate
                    tensor = tensor.view(-1, 1)
                if tensor.dim() == 2:
                    # Legacy ts_qa records often encode multivariate data as
                    # (num_vars, steps). Normalize to (T, V) without relying
                    # on a fixed configured variate count.
                    if tensor.shape[0] == tensor.shape[1]:
                        # Square tensor: T == V, orientation is ambiguous.
                        # Assume (T, V) and leave as-is to avoid silent data
                        # corruption. Callers with square time series should
                        # ensure input is already oriented as (T, V).
                        logger.warning(
                            f"[TS_QA] Square tensor shape "
                            f"{list(tensor.shape)} — cannot determine "
                            f"(T, V) vs (V, T) orientation; assuming (T, V). "
                            "Ensure the input is already oriented as (T, V)."
                        )
                    elif tensor.shape[0] < tensor.shape[1]:
                        # Heuristic: (V, T) is more common than (T, V) when the
                        # first dim is smaller; transpose to canonical (T, V).
                        tensor = tensor.t().contiguous()
                    # Real per-sample length/truncation enforcement lives in
                    # the encoder itself (src/encoders/time_series.py:
                    # _forward_timeomni or _prepare_intern_s2_batch).

                    # Interleaved merge contracts in model.py currently allocate
                    # one fixed token budget per <ts><ts/> span. The dynamic
                    # path emits one feature sequence per sample, so enforce
                    # exactly one span here (insert one if missing).
                    if self.model_config.is_interleaved_qa:
                        indices = self.model_config.modality_start_end_token_indices
                        if not indices or "time_series" not in indices:
                            raise RuntimeError(
                                "In _process_ts_qa, model config missing "
                                "modality_start_end_token_indices for time_series "
                                "(required when is_interleaved_qa=True)"
                            )
                        ts_start_token = self.tokenizer.decode([indices["time_series"][0]])
                        ts_end_token = self.tokenizer.decode([indices["time_series"][1]])
                        span = f"{ts_start_token}{ts_end_token}"
                        span_count = prompt.count(span)
                        if span_count == 0:
                            prompt = f"{prompt.strip()} {span}".strip()
                        elif span_count != 1:
                            raise RuntimeError(
                                "[TS_QA] expected exactly one "
                                f"{span!r} span in prompt, got {span_count}"
                            )
            else:
                if tensor.dim() == 1:
                    # Single-variate: reshape to (T, 1)
                    tensor = tensor.view(-1, 1)
                    if not self.model_config.normalize_ts_in_encoder:
                        mean = tensor.mean()
                        std = tensor.std() if tensor.std() > 0 else 1.0
                        # clamp std to prevent div-by-zero or tiny values
                        std = max(std, 1e-6)
                        tensor = (tensor - mean) / std
                    # Pad/Truncate to self.model_config.max_ts_length
                    if tensor.shape[0] > self.model_config.max_ts_length:
                        tensor = tensor[: self.model_config.max_ts_length, :]
                    if tensor.shape[0] < self.model_config.max_ts_length:
                        pad = torch.zeros(self.model_config.max_ts_length - tensor.shape[0], 1)
                        tensor = torch.cat([tensor, pad], dim=0)
                elif tensor.dim() == 2:
                    # --- Variate cap (issue #120) ---
                    # Each variate row becomes one <ts> span that the model expands
                    # to `max_ts_length` embedding tokens in
                    # _merge_text_input_ids_with_modality_embeds. A sample with V
                    # variates merges to ~V * max_ts_length + text tokens. Oversized
                    # merged sequences exhaust the GPU: hardware-confirmed on Aurora
                    # that OLMo-1B fwd+bwd fits seq≈4096 on a 64GB XPU tile but OOMs
                    # by seq≈4608 (attention memory grows ~O(T²)). Under DDP+oneCCL
                    # that OOM surfaces as a GPU write page-fault (issue #120 reopen,
                    # crash at step 1040 with --max-seq-length 4096), not a clean
                    # OutOfMemoryError.
                    #
                    # The cap must bound TOTAL merged length by max_seq_length, so we
                    # budget the time-series tokens (kept * max_ts_length) against
                    # max_seq_length MINUS a reserve for the surrounding prompt/target
                    # text. Budgeting only the TS contribution (the original #120 fix)
                    # left merged length = TS + text > max_seq_length, and worse, went
                    # inert at --max-seq-length 4096 for 16-variate data
                    # (4096 // 256 = 16 = all variates kept) — exactly the reopen.
                    #
                    # Cap BEFORE means are computed so the per-variate <ts><ts/>
                    # pairs injected into the prompt below stay 1:1 with the
                    # surviving tensor rows — otherwise the merge invariant
                    # (#<ts> pairs == #rows) fails at model.py:609. Only meaningful
                    # in interleaved-QA mode (where the spans expand); a no-op
                    # otherwise. Floors at 1 variate (can't emit zero); a single
                    # variate whose span alone exceeds the budget is a misconfig
                    # (max_ts_length > max_seq_length) and is logged.
                    if (
                        self.model_config.is_interleaved_qa
                        and self.max_seq_length is not None
                        and self.model_config.max_ts_length > 0
                    ):
                        # Reserve headroom for the prompt/target text that sits on
                        # top of the TS spans. 1/4 of the budget is conservative:
                        # realistic ts_qa text (base prompt + per-variate Mean/Std
                        # stats + target) is a few hundred tokens, well under this.
                        text_reserve = self.max_seq_length // _TS_QA_TEXT_RESERVE_DIVISOR
                        ts_budget = self.max_seq_length - text_reserve
                        max_variates = max(
                            1, ts_budget // self.model_config.max_ts_length
                        )
                        if tensor.shape[0] > max_variates:
                            logger.warning(
                                f"[TS_QA] Capping variates {tensor.shape[0]} -> "
                                f"{max_variates} to keep total merged length "
                                f"(variates * max_ts_length + text) within "
                                f"max_seq_length={self.max_seq_length} "
                                f"(max_ts_length={self.model_config.max_ts_length}, "
                                f"text_reserve={text_reserve})"
                            )
                            tensor = tensor[:max_variates]
                            if (
                                max_variates * self.model_config.max_ts_length
                                >= self.max_seq_length
                            ):
                                logger.warning(
                                    "[TS_QA] Even 1 variate's span "
                                    f"({self.model_config.max_ts_length} tokens) "
                                    "meets/exceeds "
                                    f"max_seq_length={self.max_seq_length}; check config "
                                    "(max_ts_length should be << max_seq_length)."
                                )

                    # Per-variate normalize (skipped when the encoder will do it).
                    means = tensor.mean(dim=1)
                    stds = tensor.std(dim=1)
                    if not self.model_config.normalize_ts_in_encoder:
                        stds = torch.where(stds == 0, torch.ones_like(stds), stds)
                        stds = torch.clamp(stds, min=1e-6)
                        tensor = (tensor - means.unsqueeze(1)) / (stds.unsqueeze(1))

                    # Prompt mutation with <ts><ts/> envelope + Mean/Std stats is
                    # only meaningful in interleaved-QA mode (the special tokens
                    # must exist in the tokenizer and the model must be configured
                    # to read them). For non-interleaved configs, leave the prompt
                    # alone and just emit the flattened tensor.
                    if self.model_config.is_interleaved_qa:
                        indices = self.model_config.modality_start_end_token_indices
                        if not indices or "time_series" not in indices:
                            raise RuntimeError(
                                "In _process_ts_qa, model config missing "
                                "modality_start_end_token_indices for time_series "
                                "(required when is_interleaved_qa=True)"
                            )
                        ts_start_token = self.tokenizer.decode(
                            [indices["time_series"][0]]
                        )
                        ts_end_token = self.tokenizer.decode(
                            [indices["time_series"][1]]
                        )
                        split_prompt = prompt.split(f"{ts_start_token}{ts_end_token}")
                        new_prompt = ""
                        idx = 0
                        for part in split_prompt:
                            new_prompt += part
                            if idx < len(means):
                                new_prompt += (
                                    f"{ts_start_token}{ts_end_token} "
                                    f"Mean: {means[idx].item():.2f}, "
                                    f"Std: {stds[idx].item():.2f} "
                                )
                            idx += 1
                        prompt = new_prompt.strip()

                    # Pad/Truncate each instance/variate to self.model_config.max_ts_length
                    if tensor.shape[1] > self.model_config.max_ts_length:
                        tensor = tensor[:, : self.model_config.max_ts_length]
                    elif tensor.shape[1] < self.model_config.max_ts_length:
                        pad = torch.zeros(
                            tensor.shape[0],
                            self.model_config.max_ts_length - tensor.shape[1],
                        )
                        tensor = torch.cat([tensor, pad], dim=1)
                    # (num_vars * max_ts_length, 1)
                    tensor = tensor.reshape(-1, 1)
            
            # When there's no template, we need to append eos_token ourselves
            if self.tokenizer:
                if target[-len(self.tokenizer.eos_token):] != self.tokenizer.eos_token:
                    target += self.tokenizer.eos_token

            # meta = f"[time_series] {item.get('id', getattr(item, 'name', 'Unknown'))}"
            return tensor, prompt + target, [prompt, target]
        except Exception as e:
            logger.warning(f"TS QA Error: {e}")
            raise e
            # return torch.randn(self.model_config.max_ts_length, 1), "Corrupt TS", "[Error] Corrupt TS Item"

    def _process_ts_instruction(self, item):
        """Handler for the local ts_instruction JSONL (zone_a dataset).

        Real schema (per <data_root>/zone_a/ts_instruction, see
        ts_instruction.local_path in src/data/datasets_config.json):
          description, description_short, description_tiny,
          characteristics, series (list[float]), metadata (dict).

        We synthesize the (prompt, target) pair _process_ts_qa expects and then
        reuse the QA handler's normalization + start/end-token envelope by
        rebuilding `item` with the keys _process_ts_qa already understands."""
        if item is None:
            raise RuntimeError("Item is None in TS Instruction (Strict)")

        description = (item.get("description") or "").strip()
        characteristics = (item.get("characteristics") or "").strip()
        short = (item.get("description_short") or "").strip()
        tiny = (item.get("description_tiny") or "").strip()
        series = item.get("series")

        if not series:
            raise RuntimeError("ts_instruction record missing 'series' field")

        prompt = "Describe this time series."
        if characteristics:
            prompt = f"{prompt}\nCharacteristics:\n{characteristics}"

        target = short or tiny or description
        if not target:
            raise RuntimeError("ts_instruction record has no description fields")

        synthetic = {
            "instruction": prompt,
            "output": target,
            "series": series,
            "id": item.get("id", "ts_instruction"),
        }
        return self._process_ts_qa(synthetic)

    def _process_geo(self, item):
        # Point Cloud: (N, 6) (XYZRGB)
        if item is None:
            raise RuntimeError("Item is None in Point Cloud (Strict)")

        # If coming from Real Micro-CT stream
        if "tensor" in item:
            return item["tensor"], "Micro-CT Scan"

        return torch.randn(1024, 6), "3D object description."

    def _process_graph(self, item):
        # Graph (SMILES -> PyG Data-like Dict)
        {
            "x": torch.zeros(128, 32),
            "edge_index": torch.empty((2, 0), dtype=torch.long),
        }  # Fallback

        # if item is None: return default_graph, "Dummy Graph"
        if item is None:
            raise ValueError("Graph Item is None")

        try:
            logger.debug(f"Graph Item Keys: {list(item.keys())}")
            description = item.get("description", "Molecule.")
            # Robust Key Lookup
            smiles = item.get(
                "SMILES",
                item.get("smiles", item.get("structure", item.get("input", ""))),
            )

            # RDKit Featurization
            if Chem and smiles:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    # Node Features (Size 32)
                    # [AtomicNum, Degree, Charge, Hybrid, Aromatic, H_Num, 0...0]
                    atoms = mol.GetAtoms()
                    x_list = []
                    for atom in atoms:
                        feats = [
                            float(atom.GetAtomicNum()),
                            float(atom.GetDegree()),
                            float(atom.GetFormalCharge()),
                            float(atom.GetHybridization()),  # Enum to float
                            float(atom.GetIsAromatic()),
                            float(atom.GetTotalNumHs()),
                        ]
                        # Pad to 32
                        feats += [0.0] * (32 - len(feats))
                        x_list.append(feats)

                    x_tensor = torch.tensor(x_list, dtype=torch.float)

                    # Edge Index
                    edges = []
                    for bond in mol.GetBonds():
                        u = bond.GetBeginAtomIdx()
                        v = bond.GetEndAtomIdx()
                        edges.append([u, v])
                        edges.append([v, u])  # Undirected

                    if edges:
                        edge_index = (
                            torch.tensor(edges, dtype=torch.long).t().contiguous()
                        )
                    else:
                        edge_index = torch.empty((2, 0), dtype=torch.long)

                    # --- Pad/Truncate to Fixed Size (128) ---
                    MAX_NODES = 128
                    num_nodes = x_tensor.size(0)

                    if num_nodes > MAX_NODES:
                        # Truncate
                        x_tensor = x_tensor[:MAX_NODES, :]
                        # Filter edges involving truncated nodes
                        mask = (edge_index[0] < MAX_NODES) & (edge_index[1] < MAX_NODES)
                        edge_index = edge_index[:, mask]
                    elif num_nodes < MAX_NODES:
                        # Pad
                        pad_size = MAX_NODES - num_nodes
                        pad = torch.zeros(pad_size, 32)
                        x_tensor = torch.cat([x_tensor, pad], dim=0)

                    return {"x": x_tensor, "edge_index": edge_index}, description

            # If standard processing fails or SMILES missing, RAISE instead of dummy
            # return default_graph, description
            raise ValueError(f"Could not parse graph from item: {list(item.keys())}")

        except Exception as e:
            # print(f"Graph Error: {e}")
            # if not self.allow_dummy_data: raise e
            # return default_graph, "Corrupt Graph"
            # FORCE RAISE to skip item in __iter__ loop
            raise e

    # --- Table Handlers ---
    def _get_tapas_tokenizer(self):
        if not hasattr(self, "tapas_tokenizer"):
            try:
                from transformers import TapasTokenizer

                self.tapas_tokenizer = TapasTokenizer.from_pretrained("google/tapas-base")
            except Exception:
                logger.warning("Warning: Could not load TapasTokenizer. Using random stubs.")
                self.tapas_tokenizer = None
        return self.tapas_tokenizer

    def _process_table(self, item):
        # Generic Table Handler using Tapas
        if item is None:
            raise RuntimeError("Item is None in Table (Strict)")

        if random.random() < 0.001:
            logger.debug(f"Table Item Keys: {list(item.keys())}")

        try:
            import pandas as pd
            # Extract Table Data
            # Spider: 'question', 'query'
            # PubTables: 'table_content' (html/str), 'question'

            data = {}
            if "table" in item:
                # Dict {'header': [], 'rows': []}
                data = item["table"]
            elif "header" in item and "rows" in item:
                data = item

            # Create DataFrame
            if data and "header" in data and "rows" in data:
                df = pd.DataFrame(data["rows"], columns=data["header"])
            else:
                # Fallback: Try flat dict
                df = pd.DataFrame([item]) if isinstance(item, dict) else pd.DataFrame()

            # Stringify for Tapas & Truncate Cell Content (Prevent ReDoS/Hang)
            # Tapas tokenization hangs on very long strings when parsing dates/numbers
            df = df.astype(str).applymap(lambda x: x[:50])

            if df.empty:
                df = pd.DataFrame({"Col1": ["Val1"], "Col2": ["Val2"]})

            tokenizer = self._get_tapas_tokenizer()
            default_ids = torch.zeros(128, dtype=torch.long)
            if tokenizer:
                # Tapas Encoding
                # We treat the table as the context.
                # Projector expects [B, S, D] -> We return input_ids [S]
                # Tapas Model expects input_ids, attention_mask, token_type_ids
                # We can package them or just return input_ids and let encoder handle?
                # Encoder.forward takes dict.
                enc = tokenizer(
                    table=df,
                    queries="Is this a table?",
                    truncation=True,
                    padding="max_length",
                    max_length=128,
                    return_tensors="pt",
                )

                # Ideally we return dict, but collator expects tensor?
                # If Union Schema allows dict, we return dict.
                # UnifiedTransformer `_process_multimodal_embeddings` handles dicts for graph.
                # Let's hope it handles dicts for table too.
                # If not, we return input_ids and Encoder assumes ids.
                # TableEncoder.forward checks for dict.
                # Check stats
                ids = enc["input_ids"][0]
                # print(f"DEBUG Table Enc Stats: Min={ids.float().min()}, Max={ids.float().max()}")
                if ids.float().max() == 0:
                    logger.warning("Table Enc is ALL ZEROS")
                    print(f"DEBUG DF Shape: {df.shape}")
                    print(f"DEBUG DF Columns: {df.columns.tolist()}")
                    print(f"DEBUG DF Head:\n{df.head()}")
                    # Check if astype(str) made it weird
                    logger.debug(f"DF Iloc[0,0]: {df.iloc[0, 0]!r}")

                return {
                    "input_ids": ids,
                    "attention_mask": enc["attention_mask"][0],
                    "token_type_ids": enc["token_type_ids"][0],
                }, "Table Data"
            else:
                logger.error("Tapas Tokenizer NOT LOADED")
                return default_ids, "Table Data (No Tokenizer)"

        except Exception as e:
            logger.error(f"Table Processing Error: {e}")
            import traceback

            traceback.print_exc()
            raise e
            # if not self.allow_dummy_data: raise e
            # return default_ids, "Corrupt Table"

    def _process_table_reasoning(self, item):
        # TableLLM: question -> answer
        ids, _ = self._process_table(item)
        if item is None:
            return ids, "Dummy Table Reasoning"

        q = item.get("question", item.get("input", item.get("prompt", "Describe this table.")))
        a = item.get("answer", item.get("output", item.get("response", "Unknown")))
        # Some datasets use 'answers' list
        if isinstance(a, list):
            a = "; ".join(a)

        text = f"{q}\nTarget: {a}"
        meta = f"[table_reasoning] {item.get('id', 'Unknown')}"
        return ids, text, meta

    def _process_table_instruction(self, item):
        ids, _ = self._process_table(item)
        if item is None:
            return ids, "Dummy Table Instruction"

        q = item.get("instruction", item.get("question", "Analyze table."))
        a = item.get("response", item.get("output", item.get("answer", "Unknown")))

        text = f"{q}\nTarget: {a}"
        return ids, text

    def _process_table_structure(self, item):
        # UniTabE: 'question' 'answers'
        ids, _ = self._process_table(item)
        if item is None:
            return ids, "Dummy Table Structure"

        q = item.get("question", "Structure Query?")
        a = item.get("answers", item.get("answer", "Unknown"))
        if isinstance(a, list):
            a = str(a)

        text = f"{q}\nTarget: {a}"
        return ids, text

    # --- Graph Handlers ---
    def _process_graph_captioning(self, item):
        # ChEBI-20: SMILES -> description
        graph_dict, _ = self._process_graph(item)
        if item is None:
            return graph_dict, "Dummy Graph Captioning"
        try:
            # WebNLG keys: 'lex' 'original_triple_sets'
            # ChEBI keys: 'SMILES' 'description'
            smiles = item.get("SMILES") or str(item.get("original_triple_sets", ""))
            desc = item.get("description") or str(item.get("lex", ""))

            # Use extracted SMILES for featurization
            graph_dict, _ = self._process_graph({"SMILES": smiles})

            text = f"Describe the following molecule/graph: {smiles}\nDescription: {desc}"
            meta = f"[graph_captioning] {item.get('CID', item.get('SMILES', 'Unknown')[:20])}"
            return graph_dict, text, meta
        # except:
        #    return torch.randint(0, 30522, (128,)), "Corrupt Graph Captioning", "[Error] Corrupt Graph Item"

        except Exception as e:
            print(f"Corrupt Graph Captioning: {e}")
            time.sleep(5)

    def _process_graph_grounding(self, item):
        # ChEBI-20-MM: SMILES + SELFIES (Motifs)
        # Parse SMILES or string slice to find substructure
        # if item is None: return torch.randint(0, 30522, (128,)), "Dummy Graph Grounding"

        try:
            smiles = item.get("SMILES", "")
            # Simple Motif Proxy: extracting double bonds or rings
            motif = "Unknown"
            target_text = ""

            if "=" in smiles:
                motif = "="
                target_text = "Double Bond"
            elif "#" in smiles:
                motif = "#"
                target_text = "Triple Bond"
            elif "1" in smiles:
                motif = "1...1"
                target_text = "Ring Structure 1"

            # Construct Grounding Task
            text = f"Locate the {target_text} in: {smiles}\nMotif Found: {motif}"
            graph_dict, _ = self._process_graph({"SMILES": smiles})
            return graph_dict, text
        except Exception:
            return {
                "x": torch.zeros(128, 32),
                "edge_index": torch.empty((2, 0), dtype=torch.long),
            }, "Corrupt Graph Grounding"

    def _process_graph_instruction(self, item):
        # GraphInstruct: query -> answer
        if item is None:
            return {
                "x": torch.zeros(128, 32),
                "edge_index": torch.empty((2, 0), dtype=torch.long),
            }, "Dummy Graph Instruction"
        if item is None:
            return {
                "x": torch.zeros(128, 32),
                "edge_index": torch.empty((2, 0), dtype=torch.long),
            }, "Dummy Graph Instruction"
        try:
            q = item.get("query", "")
            a = item.get("answer", "")
            text = f"Graph Task: {q}\nAnswer: {a}"
            graph_dict, _ = self._process_graph(item)
            return graph_dict, text
        except Exception:
            # Fallback for text-only instruction tuning rows
            return {
                "x": torch.zeros(128, 32),
                "edge_index": torch.empty((2, 0), dtype=torch.long),
            }, text + " (Text Only)"

    def _process_graph_crystal(self, item):
        # Matbench: positions + atomic_numbers
        if item is None:
            return {
                "x": torch.zeros(128, 32),
                "edge_index": torch.empty((2, 0), dtype=torch.long),
            }, "Dummy Crystal"
        try:
            atoms = item.get("atomic_numbers", [])
            pos = item.get("positions", [])

            # Construct Textual Graph Representation
            # "Atom 14 (Si) at [0.1, 0.2, 0.3]..."
            graph_str_parts = []
            for i, (a, p) in enumerate(zip(atoms, pos, strict=False)):
                if i > 10:
                    break  # Truncate for prompt
                graph_str_parts.append(f"Atom Z={a} at {p}")

            graph_desc = "; ".join(graph_str_parts)
            target = item.get("y", "Unknown Property")

            text = f"Predict property for Crystal Structure:\n{graph_desc}\nTarget: {target}"
            return {
                "x": torch.zeros(128, 32),
                "edge_index": torch.empty((2, 0), dtype=torch.long),
            }, text
        except Exception:
            return {
                "x": torch.zeros(128, 32),
                "edge_index": torch.empty((2, 0), dtype=torch.long),
            }, "Corrupt Crystal Data"

    def _process_graph_circuit(self, item):
        # Verilog: text
        if item is None:
            return {
                "x": torch.zeros(128, 32),
                "edge_index": torch.empty((2, 0), dtype=torch.long),
            }, "Dummy Circuit"
        try:
            verilog_code = item.get("text", "")
            # Just truncate code for prompt
            short_code = verilog_code[:512]
            text = f"Analyze the following Circuit Netlist (Verilog):\n{short_code}...\nTask: Synthesize/Verify"
            return {
                "x": torch.zeros(128, 32),
                "edge_index": torch.empty((2, 0), dtype=torch.long),
            }, text
        except Exception:
            return {
                "x": torch.zeros(128, 32),
                "edge_index": torch.empty((2, 0), dtype=torch.long),
            }, "Corrupt Circuit Data"

    # --- Geometry Handlers ---
    def _create_physics_stream(self, hf_id, local_path=None):
        """Custom generator for The Well HDF5 files."""
        # 1. Validation Logic

        # Heuristic: repo_id "polymathic-ai/X" -> file "data/train/X_train.h5" usually
        dataset_name = hf_id.split("/")[-1]
        filename = f"data/train/{dataset_name}_train.h5"
        target_key = None

        if "turbulence_gravity_cooling" in hf_id:
            filename = "data/train/turbulence_gravity_cooling_rho0_0.445_Z_0.1_T0_100.hdf5"
            target_key = "t0_fields/density"
        elif "MHD" in hf_id:
            filename = "data/train/MHD_Ma_0.7_Ms_0.5.hdf5"
            target_key = "t1_fields/velocity"
        elif "supernova" in hf_id:
            filename = "data/train/supernova_explosion_Msun_0.1_dim128_file_00.hdf5"
            target_key = "t0_fields/temperature"

        # Check Existance
        file_path = None
        if local_path and os.path.isdir(local_path):
            # Search for HDF5 in local path
            base_name = os.path.basename(filename)
            possible_path = os.path.join(local_path, base_name)
            if os.path.exists(possible_path):
                file_path = possible_path
                print(f"Using Local HDF5: {file_path}")
            else:
                # Deep search?
                for root, _, files in os.walk(local_path):
                    if base_name in files:
                        file_path = os.path.join(root, base_name)
                        print(f"Found Local HDF5 (deep): {file_path}")
                        break

        if not file_path:
            # Fallback to Remote (Download) - Unified Lazy Loading
            print(f"[DATA LOAD] Source: REMOTE (HF Download) | ID: {hf_id}")
            try:
                file_path = hf_hub_download(
                    repo_id=hf_id,
                    filename=filename,
                    repo_type="dataset",
                    token=self.hf_token,
                )
            except Exception as e:
                print(f"Failed to download Physics: {e}")
                return None
        else:
            print(f"[DATA LOAD] Source: LOCAL (HDF5) | Path: {file_path}")

        # 2. Define Generator (If Valid)
        def data_generator():
            while True:
                try:
                    with h5py.File(file_path, "r") as f:
                        # Physics-Specific Access
                        data = None
                        main_key = "unknown"

                        if target_key:
                            if target_key in f:
                                data = f[target_key]
                                main_key = target_key
                            elif "/" in target_key:
                                parts = target_key.split("/")
                                curr = f
                                for p in parts:
                                    if p in curr:
                                        curr = curr[p]
                                    else:
                                        break
                                if hasattr(curr, "shape"):
                                    data = curr
                                    main_key = parts[-1]

                        if data is None:
                            # Fallback Heuristics
                            keys = list(f.keys())
                            if "temperature" in keys:
                                main_key = "temperature"
                                data = f["temperature"]
                            elif "velocity" in keys:
                                main_key = "velocity"
                                data = f["velocity"]
                            elif len(keys) > 0 and hasattr(f[keys[0]], "shape"):
                                main_key = keys[0]
                                data = f[keys[0]]

                        if data is None:
                            time.sleep(1)
                            yield {
                                "tensor": torch.zeros(1, 1, 1),
                                "id": dataset_name,
                                "field": "error",
                                "stats": {},
                            }
                            continue

                        length = data.shape[0]
                        indexes = list(range(length))
                        random.shuffle(indexes)

                        for idx in indexes:
                            tensor = torch.tensor(data[idx]).float()
                            mean = tensor.mean()
                            std = tensor.std() + 1e-6
                            tensor = (tensor - mean) / std  # Instance Norm
                            yield {
                                "tensor": tensor,
                                "id": dataset_name,
                                "field": main_key,
                                "stats": {},
                            }

                    time.sleep(0.01)  # Throttle Re-open
                except Exception as e:
                    print(f"Physics Reader Error: {e}")
                    time.sleep(5)

        return data_generator()

    def _download_berea_if_needed(self):
        """Helper to download Berea Sandstone .npz"""
        import os

        local_path = "data/berea_200.npz"
        url = "https://raw.githubusercontent.com/PMEAL/OpenPNM/dev/tests/fixtures/berea_100_to_300.npz"

        if not os.path.exists(local_path):
            print(f"Downloading Berea Sandstone from {url}...")
            try:
                os.makedirs("data", exist_ok=True)
                r = requests.get(url)
                if r.status_code == 200:
                    with open(local_path, "wb") as f:
                        f.write(r.content)
                    print("Download complete.")
                    return local_path
                else:
                    print(f"Download failed: {r.status_code}")
                    return None
            except Exception as e:
                print(f"Berea download exception: {e}")
                return None
        return local_path

    def _create_real_microct_stream(self, local_path=None):
        """Streams real Berea Sandstone crops."""
        path = None
        if local_path and os.path.isdir(local_path):
            for f in os.listdir(local_path):
                if f.endswith(".npz"):
                    path = os.path.join(local_path, f)
                    break

        # 1. Validation (Return None if skipping)
        # 1. Validation (Auto-Download Fallback)
        if not path:
            print("[DATA LOAD] Source: REMOTE (Download Berea) | URL: OpenPNM")
            path = self._download_berea_if_needed()
        else:
            print(f"[DATA LOAD] Source: LOCAL (NPZ) | Path: {path}")

        # 2. Generator Definition
        def data_generator():
            if not path:
                print("Failed to get Berea data. Yielding mocks.")
                yield from self._create_mock_geometry_stream()
                return

            import numpy as np

            try:
                try:
                    data = np.load(path)
                except Exception:
                    data = None

                if data is None:
                    yield from self._create_mock_geometry_stream()
                    return

                if "im" in data:
                    vol = data["im"]
                else:
                    key = list(data.keys())[0]
                    vol = data[key]

                vol = vol.astype(np.float32)
                D, H, W = vol.shape

                while True:
                    try:
                        # Voxel to Point Cloud for PRISM Geo Projector (N, 6)
                        # 1. Crop 32x32x32
                        cz, cy, cx = (
                            random.randint(0, D - 32),
                            random.randint(0, H - 32),
                            random.randint(0, W - 32),
                        )
                        crop = vol[cz : cz + 32, cy : cy + 32, cx : cx + 32]

                        # 2. Convert to Points (Indices of non-zero or just grid)
                        # Berea is porous (0/1 or density).
                        # Let's sample points.
                        try:
                            # Grid coordinates
                            z, y, x = np.indices((32, 32, 32))
                            # Fallback: Just take 1024 random points from the block
                            # Features: [z, y, x, val, 0, 0] -> (N, 6)

                            # Flatten
                            z_f, y_f, x_f, v_f = (
                                z.flatten(),
                                y.flatten(),
                                x.flatten(),
                                crop.flatten(),
                            )

                            # Subsample 1024
                            if len(v_f) > 0:
                                indices = np.random.choice(len(v_f), 1024, replace=True)

                                points = np.stack(
                                    [
                                        z_f[indices] / 32.0,
                                        y_f[indices] / 32.0,
                                        x_f[indices] / 32.0,
                                        v_f[indices],
                                        np.zeros(1024),
                                        np.zeros(1024),
                                    ],
                                    axis=1,
                                )  # (1024, 6)

                                yield {
                                    "tensor": points,
                                    "id": "berea_sandstone",
                                    "field": "real_microct_pc",
                                    "stats": {},
                                }
                            else:
                                yield {
                                    "tensor": np.zeros((1024, 6)),
                                    "id": "berea_error",
                                    "field": "error",
                                    "stats": {},
                                }

                        except Exception:
                            yield {
                                "tensor": np.zeros((1024, 6)),
                                "id": "berea_error",
                                "field": "error",
                                "stats": {},
                            }

                        time.sleep(0.01)
                    except GeneratorExit:
                        return
                    except Exception as e:
                        print(f"Berea Stream Loop Error: {e}")
                        time.sleep(1)
            except GeneratorExit:
                return
            except Exception as e:
                print(f"Error streaming Berea (Outer): {e}")
                yield from self._create_mock_geometry_stream()

        # 3. Return Generator Call
        return data_generator()

    def _create_chebi_stream(self, hf_id):
        """Custom CSV stream for ChEBI-20 to handle parsing errors (Robust)."""
        try:
            # Try offline first
            logger.info(f"Initializing ChEBI Stream Manual Download: {hf_id}")
            path = hf_hub_download(
                repo_id=hf_id,
                filename="train.csv",
                repo_type="dataset",
                token=self.hf_token,
                local_files_only=True,
            )
        except Exception:
            # Fallback to online
            try:
                logger.info("Downloading ChEBI-20 CSV (Network)...")
                path = hf_hub_download(
                    repo_id=hf_id,
                    filename="train.csv",
                    repo_type="dataset",
                    token=self.hf_token,
                    local_files_only=False,
                )
            except Exception as e:
                logger.warning(f"Failed to download ChEBI-20: {e}")
                # Fallback to test.csv if train missing?
                try:
                    logger.info("Trying test.csv fallback...")
                    path = hf_hub_download(
                        repo_id=hf_id,
                        filename="test.csv",
                        repo_type="dataset",
                        token=self.hf_token,
                        local_files_only=False,
                    )
                except Exception:
                    yield from self._dummy_generator()
                    return

        while True:
            try:
                # Robust Read with Sniffing
                with open(path, encoding="utf-8", errors="replace") as f:
                    try:
                        # Sniff delimiter (TSV vs CSV) - ChEBI is usually Tab
                        sample = f.read(8192)
                        f.seek(0)
                        try:
                            dialect = csv.Sniffer().sniff(sample)
                            reader = csv.DictReader(f, dialect=dialect)
                        except Exception:
                            # Sniffer failed, force TSV as most likely for ChEBI
                            f.seek(0)
                            reader = csv.DictReader(f, delimiter="\t")

                    except Exception:
                        f.seek(0)
                        reader = csv.DictReader(f)  # Default Excel-CSV

                    count = 0
                    for row in reader:
                        # Normalize Keys (some versions use 'smiles', others 'SMILES')
                        if "SMILES" in row:
                            row["SMILES"] = row["SMILES"]
                        elif "smiles" in row:
                            row["SMILES"] = row["smiles"]

                        if "SMILES" in row and ("description" in row or "text" in row):
                            if "description" not in row and "text" in row:
                                row["description"] = row["text"]
                            yield row
                            count += 1

                    if count == 0:
                        logger.warning(
                            "Warning: ChEBI Stream yielded 0 rows. Retrying with alternate delimiter..."
                        )
                        time.sleep(5)
            except Exception as e:
                logger.warning(f"Error in ChEBI Stream Loop: {e}")
                time.sleep(5)

    def _create_mock_geometry_stream(self):
        """Custom generator for Mock Micro-CT (Fallback)."""
        while True:
            crop = torch.rand(32, 32, 32)
            porosity = crop.mean().item()
            yield {"voxel": crop, "porosity": porosity, "source": "Synthetic Proxy"}

    def _process_geo_physics(self, item):
        # The Well HDF5 -> Description
        if item is None:
            if not self.allow_dummy_data:
                raise RuntimeError("Item is None in Physics Stream (Strict Mode)")
            raise RuntimeError("Item is None or Invalid in Geo PDE (Strict)")
        try:
            raw = item["tensor"]
            if isinstance(raw, torch.Tensor):
                tensor = raw.clone().detach().float()
            else:
                tensor = torch.tensor(raw).float()
            # Resize/Crop to standard input if needed, or pass raw.
            # PRISM Geometry Projector likely expects specific shape.

            # Note: Normalization is applied in the stream (_create_physics_stream) for efficiency,
            # but if item comes from elsewhere, we apply here too.
            # Double normalization is fine if stats are roughly unit.
            # Let's verify if already normalized.
            # For robustness, we can re-normalize or assume stream is good.
            # Given we just edited the stream, let's assume stream is good.
            # But wait, self-contained method is better.

            # Instance Norm (Safety)
            mean = tensor.mean()
            std = tensor.std() + 1e-6
            tensor = (tensor - mean) / std

            # For now, just pass tensor.

            # Caption
            d_id = item.get("id", "sim")
            field = item.get("field", "scalar")
            stats = item.get("stats", {})

            # Enrich Caption with Stats
            mean_val = "Unknown"
            if field in stats.get("mean", {}):
                mean_val = f"{stats['mean'][field]:.2e}"

            text = f"Physics Simulation: {d_id}\nPhysical Field: {field}\nGlobal Mean: {mean_val}\nDescription: 3D volumetric evolution of {field} dynamics."
            return tensor, text
        except Exception as e:
            if not self.allow_dummy_data:
                raise RuntimeError(f"Corrupt Physics Data: {e}") from e
            return torch.randn(3, 32, 32, 32), "Corrupt Physics Data"

    def _process_geo_mat_tomo(self, item):
        # Micro-CT (Real or Mock)
        if item is None:
            if not self.allow_dummy_data:
                raise RuntimeError("Item is None in Tomography Stream (Strict Mode)")
            raise RuntimeError("Item is None in Geo Tomography (Strict)")
        try:
            # Item formatting from stream
            if "voxel" in item:
                # Handle Numpy Conversion Explicitly
                raw = item["voxel"]
                if not isinstance(raw, torch.Tensor):
                    raw = torch.tensor(raw)
                tensor = raw.float().unsqueeze(0)
            else:
                raw = item.get("tensor", torch.rand(32, 32, 32))
                if not isinstance(raw, torch.Tensor):
                    raw = torch.tensor(raw)
                tensor = raw.float().unsqueeze(0)

            # Instance Norm
            mean = tensor.mean()
            std = tensor.std() + 1e-6
            tensor = (tensor - mean) / std

            por = item.get("porosity", 0.2)
            src = item.get("source", "Unknown")

            text = (
                f"Micro-CT Scan of Porous Media ({src}). "
                f"Volumetric density map showing pore structure. "
                f"Porosity: {por:.1%}. "
                f"Material: Sandstone."
            )
            return tensor, text
        except Exception as e:
            # if not self.allow_dummy_data: raise RuntimeError(f"Corrupt Tomography Data: {e}")
            raise e
            # return torch.randn(1, 32, 32, 32), "Corrupt Tomography Data"

    def _process_geo_pde(self, item):
        # PDEBench -> Problem Statement
        if item is None:
            if not self.allow_dummy_data:
                raise RuntimeError("Item is None in PDE Stream (Strict Mode)")
            return torch.zeros(1, 10), "Dummy PDE"
        try:
            # PDEBench 1D Tensor [Time, X]
            t_data = item.get("tensor", [])
            # Take Initial Condition (Index 0)
            ic = torch.tensor(t_data[0]).float()

            # Format
            params = item.get("parameters", "Unknown")
            text = f"PDE Problem Statement: Predict evolution for {params}. Initial Condition provided."
            return ic, text
        except Exception as e:
            # if not self.allow_dummy_data: raise RuntimeError(f"Corrupt PDE Data: {e}")
            raise e
            # return torch.zeros(1, 10), "Corrupt PDE Data"

    # --- Time-MMD Helper ---
    def _load_time_mmd_map(self):
        """Lazily load Time-MMD (Economy) Series from GitHub into a date-lookup map."""
        if hasattr(self, "_time_mmd_map") and self._time_mmd_map:
            return self._time_mmd_map

        print("Lazy Loading Time-MMD Economy Series from GitHub...")
        url = "https://raw.githubusercontent.com/AdityaLab/Time-MMD/main/numerical/Economy/Economy.csv"
        try:
            import csv

            import requests


            r = requests.get(url)
            if r.status_code != 200:
                print(f"Failed to fetch Time-MMD CSV: {r.status_code}")
                self._time_mmd_map = {}
                return {}

            # Parse CSV
            # Header: Month,Exports,Imports,OT,start_date,date,end_date
            # We want to map start_date -> series (Exports, Imports, OT)
            self._time_mmd_map = {}
            lines = r.text.splitlines()
            reader = csv.DictReader(lines)
            for row in reader:
                s_date = row.get("start_date")
                # Extract numerical columns
                try:
                    vals = [
                        float(row["Exports"].replace(",", "")),
                        float(row["Imports"].replace(",", "")),
                        float(row["OT"].replace(",", "")),
                    ]
                    # We need a 512x1 tensor projection.
                    # We have 3 features. We can project them later or just normalize.
                    # For now, let's just take the first feature (Exports) as the univariate series
                    # OR we can keep all 3 if the projector handles multivariate (it expects 1 channel usually for now).
                    # Let's pivot to univariate 'Experts' for simplicity or flatten.
                    # Given the task is 30k iters, let's keep it simple: Feature 0 (Exports).
                    self._time_mmd_map[s_date] = vals[0]
                except Exception:
                    continue
            print(f"Loaded {len(self._time_mmd_map)} Time-MMD entries.")
            return self._time_mmd_map
        except Exception as e:
            print(f"Error loading Time-MMD map: {e}")
            self._time_mmd_map = {}
            return {}

    def _process_ts_time_mmd(self, item):
        # Merges HF Text (lamblamb) with GitHub Series (Time-MMD)
        if item is None:
            raise RuntimeError("Item is None in Time-MMD (Strict)")
        # if item is None: return torch.randn(512, 1), "Dummy Time-MMD"

        # 1. Get Series
        series_map = self._load_time_mmd_map()
        start_date = item.get("start_date", "")

        tensor = None  # noqa: F841
        # Try exact match first
        if start_date in series_map:
            val = series_map[start_date]
        else:
            # Try Month-Match: 1993-08-30 -> 1993-08-01
            try:
                # Check format YYYY-MM-DD
                parts = start_date.split("-")
                if len(parts) == 3:
                    # Reconstruct YYYY-MM-01
                    month_key = f"{parts[0]}-{parts[1]}-01"
                    if month_key in series_map:
                        val = series_map[month_key]
                    else:
                        # CLAMPING LOGIC: Check if out of range
                        sorted_keys = sorted(series_map.keys())
                        min_date = sorted_keys[0]
                        max_date = sorted_keys[-1]

                        if start_date < min_date:
                            # Clamp to start
                            val = series_map[min_date]
                            # Optional: print(f"Warning: Clamping {start_date} to {min_date}")
                        elif start_date > max_date:
                            # Clamp to end
                            val = series_map[max_date]
                        else:
                            # Truly missing within range
                            raise RuntimeError(
                                f"Start Date {start_date} (Month {month_key}) not found in Time-MMD Map (Strict)"
                            )
                else:
                    raise RuntimeError(f"Invalid Date Format {start_date} (Strict)")
            except Exception as e:
                raise RuntimeError(f"Time-MMD Lookup Failed for {start_date}: {e}") from e

        # If we reached here, we have 'val'.
        # Generate synthetic series around this value
        # For now, let's create a random walk starting at val.
        series = [val]
        curr = val
        for _ in range(511):
            curr += (random.random() - 0.5) * 100  # Drift
            series.append(curr)
        series_tensor = torch.tensor(series).float().unsqueeze(-1)

        # 2. Get Text
        fact = item.get("fact", "")
        preds = item.get("preds", "")
        text = f"Fact: {fact}\nPredictions: {preds}"

        return series_tensor, text

    # --- DNA Helpers ---

    def _render_dna_prompt_text(self, prompt: list) -> tuple:
        """
        Renders the chat-format prompt list produced by _process_dna_bioreason into
        two flat strings that match the Qwen/ChatML format expected by the tokenizer
        and the interleaved merge machinery.

        The label-masking logic in _merge_text_input_ids_with_modality_embeds
        locates the boundary between question and answer via the "P T" _metadata
        token counts, while the collate fn scans for the literal substring
        "<|im_end|>\\n<|im_start|>assistant\\n" to find where the assistant turn
        begins. This renderer must produce exactly that substring.

        User turn rendering:
          <|im_start|>user\\n
          [for each content slot]
            type=="dna_reference" → Reference sequence: <dna_ref_start><dna_ref_end>\\n
            type=="dna_variant"   → Variant sequence: <dna_var_start><dna_var_end>\\n
                           (the tokens registered per-modality in
                           modality_start_end_token_indices, NOT the chat-template
                           <|dna_start|><|dna_pad|><|dna_end|> variant)
            type=="text" → the question text
          <|im_end|>\\n

        Assistant turn rendering (SFT only):
          <|im_start|>assistant\\n
          [if reasoning] <think>\\n{reasoning}\\n</think>\\n\\n
          {answer text}
          <|im_end|>\\n

        Returns:
            full_text   — user turn + assistant turn (use for example["text"])
            prompt_text — user turn only + "<|im_start|>assistant\\n" suffix
                          (use to measure prompt_tokens for _metadata)
        """
        modality_tags = self.model_config.modality_start_end_token_indices or {}
        dna_ref_tags = modality_tags.get("dna_reference", ("<dna_ref_start>", "<dna_ref_end>"))
        dna_var_tags = modality_tags.get("dna_variant", ("<dna_var_start>", "<dna_var_end>"))
        dna_ref_placeholder = f"Reference sequence: {dna_ref_tags[0]}{dna_ref_tags[1]}\n"
        dna_var_placeholder = f"Variant sequence: {dna_var_tags[0]}{dna_var_tags[1]}\n"

        user_body = ""
        assistant_body = ""

        for msg in prompt:
            role = msg["role"]
            if role == "user":
                parts = []
                for c in msg["content"]:
                    # DNA-LLM mode: emit the registered placeholder tokens so
                    # _merge_text_input_ids_with_modality_embeds can find and
                    # expand this slot. LLM mode: sequences already inlined in
                    # the text content slot; the dna slots render as empty.
                    if c.get("type") == "dna_reference":
                        if self.model_name == "dna-llm":
                            parts.append(dna_ref_placeholder)
                    elif c.get("type") == "dna_variant":
                        if self.model_name == "dna-llm":
                            parts.append(dna_var_placeholder)
                    elif c.get("type") == "text":
                        parts.append(c["text"])
                user_body = f"<|im_start|>user\n{''.join(parts)}<|im_end|>\n"

            elif role == "assistant":
                reasoning = msg.get("reasoning_content", "")
                answer = msg["content"][0]["text"]
                if reasoning:
                    assistant_body = (
                        f"<|im_start|>assistant\n"
                        f"<think>\n{reasoning}\n</think>\n\n"
                        f"{answer}<|im_end|>\n"
                    )
                else:
                    assistant_body = f"<|im_start|>assistant\n{answer}<|im_end|>\n"

        # prompt_text ends exactly at the assistant marker so that:
        #   len(tokenize(prompt_text)) == P  (the P in the "P T" _metadata string)
        # The marker "<|im_end|>\n<|im_start|>assistant\n" is the literal string
        # relied on elsewhere to find where the assistant turn begins.
        prompt_text = user_body + "<|im_start|>assistant\n"

        # SFT: full_text = user turn + complete assistant turn (answer in the sequence).
        # RL:  full_text = user turn + generation prompt only (model generates the answer).
        #      assistant_body is empty when is_sft=False, so we append the marker explicitly.
        if assistant_body:
            full_text = user_body + assistant_body
        else:
            full_text = prompt_text  # generation-prompt-only for RL

        return full_text, prompt_text

    def _count_kegg_answers(self, counts) -> None:
        """Full non-streaming load of wanglab/kegg train split (~1,159 rows —
        confirmed small enough that streaming/capping isn't needed) into `counts`.
        """
        from datasets import load_dataset

        logger.info("[ClassWeights] Loading wanglab/kegg train split to compute class frequencies...")
        ds = load_dataset("wanglab/kegg", split="train", trust_remote_code=True)
        for item in ds:
            counts[clean_kegg_answer(str(item["answer"]))] += 1

    def _count_streamed_answers(self, counts, hf_id: str, clean_fn, max_rows: int) -> None:
        """Streaming, row-capped frequency count for a large dataset.

        variant_effect_coding (48,850 train rows) and variant_effect_non_snv
        (35,215 train rows) are 30-50x larger than KEGG (1,159 rows) — a full
        non-streaming load here would add much more startup latency than KEGG's
        equivalent load, so this streams and caps at max_rows instead, trading a
        small amount of frequency-estimate accuracy for bounded startup time.
        """
        import itertools

        from datasets import load_dataset

        logger.info(
            f"[ClassWeights] Streaming {hf_id} train split (capped at {max_rows} rows) "
            f"to compute class frequencies..."
        )
        ds = load_dataset(hf_id, split="train", streaming=True)
        n_seen = 0
        for item in itertools.islice(ds, max_rows):
            counts[clean_fn(str(item["answer"]))] += 1
            n_seen += 1
        logger.info(f"[ClassWeights] {hf_id}: counted {n_seen} rows.")

    def _build_class_weights(self, weight_max: float, streamed_cap: int = 20_000) -> None:
        """Compute inverse-frequency class weights across all active DNA
        BioReason datasets (KEGG + variant_effect_coding + variant_effect_non_snv),
        merged into one shared label space.

        Produces balanced weights: w_c = (N / (C * n_c)) capped at weight_max,
        where N/C are the *combined* sample/class counts across every dataset
        counted below. Weights are stored in self.class_weights_map {label: weight}.

        Each dataset's counting is independently fault-isolated: a failure
        loading one dataset (e.g. a network hiccup on a variant_effect dataset)
        logs a warning and is skipped, rather than discarding every other
        dataset's already-successfully-counted frequencies. If every dataset
        fails, class_weights_map ends up {} (uniform weights).
        """
        import collections

        counts: collections.Counter = collections.Counter()

        # dna_bioreason (KEGG) — always counted via the original full-load path
        # if present and not skipped, regardless of the streamed datasets below.
        kegg_info = self.datasets_map.get("dna_bioreason")
        if kegg_info and not kegg_info.get("skip", False):
            try:
                self._count_kegg_answers(counts)
            except Exception as e:
                logger.warning(f"[ClassWeights] Failed to count wanglab/kegg answers: {e}. Skipping.")

        streamed_datasets = [
            (
                "dna_bioreason_variant_effect_coding",
                "wanglab/variant_effect_coding",
                clean_variant_effect_coding_answer,
            ),
            (
                "dna_bioreason_variant_effect_non_snv",
                "wanglab/variant_effect_non_snv",
                clean_variant_effect_non_snv_answer,
            ),
        ]
        for dataset_key, hf_id, clean_fn in streamed_datasets:
            info = self.datasets_map.get(dataset_key)
            if not info or info.get("skip", False):
                continue
            try:
                self._count_streamed_answers(counts, hf_id, clean_fn, streamed_cap)
            except Exception as e:
                logger.warning(f"[ClassWeights] Failed to count {hf_id} answers: {e}. Skipping.")

        if not counts:
            logger.warning("[ClassWeights] No class frequencies counted from any dataset. Using uniform weights.")
            self.class_weights_map = {}
            return

        N = sum(counts.values())
        C = len(counts)
        for label, n in counts.items():
            w = N / (C * n)
            self.class_weights_map[label] = min(w, weight_max)
        logger.info(
            f"[ClassWeights] {C} classes, {N} samples (merged across "
            f"{1 + len(streamed_datasets)} datasets). "
            f"Weight range: [{min(self.class_weights_map.values()):.2f}, "
            f"{max(self.class_weights_map.values()):.2f}] (cap={weight_max})"
        )

    def _process_dna_bioreason(self, item):
        """
        Format a KEGG BioReason example into the structured chat dict consumed by
        qwen_dna_collate_fn.  Mirrors _format_kegg in BioReason/bioreason/dataset/kegg.py.

        model_name="llm"     — DNA sequences are inlined as plain text; dna_sequences is empty.
        model_name="dna-llm" — DNA sequences are passed as a separate modality; text is question-only.

        is_sft=True  — assistant turn (reasoning + answer) is appended; collate fn masks prompt tokens.
        is_sft=False — assistant turn is omitted; answer is kept for reward computation (RL/GRPO).

        Returns a dict:
            {
                "prompt":       list[dict],   # chat-format messages
                "dna_sequences": list[str],   # ["ref", "var"] or ["", ""] for llm mode
                "answer":       str,          # ground-truth answer
            }
        """
        if self.model_name not in ("llm", "dna-llm"):
            raise ValueError(f"Unsupported model_name for DNA BioReason: {self.model_name!r}")

        # --- fallback for None / corrupt items ---
        if item is None:
            question = "What disease does this DNA sequence exhibit?"
            ref, var = ("", "") if self.model_name == "llm" else ("CTGA", "CTGA")
            if self.model_name == "llm":
                question_text = f"Reference sequence: CTGA\nVariant sequence: CTGA\nQuestion: {question}"
            else:
                question_text = question

            fallback_prompt = [
                {
                    "role": "user",
                    "content": [
                        {"type": "dna_reference", "text": None},
                        {"type": "dna_variant", "text": None},
                        {"type": "text", "text": question_text},
                    ],
                }
            ]
            if self.is_sft:
                fallback_prompt.append({
                    "role": "assistant",
                    "reasoning_content": "",
                    "content": [{"type": "text", "text": "<answer>No DNA sequence information available.</answer>"}],
                })
            return {
                "prompt": fallback_prompt,
                "dna_sequences": [ref, var],
                "answer": "No DNA sequence information available.",
                "class_weight": torch.tensor(1.0, dtype=torch.float32),
            }

        try:
            ref = item["reference_sequence"]
            var = item["variant_sequence"]
            question = item.get("question", "What disease does this DNA sequence exhibit?")
            answer = clean_kegg_answer(item["answer"])
            reasoning = item.get("reasoning", "")

            # LLM mode: DNA inlined into text, no separate modality slot.
            # DNA-LLM mode: question only in text, DNA passed via dna_sequences.
            if self.model_name == "llm":
                question_text = f"Reference sequence: {ref}\nVariant sequence: {var}\nQuestion: {question}"
                dna_sequences = ["", ""]
            else:
                question_text = question
                dna_sequences = [ref, var]

            prompt = [
                {
                    "role": "user",
                    "content": [
                        {"type": "dna_reference", "text": None},
                        {"type": "dna_variant", "text": None},
                        {"type": "text", "text": question_text.strip()},
                    ],
                }
            ]

            if self.is_sft:
                # SFT: append full assistant turn so collate fn can mask prompt tokens from loss.
                # Optionally truncate reasoning to stay within token budget.
                if self.use_reasoning_traces and reasoning and self.tokenizer is not None:
                    MAX_SEQ = 2048
                    prompt_text = question_text
                    prompt_len = len(self.tokenizer.encode(prompt_text, add_special_tokens=False))
                    answer_overhead = len(self.tokenizer.encode(
                        f"<answer>{answer.strip()}</answer>", add_special_tokens=False
                    ))
                    reasoning_budget = MAX_SEQ - prompt_len - answer_overhead - 16
                    if reasoning_budget > 0:
                        reasoning_ids = self.tokenizer.encode(reasoning, add_special_tokens=False)
                        if len(reasoning_ids) > reasoning_budget:
                            reasoning = self.tokenizer.decode(
                                reasoning_ids[:reasoning_budget], skip_special_tokens=True
                            )
                    else:
                        reasoning = ""

                prompt.append({
                    "role": "assistant",
                    "reasoning_content": reasoning.strip() if self.use_reasoning_traces else "",
                    "content": [{"type": "text", "text": f"<answer>{answer.strip()}</answer>"}],
                })

            # RL: no assistant turn — answer is kept as the reward signal only.
            class_weight = self.class_weights_map.get(answer.strip().lower(), 1.0)
            return {
                "prompt": prompt,
                "dna_sequences": dna_sequences,
                "answer": answer,
                "class_weight": torch.tensor(class_weight, dtype=torch.float32),
            }

        except Exception:
            question = "What disease does this DNA sequence exhibit?"
            ref, var = ("", "") if self.model_name == "llm" else ("CTGA", "CTGA")
            if self.model_name == "llm":
                question_text = f"Reference sequence: CTGA\nVariant sequence: CTGA\nQuestion: {question}"
            else:
                question_text = question

            fallback_prompt = [
                {
                    "role": "user",
                    "content": [
                        {"type": "dna_reference", "text": None},
                        {"type": "dna_variant", "text": None},
                        {"type": "text", "text": question_text},
                    ],
                }
            ]
            if self.is_sft:
                fallback_prompt.append({
                    "role": "assistant",
                    "reasoning_content": "",
                    "content": [{"type": "text", "text": "<answer>No information available.</answer>"}],
                })
            return {
                "prompt": fallback_prompt,
                "dna_sequences": [ref, var],
                "answer": "No information available.",
                "class_weight": torch.tensor(1.0, dtype=torch.float32),
            }

    def _process_dna_bioreason_variant_effect_coding(self, item):
        """
        Format a wanglab/variant_effect_coding example into the structured chat
        dict consumed by qwen_dna_collate_fn. Mirrors format_variant_effect_for_dna_llm
        / format_variant_effect_for_llm in BioReason/bioreason/dataset/variant_effect.py,
        with two deliberate deviations from BioReason's reference implementation:

          1. clean_variant_effect_coding_answer (this module) is applied to the
             actual training answer, not just a throwaway label-vocab copy — see
             its docstring for the BioReason bug this fixes.
          2. The assistant turn always wraps the answer as <answer>{answer}</answer>
             (never BioReason's literal "Answer: {answer}"), with an empty
             reasoning_content (this dataset has no reasoning/CoT field). This is
             necessary so the GRPO reward functions in trainer_grpo.py
             (xmlcount_reward, soft_format_reward, strict_format_reward,
             correctness_reward), which all key off <think>/<answer> tags, stay
             meaningful when this dataset is mixed with KEGG in the same GRPO run.

        model_name="llm"     — DNA sequences are inlined as plain text; dna_sequences is empty.
        model_name="dna-llm" — DNA sequences are passed as a separate modality; text is question-only.

        is_sft=True  — assistant turn (answer only, no reasoning) is appended.
        is_sft=False — assistant turn is omitted; answer is kept for reward computation (RL/GRPO).

        Returns the same dict shape as _process_dna_bioreason:
            {"prompt": list[dict], "dna_sequences": list[str], "answer": str, "class_weight": Tensor}
        """
        if self.model_name not in ("llm", "dna-llm"):
            raise ValueError(f"Unsupported model_name for DNA BioReason: {self.model_name!r}")

        if item is None:
            question = "Is this variant pathogenic or benign?"
            ref, var = ("", "") if self.model_name == "llm" else ("CTGA", "CTGA")
            if self.model_name == "llm":
                question_text = f"Reference sequence: CTGA\nVariant sequence: CTGA\nQuestion: {question}"
            else:
                question_text = question

            fallback_prompt = [
                {
                    "role": "user",
                    "content": [
                        {"type": "dna_reference", "text": None},
                        {"type": "dna_variant", "text": None},
                        {"type": "text", "text": question_text},
                    ],
                }
            ]
            if self.is_sft:
                fallback_prompt.append({
                    "role": "assistant",
                    "reasoning_content": "",
                    "content": [{"type": "text", "text": "<answer>No variant information available.</answer>"}],
                })
            return {
                "prompt": fallback_prompt,
                "dna_sequences": [ref, var],
                "answer": "No variant information available.",
                "class_weight": torch.tensor(1.0, dtype=torch.float32),
            }

        try:
            ref = item["reference_sequence"]
            var = item["variant_sequence"]
            question = item.get("question", "Is this variant pathogenic or benign?")
            answer = clean_variant_effect_coding_answer(item["answer"])

            if self.model_name == "llm":
                question_text = f"Reference sequence: {ref}\nVariant sequence: {var}\nQuestion: {question}"
                dna_sequences = ["", ""]
            else:
                question_text = question
                dna_sequences = [ref, var]

            prompt = [
                {
                    "role": "user",
                    "content": [
                        {"type": "dna_reference", "text": None},
                        {"type": "dna_variant", "text": None},
                        {"type": "text", "text": question_text.strip()},
                    ],
                }
            ]

            if self.is_sft:
                # No reasoning/CoT field in this dataset — assistant turn is answer-only,
                # but still tag-wrapped (see docstring) for reward-function compatibility.
                prompt.append({
                    "role": "assistant",
                    "reasoning_content": "",
                    "content": [{"type": "text", "text": f"<answer>{answer.strip()}</answer>"}],
                })

            class_weight = self.class_weights_map.get(answer.strip().lower(), 1.0)
            return {
                "prompt": prompt,
                "dna_sequences": dna_sequences,
                "answer": answer,
                "class_weight": torch.tensor(class_weight, dtype=torch.float32),
            }

        except Exception:
            question = "Is this variant pathogenic or benign?"
            ref, var = ("", "") if self.model_name == "llm" else ("CTGA", "CTGA")
            if self.model_name == "llm":
                question_text = f"Reference sequence: CTGA\nVariant sequence: CTGA\nQuestion: {question}"
            else:
                question_text = question

            fallback_prompt = [
                {
                    "role": "user",
                    "content": [
                        {"type": "dna_reference", "text": None},
                        {"type": "dna_variant", "text": None},
                        {"type": "text", "text": question_text},
                    ],
                }
            ]
            if self.is_sft:
                fallback_prompt.append({
                    "role": "assistant",
                    "reasoning_content": "",
                    "content": [{"type": "text", "text": "<answer>No information available.</answer>"}],
                })
            return {
                "prompt": fallback_prompt,
                "dna_sequences": [ref, var],
                "answer": "No information available.",
                "class_weight": torch.tensor(1.0, dtype=torch.float32),
            }

    def _process_dna_bioreason_variant_effect_non_snv(self, item):
        """
        Format a wanglab/variant_effect_non_snv example into the structured chat
        dict consumed by qwen_dna_collate_fn. Mirrors format_variant_effect_for_dna_llm
        / format_variant_effect_for_llm in BioReason/bioreason/dataset/variant_effect.py,
        applied to this dataset's variant_effect_non_snv branch in train_dna_qwen.py
        (train_dna_qwen.py:444-461), with the same tag-format deviation documented in
        _process_dna_bioreason_variant_effect_coding above (forced <answer>...</answer>
        wrapping, empty reasoning_content, for GRPO reward-function compatibility).

        Field mapping deviation from _process_dna_bioreason_variant_effect_coding:
        this dataset's raw HF schema uses "mutated_sequence" where KEGG/coding use
        "variant_sequence" (confirmed directly against wanglab/variant_effect_non_snv's
        real columns: question, answer, reference_sequence, mutated_sequence,
        cleaned_pathogenicity, __index_level_0__). BioReason renames this column at
        the dataset level (train_dna_qwen.py:448); here it is read directly by its
        raw name instead, since items arrive as individual dicts, not a
        datasets.Dataset that can be column-renamed in bulk.

        answer cleaning uses clean_variant_effect_non_snv_answer (this module),
        which matches BioReason's own train-time cleaning exactly (unlike the coding
        dataset, BioReason's non_snv path applies this cleaner to the real training
        answer already — see that function's docstring). Confirmed against 1000 real
        rows: 228 are a bare pathogenicity string (e.g. "benign", cleaning is a
        no-op), 772 are "pathogenicity; ['term', ...]", 0 are any other shape.
        The dataset's own "answer" field (not "cleaned_pathogenicity") is used as
        the training target, i.e. pathogenicity + consequence terms, matching
        BioReason's real training behavior — "cleaned_pathogenicity" is a
        pathogenicity-only convenience column the dataset authors provide but
        BioReason itself does not train on.

        model_name="llm"     — DNA sequences are inlined as plain text; dna_sequences is empty.
        model_name="dna-llm" — DNA sequences are passed as a separate modality; text is question-only.

        is_sft=True  — assistant turn (answer only, no reasoning) is appended.
        is_sft=False — assistant turn is omitted; answer is kept for reward computation (RL/GRPO).

        Returns the same dict shape as _process_dna_bioreason:
            {"prompt": list[dict], "dna_sequences": list[str], "answer": str, "class_weight": Tensor}
        """
        if self.model_name not in ("llm", "dna-llm"):
            raise ValueError(f"Unsupported model_name for DNA BioReason: {self.model_name!r}")

        if item is None:
            question = "Is this variant pathogenic or benign, and what condition does it cause?"
            ref, var = ("", "") if self.model_name == "llm" else ("CTGA", "CTGA")
            if self.model_name == "llm":
                question_text = f"Reference sequence: CTGA\nVariant sequence: CTGA\nQuestion: {question}"
            else:
                question_text = question

            fallback_prompt = [
                {
                    "role": "user",
                    "content": [
                        {"type": "dna_reference", "text": None},
                        {"type": "dna_variant", "text": None},
                        {"type": "text", "text": question_text},
                    ],
                }
            ]
            if self.is_sft:
                fallback_prompt.append({
                    "role": "assistant",
                    "reasoning_content": "",
                    "content": [{"type": "text", "text": "<answer>No variant information available.</answer>"}],
                })
            return {
                "prompt": fallback_prompt,
                "dna_sequences": [ref, var],
                "answer": "No variant information available.",
                "class_weight": torch.tensor(1.0, dtype=torch.float32),
            }

        try:
            ref = item["reference_sequence"]
            var = item["mutated_sequence"]
            question = item.get(
                "question", "Is this variant pathogenic or benign, and what condition does it cause?"
            )
            answer = clean_variant_effect_non_snv_answer(item["answer"])

            if self.model_name == "llm":
                question_text = f"Reference sequence: {ref}\nVariant sequence: {var}\nQuestion: {question}"
                dna_sequences = ["", ""]
            else:
                question_text = question
                dna_sequences = [ref, var]

            prompt = [
                {
                    "role": "user",
                    "content": [
                        {"type": "dna_reference", "text": None},
                        {"type": "dna_variant", "text": None},
                        {"type": "text", "text": question_text.strip()},
                    ],
                }
            ]

            if self.is_sft:
                # No reasoning/CoT field in this dataset — assistant turn is answer-only,
                # but still tag-wrapped (see docstring) for reward-function compatibility.
                prompt.append({
                    "role": "assistant",
                    "reasoning_content": "",
                    "content": [{"type": "text", "text": f"<answer>{answer.strip()}</answer>"}],
                })

            class_weight = self.class_weights_map.get(answer.strip().lower(), 1.0)
            return {
                "prompt": prompt,
                "dna_sequences": dna_sequences,
                "answer": answer,
                "class_weight": torch.tensor(class_weight, dtype=torch.float32),
            }

        except Exception:
            question = "Is this variant pathogenic or benign, and what condition does it cause?"
            ref, var = ("", "") if self.model_name == "llm" else ("CTGA", "CTGA")
            if self.model_name == "llm":
                question_text = f"Reference sequence: CTGA\nVariant sequence: CTGA\nQuestion: {question}"
            else:
                question_text = question

            fallback_prompt = [
                {
                    "role": "user",
                    "content": [
                        {"type": "dna_reference", "text": None},
                        {"type": "dna_variant", "text": None},
                        {"type": "text", "text": question_text},
                    ],
                }
            ]
            if self.is_sft:
                fallback_prompt.append({
                    "role": "assistant",
                    "reasoning_content": "",
                    "content": [{"type": "text", "text": "<answer>No information available.</answer>"}],
                })
            return {
                "prompt": fallback_prompt,
                "dna_sequences": [ref, var],
                "answer": "No information available.",
                "class_weight": torch.tensor(1.0, dtype=torch.float32),
            }

    def __iter__(self):
        # Rank-aware initialization
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            rank = 0
            world_size = 1

        # Determine Worker Info for Seeding
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # Unique seed per worker
            worker_id = worker_info.id
            # Combine random state of main process (random.randint) with worker_id
            # This assumes random state is forked or set deterministically.
            seed_offset = 42 + rank * 1000 + worker_id * 100
        else:
            seed_offset = 0

        # Create iterators FRESH (Fixes duplication across workers)
        iterators = {}

        # Phase 3.5: deterministic shuffle + per-rank sharding for streaming
        # HF datasets. Two runs with the same base seed must produce identical
        # first-N sample ids per rank. Base seed comes from cfg.seed when set,
        # falls back to 0 so behavior stays stable when the run wasn't seeded.
        base_seed = int(getattr(self.model_config, "seed", 0) or 0)
        for k, v in self.datasets_objs.items():
            dataset_stream = v

            # Per-rank sharding FIRST (so two ranks never see the same sample),
            # THEN shuffle (so each rank slice is randomized deterministically).
            # If shard() raises, every rank would otherwise fall through and
            # iterate the full stream → silent cross-rank duplication, which
            # is a correctness bug masquerading as throughput. Fail loud.
            _used_filter_split_for_this_stream = False
            if world_size > 1 and hasattr(dataset_stream, "shard"):
                # HF IterableDataset.shard() requires n_shards >= world_size.
                # With few JSON files (ts_qa = 1, ts_instruction = 3) and
                # world_size = 12, shard() raises IndexError in
                # _merge_gen_kwargs because gen_kwargs_list[index] is empty.
                # When that happens, fall back to an index-strided filter
                # which yields one sample to one rank in round-robin fashion
                # — still no cross-rank duplication, just a different split.
                n_shards = getattr(dataset_stream, "n_shards", None)
                if (
                    n_shards is not None
                    and isinstance(n_shards, int)
                    and n_shards < world_size
                    and hasattr(dataset_stream, "filter")
                ):
                    logger.info(
                        f"[DATA SHARD] {k}: n_shards={n_shards} < world_size={world_size}, "
                        f"using filter-strided split instead of .shard()"
                    )
                    _rank = rank
                    _ws = world_size
                    dataset_stream = dataset_stream.filter(
                        lambda _ex, idx, _r=_rank, _w=_ws: (idx % _w) == _r,
                        with_indices=True,
                    )
                    _used_filter_split_for_this_stream = True
                else:
                    dataset_stream = dataset_stream.shard(
                        num_shards=world_size, index=rank
                    )

            # Apply Shuffle dynamically here!
            if hasattr(dataset_stream, "shuffle"):
                step_seed = base_seed + seed_offset

                # Buffer size tuning: for filter-strided streams (n_shards <
                # world_size), shuffle(buffer_size=10000) forces reading 10K
                # filtered records before the first batch. With filter
                # rejecting (world_size-1)/world_size of records, that's
                # 10K × world_size = ~120K records read per rank just to
                # warm up — dominating step time on Lustre. Smaller buffer
                # for these streams trades shuffle quality for first-batch
                # latency. Image (WebDataset) and properly-sharded HF
                # datasets keep the full 10K buffer.
                _buffer_size = (
                    256 if _used_filter_split_for_this_stream else 10000
                )
                try:
                    # Try HF datasets style first (buffer_size, seed)
                    dataset_stream = dataset_stream.shuffle(
                        buffer_size=_buffer_size, seed=step_seed
                    )
                except TypeError:
                    # WebDataset style - already shuffled in pipeline, skip
                    pass

            if hasattr(dataset_stream, "__iter__") or isinstance(dataset_stream, list | range):
                iterators[k] = iter(dataset_stream)
            else:
                iterators[k] = iter(dataset_stream)

        # Rank-0 manifest log: expected samples/epoch from datasets_config.json.
        if rank == 0 and (worker_info is None or worker_info.id == 0):
            for k in list(self.datasets_objs.keys()):
                info = self.datasets_map.get(k, {})
                size = info.get("size_formatted") or info.get("num_examples")
                if size:
                    logger.info(
                        f"[streaming manifest] {k}: expected ~{size} samples/epoch"
                    )

        # Weighted Sampling Setup (Hierarchical)

        # Weighted Sampling Setup (Hierarchical)
        # 1. Group by Modality
        modality_groups = {
            "image": [],
            "graph": [],
            "table": [],
            "time_series": [],
            "geometry": [],
            "text": [],
        }

        for name in list(self.datasets_objs.keys()):
            dataset_info = self.datasets_map[name]
            target = _get_modality(dataset_info, name)
            w = dataset_info.get("weight", 1.0)
            modality_groups[target].append((name, w))

        # Filter out empty modalities
        active_modalities = [m for m, items in modality_groups.items() if items]
        logger.info(f"Active Modalities for Sampling: {active_modalities}")

        # Normalize weights within groups
        group_samplers = {}
        for m in active_modalities:
            items = modality_groups[m]
            names = [x[0] for x in items]
            weights = [x[1] for x in items]
            # Normalize
            total = sum(weights)
            if total > 0:
                weights = [x / total for x in weights]
            group_samplers[m] = {"names": names, "weights": weights}
            logger.info(f"  Modality {m}: {names} (Weights: {weights})")

        # --- STRICT VALIDATION ---
        # 1. Dataset Coverage Check
        # Ensure every dataset configured with skip=False is actually present in the active streams.
        missing_datasets = []
        for name, info in self.datasets_map.items():
            should_skip = info.get("skip", False)
            if not should_skip:
                if name not in self.datasets_objs:
                    missing_datasets.append(name)

        if missing_datasets:
            err_msg = f"CRITICAL: The following datasets are configured as active (skip=False) but failed to load: {missing_datasets}. Please check local paths or HF_TOKEN."
            logger.error(err_msg)
            raise RuntimeError(err_msg)

        # Seeded RNG
        rng = random.Random()
        for _ in range(self.max_steps):
            # 1. Sample Modality (Uniform)
            # This ensures equal representation of active modalities
            mod_choice = rng.choice(active_modalities)

            # 2. Sample Dataset (Weighted within Modality)
            sampler = group_samplers[mod_choice]
            choice = rng.choices(sampler["names"], weights=sampler["weights"], k=1)[0]

            # Lookup handler
            handler_key = self.datasets_map[choice]["handler"]

            # Implicit Helper for new handlers
            if handler_key == "ts_qa":
                processor = self._process_ts_qa
            elif handler_key == "ts_mmd":
                processor = self._process_ts_mmd
            elif handler_key == "ts_time_mmd":
                processor = self._process_ts_time_mmd
            elif handler_key == "ts_caption":
                processor = self._process_ts_caption
            elif handler_key == "ts_instruction":
                processor = self._process_ts_instruction

            # Table Handlers
            elif handler_key == "table_reasoning":
                processor = self._process_table_reasoning
            elif handler_key == "table_instruction":
                processor = self._process_table_instruction
            elif handler_key == "table_structure":
                processor = self._process_table_structure

            # Graph Handlers
            elif handler_key == "graph_captioning":
                processor = self._process_graph_captioning
            elif handler_key == "graph_grounding":
                processor = self._process_graph_grounding
            elif handler_key == "graph_instruction":
                processor = self._process_graph_instruction
            elif handler_key == "graph_crystal":
                processor = self._process_graph_crystal
            elif handler_key == "graph_circuit":
                processor = self._process_graph_circuit

            # Geometry Handlers
            elif handler_key.startswith("geo_physics"):
                processor = self._process_geo_physics
            elif handler_key == "geo_mat_tomo":
                processor = self._process_geo_mat_tomo
            elif handler_key == "geo_pde":
                processor = self._process_geo_pde

            # DNA Handlers
            elif handler_key == "dna_bioreason":
                processor = self._process_dna_bioreason
            elif handler_key == "dna_bioreason_variant_effect_coding":
                processor = self._process_dna_bioreason_variant_effect_coding
            elif handler_key == "dna_bioreason_variant_effect_non_snv":
                processor = self._process_dna_bioreason_variant_effect_non_snv

            # Existing specific overrides
            elif handler_key == "image_pixmo":
                processor = self._process_image_pixmo
            elif handler_key == "image_points":
                processor = self._process_image_points
            elif handler_key == "text_sft":
                processor = self._process_text_sft

            else:
                processor = self.handlers_map.get(handler_key, self._process_image)

            iterator = iterators[choice]

            # Homogeneous Batch Logic
            count = 0
            retries = 0
            while count < self.batch_size:
                try:
                    item = None
                    try:
                        item = next(iterator)

                        # Fix Relative and Stale Absolute Paths
                        if isinstance(item, dict) and "local_path" in item:
                            lp = item["local_path"]
                            d_config = self.datasets_map.get(choice)

                            if lp and d_config and "local_path" in d_config:
                                config_root = d_config["local_path"]

                                # Case A: Relative Path -> Prepend Config Root
                                if not os.path.isabs(lp):
                                    item["local_path"] = os.path.join(config_root, lp)

                                # Case B: Absolute Path that doesn't exist (Stale Metadata)
                                elif not os.path.exists(lp):
                                    # logger.warning(f"DEBUG: Stale path detected: {lp}")
                                    # Try to repair path if it looks like a standard structure (images/...)
                                    if "images/" in lp:
                                        # Extract everything after the last 'images/'
                                        # e.g. /old/.../images/foo.jpg -> images/foo.jpg
                                        suffix = lp.split("images/")[-1]
                                        new_path = os.path.join(config_root, "images", suffix)

                                        # logger.warning(f"DEBUG: Trying repair: {new_path}")

                                        if os.path.exists(new_path):
                                            item["local_path"] = new_path
                                            # logger.warning(f"DEBUG: Repair successful: {new_path}")

                                        # Fallback: Try simple basename
                                        elif os.path.exists(
                                            os.path.join(config_root, os.path.basename(lp))
                                        ):
                                            item["local_path"] = os.path.join(
                                                config_root, os.path.basename(lp)
                                            )

                                        else:
                                            logger.warning(
                                                f"DEBUG: Repair failed. Path not found: {new_path}"
                                            )
                    except StopIteration:
                        # Promote to WARNING so retries are visible — silent retries
                        # mask the "persistently empty" failure mode that bites
                        # filter-strided sweep cells with n_shards < world_size.
                        logger.warning(
                            f"[RANK={rank}] Stream {choice} exhausted (retry {retries + 1}/5). "
                            f"Restarting from iterable."
                        )
                        if choice in self.datasets_objs:
                            iterators[choice] = iter(self.datasets_objs[choice])
                        else:
                            # Fallback if somehow missing
                            logger.warning(
                                f"[RANK={rank}] Warning: Stream {choice} has no stored iterable. Cannot restart."
                            )
                            raise RuntimeError(
                                f"[RANK={rank}] Stream {choice} exhausted and cannot be restarted."
                            ) from None

                        iterator = iterators[choice]
                        retries += 1
                        if retries > 5:
                            if not self.allow_dummy_data:
                                raise RuntimeError(
                                    f"[RANK={rank}] Stream {choice} is persistently empty or failing after 5 restarts."
                                ) from None
                            else:
                                logger.warning(
                                    f"[RANK={rank}] Warning: Stream {choice} empty. Yielding None."
                                )
                                item = None
                                break  # Exit try/except, proceed to processor(None)
                        continue
                    except (OSError, FileNotFoundError) as e:
                        logger.warning(
                            f"[RANK={rank}] Warning: Stream {choice} raised FileNotFoundError/IOError: {e}. Skipping item."
                        )
                        continue
                    except Exception as e:
                        logger.warning(f"[RANK={rank}] Warning: Error fetching item from stream {choice}: {e}. Retrying...")
                        time.sleep(1)
                        continue

                    try:
                        # check if `item` is None before processing
                        used_dummy_fallback = False
                        if item is None:
                            # When fallback_dummy is allowed, synthesize a modality-correct
                            # dummy here instead of routing through per-processor None
                            # handling (which is inconsistent across processors and was
                            # always preempted by this RuntimeError anyway).
                            if self.allow_dummy_data and self.datasets_map[choice].get(
                                "fallback_dummy", False
                            ):
                                modality = _get_modality(self.datasets_map[choice], choice)
                                data_tensor = make_dummy_batch(modality)
                                caption = f"Dummy {modality.value}"
                                metadata_str = f"[{choice}] (Dummy Fallback)"
                                used_dummy_fallback = True
                            else:
                                raise RuntimeError(
                                    f"[RANK={rank}] Item is None"
                                )
                        else:
                            processed_result = processor(item)

                        if not used_dummy_fallback:
                            if processed_result is None:
                                # Processor returned bare None on a real item (legacy
                                # behavior in a few graph/table handlers) — treat as
                                # skip rather than crashing on len(None) below.
                                logger.warning(
                                    f"[RANK={rank}] Processor for {choice} returned None on non-None item. Skipping."
                                )
                                continue
                            elif isinstance(processed_result, dict) and "dna" in handler_key:
                                # DNA BioReason handlers return a structured dict
                                # (prompt/dna_sequences/answer/class_weight), not the
                                # generic (data_tensor, caption, metadata_str) 3-tuple
                                # every other modality returns -- handle it before the
                                # tuple-length checks below, whose
                                # `elif len(processed_result) == 3` would otherwise try
                                # to unpack this dict's *keys* as three values. caption/
                                # metadata_str are placeholders here -- the dedicated
                                # Modality.DNA branch further below overwrites both from
                                # dna_result directly and never reads these placeholders.
                                data_tensor = processed_result
                                caption = ""
                                metadata_str = f"[{choice}] (DNA BioReason)"
                            elif len(processed_result) == 3:
                                data_tensor, caption, metadata_str = processed_result
                            elif len(processed_result) == 2:
                                data_tensor, caption = processed_result
                                metadata_str = f"[{choice}] (Legacy Handler)"
                            else:
                                raise RuntimeError(
                                    f"[RANK={rank}] Handler {choice} returned unexpected tuple len {len(processed_result)}"
                                )

                    except RuntimeError as e:
                        logger.warning(e)
                        # Strict Mode: Skip items causing data errors (404s, None, Corrupt)
                        err_str = str(e)
                        if any(
                            x in err_str
                            for x in [
                                "Download Failed",
                                "No image found",
                                "Corrupt",
                                "Item is None",
                            ]
                        ):
                            logger.warning(f"[RANK={rank}] Warning: Skipping item due to data error: {e}")
                            continue
                        else:
                            raise e

                    # DEBUG ITER DATA
                    # if random.random() < 0.1:
                    if item is None:
                        if not self.allow_dummy_data:
                            raise RuntimeError(
                                f"[RANK={rank}] Stream {choice} yielded NONE (Dummy Data) and allow_dummy_data=False."
                            )
                        logger.debug(f"[RANK={rank}] DEBUG ITER: Stream {choice} yielded NONE (Dummy).")

                    elif data_tensor is None and not (
                        _get_modality(self.datasets_map[choice], choice) is Modality.DNA
                        and self.model_name == "llm"
                    ):
                        # Processed result yielded explicit None (e.g. skipped item).
                        # Exception: DNA in "llm" mode legitimately returns data_tensor=None
                        # — sequences are inlined into caption as plain text, no modality
                        # slot needed — that is not a skip signal.
                        logger.warning(
                            f"[RANK={rank}] Warning: Processor for {choice} returned None tensor. Skipping."
                        )
                        continue

                    else:
                        if data_tensor is None:
                            shape_str = "None (DNA llm-mode: inlined as text)"
                        elif isinstance(data_tensor, dict):
                            if "x" in data_tensor:
                                shape_str = str(data_tensor["x"].shape)
                            elif "input_ids" in data_tensor:
                                shape_str = str(data_tensor["input_ids"].shape)
                            else:
                                shape_str = f"Keys: {list(data_tensor.keys())}"
                        else:
                            shape_str = str(data_tensor.shape)
                        logger.debug(
                            f"DEBUG ITER: Stream {choice} yielded VALID. Tensor: {shape_str}"
                        )

                    # Union Schema Construction
                    example = {
                        "text": "",
                        "_metadata": metadata_str,  # INJECTED METADATA
                    }

                    modality = _get_modality(self.datasets_map[choice], choice)
                    if modality is Modality.IMAGE:
                        example["image"] = data_tensor
                    elif modality is Modality.GRAPH:
                        if isinstance(data_tensor, dict):
                            # Nest graph items; ensure edge_index is LongTensor.
                            example["graph"] = {
                                "x": data_tensor.get("x", torch.zeros(128, 32)),
                                "edge_index": data_tensor.get("edge_index")
                                if data_tensor.get("edge_index") is not None
                                else torch.empty((2, 0), dtype=torch.long),
                            }
                        else:
                            # Fallback for dummy
                            example["graph"] = {
                                "x": data_tensor,
                                "edge_index": torch.empty((2, 0), dtype=torch.long),
                            }
                    elif modality is Modality.TIME_SERIES:
                        example["time_series"] = data_tensor
                    elif modality is Modality.TABLE:
                        # Tapas returns dict or tensor; MultimodalCollator handles dicts.
                        # Prefer input_ids when present (collator-friendly), else pass through.
                        if isinstance(data_tensor, dict) and "input_ids" in data_tensor:
                            example["table"] = data_tensor["input_ids"]
                        else:
                            example["table"] = data_tensor
                    elif modality is Modality.GEOMETRY:
                        example["geometry"] = data_tensor
                    elif modality is Modality.DNA:
                        dna_result = data_tensor

                        # 1. Render prompt list -> flat strings that match the ChatML
                        #    format the tokenizer and interleaved merge expect.
                        full_text, prompt_text = self._render_dna_prompt_text(
                            dna_result["prompt"]
                        )

                        # 2. Tokenize text -> example["text"] (consumed by the backbone
                        #    embedding layer; contains <dna_ref_start><dna_ref_end> and
                        #    <dna_var_start><dna_var_end> placeholders for DNA-LLM mode,
                        #    or the inlined sequences for LLM mode).
                        example["text"] = self.tokenizer(
                            full_text,
                            return_tensors="pt",
                            padding=False,
                            truncation=True,
                            max_length=2048,
                            add_special_tokens=False,
                        ).input_ids.squeeze(0)

                        # 3. Tokenize DNA sequences -> example["dna"] with the sub-keys
                        #    "dna_reference" and "dna_variant" that
                        #    _process_multimodal_embeddings checks in model.py.
                        #    LLM mode: no dna key -- sequences are already in full_text.
                        ref_seq, var_seq = dna_result["dna_sequences"]
                        if self.dna_tokenizer is not None and ref_seq and var_seq:
                            # padding="max_length" (not True): a single unbatched string
                            # makes padding=True a no-op, so the span length used to be
                            # min(actual_token_count, max_dna), varying per example. The
                            # interleaved merge path assumes a FIXED length per DNA span
                            # (DNAEncoder.tokens_per_instance(), now itself set from this
                            # same max_dna_length -- see src/model.py's DNAEncoder(...)
                            # construction), so every span must always be exactly max_dna
                            # tokens long.
                            max_dna = self.model_config.max_dna_length
                            ref_enc = self.dna_tokenizer(
                                ref_seq,
                                return_tensors="pt",
                                padding="max_length",
                                truncation=True,
                                max_length=max_dna,
                            )
                            var_enc = self.dna_tokenizer(
                                var_seq,
                                return_tensors="pt",
                                padding="max_length",
                                truncation=True,
                                max_length=max_dna,
                            )
                            example["dna"] = {
                                "dna_reference": {
                                    "input_ids": ref_enc["input_ids"].squeeze(0),
                                    "attention_mask": ref_enc["attention_mask"].squeeze(0),
                                },
                                "dna_variant": {
                                    "input_ids": var_enc["input_ids"].squeeze(0),
                                    "attention_mask": var_enc["attention_mask"].squeeze(0),
                                },
                            }

                        # 4. _metadata: controls label masking in the merge function.
                        #
                        #    "P T"            -- SFT/GRPO: mask the first P prompt tokens,
                        #                       compute loss only on the T answer tokens.
                        #    "[dna_bioreason]" -- Projector: compute causal LM loss over
                        #                       the entire sequence (question + answer),
                        #                       giving the projector a rich reconstruction
                        #                       signal rather than a narrow answer-only one.
                        #
                        # NOTE: this block intentionally does NOT reuse the generic
                        # `if self.model_config.is_interleaved_qa: example["_metadata"][0]/[1]`
                        # block further below (which assumes _metadata is already a
                        # [prompt, target] list) -- that block has no isinstance guard and
                        # would character-index this DNA branch's string-typed _metadata,
                        # corrupting it. yield+continue below deliberately skips that block
                        # entirely, matching how DNA metadata is computed here instead, from
                        # full_text/prompt_text directly.
                        if self.task == "bioreason_projector":
                            example["_metadata"] = "[dna_bioreason]"
                        elif self.model_config.is_interleaved_qa:
                            joint_ids = self.tokenizer(
                                full_text,
                                return_tensors="pt",
                                padding=False,
                                truncation=True,
                                max_length=2048,
                                add_special_tokens=False,
                            ).input_ids.squeeze(0)
                            prompt_ids = self.tokenizer(
                                prompt_text,
                                return_tensors="pt",
                                padding=False,
                                truncation=True,
                                max_length=2048,
                                add_special_tokens=False,
                            ).input_ids.squeeze(0)
                            prompt_tokens = prompt_ids.shape[0]
                            target_tokens = joint_ids.shape[0] - prompt_tokens
                            example["_metadata"] = f"{prompt_tokens} {target_tokens}"
                        else:
                            example["_metadata"] = "[dna_bioreason]"

                        # 5. Keep answer for reward / eval -- a real top-level dict key,
                        # not embedded in _metadata (see trainer_grpo.py's
                        # batch.get("answer", ...), which reads this directly).
                        example["answer"] = dna_result["answer"]

                        # 6. Per-sample class weight (1.0 if weighting disabled).
                        example["class_weight"] = dna_result.get(
                            "class_weight", torch.tensor(1.0, dtype=torch.float32)
                        )

                        yield example
                        count += 1
                        continue
                    elif modality is Modality.TEXT:
                        pass  # text is handled by the tokenizer block below
                    else:
                        raise ValueError(
                            f"Unknown modality {modality!r} for dataset {choice!r}"
                        )

                    # Tokenize
                    tokens = self.tokenizer(
                        caption,
                        return_tensors="pt",
                        padding=False,
                        truncation=True,
                        max_length=2048,
                    ).input_ids.squeeze(0)
                    example["text"] = tokens
                    if self.model_config.is_interleaved_qa:
                        prompt = example["_metadata"][0]
                        target = example["_metadata"][1]
                        # Tokenize jointly to avoid BPE boundary mismatch:
                        # len(tokenize(prompt)) + len(tokenize(target)) != len(tokenize(prompt+target))
                        joint_ids = self.tokenizer(
                            prompt + target,
                            return_tensors="pt",
                            padding=False,
                            truncation=True,
                            max_length=2048,
                        ).input_ids.squeeze(0)
                        prompt_ids = self.tokenizer(
                            prompt,
                            return_tensors="pt",
                            padding=False,
                            truncation=True,
                            max_length=2048,
                        ).input_ids.squeeze(0)
                        prompt_tokens = prompt_ids.shape[0]
                        target_tokens = joint_ids.shape[0] - prompt_tokens
                        example["_metadata"] = (
                            f"{prompt_tokens} {target_tokens}"  # Store token counts in metadata for interleaved QA
                        )
                    else:
                        # Non-interleaved (prefix) path: the model masks the
                        # first `_prompt_len` text tokens to -100 so the loss is
                        # answer-only, matching what the interleaved path gets
                        # from new_prompt_len. 0 => supervise everything, which
                        # is right for captioning-style data with no Q/A split.
                        meta = example.get("_metadata")
                        prompt_len = 0
                        if isinstance(meta, (list, tuple)) and len(meta) == 2 and meta[0]:
                            prompt_len = int(
                                self.tokenizer(
                                    meta[0],
                                    return_tensors="pt",
                                    padding=False,
                                    truncation=True,
                                    max_length=2048,
                                ).input_ids.squeeze(0).shape[0]
                            )
                        example["_prompt_len"] = prompt_len

                    yield example
                    count += 1

                except StopIteration:
                    # Restart logic
                    iterators[choice] = self._dummy_generator()  # Fallback
                    iterator = iterators[choice]
                    continue
                except Exception as e:
                    if not self.allow_dummy_data:
                        raise e
                    continue
