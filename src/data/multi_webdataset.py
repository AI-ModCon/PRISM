#!/usr/bin/env python3
"""
multi_webdataset.py - Multi-dataset WebDataset loader with weighted sampling

Provides a unified interface for loading multiple WebDataset sources from DAOS
with configurable sampling weights.

Features:
- Load multiple WebDataset directories
- Configurable per-dataset sampling weights
- Automatic shard distribution across nodes
- Compatible with PyTorch DataLoader

Usage:
    from src.data.multi_webdataset import MultiWebDataset, load_daos_config

    # Load configuration
    config = load_daos_config("src/conf/data/daos_datasets.yaml")

    # Create multi-dataset loader
    dataset = MultiWebDataset(
        config=config,
        groups=["pixmo", "s1mmalign"],  # or "all"
        daos_mount="/tmp/<user>/AuroraGPT/prism_training_data",
        world_size=24,
        rank=0,
    )

    # Iterate
    for sample in dataset:
        image = sample['image']
        caption = sample['caption']
        metadata = sample['metadata']
"""

import glob
import json
import logging
import os
import random
from collections.abc import Iterator
from typing import Any

import yaml

from .image_transforms import build_image_transform

# Distributed training support
try:
    import torch
    import torch.distributed as dist

    HAS_TORCH_DISTRIBUTED = True
except ImportError:
    dist = None
    HAS_TORCH_DISTRIBUTED = False

try:
    import webdataset as wds

    HAS_WEBDATASET = True
except ImportError:
    HAS_WEBDATASET = False
    wds = None

logger = logging.getLogger(__name__)


def _identity(src):
    """Identity nodesplitter — passes shards through unchanged.

    Defined at module level (not as a lambda) so it's picklable,
    which is required when DataLoader uses multiprocessing_context='spawn'.
    """
    return src


class _SampleStride:
    """Keep sample ``i`` only when ``i % stride == offset``.

    Used when a dataset has fewer shards than ranks, where splitting by shard
    cannot give every rank a disjoint subset. Each rank instead opens every
    shard and takes a strided slice of the sample stream, which keeps the
    ranks disjoint at sample granularity.

    A class rather than a closure so it survives pickling under DataLoader's
    ``multiprocessing_context='spawn'``, and so each worker gets its own
    counter rather than sharing one via a captured cell.

    Note the counter is per-instance: with ``num_workers > 1`` webdataset
    gives each worker its own copy of the pipeline, and workers already split
    the shard list between themselves, so per-worker counting is correct.
    Verified against the real 3-shard SciTS val split at 12 ranks with
    num_workers in {0, 1, 2, 4}: 1440 distinct keys, zero duplication either
    across ranks or within a rank, in every case.
    """

    __slots__ = ("offset", "stride", "_i")

    def __init__(self, offset: int, stride: int):
        self.offset = offset
        self.stride = stride
        self._i = 0

    def __call__(self, sample) -> bool:
        keep = (self._i % self.stride) == self.offset
        self._i += 1
        return keep


class _DeterministicRandomMix:
    """Deterministic finite replacement for WebDataset RandomMix.

    Each source is consumed without replacement. Source choice is weighted for
    ordering only; exhausted sources are dropped.
    """

    def __init__(self, datasets, probs=None, longest=False, seed: int = 0):
        self.datasets = datasets
        self.probs = probs
        self.longest = longest
        self.seed = seed
        self.epoch = -1

    def __call__(self):
        return iter(self)

    def __iter__(self):
        self.epoch += 1
        rng = random.Random(self.seed + self.epoch)
        sources = [iter(d) for d in self.datasets]
        probs = list(self.probs) if self.probs is not None else [1.0] * len(sources)

        while sources:
            total = sum(probs)
            if total <= 0:
                probs = [1.0] * len(sources)
                total = float(len(sources))

            threshold = rng.random() * total
            cumulative = 0.0
            index = len(probs) - 1
            for i, prob in enumerate(probs):
                cumulative += prob
                if threshold <= cumulative:
                    index = i
                    break

            try:
                yield next(sources[index])
            except StopIteration:
                if not self.longest:
                    return
                del sources[index]
                del probs[index]


def load_daos_config(config_path: str) -> dict:
    """Load DAOS dataset configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_datasets_for_groups(
    config: dict,
    groups: str | list[str],
    weight_overrides: dict[str, float] | None = None,
    proportion_overrides: dict[str, float] | None = None,
) -> list[dict]:
    """
    Get list of datasets for specified groups with their weights and proportions.

    Args:
        config: Loaded YAML configuration
        groups: Either a preset name (e.g., "all"), a group name,
                or list of group names
        weight_overrides: Optional dict of dataset_name -> weight overrides
        proportion_overrides: Optional dict of dataset_name -> proportion overrides
            (0.0-1.0, controls what fraction of shards to use)

    Returns:
        List of dataset dicts with 'name', 'path', 'weight', 'samples', 'shards', 'proportion'
    """
    datasets = []
    weight_overrides = weight_overrides or {}
    proportion_overrides = proportion_overrides or {}

    # Check if groups is a preset
    if isinstance(groups, str):
        if groups in config.get("presets", {}):
            preset = config["presets"][groups]
            group_list = preset["groups"]
            # Apply preset weight overrides first, then CLI overrides
            preset_weight_overrides = preset.get("weight_overrides", {})
            weight_overrides = {**preset_weight_overrides, **weight_overrides}
            # Apply preset proportion overrides first, then CLI overrides
            preset_proportion_overrides = preset.get("proportion_overrides", {})
            proportion_overrides = {
                **preset_proportion_overrides,
                **proportion_overrides,
            }
        else:
            group_list = [groups]
    else:
        group_list = groups

    # Collect datasets from each group
    for group_name in group_list:
        if group_name not in config.get("groups", {}):
            logger.warning(f"Unknown group: {group_name}")
            continue

        group = config["groups"][group_name]
        for ds_name, ds_info in group.get("datasets", {}).items():
            # Check if dataset should be skipped
            if ds_info.get("skip", False):
                skip_reason = ds_info.get("skip_reason", "Configured to skip")
                logger.warning(f"Skipping dataset {ds_name}: {skip_reason}")
                continue

            weight = weight_overrides.get(ds_name, ds_info.get("weight", 1.0))

            # Skip datasets with weight=0 (excluded from this training phase)
            if weight <= 0:
                logger.info(
                    f"Excluding dataset {ds_name}: weight={weight} (use SFT preset to include)"
                )
                continue

            proportion = proportion_overrides.get(
                ds_name, ds_info.get("proportion", 1.0)
            )
            datasets.append(
                {
                    "name": ds_name,
                    "group": group_name,
                    "path": ds_info["path"],
                    "weight": weight,
                    "samples": ds_info.get("samples", 0),
                    "shards": ds_info.get("shards", 0),
                    "proportion": proportion,
                    "description": ds_info.get("description", ""),
                }
            )

    return datasets


def normalize_weights(datasets: list[dict]) -> list[dict]:
    """Normalize weights to sum to 1.0."""
    total_weight = sum(d["weight"] for d in datasets)
    if total_weight > 0:
        for d in datasets:
            d["normalized_weight"] = d["weight"] / total_weight
    else:
        # Equal weights if all zero
        equal = 1.0 / len(datasets) if datasets else 0
        for d in datasets:
            d["normalized_weight"] = equal
    return datasets


class MultiWebDataset:
    """
    Multi-source WebDataset with weighted sampling.

    Combines multiple WebDataset sources into a single iterable,
    with configurable sampling weights per dataset.
    """

    # Per-modality WebDataset tuple specs. Each entry is (tuple_spec_args,
    # process_fn_attr). Extensions match what tools/shard_modality.py writes
    # (text/.ts.npy/.graph.pt) plus the legacy image set already used by
    # MultiWebDatasetWrapper. Adding a modality means adding one entry +
    # implementing the matching _process_<modality>_sample method below.
    _MODALITY_PIPELINE: dict[str, tuple[tuple[str, ...], str]] = {
        "image": (
            ("jpg;png;jpeg;webp;gif", "txt", "json"),
            "_process_sample",
        ),
        "time_series": (
            ("ts.npy", "text", "meta.json"),
            "_process_time_series_sample",
        ),
        "graph": (
            ("graph.pt", "text", "meta.json"),
            "_process_graph_sample",
        ),
        # Composite CALVIN sample. All keys come from a single shard
        # (one per (obs_t, action_{t+1}) pair); the loader must keep them
        # joined by `__key__`. Sharder: tools/shard_calvin_vla.py.
        "vla": (
            (
                "head.jpg",
                "wrist.jpg",
                "pose.npy",
                "action.npy",
                "instruction.txt",
                "meta.json",
            ),
            "_process_vla_sample",
        ),
    }

    def __init__(
        self,
        config: dict | None = None,
        groups: str | list[str] = "all",
        daos_mount: str | None = None,
        weight_overrides: dict[str, float] | None = None,
        proportion_overrides: dict[str, float] | None = None,
        world_size: int = 1,
        rank: int = 0,
        shuffle_shards: bool = True,
        shuffle_buffer: int = 1000,
        seed: int = 42,
        modality: str = "image",
        local_shards_dir: str | None = None,
        partition_by: str = "global",
        resampled: bool = True,
    ):
        """
        Initialize multi-dataset loader.

        Args:
            config: DAOS dataset configuration dict. Required for the
                manifest+shards/ layout (default). Ignored only by the legacy
                local_shards_dir path, which reads a single flat shard list.
            groups: Preset name, group name, or list of group names
            daos_mount: Override for DAOS mount path
            weight_overrides: Optional per-dataset weight overrides
            proportion_overrides: Optional per-dataset proportion overrides (0.0-1.0)
                Controls what fraction of each dataset's shards to use.
                Useful for faster startup with large datasets.
            world_size: Total number of workers (for shard splitting)
            rank: Current worker rank
            shuffle_shards: Whether to shuffle shards
            shuffle_buffer: Buffer size for sample shuffling
            seed: Random seed for reproducibility
            local_shards_dir: When set, read shards from a single local dir
                (scripts/stage_shards.py output). Reads <dir>/local_manifest.json
                (or globs *.tar if absent), bypasses DAOS shard discovery /
                broadcast, and partitions across LOCAL ranks by default (each
                node's tmpfs has its own subset).
            partition_by: "global" (default) divides shards across all ranks
                in world_size and uses rank-0 shard discovery/broadcast.
                "local" makes each rank discover its node-local dataset tree
                and divide each node-private shard subset across LOCAL ranks.
                local_shards_dir forces partition_by="local" automatically.
            resampled: If True, sample shards with replacement and remove epoch
                boundaries. If False, iterate each rank's shard partition once
                per iterator, letting the training loop restart for a new epoch.
        """
        if not HAS_WEBDATASET:
            raise ImportError(
                "webdataset is required. Install with: pip install webdataset"
            )

        if modality not in self._MODALITY_PIPELINE:
            raise ValueError(
                f"MultiWebDataset: unknown modality {modality!r}. "
                f"Known: {list(self._MODALITY_PIPELINE.keys())}"
            )
        if partition_by not in ("global", "local"):
            raise ValueError(
                f"MultiWebDataset: partition_by must be 'global' or 'local', "
                f"got {partition_by!r}"
            )
        self.modality = modality
        self.config = config
        self.world_size = world_size
        self.rank = rank
        self.shuffle_shards = shuffle_shards
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.local_shards_dir = local_shards_dir
        self.resampled = resampled
        self._start_sample_offset = 0
        # local-shards mode is a per-node tmpfs layout; partition_by="local"
        # is the only sensible default. Operators can still pass "global"
        # explicitly if they have a single shared local_shards_dir on a
        # network mount (untested but not blocked).
        self.partition_by = "local" if local_shards_dir else partition_by

        if local_shards_dir:
            # Local-shards mode skips DAOS config entirely. Build a single
            # synthetic dataset entry so downstream code (_build_dataset,
            # get_stats, __len__) sees the same data shape it does for the
            # normal path. shards/samples are estimated from local_manifest
            # in _discover_local_shards; group is "local" so anything
            # filtering by group ignores it cleanly.
            self.daos_mount = ""
            self.datasets_info = [{
                "name": f"local_shards:{os.path.basename(local_shards_dir.rstrip('/'))}",
                "path": local_shards_dir,
                "group": "local",
                "weight": 1.0,
                "normalized_weight": 1.0,
                "samples": 0,  # filled in by local_manifest if present
                "shards": 0,   # filled in by _discover_local_shards
            }]
            logger.info(
                f"MultiWebDataset: local-shards mode from {local_shards_dir} "
                f"(partition_by=local)"
            )
        else:
            if config is None:
                raise ValueError(
                    "MultiWebDataset: config is required when local_shards_dir "
                    "is not set"
                )
            # Get DAOS mount path
            if daos_mount:
                self.daos_mount = daos_mount
            else:
                # Try environment variable, then config default
                self.daos_mount = os.environ.get(
                    "DAOS_MOUNT",
                    config.get("daos", {}).get(
                        "mount_base", "/tmp/AuroraGPT/prism_training_data"
                    ),
                )

            # Expand user in path
            user = os.environ.get("USER", "unknown")
            self.daos_mount = self.daos_mount.replace("${USER}", user)

            # Get datasets for specified groups (with proportion support)
            self.datasets_info = get_datasets_for_groups(
                config, groups, weight_overrides, proportion_overrides
            )
            self.datasets_info = normalize_weights(self.datasets_info)

            logger.info(f"MultiWebDataset: Loading {len(self.datasets_info)} datasets")
            for ds in self.datasets_info:
                proportion_str = (
                    f", proportion={ds['proportion']:.2f}"
                    if ds.get("proportion", 1.0) < 1.0
                    else ""
                )
                logger.info(
                    f"  - {ds['name']}: {ds['samples']} samples, weight={ds['weight']:.2f} "
                    f"(norm={ds['normalized_weight']:.4f}){proportion_str}"
                )

        # Build the combined dataset
        self._dataset = self._build_dataset()

    def _get_shards_for_dataset(
        self, ds_info: dict, validation: bool = False
    ) -> list[str]:
        """
        Get list of shard paths for a dataset.

        Uses manifest.json if available (fast path - single file read),
        falls back to glob.glob() if not (slow path - many stat calls).

        Applies proportion limiting if ds_info contains a 'proportion' field < 1.0.

        Args:
            ds_info: Dataset info dict with 'path', 'name', etc.
            validation: If True, return validation shards instead of training shards.

        NOTE: This method performs file system operations. In distributed settings,
        it should only be called by rank 0, with results broadcast to other ranks
        via _discover_all_shards_distributed().
        """
        ds_base_path = os.path.join(self.daos_mount, ds_info["path"])
        manifest_path = os.path.join(ds_base_path, "manifest.json")

        # Validation shards are in val_shards/, training shards in shards/
        if validation:
            shards_dir = os.path.join(ds_base_path, "val_shards")
            manifest_key = "val_shards"
            count_key = "num_val_shards"
        else:
            shards_dir = os.path.join(ds_base_path, "shards")
            manifest_key = "shards"
            count_key = "num_shards"

        shards = []
        total_shards_available = 0

        # Fast path: Read shard list from manifest.json
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path) as f:
                    manifest = json.load(f)

                # Extract shard names from manifest (supports multiple formats)
                #
                # For training shards (manifest_key="shards"):
                #   List format: {"shards": [{"name": "shard-00000.tar", ...}, ...]}
                #   Count format: {"num_shards": N}, {"total_shards": N}, {"num_train_shards": N},
                #                 {"train_shards": N}
                #
                # For validation shards (manifest_key="val_shards"):
                #   List format: {"val_shards": [{"name": "shard-00000.tar", ...}, ...]}
                #   Count format: {"val_shards": N}, {"num_val_shards": N}
                #
                # Multiple naming conventions exist because manifests were
                # generated by different scripts over time.
                shard_names = None
                num_shards = None

                if manifest_key in manifest and isinstance(
                    manifest[manifest_key], list
                ):
                    shard_names = [s["name"] for s in manifest[manifest_key]]
                    # Old-format manifests include ALL shards (train+val) in the
                    # "shards" list. Use the dataset config's shard count to cap
                    # the list to training-only when num_train_shards is absent.
                    if not validation and "num_train_shards" not in manifest:
                        config_shards = ds_info.get("shards")
                        if config_shards and len(shard_names) > config_shards:
                            logger.warning(
                                f"{ds_info['name']}: Manifest has {len(shard_names)} "
                                f"shards but config says {config_shards} training "
                                f"shards — capping to {config_shards} "
                                f"(manifest missing num_train_shards)"
                            )
                            shard_names = shard_names[:config_shards]
                elif manifest_key in manifest and isinstance(
                    manifest[manifest_key], int
                ):
                    # Handle case where manifest_key (e.g. "val_shards") is an
                    # integer count rather than a list of shard objects
                    num_shards = manifest[manifest_key]
                elif count_key in manifest:
                    num_shards = manifest[count_key]
                else:
                    # Try all known aliases for shard counts
                    alias_keys = []
                    if validation:
                        alias_keys = ["num_val_shards", "val_shards"]
                    else:
                        alias_keys = [
                            "total_shards",
                            "num_train_shards",
                            "train_shards",
                        ]
                    for alias in alias_keys:
                        if alias in manifest and isinstance(manifest[alias], int):
                            num_shards = manifest[alias]
                            logger.info(
                                f"{ds_info['name']}: Using manifest key '{alias}' "
                                f"for {manifest_key} count ({num_shards})"
                            )
                            break

                if shard_names is None and num_shards is None:
                    logger.warning(
                        f"{ds_info['name']}: Manifest exists but has no {manifest_key} info "
                        f"(keys found: {list(manifest.keys())}), falling back to glob"
                    )

                # Generate shard names from count if we got a numeric count
                if num_shards is not None and shard_names is None:
                    if num_shards == 0:
                        logger.info(f"{ds_info['name']}: No {manifest_key} in manifest")
                        return []

                    # For validation shards, naming typically continues from training
                    # e.g., train: pixmo-000000.tar to pixmo-000612.tar
                    #       val:   pixmo-000613.tar (starts at num_train_shards)
                    num_train_shards = manifest.get(
                        "num_train_shards",
                        manifest.get("num_shards", manifest.get("total_shards", 0)),
                    )
                    start_idx = num_train_shards if validation else 0

                    # Check which naming pattern is used by testing first shard
                    test_shard_generic = os.path.join(
                        shards_dir, f"shard-{start_idx:06d}.tar"
                    )
                    if os.path.exists(test_shard_generic):
                        # Generic "shard-XXXXXX.tar" pattern
                        shard_names = [
                            f"shard-{start_idx + i:06d}.tar" for i in range(num_shards)
                        ]
                    else:
                        # Dataset-specific "{name}-XXXXXX.tar" pattern
                        ds_name = manifest.get("dataset", ds_info["name"]).replace(
                            "_", "-"
                        )
                        shard_names = [
                            f"{ds_name}-{start_idx + i:06d}.tar"
                            for i in range(num_shards)
                        ]

                if shard_names:
                    shards = [os.path.join(shards_dir, name) for name in shard_names]
                    total_shards_available = len(shards)
                    shard_type = "validation" if validation else "training"
                    logger.info(
                        f"{ds_info['name']}: Loaded {len(shards)} {shard_type} shards from manifest (fast path)"
                    )

            except (OSError, json.JSONDecodeError, KeyError) as e:
                logger.warning(
                    f"{ds_info['name']}: Failed to read manifest ({e}), falling back to glob"
                )
                shards = []

        # Slow path: Fall back to directory listing if manifest didn't work
        # NOTE: Using os.listdir instead of glob.glob for DAOS compatibility
        # (glob.glob can hang on dfuse mounts under certain conditions)
        if not shards:
            if not os.path.isdir(shards_dir):
                if validation:
                    # Validation shards are optional - don't warn loudly
                    logger.debug(f"Validation shard directory not found: {shards_dir}")
                else:
                    logger.warning(f"Shard directory not found: {shards_dir}")
                return []

            # For validation shards, if no manifest info, try quick directory listing
            # but don't block forever on slow filesystems like DAOS/dfuse
            if validation:
                logger.info(
                    f"{ds_info['name']}: No val_shards in manifest, using directory listing"
                )
                try:
                    # Use os.listdir which is faster than glob on DAOS
                    files = os.listdir(shards_dir)
                    tar_files = sorted([f for f in files if f.endswith(".tar")])
                    shards = [os.path.join(shards_dir, f) for f in tar_files]
                    total_shards_available = len(shards)
                    if shards:
                        logger.info(
                            f"{ds_info['name']}: Found {len(shards)} validation shards via directory listing"
                        )
                    else:
                        logger.debug(
                            f"{ds_info['name']}: No validation shards found (optional)"
                        )
                        return []
                except OSError as e:
                    logger.warning(
                        f"{ds_info['name']}: Failed to list validation shards ({e}), skipping"
                    )
                    return []
            else:
                # Training shards - use original glob behavior
                logger.warning(
                    f"{ds_info['name']}: No manifest found at {manifest_path}, using glob (slow)"
                )
                shards = sorted(glob.glob(os.path.join(shards_dir, "*.tar")))
                total_shards_available = len(shards)

            if not shards:
                if not validation:
                    logger.warning(f"No shards found in: {shards_dir}")
                return []

        # Apply proportion limiting (only for training, not validation)
        if not validation:
            proportion = ds_info.get("proportion", 1.0)
            if proportion < 1.0 and proportion > 0:
                num_shards_to_use = max(1, int(len(shards) * proportion))
                shards = shards[:num_shards_to_use]
                logger.info(
                    f"{ds_info['name']}: Using {num_shards_to_use}/{total_shards_available} shards "
                    f"(proportion={proportion:.2f})"
                )

        return shards

    def _partition_slice(self) -> tuple[int, int]:
        """Resolve (slice_rank, slice_size) for partitioning a shard list.

        global mode: (self.rank, self.world_size) — every rank takes its
        global index.

        local mode: (LOCAL_RANK, LOCAL_WORLD_SIZE) — resolved from env. Each
        node's local ranks divide that node's shard subset. Fallback chain
        matches what every Aurora launcher exports
        (tools/launch_aurora_web.py:690, launch_aurora_daos.py:1109, etc).
        """
        if self.partition_by == "global":
            return self.rank, self.world_size
        local_rank = int(os.environ.get(
            "LOCAL_RANK",
            os.environ.get("PMI_LOCAL_RANK",
                os.environ.get("PALS_LOCAL_RANKID", "0"))
        ))
        local_world = int(os.environ.get(
            "LOCAL_WORLD_SIZE",
            os.environ.get("PMI_LOCAL_SIZE",
                os.environ.get("PALS_LOCAL_SIZE", "1"))
        ))
        return local_rank, local_world

    def _discover_local_shards(self) -> list[str]:
        """Read shard list from local_shards_dir.

        scripts/stage_shards.py writes <local_dir>/local_manifest.json with
        a "shards" array of basenames and the .tar files directly in
        <local_dir> (no shards/ subdir). Each node's tmpfs is independent,
        so every rank reads its own node's manifest — no rank-0 broadcast
        is needed (and would be wrong, since rank 0's shards live on a
        different node's tmpfs).

        Falls back to globbing *.tar if the manifest is missing, matching
        the historical LocalShardDataset behavior.
        """
        local_dir = self.local_shards_dir
        manifest_path = os.path.join(local_dir, "local_manifest.json")
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path) as f:
                    manifest = json.load(f)
                shard_names = manifest.get("shards", [])
                if shard_names:
                    shards = [os.path.join(local_dir, n) for n in shard_names]
                    samples_est = manifest.get("total_samples_estimate")
                    if samples_est is not None:
                        self.datasets_info[0]["samples"] = int(samples_est)
                    self.datasets_info[0]["shards"] = len(shards)
                    logger.info(
                        f"Rank {self.rank}: local_manifest.json declares "
                        f"{len(shards)} shards from {local_dir}"
                    )
                    return shards
                logger.warning(
                    f"Rank {self.rank}: local_manifest.json has empty 'shards', "
                    f"falling back to glob"
                )
            except (OSError, json.JSONDecodeError, KeyError) as e:
                logger.warning(
                    f"Rank {self.rank}: failed to read {manifest_path} ({e}), "
                    f"falling back to glob"
                )
        # Fallback: glob the directory. Matches the original LocalShardDataset
        # behavior pre-stage_shards.py.
        shards = sorted(glob.glob(os.path.join(local_dir, "*.tar")))
        self.datasets_info[0]["shards"] = len(shards)
        logger.info(
            f"Rank {self.rank}: globbed {len(shards)} *.tar shards from {local_dir}"
        )
        return shards

    def _discover_all_shards_distributed(self) -> dict[str, list[str]]:
        """
        Discover shards for all datasets using rank-0 coordination.

        This is critical for scaling: instead of ALL ranks hitting the filesystem
        (which causes O(N) contention on DAOS/dfuse), only rank 0 performs discovery
        and broadcasts results to all other ranks.

        IMPORTANT: When world_size=1 (single-process mode, e.g., for validation),
        this method skips distributed primitives and does direct filesystem access.
        This avoids deadlocks when called from only one rank.

        local partition reads per-node and bypasses the rank-0 broadcast.
        Broadcasting would be wrong when each node's tmpfs holds a different
        shard subset.

        Returns:
            Dict mapping dataset name to list of shard paths
        """
        if self.local_shards_dir:
            ds_name = self.datasets_info[0]["name"]
            return {ds_name: self._discover_local_shards()}

        all_shards: dict[str, list[str]] = {}

        if self.partition_by == "local":
            logger.info(
                f"Rank {self.rank}: Discovering node-local shards for "
                f"{len(self.datasets_info)} datasets..."
            )
            for ds_info in self.datasets_info:
                all_shards[ds_info["name"]] = self._get_shards_for_dataset(ds_info)
            return all_shards

        # Check if we're in a distributed environment AND using multiple workers
        # world_size=1 means single-process mode (e.g., validation loader) - skip distributed ops
        is_distributed = (
            HAS_TORCH_DISTRIBUTED
            and dist is not None
            and dist.is_initialized()
            and self.world_size
            > 1  # CRITICAL: Only use collective ops if world_size > 1
        )

        if is_distributed:
            # Only rank 0 performs filesystem operations
            if self.rank == 0:
                logger.info("Rank 0: Discovering shards for all datasets...")
                for ds_info in self.datasets_info:
                    shards = self._get_shards_for_dataset(ds_info)
                    all_shards[ds_info["name"]] = shards
                logger.info(f"Rank 0: Discovered shards for {len(all_shards)} datasets")

            # Synchronize: ensure rank 0 has finished discovery
            dist.barrier()

            # Broadcast shard lists from rank 0 to all ranks
            # We serialize to JSON for easy tensor-free broadcast
            if self.rank == 0:
                shards_json = json.dumps(all_shards)
                shards_bytes = shards_json.encode("utf-8")
                size_tensor = torch.tensor([len(shards_bytes)], dtype=torch.long)
            else:
                size_tensor = torch.tensor([0], dtype=torch.long)

            # Move to appropriate device if available
            if torch.cuda.is_available():
                size_tensor = size_tensor.cuda()
            elif hasattr(torch, "xpu") and torch.xpu.is_available():
                size_tensor = size_tensor.to("xpu")

            # Broadcast size first
            dist.broadcast(size_tensor, src=0)
            data_size = size_tensor.item()

            # Create buffer and broadcast data
            if self.rank == 0:
                data_tensor = torch.frombuffer(
                    bytearray(shards_bytes), dtype=torch.uint8
                ).clone()
            else:
                data_tensor = torch.zeros(data_size, dtype=torch.uint8)

            # Move to appropriate device
            if torch.cuda.is_available():
                data_tensor = data_tensor.cuda()
            elif hasattr(torch, "xpu") and torch.xpu.is_available():
                data_tensor = data_tensor.to("xpu")

            dist.broadcast(data_tensor, src=0)

            # Deserialize on non-rank-0
            if self.rank != 0:
                shards_bytes = bytes(data_tensor.cpu().numpy().tolist())
                shards_json = shards_bytes.decode("utf-8")
                all_shards = json.loads(shards_json)
                logger.info(
                    f"Rank {self.rank}: Received shards for {len(all_shards)} datasets from rank 0"
                )

            # Final sync to ensure all ranks have the data
            dist.barrier()
        else:
            # Non-distributed: each "rank" discovers its own (single process)
            for ds_info in self.datasets_info:
                shards = self._get_shards_for_dataset(ds_info)
                all_shards[ds_info["name"]] = shards

        return all_shards

    def _build_dataset(self):
        """
        Build combined WebDataset with weighted sampling.

        Uses wds.RandomMix for weighted combination of multiple sources.

        OPTIMIZATION: Uses rank-0 coordinated shard discovery to avoid
        O(N) filesystem contention on DAOS/dfuse when scaling to many nodes.
        """
        start_sample_offset = int(getattr(self, "_start_sample_offset", 0) or 0)
        if start_sample_offset:
            logger.info(
                f"Rank {self.rank}: Fast-forwarding WebDataset stream by "
                f"{start_sample_offset} raw sample(s) before decode"
            )

        # CRITICAL: Use distributed shard discovery to avoid scaling contention
        # Only rank 0 hits the filesystem, then broadcasts to all ranks
        all_shards = self._discover_all_shards_distributed()

        # Collect all dataset sources with their weights
        sources = []
        weights = []

        for ds_info in self.datasets_info:
            shards = all_shards.get(ds_info["name"], [])
            if not shards:
                logger.warning(f"No shards found for {ds_info['name']}")
                continue

            # Create WebDataset for this source.
            # Manually split shards by rank BEFORE creating WebDataset —
            # avoids issues with nodesplitter + DataLoader workers.
            #
            # partition_by="global" (default): every rank in world_size
            # takes 1/world_size of the shared shard list. Right when all
            # ranks see the same shards (DAOS, network FS).
            #
            # partition_by="local": each node's tmpfs holds a node-private
            # subset; each node's local ranks split that subset 1/LOCAL_WORLD_SIZE.
            # Right for scripts/stage_shards.py output, where the global
            # shard list at rank 0 has no relation to what's actually on
            # rank N's tmpfs.
            slice_rank, slice_size = self._partition_slice()

            # When there are at least as many shards as ranks, split by shard:
            # each rank opens a disjoint subset, which is the cheapest way to
            # keep ranks from reading the same bytes.
            #
            # When shards < ranks, shard-splitting cannot produce a disjoint
            # cover — some ranks get nothing. Handing those ranks shards[:1]
            # (the previous behaviour) gave every starved rank the SAME shard,
            # so the same samples were trained on repeatedly while the rest of
            # the dataset went unseen, silently. Instead, fall back to
            # sample-level striding: every rank opens the full shard list and
            # keeps sample i only when i % slice_size == slice_rank. That is
            # still a disjoint cover, just at sample rather than shard
            # granularity. Same trick `_StridedTSValidation` uses for the
            # 3-shard SciTS validation split.
            #
            # This is what lets a 27-shard dataset train on 12 ranks/node
            # without capping ngpus to the shard count.
            sample_stride = None
            if slice_size > 1 and len(shards) >= slice_size:
                shards_for_rank = shards[slice_rank::slice_size]
                logger.info(
                    f"Rank {self.rank}: Using {len(shards_for_rank)}/{len(shards)} "
                    f"shards for {ds_info['name']} (partition_by={self.partition_by}, "
                    f"slice={slice_rank}/{slice_size})"
                )
            elif slice_size > 1:
                shards_for_rank = shards
                sample_stride = (slice_rank, slice_size)
                logger.info(
                    f"Rank {self.rank}: {len(shards)} shards < {slice_size} ranks for "
                    f"{ds_info['name']}; reading all shards and striding at the "
                    f"sample level (keep i where i %% {slice_size} == {slice_rank}, "
                    f"partition_by={self.partition_by})"
                )
            else:
                shards_for_rank = shards

            if not shards_for_rank:
                continue  # Skip this dataset entirely if no shards available

            shardshuffle = False if self.resampled else (100 if self.shuffle_shards else False)
            if sample_stride is not None:
                # Striding assigns sample i to rank i % slice_size, so every
                # rank must walk the shards in the SAME order — otherwise the
                # per-rank orderings disagree, ranks collide on some samples
                # and miss others. Pin shard order and resampling off for this
                # source; the .shuffle() buffer below still randomizes what
                # each rank sees, and the stride keeps the ranks disjoint.
                shardshuffle = False
                source_resampled = False
            else:
                source_resampled = self.resampled

            base = wds.WebDataset(
                shards_for_rank,
                shardshuffle=shardshuffle,
                empty_check=False,  # Don't error on empty - let training handle it
                nodesplitter=_identity,  # Shards already split; module-level for pickling
                resampled=source_resampled,
                detshuffle=not source_resampled,
                seed=self.seed,
            )

            # Stride BEFORE .shuffle()/.decode() so the index each rank tests
            # is the raw shard-order index (identical on every rank), and so
            # dropped samples are never decoded.
            if sample_stride is not None:
                base = base.select(_SampleStride(*sample_stride))
                if self.resampled:
                    # `resampled=True` normally supplies the endless stream the
                    # training loop expects (it never checks for exhaustion).
                    # We had to turn it off above to keep shard order identical
                    # across ranks, so restore the endlessness with .repeat().
                    # Epoch boundaries reappear, but every rank hits them on the
                    # same step, so no collective goes unmatched.
                    base = base.repeat()

            base = base.shuffle(self.shuffle_buffer, seed=self.seed)

            if start_sample_offset:
                # Keep sources raw so the global slice below can skip samples
                # before PIL decode, tensor conversion, tokenization, and collate.
                sources.append(base)
            else:
                tuple_spec, process_attr = self._MODALITY_PIPELINE[self.modality]
                process_fn = getattr(self, process_attr)
                # `.decode("pil")` only makes sense for image shards — running it
                # on numpy/torch payloads from shard_modality.py would call the PIL
                # decoder on raw bytes and crash. Non-image modalities skip decode
                # and rely on the per-modality _process fn to deserialize.
                if self.modality == "image":
                    base = base.decode("pil")

                ds = base.to_tuple(*tuple_spec, handler=wds.warn_and_continue).map(process_fn)
                sources.append(ds)
            weights.append(ds_info["normalized_weight"])

        if not sources:
            if not self.resampled:
                # Finite mode skips (rather than duplicates) sources a rank has
                # no shards for. When world_size exceeds every source's shard
                # count, high-index ranks end up with zero sources and crash
                # here — which in DDP/FSDP hangs the whole job. Surface the
                # actual cause instead of a generic message.
                raise ValueError(
                    f"Rank {self.rank}: no shards assigned for any source in "
                    f"finite (resampled=False) mode. This happens when world_size "
                    f"exceeds the per-source shard count. Reduce nodes/ranks, "
                    f"reshard the dataset so num_shards >= world_size, or use "
                    f"resampled mode."
                )
            raise ValueError("No valid dataset sources found!")

        logger.info(f"Built {len(sources)} WebDataset sources (resampled={self.resampled})")

        # Combine with weighted random mixing
        # longest=True: if a source somehow exhausts, drop it and continue
        # with remaining sources (safety net - shouldn't happen with resampled=True)
        if len(sources) == 1:
            mixed = sources[0]
        elif not self.resampled:
            mixed = _DeterministicRandomMix(
                sources, weights, longest=True, seed=self.seed
            )
        else:
            mixed = wds.RandomMix(sources, weights, longest=True)

        if not start_sample_offset:
            return mixed

        tuple_spec, process_attr = self._MODALITY_PIPELINE[self.modality]
        process_fn = getattr(self, process_attr)
        stages = [
            mixed,
            wds.filters.slice(start_sample_offset, None),
        ]
        if self.modality == "image":
            stages.append(wds.filters.decode("pil"))
        stages.extend(
            [
                wds.filters.to_tuple(*tuple_spec, handler=wds.warn_and_continue),
                wds.filters.map(process_fn),
            ]
        )
        return wds.DataPipeline(*stages)

    def set_start_sample_offset(self, offset: int) -> int:
        """Rebuild the stream so iteration starts after `offset` raw samples."""
        offset = max(0, int(offset))
        self._start_sample_offset = offset
        self._dataset = self._build_dataset()
        return offset

    def _process_sample(self, sample: tuple) -> dict[str, Any]:
        """Process a raw WebDataset sample into standard format.

        Handles three data types:
        1. Conversation format (CoSyn-point): metadata has 'conversations' list
        2. Pointing format (pixmo-points): metadata has 'points' + 'label'
        3. Caption format (default): text is the caption
        """
        image, text, metadata = sample

        # Handle text decoding
        if isinstance(text, bytes):
            text = text.decode("utf-8")

        # Handle metadata
        if isinstance(metadata, bytes):
            try:
                metadata = json.loads(metadata.decode("utf-8"))
            except Exception:
                metadata = {}
        elif not isinstance(metadata, dict):
            metadata = {}

        # Check for pointing/conversation data formats
        caption = None

        # Format 1: Conversation format (CoSyn-point style)
        # metadata has 'conversations': [{'role': 'user', 'content': ...}, {'role': 'assistant', 'content': '<points>...'}]
        if "conversations" in metadata and isinstance(metadata["conversations"], list):
            conversations = metadata["conversations"]
            if len(conversations) >= 2:
                # Format as "user: ... assistant: ..." for training
                parts = []
                for turn in conversations:
                    role = turn.get("role", "user")
                    content = turn.get("content", "")
                    parts.append(f"{role}: {content}")
                caption = "\n".join(parts)
                metadata["_data_type"] = "pointing_conversation"

        # Format 2: Points + Label format (pixmo-points style)
        # metadata has 'points': [{'x': ..., 'y': ...}, ...] and 'label': '...'
        # OR 'points': {'x': [...], 'y': [...]} (pixmo-count parallel array format)
        elif "points" in metadata and "label" in metadata:
            points = metadata["points"]
            label = metadata["label"]

            # Convert parallel array format to list of dicts
            # pixmo_count uses: {"x": [x1, x2, ...], "y": [y1, y2, ...]}
            if isinstance(points, dict) and "x" in points and "y" in points:
                x_coords = points["x"]
                y_coords = points["y"]
                if isinstance(x_coords, list) and isinstance(y_coords, list):
                    points = [{"x": x, "y": y} for x, y in zip(x_coords, y_coords, strict=False)]

            if isinstance(points, list) and len(points) > 0:
                # Construct Molmo2-style conversation
                # Question: "Point to [label]"
                # Answer: <points coords="1 1 X Y;...">label</points>
                question = f"Point to {label}"

                # Format points as Molmo2 coords (normalized to 0-1000)
                coords_parts = []
                for i, pt in enumerate(points):
                    if isinstance(pt, dict):
                        x = pt.get("x", 0)
                        y = pt.get("y", 0)
                    elif isinstance(pt, list | tuple) and len(pt) >= 2:
                        x, y = pt[0], pt[1]
                    else:
                        continue

                    # Normalize to 0-1000 range if not already
                    # pixmo-points uses 0-100 range, so multiply by 10
                    if x <= 100 and y <= 100:
                        x = int(x * 10)
                        y = int(y * 10)
                    else:
                        x = int(x)
                        y = int(y)

                    # Format: "ImageIndex ObjectID X Y"
                    coords_parts.append(f"1 {i + 1} {x} {y}")

                coords_str = ";".join(coords_parts)
                answer = f'<points coords="{coords_str}">{label}</points>'

                caption = f"user: {question}\nassistant: {answer}"
                metadata["_data_type"] = "pointing_pixmo"

        # Format 3: Default caption format
        if caption is None:
            caption = text
            metadata["_data_type"] = "caption"

        # CRITICAL: Validate text is not empty
        # Empty text causes NaN loss because all labels become -100 (ignored)
        if not caption or not caption.strip():
            # Log detailed info to help identify the source dataset
            source = metadata.get("source", "unknown")
            sample_id = metadata.get(
                "id", metadata.get("image_id", metadata.get("global_idx", "unknown"))
            )
            logger.warning(
                f"[DATA QUALITY] Empty caption found! Source: {source}, ID: {sample_id}, "
                f"Full metadata: {metadata}. Using fallback caption."
            )
            caption = "An image."  # Fallback caption

        return {
            "image": image,
            "caption": caption,
            "metadata": metadata,
        }

    def _process_time_series_sample(self, sample: tuple) -> dict[str, Any]:
        """Decode (ts_bytes, text_bytes, meta_bytes) → dict for the wrapper.

        Yields the same shape the downstream `ModalityAwareWebDatasetWrapper`
        expects: a per-modality key (`time_series`) with the raw numpy/bytes
        payload, plus `text` and `metadata`. The wrapper handles the actual
        tensor conversion + tokenization.
        """
        import io

        import numpy as np

        payload, text, metadata = sample
        if isinstance(payload, (bytes, bytearray)):
            arr = np.load(io.BytesIO(payload), allow_pickle=False)
        elif isinstance(payload, np.ndarray):
            arr = payload
        else:
            arr = payload  # tensor / other — wrapper decoder will coerce
        if isinstance(text, bytes):
            text = text.decode("utf-8", errors="replace")
        if isinstance(metadata, bytes):
            try:
                metadata = json.loads(metadata.decode("utf-8"))
            except Exception:
                metadata = {}
        elif not isinstance(metadata, dict):
            metadata = {}
        metadata.setdefault("_data_type", "time_series")
        return {"time_series": arr, "text": text, "metadata": metadata}

    def _process_graph_sample(self, sample: tuple) -> dict[str, Any]:
        """Decode (graph_bytes, text_bytes, meta_bytes) → dict for the wrapper."""
        import io

        import torch

        payload, text, metadata = sample
        if isinstance(payload, (bytes, bytearray)):
            graph = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
        else:
            graph = payload
        if isinstance(text, bytes):
            text = text.decode("utf-8", errors="replace")
        if isinstance(metadata, bytes):
            try:
                metadata = json.loads(metadata.decode("utf-8"))
            except Exception:
                metadata = {}
        elif not isinstance(metadata, dict):
            metadata = {}
        metadata.setdefault("_data_type", "graph")
        return {"graph": graph, "text": text, "metadata": metadata}

    def _process_vla_sample(self, sample: tuple) -> dict[str, Any]:
        """Decode CALVIN VLA composite (head, wrist, pose, action, text, meta).

        Yields a dict keyed by the per-modality names the
        `ModalityAwareWebDatasetWrapper`'s VLA branch consumes:
            image_head, image_wrist, pose, action, text, metadata
        The wrapper handles tensor conversion + tokenization; we keep
        decoding minimal so JSON/bytes hand-off stays consistent with the
        other modalities.
        """
        head_bytes, wrist_bytes, pose_bytes, action_bytes, text_bytes, meta_bytes = sample

        if isinstance(text_bytes, bytes):
            text = text_bytes.decode("utf-8", errors="replace")
        else:
            text = text_bytes

        if isinstance(meta_bytes, (bytes, bytearray)):
            try:
                metadata = json.loads(meta_bytes.decode("utf-8"))
            except Exception:
                metadata = {}
        elif isinstance(meta_bytes, dict):
            metadata = meta_bytes
        else:
            metadata = {}
        metadata.setdefault("_data_type", "vla")

        return {
            "image_head": head_bytes,
            "image_wrist": wrist_bytes,
            "pose": pose_bytes,
            "action": action_bytes,
            "text": text,
            "metadata": metadata,
        }

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over samples from all datasets."""
        return iter(self._dataset)

    def __len__(self) -> int:
        """Approximate total samples (sum across all datasets)."""
        return sum(ds["samples"] for ds in self.datasets_info)

    def get_stats(self) -> dict:
        """Get dataset statistics."""
        return {
            "num_datasets": len(self.datasets_info),
            "total_samples": sum(ds["samples"] for ds in self.datasets_info),
            "total_shards": sum(ds["shards"] for ds in self.datasets_info),
            "datasets": [
                {
                    "name": ds["name"],
                    "group": ds["group"],
                    "samples": ds["samples"],
                    "weight": ds["weight"],
                    "normalized_weight": ds["normalized_weight"],
                }
                for ds in self.datasets_info
            ],
        }

    def get_validation_shards(self) -> dict[str, list[str]]:
        """
        Get validation shard paths for all loaded datasets.

        Reads val_shards from manifest.json for each dataset.
        Use this to create a validation DataLoader separately.

        Returns:
            Dict mapping dataset name to list of validation shard paths.
            Datasets without validation shards will have empty lists.
        """
        val_shards: dict[str, list[str]] = {}

        for ds_info in self.datasets_info:
            shards = self._get_shards_for_dataset(ds_info, validation=True)
            val_shards[ds_info["name"]] = shards
            if shards:
                logger.info(f"{ds_info['name']}: Found {len(shards)} validation shards")

        return val_shards

    def get_all_validation_shard_paths(self) -> list[str]:
        """
        Get flat list of all validation shard paths across all datasets.

        Convenience method for creating a simple validation loader.

        Returns:
            List of all validation shard paths (combined from all datasets).
        """
        all_paths = []
        for ds_info in self.datasets_info:
            shards = self._get_shards_for_dataset(ds_info, validation=True)
            all_paths.extend(shards)
        return all_paths


try:
    from torch.utils.data import IterableDataset as _IterableDataset

    _HAS_ITERABLE_DATASET = True
except ImportError:
    _IterableDataset = object
    _HAS_ITERABLE_DATASET = False


class MultiWebDatasetWrapper(_IterableDataset):
    """
    Wrapper around MultiWebDataset that produces samples in the format expected
    by the training pipeline (matching StreamingMultimodalDataset output).

    This wrapper:
    1. Transforms PIL images to tensors (224x224, normalized)
    2. Tokenizes captions
    3. Returns samples with keys: {'image': tensor, 'text': tokens, '_metadata': str}

    Note: This is an IterableDataset - use DataLoader with num_workers and no shuffle.
    """

    def __init__(
        self,
        tokenizer,
        config: dict | None = None,
        config_path: str = "src/conf/data/daos_datasets.yaml",
        groups: str | list[str] = "all",
        daos_mount: str | None = None,
        weight_overrides: dict[str, float] | None = None,
        proportion_overrides: dict[str, float] | None = None,
        world_size: int = 1,
        rank: int = 0,
        max_length: int = 2048,
        batch_size: int = 1,  # Not used directly, kept for API compatibility
        max_steps: int = 1000000,  # Not used directly, kept for API compatibility
        modalities=("image",),
        model_config=None,
        local_shards_dir: str | None = None,
        partition_by: str = "global",
        resampled: bool = True,
        shuffle_buffer: int = 32768,
    ):
        """
        Initialize the wrapper.

        Args:
            tokenizer: HuggingFace tokenizer for text encoding
            config: Pre-loaded config dict (optional, overrides config_path)
            config_path: Path to DAOS dataset configuration
            groups: Preset name, group name, or list of group names
            daos_mount: Override DAOS mount path
            weight_overrides: Per-dataset weight overrides
            proportion_overrides: Per-dataset proportion overrides
            world_size: Total distributed workers
            rank: Current worker rank
            max_length: Maximum token length for tokenization
            batch_size: Ignored (kept for API compatibility with StreamingMultimodalDataset)
            max_steps: Ignored (kept for API compatibility)
            modalities: Modalities exposed via `active_modalities`. Defaults to
                `("image",)` for backward compatibility — the historical
                wrapper only handled image samples. Pass `("text", "image",
                "time_series")` (or any subset) when the underlying shards
                actually mix modalities. Tracker for IsoFLOP cells that
                exercise the `text_<mod>` slate via a single wrapper.
            local_shards_dir: When set, read shards from a single local dir
                produced by scripts/stage_shards.py (LOCAL_SHARDS_DIR env);
                forwarded to MultiWebDataset which switches to per-node
                tmpfs mode with local-rank shard partitioning. Disables
                DAOS config loading (config/config_path/daos_mount/groups
                become ignored).
            partition_by: Forwarded to MultiWebDataset. Use "local" when
                daos_mount points to a node-local mirror that differs by node.
            resampled: Whether the underlying WebDataset samples shards with
                replacement. Set false for finite no-replacement epochs.
            shuffle_buffer: Sample-level shuffle buffer passed to WebDataset.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self._modalities = set(modalities)

        # Load config only when not in local-shards mode (DAOS config is
        # unused there). Avoids surprising config-file requirement for the
        # tmpfs-staging path used by Stage A.
        if config is None and not local_shards_dir:
            config = load_daos_config(config_path)

        # Create underlying MultiWebDataset
        self.multi_ds = MultiWebDataset(
            config=config,
            groups=groups,
            daos_mount=daos_mount,
            weight_overrides=weight_overrides,
            proportion_overrides=proportion_overrides,
            world_size=world_size,
            rank=rank,
            local_shards_dir=local_shards_dir,
            partition_by=partition_by,
            resampled=resampled,
            shuffle_buffer=shuffle_buffer,
        )

        self.image_transform = build_image_transform(model_config)

        # Store stats for external access
        self._stats = self.multi_ds.get_stats()
        logger.info(
            f"MultiWebDatasetWrapper: Loaded {self._stats['num_datasets']} datasets, "
            f"{self._stats['total_samples']} total samples"
        )

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over samples, yielding format compatible with training pipeline."""
        import torch
        from PIL import Image

        for sample in self.multi_ds:
            try:
                # 1. Transform image
                pil_image = sample["image"]
                if self.image_transform and isinstance(pil_image, Image.Image):
                    # Ensure RGB
                    if pil_image.mode != "RGB":
                        pil_image = pil_image.convert("RGB")
                    image_tensor = self.image_transform(pil_image)
                else:
                    # Fallback: create dummy tensor
                    image_tensor = torch.zeros(3, 224, 224)

                # 2. Tokenize caption
                caption = sample["caption"]
                tokens = self.tokenizer(
                    caption,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=self.max_length,
                ).input_ids.squeeze(0)

                # 3. Build metadata string
                metadata = sample.get("metadata", {})
                data_type = metadata.get("_data_type", "unknown")
                sample_id = metadata.get(
                    "id",
                    metadata.get("image_id", metadata.get("global_idx", "unknown")),
                )
                metadata_str = f"[{data_type}] {sample_id}"

                yield {
                    "image": image_tensor,
                    "text": tokens,
                    "_metadata": metadata_str,
                }

            except Exception as e:
                logger.warning(f"Error processing sample: {e}")
                continue

    def __len__(self) -> int:
        """Approximate total samples."""
        return len(self.multi_ds)

    def get_stats(self) -> dict:
        """Get dataset statistics."""
        return self._stats

    def fast_forward_batches(self, num_batches: int, batch_size: int) -> int:
        """Skip completed local batches before expensive decode/tokenization."""
        num_samples = max(0, int(num_batches)) * max(1, int(batch_size))
        return self.multi_ds.set_start_sample_offset(num_samples)

    @property
    def active_modalities(self):
        """Return active modalities for compatibility with StreamingMultimodalDataset.

        Mirrors `ModalityAwareWebDatasetWrapper.active_modalities` (line ~1467).
        Defaults to `{"image"}` when nothing was passed to `__init__` so the
        production VLM path stays bit-identical; callers that mix modalities
        via this wrapper should pass `modalities=` explicitly.
        """
        return set(self._modalities)


class ModalityAwareWebDatasetWrapper(_IterableDataset):
    """Phase 3.1: non-image WebDataset wrapper.

    Composes a `MultiWebDataset` (so DAOS shard discovery, weighted sampling,
    and validation-shard helpers come for free) but replaces the hardcoded
    image PIL decode with a per-modality dispatch table. The existing
    `MultiWebDatasetWrapper` is left bit-identical for image-only runs so
    production VLM training has zero regression risk.

    Each shard's manifest declares a `modality` key (e.g. "time_series");
    samples are decoded by the matching entry in `_modality_decoders`.
    Yielded dicts always include `"text"` (tokenized) and `"<modality>"`
    (decoded tensor / dict-of-tensors).

    Decoders currently supported:
        image       -> PIL.Image → 224×224 tensor (legacy parity)
        time_series -> numpy float32 1-D tensor
        graph       -> torch.load of a dict-of-tensors blob (x, edge_index)
    """

    # Modalities this wrapper knows how to decode. Anything outside this set
    # would otherwise raise inside `_decoder_for` and get swallowed by the
    # broad `except` in `__iter__`, silently dropping 100% of samples — so
    # we validate up front instead. "vla" is a *composite* modality: the
    # underlying pipeline pulls all of {head, wrist, pose, action, text}
    # from a single shard and dispatches through `_iter_composite`.
    SUPPORTED_MODALITIES = ("image", "time_series", "graph", "vla")

    # Modalities that bundle multiple per-sample payloads from a single shard.
    # Listed separately so the single-modality invariant lift below can name
    # them explicitly (any new composite needs a matching `_decode_*` and a
    # `_process_*_sample` in MultiWebDataset).
    _COMPOSITE_MODALITIES = frozenset({"vla"})

    def __init__(
        self,
        tokenizer,
        modalities: list[str],
        config: dict | None = None,
        config_path: str = "src/conf/data/daos_datasets.yaml",
        groups: str | list[str] = "all",
        daos_mount: str | None = None,
        weight_overrides: dict[str, float] | None = None,
        proportion_overrides: dict[str, float] | None = None,
        world_size: int = 1,
        rank: int = 0,
        max_length: int = 2048,
        batch_size: int = 1,
        max_steps: int = 1000000,
        model_config=None,
        local_shards_dir: str | None = None,
        partition_by: str = "global",
        resampled: bool = True,
        shuffle_buffer: int = 32768,
    ):
        if not modalities:
            raise ValueError("modalities cannot be empty")
        unsupported = [m for m in modalities if m not in self.SUPPORTED_MODALITIES]
        if unsupported:
            raise ValueError(
                f"ModalityAwareWebDatasetWrapper: unsupported modalities {unsupported}. "
                f"Supported: {list(self.SUPPORTED_MODALITIES)}"
            )
        # A single wrapper instance binds one `MultiWebDataset` and thus one
        # WebDataset pipeline (one `to_tuple` spec). Single-modality runs
        # (image / time_series / graph) keep the historical 1:1 mapping
        # bit-identical. Composite modalities (currently just `vla`) bundle
        # multiple per-sample payloads behind a single pipeline so the loader
        # can keep them joined by `__key__` — required for CALVIN's
        # (obs_t, action_{t+1}) Markov pairs.
        if len(modalities) != 1:
            raise ValueError(
                f"ModalityAwareWebDatasetWrapper supports exactly one (possibly "
                f"composite) modality per instance; got {modalities!r}. "
                f"For VLA pass modalities=['vla']; for parallel modality "
                f"mixes instantiate one wrapper per modality and combine "
                f"downstream."
            )
        self.tokenizer = tokenizer
        self.max_length = max_length
        self._modalities = list(modalities)
        self._primary_modality = self._modalities[0]

        self._image_transform = build_image_transform(model_config)

        if config is None and not local_shards_dir:
            config = load_daos_config(config_path)

        self.multi_ds = MultiWebDataset(
            config=config,
            groups=groups,
            daos_mount=daos_mount,
            weight_overrides=weight_overrides,
            proportion_overrides=proportion_overrides,
            world_size=world_size,
            rank=rank,
            modality=self._primary_modality,
            local_shards_dir=local_shards_dir,
            partition_by=partition_by,
            resampled=resampled,
            shuffle_buffer=shuffle_buffer,
        )

        self._stats = self.multi_ds.get_stats()
        logger.info(
            f"ModalityAwareWebDatasetWrapper(modality={self._primary_modality}): "
            f"Loaded {self._stats['num_datasets']} datasets, "
            f"{self._stats['total_samples']} total samples"
        )

    # ---- modality decoders ------------------------------------------------
    def _decode_image(self, value):
        import torch
        from PIL import Image

        if isinstance(value, Image.Image):
            if value.mode != "RGB":
                value = value.convert("RGB")
            if self._image_transform is not None:
                return self._image_transform(value)
        return torch.zeros(3, 224, 224)

    @staticmethod
    def _decode_time_series(value):
        import io

        import numpy as np
        import torch

        if isinstance(value, torch.Tensor):
            return value.float()
        if isinstance(value, np.ndarray):
            return torch.from_numpy(value.astype("float32"))
        if isinstance(value, (bytes, bytearray)):
            arr = np.load(io.BytesIO(value), allow_pickle=False)
            return torch.from_numpy(arr.astype("float32"))
        raise TypeError(f"time_series decoder: unsupported value type {type(value)}")

    @staticmethod
    def _decode_graph(value):
        import io

        import torch

        if isinstance(value, dict):
            return value
        if isinstance(value, (bytes, bytearray)):
            # torch.save'd dict-of-tensors with keys {x, edge_index, num_nodes}.
            return torch.load(io.BytesIO(value), map_location="cpu", weights_only=True)
        raise TypeError(f"graph decoder: unsupported value type {type(value)}")

    @staticmethod
    def _decode_pose(value):
        """Decode CALVIN pose tensor — float32 rank-1 (typically 15 dims)."""
        import io

        import numpy as np
        import torch

        if isinstance(value, torch.Tensor):
            return value.float()
        if isinstance(value, np.ndarray):
            return torch.from_numpy(value.astype("float32"))
        if isinstance(value, (bytes, bytearray)):
            arr = np.load(io.BytesIO(value), allow_pickle=False)
            return torch.from_numpy(arr.astype("float32"))
        raise TypeError(f"pose decoder: unsupported value type {type(value)}")

    @staticmethod
    def _decode_action(value):
        """Decode CALVIN action tensor — float32 rank-1 (typically 7 dims)."""
        import io

        import numpy as np
        import torch

        if isinstance(value, torch.Tensor):
            return value.float()
        if isinstance(value, np.ndarray):
            return torch.from_numpy(value.astype("float32"))
        if isinstance(value, (bytes, bytearray)):
            arr = np.load(io.BytesIO(value), allow_pickle=False)
            return torch.from_numpy(arr.astype("float32"))
        raise TypeError(f"action decoder: unsupported value type {type(value)}")

    def _decode_jpeg_to_image(self, value):
        """Decode JPEG/PNG bytes → normalized 224×224 tensor via _decode_image."""
        import io

        from PIL import Image

        if isinstance(value, (bytes, bytearray)):
            img = Image.open(io.BytesIO(value)).convert("RGB")
            return self._decode_image(img)
        return self._decode_image(value)

    def _decoder_for(self, modality: str):
        return {
            "image": self._decode_image,
            "time_series": self._decode_time_series,
            "graph": self._decode_graph,
            "pose": self._decode_pose,
            "action": self._decode_action,
        }[modality]

    # Delimiter written by the WebDataset converters (see
    # scripts/convert_scits_to_webdataset._compose_text, which emits
    # "Question: ...\nAnswer: ...").
    _QA_ANSWER_DELIM = "Answer:"

    def _prompt_token_len(self, caption: str) -> int:
        """Token length of the prompt portion of a QA caption.

        Returns 0 when the caption has no answer delimiter — i.e. it is not
        QA-structured (captioning), so the entire string is the target and
        nothing should be masked.

        Without this, `labels=batch["text"]` supervises the question as well as
        the answer. On SciTS that is ~75% of supervised tokens spent on one of
        only 4 fixed question templates, so loss falls by memorising the
        templates while learning nothing about the series.
        """
        if not caption or self._QA_ANSWER_DELIM not in caption:
            return 0
        head, _, _tail = caption.partition(self._QA_ANSWER_DELIM)
        prompt_str = head + self._QA_ANSWER_DELIM
        # Tokenize the prompt alone and use its length as a prefix count. This
        # mirrors the interleaved path's accounting in
        # StreamingMultimodalDataset (BPE boundary effects at the split point
        # can shift this by a token; the answer is the remainder either way).
        ids = self.tokenizer(
            prompt_str,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=self.max_length,
        ).input_ids.squeeze(0)
        return int(ids.shape[0])

    # ---- iter -------------------------------------------------------------
    def __iter__(self) -> Iterator[dict[str, Any]]:
        modality = self._primary_modality
        if modality in self._COMPOSITE_MODALITIES:
            yield from self._iter_composite(modality)
            return
        decoder = self._decoder_for(modality)
        for sample in self.multi_ds:
            try:
                if modality not in sample:
                    # MultiWebDataset's per-modality _process fn is supposed to
                    # always produce this key — if it didn't, something is
                    # wrong with the shard layout. Skip + warn instead of
                    # yielding a text-only sample that would crash the collator.
                    logger.warning(
                        f"ModalityAwareWebDatasetWrapper: sample missing key "
                        f"{modality!r}; available keys={list(sample.keys())}; skipping"
                    )
                    continue
                out: dict[str, Any] = {modality: decoder(sample[modality])}

                caption = sample.get("text") or sample.get("caption", "")
                if isinstance(caption, (bytes, bytearray)):
                    caption = caption.decode("utf-8", errors="replace")
                tokens = self.tokenizer(
                    str(caption),
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=self.max_length,
                ).input_ids.squeeze(0)
                out["text"] = tokens
                # Number of leading text tokens that are prompt, not target.
                # The model masks these to -100 so the loss is computed on the
                # answer only. 0 means "supervise everything" (correct for
                # captioning, where the whole string is the target).
                out["_prompt_len"] = self._prompt_token_len(str(caption))

                metadata = sample.get("metadata", {})
                data_type = metadata.get("_data_type", "unknown") if isinstance(metadata, dict) else "unknown"
                sample_id = (
                    metadata.get("id", metadata.get("global_idx", "unknown"))
                    if isinstance(metadata, dict)
                    else "unknown"
                )
                out["_metadata"] = f"[{data_type}] {sample_id}"

                yield out
            except Exception as e:  # noqa: BLE001
                logger.warning(f"ModalityAwareWebDatasetWrapper: skipping sample: {e}")
                continue

    def _iter_composite(self, modality: str) -> Iterator[dict[str, Any]]:
        """Yield composite-modality samples.

        Currently used for `vla`: the underlying `MultiWebDataset` yields a
        dict with all of {image_head, image_wrist, pose, action, text,
        metadata} (per `_process_vla_sample`). We decode each typed payload
        through its single-modality decoder so callers see the same tensor
        shapes/dtypes as if they had run a single-modality wrapper for
        each. Keeps the VLA trainer's `VLABatch` reshape narrow.
        """
        if modality != "vla":
            raise NotImplementedError(
                f"_iter_composite: unsupported composite modality {modality!r}"
            )
        for sample in self.multi_ds:
            try:
                missing = [
                    k for k in ("image_head", "image_wrist", "pose", "action")
                    if k not in sample
                ]
                if missing:
                    logger.warning(
                        f"ModalityAwareWebDatasetWrapper(vla): sample missing "
                        f"keys={missing}; available={list(sample.keys())}; skipping"
                    )
                    continue

                out: dict[str, Any] = {
                    "image_head": self._decode_jpeg_to_image(sample["image_head"]),
                    "image_wrist": self._decode_jpeg_to_image(sample["image_wrist"]),
                    "pose": self._decode_pose(sample["pose"]),
                    "action": self._decode_action(sample["action"]),
                }

                caption = sample.get("text", "")
                if isinstance(caption, (bytes, bytearray)):
                    caption = caption.decode("utf-8", errors="replace")
                tokens = self.tokenizer(
                    str(caption),
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=self.max_length,
                ).input_ids.squeeze(0)
                out["text"] = tokens
                out["text_attention_mask"] = torch.ones_like(tokens, dtype=torch.long)

                metadata = sample.get("metadata", {})
                if isinstance(metadata, dict):
                    data_type = metadata.get("_data_type", "vla")
                    ep = metadata.get("episode", "?")
                    step = metadata.get("step", "?")
                    task = metadata.get("task_index", "?")
                    out["_metadata"] = f"[{data_type}] ep={ep} step={step} task={task}"
                else:
                    out["_metadata"] = "[vla] unknown"

                yield out
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    f"ModalityAwareWebDatasetWrapper(vla): skipping sample: {e}"
                )
                continue

    def __len__(self) -> int:
        return len(self.multi_ds)

    def get_stats(self) -> dict:
        return self._stats

    @property
    def active_modalities(self):
        return set(self._modalities)


class BucketedMultiWebDatasetWrapper(_IterableDataset):
    """
    Wrapper around MultiWebDatasetWrapper that groups samples by sequence length.

    This significantly improves training throughput when using datasets with
    variable sequence lengths by:
    1. Buffering samples into a pool
    2. Sorting by text length
    3. Yielding batches of similar-length samples

    This reduces padding waste from O(max_len_in_batch) to O(bucket_range).

    Example:
        Without bucketing: batch with lengths [10, 500, 20, 15] pads to 500 tokens
        With bucketing: batch with lengths [10, 15, 12, 20] pads to 20 tokens

    Usage:
        dataset = BucketedMultiWebDatasetWrapper(
            tokenizer=tokenizer,
            buffer_size=1000,  # Buffer 1000 samples before sorting
            ...
        )
    """

    def __init__(
        self,
        tokenizer,
        config: dict | None = None,
        config_path: str = "src/conf/data/daos_datasets.yaml",
        groups: str | list[str] = "all",
        daos_mount: str | None = None,
        weight_overrides: dict[str, float] | None = None,
        proportion_overrides: dict[str, float] | None = None,
        world_size: int = 1,
        rank: int = 0,
        max_length: int = 2048,
        batch_size: int = 1,
        max_steps: int = 1000000,
        # Bucketing parameters
        buffer_size: int = 1000,
        num_buckets: int = 8,
        shuffle_buckets: bool = True,
        shuffle_buffer: int = 32768,
        # Phase 3.2: pluggable bucket key (default = text length, preserves
        # historical image-only behavior). For time_series shards pass a
        # function that reads sample["time_series"].shape[0]; for graph
        # shards pass one that reads num_nodes. None = legacy text-length.
        bucket_key_fn=None,
        model_config=None,
        partition_by: str = "global",
        resampled: bool = True,
    ):
        """
        Initialize the bucketed wrapper.

        Args:
            tokenizer: HuggingFace tokenizer
            config: Pre-loaded config dict
            config_path: Path to DAOS dataset configuration
            groups: Dataset groups to load
            daos_mount: Override DAOS mount path
            weight_overrides: Per-dataset weight overrides
            proportion_overrides: Per-dataset proportion overrides
            world_size: Total distributed workers
            rank: Current worker rank
            max_length: Maximum token length
            batch_size: Not used directly (for API compat)
            max_steps: Not used directly (for API compat)
            buffer_size: Number of samples to buffer before sorting (larger = better bucketing)
            num_buckets: Number of length buckets (more = tighter grouping)
            shuffle_buckets: Whether to shuffle within buckets (recommended)
            shuffle_buffer: Sample-level shuffle buffer passed to WebDataset.
            partition_by: Forwarded to MultiWebDatasetWrapper.
            resampled: Whether the underlying WebDataset is infinite/resampled.
        """
        # Create underlying wrapper
        self._base = MultiWebDatasetWrapper(
            tokenizer=tokenizer,
            config=config,
            config_path=config_path,
            groups=groups,
            daos_mount=daos_mount,
            weight_overrides=weight_overrides,
            proportion_overrides=proportion_overrides,
            world_size=world_size,
            rank=rank,
            max_length=max_length,
            batch_size=batch_size,
            max_steps=max_steps,
            model_config=model_config,
            shuffle_buffer=shuffle_buffer,
            partition_by=partition_by,
            resampled=resampled,
        )

        self.buffer_size = buffer_size
        self.num_buckets = num_buckets
        self.shuffle_buckets = shuffle_buckets
        self.resampled = resampled
        self.max_length = max_length
        self.bucket_key_fn = bucket_key_fn  # None ⇒ legacy text-length

        # Stats
        self._stats = self._base._stats.copy()
        self._stats["bucketing"] = {
            "buffer_size": buffer_size,
            "num_buckets": num_buckets,
        }

        logger.info(
            f"BucketedMultiWebDatasetWrapper: buffer_size={buffer_size}, "
            f"num_buckets={num_buckets}, shuffle={shuffle_buckets}"
        )

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """
        Iterate with length-based bucketing.

        Buffers samples, sorts by length, then yields in order.

        With resampled=True on the underlying WebDatasets, the base iterator
        is infinite (never raises StopIteration), so this loop runs until the
        training loop stops requesting samples. With resampled=False, this
        iterator flushes its final partial buffer and stops at the epoch
        boundary.
        """

        buffer = []
        base_iter = iter(self._base)

        while True:
            exhausted = False
            # Fill buffer from current iterator
            while len(buffer) < self.buffer_size:
                try:
                    sample = next(base_iter)
                except StopIteration:
                    if not self.resampled:
                        if buffer:
                            logger.info(
                                f"[BUCKETING] Base iterator exhausted with {len(buffer)} "
                                f"samples in final buffer (buffer_size={self.buffer_size})."
                            )
                            exhausted = True
                            break
                        return
                    logger.info(
                        f"[BUCKETING] Base iterator exhausted with {len(buffer)} "
                        f"samples in buffer (buffer_size={self.buffer_size}). "
                        f"Restarting iterator."
                    )
                    base_iter = iter(self._base)
                    continue

                if self.bucket_key_fn is not None:
                    try:
                        key = int(self.bucket_key_fn(sample))
                    except Exception as _e_bk:  # noqa: BLE001
                        logger.debug(f"[BUCKETING] bucket_key_fn failed: {_e_bk}; falling back to text len")
                        key = (
                            sample["text"].shape[0]
                            if hasattr(sample["text"], "shape")
                            else len(sample["text"])
                        )
                else:
                    key = (
                        sample["text"].shape[0]
                        if hasattr(sample["text"], "shape")
                        else len(sample["text"])
                    )
                buffer.append((key, sample))

            # Yield the full sorted buffer
            yield from self._yield_sorted_buffer(buffer)
            buffer = []

            if exhausted:
                return

    def _yield_sorted_buffer(self, buffer):
        """
        Sort buffer by length and yield samples.

        Optionally divides into buckets and shuffles within buckets
        to maintain some randomness while keeping similar lengths together.
        """

        # Sort by length
        buffer.sort(key=lambda x: x[0])

        if self.shuffle_buckets and self.num_buckets > 1:
            # Divide into buckets and shuffle within each
            bucket_size = len(buffer) // self.num_buckets
            if bucket_size > 0:
                shuffled = []
                for i in range(self.num_buckets):
                    start = i * bucket_size
                    end = (
                        start + bucket_size if i < self.num_buckets - 1 else len(buffer)
                    )
                    bucket = buffer[start:end]
                    random.shuffle(bucket)
                    shuffled.extend(bucket)
                buffer = shuffled

        # Yield samples (without length)
        for _, sample in buffer:
            yield sample

    def __len__(self) -> int:
        return len(self._base)

    def get_stats(self) -> dict:
        return self._stats

    def fast_forward_batches(self, num_batches: int, batch_size: int) -> int:
        """Delegate resume fast-forward to the underlying WebDataset wrapper."""
        return self._base.fast_forward_batches(num_batches, batch_size)

    @property
    def active_modalities(self):
        return self._base.active_modalities


def create_multi_webdataset_loader(
    config_path: str = "src/conf/data/daos_datasets.yaml",
    groups: str | list[str] = "all",
    daos_mount: str | None = None,
    weight_overrides: dict[str, float] | None = None,
    batch_size: int = 8,
    num_workers: int = 4,
    world_size: int = 1,
    rank: int = 0,
    resampled: bool = True,
) -> "torch.utils.data.DataLoader":
    """
    Create a DataLoader for multi-dataset WebDataset.

    Args:
        config_path: Path to DAOS dataset configuration
        groups: Preset, group name, or list of groups
        daos_mount: Override DAOS mount path
        weight_overrides: Per-dataset weight overrides
        batch_size: Batch size per GPU
        num_workers: DataLoader workers
        world_size: Total distributed workers
        rank: Current rank
        resampled: Whether to sample shards with replacement

    Returns:
        DataLoader instance
    """
    import torch.utils.data as data

    config = load_daos_config(config_path)

    dataset = MultiWebDataset(
        config=config,
        groups=groups,
        daos_mount=daos_mount,
        weight_overrides=weight_overrides,
        world_size=world_size,
        rank=rank,
        resampled=resampled,
    )

    # WebDataset is already an IterableDataset
    loader = data.DataLoader(
        dataset._dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )

    return loader


if __name__ == "__main__":
    # Quick test
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="src/conf/data/daos_datasets.yaml")
    parser.add_argument("--groups", default="all")
    parser.add_argument("--daos-mount", default=None)
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    config = load_daos_config(args.config)

    print("\n=== Dataset Configuration ===")
    groups = args.groups.split(",") if "," in args.groups else args.groups
    datasets = get_datasets_for_groups(config, groups)
    datasets = normalize_weights(datasets)

    for ds in datasets:
        print(
            f"  {ds['name']} ({ds['group']}): {ds['samples']} samples, weight={ds['weight']:.2f}"
        )

    print(
        f"\nTotal: {len(datasets)} datasets, {sum(d['samples'] for d in datasets)} samples"
    )

    # Try loading if DAOS is mounted
    daos_mount = args.daos_mount or os.environ.get("DAOS_MOUNT")
    if daos_mount and os.path.isdir(daos_mount):
        print(f"\n=== Testing data loading from {daos_mount} ===")
        try:
            dataset = MultiWebDataset(
                config=config,
                groups=groups,
                daos_mount=daos_mount,
            )

            print(f"\nSampling {args.samples} examples...")
            for i, sample in enumerate(dataset):
                if i >= args.samples:
                    break
                print(
                    f"  [{i}] image: {sample['image'].size}, caption: {sample['caption'][:50]}..."
                )

            print("\nSUCCESS!")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print(f"\nSkipping data loading test (DAOS not mounted at {daos_mount})")
