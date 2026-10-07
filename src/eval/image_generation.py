"""Offline image validation records, boundary parity, and paired qualification.

This module deliberately loads neither models nor network clients. Fixture tests of
these utilities are not evidence of released-checkpoint or accelerator parity.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import random
import re
import subprocess
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

TASKS = ("text_to_image", "editing", "multi_reference")
SHA_REVISION = re.compile(r"^[0-9a-f]{40}$")
PARITY_BOUNDARIES = (
    "condition.positive",
    "mask.positive",
    "latents.initial",
    "prediction.step0",
    "latents.final",
    "pixels",
)
FULL_PIPELINE_BOUNDARIES = PARITY_BOUNDARIES + (
    "token_ids.positive",
    "token_mask.positive",
    "position_ids.positive",
    "condition.negative",
    "mask.negative",
    "latents.step0",
    "latents.vae_input",
    "pixels.vae_output",
    "schedule.timesteps",
)


class ValidationBlocked(ValueError):
    """A missing input makes the requested validation unproven."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str | Path, value: Any) -> None:
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


@dataclass(frozen=True)
class ImageValidationCase:
    case_id: str
    task: str
    prompt: str
    reference_paths: tuple[str, ...]
    group: str
    split: str
    seeds: tuple[int, ...]
    stratum: str = "unspecified"
    width: int = 512
    height: int = 512

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_cases(
    path: str | Path, data_root: str | Path | None = None, *, smoke: bool = False
) -> list[ImageValidationCase]:
    """Read every fixed case, rejecting invalid records rather than dropping them.

    ``reference_paths`` are source inputs. Target images or target latents have no
    field in this generation schema. Every record has three predetermined seeds.
    """
    manifest = Path(path).resolve()
    base = Path(data_root).resolve() if data_root else manifest.parent
    cases: list[ImageValidationCase] = []
    seen: set[str] = set()
    group_splits: dict[str, str] = {}
    source_groups: dict[str, str] = {}
    for line_number, line in enumerate(manifest.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if any(
                k in row
                for k in ("target", "target_path", "target_paths", "target_image", "target_latents")
            ):
                raise ValidationBlocked("targets are not generation conditioning")
            identifier = row.get("case_id", row.get("id"))
            if (
                not isinstance(identifier, str)
                or not re.fullmatch(r"[A-Za-z0-9_.-]+", identifier)
                or identifier in (".", "..")
            ):
                raise ValidationBlocked("case_id must be a safe nonempty identifier")
            if identifier in seen:
                raise ValidationBlocked(f"duplicate case_id: {identifier}")
            task = {"t2i": "text_to_image", "edit": "editing", "in_context": "multi_reference"}.get(
                row["task"], row["task"]
            )
            if task not in TASKS:
                raise ValidationBlocked(f"unknown task: {task}")
            prompt = row["prompt"]
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValidationBlocked("prompt must be nonempty text")
            refs = row.get("reference_paths", row.get("source_images"))
            if not isinstance(refs, list) or any(not isinstance(v, str) for v in refs):
                raise ValidationBlocked("reference_paths must be a list of paths")
            if (
                (task == "text_to_image" and refs)
                or (task == "editing" and not refs)
                or (task == "multi_reference" and len(refs) < 2)
            ):
                raise ValidationBlocked("source count does not match task")
            paths = tuple(str((base / value).resolve()) for value in refs)
            if any(not Path(value).is_file() for value in paths):
                raise ValidationBlocked("source/reference asset missing")
            seeds = row["seeds"]
            allowed_counts = (1, 2, 3) if smoke else (3,)
            if (
                not isinstance(seeds, list)
                or len(seeds) not in allowed_counts
                or any(type(v) is not int or v < 0 or v >= 2**63 for v in seeds)
                or len(set(seeds)) != len(seeds)
            ):
                raise ValidationBlocked(
                    "exactly three distinct nonnegative 63-bit seeds are required"
                )
            group = row.get("group")
            if group is None and row.get("group_ids"):
                groups = row["group_ids"]
                if not isinstance(groups, list) or len(groups) != 1:
                    raise ValidationBlocked(
                        "evaluation needs one preregistered connected-component group per case"
                    )
                group = groups[0]
            split = row["split"]
            if not isinstance(group, str) or not group or not isinstance(split, str) or not split:
                raise ValidationBlocked("group and split are required nonempty strings")
            if group in group_splits and group_splits[group] != split:
                raise ValidationBlocked(f"source/subject group {group} leaks across splits")
            for source_path in paths:
                if source_path in source_groups and source_groups[source_path] != group:
                    raise ValidationBlocked(
                        "shared source images must use the same connected-component group"
                    )
                source_groups[source_path] = group
            width, height = row.get("width", 512), row.get("height", 512)
            if any(type(v) is not int or v <= 0 or v % 16 for v in (width, height)):
                raise ValidationBlocked("width/height must be positive multiples of 16")
            cases.append(
                ImageValidationCase(
                    identifier,
                    task,
                    prompt,
                    paths,
                    group,
                    split,
                    tuple(seeds),
                    row.get("stratum", "unspecified"),
                    width,
                    height,
                )
            )
            seen.add(identifier)
            group_splits[group] = split
        except (KeyError, TypeError, ValueError) as exc:
            raise ValidationBlocked(f"{manifest}:{line_number}: {exc}") from exc
    if not cases:
        raise ValidationBlocked("case manifest is empty")
    return cases


def suite_summary(cases: list[ImageValidationCase], stage: str = "P0") -> dict[str, Any]:
    counts = Counter(case.task for case in cases)
    required = 10 if stage == "P0" else 200
    complete = all(counts[task] == required for task in TASKS) and all(
        len(case.seeds) == 3 for case in cases
    )
    return {
        "stage": stage,
        "case_count": len(cases),
        "output_count": sum(len(c.seeds) for c in cases),
        "task_counts": dict(counts),
        "complete_prespecified_size": complete,
    }


def environment_preflight(
    checkpoint: str | Path | None,
    revision: str,
    upstream: str | Path | None,
    upstream_revision: str,
) -> dict[str, Any]:
    """Verify installed packages, cache metadata and file hashes without model imports."""
    errors: list[str] = []
    packages: dict[str, Any] = {}
    for name in (
        "torch",
        "numpy",
        "Pillow",
        "diffusers",
        "transformers",
        "accelerate",
        "safetensors",
        "omegaconf",
        "timm",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
            errors.append(f"missing distribution: {name}")
    for name, expected in (("diffusers", "0.33.1"), ("transformers", "4.51.3")):
        if packages[name] != expected:
            errors.append(f"pinned reference requires {name}=={expected}, found {packages[name]!r}")
    for label, value in (
        ("checkpoint revision", revision),
        ("upstream revision", upstream_revision),
    ):
        if not SHA_REVISION.fullmatch(value or ""):
            errors.append(f"{label} must be a full lowercase 40-character commit SHA")
    checkpoint_info: dict[str, Any] = {
        "path": str(checkpoint) if checkpoint else None,
        "revision": revision,
    }
    if not checkpoint or not Path(checkpoint).is_dir():
        errors.append("local checkpoint snapshot missing; downloads are disabled")
    else:
        root = Path(checkpoint).resolve()
        # A detached copied directory needs an explicit provenance sidecar.
        pinned_snapshot = root.name == revision and root.parent.name == "snapshots"
        sidecar = root / "prism_checkpoint_provenance.json"
        verified_files = False
        if sidecar.is_file():
            try:
                provenance = json.loads(sidecar.read_text())
                entries = provenance.get("files", {})
                verified_files = provenance.get("revision") == revision and bool(entries)
                for name, digest in entries.items():
                    item = (root / name).resolve()
                    if (
                        not item.is_relative_to(root)
                        or not item.is_file()
                        or sha256_file(item) != digest
                    ):
                        verified_files = False
            except (ValueError, OSError, AttributeError):
                verified_files = False
        if not pinned_snapshot and not verified_files:
            errors.append(
                "checkpoint revision unverifiable: use a pinned HF snapshots/<sha> directory or a verified prism_checkpoint_provenance.json file inventory"
            )
        weights = sorted(str(p.relative_to(root)) for p in root.rglob("*.safetensors"))
        configs = sorted(str(p.relative_to(root)) for p in root.rglob("*.json"))
        checkpoint_info.update(
            {
                "weights": weights,
                "configs": configs,
                "revision_verified": pinned_snapshot or verified_files,
            }
        )
        if not weights or not configs:
            errors.append("checkpoint weights/configuration missing")
        for component in ("transformer", "vae", "mllm", "processor", "scheduler"):
            if not (root / component).is_dir():
                errors.append(f"checkpoint component missing: {component}")
        for index in root.rglob("*.safetensors.index.json"):
            try:
                weight_map = json.loads(index.read_text())["weight_map"]
                if any(not (index.parent / shard).is_file() for shard in set(weight_map.values())):
                    errors.append(f"checkpoint shard missing: {index.relative_to(root)}")
            except (ValueError, KeyError, TypeError):
                errors.append(f"invalid checkpoint index: {index.relative_to(root)}")
    upstream_info: dict[str, Any] = {
        "path": str(upstream) if upstream else None,
        "revision": upstream_revision,
    }
    if not upstream or not Path(upstream).is_dir():
        errors.append("local pinned OmniGen2 source checkout missing")
    else:
        try:
            actual = subprocess.check_output(
                ["git", "--no-optional-locks", "-C", str(upstream), "rev-parse", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            dirty = subprocess.check_output(
                [
                    "git",
                    "--no-optional-locks",
                    "-C",
                    str(upstream),
                    "status",
                    "--porcelain",
                    "--untracked-files=no",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            upstream_info.update({"actual_revision": actual, "tracked_changes": bool(dirty)})
            if actual != upstream_revision or dirty:
                errors.append("OmniGen2 source does not match the clean pinned upstream revision")
        except (OSError, subprocess.CalledProcessError):
            errors.append("cannot verify upstream Git revision")
    return {
        "status": "blocked" if errors else "ready",
        "model_loaded": False,
        "downloads_allowed": False,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "checkpoint": checkpoint_info,
        "upstream": upstream_info,
        "errors": errors,
    }


def _array(value: Any) -> Any:
    import numpy as np

    if hasattr(value, "detach"):
        value = value.detach().cpu()
        # NumPy has no portable bfloat16; dtype is recorded separately by caller.
        if str(value.dtype) == "torch.bfloat16":
            value = value.float()
        value = value.numpy()
    result = np.asarray(value)
    if result.dtype.hasobject:
        raise ValidationBlocked("object tensors cannot be archived")
    return result


def archive_trace(directory: str | Path, trace: Mapping[str, Any]) -> dict[str, Any]:
    """Store every supplied boundary with shape/dtype/hash and JSON provenance."""
    import numpy as np

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    record: dict[str, Any] = {
        "tensors": {},
        "discrete": trace.get("discrete", {}),
        "provenance": trace.get("provenance", {}),
    }
    tensors = trace.get("tensors", {})
    if not tensors:
        raise ValidationBlocked("backend captured no numerical boundaries")
    for index, (name, value) in enumerate(sorted(tensors.items())):
        array = _array(value)
        filename = f"tensor-{index:03d}.npy"
        np.save(root / filename, array, allow_pickle=False)
        record["tensors"][name] = {
            "path": filename,
            "shape": list(array.shape),
            "dtype": str(getattr(value, "dtype", array.dtype)),
            "storage_dtype": str(array.dtype),
            "sha256": sha256_file(root / filename),
        }
    write_json(root / "trace.json", record)
    return record


def read_trace(directory: str | Path) -> dict[str, Any]:
    import numpy as np

    root = Path(directory).resolve()
    record = json.loads((root / "trace.json").read_text())
    tensors = {}
    for name, entry in record["tensors"].items():
        path = (root / entry["path"]).resolve()
        if not path.is_relative_to(root) or sha256_file(path) != entry["sha256"]:
            raise ValidationBlocked(f"artifact hash/path mismatch: {name}")
        tensors[name] = np.load(path, allow_pickle=False)
    return {
        "tensors": tensors,
        "discrete": record["discrete"],
        "provenance": record["provenance"],
        "dtypes": {key: entry["dtype"] for key, entry in record["tensors"].items()},
    }


def compare_traces(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    tolerances: Mapping[str, Any] | None = None,
    required: tuple[str, ...] = PARITY_BOUNDARIES,
) -> dict[str, Any]:
    """Exact by default; only preregistered boundary-specific tolerances may relax it."""
    import numpy as np

    tolerances = tolerances or {}
    left, right = reference.get("tensors", {}), candidate.get("tensors", {})
    failures: list[str] = []
    results = {}
    if reference.get("discrete") != candidate.get("discrete"):
        failures.append("discrete inputs/masks/positions/specification differ")
    for name in sorted(set(left) | set(right) | set(required)):
        if name not in left or name not in right:
            failures.append(f"missing boundary: {name}")
            continue
        a, b = _array(left[name]), _array(right[name])
        dtype_a = reference.get("dtypes", {}).get(name, str(getattr(left[name], "dtype", a.dtype)))
        dtype_b = candidate.get("dtypes", {}).get(name, str(getattr(right[name], "dtype", b.dtype)))
        dtype_a, dtype_b = dtype_a.removeprefix("torch."), dtype_b.removeprefix("torch.")
        if a.shape != b.shape or dtype_a != dtype_b:
            failures.append(f"shape/dtype mismatch: {name}")
            continue
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            failures.append(f"nonfinite boundary: {name}")
            continue
        tolerance = tolerances.get(name, {})
        atol, rtol = float(tolerance.get("atol", 0)), float(tolerance.get("rtol", 0))
        if not math.isfinite(atol) or not math.isfinite(rtol) or min(atol, rtol) < 0:
            raise ValidationBlocked(f"invalid registered tolerance: {name}")
        if (atol or rtol) and (
            not tolerance.get("rationale") or not tolerance.get("repeatability_artifact")
        ):
            raise ValidationBlocked(
                f"nonzero tolerance requires rationale and repeatability_artifact: {name}"
            )
        if (
            name.startswith("mask.")
            or name == "latents.initial"
            or not np.issubdtype(a.dtype, np.floating)
        ) and (atol or rtol):
            raise ValidationBlocked(f"discrete/pass-through boundary must be exact: {name}")
        matches = (
            bool(np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=False))
            if (atol or rtol)
            else bool(np.array_equal(a, b))
        )
        max_abs = (
            float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))) if a.size else 0.0
        )
        results[name] = {
            "passed": matches,
            "max_absolute_error": max_abs,
            "atol": atol,
            "rtol": rtol,
        }
        if not matches:
            failures.append(f"numerical mismatch: {name}")
    return {"passed": not failures, "boundaries": results, "failures": failures}


def clustered_paired_interval(
    pairs: list[tuple[str, float, float]],
    *,
    bootstrap_samples: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """One-sided percentile CI, resampling whole source/subject groups.

    Each pair is already averaged over the prespecified sampling/training seeds.
    The estimand is the case-weighted mean; repeated seeds are not independent cases.
    """
    if not pairs or bootstrap_samples < 100 or not 0 < alpha < 0.5:
        raise ValidationBlocked("invalid paired bootstrap specification")
    clusters: dict[str, list[float]] = defaultdict(list)
    for group, full, comparator in pairs:
        if not all(math.isfinite(x) for x in (full, comparator)):
            raise ValidationBlocked("nonfinite evaluation score")
        clusters[group].append(full - comparator)
    keys = sorted(clusters)
    if len(keys) < 2:
        raise ValidationBlocked("at least two independent source/subject clusters required")
    rng = random.Random(seed)
    draws = []
    for _ in range(bootstrap_samples):
        values = [x for group in rng.choices(keys, k=len(keys)) for x in clusters[group]]
        draws.append(sum(values) / len(values))
    draws.sort()
    lower = draws[max(0, math.ceil(alpha * bootstrap_samples) - 1)]
    delta = sum(x for values in clusters.values() for x in values) / len(pairs)
    return {
        "difference": delta,
        "lower_one_sided_95": lower,
        "alpha": alpha,
        "clusters": len(keys),
        "cases": len(pairs),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
    }


def evaluate_qualification(
    cases: list[ImageValidationCase], evaluation: Mapping[str, Any], protocol: Mapping[str, Any]
) -> dict[str, Any]:
    """Score a complete, fixed evaluator export; never silently drop failed outputs.

    Protocol declares training_seeds, necessary_ablations per task, auxiliary_gates,
    and preregistration metadata. Rows: case_id, seed, training_seed, variant,
    scores ({success: [0,1], ...}), failed (optional). Reference/ablations are
    evaluated for each training seed too, preserving paired sampling settings.
    """
    failures: list[str] = []
    for field in ("registered_at", "protocol_id", "evaluator_revision", "max_statistical_looks"):
        if not protocol.get(field):
            failures.append(f"missing preregistration field: {field}")
    if protocol.get("max_statistical_looks") != 1:
        failures.append("this implementation supports one prespecified statistical look only")
    train_seeds = protocol.get("training_seeds", [])
    if len(set(train_seeds)) < 2:
        failures.append("at least two finalist training seeds required")
    summary = suite_summary(cases, "P4")
    if not summary["complete_prespecified_size"]:
        failures.append("sealed qualification requires exactly 200 cases per task")
    if any(c.split not in ("test", "qualification", "sealed") for c in cases):
        failures.append("qualification cases must belong to a sealed test split")
    if evaluation.get("evaluator_revision") != protocol.get("evaluator_revision"):
        failures.append("evaluator revision differs from preregistration")
    case_map = {case.case_id: case for case in cases}
    rows = {}
    for row in evaluation.get("records", []):
        try:
            key = (row["case_id"], row["seed"], row["training_seed"], row["variant"])
            if key in rows or key[0] not in case_map:
                raise ValidationBlocked("duplicate or unknown scored case")
            if key[1] not in case_map[key[0]].seeds or key[2] not in train_seeds:
                raise ValidationBlocked("unregistered sampling/training seed")
            scores = row["scores"]
            if any(
                not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1
                for v in scores.values()
            ):
                raise ValidationBlocked("scores must be finite values in [0,1]")
            if row.get("failed") and scores.get("success") != 0:
                raise ValidationBlocked("generation failures must receive zero success")
            rows[key] = scores
        except (KeyError, TypeError, ValueError) as exc:
            failures.append(str(exc))
    task_reports = {}
    necessary = protocol.get("necessary_ablations", {})
    auxiliary = protocol.get("auxiliary_gates", {})
    min_clusters = protocol.get("min_clusters", 10)
    for task in TASKS:
        selected = [case for case in cases if case.task == task]
        ablations = necessary.get(task, [])
        if not ablations:
            failures.append(f"no preregistered necessary-input ablation for {task}")
        required_auxiliary = {"editing": "preservation", "multi_reference": "identity"}.get(task)
        if required_auxiliary and required_auxiliary not in auxiliary.get(task, {}):
            failures.append(f"separate preservation/identity gate missing for {task}")
        if {case.stratum for case in selected} != {"natural", "procedural"}:
            failures.append(f"both natural and procedural strata must be reported for {task}")
        variants = [
            "full",
            "reference",
            "native_only",
            *ablations,
            *protocol.get("diagnostic_ablations", {}).get(task, []),
        ]
        means: dict[tuple[str, str, str], float] = {}
        task_errors = []
        metrics = ["success", *auxiliary.get(task, {}).keys()]
        for case in selected:
            for variant in variants:
                for metric in metrics:
                    values = []
                    for training_seed in train_seeds:
                        for seed in case.seeds:
                            score = rows.get((case.case_id, seed, training_seed, variant), {}).get(
                                metric
                            )
                            if score is None:
                                task_errors.append(
                                    f"missing score {case.case_id}/{seed}/{training_seed}/{variant}/{metric}"
                                )
                            else:
                                values.append(score)
                    if values:
                        means[(case.case_id, variant, metric)] = sum(values) / len(values)
        report: dict[str, Any] = {"failures": task_errors, "comparisons": {}}
        if not task_errors and selected:
            for variant in variants[1:]:
                pairs = [
                    (
                        c.group,
                        means[c.case_id, "full", "success"],
                        means[c.case_id, variant, "success"],
                    )
                    for c in selected
                ]
                try:
                    interval = clustered_paired_interval(
                        pairs,
                        bootstrap_samples=protocol.get("bootstrap_samples", 2000),
                        seed=protocol.get("bootstrap_seed", 0),
                    )
                    report["comparisons"][variant] = interval
                    if interval["clusters"] < min_clusters:
                        failures.append(f"too few source/subject clusters for {task}")
                except ValidationBlocked as exc:
                    task_errors.append(str(exc))
            comparisons = report["comparisons"]
            if "reference" in comparisons:
                report["noninferiority_passed"] = comparisons["reference"][
                    "lower_one_sided_95"
                ] > -float(protocol.get("noninferiority_margin", 0.05))
                if not report["noninferiority_passed"]:
                    failures.append(f"noninferiority unproven: {task}")
            if ablations and all(a in comparisons for a in ablations):
                strongest = min(ablations, key=lambda a: comparisons[a]["difference"])
                report["strongest_necessary_ablation"] = strongest
                # Requiring each preregistered ablation also avoids post-hoc selection
                # of an easy comparator; report the strongest point estimate.
                report["necessary_input_passed"] = all(
                    comparisons[a]["difference"] >= float(protocol.get("ablation_margin", 0.15))
                    and comparisons[a]["lower_one_sided_95"] > 0
                    for a in ablations
                )
                if not report["necessary_input_passed"]:
                    failures.append(f"necessary-input dependence unproven: {task}")
            report["auxiliary"] = {}
            for metric, threshold in auxiliary.get(task, {}).items():
                pairs = [
                    (c.group, means[c.case_id, "full", metric], float(threshold)) for c in selected
                ]
                try:
                    interval = clustered_paired_interval(
                        pairs,
                        bootstrap_samples=protocol.get("bootstrap_samples", 2000),
                        seed=protocol.get("bootstrap_seed", 0),
                    )
                    report["auxiliary"][metric] = interval
                    if interval["lower_one_sided_95"] <= 0:
                        failures.append(f"{task} {metric} criterion unproven")
                except ValidationBlocked as exc:
                    task_errors.append(str(exc))
            report["strata"] = {}
            for stratum in sorted({c.stratum for c in selected}):
                subset = [c for c in selected if c.stratum == stratum]
                stratum_report = {
                    "cases": len(subset),
                    "full_success": sum(means[c.case_id, "full", "success"] for c in subset)
                    / len(subset),
                    "comparisons": {},
                }
                for variant in variants[1:]:
                    try:
                        stratum_report["comparisons"][variant] = clustered_paired_interval(
                            [
                                (
                                    c.group,
                                    means[c.case_id, "full", "success"],
                                    means[c.case_id, variant, "success"],
                                )
                                for c in subset
                            ],
                            bootstrap_samples=protocol.get("bootstrap_samples", 2000),
                            seed=protocol.get("bootstrap_seed", 0),
                        )
                    except ValidationBlocked as exc:
                        task_errors.append(f"{task}/{stratum}: {exc}")
                report["strata"][stratum] = stratum_report
            report["training_seeds"] = {
                str(seed): sum(
                    rows[c.case_id, sample_seed, seed, "full"]["success"]
                    for c in selected
                    for sample_seed in c.seeds
                )
                / sum(len(c.seeds) for c in selected)
                for seed in train_seeds
            }
        failures.extend(task_errors)
        task_reports[task] = report
    return {
        "status": "passed" if not failures else "unproven",
        "suite": summary,
        "tasks": task_reports,
        "failures": failures,
        "claim": "image task qualification only; no scientific-modality qualification",
    }
