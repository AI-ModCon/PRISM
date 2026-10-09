#!/usr/bin/env python3
"""Convert OpenTSLab/SciTS to PRISM-compatible WebDataset shards.

Output layout:
    <output_dir>/
        shards/
            <dataset_name>-000000.tar
            ...
        val_shards/
            <dataset_name>-000000.tar
            ...
        manifest.json

Each sample inside a shard contains exactly the keys expected by the
time-series path in MultiWebDataset:
    <key>.ts.npy
    <key>.text
    <key>.meta.json

The manifest schema is compatible with:
  - src/data/multi_webdataset.py (_get_shards_for_dataset)
  - scripts/stage_shards.py

python applications/timeseries/convert_scits_to_webdataset.py \
    --scits-dir /flare/ModCon/pemami/data/SciTS \
    --output-dir /flare/ModCon/pemami/data/SciTS-forecast \
    --dataset-name scits \
    --samples-per-shard 1000 \
    --val-ratio 0.10 
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

FORECAST_CAP = "Forecast the time series."

# Synced from SciTS/process/infer_eval_utils.py.
DATASET_TO_TASK = {
    "ASU01_ASG02": "event_detection", "ASU03": "classification", "EAU01_EAG02": "event_detection", "BIU01": "classification", "BIU02": "classification", "BIU03": "classification", "MEU01": "anomaly_detection", "MEU02": "anomaly_detection", "MEG03": "forecasting", "MEU04": "mcq", "ECG01": "forecasting", "ECG02": "forecasting", "ECU03": "mcq", "NEU01": "anomaly_detection", "NEU02": "classification", "NEG03": "forecasting", "NEG04": "imputation", "NEU05": "classification", "NEU06": "classification", "ENG01": "synthesize", "ENG02": "forecasting", "ENG03": "forecasting", "ENG04": "forecasting", "ENG05": "imputation", "PHU01": "classification", "PHG02": "forecasting", "PHG03": "imputation", "PHU04": "anomaly_detection", "PHU05": "anomaly_detection", "PHU06": "classification", "URG01": "forecasting", "URG02": "forecasting", "URG03": "imputation", "URU04": "anomaly_detection", "URG05": "forecasting", "MFU01_MFU02": "classification", "MFU03": "anomaly_detection", "RAU01": "classification", "RAU02": "classification", "MAG01": "forecasting",
}

def _as_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _extract_input_ts_path(example: dict[str, Any]) -> str:
    """Resolve SciTS input time-series path from supported metadata layouts."""
    input_ts = example.get("input_ts")
    if isinstance(input_ts, dict):
        return _as_str(input_ts.get("path")).strip()
    if isinstance(input_ts, str):
        return input_ts.strip()
    return ""


def _extract_gt_ts_path(example: dict[str, Any]) -> str:
    """Resolve SciTS ground-truth time-series path when present."""
    gt_ts = example.get("gt_ts")
    if isinstance(gt_ts, dict):
        return _as_str(gt_ts.get("path")).strip()
    if isinstance(gt_ts, str):
        return gt_ts.strip()
    return ""


def _normalize_task_id(task_id: Any) -> str:
    if isinstance(task_id, (list, tuple)):
        return "_".join(_as_str(value).strip() for value in task_id if _as_str(value).strip())
    return _as_str(task_id).strip()


def _gt_result_to_text(gt_result: Any) -> str:
    """Render structured SciTS gt_result into a compact textual answer."""
    if not isinstance(gt_result, dict):
        return _as_str(gt_result).strip()

    # MCQ / direct answer-style payloads.
    if gt_result.get("answer") is not None:
        answer = _as_str(gt_result.get("answer")).strip()
        if answer:
            return answer

    # Classification payload.
    gt_class = gt_result.get("gt_class")
    if isinstance(gt_class, dict):
        labels: list[str] = []
        for values in gt_class.values():
            if isinstance(values, list):
                labels.extend([_as_str(v).strip() for v in values if _as_str(v).strip()])
            else:
                label = _as_str(values).strip()
                if label:
                    labels.append(label)
        if labels:
            return ", ".join(labels)

    # Anomaly detection / event localization payload.
    if "contain" in gt_result:
        contain = bool(gt_result.get("contain"))
        start_time = gt_result.get("start_time")
        if start_time is not None:
            return f"contain={contain}, start_time={start_time}"
        return f"contain={contain}"

    return json.dumps(gt_result, ensure_ascii=True, sort_keys=True)


def _compose_text(example: dict[str, Any], is_forecasting: bool = False) -> str:
    """Build instruction/response text for PRISM time-series training."""
    input_text = _as_str(
        example.get("input_text")
        or example.get("input")
        or example.get("instruction")
        or example.get("question")
        or example.get("prompt")
    ).strip()
    if is_forecasting:
        return input_text
    output_text = _as_str(
        example.get("gt_text")
        or example.get("output")
        or example.get("answer")
        or example.get("response")
        or example.get("target_text")
    ).strip()
    if not output_text and example.get("gt_result") is not None:
        output_text = _gt_result_to_text(example.get("gt_result"))

    if input_text and output_text:
        return f"Question: {input_text}\nAnswer: {output_text}".strip()
    if input_text:
        return f"Question: {input_text}".strip()
    if output_text:
        return f"Answer: {output_text}".strip()
    return FORECAST_CAP

def _stable_val_bucket(example: dict[str, Any], val_ratio: float) -> bool:
    """Deterministically assign a sample to validation using a stable hash."""
    if val_ratio <= 0:
        return False

    sample_id = (
        _as_str(example.get("id"))
        or _as_str(example.get("sample_id"))
        or _as_str(example.get("uuid"))
        or _as_str(example.get("idx"))
        or json.dumps(example, sort_keys=True, default=str)
    )
    digest = hashlib.md5(sample_id.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return bucket < val_ratio


def _load_series_from_file(file_path: Path, data_type: str | None = None) -> np.ndarray:
    """Read data with the conventions in SciTS/process/infer_eval_utils.py."""
    del data_type
    path_str = str(file_path)
    if path_str.endswith(".csv"):
        rows = []
        with file_path.open() as raw_data_reader:
            for line in raw_data_reader.readlines():
                line = line.strip("\ufeff")
                rows.append(line.strip().split(",") if "," in line else line.strip())
        if "X" not in rows:
            return np.array(rows, dtype=np.float32)
        return np.array(rows)
    if path_str.endswith(".npy"):
        return np.load(file_path, allow_pickle=False)
    if path_str.endswith((".wav", ".flac")):
        try:
            import librosa  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - optional dependency path
            raise RuntimeError("Reading audio requires the optional 'librosa' dependency") from exc
        data, _ = librosa.core.load(file_path, mono=False)
        return data
    raise ValueError(f"Unsupported SciTS data type: {file_path.suffix}")


def _resolve_scits_metadata(root: Path) -> Path:
    metadata_path = root / "meta_data.jsonl"
    if metadata_path.exists():
        return metadata_path
    raise FileNotFoundError(f"Could not find meta_data.jsonl under {root}")


def _iter_scits_records(metadata_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with metadata_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {metadata_path}") from exc
    return records


@dataclass
class TarShardWriter:
    out_dir: Path
    dataset_name: str
    samples_per_shard: int

    def __post_init__(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._current_tar: tarfile.TarFile | None = None
        self._current_shard_name: str | None = None
        self._current_samples = 0
        self._shard_idx = 0
        self._start_idx = 0
        self._written_total = 0
        self._shards: list[dict[str, Any]] = []

    def _open_next_shard(self) -> None:
        self._current_shard_name = f"{self.dataset_name}-{self._shard_idx:06d}.tar"
        shard_path = self.out_dir / self._current_shard_name
        self._current_tar = tarfile.open(shard_path, "w")
        self._current_samples = 0
        self._start_idx = self._written_total

    def _close_current_shard(self) -> None:
        if self._current_tar is None or self._current_shard_name is None:
            return
        self._current_tar.close()
        shard_path = self.out_dir / self._current_shard_name
        self._shards.append(
            {
                "name": self._current_shard_name,
                "samples": self._current_samples,
                "size_bytes": shard_path.stat().st_size,
                "start_idx": self._start_idx,
            }
        )
        self._current_tar = None
        self._current_shard_name = None
        self._shard_idx += 1

    @staticmethod
    def _add_bytes(tar: tarfile.TarFile, name: str, payload: bytes) -> None:
        info = tarfile.TarInfo(name=name)
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))

    def write(
        self,
        key: str,
        ts_array: np.ndarray,
        text: str,
        metadata: dict[str, Any],
        gt_ts_array: np.ndarray | None = None,
    ) -> None:
        if self._current_tar is None:
            self._open_next_shard()

        assert self._current_tar is not None

        ts_buf = io.BytesIO()
        np.save(ts_buf, ts_array, allow_pickle=False)
        self._add_bytes(self._current_tar, f"{key}.ts.npy", ts_buf.getvalue())

        if gt_ts_array is not None:
            gt_ts_buf = io.BytesIO()
            np.save(gt_ts_buf, gt_ts_array, allow_pickle=False)
            self._add_bytes(self._current_tar, f"{key}.gt_ts.npy", gt_ts_buf.getvalue())

        self._add_bytes(self._current_tar, f"{key}.text", text.encode("utf-8"))
        self._add_bytes(
            self._current_tar,
            f"{key}.meta.json",
            json.dumps(metadata, ensure_ascii=True).encode("utf-8"),
        )

        self._current_samples += 1
        self._written_total += 1

        if self._current_samples >= self.samples_per_shard:
            self._close_current_shard()

    def finish(self) -> list[dict[str, Any]]:
        self._close_current_shard()
        return self._shards

    @property
    def total_written(self) -> int:
        return self._written_total


def convert_scits_to_webdataset(
    output_dir: str,
    dataset_name: str = "scits",
    scits_dir: str = "/Users/pemami/Workspace/Genesis/SciTS",
    samples_per_shard: int = 1000,
    val_ratio: float = 0.01,
    max_samples: int | None = None,
) -> dict[str, Any]:
    source_root = Path(scits_dir).expanduser().resolve()
    root = Path(output_dir)
    shards_dir = root / "shards"
    val_shards_dir = root / "val_shards"

    train_writer = TarShardWriter(shards_dir, dataset_name, samples_per_shard)
    val_writer = TarShardWriter(val_shards_dir, dataset_name, samples_per_shard)

    metadata_path = _resolve_scits_metadata(source_root)
    examples = _iter_scits_records(metadata_path)

    pbar = tqdm(total=len(examples), desc="Converting SciTS")

    # Check whether the output directories already exist and contains all the expected shards.
    # If so, we skip the conversion and return the manifest. Never resume-skip
    # a --max-samples run: expected_*_shards below is computed from the FULL
    # example count, so a smoke run against an already-fully-converted output
    # dir would see shard counts that "exceed expected" and silently return
    # the full manifest instead of the requested subset.
    if max_samples is None and shards_dir.exists() and val_shards_dir.exists():
        expected_train_shards = (len(examples) * (1 - val_ratio)) // samples_per_shard + 1
        expected_val_shards = (len(examples) * val_ratio) // samples_per_shard + 1

        # os.listdir() + filter, not Path.glob() — glob can hang on
        # dfuse/DAOS-mounted shard directories.
        actual_train_shards = sum(1 for f in os.listdir(shards_dir) if f.endswith(".tar"))
        actual_val_shards = sum(1 for f in os.listdir(val_shards_dir) if f.endswith(".tar"))

        if (
            actual_train_shards >= expected_train_shards
            and actual_val_shards >= expected_val_shards
        ):
            print(
                f"Found existing shards in {shards_dir} and {val_shards_dir}. "
                "Skipping conversion."
            )
            manifest_path = root / "manifest.json"
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                return manifest
            else:
                raise FileNotFoundError(
                    f"Manifest file not found at {manifest_path}. Cannot skip conversion."
                )
        else:
            print(
                f"Existing shards found but do not match expected counts. "
                f"Expected train shards: {expected_train_shards}, actual: {actual_train_shards}. "
                f"Expected val shards: {expected_val_shards}, actual: {actual_val_shards}. "
                "Proceeding with conversion."
            )

    # try/finally so a mid-conversion exception (bad record, disk full, ^C)
    # still closes the in-progress tar files via TarShardWriter.finish()
    # instead of leaving a shard with an unwritten tar end-of-archive marker.
    try:
        processed, skipped_missing_series, skipped_forecasting_series = _convert_examples(
            examples,
            source_root,
            metadata_path,
            dataset_name,
            val_ratio,
            max_samples,
            train_writer,
            val_writer,
            pbar,
        )
    finally:
        pbar.close()
        train_shards = train_writer.finish()
        val_shards = val_writer.finish()

    manifest = {
        "dataset": dataset_name,
        "source": str(source_root),
        "metadata_path": str(metadata_path.relative_to(source_root)),
        "total_samples": processed,
        "train_samples": train_writer.total_written,
        "val_samples": val_writer.total_written,
        "samples_per_shard": samples_per_shard,
        "num_train_shards": len(train_shards),
        "num_val_shards": len(val_shards),
        "num_shards": len(train_shards),
        "sample_ext": "ts.npy",
        "text_ext": "text",
        "meta_ext": "meta.json",
        "shards": train_shards,
        "val_shards": val_shards,
        "skipped_missing_series": skipped_missing_series,
        "skipped_forecasting_series": skipped_forecasting_series,
    }

    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\n=== SciTS Conversion Complete ===")
    print(f"SciTS source:      {source_root}")
    print(f"Output dir:        {root}")
    print(f"Total processed:   {processed:,}")
    print(f"Train samples:     {train_writer.total_written:,}")
    print(f"Val samples:       {val_writer.total_written:,}")
    print(f"Train shards:      {len(train_shards)}")
    print(f"Val shards:        {len(val_shards)}")
    print(f"Skipped (no ts):   {skipped_missing_series:,}")
    print(f"Skipped (forecasting):   {skipped_forecasting_series:,}")
    print(f"Manifest:          {manifest_path}")

    return manifest


def _convert_examples(
    examples: list[dict[str, Any]],
    source_root: Path,
    metadata_path: Path,
    dataset_name: str,
    val_ratio: float,
    max_samples: int | None,
    train_writer: TarShardWriter,
    val_writer: TarShardWriter,
    pbar: tqdm,
) -> tuple[int, int, int]:
    """Runs the per-record conversion loop, writing into `train_writer`/
    `val_writer`. Returns (processed, skipped_missing_series,
    skipped_forecasting_series)."""
    skipped_missing_series = 0
    skipped_missing_gt_series = 0
    skipped_forecasting_series = 0
    processed = 0

    for idx, example in enumerate(examples):
        if max_samples is not None and processed >= max_samples:
            break

        input_path_str = _extract_input_ts_path(example)
        if not input_path_str:
            skipped_missing_series += 1
            print(f"Missing {input_path_str}")
            pbar.update(1)
            continue

        series_path = source_root / input_path_str
        if not series_path.exists():
            skipped_missing_series += 1
            print(f"Missing {series_path}")
            pbar.update(1)
            continue

        data_type = _as_str(series_path.suffix.lstrip(".") or example.get("data_type") or "npy")

        try:
            series = _load_series_from_file(series_path, data_type)
        except Exception as exc:
            skipped_missing_series += 1
            # Previously swallowed the exception entirely — 43% of the
            # source dropped in one run with no way to tell why (bad dtype?
            # corrupt npy? missing soundfile?). Print the real cause.
            print(
                f"Failed to load series from {series_path} "
                f"with data_type {data_type}: {type(exc).__name__}: {exc}"
            )
            pbar.update(1)
            continue

        gt_series: np.ndarray | None = None
        gt_path_str = _extract_gt_ts_path(example)
        is_forecasting = False
        if gt_path_str:
            # print("Skipping forecasting...")
            # skipped_forecasting_series += 1
            # pbar.update(1)
            #continue
            gt_series_path = source_root / gt_path_str
            if not gt_series_path.exists():
                skipped_missing_gt_series += 1
                print(f"Missing ground-truth series {gt_series_path}")
                pbar.update(1)
                continue
            try:
                gt_series = _load_series_from_file(gt_series_path, data_type)
                is_forecasting = True
            except Exception:
                skipped_missing_gt_series += 1
                print(
                    f"Failed to load ground-truth series from {gt_series_path} "
                    f"with data_type {data_type}"
                )
                pbar.update(1)
                continue
        text = _compose_text(example, is_forecasting)

        key = _as_str(example.get("id")) or f"{processed:08d}"
        task_id = _normalize_task_id(example.get("task_id"))
        meta = {
            "source": str(source_root),
            "dataset": dataset_name,
            "metadata_path": str(metadata_path.relative_to(source_root)),
            "source_path": input_path_str,
            "original_index": idx,
            "shape": list(series.shape),
            "dtype": str(series.dtype),
            "task_id": task_id,
            "task_type": DATASET_TO_TASK.get(task_id, "unknown"),
            "id": key,
            "data_type": data_type,
            "gt_result": example.get("gt_result"),
            "gt_text": example.get("gt_text"),
            "gt_ts": example.get("gt_ts"),
            "meta_data": example.get("meta_data", {}),
        }

        if gt_series is not None:
            meta["gt_ts_shape"] = list(gt_series.shape)
            meta["gt_ts_dtype"] = str(gt_series.dtype)

        if _stable_val_bucket(example, val_ratio):
            val_writer.write(key, series, text, meta, gt_ts_array=gt_series)
        else:
            train_writer.write(key, series, text, meta, gt_ts_array=gt_series)

        processed += 1
        pbar.update(1)

    return processed, skipped_missing_series, skipped_forecasting_series


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert OpenTSLab/SciTS into PRISM-compatible WebDataset shards."
    )
    parser.add_argument(
        "--scits-dir",
        required=True,
        help="Path to a local SciTS checkout or staged SciTS directory",
    )
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--dataset-name", default="scits", help="Shard filename prefix")
    parser.add_argument(
        "--samples-per-shard",
        type=int,
        default=1000,
        help="Samples per tar shard",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.01,
        help="Validation ratio (deterministic hash split)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap for smoke conversion",
    )

    args = parser.parse_args()

    if args.samples_per_shard <= 0:
        raise ValueError("--samples-per-shard must be > 0")
    if args.val_ratio < 0 or args.val_ratio >= 1:
        raise ValueError("--val-ratio must satisfy 0 <= val-ratio < 1")

    convert_scits_to_webdataset(
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        scits_dir=args.scits_dir,
        samples_per_shard=args.samples_per_shard,
        val_ratio=args.val_ratio,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
