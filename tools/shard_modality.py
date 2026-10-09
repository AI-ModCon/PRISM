#!/usr/bin/env python3
"""Phase 3.3: one-shot shard creator for non-image PRISM modalities.

Reads a Hugging Face (or local) source dataset and writes 256–512 MB WebDataset
shards plus a `shards.json` manifest compatible with `MultiWebDataset`'s
discovery path. The launcher's existing DAOS staging picks up the manifest
unchanged.

Login-node tool — no GPU / MPI / Aurora dependencies.

Currently supported modalities:
    time_series    source: ChatTSRepo/ChatTS-Training-Dataset (ts_qa)
    graph          source: liupf/ChEBI-20-MM (graph_captioning)

Sample schema (one per WebDataset entry):
    <basename>.text            UTF-8 caption / instruction-output pair
    <basename>.<ext>           modality payload (numpy npy for time_series,
                               torch.save dict-of-tensors for graph)
    <basename>.meta.json       small JSON with sample id + provenance

Example:
    python tools/shard_modality.py --modality time_series \\
        --source hf --hf-id ChatTSRepo/ChatTS-Training-Dataset \\
        --hf-config align_256 --hf-split train \\
        --out-uri /flare/ModCon/<you>/shards/ts_qa --max-samples 105000
"""

from __future__ import annotations

import argparse
import io
import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

logger = logging.getLogger("shard_modality")


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )


# -----------------------------------------------------------------------------
# Modality-specific iterators. Each yields (basename, text, payload_bytes, ext, meta).
# Adding a new modality means writing one new iterator + registering it in the
# `_ITERATORS` table; everything else (shard writer, manifest) is generic.
# -----------------------------------------------------------------------------

def _iter_time_series_hf(hf_id: str, split: str, config: str | None, max_samples: int | None) -> Iterator[tuple]:
    import numpy as np
    from datasets import load_dataset

    kwargs: dict[str, Any] = {"split": split, "streaming": True}
    if config:
        kwargs["name"] = config
    ds = load_dataset(hf_id, **kwargs)
    n = 0
    for i, ex in enumerate(ds):
        # ChatTS schema: input / output / timeseries (list of floats)
        ts_raw = ex.get("timeseries") or ex.get("series") or ex.get("ts")
        if ts_raw is None:
            continue
        arr = np.asarray(ts_raw, dtype="float32").reshape(-1)
        buf = io.BytesIO()
        np.save(buf, arr, allow_pickle=False)
        text = f"{ex.get('input', '')}\n{ex.get('output', '')}".strip()
        basename = f"{i:09d}"
        meta = {"id": basename, "src": hf_id, "len": int(arr.shape[0])}
        yield basename, text, buf.getvalue(), "ts.npy", meta
        n += 1
        if max_samples and n >= max_samples:
            return


def _iter_graph_hf(hf_id: str, split: str, config: str | None, max_samples: int | None) -> Iterator[tuple]:
    import torch
    from datasets import load_dataset

    try:
        from rdkit import Chem
    except ImportError as exc:
        raise SystemExit("graph sharding requires rdkit (pip install rdkit-pypi)") from exc

    kwargs: dict[str, Any] = {"split": split, "streaming": True}
    if config:
        kwargs["name"] = config
    ds = load_dataset(hf_id, **kwargs)

    n = 0
    for i, ex in enumerate(ds):
        smiles = ex.get("SMILES") or ex.get("smiles")
        text = ex.get("description") or ex.get("caption") or ""
        if not smiles or not text:
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        num_nodes = mol.GetNumAtoms()
        if num_nodes == 0:
            continue
        # 1-hot atomic number as node feature (cheap; encoder embeds further).
        x = torch.zeros((num_nodes, 1), dtype=torch.float32)
        for atom_idx, atom in enumerate(mol.GetAtoms()):
            x[atom_idx, 0] = atom.GetAtomicNum()
        edges = []
        for bond in mol.GetBonds():
            a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edges.append([a, b])
            edges.append([b, a])
        edge_index = (
            torch.tensor(edges, dtype=torch.long).t().contiguous()
            if edges
            else torch.empty((2, 0), dtype=torch.long)
        )
        graph_blob = {
            "x": x,
            "edge_index": edge_index,
            "num_nodes": torch.tensor(num_nodes, dtype=torch.long),
        }
        buf = io.BytesIO()
        torch.save(graph_blob, buf)
        basename = f"{i:09d}"
        meta = {"id": basename, "src": hf_id, "smiles": smiles, "num_nodes": num_nodes}
        yield basename, text, buf.getvalue(), "graph.pt", meta
        n += 1
        if max_samples and n >= max_samples:
            return


_ITERATORS = {
    "time_series": _iter_time_series_hf,
    "graph": _iter_graph_hf,
}


# -----------------------------------------------------------------------------
# Shard writer + manifest
# -----------------------------------------------------------------------------

def _open_writer(out_dir: Path, shard_pattern: str, shard_maxbytes: int):
    import webdataset as wds

    return wds.ShardWriter(
        str(out_dir / shard_pattern),
        maxcount=10_000_000,  # cap by bytes, not count
        maxsize=shard_maxbytes,
    )


def write_shards(
    iterator: Iterator[tuple],
    out_dir: Path,
    modality: str,
    shard_maxbytes: int,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = f"{modality}-%06d.tar"
    writer = _open_writer(out_dir, pattern, shard_maxbytes)

    sample_count = 0
    payload_ext: str | None = None
    try:
        for basename, text, payload, ext, meta in iterator:
            sample = {
                "__key__": basename,
                "text": text,
                ext: payload,
                "meta.json": json.dumps(meta),
            }
            writer.write(sample)
            payload_ext = ext  # actual WebDataset sample-key suffix
            sample_count += 1
            if sample_count % 1000 == 0:
                logger.info(f"  wrote {sample_count} samples …")
    finally:
        writer.close()

    # Discover what we just wrote — `webdataset.ShardWriter` doesn't expose
    # the final shard list directly, so glob the directory.
    shard_files = sorted(out_dir.glob(f"{modality}-*.tar"))
    manifest = {
        "modality": modality,
        "num_samples": sample_count,
        "num_shards": len(shard_files),
        "shards": [str(p.name) for p in shard_files],
        # The actual WebDataset sample-key suffix the loader's `to_tuple` must
        # consume (e.g. "ts.npy", "graph.pt"). MultiWebDataset's per-modality
        # pipeline uses this to validate that its hardcoded tuple spec matches
        # what was written.
        "sample_ext": payload_ext,
    }
    manifest_path = out_dir / "shards.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info(
        f"wrote {sample_count} samples → {len(shard_files)} shards in {out_dir}; "
        f"manifest at {manifest_path}"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--modality", required=True, choices=sorted(_ITERATORS.keys()))
    p.add_argument(
        "--source", default="hf", choices=["hf"],
        help="Source backend (only 'hf' supported today; 'local' tbd).",
    )
    p.add_argument("--hf-id", required=True, help="HuggingFace dataset id, e.g. liupf/ChEBI-20-MM")
    p.add_argument("--hf-split", default="train")
    p.add_argument("--hf-config", default=None, help="HF config / subset name (optional)")
    p.add_argument("--out-uri", required=True, help="Output directory (local path or daos://...)")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--shard-mb", type=int, default=384, help="Approx shard size in MB (default 384)")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    _setup_logging(args.log_level)

    if args.out_uri.startswith("daos://"):
        # The launcher mounts DAOS at runtime; for the login-node tool we
        # write to the dfuse mount path. Users pass the dfuse path directly.
        raise SystemExit(
            "out-uri=daos://... not supported on login node; pass the dfuse mount path "
            "(e.g. /flare/ModCon/<you>/shards/<modality>) and the launcher will see it."
        )

    out_dir = Path(args.out_uri).expanduser().resolve()
    iterator = _ITERATORS[args.modality](
        args.hf_id, args.hf_split, args.hf_config, args.max_samples
    )
    write_shards(iterator, out_dir, args.modality, args.shard_mb * 1024 * 1024)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
