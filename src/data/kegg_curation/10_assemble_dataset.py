"""Stage 10 — Assemble the final HF Dataset.

Recreates BioReason's KEGG_Data_3.ipynb steps 27-29 (final tabular
assembly), matching wanglab/kegg's exact published schema so this is a
drop-in for the existing consumer in src/data/multimodal.py's
_process_dna_bioreason (mirroring BioReason/bioreason/dataset/kegg.py's
_format_kegg):

    question, answer, reasoning, reference_sequence, variant_sequence

Disease-name standardization (CONFIG.disease_name_overrides) is applied to
`answer` here if populated — starts as a no-op passthrough by design (the
original hand-curated a ~90-entry mapping specific to its own variant set;
this pipeline doesn't try to pre-populate an equivalent, since it's a fresh
curation run over a different, freshly-fetched variant set — see
docs/applications/kegg_curation_pipeline.md).

Saves the assembled dataset locally as a HF `datasets.Dataset`
(parquet + dataset_info.json). Matching the original notebook's own
behavior, this stage does NOT call push_to_hub — that remains a manual,
deliberate step (see the printed instructions at the end of main()).

Input:  checkpoints/variants_with_reasoning.tsv   (stage 09)
        checkpoints/reasoning/{Var_ID}.json        (stage 09)
        checkpoints/sequences/{Var_ID}.json        (stage 08)
Output: checkpoints/final_dataset/                 HF Dataset (parquet +
        dataset_info.json), single "train" split

Usage:
    python -m src.data.kegg_curation.10_assemble_dataset
"""

import csv
import json
import logging

from src.data.kegg_curation.config import CONFIG

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def standardize_disease_name(answer: str) -> str:
    """Applies CONFIG.disease_name_overrides if populated; no-op otherwise."""
    return CONFIG.disease_name_overrides.get(answer, answer)


def main() -> None:
    CONFIG.ensure_dirs()
    in_path = CONFIG.checkpoint("variants_with_reasoning.tsv")
    with open(in_path, newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    logger.info(f"Loaded {len(rows)} variants from {in_path}.")

    seq_dir = CONFIG.checkpoint("sequences")
    reasoning_dir = CONFIG.checkpoint("reasoning")

    records: list[dict] = []
    n_skipped = 0

    for row in rows:
        if row.get("ReasoningStatus") != "ok":
            n_skipped += 1
            continue

        var_id = row["Var_ID"]
        seq_path = seq_dir / f"{var_id}.json"
        reasoning_path = reasoning_dir / f"{var_id}.json"
        if not seq_path.exists() or not reasoning_path.exists():
            logger.warning(f"{var_id}: missing sequence or reasoning file — skipping.")
            n_skipped += 1
            continue

        sequences = json.loads(seq_path.read_text())
        reasoning = json.loads(reasoning_path.read_text())

        # Stage 09 only rejects a backend response that carries an "error"
        # key, which reasoning.py sets when the text could not be parsed as
        # JSON at all. A response that IS valid JSON but omits one of the
        # three required keys passes that gate and gets written verbatim,
        # so indexing it here raised KeyError — aborting the final stage of
        # a multi-hour, real-API-cost run over one malformed row. Skip and
        # count the row instead; the checkpoint stays on disk for
        # inspection or a targeted re-run.
        missing = [k for k in ("question", "answer", "reasoning") if k not in reasoning]
        if missing:
            logger.warning(
                f"{var_id}: reasoning JSON is missing {missing} — skipping. "
                f"Inspect {reasoning_path} and re-run stage 09 for this variant."
            )
            n_skipped += 1
            continue

        records.append({
            "question": reasoning["question"],
            "answer": standardize_disease_name(reasoning["answer"]),
            "reasoning": reasoning["reasoning"],
            "reference_sequence": sequences["reference_sequence"],
            "variant_sequence": sequences["variant_sequence"],
        })

    logger.info(f"Assembled {len(records)} records ({n_skipped} skipped: no reasoning/sequence).")

    if not records:
        logger.error("No records to assemble — nothing to write. Check earlier stages' output.")
        return

    from datasets import Dataset, DatasetDict

    dataset = Dataset.from_list(records)
    dataset_dict = DatasetDict({"train": dataset})

    out_path = CONFIG.checkpoint("final_dataset")
    dataset_dict.save_to_disk(str(out_path))
    logger.info(f"Saved dataset ({len(records)} rows, 'train' split) to {out_path}")

    print(
        "\n"
        "Dataset assembled and saved locally. This pipeline does NOT push to\n"
        "the Hugging Face Hub automatically — to publish, load it back and\n"
        "call push_to_hub yourself once you're satisfied with the content:\n\n"
        "    from datasets import load_from_disk\n"
        f"    ds = load_from_disk('{out_path}')\n"
        "    ds.push_to_hub('your-username/your-dataset-name')\n"
    )


if __name__ == "__main__":
    main()
