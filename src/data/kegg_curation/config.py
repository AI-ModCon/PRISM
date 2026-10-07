"""Single source of truth for the KEGG curation pipeline.

BioReason's original notebooks regenerated the reference/variant sequence
window 3 separate times with 2 different values (1000nt in KEGG_Data_1,
2000nt in KEGG_Data_2's CONFIG dict, 2000nt again in KEGG_Data_3's final
pass) with no single place reconciling them. This module exists so that
never happens here — every stage imports `CONFIG` from this file instead of
hardcoding its own copy.

See docs/applications/kegg_curation_pipeline.md for the full pipeline design.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

#: Root for all pipeline outputs (checkpoints + downloaded genome data) —
#: kept outside the repo checkout since this includes multi-GB downloads
#: (reference genome FASTA) that must never land in git.
DATASET_ROOT = Path(
    os.environ.get(
        "KEGG_CURATION_DATA_ROOT",
        # Placeholder: set KEGG_CURATION_DATA_ROOT (or PRISM_DATA_ROOT) to a
        # writable root outside the checkout. Deliberately not a real path.
        "/lus/<filesystem>/projects/<project>/<user>/dataset/kegg_curation",
    )
)


@dataclass
class KeggCurationConfig:
    # --- Paths ---
    checkpoint_dir: Path = field(default_factory=lambda: DATASET_ROOT / "checkpoints")
    # Where the subsetted reference genome FASTA lives.
    genome_dir: Path = field(default_factory=lambda: DATASET_ROOT / "genome")

    # --- KEGG retrieval ---
    kegg_rest_base: str = "https://rest.kegg.jp"
    kegg_species: str = "hsa"  # human

    # --- Reference genome ---
    # GRCh38, RefSeq assembly — same accession the original notebooks used.
    genome_assembly_accession: str = "GCF_000001405.26"

    # --- Sequence window ---
    # Nucleotides extracted on EACH side of a variant. The original's final
    # published dataset almost certainly used 2000 (KEGG_Data_3's final
    # pass, variable named "KEGG_2000") — kept as the default here.
    sequence_window: int = 2000

    # --- Stage 03: cross-DB ID resolution ---
    # Unlike BioReason's original pipeline (which parsed cross-DB IDs out of
    # free NETWORK-entry text via a one-off ChatGPT paste — no surviving
    # code), the individual KEGG "hsa_var:<token>" entries carry a
    # structured VARIATION field with direct "Source: ID [ID2 ...]" lines
    # (e.g. "ClinVar: 12582", "dbSNP: rs121913529") for the large majority
    # of tokens — confirmed by direct inspection, not documented anywhere.
    # Stage 03 therefore parses this field directly as the primary path
    # (zero cost, high precision) and only falls back to an LLM resolver
    # for tokens whose hsa_var: entry is missing (404) or lacks a VARIATION
    # field entirely.
    id_resolver_llm_fallback: bool = True

    # --- Stage 04: NCBI Entrez (ClinVar / dbSNP / OMIM) ---
    # NCBI usage policy requires a contact email for Entrez API use; an API
    # key is optional but raises the rate limit from 3 req/sec to 10
    # req/sec. This is a SEPARATE credential from ANTHROPIC_API_KEY/HF_TOKEN
    # — register free at https://www.ncbi.nlm.nih.gov/account/ to get one.
    ncbi_entrez_email: str | None = field(
        default_factory=lambda: os.environ.get("NCBI_ENTREZ_EMAIL")
    )
    ncbi_api_key: str | None = field(
        default_factory=lambda: os.environ.get("NCBI_API_KEY")
    )

    # --- Stage 04: COSMIC (optional) ---
    # Path to a user-supplied, license-gated COSMIC "CompleteTargetedScreens
    # Mutant" TSV export (GRCh38 build). If None, COSM-sourced variants are
    # skipped rather than erroring — COSMIC access is licensed per-user and
    # can't be fetched programmatically.
    cosmic_tsv_path: str | None = None

    # --- Stage 09: reasoning generation backend ---
    # "claude" (real generation) or "mock" (zero-cost dry run — returns
    # synthetic placeholder question/answer/reasoning so the rest of the
    # pipeline can be validated without API cost). Defaults to "mock";
    # override via REASONING_BACKEND=claude for a real run.
    reasoning_backend: str = field(
        default_factory=lambda: os.environ.get("REASONING_BACKEND", "mock")
    )
    # Current Claude model — intentionally NOT pinned to the original's
    # dated claude-3-7-sonnet-20250219 snapshot, which will eventually be
    # deprecated. Override via ANTHROPIC_MODEL if you need a specific one.
    anthropic_model: str = field(
        default_factory=lambda: os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")
    )
    anthropic_api_key: str | None = field(
        default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY")
    )
    reasoning_max_tokens: int = 6000
    reasoning_temperature: float = 0.2

    # --- Stage 10: disease-name standardization ---
    # No-op by default. The original hand-curated a ~90-entry mapping
    # specific to its own variant set; populate this after inspecting the
    # actual distinct disease labels your run produces, if needed.
    disease_name_overrides: dict[str, str] = field(default_factory=dict)

    # --- Run scale ---
    # None = process every variant found. Set to a small number for a
    # smoke-scale dry run before committing to a full/expensive run.
    max_variants: int | None = None

    def __post_init__(self) -> None:
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.genome_dir = Path(self.genome_dir)

    def ensure_dirs(self) -> None:
        """Creates checkpoint_dir. Called by each stage's main(), NOT at
        import time: `CONFIG` below is module-level, so mkdir-ing inside
        __post_init__ ran on every import of every stage module. Combined
        with DATASET_ROOT defaulting to a path under one specific user's
        project directory, that made `import`ing any stage raise
        PermissionError for every other user (including at test-collection
        time). Creating the output tree is a side effect of *running* a
        stage, not of importing it.
        """
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def checkpoint(self, name: str) -> Path:
        """Path to a named checkpoint file inside checkpoint_dir."""
        return self.checkpoint_dir / name


CONFIG = KeggCurationConfig()
