# The 646 AION image–caption overlap candidates

`overlap_candidates_646.csv.gz` is the complete, byte-identical candidate roster from the metadata audit completed on 2026-09-16. It contains identifiers, sky positions, angular separations, and exact source shard/row locators. **It does not contain caption text, image or spectrum arrays, features, or encoder weights.** Recover these on Aurora before accepting examples for training.

- Compressed size: **39,219 bytes**; CSV data rows: **646**.
- SHA-256: `650935faa58953c9929e7379429bfb332a7486dab54544fedef64ca385abb3bc`.
- All 646 paired IDs and all 646 caption IDs are distinct. Every row has exactly one candidate in each direction within **1 arcsecond**; all three candidate/uniqueness flags are `True`.
- Median separation: **0.0191149 arcsec**; maximum: **0.2309893 arcsec**. All distances were independently checked with a spherical haversine calculation.
- The audit covered all **95,895** paired-feature objects and **164,911** Legacy DR10 caption records; it excluded the **110,702** HSC caption records. This is a positional candidate roster, not 646 already accepted training pairs.

## License and attribution

The roster is a derived table — identifiers, coordinates, separations and shard/row
locators — computed from the three pinned sources below. It carries no caption text, pixels,
spectra, features or weights.

| Source | License |
|---|---|
| [`EiffL/desi_x_ls_aion`](https://huggingface.co/datasets/EiffL/desi_x_ls_aion) | CC BY 4.0 |
| [`astronolan/galaxy-descriptions`](https://huggingface.co/datasets/astronolan/galaxy-descriptions) | MIT |
| [AION tutorial HDF5](https://users.flatironinstitute.org/~polymathic/data/provabgs_desi_ls.hdf5) (Polymathic AI / Flatiron Institute) | see upstream |

Both Hugging Face datasets permit redistribution of derived work with attribution, which this
section and the pinned revisions below provide. The underlying observations come from DESI and
the DESI Legacy Imaging Surveys; cite them directly in any downstream publication.

## Pinned sources and identifier meanings

| Source | Repository / location | Exact revision |
|---|---|---|
| Paired AION features | [`EiffL/desi_x_ls_aion`](https://huggingface.co/datasets/EiffL/desi_x_ls_aion/tree/57f7c7575260811ea076163bc743787d75b9aa49) | `57f7c7575260811ea076163bc743787d75b9aa49` |
| AION-Search image descriptions | [`astronolan/galaxy-descriptions`](https://huggingface.co/datasets/astronolan/galaxy-descriptions/tree/6890dc0c8fc793867c07a6ce4fce11e51c167d6e) | `6890dc0c8fc793867c07a6ce4fce11e51c167d6e` |
| Raw paired observations | [Official AION tutorial HDF5](https://users.flatironinstitute.org/~polymathic/data/provabgs_desi_ls.hdf5) | HTTP ETag `"6a62487d-1c2e3b0ad4"`; **121,034,705,620 bytes** |

For captions use **config `default`, split `default`**, and require `survey == "legacy_dr10"`. Do not substitute the `tiny` split. The recorded caption locators refer directly to the pinned Parquet files, not an index in a filtered Hugging Face dataset.

| Roster columns | Meaning |
|---|---|
| `paired_targetid` | String representation of the feature table's `object_id_provabgs`; use its PROVABGS namespace when recovering raw rows. |
| `caption_object_id` | Legacy imaging identifier from the caption table's `object_id`; it is a different namespace from the paired ID. |
| `paired_ra`, `paired_dec`, `caption_ra`, `caption_dec` | Source coordinates in degrees. |
| `separation_arcsec` | Spherical angular separation in arcseconds. |
| `paired_candidate_count`, `caption_candidate_count` | Number of neighbors within the matching radius in each direction. |
| `one_to_one_row_candidate`, `unique_source_ids`, `provisional_unambiguous_candidate` | Metadata matching flags; they do not certify physical identity, image quality, or caption correctness. |
| `paired_shard`, `caption_shard` | Paths relative to the corresponding pinned dataset repository. |
| `paired_row`, `caption_row` | **Zero-based row offset in the original, entire Parquet shard**, before any filtering. Row numbering continues across its row groups. |

Read identifiers as strings or exact integers, never floating-point numbers; these astronomical IDs can exceed JavaScript's exact integer range. The row numbering was verified against the audit's `enumerate` over the complete Parquet table.

## Recover the original captions on Aurora

For either source, construct its file URL as:

```text
https://huggingface.co/datasets/{repository}/resolve/{revision}/{shard_path}
```

The 646 caption candidates occupy just these two files, totaling **501,366,207 bytes** if downloaded in full:

| Caption shard | SHA-256 of full Parquet file, from the pinned repository's LFS metadata |
|---|---|
| `data/default-00008-of-00097.parquet` | `b23b3742eb905213299d320d808841a5d208f140482c351d85436465c94cfd46` |
| `data/default-00009-of-00097.parquet` | `fea00a8b73d16fda8883009a1c401f0f7af99a57e508f25cfec7e47a05a2b652` |

Minimal extraction recipe (future remote work; no downloads were executed to prepare this package):

```python
# Requires huggingface_hub and pyarrow in the remote environment.
import csv, gzip
from collections import defaultdict
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq

with gzip.open("docs/applications/astroprism_data/overlap_candidates_646.csv.gz", "rt") as f:
    roster = list(csv.DictReader(f))  # IDs remain strings.
groups = defaultdict(list)
for row in roster:
    groups[row["caption_shard"]].append(row)
for shard, candidates in groups.items():
    path = hf_hub_download(
        repo_id="astronolan/galaxy-descriptions", repo_type="dataset",
        revision="6890dc0c8fc793867c07a6ce4fce11e51c167d6e", filename=shard,
    )
    # Verify the downloaded file's SHA-256 against the table above.
    table = pq.read_table(path, columns=[
        "object_id", "survey", "ra", "dec", "caption", "summary",
    ])
    for row in candidates:
        item = table.slice(int(row["caption_row"]), 1).to_pylist()[0]
        assert str(item["object_id"]) == row["caption_object_id"]
        assert item["survey"] == "legacy_dr10"
        # Also check coordinates against the roster before accepting the row.
        # Save full caption, summary, original locator, and provenance together.
```

Use the full `caption` as the candidate target. Keep `summary` for provenance and inspection; it is not a substitute for validating the full description. These descriptions are synthetic weak supervision. Preserve the original text separately from any crop-validated revision. Grouping by shard avoids downloading the same file per object. Selective Parquet column/range reads can reduce transferred bytes further.

The paired-feature locators span **18 shards** totaling **4,705,846,446 bytes** if downloaded in full. They allow retrieval of the released `embedding_image` and `embedding_spectrum` and verification of `object_id_provabgs`, `ra_provabgs`, and `dec_provabgs`. They do **not** locate raw pixels or flux arrays; use the raw tutorial source below for encoder inputs. Downloading these feature shards is optional when recomputing features with the official encoders.

## Recover matching raw images and spectra

The official tutorial HDF5 contains **115,404** unique PROVABGS objects, including every ID in the paired-feature release. Its URL is not a Git revision: check the recorded ETag and length before reading, and stop to revalidate if either changes. Avoid downloading the entire 121 GB file solely to extract these candidates; use a seekable HTTP-range reader with HDF5, require HTTP 206 responses, and restrict reads to IDs and selected rows.

1. Read `provabgs_desi_edr_sv3/object_id` as exact IDs and construct `str(ID) -> raw row index`. Require uniqueness. Map each roster `paired_targetid` through this lookup; do not use the Parquet row offset as an HDF5 row offset.
2. Cross-check `desi/object_id`, `desi/ra`, `desi/dec`, the PROVABGS coordinates, and `legacysurvey/object_id`, `legacysurvey/ra`, `legacysurvey/dec`. Retain all original IDs and the raw source index.
3. Extract each selected row's `legacysurvey/image_array`, `image_band`, `image_ivar`, `image_mask`, `image_scale`, and `object_mask` under the `legacysurvey/` group, plus `desi/spectrum_flux`, `desi/spectrum_ivar`, `desi/spectrum_lambda`, and `desi/spectrum_mask`. Preserve units and masks. The image shape is `[4,160,160]`; the spectrum length is `7781`.
4. Read per-object band labels and use the official AION preprocessing; do not infer channel order from the array shape. Retain calibrated arrays, not only RGB previews. Apply and record the AION image crop and spectral preprocessing before caption acceptance.

The full raw file has **95 rows where DESI and PROVABGS IDs differ**, so blindly joining through `desi/object_id` is unsafe. A local cross-check of this particular roster found **646 unique raw PROVABGS matches, with DESI and PROVABGS IDs agreeing for all 646**. Recheck this on extraction; any discrepancy must be resolved from source coordinates/identities or excluded, never silently relabeled. Identifier agreement still does not establish image–spectrum physical correspondence.

## Acceptance and training handoff

Before any candidate becomes supervision, verify the physical galaxy match, imaging survey/release, spectroscopic aperture, and that every accepted caption claim is supported by the **exact student image crop**. Remove or revise claims about structures or companions outside the crop. Record raw-source checksums, preprocessing, rejected claims, quality flags, and accepted/rejected status per object. Retain the original roster unchanged and write a separate accepted manifest.

Freeze object-level train/validation/test assignments before deriving image captions, spectral captions, conversations, preferences, or reward examples. Every view, question, and generated derivative of one galaxy stays in the same split. The accepted count can be smaller than 646; 646 is the total candidate pool, not the training count. The training plan defines the subsequent image alignment, spectral captioning, joint model, instruction SFT, evaluation, and optional RL sequence.

This directory is self-contained provenance and source-location information. The local analysis scripts, reports, environments, caches, and 32-object diagnostic arrays are not required to interpret this roster and are intentionally outside the portable package.
