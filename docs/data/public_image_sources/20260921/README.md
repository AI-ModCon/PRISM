# Public image-caption data collection: manifests and provenance

The complete survey is [Public image-caption data collection for PRISM](../../../reports/2026-09-21-public-image-caption-data-survey.md), audited 2026-09-21. It covers the selected three-source pool, the wider nine-source inventory, count and storage estimates, access limitations, caption-conditioned training, and validation required before further training.

This directory contains the compact, tracked evidence supporting that survey:

- [Selected subset](selected_subset.json): candidate counts, exact download bytes, and assumptions.
- [Download manifest](download_manifest.jsonl): 73 revision-pinned source objects, including 60 image archives and 13 annotation/prompt files, for possible future acquisition.
- [Small-source summary](small_sources.summary.json): DOCCI, ShareGPT4V and BLIP3-o.
- [Dense-source summary](dense_sources.summary.json): DenseFusion, ALLaVA and JourneyDB.
- [Large-source summary](large_sources.summary.json): Recap, SAM-LLaVA and LAION.
- [DenseFusion verification](densefusion4v_content_verification.json): parsed caption/URL schema, counts and per-source identifier uniqueness.
- [Source-bundle checksums](source_bundle_checksums.json): hashes for the complete external metadata bundle, rather than just these tracked summaries.
- [Delivery receipt](delivery-receipt.json): successful Aurora transfer and hash verification.

Only metadata/manifests were collected. The full 134-file metadata bundle totals 149,693,911 bytes and is stored locally at `outputs/public_image_data_audit/20260921/` and on Aurora at:

```text
/lus/flare/projects/ModCon/sandeep/prism-public-image-data-20260921/metadata/
```

Run `python3 verify_bundle.py` inside the full bundle to verify its file hashes. Caption rows and raw source snapshots remain in that bundle outside Git. No image payloads or image archives were downloaded, and this audit did not launch expanded connector training.
