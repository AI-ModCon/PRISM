# AstroPrism: the 646-candidate pilot

Start with the **646 positional matches between released paired AION galaxies and AION-Search image captions**. Recover and validate these objects on Aurora, train image-to-language caption alignment, prepare grounded spectral captions in parallel, then extend to image + spectrum + language, instruction SFT, and a later verifier-based RL experiment.

The earlier goal of **10,000 training galaxies plus separate validation/test galaxies** remains a later expansion after this small pipeline works.

## Contents

| File | Purpose |
|---|---|
| [astroprism_training_plan.md](astroprism_training_plan.md) | AION-1/Qwen architecture, phases, splits, training settings and implementation work |
| [astroprism_synthetic_data_prompts.md](astroprism_synthetic_data_prompts.md) | Image verification, spectral facts/captions, joint captions, instruction conversations and validation prompts |
| [astroprism_benchmark_landscape.md](astroprism_benchmark_landscape.md) | AstroSpecQA, image benchmarks, paired/interleaved evaluation and RL evaluation boundaries |
| [astroprism_data.md](astroprism_data.md) | Dataset pins, roster checksum, field definitions and remote recovery procedure |
| [astroprism_data/overlap_candidates_646.csv.gz](astroprism_data/overlap_candidates_646.csv.gz) | Exact 646-row candidate roster with IDs, coordinates, separations and source shard/row locators |

This page and the four it links to are the whole package. It is a **plan**: no AION inference, no astronomy-specific PRISM integration, no caption synthesis, no SFT and no RL is implemented here. The analysis scripts, audit tables and diagnostic arrays behind the roster are not part of it.

## What is ready

The positional-match roster is committed here. Its 646 paired catalog IDs all map uniquely into the official tutorial raw-file index, with matching DESI IDs for these candidates. Physical association, raw observation quality and compatibility of each caption with AION's exact image crop still need checking. Caption text has not yet been extracted for this roster.

The proposal uses AION-1 Base's modality-specific codecs and shared frozen encoder, separate image/spectrum projectors, and Qwen3-4B. Nothing here establishes model readiness; the numbers below are proposals, not measurements.

For a 646-object accepted pool, the proposed split is approximately **516 train / 65 validation / 65 test**, grouped by physical source before training or instruction generation. Rejections or grouping can change those counts. The 646 rows are the complete candidate pool, not 646 training objects plus additional holdouts.

Remote data recovery, model loading, job submission and training are all subsequent work. [astroprism_data.md](astroprism_data.md) documents how to recover the captions and raw observations from the pinned public sources.
