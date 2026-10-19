# AstroPrism: image, spectrum and language pilot with 646 candidates

**Last updated**: 2026-09-16. The plan below is written against PRISM's existing Zone-A training path; it proposes work, it does not describe shipped code.

## 1. Goal and sequence

Prototype a model that accepts an image, a spectrum or their matched pair and produces grounded descriptions and answers. Begin with the **646 candidate matches** between released paired AION galaxies and AION-Search image captions, using **AION-1 Base** and **Qwen3-4B**.

```text
Recover candidates -> verify identity, crop and captions -> freeze source splits
                                  |
               +------------------+-------------------+
               |                                      |
   Stage 1A: image-caption alignment       Measure spectral facts and generate
   with AION image tokens                 verified captions for the same objects
               |                                      |
               +------------------+-------------------+
                                  |
   Stage 1B: spectrum-projector warmup, then mixed image / spectrum / joint captions
                                  |
   Stage 2: verified instruction conversations -> projectors + Qwen LoRA SFT
                                  |
   Stage 3: small verifier-grounded RL experiment -> compare against SFT
```

Stages 1A/1B are **caption alignment**, Stage 2 is **instruction SFT**, and Stage 3 is a later **RL extension**. These are our proposed stages. [AION-Search](https://arxiv.org/abs/2512.11982) supplies candidate image descriptions but trains contrastive retrieval. [AstroSpecLM](https://arxiv.org/abs/2609.07102) motivates evidence-first spectral supervision; its main model uses SpecMAE and SFT, and its caption-alignment ablation trains projector plus LoRA. Our AION-based, frozen-Qwen caption stage is a deliberate adaptation.

**Status:** candidate metadata and this plan exist. Caption text has not been extracted for this roster. The accepted corpus/splits, AION/Qwen integration, generated instructions and training runs remain future work. Local data audits and offline tests did not run AION inference.

## 2. Recover and accept the candidate observations

Use the [646-row roster](astroprism_data/overlap_candidates_646.csv.gz) and [pinned source/recovery instructions](astroprism_data.md). It joins [`EiffL/desi_x_ls_aion`](https://huggingface.co/datasets/EiffL/desi_x_ls_aion) paired features to `legacy_dr10` records in [`astronolan/galaxy-descriptions`](https://huggingface.co/datasets/astronolan/galaxy-descriptions).

All 646 paired IDs and caption IDs are unique, with one-to-one positional candidates within one arcsecond; the largest separation is 0.231 arcseconds. All 646 also map uniquely into the official tutorial raw-file index with matching DESI/PROVABGS IDs. This resolves the larger raw catalog's identifier-discrepancy concern for this roster, but still requires physical identity and input-quality verification.

Recover raw observations from the [official tutorial HDF5 file](https://users.flatironinstitute.org/~polymathic/data/provabgs_desi_ls.hdf5), following the [AION tutorial](https://github.com/PolymathicAI/AION/blob/main/notebooks/Tutorial.ipynb). Select by verified IDs; the first 646 rows are not this roster. Public full-token features may accelerate training after checking extraction provenance and numerical agreement with representative raw-input recomputation.

Where additional raw fields or reference measurements are needed, use release-compatible sources:

| Resource | Role |
|---|---|
| [MMU Legacy DR10 South](https://huggingface.co/datasets/hugging-science/mmu_legacysurvey_dr10_south_21) | Calibrated multiband images, uncertainties, masks and imaging metadata |
| [MMU DESI EDR SV3](https://huggingface.co/datasets/UniverseTBD/mmu_desi_edr_sv3) | Observed spectra, wavelength, inverse variance, masks and reduction metadata |
| [MMU DESI PROVABGS](https://huggingface.co/datasets/UniverseTBD/mmu_desi_provabgs) | Catalog identity and separately marked model-derived physical-property references |
| [FastSpecFit EDR](https://data.desi.lbl.gov/doc/releases/edr/vac/fastspecfit/) | Compatible continuum/line measurements, with fit quality and provenance |
| [Galaxy Zoo](https://data.galaxyzoo.org/) | Independent morphology references where matched human classifications exist |

Do not mix EDR and DR1 fits without reconciling releases. Some fitted properties use photometry jointly with spectra; preserve their actual evidence requirements. Catalog properties are reference targets, not automatically student inputs or directly visible facts.

Acceptance procedure:

1. Read each pinned shard/row and verify its ID and coordinates. Keep large identifiers as strings. Verify raw image/spectrum association, blends, nearby companions and fiber aperture.
2. Recover calibrated four-band images and observed-frame spectra with wavelength, inverse variance and mask. Check finite values, usable coverage and metadata.
3. Verify each original caption against a rendering of the **exact AION image crop**. Revise unsupported/out-of-crop claims, retain original text in provenance, and label `released_reused`, `released_revised` or `rejected`.
4. Group repeat observations, aliases and overlapping sources before assigning splits. Record every exclusion; do not silently add unrelated objects to maintain 646.

Use the full caption as a weak reference, not an unchecked summary or expert label. This overlap is a selected galaxy sample; it does not establish broad performance on other surveys, stars or quasars.

## 3. Split once before fitting or instruction generation

If all candidates survive, target **516 train / 65 validation / 65 test galaxies**, with seed 42. Treat this as an approximately 80/10/10 allocation. Keep physical-source and overlapping-cutout groups intact even when exact totals change. Report the accepted counts after rejection/grouping.

- A source keeps its split through image alignment, spectrum alignment, SFT and RL. All views, paraphrases, conversations and rollouts inherit it.
- Captions can be generated for every eligible object, but held-out captions/facts stay outside training, reward fitting and training-teacher prompt tuning. Validation sources may support inference-prompt/scorer selection; test sources may not support any tuning. Training on all 646 images and only later holding out some spectra would leak those objects.
- Use training-only sources for interactive debugging, training-teacher prompt revision, tiny overfit checks and demonstrations. Use validation for checkpoint, inference-prompt and scorer selection. Independently curate and seal test references before selection.
- Stratify approximately over brightness, redshift, spectral S/N and reliable coarse morphology. Leakage control takes precedence over exact balance.
- Describe this as evaluation on objects held out from adaptation. Disjointness from AION/Qwen foundation-model pretraining is not established.

The accepted manifest must bind `source_group_id`, split, survey/release/IDs, raw and feature locators/checksums, crop signature, encoder/codec revisions, caption provenance, permitted fact references and quality flags. The committed candidate roster is the input to this process, not an accepted split manifest.

## 4. AION-1 encoders and Qwen interface

AION has modality-specific codecs followed by **one shared pretrained transformer encoder**. Use separate logical image/spectrum passes with shared frozen weights, allowing either modality independently. Follow the [official code](https://github.com/PolymathicAI/AION) and [Base checkpoint](https://huggingface.co/polymathic-ai/aion-base).

| Component | Image branch | Spectrum branch |
|---|---|---|
| Raw input | Calibrated `g,r,i,z`, typically `4 × 160 × 160` | Observed wavelength, flux, inverse variance, mask |
| Official codec | `LegacySurveyImage`, `96 × 96` center crop | `DESISpectrum`, official normalization/padding |
| Full Base feature tensor | `B × 576 × 768` | `B × 273 × 768` |
| Trainable projector | `Linear(768,2560) → GELU → Linear(2560,2560)` | Separate projector with identical dimensions |
| Qwen observation embeddings | `B × 576 × 2560` | `B × 273 × 2560` |

The released feature card specifies the token shapes; Qwen3-4B's [configuration](https://huggingface.co/Qwen/Qwen3-4B/blob/main/config.json) specifies width 2560. Verify shapes on the pinned real checkpoint. Start with full tokens and no resampler, trainable modality type embeddings, and an explicit projector-normalization setting of `none`; monitor embedding norms.

- Pin AION package, model, codecs and Qwen/tokenizer revisions. Local AION metadata recorded checkpoint revision `40541618104bab0fa85c8af68daeb867a720bb8c`; verify existing Aurora weights against that or document the chosen alternative. Codec pins must be explicit, not assumed from the main model loader.
- Request 576/273 tokens explicitly from `AION.encode`; its default budget can truncate output. The spectrum sequence includes a normalization token plus spectral codes, not 273 raw wavelength bins. Preserve token positions/masks and deterministic ordering.
- Keep AION codecs and encoder in evaluation mode. Use official preprocessing instead of generic time-series normalization. Check that narrow-line claims survive the lossy codec before making them caption targets.
- Cache full frozen AION tokens only after provenance and raw/cache agreement checks. Trainable projector outputs cannot be cached across updates. The nominal float16 cache for all 646 paired objects is about **0.84 GB decimal**, excluding metadata/raw inputs.

The first joint baseline supplies `[image embeddings][spectrum embeddings][question/history][assistant answer]`. Its **849 observation positions** leave 1,199 positions for text/delimiters in a 2,048-position budget. Reject overflow instead of silently truncating observations or targets.

Label observation positions, system/user text and padding with `-100`; label only assistant answer tokens for next-token prediction. The model predicts text, not the input observation-code IDs. Use the pinned Qwen non-thinking chat template and verify every multi-turn assistant span. This all-upfront prefix baseline is distinct from a later conversation in which new observations arrive between turns; that needs explicit causal/source-binding checks.

## 5. Stage 1A: image-caption alignment

Train the image projector and image type embedding using accepted training image/caption pairs. Freeze AION and Qwen. Optimize assistant-caption cross entropy and validate grounding and image sensitivity, not only wording similarity. Save the selected image-only checkpoint as a control.

Start with a representative training-only encoder check and a tiny overfit check to establish real gradients, masking and save/reload. These are future implementation checks. At 516 accepted training objects, the nominal corpus is **516 image-caption records**; paraphrases do not add independent galaxies.

Meanwhile, measure spectral facts and generate captions in a separate process using the same source splits. This parallel work requires separately provisioned resources; do not run teacher or training inference on an Aurora login node.

## 6. Parallel spectral caption preparation

Use the [synthetic-data prompts](astroprism_synthetic_data_prompts.md). First distill a compact fact record from measured spectra and compatible catalogs: wavelength coverage, quality, continuum/breaks, reliable line identifications, strengths/EWs and uncertainties. Every fact carries provenance, units, uncertainty, detection/coverage status and permitted student modalities.

Catalog-redshift rest-frame measurements may help the teacher, but the student receives an observed-frame spectrum. Never leak true redshift through student preprocessing or prompt text. Distinguish nondetection, masked/missing coverage and an unassessed feature. Keep model-derived physical properties separately marked.

Generate one concise spectral caption per accepted source, with a separate claim-to-fact audit. Verify numerical targets and recoverability from the processed AION input. Generate joint captions from verified image and spectral facts, attribute evidence to each observation, and respect whole-galaxy versus fiber-aperture limitations. Coexistence of two features does not establish a causal explanation.

Generation may cover the accepted roster, but outputs stay separated by split. Independently review evaluation references; the teacher agreeing with its own prose is insufficient. Record model/prompt versions, generation settings and accepted/revised/rejected counts.

## 7. Stage 1B: spectrum and joint caption alignment

Initialize from the selected Stage 1A checkpoint and add a fresh spectrum projector.

1. **Spectrum warmup:** freeze the image projector/type embedding, AION and Qwen; train only the spectrum projector/type embedding on training spectral captions.
2. **Mixed alignment:** train both projectors/type embeddings using image-only, spectrum-only and paired captions while AION/Qwen remain frozen. Replay image-only examples to preserve their grounding.
3. Select using per-modality validation evidence accuracy. Compare image-only performance with the saved Stage 1A checkpoint.

With 516 accepted training sources, mixed alignment has a nominal **1,548 caption records** (516 for each modality condition), subject to caption acceptance. Report actual records and objects separately. Never fabricate a missing observation or treat unavailable facts as zero measurements.

## 8. Stage 2: instruction generation and SFT

Build conversations from validated fact records, not raw unfiltered catalogs or caption paraphrases alone:

- **Image-only:** morphology, visible structures, quality and limitations supported by the image.
- **Spectrum-only:** line/continuum evidence, supported redshift/numerical tasks and uncertainty/coverage questions.
- **Joint:** complementary evidence, modality attribution, aperture limitations and tasks that require both observations.

Start with two accepted conversations per training source per modality condition: nominally **3,096 conversations** for 516 sources. Most are single-turn; approximately 20% can contain a short evidence/uncertainty follow-up. Reject unsupported tasks rather than filling a quota. Count conversations and assistant turns separately.

Initialize from mixed alignment. Freeze AION and Qwen base weights; train projectors/type embeddings plus Qwen LoRA. Verify trainable names and optimizer membership. Use explicit assistant-only labels for every turn and homogeneous modality batches initially. All source splits remain fixed.

### Proposed starting settings

These settings are initial proposals, not measured optima. This small corpus primarily tests integration and grounding.

| Setting | Caption alignment | Instruction SFT |
|---|---|---|
| Projector LR | `1e-4` initially | `5e-5` |
| Qwen adaptation | Frozen | LoRA rank 16, alpha 32, dropout 0.05, q/k/v/o projections |
| LoRA LR | — | `2e-5` |
| Optimizer | AdamW, weight decay 0.01, clip 1.0 | Same |
| Effective batch | 16 records via accumulation | 16 conversations via accumulation |
| Initial duration | Evaluate each epoch; initially 1–3 per subphase | Initially 1–2 epochs |
| Precision/schedule | bf16 where supported; ~5% warmup then cosine | Same |
| Full sequence budget | 2,048 including observation tokens | Same; reject overflow |

Nominal updates per epoch are about 33 for image alignment, 33 for spectrum warmup, 97 for mixed captions, and 194 for SFT. Actual counts depend on accepted records and sampling. Check validation at least each epoch and reduce training when held-out grounding regresses. Establish real XPU memory/throughput before estimating duration.

## 9. Stage 3: verifier-grounded RL

Treat RL as a separately evaluated extension initialized from SFT. With roughly 516 training galaxies, memorization and unreliable rewards are substantial concerns. Build and calibrate deterministic verifiers before generating optimization rollouts.

Start with objectively scored tasks: independently supported morphology labels, line identity/coverage decisions, modality attribution, or numeric answers with declared units/tolerances. Use training-source references only and require the relevant student inputs. Explanatory prose may accompany an answer, but verbosity or teacher preference alone must not determine reward.

A possible first algorithm is group-relative policy optimization: sample several completions per training prompt, keep a frozen SFT reference for KL control, and update Qwen LoRA while AION and aligned projectors stay frozen. Choose exact rollout counts, LR and KL settings after Aurora memory/throughput and reward-calibration checks; this document supplies no runnable RL implementation.

Log separate reward components for answer correctness, unit/schema validity, verified evidence/coverage consistency and checkable unsupported claims. Appropriate abstention needs a reference answerability label; blanket refusal must not earn reward. Reject malformed outputs deterministically and test reward exploitation against independently checked cases.

Compare **SFT versus SFT+RL** on the same locked, source-disjoint tasks and decoding budget. Track independent scientific accuracy, uncertainty and image-only retention alongside reward. Higher training reward alone does not select the model. Leave broad unverified free-form reasoning outside the first RL objective.

## 10. Evaluation and implementation requirements

The [benchmark protocol](astroprism_benchmark_landscape.md) maps AstroSpecQA, image resources, paired evaluations and later interleaving checks. Review references for all eligible held-out sources in this small pilot. A 200-source expert panel belongs to future expansion, not a 65-object test split.

Measure supported-claim precision/recall, unsupported claims, uncertainty, task-specific classification/numeric accuracy and modality attribution. Bootstrap by physical object, not paraphrase; report small-sample uncertainty and selection bias. Retain a fixed text-only instruction control.

Compare image-only retention after mixed alignment; image versus spectrum versus joint on identical tasks; matched versus removed/swapped inputs; aligned SFT versus direct SFT where resources permit; and SFT versus RL. A random swapped pair tests sensitivity, not necessarily a recognizable physical contradiction. Removing an irrelevant modality should not alter an answer.

For later interleaved arrival (image → question → spectrum → joint question), earlier answers must not attend to later spectral tokens. Verify this masking/source-binding contract before claiming interleaving support.

Required PRISM work includes AION codecs/shared encoder adapters, a provenance-checked raw/cached feature path, paired source loading, modality masks and assistant-only labels, stage-specific optimizers, Qwen LoRA/save-reload, astronomy evaluation, and separate RL rollouts/verifiers. None is delivered by this planning commit.

Inspect [`src/model.py`](../../src/model.py), [`src/data/multimodal.py`](../../src/data/multimodal.py), [`src/data/collate.py`](../../src/data/collate.py) and [`trainer_zone_a.py`](../../src/training/trainer_zone_a.py). Existing multimodal fusion is not proof of AION support. At this baseline the trainer passes text IDs as labels and toggles backbone gradients broadly; verify explicit multi-turn labels and LoRA-specific freeze rules before use.

On Aurora, first inventory existing weights and the XPU Python environment. Preserve its accelerator-enabled PyTorch. Model loading, feature extraction, teacher inference and training belong on allocated compute resources following repository Aurora/PBS guidance. The publication/initial continuation task downloads neither data nor weights and launches no experiments; execution starts separately on the remote.

## 11. Return to 10,000 training galaxies later

After this pipeline works, expand to **10,000 distinct training galaxies plus separate validation/test**, proposed 1,000 each. The released overlap alone cannot supply this: even allocating all 646 candidates to training leaves at least 9,354 training images needing new captions; holdouts and rejections increase that number.

Generate additional image captions with the same evidence controls, recover spectra, and preserve evaluation integrity when defining larger splits. More conversations from the same 646 objects do not substitute for more independent galaxies. Scale unique qualified sources and independently reviewed evaluation references before reassessing alignment, SFT and RL gains.
