# Astronomy benchmark landscape for the PRISM galaxy pilot

Updated: 2026-09-16. This is an evaluation plan, not an executed benchmark. Start from the [646 positional candidates](astroprism_data/overlap_candidates_646.csv.gz), confirm eligible physical pairs using the [data contract](astroprism_data.md), and reserve source-disjoint validation/test sets before any training. A provisional split is **516 train / 65 validation / 65 test**, conditional on all 646 being eligible and group constraints permitting it. The [training plan](astroprism_training_plan.md) covers image-caption alignment, spectrum/joint alignment, instruction SFT and later RL; [generation prompts](astroprism_synthetic_data_prompts.md) specify evidence permissions. The 10,000-training-galaxy design is a later expansion.

## What to use

Use three complementary evaluations: **scientific prediction from real observations**, **answers grounded in the available observation**, and **modality-dependence controls**. Existing resources cover parts of this combination. I did not locate a ready-to-run public benchmark that combines matched galaxy images, numerical 1D spectra, and independently validated open-ended joint answers on the same sources. This is a bounded search finding, not a claim that no such resource can exist.

For this galaxy-only pilot, prioritize held-out redshift and spectral-feature tasks, human-supported morphology labels, and a small expert-reviewed joint QA set. Use paper-figure QA and text knowledge benchmarks as separate optional diagnostics. Do not report all-galaxy object classification or stellar subtype/temperature tasks as meaningful core pilot metrics.

To stay close to the requested AION examples, its image evaluation already crossmatches Galaxy10 with Legacy DR10, yielding approximately 8,000 galaxies with four-band cutouts. This is a useful morphology protocol to adapt, but it does not establish how many of our paired pilot galaxies carry those labels. [AION §7.2.2](https://arxiv.org/html/2510.17960)

## AstroSpecQA: the exact published reference

AstroSpecQA uses numerical DESI spectra, held apart from DESI-SpecInstruct. Its **11,698 questions** comprise:

| Task | Questions | Published metric |
|---|---:|---|
| Object class | 5,000 | Accuracy |
| Redshift, galaxies/QSOs | 3,247 | MAE, R² |
| Stellar subtype | 1,753 | Accuracy |
| Stellar effective temperature | 1,698 | MAE, R² |

Appendix A.4 gives object counts: STAR 1,753; GALAXY 1,648; QSO 1,599. Stellar subtypes: K 620; G 386; M 322; F 274; WD 94; A 53; B 3; CV 1. Thus subtype accuracy can hide rare-class failure. Questions must not be counted as distinct physical objects. These details come from [§3.4, §5.1 and Appendix A.4](https://arxiv.org/html/2609.07102).

The paper scores explanations with a fact-conditioned judge and a 100-output human check. The original benchmark files have not been obtained in this project, so its task design is a reference rather than an executable reproduction. [Explanation evaluation](https://arxiv.org/html/2609.07102)

## Existing resources and their actual input modalities

| Resource | What the model receives / does | Verified scope and availability | Role for PRISM |
|---|---|---|---|
| **Galaxy Zoo / Galaxy10** | Real galaxy images → morphology labels/vote fractions | Public catalogues/images. Galaxy Zoo DECaLS includes volunteer classifications and machine predictions; Galaxy Zoo DESI's 8.67M catalogue is automated morphology inference. [Official data](https://data.galaxyzoo.org/), [MMU Galaxy10](https://huggingface.co/datasets/UniverseTBD/mmu_gz10) | Strong image-grounding evaluation when the gold labels are human-supported. A machine prediction is a weaker reference, not independent human truth. |
| **AION galaxy-property evaluation** | Calibrated photometry, images and numerical spectra → physical-property estimates | Published PROVABGS evaluation, R², with photometry / photometry+image / photometry+image+spectrum conditions. [Paper §7.2.1](https://arxiv.org/html/2510.17960), [code](https://github.com/PolymathicAI/AION) | Closest existing multimodal scientific-prediction protocol. Keep input conditions explicit; the paper's strongest joint condition includes photometry. It is not an open-ended QA benchmark. |
| **AstroCLIP** | Real Legacy images and numerical DESI spectra → embeddings, cross-modal retrieval and downstream predictions | Public model/data workflow. [Official repository](https://github.com/PolymathicAI/AstroCLIP) | Encoder/probe and retrieval baselines. Retrieval performance alone does not establish grounded conversation. |
| **AION-Search** | Galaxy image / text query → retrieval ranking | Evaluates spirals, mergers and lenses using nDCG@10; also reports Galaxy10 classification. [Paper §§II.4–II.6](https://arxiv.org/html/2512.11982v2) | Optional retrieval diagnostic. Its synthetic captions are training supervision, not independent evaluation answers. |
| **AstroMMBench** | A rasterized astronomy-paper figure plus a four-option question | 621 MCQs across six subfields; generated from arXiv figure/context pairs and reviewed by experts. Accuracy overall/by subfield. [Paper](https://arxiv.org/html/2510.00063), [public release](https://huggingface.co/datasets/sjh11/AstroMMBench) | Optional figure-understanding/knowledge diagnostic. A paper figure, including a plotted spectrum, is not a numerical 1D DESI input or necessarily a raw sky cutout. |
| **AstroVLBench / ASIB** | Separate optical-image and DESI-spectrum tasks; spectrum plots and numerical CSVs are distributed | Public anonymous release; ASIB provides line-presence, broad-line and BPT tasks. [Dataset and task definitions](https://huggingface.co/datasets/anonymous4ai/AstroVLBench) | Possible later external check. Inspect the original inference path, spectrum frame and labels first; a native 1D adaptation is a new protocol. Its separate task populations do not establish matched image+spectrum joint QA. |

Galaxy Zoo morphology is the closest ready-made image-only scientific check for this pilot. Match human labels to the exact student view; machine-generated catalogue predictions are weaker references. AstroMMBench is useful for figure QA, but calibrated four-band AION image inputs are not automatically interchangeable with arbitrary RGB paper figures. Neither resource supplies the native paired-encoder evaluation we need.

## Proposed evaluation for the 646-candidate pilot

Everything below is a **new PRISM evaluation proposal**, not an existing published benchmark or a reported result.

### Source and split control

Use the proposed 65 validation galaxies for model selection, inference-prompt/scorer development and threshold selection. Training-data teacher prompts and their demonstrations use training sources only. Lock the proposed 65 test galaxies before new caption generation or instruction synthesis. These are targets, not an existing split: resolve candidate matches first and record the actual numbers after quality cuts and grouping. Group all repeat observations of each physical object; prevent near-duplicate/overlapping image leakage, and report sky/field and observing-condition slices. Exclude test identities from every training stage, the RL reward pool, all augmentation descendants, and demonstrations supplied in prompts.

Document possible AION pretraining exposure separately. A downstream split does not prove that the pretrained encoder never saw a source or an associated catalog label. Record object identity, survey/reduction, raw-data references, processed student views, input metadata, question, target provenance, evidence locations, scoring version, and allowed modalities in a frozen test manifest.

### Core task panel

| Panel | Proposed population | Inputs / questions | Scoring |
|---|---|---|---|
| **Galaxy redshift** | Up to 65 eligible test galaxies; report quality exclusions explicitly | Same numerical prediction task under image-only, spectrum-only and joint inputs; no target redshift supplied | MAE and R² for comparability; additionally bias, NMAD and outlier rate for normalized residual Δz/(1+z). Fix the outlier threshold on validation; a possible pilot choice is 0.05, explicitly a design choice. Report parse failures and numeric-answer coverage. |
| **Image morphology** | Human-labeled crossmatch subset `N_morph ≤ 65`, or a newly expert-labeled subset | Visible structure, orientation and feature presence. Use the exact crop/bands seen by the student | Macro-F1/balanced accuracy for declared discrete classes; proper probability scores for retained vote distributions. Keep machine-only labels in a separate analysis. Do not invent a guaranteed crossmatch count. |
| **Spectral evidence** | Valid, covered line/continuum items across the test sources; denominator varies by feature | Presence/strength of measured features, line ratios when valid, quality/coverage limitations | Precision/recall and macro-F1 on feature states; numeric errors with units/tolerances; separately score upper limits, missing coverage and masked regions. Compare against versioned measurements, not teacher prose. |
| **Joint answers** | All eligible test galaxies, up to 65, with expert review | One expert-reviewed joint question per source, evaluated with joint, image-only and spectrum-only views | Claim correctness, evidence attribution, unsupported-claim rate, and correct partial answers/abstention. Each view needs an admissible reference answer. Two independent reviewers adjudicate disagreements; automated judging is supplementary. |
| **Observation interventions** | A declared subset of the same test sources | Remove relevant lines or image regions, remove a modality, or swap one input with controlled alternatives | Direction/consistency of scientifically justified answer changes. Some swaps are observationally indistinguishable; do not force a mismatch verdict. |

With 65 eligible test galaxies, the redshift panel yields up to **195 predictions on 65 galaxies**; evaluating one joint question under three input conditions yields up to **195 answers on those same galaxies**. These are evaluation records, not additional independent objects. With this small test set, report uncertainty and treat results as feasibility evidence rather than a general astronomy benchmark. Count spectrum-feature items and morphology labels separately. Bootstrap confidence intervals by galaxy, not by duplicated question, and report class, brightness, signal-to-noise and redshift slices.

For the proposed redshift scorer, define `r = (z_pred − z_ref)/(1 + z_ref)`, bias as `median(r)`, and `NMAD = 1.4826 × median(|r − median(r)|)`. Report the fixed-threshold fraction `|r| > 0.05` separately from parse failures; report answer coverage on the complete eligible roster. These explicit scorer conventions are our proposal, not a claim that AstroSpecQA used them.

For explanation scoring, retain concise claim-level evidence references in private audit records. The scoring facts should come from independently checked measurements/labels and expert review, rather than reuse of the same teacher's generated answer. A fluent explanation can be wrong even when a final label happens to be correct.

### Controls and baselines

- Compare direct instruction tuning against caption alignment followed by instruction tuning, with matched training data and evaluation sources.
- Evaluate image-only, spectrum-only, joint and text-only conditions. Include question-only or shuffled-input controls to expose answer priors.
- Train simple frozen-AION probes on the same training roster for numerical/categorical tasks. Match available inputs: an AION photometry+image+spectrum result is not an input-matched baseline for image+spectrum-only PRISM.
- Keep observational captions separate from supervised property estimates. Catalog masses, SFRs or redshifts may be valid prediction labels; their presence in a reference table does not make them directly visible evidence in every modality.
- In independent redshift evaluation, retain observed-frame spectra and exclude true-redshift rest-frame preprocessing and target-derived student features. Ensure a later conversation turn does not receive the target through an earlier answer.
- Use only information retained after AION preprocessing. A teacher's wider image, additional band, or uncropped wavelength range cannot justify a claim about evidence the student does not receive.
- Score final answers and evidence separately. Fix parsing rules, numeric conventions and invalid-answer handling before the test run. Never hide missing/malformed predictions by reporting accuracy only on successful parses.

## Interleaving: input formatting versus scientific capability

**Interleaving is a way to arrange the inputs, not a scientific task by itself.** A model may receive `text → image → text → spectrum → question` yet answer using only one modality. Conversely, a model receiving an image prefix followed by a spectrum prefix can perform a genuinely joint task. Demonstrate joint use through the scientific panels and observation interventions above, independently of serialization format.

The initial proposal uses **all declared image/spectrum prefixes before the text conversation** for one galaxy. This is the baseline to implement and verify first. Incremental interleaved observations, repeated modality spans, and multiple-object conversations are optional future work; this note does not establish that the current PRISM loader, attention masks, or generation path support them.

### Three distinct tests

| Test | Example / contract | What it establishes |
|---|---|---|
| **Same object, different ordering** | Present the same galaxy image, spectrum, metadata and question before the answer, using image-first, spectrum-first, or text-separated layouts | Robustness to presentation order when the available evidence is unchanged. Hold the facts and permitted answer fixed; do not count permutations as new objects. |
| **New evidence over multiple turns** | Turn 1 provides an image and asks about visible morphology. Turn 2 introduces that galaxy's spectrum and asks for spectral interpretation or a combined assessment | Whether the model incorporates newly available evidence without using it prematurely. The first answer must have an image-only reference, the later answer a reference valid for the expanded input. |
| **Multiple objects and source binding** | Local alias A has image A and spectrum A; alias B has image B and spectrum B; the question explicitly asks about A, B or their comparison | Whether findings attach to the correct source. Reorder whole source-bound spans and include controlled distractors. Real survey IDs remain in the audit manifest; neutral A/B aliases may appear in the conversation. |

These examples describe **proposed test structures**, not published example records or executed model results. Keep text references unambiguous: every “this galaxy,” “its spectrum,” or “the second source” must resolve to a source and an observation already available at that answer position.

### Causal availability is part of the gold answer

For each assistant turn, record the set of available observation IDs. Its question and reference answer must use that set, not modalities supplied later in the conversation. An image-only turn cannot claim line detections from a spectrum introduced afterward. Teacher generation and judging must follow the same turn-specific restriction.

A prefix-only implementation that places **all** observations at the beginning would leak future spectra into earlier answers in an incremental-observation test. Do not run or claim this test until the model and collator implement the intended positions and causal visibility. For the initial prefix baseline, supply all observations up front and evaluate only that declared condition.

Order permutations are scientifically comparable only when evidence availability and referents remain equivalent. Introducing a spectrum after an earlier answer changes the information available and therefore requires a different reference answer; it is not merely a harmless formatting permutation.

### Engineering checks before scientific benchmarking

Validate the serialization contract with small deterministic fixtures before model-quality comparisons:

- Each modality placeholder maps to exactly one intended tensor/embedding span, with matching sample, source alias, modality type, ordering, token length and positions.
- Repeated images/spectra and multiple objects retain correct bindings through batching, collation, caching and generation; no span or cached feature crosses sample boundaries.
- Padding, truncation and missing modalities preserve valid span boundaries and attention masks; truncation must not silently retain a question whose required evidence was removed.
- Assistant-only loss labels exclude user text, modality prefixes, padding and audit sidecars. For incremental inputs, verify that earlier predictions cannot attend to later modality spans, including during cached generation.
- Dataset pixel-validity/bad-pixel masks, modality-presence masks, and language-model attention masks have separate named contracts and must not be conflated.

These are **placeholder/alignment and causal-visibility checks**, not evidence of astronomy competence. After they pass, evaluate scientific correctness, source attribution, partial-answer behavior, order sensitivity and observation interventions on the held-out sources. Report both engineering pass/fail results and scientific scores; success in either category alone does not prove the other.

### Evaluation after each training stage

| Saved checkpoint | Required check | What a positive result establishes |
|---|---|---|
| Image caption alignment | Image claims against independently checked morphology/quality facts; compare image-conditioned and text-only output | Whether the image projector supplies useful visual evidence on unseen sources |
| Spectrum and joint caption alignment | Spectral feature correctness, uncertainty, image retention, and per-claim modality attribution | Whether adding the spectrum branch supports its own facts without erasing image grounding |
| Instruction SFT | All applicable task panels, answerability and multi-turn consistency; matched direct-SFT baseline if feasible | Whether the model follows instructions and keeps supported scientific content |
| Later RL pilot | The identical locked panels plus reward edge-case audit and comparison with the saved SFT checkpoint | Whether optimization of the independent verifier improves held-out scientific results, rather than only training reward |

Save all intermediate checkpoints and define the comparisons before seeing test results. Use validation for early stopping and prompt changes. Score predeclared checkpoints on the test set in one frozen evaluation; repeated test inspection followed by edits turns it into development data. If exploratory tests are unavoidable, disclose them and obtain a fresh independent test pool for final claims.

### RL reward and evaluation are separate

The proposed RL pool contains training sources only and tasks with independent numerical/categorical checkers. Fix units, tolerances, quality eligibility and parse rules on training/validation data. Do not reward teacher agreement, answer length, eloquence or a model's self-assessment. A reward for a correct scalar does not verify the accompanying scientific explanation. Test the checker using valid, invalid, boundary and malformed outputs before policy optimization.

Measure held-out correctness, unsupported claims, appropriate abstention and retained image/spectrum capabilities separately from reward. Keep a non-RL SFT checkpoint as the baseline. A gain on a reward built from the same synthetic teacher that produced training answers is not independent evidence; use catalog/measurement targets and expert-reviewed claims as applicable. Given the small pool, proceed beyond a bounded RL pilot only if the verifier is sound and the independent results justify it.

### Optional external checks

1. **Galaxy10 / human Galaxy Zoo labels:** morphology test after identity-overlap checks and matching the image view. A re-split or re-crop is a new evaluation condition, not an exact reproduction of a published score.
2. **AstroVLBench ASIB:** inspect and reproduce its original input protocol first, then separately identify any numerical-spectrum adaptation. Verify survey release, preprocessing, wavelength frame and target definitions; do not assume DESI releases are interchangeable.
3. **AstroMMBench:** only through an interface that handles its paper figures. Report figure QA separately from raw-observation performance.
4. **AION / AstroCLIP probes and retrieval:** use the same held-out source roster and declared inputs. Retrieval and property estimation do not by themselves establish instruction following.

## Acceptance before implementing benchmark code

Before scoring, assemble a reviewable evaluation manifest: exact source roster, overlap report, task eligibility counts, independent targets, questions/reference answers, per-view permissions, frozen scorers, and baseline configurations. Complete a small expert audit spanning ambiguous morphology, low signal-to-noise, nondetections, out-of-coverage features and aperture mismatch. Then freeze the benchmark and run the proposed comparisons.

Document the resulting native image+spectrum QA panel as a **new AstroPrism pilot evaluation**, with its small sample size, construction and limitations. It should not be called AstroSpecQA reproduction unless the original benchmark data and evaluation protocol are obtained and preserved.
