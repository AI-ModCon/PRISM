# DOCCI: conditioning diagnosis, repeated small-set training, and full-data follow-through

The user authorized steps 1–3: demonstrate learning on a repeated small set,
diagnose conditioning, then train on the complete prepared DOCCI pool. This report
distinguishes completed checks, submitted experiments, and candidate settings.

**Current direction, September 22:** the user explicitly requested joint training
on the full prepared dataset instead of further small-set learning-rate or
feature-retention gates. Full-data job **8848858** completed all 1,809 updates;
post-training caption-conditioning evaluation **8854234** also completed. The earlier
quality-first prerequisites below are historical decisions, superseded for this
larger-data learning experiment. PRISM, VAE and the native conditioner stay frozen;
the connector and diffusion transformer learn together from image flow loss.

## Completed preparation

The six-update batch-four smoke, Aurora job **8847986**, completed successfully.
It used the existing 500-step connector and original OmniGen2 diffusion weights.
Both trainable groups changed, while the audited frozen state remained unchanged.
Runtime was **384.79 seconds**, including loading, validation and checkpoint
output. Peak device allocation was **27,889,233,408 bytes (25.97 GiB)**. The last
five updates took approximately 7.34–7.72 seconds each. These measurements validate
the batch size, not generation quality.

The staged runner adds deterministic subset selection, exact per-ID exposure
counts, and fresh-stage initialization from a completed full joint checkpoint.
Fresh-stage initialization retains trained FP32 master weights and buffers but
resets optimizer and sampling progress. Exact resume remains bound to the original
protocol. PRISM, its input alignment, VAE and native conditioner stay frozen.

The conditioning diagnostic compares native and PRISM features and paired losses
at fixed noise levels; validates identical actual noisy diffusion inputs; and
compares native, mixed-conditioner guidance, no guidance, and PRISM negative
conditioning. Explicit raw/chat formatting is recorded. An EOS anchor used for
empty raw PRISM conditioning is experimental, not a trained unconditional token.

Local validation passed **106 distinct focused/regression tests** across the
staged runner, conditioning controls, launcher, CPU optimizer and trainability
contracts. Ruff checks and formatting passed. An independent review found and
corrected one diagnostic issue: wrong training captions now stay within the
recorded small-set membership. No unresolved blocker was found in that review.
Fixture tests do not establish real generation quality.

## Submitted experiments

| Experiment | Aurora job | Configuration | Result status |
| --- | --- | --- | --- |
| `conditioning-01` | 8848014 | Four train and four validation captions; three fixed timesteps; raw/chat features; two validation image comparisons at 50 diffusion steps | Completed, frozen state unchanged; 389.66 seconds |
| `overfit32-256-01` | 8848017 | 32 selected pairs; 256 updates; batch 4; connector LR 1e-4; diffusion LR 1e-5; fixed probes every 64 updates; checkpoints and images every 128 | Completed; both groups changed, frozen state unchanged; 2,479.83 seconds |
| `conditioning-overfit256-01` | 8848158 | Completed joint checkpoint; eight train/eight validation captions; three fixed timesteps; raw prompts; two validation image comparisons | Completed; frozen state unchanged; 502.64 seconds |
| `feature-alignment32-1000-01` | 8848159 | Connector-only feature alignment; 32 train/eight validation captions; 1,000 updates at batch four; LR 1e-4; all other components frozen | Completed; connector changed, frozen state unchanged; 263.76 seconds |
| `conditioning-aligned1000-01` | 8848222 | Verified aligned connector; original frozen DiT; chat prompts; eight train/eight validation captions and two visual cases | Completed; frozen state unchanged; 381.64 seconds; principal subjects still missing |
| `conditioning-template-swap-01` | 8848325 | Two train/two validation captions; native/aligned/prefix/suffix/both-template CFG1 routes; two held-out image cases at 50 steps | Completed; all audits passed; 414.69 seconds; prefix replacement removes most excess flow MSE |
| `feature-alignment-regions32-1000-01` | 8848399 | Connector-only equal prefix/content/suffix feature loss; initialized from verified content-alignment checkpoint; same 32 train/eight held-out captions; 1,000 updates at batch four | Completed; 264.87 seconds; all execution/artifact audits passed; feature improvements require generation validation |
| `conditioning-regions1000-01` | 8848573 | Verified region-alignment connector; original frozen DiT; eight train/eight held-out captions, three flow times, two image cases with native CFG1 and CFG5 references | Completed; 446.82 seconds; all execution/artifact checks passed; partial subject recovery, insufficient for full-data quality gate |
| `aligned-regions-joint-smoke-01` | 8848651 | Six joint updates from strict region alignment; 32-pair subset; batch four, accumulation four; chat, CFG5 native-negative galleries | Completed; 548.74 seconds; execution and terminal checkpoint audits passed; no demonstrated quality gain |
| `conditioning-aligned-joint6-01` | 8848734 | Audited joint step-six checkpoint; earlier diagnostic cases/seeds; original/adapted native CFG1/CFG5 controls and PRISM variants | Completed; 485.92 seconds; execution/artifact checks passed; PRISM image degradation confirmed at matched seeds |
| `joint-components-01` | 8848800 | Frozen 2×2 region/joint-six connector and original/joint-six DiT controls, plus native conditioning on each DiT; two train/two held-out targets; twelve CFG5 images | Completed; 1,010.69 seconds; all execution/artifact checks passed; updated connector reproduces degradation with either DiT |
| `aligned-regions-joint-lr1e5-smoke-01` | 8848832 | Six-update replay from the same region initializer; connector LR reduced tenfold to 1e-5; DiT LR remains 2e-6; all training/evaluation settings otherwise unchanged | Completed; 559.67 seconds; terminal checkpoint/artifact checks passed; retained as historical evidence |
| `full-docci-joint-3epochs-01` | 8848858 | Full 9,647-pair pool; 1,809 updates; batch four/accumulation four; connector LR 1e-4 and DiT LR 2e-6; fresh stage from audited joint-six checkpoint | Completed, PBS exit 0; 19,522.59 seconds; final checkpoint and frozen-state collection checks passed |
| `full-docci-joint-conditioning-01` | 8854234 | Verified terminal joint checkpoint; eight train/eight held-out captions, three flow times; fourteen images with original/adapted native CFG1/CFG5 and PRISM controls | Completed, PBS exit 0; 521.15 seconds; all requested controls and images collected |

The completed overfit run exposed every selected pair exactly **32 times**:
1,024 total exposures across 32 unique pairs. The terminal checkpoint digest was
independently verified on Aurora; source and artifact acceptance passed.
The higher diffusion LR is a deliberate small-set fitting experiment, not the
full-data learning rate. Its primary galleries use raw prompts and guidance 1;
the diagnostic separately compares guidance 5 and alternative negative branches.
The overfit runner's native gallery also follows its explicit guidance setting.
Do not conflate it with the diagnostic's native CFG5 baseline.

All generation is caption-only and target-free. Training images enter the VAE
loss target, not PRISM's vision encoder. The 100 validation examples are excluded
from optimization. DOCCI was an OmniGen2 training source, so this is not evidence
of unseen-data generalization for the pretrained generator.

The intermediate step-64 evaluation over 32 fixed examples per split gives
training flow MSE **0.432859 → 0.419779** (3.02% lower) and validation flow MSE
**0.457064 → 0.457979** (0.20% higher). This is preliminary fitting evidence,
not generation accuracy. The snapshot below was saved while the run was at
step 90. A second explicitly interim snapshot at step 192 contains 196 optimizer
log rows because collection occurred while training continued. The completed
report and final frozen-state checks are now saved separately at the run root.

| Fixed evaluation step | Train flow MSE, 32 examples | Validation flow MSE, 32 examples |
| ---: | ---: | ---: |
| 0 | 0.432859 | 0.457064 |
| 64 | 0.419779 | 0.457979 |
| 128 | 0.412369 | 0.454402 |
| 192 | 0.402326 | 0.458851 |
| 256 | 0.390641 | 0.457206 |

The final validation value in this table uses the same original 32-example
cohort, recomputed from the full evaluation. Final mean loss over all 100
validation examples is **0.464707**, which must not be compared directly to the
32-example initial mean. Training loss fell **9.75%**; the unchanged validation
cohort rose **0.03%**, effectively flat.

At step 256, one of the two shown training cases partially learns its subject:
the pine-tree caption now produces a recognizable small pine tree, without the
requested firewood/tools/fence arrangement. The digital clock remains absent.
Neither held-out sample produces its requested animals or airplanes. Thus some
fitting occurred, but reliable tiny-set caption following and a held-out gain
were not established. This supports the bounded feature-alignment experiment
before full-data training; these four images are not a benchmark.

See the [final training gallery](../assets/image_generation/2026-09-22-docci-alignment/runs/overfit32-256-01/train-comparison-step-000256.png),
[final validation gallery](../assets/image_generation/2026-09-22-docci-alignment/runs/overfit32-256-01/validation-comparison-step-000256.png),
and [loss curves](../assets/image_generation/2026-09-22-docci-alignment/runs/overfit32-256-01/plots/connector-diffusion-losses.png).

At step 192, training loss is 7.05% below its initial value, while validation
loss is 0.39% higher. This is fitting of the selected set without a demonstrated
held-out gain. The saved step-128 images still miss the principal subjects in all
four inspected cases: the two training captions for a clock and pine tree give
an unrelated suspended object and a stone sculpture; the two held-out captions
for helmeted animals and airplanes give furniture/interior scenes. Native
pretrained controls at the **same CFG 1** produce recognizable animals and
airplanes. This rules out guidance being the only difference in these examples.
Image and target hashes and starting-latent identity were checked. The generated
images were not altered. This remains an interim qualitative diagnostic.

See the [interim training gallery](../assets/image_generation/2026-09-22-docci-alignment/runs/overfit32-256-01/interim-step192/train-comparison-step-000128.png)
and [interim validation gallery](../assets/image_generation/2026-09-22-docci-alignment/runs/overfit32-256-01/interim-step192/validation-comparison-step-000128.png).

Training-run wrong-caption probes reset global RNG but do not record the actual
noisy latent and timestep hashes. Their individual gaps are therefore left
provisional until the completed checkpoint is checked with the separate audited
conditioning diagnostic. The diagnostic results below do verify actual paired
inputs. No running source snapshot was changed.

## Completed conditioning findings

The diagnostic completed 24 paired noise/timestep probes across eight unique
captions and saved 14 fixed-noise generated images. Source, target-byte and
frozen-state checks passed. Each table row aggregates four captions at three
noise levels; these are not twelve independent images or accuracy scores.

| Route | Train wrong-minus-matched loss | Validation wrong-minus-matched loss | Matched-caption wins, train / validation |
| --- | ---: | ---: | ---: |
| Native pretrained OmniGen2 | 0.024812 | 0.028594 | 12/12 / 12/12 |
| PRISM raw | 0.002768 | 0.001726 | 6/12 / 8/12 |
| PRISM chat | 0.002426 | 0.001043 | 6/12 / 8/12 |

Native conditioning shows substantially more useful caption sensitivity in this
small controlled comparison. Raw/chat formatting alone did not close that gap.
The raw and chat image comparisons were inspected: original native OmniGen2
renders the requested dog/penguin in blue helmets and two airplanes. None of the
PRISM variants restores those principal objects. Disabling guidance changes the
outputs to unrelated foliage or garbled text; replacing the negative branch with
an untrained PRISM anchor does not repair them. Raw EOS-negative outputs are
nearly uniform colors. These two cases are diagnostic, not a global benchmark.

The raw feature RMS is about five times smaller than native, but the upstream
caption embedding starts with `RMSNorm → Linear`. A simple global rescaling would
largely cancel there, so scale alone is not an established cause.

For all eight matched and eight wrong-caption pairs inspected, native and PRISM
chat inputs have exactly equal formatted text, token IDs and attention masks.
This makes per-token teacher alignment a feasible fallback for these examples.
Every additional pair must pass the same assertions; this is not established
across the full dataset. A useful fallback would keep both parents and the DiT
frozen and train the connector against the native features after the actual
frozen caption RMSNorm, separating caption-content and repeated-template errors.
The completed feature experiment below evaluates this hypothesis.

The feature-alignment experiment is implemented as
[`align_prism_image_conditioning.py`](../../tools/align_prism_image_conditioning.py).
It trains only the connector against detached native caption features after the
actual frozen caption RMSNorm. Every caption must pass exact formatted-input,
token, mask and content-span checks, including observing the native model's
actual forward inputs. Image pixels are not read. Template positions are measured
separately and excluded from the optimization loss. Defaults are 32 training
captions, eight validation captions and 1,000 updates. Its checkpoint kind is
distinct from flow training. Job **8848159** completed in 263.76 seconds within
the reviewed one-node, one-tile, 20-minute debug allocation. Thirteen focused fixture tests
passed after integration; independent review caught and corrected the native
model's positional token-input contract. These tests add to the 106 above and
do not establish real-checkpoint training or image quality.

The explicit weights-only restore helper is now implemented in
[`prism_image_alignment_checkpoint.py`](../../tools/prism_image_alignment_checkpoint.py)
and integrated through mutually exclusive `--alignment-checkpoint` and
`--joint-checkpoint` diagnostic flags. Before copying connector weights it
validates the terminal report/digest, parent and original DiT, frozen/norm state,
audited 500-step starting connector, data/held-out identity, exact token spans and
exposures. The original native baseline precedes restoration. It never restores
optimizer/RNG/sampler state and never treats feature MSE as generation accuracy.
The alignment runner/helper/diagnostic suite passed **58 tests**, launcher suite
**29 tests**, and plotter suite **14 tests**. The combined integration run passed
97 tests with four plotting tests skipped in the model environment; all four
also passed in the plotting environment as part of the 14-test suite. Independent
review found no remaining blocker. Running Aurora source snapshots were preserved.

Alignment training uses `prism-feature-alignment`; its follow-up diagnostic uses
the separate `prism-alignment-validation` snapshot. The generation diagnostic
received the verified terminal alignment checkpoint digest and was submitted as
job **8848222** after dry-run and shell checks. Its completed result is below.

See the [raw comparison](../assets/image_generation/2026-09-22-docci-alignment/runs/conditioning-01/conditioning-raw-comparison.png),
[chat comparison](../assets/image_generation/2026-09-22-docci-alignment/runs/conditioning-01/conditioning-chat-comparison.png),
and [interpreted measurements](../assets/image_generation/2026-09-22-docci-alignment/runs/conditioning-01/interpreted-summary.json).

## Completed joint-checkpoint diagnostic

The step-256 checkpoint was checked with 16 unique target/caption pairs at three
fixed timesteps: 48 paired evaluations per route, not 48 independent examples.
Actual noisy-input and timestep hashes matched within every control. All frozen
hashes remained unchanged. Native conditioning in this table uses the **adapted
DiT**, exactly as the PRISM route does.

| Route | Train wrong-minus-matched MSE | Validation wrong-minus-matched MSE | Matched-caption wins, train / validation |
| --- | ---: | ---: | ---: |
| Native conditioner / adapted DiT | 0.025577 | 0.029261 | 24/24 / 23/24 |
| PRISM raw / adapted DiT | 0.009406 | 0.001662 | 23/24 / 13/24 |

The 24 probes in each cell represent eight captions at three timesteps. PRISM
has useful sensitivity on the trained subset but weak held-out caption use.
Both original and adapted native routes retain recognizable animals/airplanes.
PRISM CFG1 and mixed CFG5 still miss those subjects; the untrained EOS negative
branch produces nearly uniform colors. See the
[controlled image comparison](../assets/image_generation/2026-09-22-docci-alignment/runs/conditioning-overfit256-01/conditioning-raw-comparison.png).

## Completed connector-feature alignment

All 40 caption/token/mask/content-span audits passed, including actual native
forward inputs. The run used exactly **4,000 exposures**, or **125 uses of every
selected training caption**. Eight held-out captions never entered optimization.
Only connector weights changed. PRISM, the native teacher, VAE and original DiT
hashes stayed unchanged; no target pixels or saved activation caches were used.
The terminal checkpoint digest was independently verified and artifact checks
passed.

| Feature metric after frozen caption RMSNorm | Training, initial → final | Held out, initial → final |
| --- | ---: | ---: |
| Caption-content MSE | 0.888033 → 0.026440 | 0.888760 → 0.063224 |
| Caption-content cosine similarity | 0.0320 → 0.9382 | 0.0335 → 0.8605 |
| Template MSE, excluded from optimization | 1.106454 → 0.680301 | 1.106582 → 0.680161 |
| Template cosine similarity | 0.0425 → 0.2595 | 0.0424 → 0.2594 |

These are feature-matching measurements, not generation accuracy. Content
features align substantially on this small held-out set, while template tokens
remain less aligned. Job 8848222 tested whether the improvement reaches actual
flow conditioning and caption-only generation; its results below do not yet
support full-data training with this connector.

The submitted diagnostic's native image baseline uses CFG5, so its native and
PRISM CFG1 images must not be described as matched-guidance comparisons. If the
aligned images fail, first add original-native CFG1 with identical noise and
settings. A bounded next diagnostic can then replace only native prefix or suffix
features in the aligned sequence, separately. All 40 audited sequences have 23
prefix and two suffix template tokens; causal suffix states may contain caption
information. Such teacher-assisted swaps could locate residual conditioning
errors without establishing a trained PRISM route. The subsequent completed swap
experiment and its limitations are recorded below.

Checkpoint:
`/lus/flare/projects/ModCon/sandeep/prism-docci-alignment-20260922/runs/feature-alignment32-1000-01/connector-feature-alignment-step-001000.pt`.

SHA256: `431938ef0fe99e480a28a71fd8eb716d067050cdde7388dd0ed33a2af794a642`.

The [feature loss and cosine curves](../assets/image_generation/2026-09-22-docci-alignment/runs/feature-alignment32-1000-01/plots/feature-alignment.png)
are also available as a [PDF](../assets/image_generation/2026-09-22-docci-alignment/runs/feature-alignment32-1000-01/plots/feature-alignment.pdf)
and [numeric series](../assets/image_generation/2026-09-22-docci-alignment/runs/feature-alignment32-1000-01/plots/feature-alignment-series.json).
Their renderer checks report/log agreement, finite values, fixed cohorts and exact
exposure counts. Template curves are explicitly excluded from the training loss.

## Completed generation check of the aligned connector

Job **8848222** completed successfully. The strict adapter restored only the
verified step-1000 connector, retained the original DiT and revalidated caption,
data and checkpoint identities. All 2,220 frozen tensor hashes stayed unchanged.
All 48 actual noisy-input/timestep controls passed, source bytes matched, and
the eight generated images were collected with verified hashes and starting noise.

| Route with original DiT | Train matched MSE | Held-out matched MSE | Train / held-out wrong-minus-matched MSE | Matched-caption wins, train / held out |
| --- | ---: | ---: | --- | --- |
| Native conditioner | 0.453660 | 0.485447 | 0.024849 / 0.026605 | 24/24 / 24/24 |
| Feature-aligned PRISM chat | 0.673481 | 0.724732 | 0.021485 / 0.013281 | 18/24 / 16/24 |

Each split contains eight captions at three timesteps, not 24 independent
examples. The aligned connector has measurable caption sensitivity, but its
matched denoising losses remain substantially above native. Only five of eight
held-out captions have a positive mean caption gap across the three timesteps.

In both inspected cases, PRISM CFG1 and positive-PRISM/native-negative CFG5 still
miss the principal subjects: the dog/penguin and airplanes are not recognizable.
The images contain blurred textures, window-like shapes or background colors.
The native CFG5 control renders the requested subjects. The PRISM-negative CFG5
ablation also fails; its empty-message anchor was never trained.
See the [aligned image comparison](../assets/image_generation/2026-09-22-docci-alignment/runs/conditioning-aligned1000-01/conditioning-chat-comparison.png).

The bounded diagnostic completed as job **8848325**: matched native and aligned PRISM
CFG1 controls, then replacement of native prefix only, suffix only, or both
template regions in the aligned feature sequence. It used two train and two
held-out captions, three fixed timesteps, and two held-out image cases. Exact
observed token IDs/masks and content spans must match before any replacement.
Replacement occurs before DiT RMSNorm, with all weights frozen. These hybrids
are teacher-assisted diagnostic routes, not deployable PRISM improvements. A
suffix rescue may reflect caption information in causal suffix states. This
experiment localizes a major prefix mismatch, as detailed below; full-data training remains pending.

The optional `--template-swap-controls` implementation and its integration passed
**134 tests** and Ruff; independent review found no blocker. Tests include strict
alignment admission, malformed token/mask/span rejection, unchanged caption-content
states, actual DiT conditioning hashes, and identical flow/sampling noise. These
fixture tests validate the evidence machinery, not real image quality. The
dedicated `prism-template-swap-diagnostic` source snapshot was overlaid with seven
reviewed files and checksum-verified before submission. Launcher dry-run and both
PBS/worker shell checks passed. The allocation is one node, one process and one
XPU tile for 12 minutes, based on the previous 381.64-second diagnostic and fewer
flow forwards. All previously executed source snapshots remain unchanged.

The completed run was collected through `pack_run.py` with `--source-root` set to
`prism-template-swap-diagnostic`, then rendered and audited through
`provenance/render_template_swap.py` on the collected run directory. The latter
refuses incomplete or mismatched evidence and displays the five primary CFG1
routes. Original native CFG5 is an extra checked reference, not one of those five
columns. The renderer passed an actual diagnostic fixture and seven deliberately
corrupted or incomplete evidence cases. Checkpoint bytes remain on Aurora.

## Completed prefix/suffix diagnosis

The completed template-swap report passed both the generic collector and the
strict template artifact audit: eight matched/wrong caption-input audits, twelve
paired flow controls across five routes, twelve image hashes, exact feature
partition/conditioning hashes, identical starting noise, and all 2,220 frozen
tensor hashes. This is four target/caption pairs at three timesteps, not twelve
independent examples. Every primary image route uses CFG1 with the same noise.

| Conditioning route, original frozen DiT | Train matched flow MSE | Held-out matched flow MSE |
| --- | ---: | ---: |
| Native | 0.439760 | 0.507843 |
| Aligned PRISM | 0.659163 | 0.725752 |
| Native prefix, PRISM content and suffix | 0.442901 | 0.511934 |
| PRISM prefix and content, native suffix | 0.651322 | 0.712202 |
| Native prefix and suffix, PRISM content | 0.441780 | 0.512074 |

Replacing the 23 prefix positions removes about 98% of the aligned-versus-native
excess matched MSE on this tiny cohort while leaving caption-content states
unchanged. Replacing only the two suffix positions has a much smaller effect.
This identifies a major mismatch in positions that the content-only objective
excluded. It does not establish general performance across DOCCI.

Prefix replacement also changes the images from garbled textures into coherent
indoor/outdoor scenes. Subject fidelity remains insufficient: the animal is
distorted and the second scene contains vehicle-like objects rather than the
requested two airplanes. Suffix-only images remain similar to aligned PRISM;
using both template regions looks similar to prefix-only. The matched native
CFG1 control is itself less faithful than the separate native CFG5 reference.
See the [matched-CFG1 template comparison](../assets/image_generation/2026-09-22-docci-alignment/runs/conditioning-template-swap-01/template-swap-comparison.png).

All twelve shared flow rows exactly reproduce the previous aligned/native
diagnostic, including wrong-caption IDs and noisy-input hashes. The eight input
audits (six unique captions) also match the original feature cache's native
feature hashes. Native prefix hashes vary slightly across captions; a single
constant native prefix tensor was not tested.

## Completed alignment of prefix, content and suffix

Job **8848399**, `feature-alignment-regions32-1000-01`, completed on September 22
in 264.87 seconds (PBS wall time 4:37, exit 0). It assigned equal weight to
prefix, caption-content and suffix MSE after the actual frozen caption RMSNorm,
then averaged equally over captions. It initialized from the audited step-1000
content-alignment connector with a fresh optimizer, sampler and RNG, retained
the same 32 training and eight held-out captions, and trained only the connector
for 1,000 updates at batch four and LR 1e-4. Exactly 4,000 caption uses were
recorded, or 125 per training caption. All 40 rebuilt token/feature audits matched
the initializer. PRISM, the native teacher, VAE and original DiT stayed unchanged.

This objective has schema version 2 and the distinct checkpoint kind
`real_checkpoint_connector_native_region_feature_alignment`, with objective
`native_caption_rmsnorm_equal_region_mse_v2`. Strict initialization validates the
step-500 → content-alignment → region-alignment chain before copying any weights.
Region weighting is an experiment; feature improvements must be checked through
actual generation. Feature checkpoints remain distinct from
joint flow-training checkpoints.

| Feature metric after frozen caption RMSNorm | Training, initial → final | Held out, initial → final |
| --- | ---: | ---: |
| Equal-region MSE | 0.609071 → 0.008978 | 0.621051 → 0.022792 |
| Prefix MSE | 0.638380 → 0.000527 | 0.638294 → 0.000527 |
| Prefix cosine | 0.2745 → 0.9989 | 0.2744 → 0.9989 |
| Caption-content MSE | 0.026440 → 0.024105 | 0.063224 → 0.062828 |
| Caption-content cosine | 0.9382 → 0.9437 | 0.8605 → 0.8634 |
| Suffix MSE | 1.162393 → 0.002302 | 1.161634 → 0.005022 |
| Suffix cosine | 0.0868 → 0.9981 | 0.0867 → 0.9959 |

The held-out equal-region feature loss fell 96.33%, with caption-content loss
slightly improved rather than degraded. This addresses the previously excluded
template positions on eight held-out captions; it is not image-quality evidence.
The [nine-panel curves](../assets/image_generation/2026-09-22-docci-alignment/runs/feature-alignment-regions32-1000-01/plots/feature-alignment.png),
[PDF](../assets/image_generation/2026-09-22-docci-alignment/runs/feature-alignment-regions32-1000-01/plots/feature-alignment.pdf),
and [numeric series](../assets/image_generation/2026-09-22-docci-alignment/runs/feature-alignment-regions32-1000-01/plots/feature-alignment-series.json)
were collected, checked and visually inspected. Both initializer and terminal
checkpoint digests were independently verified on Aurora; the collector's full
execution/artifact acceptance passed. No checkpoint bytes were downloaded.

Terminal checkpoint:
`/lus/flare/projects/ModCon/sandeep/prism-docci-alignment-20260922/runs/feature-alignment-regions32-1000-01/connector-region-feature-alignment-step-001000.pt`.

SHA256: `5cb963cbf1d0c5d231b2df7183d488133d84f3d90439971ec85736546739cc6d`.

The integrated writer, restore helper, diagnostic, joint runner and launcher
passed **171 tests**, Ruff checks and independent review. The ordinary diagnostic
now supports `--native-cfg1-control`, preserving the separate native CFG5
reference. The executed source snapshot is `prism-feature-alignment-regions`;
its eight-file overlay was checksum-verified. The reviewed launcher dry-run and
PBS/worker shell checks passed. Allocation is one node, one process and one XPU
tile in the capacity queue for ten minutes. Older source snapshots are unchanged.

Collect with the versioned `provenance/regions/pack_feature_alignment.py`, its
sibling `pack_run.py`, `--source-root ROOT/prism-feature-alignment-regions`, and
`--verify-checkpoint`. This verifies both initializer and terminal checkpoint
digests and preserves the initializer report plus exact collector source bytes.
Use `provenance/regions/plot_feature_alignment.py` for prefix/content/suffix and
combined-template curves. Preserve collected source snapshots under the run's
own provenance directory, and do not overwrite newer local rendering helpers
with historical copies carried by an archive.

## Completed generation check of region alignment

Job **8848573**, `conditioning-regions1000-01`, was submitted at 05:05 UTC on
September 22 through the reviewed launcher after checkpoint acceptance,
independent review, source hash checks, dry-run and both shell syntax checks.
It completed in **446.82 seconds**, with PBS wall time 7:40 and exit 0.
It used the unchanged `prism-feature-alignment-regions` snapshot and original
frozen DiT. Allocation is one node, one process and one XPU tile for fifteen
minutes, based on the preceding 381.64-second diagnostic plus two native CFG1
images. The corrected submitted command uses the actual schema-2 checkpoint
basename and verified digest; the original `NOT-SUBMITTED` proposal is historical.

The diagnostic compares eight training and eight held-out captions at three
fixed flow times (48 paired cases per route), plus two held-out generated cases
at 50 sampling steps. Each case includes native CFG1 and CFG5, PRISM CFG1,
PRISM-positive/native-negative CFG5, and PRISM-positive/negative CFG5: ten images
with matched initial noise. The last negative anchor is untrained and remains
an ablation. Strict schema-2 restoration revalidated the complete checkpoint
lineage and copied four connector tensors. All 48 actual paired noisy-input and
timestep controls, ten image hashes, matched initial noise, source checks and
all 2,220 frozen tensor hashes passed. No training occurred in this diagnostic.

| Route with original frozen DiT | Train matched MSE | Held-out matched MSE | Train / held-out wrong-minus-matched MSE | Matched-caption wins, train / held out |
| --- | ---: | ---: | --- | --- |
| Native conditioner | 0.453660 | 0.485447 | 0.024849 / 0.026605 | 24/24 / 24/24 |
| Region-aligned PRISM chat | 0.455404 | 0.493149 | 0.022267 / 0.016004 | 23/24 / 20/24 |

Each split contains eight captions at three fixed timesteps. PRISM's held-out
matched loss fell **31.95%** from the content-only aligned result and is now
**1.59% above native**, versus 49.29% above native previously. The training excess
is 0.38%. Caption sensitivity also improves, but the held-out caption gap remains
smaller than native. Near-native denoising loss is not proof of equivalent images.

The [matched-guidance image comparison](../assets/image_generation/2026-09-22-docci-alignment/runs/conditioning-regions1000-01/conditioning-chat-comparison.png)
shows a recognizable yellow aircraft and a second purple/blue aircraft-like
object in PRISM CFG5, with incorrect colors and placement relative to the caption.
The stuffed-animal scene is coherent but merges or misses subjects: a dog-like
toy appears under a cabinet and the requested penguin is absent. PRISM CFG1
retains room/runway backgrounds but fails the principal objects. Native CFG5
retains both animals and recognizable aircraft; native CFG1 is itself weaker.
Thus the revised connector provides substantial recovery without full semantic
recovery. These two images are a qualitative diagnostic, not a benchmark.

A bounded six-update joint smoke is justified to validate strict schema-2
initialization, both optimizer groups and batch-four/accumulation-four timing,
with initial/final CFG5 galleries to detect degradation. That mechanics check
does not satisfy the full-data quality gate by itself. The missing animal and
remaining caption errors must still be investigated through controlled small-set
training/evaluation before scaling. Full-data training remains pending.

## Completed joint initialization smoke from region alignment

The joint runner now accepts an explicit `--init-alignment-checkpoint` and digest.
It requires the audited step-500 connector, original DiT and chat formatting,
then calls the strict feature-checkpoint adapter before creating trainable groups
or FP32 CPU masters. The aligned connector retains its FP32 weights; optimizer,
sampler and RNG start fresh, with alignment lineage recorded. This route is
mutually exclusive with joint-checkpoint initialization. Exact resume retains
the alignment identity and original protocol.

The integrated runner, adapter and launcher passed **91 tests** and Ruff checks;
independent review found no blocker. Gallery labels distinguish this initialization
from the original step-500 connector. The expanded schema-2 integration is part
of the subsequent 171-test suite. No submitted or completed source snapshot was
edited.

Job **8848651**, `aligned-regions-joint-smoke-01`, was submitted at 05:32 UTC on
September 22 and completed in **548.74 seconds** (PBS 9:21, exit 0).
Its new `prism-aligned-joint-smoke` source snapshot copies the
reviewed region-alignment source; eight key files were checksum-verified before
launch. Independent command review, remote dry-run, PBS/worker syntax checks and
32-thread CPU optimizer binding checks passed. The one-node, one-process,
one-XPU-tile capacity allocation was **twenty minutes**. The run saved ten images
plus a full joint checkpoint, whose terminal digest was independently verified.

The command restores the verified schema-2 connector through the strict helper,
uses original FP32 DiT master weights, and creates fresh optimizer/sampler/RNG
state. It completed six updates at batch four with accumulation four: **96 exact
image exposures**, or three uses of every selected training pair. Connector LR is
1e-4 and DiT LR is 2e-6. PRISM, native conditioner and VAE stay frozen. The initial
and terminal probes use the same eight training/eight held-out examples; the
gallery includes two examples per split before and after updates plus two native
validation references. All use 50 diffusion steps and CFG5 with native negative
conditioning. The joint checkpoint is approximately 47.7 GB and remains
on Aurora.

Strict alignment lineage, FP32 masters, finite gradients, changes in both groups,
all 1,634 frozen tensor hashes, source/image identities and exact exposures passed. DiT masters
were restored from 582 original FP32 tensors (3,967,161,400 parameters), with
4,200,448 connector parameters. Peak device allocation was **27.74 GiB**.
Update durations were 13.72, 11.73, 11.24, 10.69, 11.24 and 10.79 seconds;
the last five average **11.14 seconds per 16-image accumulated update**.
Loading, evaluation, sampling and checkpoint I/O are additional costs.

| Fixed eight-example probe | Initial flow MSE | After six updates | Change |
| --- | ---: | ---: | ---: |
| Training | 0.375445 | 0.378340 | +0.77% |
| Held out | 0.526330 | 0.527005 | +0.13% |

The smoke establishes execution, not a quality gain. The training clock changes
from the requested `3:41` to `3:44`; the pine tree remains recognizable but is
placed in a basket. The held-out dog/penguin remains merged or missing, and the
aircraft case becomes a stylized road/sign scene. See the
[training comparison](../assets/image_generation/2026-09-22-docci-alignment/runs/aligned-regions-joint-smoke-01/train-comparison-step-000006.png),
[held-out comparison](../assets/image_generation/2026-09-22-docci-alignment/runs/aligned-regions-joint-smoke-01/validation-comparison-step-000006.png)
and [loss curves](../assets/image_generation/2026-09-22-docci-alignment/runs/aligned-regions-joint-smoke-01/plots/connector-diffusion-losses.png).
Initial/final comparisons within this smoke use identical noise. Its held-out
seeds (200043/200045) differ from the preceding diagnostic (200042/200043), so
cross-run changes cannot be attributed entirely to the six optimizer updates.

Terminal checkpoint:
`/lus/flare/projects/ModCon/sandeep/prism-docci-alignment-20260922/runs/aligned-regions-joint-smoke-01/connector-diffusion-pilot-step-000006.pt`.

SHA256: `8525b06bae8a61990ba75bce6604b8963c8d1b4e7f1d692c1d813ac6a524e09d`.

The training runner's wrong-caption gaps remain provisional because its global
RNG reset does not verify actual noisy-input/timestep identity. The bounded
diagnostic below restores this completed joint checkpoint and replays the earlier
diagnostic's seeds, separating PRISM from native conditioning with the same
adapted DiT. These results do not support a full-data jump. A later fresh joint
stage must explicitly retain chat prompts (the default is raw), pin the joint
checkpoint digest, and preserve the smoke report as indirect alignment lineage;
it is not a new direct feature-checkpoint restore.

## Completed controlled diagnostic of the six-update checkpoint

Job **8848734**, `conditioning-aligned-joint6-01`, was submitted at 06:01 UTC on
September 22 and completed in **485.92 seconds** (PBS 8:17, exit 0).
It used the unchanged `prism-aligned-joint-smoke` source and verified
terminal joint digest. Independent command/restore review, remote launcher
dry-run and PBS/worker syntax checks passed. Allocation is one node, one process
and one XPU tile for twenty minutes to allow the 47.7 GB checkpoint verification
and restoration plus fourteen generated images.

It reuses the region diagnostic's eight training/eight validation captions,
three fixed flow times and image seeds. Native and PRISM flow controls both use
the adapted DiT. Each image case has original-native CFG1/CFG5 references before
restore, then adapted-native CFG1/CFG5 and three PRISM routes afterward. The main
renderer selects adapted-native references; original-native images remain saved
separately. All 48 actual paired noisy-input/timestep controls, eight archived
source identities, fourteen image hashes and 2,220 frozen tensor hashes passed.
The four original-native CFG1/CFG5 images replay byte-for-byte. All 48 flow IDs,
wrong-caption IDs, seeds and actual noisy-input/timestep hashes also match the
pre-joint diagnostic. The ordinary diagnostic records initial sampling latents;
it does not attest final-latent hashes or each sampling forward's conditioning.

| Route with adapted DiT | Train matched MSE | Held-out matched MSE | Train / held-out wrong-minus-matched MSE | Matched-caption wins, train / held out |
| --- | ---: | ---: | --- | --- |
| Native conditioner | 0.452946 | 0.485011 | 0.025003 / 0.026759 | 24/24 / 24/24 |
| PRISM chat after six updates | 0.456802 | 0.493165 | 0.019792 / 0.014181 | 23/24 / 21/24 |

PRISM held-out matched MSE is effectively flat against the pre-joint diagnostic
(0.493149 → 0.493165), but its mean caption gap fell **11.39%**. The correct-caption
win count increased by one; these threshold counts and mean gaps capture different
properties and neither establishes image quality.

At the same diagnostic seeds, the PRISM animal scene becomes a white, largely
featureless figure with a blue helmet-like shape and still no separate penguin.
The aircraft scene becomes a flat stylized airplane/sign graphic and loses the
second aircraft. Native conditioning on the same adapted DiT retains both animals
and recognizable aircraft, visually close to the original native reference.
See the [matched-seed before/after comparison](../assets/image_generation/2026-09-22-docci-alignment/runs/conditioning-aligned-joint6-01/matched-seed-before-after.png)
and [adapted-DiT comparison](../assets/image_generation/2026-09-22-docci-alignment/runs/conditioning-aligned-joint6-01/conditioning-chat-comparison.png).
This confirms degradation of these PRISM examples after the joint updates;
native stability suggests a connector-related effect but does not isolate it
from an interaction with the adapted DiT.

## Completed frozen connector/DiT component diagnostic

Job **8848800**, `joint-components-01`, completed September 22 in **1,010.69
seconds** (PBS 17:03, exit 0). It crosses both verified connector states (region-aligned and joint-six)
with both DiT states (original and joint-six), alongside native conditioning for
each DiT. All components remain frozen; there are no optimizer updates or saved
weights. The new `prism-joint-components` source snapshot preserves the executed
joint-smoke snapshot. Ten inherited source files and four overlay files passed
checksum verification before submission.

Strict region admission occurs against the original DiT before strict joint-six
restoration. Detached CPU snapshots retain full FP32 connector weights and BF16
DiT runtime tensors, including nonpersistent buffers. Every phase verifies its
complete intended state and the unchanged PRISM/VAE/native state; the original
DiT and aligned connector are restored at exit. This uses its own evidence kind,
`real_checkpoint_joint_component_diagnostic`, rather than treating intentional
component swaps as a normal unchanged-model diagnostic.

Two training and two held-out targets at three flow times provide twelve paired
cases per route, with actual conditioning/noisy-input/timestep hashes checked
across all phases. The two held-out captions generate twelve images across the
six routes at **CFG5**, with the same native negative condition, initial noise,
and 50-step schedule. Actual positive/negative CFG branches and final latent
hashes are recorded. Prefix/content/suffix feature comparisons always use the
captured original caption RMSNorm, separating connector changes from changes in
the adapted normalization layer.

The diagnostic, helper, runner and launcher integration passed **218 tests** and
Ruff. Independent source review found no execution blocker. The separate
metadata collector/renderer passed five test methods, including a full archive
round trip, 32 negative evidence cases and corrupted-checkpoint rejection; its
explicitly synthetic gallery was visually inspected. Remote launcher dry-run
and PBS/worker syntax checks passed. The completed accelerator run passed the
strict collector and standalone renderer audit. The allocation was **twenty capacity
minutes, one node, one process and one XPU tile**.

Collect with `provenance/components/pack_joint_components.py` and
`--source-root .../prism-joint-components --verify-checkpoints` (plural), using
the existing image environment's Python. Preserve its sibling `pack_run.py` and
`render_joint_components.py`. Completed acceptance requires independently
streamed region/joint digests, source report identities, complete phase/flow/image
controls and restored baseline hashes. Keep checkpoint bytes on Aurora. These
factorial comparisons distinguish connector effects from DiT effects or their
interaction before choosing further training.

Both checkpoint digests were independently streamed and verified. All fifteen
executed source identities, twelve paired flow cases across six routes, twelve
images, actual CFG branches/initial and final latents, all four 2,220-tensor phase
audits and final restored baseline hashes passed. Weights remained on Aurora.
Independent review reproduced the artifact audit: all four shared routes match
the twelve relevant prior flow rows exactly, and eight corresponding CFG5 images
replay byte-for-byte.

The controlled images identify the **updated connector as the dominant cause of
degradation in these two cases**. Its combination with the original DiT already
reproduces the malformed white animal/helmet and flat one-aircraft graphic. Keeping
the aligned connector while switching to the adapted DiT retains images visually
close to the pre-joint baseline. Both native references remain similar. This is
a bounded causal comparison; it does not prove that DiT changes have no effect
on other captions or after longer training.

| Held-out route, two captions × three times | Matched flow MSE | Wrong-minus-matched MSE |
| --- | ---: | ---: |
| Native / original DiT | 0.507843 | 0.025626 |
| Native / adapted DiT | 0.507307 | 0.026032 |
| Aligned connector / original DiT | 0.511325 | 0.019883 |
| Aligned connector / adapted DiT | 0.510636 | 0.020113 |
| Updated connector / original DiT | 0.510851 | 0.017436 |
| Updated connector / adapted DiT | 0.510861 | 0.017321 |

All six routes win all six matched-caption comparisons in this small held-out
cohort. The losses and win counts still conceal the visible semantic degradation.
Do not compare these two-caption averages directly to preceding eight-caption
means.

Under the fixed original DiT RMSNorm, the two matched held-out captions show:

| Feature region | Aligned connector MSE | Updated connector MSE | Change |
| --- | ---: | ---: | ---: |
| Prefix | 0.0005794 | 0.0116890 | 20.18× |
| Caption content | 0.0563248 | 0.0627135 | +11.34% |
| Suffix | 0.0053065 | 0.0087664 | +65.2% |

The six flow-training updates therefore moved the connector away from its native
feature alignment, especially at prefix positions. See the
[component comparison](../assets/image_generation/2026-09-22-docci-alignment/runs/joint-components-01/component-comparison.png)
and [feature drift plot](../assets/image_generation/2026-09-22-docci-alignment/runs/joint-components-01/feature-drift.png).
All three feature regions drift, so the factorial does not isolate prefix drift
as the sole cause. Lower LR may preserve conditioning without repairing the
baseline's missing subjects; it is not a full-data gate.
The earlier decision to wait for stronger generation evidence was subsequently
superseded by the user's explicit full-data training instruction.

## Completed lower-connector-learning-rate replay

Job **8848832**, `aligned-regions-joint-lr1e5-smoke-01`, completed September 22
in **559.67 seconds**. The compact archive and terminal checkpoint passed collection
acceptance; large checkpoint bytes remain on Aurora.
It repeats the six-update joint smoke from the same verified region checkpoint,
changing connector LR from **1e-4 to 1e-5**. DiT LR remains **2e-6**. The exact
submitted command differs from job 8848651 only in that learning rate, the new
output directory, and the config path in the unchanged `prism-joint-components`
snapshot. Config bytes and training source identities match the original smoke.
No source snapshot was edited and no new training objective was introduced.

Initialization starts again from warm500 followed by strict schema-2 region
restoration, original FP32 DiT masters and fresh optimizer/sampler/RNG. It does
not initialize from the degraded joint-six checkpoint. The same 32 pairs,
seed 42, batch four/accumulation four produced **96 audited exposures, three per
pair**. Chat prompts, eight/eight initial/final probes, original-native baseline,
50-step CFG5 generation, native negative conditioning and all other settings
remain the same. Both optimizer groups remain trainable; shared clipping and
coupled later updates mean this is a learning-rate intervention, not an assertion
that resulting DiT weights must remain identical.

Independent command review, parser/budget checks, remote source/config hash
checks, launcher dry-run and PBS/worker syntax checks passed. The allocation is
twenty capacity minutes on one node/process/XPU tile, with the same 32-thread CPU
optimizer binding. Source code is unchanged from the 218-test integration.

The verified terminal checkpoint SHA256 is
`117e63f67d309fffdd4d13a7f0ca551e76dc8fe55d556fa25809ae7423716c06`.
Independent audit verified all sixteen archived source files, 1,634 unchanged
frozen tensors, exact 32-by-three exposure, FP32 initialization and finite
updates in both groups. Fixed eight-example train loss fell 0.38% and held-out
loss fell 0.09%; the saved galleries still show incorrect clock time and missing
or unclear held-out subjects. Loss plots, galleries and an interpreted summary
are retained as historical evidence.

Its proposed `joint-components-lr1e5-command-NOT-SUBMITTED.json` follow-up is now
**superseded and must not be automatically submitted**. The user requested that
the larger-data joint experiment proceed instead. The tenfold reduction was a
diagnostic heuristic for preserving the earlier interface, not an established
optimal learning rate for jointly learning a new interface on the full pool.

## Full-data joint training configuration

Job **8848858**, `full-docci-joint-3epochs-01`, was submitted September 22 and was
initially queued in capacity, then started at **07:39:45 UTC**. It runs **1,809 updates**, batch 4 with accumulation
4, for **28,944 image
exposures**: three complete passes through 9,647 pairs plus three examples from
the next pass. Learning rates are **1e-4 for the connector and 2e-6 for the DiT**.
There is no feature-retention loss: the existing image flow-matching loss jointly
updates both modules, allowing their interface to adapt across diverse captions.
The full-pool sampler was simulated before launch: 9,644 examples receive three
exposures and three receive four, with every training ID included. The 100
validation pairs remain excluded from optimization; index hashes and disjoint
IDs were checked directly on Aurora.

The fresh stage restores the independently audited completed joint-six checkpoint
from job 8848651, SHA256
`8525b06bae8a61990ba75bce6604b8963c8d1b4e7f1d692c1d813ac6a524e09d`.
Its report SHA256 is
`b479d9412b50fbc016b5fb7af872eac83445a29bd8f7d7d59bca20f022f13590`.
Full FP32 connector/DiT masters and runtime buffers are restored; optimizer
moments, sampler and RNG start fresh. Alignment provenance is indirect through
that completed joint report. This is not a direct feature-checkpoint restore and
does not use the lower-learning-rate smoke as its initializer.

Chat prompts, seed 42, CFG5, native negative conditioning, 50 diffusion sampling
steps and 256-square resolution are explicit. Fixed 32-example train/validation
probes run initially and every 200 updates; terminal validation uses all 100
held-out pairs. Compare the same 32 validation IDs over time rather than treating
the final 100-example mean as the same cohort. Checkpoints and two-per-split
galleries are saved at 600, 1,200, 1,800 and terminal 1,809 updates, with initial
images for comparison. Original-native baselines belong in a separate diagnostic
because this run begins with an adapted DiT.

The immutable `prism-joint-components` source is unchanged. Independent command
review, current parser/budget checks, source/config/initializer-report identities,
dataset hashes and separation, remote launcher dry-run, PBS/worker syntax and
32-thread CPU optimizer binding checks passed. Existing implementation tests
remain the 218-test integration. Measured update time gives approximately
**5.60 hours of optimizer work**; the **twelve-hour, one-node, one-process,
one-XPU-tile** allocation includes margin for caption variation, restore,
evaluation, sampling and roughly 191 GB of additional checkpoint writes.

Execution evidence is the actual submitted
`provenance/full-docci-joint-3epochs-command.json` plus job scripts/manifests.
The older `full-data-candidate-NOT-SUBMITTED.json` is historical and uses different
initialization/conditioning; do not execute it. Monitor numerical health and
data coverage, retain train/validation curves and evaluate caption-following
images after training. Poor early image quality alone is not a reason to return
to tiny-set gates or stop this user-requested larger-data experiment. Submission
does not establish completed training or improved generation quality.

The September 22 08:01 UTC health check found 97 completed optimizer log rows,
1,552 exposures across 1,552 distinct training IDs, finite recorded metrics and
no missing gradients. Both groups had nonzero gradients and changed runtime
weights. The report had refreshed through step 90; live report and log reads are
not atomic. No periodic evaluation or checkpoint had yet been reached. This is
interim execution health, not terminal acceptance.

The prepared, **not submitted** post-training evaluation is
`provenance/full-docci-joint-conditioning-command-NOT-SUBMITTED.json`, with a
separate command review. It uses the ordinary conditioning diagnostic, which
supports the completed full-stage checkpoint, and retains a mandatory unfilled
terminal SHA256 until independent verification. It plans eight train/eight
held-out captions at three flow times and fourteen images across two held-out
cases, including original/adapted native CFG1 and CFG5 controls. The full-pool
training probes are indices 0–7 and differ from the previous subset diagnostics;
held-out probes and seeds remain comparable. This evaluates the requested
training outcome and does not add a prerequisite for that training.

Collect interim evidence under distinct step/time directories with unique archive
names, preserving the report/log step distinction and leaving terminal acceptance
unset. At completion, independently verify all 100 final validation IDs as well
as the fixed 32-example comparison cohort: the generic collector's automatic
full-validation check does not cover an explicit `final_validation_count=100`.

### First full-data evaluation: step 200 (interim)

The separately preserved `interim-eval0200-20260922T0823Z` collection contains the
step-200 evaluation, a report refreshed through step 210, and 213 complete
optimizer records (3,408 distinct pairs/exposures). The non-atomic collection
occurred at 08:22:56 UTC while training continued. All fourteen recorded source
identities and finite-loss/group-gradient checks passed; terminal acceptance
remains unset. No checkpoint bytes were collected.
An independent metadata audit reproduced the exact sampler, microbatch order
and seeds through step 213, verified unchanged evaluation IDs/seeds and rebuilt
both means from all per-example losses. Both optimizer groups had finite nonzero
gradients and reported runtime changes at every logged update; final frozen-state
verification awaits completion.

| Fixed cohort | Step 0 flow MSE | Step 200 flow MSE | Change |
| --- | ---: | ---: | ---: |
| Training, 32 examples | 0.439716 | 0.438146 | −0.357% |
| Validation, 32 examples | 0.455094 | 0.453521 | −0.346% |

The [interim loss curves](../assets/image_generation/2026-09-22-docci-alignment/runs/full-docci-joint-3epochs-01/interim-eval0200-20260922T0823Z/plots/connector-diffusion-losses.png)
retain fixed IDs and seeds and show stochastic optimizer losses separately.
This is a small preliminary loss change, not generation-quality evidence.
There are no newly trained images yet: the four collected images are from
initialization, and the first trained gallery/checkpoint is planned at step 600.
Updates 2–213 averaged 10.49 seconds. Training continues unchanged.

### Step 400 evaluation (interim)

The separate `interim-eval0400-20260922T0901Z` snapshot was collected at
09:00:39 UTC. Its report records step 420, while its optimizer log contains
425 complete updates and 6,800 distinct training pairs/exposures. Source identity,
finite group gradients and training/validation separation checks passed. Final
frozen-state and checkpoint verification remain pending.
The independent audit reproduced all 425 updates' sampler/microbatch seeds and
confirmed that the earlier 213 log rows, evaluations and initial images were
unchanged. All three evaluations use identical fixed cohorts and seeds.

| Fixed cohort | Step 0 flow MSE | Step 400 flow MSE | Change from step 0 |
| --- | ---: | ---: | ---: |
| Training, 32 examples | 0.439716 | 0.438619 | −0.250% |
| Validation, 32 examples | 0.455094 | 0.452820 | −0.500% |

Training-cohort loss rose 0.108% relative to step 200 while validation loss fell
another 0.155%. These are small interim variations, without new image-quality
evidence. The [updated loss curves](../assets/image_generation/2026-09-22-docci-alignment/runs/full-docci-joint-3epochs-01/interim-eval0400-20260922T0901Z/plots/connector-diffusion-losses.png)
include all three fixed-cohort evaluations and 425 optimizer records. No trained
gallery or checkpoint exists yet; the first is scheduled at step 600. Updates
2–425 averaged 10.50 seconds. The authorized full-data run continues unchanged.

## Completed full-data results

Aurora job **8848858** finished with PBS exit **0**, wall time **5:25:34**, and
19,522.59 seconds of recorded execution. All **1,809 updates** completed with
finite losses and gradients. The run consumed **28,944 exposures across all
9,647 training IDs**: 9,644 IDs appeared three times and three IDs four times.
Both connector and full diffusion groups changed; all **1,634 frozen tensor
hashes** stayed unchanged. Full FP32 master initialization and fresh optimizer,
sampler and RNG lineage remain recorded in the completed report.

The terminal checkpoint stays on Aurora:
`runs/full-docci-joint-3epochs-01/connector-diffusion-pilot-step-001809.pt`,
47,674,885,973 bytes. Its SHA256 was independently streamed and verified:
`14033752ca0bafa947ffc29d80e044900b64e3253260486854205f706cabaee4`.
The completed report SHA256 is
`0da348d96788ab928ceb8d22dc51a6f9ebb84fdb84d7c7c7f04a10773af2114c`.
Compact evidence, all twenty saved image hashes, all fourteen recorded training
source identities and terminal execution checks passed collection acceptance.
The earlier interim directories remain separate.

| Cohort | Initial flow MSE | Final flow MSE | Relative change |
| --- | ---: | ---: | ---: |
| Fixed training, 32 examples | 0.439716 | 0.436508 | −0.73% |
| Fixed validation, same 32 examples | 0.455094 | 0.449456 | −1.24% |
| Complete final validation, 100 examples | Not evaluated initially | 0.451538 | Not a matched comparison |

The terminal fixed validation mean is recomputed from the same 32 per-example
entries within the 100-example evaluation. The
[complete loss curves](../assets/image_generation/2026-09-22-docci-alignment/runs/full-docci-joint-3epochs-01/plots/connector-diffusion-losses.png)
keep that comparison separate from the full-100 marker.

At matched within-run starting noise, the
[held-out gallery](../assets/image_generation/2026-09-22-docci-alignment/runs/full-docci-joint-3epochs-01/validation-comparison-step-001809.png)
now separates a recognizable dog and penguin, and recovers two aircraft on a
runway. The penguin is much too small and lacks its requested blue hard hat;
aircraft colors, placement and background remain inaccurate. These two cases
show useful subject recovery without establishing broad caption fidelity.
The [training gallery](../assets/image_generation/2026-09-22-docci-alignment/runs/full-docci-joint-3epochs-01/train-comparison-step-001809.png)
is mixed: pyramid shapes improve, while the genie becomes more cartoon-like and
its sign text and pose deteriorate. These are caption-only generations; target
images enter the training loss, not PRISM's vision encoder at sampling time.

The bounded diagnostic **8854234**, `full-docci-joint-conditioning-01`, was
submitted at 16:22 UTC and initially queued. It compares the completed PRISM route
to original and adapted native conditioning with actual paired flow-noise/timestep
controls: eight train/eight held-out captions at three times, plus fourteen images
for two held-out cases at fifty sampling steps. The command pins the independently
verified terminal digest. Eighteen immutable source hashes, reviewed command,
remote launcher dry-run and both PBS/worker syntax checks passed; submitted scripts
are byte-identical to the checked dry-run. Allocation is twenty capacity minutes,
one node, one process and one XPU tile. No model was loaded on a login node.
The unchanged ordinary diagnostic launcher does not apply the CPU32 binding
used by the training entry point.

The diagnostic's training targets are the first eight full-pool indices, which
differ from earlier tiny-set cohorts; held-out cases and seeds remain comparable
to the prior ordinary diagnostics. The PRISM negative anchor is untrained and is
only an ablation. The main rendered gallery will use adapted-native references;
original-native images are saved separately. This is evaluation of the authorized
full-data experiment, not a new training prerequisite.
Training-run wrong-caption gaps remain provisional until that diagnostic.
Neither low flow loss nor four saved examples establishes a generation benchmark.

## Completed caption-conditioning evaluation

Job **8854234** completed with PBS exit **0** and runtime **521.15 seconds**.
Strict restoration admitted the full-stage step-1,809 checkpoint and its exact
completed source report. All **48 paired flow cases**, **14 images**, source
identity checks and **2,220 frozen tensor hashes** passed collection acceptance.
The flow diagnostic verifies actual noisy-latent and timestep inputs for its
correct/wrong-caption comparisons. Its ordinary image sampler records initial
noise and image hashes, not every sampling forward or final latent.
The independent audit passed all **53 checks**: all 24 held-out flow inputs
exactly replay both prior diagnostics, and all four original-native reference
images replay byte-for-byte.

The compact archive SHA256 is
`ea17ef2f682913e83e99d068a6d8c05c1e44f0855e5c053681f8ad4f28f631d6`;
the completed diagnostic report SHA256 is
`896ffae828c792b6e3735b9a2b1efff37620ffe6e56800401b56bbec6be83606`.

The held-out cohort is eight captions at three fixed flow times, giving 24 paired
cases. Both final conditioning routes below use the same adapted DiT. The
joint-six column is the matched pre-full-training diagnostic, with its earlier DiT.

| Held-out measure | PRISM before full training | PRISM after full training | Native conditioning / final DiT |
| --- | ---: | ---: | ---: |
| Matched-caption flow MSE | 0.493165 | 0.482828 | 0.481150 |
| Correct-caption lower-loss cases | 21/24 | 22/24 | 24/24 |
| Wrong-minus-correct caption MSE | 0.014181 | 0.025593 | 0.029362 |

PRISM's matched loss decreased **2.10%** on these matched diagnostic cases and
is **0.35% above** native conditioning on the final DiT. Its caption gap increased
**80.47%** from the joint-six diagnostic and reaches **87.16%** of the native gap.
These are bounded conditioning measurements, not image accuracy. The two remaining
PRISM non-wins concern the airplane caption at flow times 0.1 and 0.9. Training
diagnostic probes now use full-pool indices 0–7; their averages are not directly
comparable to the earlier tiny-set diagnostic cohorts.

The [matched-seed before/after comparison](../assets/image_generation/2026-09-22-docci-alignment/runs/full-docci-joint-conditioning-01/matched-seed-before-after.png)
shows a distinct dog and penguin and recognizable aircraft after full training,
instead of the earlier merged animal and cartoon aircraft scene. The dog has a
cone-shaped hat; the undersized penguin lacks its helmet; aircraft colors and
runway placement remain wrong. Adapted-native CFG5 preserves both helmets better.
CFG1 also recovers more scene structure but subjects remain distorted. Root and
independent visual inspection agree that subject recovery improves in these two
cases while attribute fidelity remains limited.

The [complete guidance comparison](../assets/image_generation/2026-09-22-docci-alignment/runs/full-docci-joint-conditioning-01/conditioning-chat-comparison.png)
keeps CFG1 and CFG5 separate. Original-native controls are saved separately from
the main gallery's adapted-native references. The PRISM-negative CFG5 route is an
untrained empty-chat ablation. Both connector and DiT changed during full training,
so these comparisons do not isolate each component's contribution. Two image
cases do not establish broad generation quality.

The authorized full-data training and bounded evaluation are complete. Reports,
audits, loss plots and image comparisons are saved; no additional training,
dataset download, commit or push was performed.

## Follow-up and locations

The task heartbeat **`continue-prism-docci-alignment-and-training`** is no longer
needed after the completed training and evaluation; its scheduled follow-up is
stopped. Additional training or broader benchmarking requires new user direction.
PRISM unfreezing, new datasets, commits and pushes remain outside this authorization.

Aurora root:
`/lus/flare/projects/ModCon/sandeep/prism-docci-alignment-20260922`.

Local artifact bundle:
`docs/assets/image_generation/2026-09-22-docci-alignment/`.

Implementation and protocol:
[staged adaptation](../modalities/image_decoder/docci_alignment_stages.md),
[training runner](../../tools/train_prism_image_diffusion.py), and
[conditioning diagnostic](../../tools/diagnose_prism_image_conditioning.py).
Large model checkpoints remain on Aurora. No new commits or pushes were made.
