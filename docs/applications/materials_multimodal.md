# Materials property prediction with text + crystal graphs

This example learns a scalar property by joining three files on Materials Project ID:

- `Materials/targets_material.csv` supplies the target (by default `band_gap`);
- `Materials/text/<id>.txt` supplies the material description;
- `Materials/bulk_data_full/<id>.cif` supplies the periodic crystal structure.

The data and regression integration live in `src/materials`; the reusable periodic crystal encoder
lives in `src/encoders/crystal_graph.py`. It encodes a crystal into a fixed number of graph tokens,
projects them into Qwen's embedding space with PRISM's modality projector, prepends them to the
description tokens, and regresses the property from Qwen's final valid hidden state. The graph
encoder uses periodic radius edges with Gaussian distance features.
By default, the graph retains the 16 closest periodic neighbors
within 5 Å of each atom; tune this with `--cutoff` and `--max-neighbors`.

## Setup and smoke run

Install the add-on CIF parser in the same environment as PRISM:

```bash
pip install -r requirements/materials.txt
```

Run a small PRISM end-to-end experiment first (Qwen must already be staged locally):

```bash
python applications/material_science/train_materials_prism.py \
  --text-model Qwen/Qwen3-0.6B \
  --target band_gap \
  --max-samples 1000 \
  --epochs 3 \
  --batch-size 8 \
  --output-dir outputs/materials-prism-bandgap-smoke
```

The first epoch parses CIFs and writes graph tensors below the run directory. Later epochs reuse that
cache. On a shared filesystem, caching all structures can require many small files; pass
`--graph-cache-dir /tmp/$USER/material-graphs` to use node-local storage, or `--no-graph-cache` to
disable caching. Use `--num-workers` only after confirming that concurrent CIF parsing improves
throughput on the target machine.

On Polaris, `bash applications/material_science/run_materials_polaris.sh` submits the PRISM implementation. For a short validation
job, use `QUEUE=debug WALLTIME=01:00:00 EPOCHS=3 MAX_SAMPLES=1000 RUN_NOTE=smoke bash
applications/material_science/run_materials_polaris.sh`. The default PRISM batch size is 8.
The launcher captures the active environment's absolute Python path at submission time and passes
it through PBS rather than relying on the compute job's reconstructed `PATH`. Set
`MATERIALS_PYTHON=/absolute/path/to/environment/bin/python` to select a different environment
explicitly. It does not activate or fall back to the repository's `.venv-polaris` environment.

The current `conda/2025-09-28` modulefile references retired CPE module versions, so the launcher
leaves the selected shell environment in place and supplies the current Polaris CUDA/MPI runtime
paths directly. If its dependency preflight reports that `pymatgen` is missing, install it into the
selected Python environment with
`$MATERIALS_PYTHON -m pip install -r requirements/materials.txt` and resubmit.

## Required comparisons

Use exactly the same seed and split to measure whether joint learning helps. For PRISM:

```bash
python applications/material_science/train_materials_prism.py --mode text  --seed 17 --output-dir outputs/materials-prism-bandgap-text
python applications/material_science/train_materials_prism.py --mode graph --seed 17 --output-dir outputs/materials-prism-bandgap-graph
python applications/material_science/train_materials_prism.py --mode joint --seed 17 --output-dir outputs/materials-prism-bandgap-joint
```

The PRISM path always uses the selected model as its LLM backbone. It is frozen by default:

```bash
python applications/material_science/train_materials_prism.py \
  --mode joint \
  --text-model Qwen/Qwen3-0.6B \
  --output-dir outputs/materials-prism-bandgap-qwen
```

Add `--train-text-backbone` to fine-tune the whole transformer, which requires substantially more
accelerator memory. On Polaris, add `TRAIN_TEXT_BACKBONE=1`. The model and tokenizer
must already be available in the Hugging Face cache if compute-node network access is disabled. The
fine-tuned backbone uses a separate `1e-5` learning rate by default; override it with
`--text-learning-rate` locally or `TEXT_LEARNING_RATE` in the Polaris launcher.

Stage Qwen once from a Polaris login node before submitting the job:

```bash
python applications/material_science/stage_materials_text_model.py \
  --model Qwen/Qwen3-0.6B \
  --cache-dir /eagle/projects/ModCon/$USER/huggingface/hub
```

The launcher points `HF_HUB_CACHE` at that shared directory and enables Hugging Face offline mode on
the compute node. Override the paths with `MATERIALS_HF_HOME` and `MATERIALS_HF_HUB_CACHE` if the
shared cache lives elsewhere. Before training, the job verifies that the requested snapshot is
complete and prints the staging command if it is missing.

Each directory contains:

- `best.pt`: weights selected by validation MAE (trainable PRISM adapters/head and, when enabled,
  the trainable backbone weights);
- `run.json`: arguments, exact split IDs, train-target normalization, vocabulary, learning curves,
  and final test metrics;
- `test_predictions.csv`: material-level targets and predictions;
- `graph_cache/`: reusable parsed graphs unless disabled.

MAE and RMSE are reported in the target's original units (eV for band gap), along with R². The split
is a deterministic hash of the material ID, and both vocabulary and target normalization are fit only
on training examples.

## Interpretation caveat

Material descriptions may explicitly discuss electronic properties. That can be useful for a
multimodal predictor, but it may also make band-gap evaluation easier than a structure-only discovery
setting. Inspect the descriptions for numeric target leakage. A useful follow-up is to redact explicit
band-gap values or compare against text generated only from composition and symmetry information.
