# Tutorial: Jointly train PRISM on material descriptions and crystal structures

This tutorial shows how to train a PRISM model to predict a material property from two inputs:

- A CIF file describing the periodic crystal structure.
- A natural language text file describing the same material.

The target property is stored in a CSV file. PRISM encodes the crystal and text, fuses their token sequences with a language model, and learns a regression head to predict the target.

## 1. Arrange the data

The current materials training command expects the following layout:

```text
Materials/
├── targets_material.csv
├── text/
│   ├── mp-1.txt
│   ├── mp-2.txt
│   └── ...
└── bulk_data_full/
    ├── mp-1.cif
    ├── mp-2.cif
    └── ...
```

The material ID joins all three sources. The CSV must have an `id` column and one or more numeric property columns. For example:

```csv
id,band_gap,formation_energy
mp-1,1.42,-2.31
mp-2,0.00,-1.87
```

For each CSV ID, provide a text file named `<id>.txt` and a structure file named `<id>.cif` in the directories above. Text files should contain the natural language descriptions. Extra CSV columns are allowed; select the property to train on with `--target`.

The loader uses rows for which both files exist. It skips rows with missing or nonnumeric targets, and fails if an ID has conflicting duplicate target values. Keep IDs consistent across filenames and the CSV.

## 2. Install the materials dependency

Run commands from the repository root in the Python environment used for PRISM:

```bash
pip install -r requirements/base.txt
pip install -r requirements/materials.txt
```

The materials add-on installs `pymatgen`, which PRISM uses to parse CIF files. The selected Hugging Face model and tokenizer must also be accessible to the environment.

## 3. Run a small joint-training example

Start with a small run to confirm the files, model, and compute environment work together:

```bash
python applications/material_science/train_materials_prism.py \
  --materials-dir Materials \
  --target band_gap \
  --text-model Qwen/Qwen3-0.6B \
  --mode joint \
  --max-samples 1000 \
  --epochs 3 \
  --batch-size 8 \
  --output-dir outputs/materials-prism-band-gap-smoke
```

Replace `Materials` with the path to your data directory and `band_gap` with a numeric column in your CSV. `--max-samples` is useful for smoke runs; omit it to use every complete record. The default device is selected automatically (CUDA, XPU, or CPU); set `--device cuda` or another PyTorch device explicitly if needed.

For a full run, omit `--max-samples` and choose `--epochs`, `--batch-size`, and `--output-dir` for your dataset and hardware. The default configuration freezes the language model backbone and trains the graph encoder, graph projector, and regression head. Add `--train-text-backbone` to fine-tune the whole language model; this uses substantially more accelerator memory.

### Useful data and graph options

- `--max-text-length 192` sets the maximum number of tokenizer tokens retained from each description.
- `--cutoff 5.0` sets the graph neighbor radius in angstroms.
- `--max-neighbors 16` caps the number of closest periodic neighbors retained per atom.
- `--graph-tokens 8` sets the number of learned graph summary tokens per crystal.
- `--seed 17` sets the deterministic data split and random seed.
- `--val-fraction 0.1` and `--test-fraction 0.1` set the validation and test proportions.
- `--num-workers 0` controls data-loader worker processes. Increase only if concurrent CIF parsing helps on your system.

Parsed crystal graphs are cached under `<output-dir>/graph_cache` by default. To put the cache on faster local storage, pass `--graph-cache-dir /path/to/cache`. Use `--no-graph-cache` to disable it.

## 4. Make the language model available on offline compute nodes

If the training machine cannot download Hugging Face models, stage the model from a machine with access. For example:

```bash
python applications/material_science/stage_materials_text_model.py \
  --model Qwen/Qwen3-0.6B \
  --cache-dir /path/to/shared/huggingface/hub
```

Then point Hugging Face at that cache when launching training, according to your cluster's setup. The tokenizer and model name passed to `--text-model` must match the staged model. On Polaris, see [the materials guide](applications/materials_multimodal.md) for the PBS launcher and cache environment settings.

## 5. What PRISM does with the two inputs

For each material, the data loader tokenizes the description and parses the CIF into atoms, periodic neighbor edges, and interatomic distances. The graph encoder embeds atomic numbers and passes distance-aware messages along the edges. Learned queries summarize each variable-size crystal into a fixed number of graph tokens. A PRISM modality projector maps these graph tokens to the language model's embedding dimension.

The text token IDs are embedded by the selected language model's own embedding table. PRISM concatenates the projected graph tokens before the text tokens and passes the combined sequence and its attention mask through the language model decoder. The decoder's self-attention layers can combine information across both modalities. A regression head reads the final valid hidden state and predicts the selected scalar property.

During training, the target is normalized using statistics from the training split. The model is optimized with smooth L1 loss. At evaluation, predictions and error metrics are converted back to the original target units.

## 6. Inspect the results

The output directory contains:

- `best.pt`: trainable model weights from the epoch with the best validation MAE.
- `run.json`: command-line arguments, exact train/validation/test IDs, target normalization, training history, and test metrics.
- `test_predictions.csv`: one row per test material, with its ID, true target, and predicted target.
- `graph_cache/`: cached parsed structures, unless caching was disabled or redirected.

The terminal reports validation metrics each epoch and final test metrics. MAE is the average absolute prediction error; RMSE weights larger errors more heavily; R² compares predictions against a constant-mean baseline. MAE and RMSE use the target's original units.

## 7. Run on your full dataset

After the smoke run completes and the outputs look sensible, train on all complete pairs. For example:

```bash
python applications/material_science/train_materials_prism.py \
  --materials-dir Materials \
  --target band_gap \
  --text-model Qwen/Qwen3-0.6B \
  --mode joint \
  --epochs 30 \
  --batch-size 8 \
  --seed 17 \
  --output-dir outputs/materials-prism-band-gap
```

The split is deterministic for a given seed and material ID. Keep the run metadata with the checkpoint so you can identify the exact data split and training configuration used.
