# PRISM Evaluation Framework

PRISM includes a unified evaluation tool, `tools/universal_evaluator.py`, designed to verify the model pipeline at every stage—from data loading to final quantitative benchmarking.

## Universal Evaluator (`tools/universal_evaluator.py`)

This tool consolidates inspection, visualization, and evaluation into a single CLI.

### Usage
```bash
python tools/universal_evaluator.py --mode [MODE] --checkpoint [PATH] [OPTIONS]
```

### Modes

#### 1. Training Data Inspection (`inspect_train`)
Verifies exactly what the model "sees" during training. It loads the `StreamingMultimodalDataset`, applies preprocessing, and yields batches.

*   **Goal**: Debug data loading, fallback logic, and tensor shapes.
*   **Visualization**: Generates composite plots showing all active modalities for each sample.
*   **Arguments**:
    *   `--limit N`: Inspect N random samples.
    *   `--visualize`: Enable plot generation.
    *   `--viz_dir DIR`: Directory to save plots.
    *   `--exhaustive`: (Optional) Iterate through **every** configured dataset individually instead of random sampling.

**Example**:
```bash
python tools/universal_evaluator.py \
    --mode inspect_train \
    --limit 10 \
    --visualize \
    --viz_dir visualization/debug
```

**Composite Visualization**:
Each sample produces a 1x5 grid image (`sample_X_composite.png`) displaying:
*   **Image**: Denormalized RGB image.
*   **Graph**: 2D projection of nodes/edges.
*   **Table**: Token ID heatmap.
*   **Time Series**: Signal plot.
*   **Geometry**: Mid-slice view or point cloud projection.
*   *Inactive modalities are explicitly marked as "BLANK".*

---

#### 2. Qualitative Evaluation (`inspect_eval`)
Runs inference on a single representative example from each modality's evaluation set.

*   **Goal**: Quick "sanity check" that the model generates coherent text and processes inputs without crashing.
*   **Evaluators**:
    *   Graph (ChEBI-20)
    *   Time (SciTS)
    *   Table (Spider)
    *   Vision (VQAv2)
    *   Geometry (MatBench)

**Example**:
```bash
python tools/universal_evaluator.py \
    --checkpoint outputs/checkpoint/model.safetensors \
    --mode inspect_eval
```

---

#### 3. Full Quantitative Evaluation (`run_eval`)
Runs the full evaluation suite across all benchmarks.

*   **Goal**: Compute metrics (BLEU, Accuracy, MAE) for model checkpoints.
*   **Arguments**:
    *   `--limit N`: Number of samples per task (default 100). Use smaller N for fast verification.

**Example**:
```bash
python tools/universal_evaluator.py \
    --checkpoint outputs/checkpoint/model.safetensors \
    --mode run_eval \
    --limit 100
```

### Supported Benchmarks

| Modality | Dataset | Metric | Status |
|----------|---------|--------|--------|
| **Graph** | ChEBI-20 | BLEU | ✅ Active |
| **Time** | SciTS | Accuracy | ✅ Active |
| **Table** | Spider | Accuracy | ✅ Active |
| **Geometry** | MatBench | MAE | ✅ Active |
| **Vision** | VQAv2 | Accuracy | ✅ Active |

---

#### 4. Time Series Verification (`verify_timeseries_interleave` / `verify_timeseries_scits`)
Runs generation on time series samples and scores predictions against ground truth.

*   **Goal**: Verify the time series encoder/projector end-to-end.
*   **Default split**: Training data via `StreamingMultimodalDataset` (`verify_timeseries_interleave`).
*   **Validation split**: SciTS val shards (Q/A format from `*.tar` WebDataset files, `verify_timeseries_scits`).

**Example — training data**:
```bash
python tools/universal_evaluator.py \
    --checkpoint outputs/checkpoint/model.safetensors \
    --mode verify_timeseries_interleave \
    --limit 20
```

**Example — SciTS validation shards**:
```bash
export PYTHONNOUSERSITE=1 && unset PYTHONPATH && PRISM_VAL_SHARDS_DIR=/flare/ModCon/pemami/data/SciTS-processed/val_shards \
python tools/universal_evaluator.py \
    --checkpoint <YOUR_CHECKPOINT>/model.safetensors \
    --mode verify_timeseries_scits \
    --limit 2000 \
    --backbone allenai/OLMo-1B-0724-hf
```

Set `PRISM_VAL_SHARDS_DIR` to any directory containing `*.tar` SciTS shards (each sample must have `.ts.npy` and `.text` files). The `.text` file should be in `Question: ...\nAnswer: ...` format.
