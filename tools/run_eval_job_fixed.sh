#!/bin/bash
#PBS -l select=1
#PBS -l walltime=00:30:00
#PBS -q debug
#PBS -A ModCon
#PBS -N PRISM-EVAL-FIXED
#PBS -l filesystems=home:flare

cd $PBS_O_WORKDIR

# --- Environment Setup (Fixed) ---
# 1. Load System Frameworks (Provides PyTorch on XPU)
module load frameworks/2025.2.0

# 2. Add User Site to PYTHONPATH (Provides transformers, accelerate, etc. installed via pip --user)
# Note: We hardcode python3.10 path matching the module version
export PYTHONPATH="$HOME/.local/lib/python3.10/site-packages:$PYTHONPATH"
export PYTHONPATH="$PBS_O_WORKDIR:$PYTHONPATH"

# Force Offline Mode to prevent hangs on compute nodes
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "Environment ready."
echo "Python: $(which python3)"
python3 -c "import torch; print(f'Torch: {torch.__file__}')"
python3 -c "import transformers; print(f'Transformers: {transformers.__file__}')"

# --- Run Evaluator ---
CHECKPOINT="/flare/ModCon/ngetty/BaseMM_PRISM/exported/prism-olmo1b-image-step500/model.safetensors"

echo "Running Evaluator..."
# Use inspect_train mode with limit to verify data loading + inference
python3 tools/universal_evaluator.py \
    --checkpoint "$CHECKPOINT" \
    --mode verify_image \
    --backbone "allenai/OLMo-1B-0724-hf" \
    --limit 10 \
    --visualize \
    --viz_dir "viz_output/eval_1b_batch"

echo "Done."
