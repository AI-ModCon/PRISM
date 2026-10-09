#!/bin/bash
set -eo pipefail
module use /soft/modulefiles
module load frameworks/2025.3.1
set -u
source /lus/flare/projects/ModCon/sandeep/prism-image-smoke-20260921/.venv-image/bin/activate
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export ZE_FLAT_DEVICE_HIERARCHY=FLAT ZE_AFFINITY_MASK=0
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
unset SYCL_DEVICE_FILTER
export PYTHONPATH=/lus/flare/projects/ModCon/sandeep/prism-docci-alignment-20260922/prism-alignment-validation:/lus/flare/projects/ModCon/sandeep/prism-image-smoke-20260921/OmniGen2:${PYTHONPATH:-}

cd /lus/flare/projects/ModCon/sandeep/prism-docci-alignment-20260922/prism-alignment-validation
/lus/flare/projects/ModCon/sandeep/prism-image-smoke-20260921/.venv-image/bin/python -u -c 'import torch; n=torch.xpu.device_count(); print({'"'"'xpu_tiles'"'"':n,'"'"'torch'"'"':str(torch.__version__)},flush=True); assert n==1'
/lus/flare/projects/ModCon/sandeep/prism-image-smoke-20260921/.venv-image/bin/python -u /lus/flare/projects/ModCon/sandeep/prism-docci-alignment-20260922/prism-alignment-validation/tools/diagnose_prism_image_conditioning.py --model-config /lus/flare/projects/ModCon/sandeep/prism-docci-alignment-20260922/prism-alignment-validation/src/conf/image_generation/qwen3_1_7b_prism_harness_omnigen2.json --checkpoint /lus/flare/projects/AuroraGPT/sww/prism_outputs/CODEX_QWEN3_1P7B_SIGLIP_CLEAN_GSHUFV1_INVSQRTLR_TRAIN25K_BS4_HSDP_FULLSHARD_16N_R1/checkpoints/step_19750/model.safetensors --tokenizer /lus/flare/projects/ModCon/sww/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e --source-processor /lus/flare/projects/ModCon/sww/huggingface/hub/models--google--siglip2-base-patch16-224/snapshots/75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2 --train-index /lus/flare/projects/ModCon/sandeep/prism-docci-qwen3-1p7b-20260921/webdataset/train.jsonl --validation-index /lus/flare/projects/ModCon/sandeep/prism-docci-qwen3-1p7b-20260921/webdataset/validation.jsonl --repeatability-report /lus/flare/projects/ModCon/sandeep/prism-docci-joint-20260921/runs/numerical-01/manifest.json --connector-checkpoint /lus/flare/projects/ModCon/sandeep/prism-docci-qwen3-1p7b-20260921/runs/pilot-500-01/connector-pilot-step-000500.pt --connector-checkpoint-sha256 e84b3550304f4cebe56b394e1e1f65dfc333bd9c6ed8e9adbb240c6694c4387f --expected-connector-step 500 --height 256 --width 256 --expected-parent-tensors 526 --device xpu --dtype bfloat16 --attention-backend math --deterministic --output-dir /lus/flare/projects/ModCon/sandeep/prism-docci-alignment-20260922/runs/conditioning-aligned1000-01 --train-probe-count 8 --validation-probe-count 8 --sample-count 2 --sampling-steps 50 --prism-formats chat --alignment-checkpoint /lus/flare/projects/ModCon/sandeep/prism-docci-alignment-20260922/runs/feature-alignment32-1000-01/connector-feature-alignment-step-001000.pt --alignment-checkpoint-sha256 431938ef0fe99e480a28a71fd8eb716d067050cdde7388dd0ed33a2af794a642
