# 1. Environment

Training uses Qwen3-0.6B, PRISM's Intern-S2 Preview 397B time-series encoder,
and a newly initialized linear quantile head. The 397B label identifies the
source model; PRISM extracts only its time-series encoder weights.

## Build the Aurora environment

The Intern-S2 implementation requires the `--intern-s2` environment variant,
which installs Transformers 5.2.0. On Aurora:

```bash
module purge
module use /opt/aurora/26.26.0/modulefiles
module load oneapi/release/2025.3.1
module load mpich/prd/5.0.0.aurora_test.51a9474 
module load frameworks/2025.3.1
export VENV_PATH=/flare/GenesisHackathonOct26/$USER/prism-envs/py3.12-intern-s2
bash tools/build_aurora_env.sh --intern-s2
source "$VENV_PATH/bin/activate"
```

Keep `VENV_PATH` exported for the launcher, or put it in your ignored local
`.env`. The launcher below uses `--use-shared-venv`. Load the pinned module
before activating the environment. This Transformers variant is not compatible
with vLLM's declared version range; vLLM is not needed for this tutorial. See
[Aurora operations](../platforms/aurora_operations.md) for module details.

## Set model paths and access

Choose shared cache locations visible from compute nodes. Set `HF_TOKEN` in
your local ignored `.env` or through your site's secret manager; the extraction
script reads it from the environment or `.env`. For the Genesis hackathon, the project field can be `GenesisHackathonOct26`

```bash
export HF_HOME=/flare/GenesisHackathonOct26/$USER/huggingface
```

Make sure the public Qwen3-0.6B checkpoint is also present in the shared
Hugging Face cache before launching a job that cannot access the network.

## Extract the time-series encoder

The extractor defaults to the 35B repository, so specify the 397B repository
explicitly. `--encoder-only` avoids downloading Intern-S2's separate native
forecaster, which this PRISM preset does not use.

```bash
python scripts/extract_intern_s2_timeseries.py \
    --repo-id internlm/Intern-S2-Preview-397B \
    --output-dir "$HF_HOME/intern-s2-preview-397b-timeseries" \
    --code-dir "$HF_HOME/intern-s2-preview-397b-timeseries/source" \
    --encoder-only

test -s "$HF_HOME/intern-s2-preview-397b-timeseries/model.safetensors"
```

The extractor writes upstream code files into `--code-dir`. Keeping that path
outside the checkout avoids overwriting PRISM's locally maintained,
vendored encoder implementation.

Next: [prepare the forecast data](data.md).
