import os

from huggingface_hub import snapshot_download

model_id = "allenai/OLMo-1B-0724-hf"
# Honour the caller's HF cache; fall back to huggingface_hub's own default
# (None) rather than hard-coding one machine's scratch path.
cache_dir = os.environ.get("HF_HOME") or None

print(f"Downloading {model_id} to {cache_dir}...")
try:
    path = snapshot_download(repo_id=model_id, cache_dir=cache_dir)
    print(f"Successfully downloaded to {path}")
except Exception as e:
    print(f"Failed to download: {e}")
    # Try generic 1B if this specific one fails
    try:
        fallback_id = "allenai/OLMo-1B-hf"
        print(f"Retrying with {fallback_id}...")
        path = snapshot_download(repo_id=fallback_id, cache_dir=cache_dir)
        print(f"Successfully downloaded to {path}")
    except Exception as e2:
        print(f"Failed fallback: {e2}")
