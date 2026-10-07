
import argparse
import json
import logging
import os
import sys

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add scripts directory to path to import prepare_pixmo_dataset
# Assuming this script is in src/data/ and scripts/ is in root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(os.path.join(REPO_ROOT, "scripts"))

try:
    from prepare_pixmo_dataset import prepare_dataset
except ImportError:
    logger.error("Could not import prepare_pixmo_dataset. Make sure scripts/ directory is accessible.")
    sys.exit(1)

def load_config():
    from src.site_paths import expand_tree

    config_path = os.path.join(os.path.dirname(__file__), "datasets_config.json")
    with open(config_path) as f:
        # Shipped dataset paths name site roots as ${PRISM_*} rather than one
        # user's directories; resolve them at load (same as DatasetManager).
        return expand_tree(json.load(f))

def download_pixmo(limit=None, workers=128, verify_ssl=False):
    config = load_config()
    zone_a_datasets = config.get("zone_a", {}).get("datasets", {})
    
    # Target datasets
    targets = ["pixmo_points", "pixmo_count"]
    
    for target in targets:
        if target not in zone_a_datasets:
            logger.warning(f"Dataset {target} not found in config.")
            continue
            
        ds_config = zone_a_datasets[target]
        hf_id = ds_config.get("hf_id")
        # Use configured available path or default
        output_dir = ds_config.get("local_path")
        
        # Heuristic: if the configured path still contains an unresolved
        # "<user>"/"<project>" placeholder (see datasets_config.json), fall
        # back to a relative ./data/ dir instead of that literal placeholder.
        if "<user>" in output_dir or "<project>" in output_dir:
            # e.g. /flare/<project>/<user>/PRISM/data/... -> ./data/...
            rel_path = output_dir.split("/data/")[-1]
            output_dir = os.path.join(REPO_ROOT, "data", rel_path)
            
        logger.info(f"Processing {target} ({hf_id}) -> {output_dir}")
        
        try:
             prepare_dataset(
                input_arrow_dir=None,
                output_dir=output_dir,
                cache_image_dirs=[], 
                workers=workers,
                hf_id=hf_id,
                limit=limit,
                split=ds_config.get("split", "train"),
                verify_ssl=verify_ssl
            )
        except Exception as e:
            logger.error(f"Failed to process {target}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="Limit samples for testing")
    parser.add_argument("--workers", type=int, default=128, help="Number of workers for hydration")
    parser.add_argument("--verify-ssl", action="store_true", help="Enable SSL verification")
    parser.add_argument("--no-verify-ssl", action="store_false", dest="verify_ssl", help="Disable SSL verification (default)")
    parser.set_defaults(verify_ssl=False)
    
    args = parser.parse_args()
    
    download_pixmo(limit=args.limit, workers=args.workers, verify_ssl=args.verify_ssl)
