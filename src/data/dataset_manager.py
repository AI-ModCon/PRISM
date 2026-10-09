import json
import os
from typing import Any

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "datasets_config.json")


class DatasetManager:
    def __init__(self, config_path: str = CONFIG_PATH):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config not found at {config_path}")

        from src.site_paths import expand_tree

        with open(config_path) as f:
            # Shipped dataset paths name site roots as ${PRISM_*} rather than
            # one user's directories; resolve them at load.
            self.full_config = expand_tree(json.load(f))

    def get_zone_config(self, zone: str) -> dict[str, Any]:
        """Returns the configuration for a specific zone (e.g., 'zone_a')."""
        if zone not in self.full_config:
            raise ValueError(f"Zone '{zone}' not found in config.")
        return self.full_config[zone]

    def get_dataset_info(self, zone: str, modality: str) -> dict[str, Any]:
        zone_cfg = self.get_zone_config(zone)
        datasets = zone_cfg.get("datasets", {})
        return datasets.get(modality)

    def verify_readiness(self, zone: str = "zone_a") -> dict[str, str]:
        """Checks availability of datasets in the zone."""
        results = {}
        datasets = self.get_zone_config(zone).get("datasets", {})

        print(f"Verifying Datasets for {zone}...")
        for modality, info in datasets.items():
            run_check = True
            if info.get("skip", False):
                status = f"Skipped ({info.get('skip_reason', 'User Request')})"
                run_check = False

            if run_check:
                try:
                    # 1. Local Check
                    local_path = info.get("local_path")
                    if local_path:
                        abs_path = os.path.abspath(
                            os.path.join(os.path.dirname(__file__), "../../..", local_path)
                        )
                        if os.path.exists(abs_path):
                            status = f"Ready (Local: {local_path})"
                        else:
                            status = f"Missing Local ({local_path})"
                    else:
                        # 2. Remote Check (Simple config check)
                        info["hf_id"]
                        status = "Ready (Remote ConfigURED)"
                except Exception as e:
                    status = f"Error ({str(e)})"

            results[modality] = status
            print(f"  [{modality.upper()}] {info['name']}: {status}")

        return results

    def list_datasets(self, zone: str = "zone_a"):
        """Prints extracted paths and details."""
        datasets = self.get_zone_config(zone).get("datasets", {})
        print(f"\nDataset Manifest for {zone.upper()}:")
        print(f"{'Modality':<15} | {'Dataset Name':<20} | {'HF ID':<40} | {'Stream'}")
        print("-" * 90)
        for modality, info in datasets.items():
            print(f"{modality:<15} | {info['name']:<20} | {info['hf_id']:<40} | {info['stream']}")


if __name__ == "__main__":
    # Script usage as requested
    manager = DatasetManager()

    for zone in manager.full_config.keys():
        print(f"\n--- {zone.upper()} ---")
        manager.list_datasets(zone)
        print("")
        manager.verify_readiness(zone)
