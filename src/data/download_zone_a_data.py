import json
import os
from io import BytesIO

import requests
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


def download_cc3m_subset(output_dir, num_samples=500):
    print(f"Downloading CC3M subset ({num_samples} samples) to {output_dir}...")

    # Create directories
    img_dir = os.path.join(output_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    # Load Conceptual Captions (streaming)
    # 'google-research-datasets/conceptual_captions'
    try:
        dataset = load_dataset(
            "google-research-datasets/conceptual_captions", split="train", streaming=True
        )
    except Exception as e:
        print(f"Error loading CC3M: {e}")
        return

    data_manifest = []

    count = 0
    for item in tqdm(dataset):
        if count >= num_samples:
            break

        caption = item["caption"]
        image_url = item["image_url"]

        try:
            # Download Image
            response = requests.get(image_url, timeout=5)
            if response.status_code != 200:
                continue

            image = Image.open(BytesIO(response.content)).convert("RGB")

            # Save Image
            img_filename = f"{count:05d}.jpg"
            img_path = os.path.join(img_dir, img_filename)
            image.save(img_path)

            # Add to manifest
            # Use absolute path for safety
            abs_img_path = os.path.abspath(img_path)

            data_manifest.append({"id": count, "image_path": abs_img_path, "text": caption})

            count += 1
        except Exception:
            # print(f"Failed to download {image_url}: {e}")
            continue

    # Save Manifest
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(data_manifest, f, indent=2)

    print(f"Saved {count} pairs to {output_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    # Output to data/zone_a/cc3m (relative to src/data/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "../../data", "zone_a", "cc3m")
    download_cc3m_subset(output_dir, num_samples=100)
