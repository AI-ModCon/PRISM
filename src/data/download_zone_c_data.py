import json
import os

from datasets import load_dataset
from tqdm import tqdm


def download_preference_data(output_dir, num_samples=100):
    print(f"Downloading UltraFeedback subset ({num_samples} samples) to {output_dir}...")

    os.makedirs(output_dir, exist_ok=True)

    # Load UltraFeedback (Binarized)
    # This dataset contains 'prompt', 'chosen', 'rejected' fields.
    try:
        dataset = load_dataset(
            "HuggingFaceH4/ultrafeedback_binarized", split="train_prefs", streaming=True
        )
    except Exception as e:
        print(f"Error loading UltraFeedback: {e}")
        return

    data_manifest = []

    count = 0
    for item in tqdm(dataset):
        if count >= num_samples:
            break

        try:
            # UltraFeedback format:
            # prompt: str
            # chosen: list[dict] (role, content)
            # rejected: list[dict] (role, content)

            prompt = item["prompt"]
            chosen = item["chosen"]
            rejected = item["rejected"]

            # Extract text content
            # Assuming standard chat format
            # chosen/rejected are lists of messages. We want the assistant response.
            # Usually prompt is user, chosen/rejected are assistant responses.

            # For simplicity, we store the full conversation or just the response.
            # DPO usually needs (prompt, chosen, rejected).

            # Let's format as:
            # prompt: User input
            # chosen: Assistant response
            # rejected: Assistant response

            # In binarized version, 'chosen' and 'rejected' might be the full conversation or just the response.
            # Let's inspect one item if we could, but for now assume standard format.
            # Actually, HuggingFaceH4/ultrafeedback_binarized usually has:
            # prompt (str), chosen (list of msgs), rejected (list of msgs)

            chosen_response = chosen[-1]["content"]
            rejected_response = rejected[-1]["content"]

            data_manifest.append(
                {
                    "id": count,
                    "prompt": prompt,
                    "chosen": chosen_response,
                    "rejected": rejected_response,
                }
            )

            count += 1
        except Exception as e:
            print(f"Skipping item {count}: {e}")
            continue

    # Save Manifest
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(data_manifest, f, indent=2)

    print(f"Saved {count} samples to {output_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    # Output to data/zone_c/ultrafeedback (relative to src/data/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "../../data", "zone_c", "ultrafeedback")
    download_preference_data(output_dir, num_samples=100)
