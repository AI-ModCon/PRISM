import json
import os

from datasets import load_dataset
from tqdm import tqdm


def download_scienceqa_subset(output_dir, num_samples=100):
    print(f"Downloading ScienceQA subset ({num_samples} samples) to {output_dir}...")

    # Create directories
    img_dir = os.path.join(output_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    # Load ScienceQA
    # 'derek-thomas/ScienceQA' is the standard HF version
    try:
        dataset = load_dataset("derek-thomas/ScienceQA", split="train", streaming=True)
    except Exception as e:
        print(f"Error loading ScienceQA: {e}")
        return

    data_manifest = []

    count = 0
    for item in tqdm(dataset):
        if count >= num_samples:
            break

        # ScienceQA structure:
        # image: PIL.Image or None
        # question: str
        # choices: list[str]
        # answer: int (index)
        # hint: str
        # solution: str

        try:
            image = item.get("image")
            question = item.get("question", "")
            choices = item.get("choices", [])
            answer_idx = item.get("answer", 0)
            solution = item.get("solution", "")
            hint = item.get("hint", "")

            # Construct Instruction
            # Format: Question + Choices
            choices_str = "\n".join([f"({i}) {c}" for i, c in enumerate(choices)])
            instruction = f"Question: {question}\nChoices:\n{choices_str}\n"
            if hint:
                instruction += f"Hint: {hint}\n"
            instruction += "Answer:"

            # Construct Output
            # Format: Answer + Explanation
            answer_text = choices[answer_idx] if choices else ""
            output = f"The answer is ({answer_idx}) {answer_text}.\nExplanation: {solution}"

            img_path = None
            if image:
                # Save Image
                img_filename = f"{count:05d}.jpg"
                img_path = os.path.join(img_dir, img_filename)
                # Ensure absolute path
                abs_img_path = os.path.abspath(img_path)
                image.convert("RGB").save(abs_img_path)
                img_path = abs_img_path

            # Add to manifest
            data_manifest.append(
                {
                    "id": count,
                    "image_path": img_path,  # Can be None
                    "instruction": instruction,
                    "output": output,
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
    # Output to data/zone_b/scienceqa (relative to src/data/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "../../data", "zone_b", "scienceqa")
    download_scienceqa_subset(output_dir, num_samples=100)
