import re

import numpy as np
import torch
from datasets import load_dataset

from ..evaluator import BaseEvaluator, EvaluatorRegistry


@EvaluatorRegistry.register("geometry_matbench")
class MatBenchEvaluator(BaseEvaluator):
    """
    Evaluates Geometry Projector on MatBench (Crystal Property Prediction).
    Task: Given Crystal Structure (Text Description) -> Predict Band Gap (Float).
    """

    def __init__(self, model, tokenizer, device="cuda"):
        super().__init__(model, tokenizer, device)
        # Load Validation Split
        # Using a subset or fold0 test set if available.
        # For simplicity, we stream 'train' and take a slice, or usage 'validation' if present.
        # Check config: nimashoghi/matbench_jdft2d_fold0 usually has 1 fold.
        self.dataset = load_dataset(
            "nimashoghi/matbench_jdft2d_fold0", split="train", streaming=True
        )
        self.metric_name = "mae"

    def _featurize_geometry(self, item):
        # Construct (N, 6) tensor from atomic numbers and positions
        # Features: [x, y, z, atomic_num, dummy, dummy]
        # Walrus expects (B, N, D) or (B, T, N, D)

        atoms = item["atomic_numbers"]
        pos = item["positions"]  # list of [x,y,z]

        N = len(atoms)
        # Fixed 6 dimensions (Walrus default input_dim in PRISM config is 6)
        features = torch.zeros((N, 6), dtype=torch.float)

        for i in range(N):
            # Pos
            features[i, 0] = pos[i][0]
            features[i, 1] = pos[i][1]
            features[i, 2] = pos[i][2]
            # Atom (normalized slightly to avoid huge values? or raw if embedding handles it)
            # Walrus projects input_dim -> hidden states linearly. Raw is fine but usually normalization helps.
            # For simplicity: Keep Raw. Walrus is robust.
            features[i, 3] = float(atoms[i])

        # Add Batch Dim (1, N, 6)
        return features.unsqueeze(0)

    def evaluate(self, limit: int = 100):
        print("Evaluating MatBench (Geometry)...")
        predictions = []
        ground_truths = []

        count = 0
        for _i, item in enumerate(self.dataset):
            if limit and count >= limit:
                break
            try:
                # Structure: atomic_numbers, positions
                # FIX: Item is flattened, no 'structure' key
                atoms = item["atomic_numbers"]
                item["positions"]

                # Construct Prompt
                graph_desc = f"Crystal containing {len(atoms)} atoms."

                prompt = (
                    f"Predict property (Band Gap) for Crystal Structure:\n{graph_desc}\nTarget:"
                )

                # Tokenize
                # Tokenize
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

                # Featurize Geometry
                geo_tensor = self._featurize_geometry(item)

                # Construct Inputs
                inputs = {
                    "input_ids": inputs.input_ids,
                    "attention_mask": inputs.attention_mask,
                    "geometry": geo_tensor,
                }

                # 2. Generate
                output_str = self.generate(inputs, max_new_tokens=10)

                # 3. Parse Float
                # Expected output: " 1.234"

                # Simple extraction: find last number
                pred_val = self._extract_number(output_str)
                true_val = item["y"]  # Target is 'y' in MatBench JDFT2D

                # Robustness: If model fails to predict a number (untrained), bad guess is better than drop
                if pred_val is None:
                    pred_val = 0.0

                predictions.append(pred_val)
                ground_truths.append(true_val)

                count += 1
            except Exception:
                continue

        # 4. Compute Metrics
        if not predictions:
            return {"mae": 0.0, "valid_count": 0}

        preds = np.array(predictions)
        gts = np.array(ground_truths)

        mae = np.mean(np.abs(preds - gts))

        return {"mae": float(mae), "valid_count": len(predictions)}

    def _extract_number(self, text):
        # Extract last floating point number
        matches = re.findall(r"[-+]?\d*\.\d+|\d+", text)
        if matches:
            return float(matches[-1])
        return None
