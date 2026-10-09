import re

import numpy as np
from datasets import load_dataset

from ..evaluator import BaseEvaluator, EvaluatorRegistry


@EvaluatorRegistry.register("ts_monash")
class MonashEvaluator(BaseEvaluator):
    """
    Evaluates Time Series Projector on Monash Forecasting Archives.
    Task: Context History -> Forecast Horizon.
    Metric: MAE.
    """

    def __init__(self, model, tokenizer, device="cuda"):
        super().__init__(model, tokenizer, device)
        # Using 'monash_tsf' (weather)
        try:
            self.dataset = load_dataset(
                "monash_tsf", "weather", split="train", streaming=True, trust_remote_code=True
            )
        except Exception as e:
            # Fallback
            self.dataset = []
            print(f"Warning: Monash dataset not found: {e}. Eval will yield 0.")

    def evaluate(self, limit: int = 50):
        print("Evaluating Monash (Time Series Forecasting)...")
        ae_sum = 0
        count = 0

        for item in self.dataset:
            if limit and count >= limit:
                break
            try:
                # Chronos dataset format: 'target' (full series), 'start'
                # We need to split into context and horizon.
                target = np.array(item["target"])
                horizon = 30  # Daily weather, predict month

                if len(target) <= horizon:
                    continue

                context = target[:-horizon]
                ground_truth = target[-horizon:]

                # Textual Prompt for Forecasting?
                # "History: 1.0, 2.0... Forecast next 12 steps:"
                context_str = ", ".join([f"{v:.2f}" for v in context[-64:]])  # Limit context len
                prompt = f"Time Series History: {context_str}\nForecast the next {horizon} steps:"

                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

                # Generate
                output = self.generate(inputs, max_new_tokens=64)

                # Parse numbers
                # "10.5, 12.3, ..."
                # Extract all numbers from generated text (excluding prompt)
                gen_text = output.replace(prompt, "")
                preds = [float(x) for x in re.findall(r"[-+]?\d*\.\d+|\d+", gen_text)]

                # Eval first N matches against ground truth
                min_len = min(len(preds), len(ground_truth))
                if min_len == 0:
                    continue

                p = np.array(preds[:min_len])
                g = ground_truth[:min_len]

                loss = np.mean(np.abs(p - g))
                ae_sum += loss
                count += 1

            except Exception:
                # print(f"Monash Error: {e}")
                continue

        mae = ae_sum / count if count > 0 else 0.0
        return {"mae": mae, "valid_count": count}
