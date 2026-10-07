import abc
from typing import Any

import torch


class BaseEvaluator(abc.ABC):
    """
    Abstract Base Class for PRISM Evaluators.
    Each modality (Geometry, Graph, etc.) will implement a subclass.
    """

    def __init__(self, model, tokenizer, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()

    @abc.abstractmethod
    def evaluate(self, limit: int = None) -> dict[str, float]:
        """
        Run the evaluation loop.
        Args:
            limit: Optional limit on number of samples to evaluate (for debugging).
        Returns:
            Dict of metrics (e.g., {"accuracy": 0.85, "bleu": 0.4})
        """
        pass

    def generate(self, inputs: dict[str, Any], max_new_tokens: int = 128) -> str:
        """
        Helper for generation.
        Args:
           inputs: Union Schema dict (moved to device).
           max_new_tokens: Max tokens to generate.
        """
        # Move inputs to device
        # Move inputs to device and cast to model dtype
        model_dtype = self.model.parameters().__next__().dtype

        def recursive_to_device(item):
            if isinstance(item, torch.Tensor):
                item = item.to(self.device)
                if item.dtype in [torch.float32, torch.float64]:
                    item = item.to(model_dtype)
                return item
            elif isinstance(item, dict):
                return {k: recursive_to_device(v) for k, v in item.items()}
            elif isinstance(item, list | tuple):
                return type(item)(recursive_to_device(i) for i in item)
            else:
                return item

        device_inputs = recursive_to_device(inputs)

        # FIX: UnifiedTransformer expects 'text' key for input_ids
        if "input_ids" in device_inputs and "text" not in device_inputs:
            device_inputs["text"] = device_inputs["input_ids"]

        with torch.no_grad():
            outputs = self.model.generate(
                inputs=device_inputs,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode
        # Note: Depending on model output format, might need slicing
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return decoded


class EvaluatorRegistry:
    _registry = {}

    @classmethod
    def register(cls, name: str):
        def decorator(eval_cls):
            cls._registry[name] = eval_cls
            return eval_cls

        return decorator

    @classmethod
    def get(cls, name: str):
        return cls._registry.get(name)
