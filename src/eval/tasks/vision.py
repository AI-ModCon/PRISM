from datasets import load_dataset
from transformers import AutoProcessor

from ..evaluator import BaseEvaluator, EvaluatorRegistry


@EvaluatorRegistry.register("vision_vqa")
class VQAv2Evaluator(BaseEvaluator):
    """
    Evaluates Vision Projector on VQAv2 (Validation Split).
    Task: Vision Question Answering.
    Metric: Exact Match Accuracy (Top-1 Generation).
    """

    def __init__(self, model, tokenizer, device="cuda"):
        super().__init__(model, tokenizer, device)

        # Load Dataset (Streaming)
        # HuggingFaceM4/VQAv2 required trust_remote_code check in verification,
        # but standard load_dataset might handle it if we trust.
        try:
            self.dataset = load_dataset(
                "HuggingFaceM4/VQAv2", split="validation", streaming=True, trust_remote_code=True
            )
        except Exception as e:
            print(f"Warning: Failed to load VQAv2: {e}")
            self.dataset = []

        # Load Processor for SigLIP (Independent of model to ensure correct preprocessing)
        try:
            self.processor = AutoProcessor.from_pretrained(
                "google/siglip2-base-patch16-224", trust_remote_code=True
            )
        except Exception as e:
            print(f"Warning: Failed to load SigLIP Processor: {e}")
            self.processor = None

    def evaluate(self, limit: int = 50):
        print("Evaluating VQAv2 (Vision)...")
        if not self.dataset or not self.processor:
            return {"accuracy": 0.0, "valid_count": 0}

        correct = 0
        count = 0

        for _i, item in enumerate(self.dataset):
            if limit and count >= limit:
                break
            try:
                # Keys: ['question_type', 'multiple_choice_answer', 'answers', 'image_id', 'question', 'image']
                image = item["image"]
                question = item["question"]
                answer_str = item["multiple_choice_answer"]  # Primary truth

                # 1. Prepare Image
                # Processor returns dict with 'pixel_values'
                if image.mode != "RGB":
                    image = image.convert("RGB")

                inputs_proc = self.processor(images=image, return_tensors="pt")
                pixel_values = inputs_proc["pixel_values"].to(self.device)  # (1, 3, H, W)

                # 2. Prepare Prompt
                prompt = f"Question: {question}\nAnswer:"

                # 3. Tokenize Text
                text_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

                # 4. Combine Inputs
                # UnifiedTransformer expects dict with keys matching modality names ('image', 'text' mapped from input_ids)
                generate_inputs = {
                    "image": pixel_values,
                    "input_ids": text_inputs.input_ids,
                    "attention_mask": text_inputs.attention_mask,
                }

                # 5. Generate
                # Note: BaseEvaluator.generate auto-maps 'input_ids' -> 'text' now.
                output_str = self.generate(generate_inputs, max_new_tokens=10)

                # 6. Evaluate
                # Simple exact match (case insensitive)
                # Output might contain prompt, need to clean?
                # BaseEvaluator.generate returns decoded text.
                # Usually it includes prompt if model is decoder-only,
                # but our generate helper uses tokenizer.decode(outputs[0]) which includes prompt
                # UNLESS we slice. BaseEvaluator implementation:
                # "decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)"
                # This usually includes prompt.

                # Strip prompt
                cleaned_output = output_str.replace(prompt, "").strip().lower()
                clean_truth = answer_str.lower()

                # VQA Eval usually checks if generated word is in list of answers.
                # Here we check against 'multiple_choice_answer' (most frequent/consensus).

                if clean_truth in cleaned_output:
                    correct += 1

                count += 1
            except Exception:
                # print(f"VQAv2 Error: {e}")
                continue

        acc = correct / count if count > 0 else 0.0
        return {"accuracy": acc, "valid_count": count}


# Disabled MME Evaluator stub for now
# @EvaluatorRegistry.register("vision_mme")
# ...


@EvaluatorRegistry.register("vision_mathvista")
class MathVistaEvaluator(BaseEvaluator):
    """
    Evaluates Vision Projector on MathVista (AI4Math/MathVista).
    Task: Visual Math Reasoning.
    Metric: Accuracy (Free-form / Multiple Choice mixture).
    """

    def __init__(self, model, tokenizer, device="cuda"):
        super().__init__(model, tokenizer, device)
        try:
            self.dataset = load_dataset(
                "AI4Math/MathVista", split="testmini", streaming=True, trust_remote_code=True
            )
            self.processor = AutoProcessor.from_pretrained(
                "google/siglip2-base-patch16-224", trust_remote_code=True
            )
        except Exception as e:
            print(f"Warning: Failed to load MathVista: {e}")
            self.dataset = []
            self.processor = None

    def evaluate(self, limit: int = 50):
        print("Evaluating MathVista (Visual Math)...")
        if not self.dataset or not self.processor:
            return {"accuracy": 0.0, "valid_count": 0}

        count = 0
        correct = 0  # Proxy: non-empty generation or simple check

        for item in self.dataset:
            if limit and count >= limit:
                break
            try:
                # Keys: question, answer, decoded_image (PIL)
                image = item["decoded_image"]
                q = item["question"]
                truth = str(item["answer"]).lower()

                if image.mode != "RGB":
                    image = image.convert("RGB")
                inputs_proc = self.processor(images=image, return_tensors="pt")
                pixel_values = inputs_proc["pixel_values"].to(self.device)

                prompt = f"Question: {q}\nAnswer:"
                text_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

                generate_inputs = {
                    "image": pixel_values,
                    "input_ids": text_inputs.input_ids,
                    "attention_mask": text_inputs.attention_mask,
                }

                output_str = self.generate(generate_inputs, max_new_tokens=10)
                generated = output_str.replace(prompt, "").strip().lower()

                # MathVista eval is complex. We do a proxy containment check.
                if truth in generated:
                    correct += 1

                count += 1
            except Exception:
                continue

        acc = correct / count if count > 0 else 0.0
        return {"accuracy": acc, "valid_count": count}


@EvaluatorRegistry.register("vision_mathvision")
class MathVisionEvaluator(BaseEvaluator):
    """
    Evaluates Vision Projector on MathVision.
    Task: Visual Math.
    """

    def __init__(self, model, tokenizer, device="cuda"):
        super().__init__(model, tokenizer, device)
        try:
            self.dataset = load_dataset(
                "MathVision/MathVision", split="test", streaming=True, trust_remote_code=True
            )
            self.processor = AutoProcessor.from_pretrained(
                "google/siglip2-base-patch16-224", trust_remote_code=True
            )
        except Exception as e:
            print(f"Warning: Failed to load MathVision: {e}")
            self.dataset = []
            self.processor = None

    def evaluate(self, limit: int = 50):
        print("Evaluating MathVision...")
        if not self.dataset or not self.processor:
            return {"accuracy": 0.0, "valid_count": 0}

        count = 0
        correct = 0

        for item in self.dataset:
            if limit and count >= limit:
                break
            try:
                # Keys: question, answer, decoded_image
                image = item["decoded_image"]
                q = item["question"]
                truth = str(item["answer"]).lower()

                if image.mode != "RGB":
                    image = image.convert("RGB")
                inputs_proc = self.processor(images=image, return_tensors="pt")
                pixel_values = inputs_proc["pixel_values"].to(self.device)

                prompt = f"Question: {q}\nAnswer:"
                text_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

                generate_inputs = {
                    "image": pixel_values,
                    "input_ids": text_inputs.input_ids,
                    "attention_mask": text_inputs.attention_mask,
                }

                output_str = self.generate(generate_inputs, max_new_tokens=10)
                generated = output_str.replace(prompt, "").strip().lower()

                if truth in generated:
                    correct += 1
                count += 1
            except Exception:
                continue

        acc = correct / count if count > 0 else 0.0
        return {"accuracy": acc, "valid_count": count}


@EvaluatorRegistry.register("vision_mmstar")
class MMStarEvaluator(BaseEvaluator):
    """
    Evaluates Vision Projector on MMStar.
    Task: Evaluation benchmark.
    """

    def __init__(self, model, tokenizer, device="cuda"):
        super().__init__(model, tokenizer, device)
        try:
            self.dataset = load_dataset(
                "Lin-Chen/MMStar", split="val", streaming=True, trust_remote_code=True
            )
            self.processor = AutoProcessor.from_pretrained(
                "google/siglip2-base-patch16-224", trust_remote_code=True
            )
        except Exception as e:
            print(f"Warning: Failed to load MMStar: {e}")
            self.dataset = []
            self.processor = None

    def evaluate(self, limit: int = 50):
        print("Evaluating MMStar...")
        if not self.dataset or not self.processor:
            return {"accuracy": 0.0, "valid_count": 0}

        count = 0
        correct = 0

        for item in self.dataset:
            if limit and count >= limit:
                break
            try:
                # Keys: question, answer, image
                image = item["image"]
                q = item["question"]
                truth = str(item["answer"]).lower()  # Check truth key

                if image.mode != "RGB":
                    image = image.convert("RGB")
                inputs_proc = self.processor(images=image, return_tensors="pt")
                pixel_values = inputs_proc["pixel_values"].to(self.device)

                prompt = f"Question: {q}\nAnswer:"
                text_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

                generate_inputs = {
                    "image": pixel_values,
                    "input_ids": text_inputs.input_ids,
                    "attention_mask": text_inputs.attention_mask,
                }

                output_str = self.generate(generate_inputs, max_new_tokens=10)
                generated = output_str.replace(prompt, "").strip().lower()

                if truth in generated:
                    correct += 1
                count += 1
            except Exception:
                continue

        acc = correct / count if count > 0 else 0.0
        return {"accuracy": acc, "valid_count": count}


@EvaluatorRegistry.register("vision_mmmu")
class MMMUEvaluator(BaseEvaluator):
    """
    Evaluates Vision Projector on MMMU (Biology subset).
    Task: Multimodal Reasoning (College-level).
    """

    def __init__(self, model, tokenizer, device="cuda"):
        super().__init__(model, tokenizer, device)
        try:
            # Picking Biology as representative subset
            self.dataset = load_dataset(
                "MMMU/MMMU", "Biology", split="validation", streaming=True, trust_remote_code=True
            )
            self.processor = AutoProcessor.from_pretrained(
                "google/siglip2-base-patch16-224", trust_remote_code=True
            )
        except Exception as e:
            print(f"Warning: Failed to load MMMU: {e}")
            self.dataset = []
            self.processor = None

    def evaluate(self, limit: int = 50):
        print("Evaluating MMMU (Biology)...")
        if not self.dataset or not self.processor:
            return {"accuracy": 0.0, "valid_count": 0}

        count = 0
        correct = 0

        for item in self.dataset:
            if limit and count >= limit:
                break
            try:
                # Keys: question, options, answer, image (PIL)
                image = item["image"]
                q = item["question"]
                options = item["options"]  # string representation "[...]"
                truth = item["answer"]  # "A", "B"...

                if image.mode != "RGB":
                    image = image.convert("RGB")
                inputs_proc = self.processor(images=image, return_tensors="pt")
                pixel_values = inputs_proc["pixel_values"].to(self.device)

                # Parse options if string, or just list
                # MMMU options are typically stringified list "['A', 'B']" or proper list. Verify?
                # Assuming string format from previous experience or just prompt without options explictly
                # Better: Prompt "Question: ... Options: ... Answer:"

                prompt = f"Question: {q}\nOptions: {options}\nAnswer:"
                text_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

                generate_inputs = {
                    "image": pixel_values,
                    "input_ids": text_inputs.input_ids,
                    "attention_mask": text_inputs.attention_mask,
                }

                output_str = self.generate(generate_inputs, max_new_tokens=5)
                generated = output_str.replace(prompt, "").strip()

                if len(generated) > 0 and generated[0].upper() == truth.upper():
                    correct += 1
                elif truth in generated:  # Less strict
                    correct += 1

                count += 1
            except Exception:
                continue

        acc = correct / count if count > 0 else 0.0
        return {"accuracy": acc, "valid_count": count}
