from datasets import load_dataset

from ..evaluator import BaseEvaluator, EvaluatorRegistry


@EvaluatorRegistry.register("text_mmlu")
class MMLUEvaluator(BaseEvaluator):
    """
    Evaluates Text Projector on MMLU (Original).
    Task: Multi-choice Reasoning (4 options).
    Metric: Accuracy.
    """

    def __init__(self, model, tokenizer, device="cuda", num_shots=0, trace_file=None):
        super().__init__(model, tokenizer, device)
        self.num_shots = num_shots
        self.trace_file = trace_file  # Path to write debug trace
        try:
            self.dataset = load_dataset("cais/mmlu", "all", split="test", streaming=True)
            if num_shots > 0:
                print(
                    f"Loading {num_shots}-shot examples from MMLU 'dev' split (Subject-Specific)..."
                )
                # Load entire dev split to memory for grouping
                dev_ds = load_dataset("cais/mmlu", "all", split="dev", streaming=False)

                # Group by subject
                from collections import defaultdict

                self.shots_by_subject = defaultdict(list)
                for item in dev_ds:
                    sub = item["subject"]
                    self.shots_by_subject[sub].append(item)
                print(f"Loaded shots for {len(self.shots_by_subject)} subjects.")
            else:
                self.shots_by_subject = {}
        except Exception as e:
            print(f"Warning: Failed to load MMLU: {e}")
            self.dataset = []

    def evaluate(self, limit: int = 100):
        print(f"Evaluating MMLU ({self.num_shots}-shot, Subject-Specific)...")
        correct = 0
        count = 0
        labels = ["A", "B", "C", "D"]

        # Open trace file if needed
        trace_f = None
        if self.trace_file:
            trace_f = open(self.trace_file, "w")
            trace_f.write(f"# MMLU Trace ({self.num_shots}-shot)\n\n")

        for item in self.dataset:
            if limit and count >= limit:
                break
            try:
                q = item["question"]
                options = item["choices"]
                answer_idx = item["answer"]
                subject = item.get("subject", "default")

                # Exact Template from Literature/Harness
                # Header
                clean_subject = subject.replace("_", " ")
                header = f"The following are multiple choice questions (with answers) about {clean_subject}.\n\n"

                # Construct shot prompt
                shot_prompt = ""
                if self.num_shots > 0 and subject in self.shots_by_subject:
                    # Get shots for this specific subject
                    # Take first N (standard) or random? Literature usually uses fixed set (first N).
                    subject_shots = self.shots_by_subject[subject][: self.num_shots]

                    for shot in subject_shots:
                        sq = shot["question"]
                        soptions = shot["choices"]
                        sans_idx = shot["answer"]
                        sops = ""
                        for i, opt in enumerate(soptions):
                            sops += f"{labels[i]}. {opt}\n"
                        shot_prompt += f"{sq}\n{sops}Answer: {labels[sans_idx]}\n\n"

                options_str = ""
                for i, opt in enumerate(options):
                    options_str += f"{labels[i]}. {opt}\n"

                # Full Prompt
                # Header + Shots + Question
                prompt = f"{header}{shot_prompt}{q}\n{options_str}Answer:"

                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

                # Trace: Log Input (truncated)
                if trace_f and count < 5:
                    trace_f.write(
                        f"## Sample {count+1} ({subject})\n**Prompt**:\n```\n{prompt[-500:]}\n...(truncated)\n```\n"
                    )

                output_str = self.generate(inputs, max_new_tokens=2)
                generated = output_str.replace(prompt, "").strip()

                # Trace: Log Output
                if trace_f and count < 5:
                    trace_f.write(f"**Output**: `{generated}`\n**Truth**: {labels[answer_idx]}\n\n")

                pred_label = None
                for char in generated:
                    if char.upper() in labels:
                        pred_label = char.upper()
                        break

                if pred_label == labels[answer_idx]:
                    correct += 1
                count += 1
            except Exception:
                continue

        if trace_f:
            trace_f.close()
        acc = correct / count if count > 0 else 0.0
        return {"accuracy": acc, "valid_count": count}


@EvaluatorRegistry.register("text_mmlu_pro")
class MMLUProEvaluator(BaseEvaluator):
    """
    Evaluates Text Projector on MMLU-Pro (TIGER-Lab/MMLU-Pro).
    Task: Multi-choice Reasoning (10 options).
    Metric: Accuracy (Option Selection).
    """

    def __init__(self, model, tokenizer, device="cuda", num_shots=0):
        super().__init__(model, tokenizer, device)
        self.num_shots = num_shots
        try:
            self.dataset = load_dataset("TIGER-Lab/MMLU-Pro", split="test", streaming=True)
            if num_shots > 0:
                print(f"Loading {num_shots}-shot examples from MMLU-Pro 'validation' split...")
                self.val_dataset = load_dataset(
                    "TIGER-Lab/MMLU-Pro", split="validation", streaming=True
                )
                self.shots = list(self.val_dataset.take(num_shots))
            else:
                self.shots = []
        except Exception:
            print("Warning: Failed to load MMLU-Pro. Using validation.")
            self.dataset = load_dataset("TIGER-Lab/MMLU-Pro", split="validation", streaming=True)
            self.shots = []

    def evaluate(self, limit: int = 50):
        print(f"Evaluating MMLU-Pro ({self.num_shots}-shot)...")
        correct = 0
        count = 0

        # Option labels A-J
        labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

        # Pre-build shot string
        shot_prompt = ""
        for shot in self.shots:
            q = shot["question"]
            options = shot["options"]
            ans_idx = shot["answer_index"]
            ops = ""
            for i, opt in enumerate(options):
                if i < len(labels):
                    ops += f"{labels[i]}. {opt}\n"
            shot_prompt += f"Question: {q}\nOptions:\n{ops}Answer: {labels[ans_idx]}\n\n"

        for item in self.dataset:
            if limit and count >= limit:
                break
            try:
                # Keys: question, options (list), answer_index (int)
                q = item["question"]
                options = item["options"]
                answer_idx = item["answer_index"]

                # Construct Prompt
                # "Question: ...
                # Options:
                # A. ...
                # ...
                # Answer:"

                options_str = ""
                for i, opt in enumerate(options):
                    if i < len(labels):
                        options_str += f"{labels[i]}. {opt}\n"

                prompt = f"{shot_prompt}Question: {q}\nOptions:\n{options_str}Answer:"

                # Tokenize
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

                # Generate (Short, we just want the letter)
                # Note: Model might output "The answer is A".
                # Ideally we check logits for A-J, but generation + parsing is standard for open-ended models.
                output_str = self.generate(inputs, max_new_tokens=5)

                # Clean Output
                # BaseEvaluator strips special tokens but might keep prompt?
                # output_str is result of generate().
                # If using our helper, it returns decoded text.
                # Remove prompt.
                generated = output_str.replace(prompt, "").strip()

                # Check for first letter
                # "A." or "A" or "Answer is A"
                # Simple heuristic: Look for valid label in first few chars.
                pred_label = None

                # 1. Exact match with label
                for char in generated:
                    if char.upper() in labels:
                        pred_label = char.upper()
                        break

                # Truth
                truth_label = labels[answer_idx]

                if pred_label == truth_label:
                    correct += 1

                count += 1
            except Exception:
                # print(f"MMLU Error: {e}")
                continue

        acc = correct / count if count > 0 else 0.0
        return {"accuracy": acc, "valid_count": count}


@EvaluatorRegistry.register("text_gpqa")
class GPQAEvaluator(BaseEvaluator):
    """
    Evaluates Text Projector on GPQA (Diamond Subset).
    Task: Graduate-Level QA (Biology, Physics, Chemistry).
    Metric: Accuracy (Multiple Choice).
    """

    def __init__(self, model, tokenizer, device="cuda"):
        super().__init__(model, tokenizer, device)
        try:
            # Token should be in env HF_TOKEN
            self.dataset = load_dataset(
                "Idavidrein/gpqa",
                "gpqa_diamond",
                split="train",
                streaming=True,
                trust_remote_code=True,
            )
        except Exception as e:
            print(f"Warning: Failed to load GPQA: {e}")
            self.dataset = []

    def evaluate(self, limit: int = 50):
        print("Evaluating GPQA (Expert Reasoning)...")
        if not self.dataset:
            return {"accuracy": 0.0, "valid_count": 0}

        correct = 0
        count = 0
        import random

        labels = ["A", "B", "C", "D"]

        for item in self.dataset:
            if limit and count >= limit:
                break
            try:
                # Keys: Question, Correct Answer, Incorrect Answer 1...3
                q = item["Question"]
                correct_ans = item["Correct Answer"]
                incorrects = [
                    item["Incorrect Answer 1"],
                    item["Incorrect Answer 2"],
                    item["Incorrect Answer 3"],
                ]

                # Shuffle options
                options = [correct_ans] + incorrects
                random.shuffle(options)

                # Find index of correct answer
                correct_idx = options.index(correct_ans)
                correct_label = labels[correct_idx]

                # Construct Prompt
                options_str = ""
                for i, opt in enumerate(options):
                    options_str += f"{labels[i]}. {opt}\n"

                prompt = f"Question: {q}\nOptions:\n{options_str}Answer:"

                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                output_str = self.generate(inputs, max_new_tokens=5)
                generated = output_str.replace(prompt, "").strip()

                # Check prediction
                pred_label = None
                for char in generated:
                    if char.upper() in labels:
                        pred_label = char.upper()
                        break

                if pred_label == correct_label:
                    correct += 1

                count += 1
            except Exception:
                # print(f"GPQA Error: {e}")
                continue

        acc = correct / count if count > 0 else 0.0
        return {"accuracy": acc, "valid_count": count}


@EvaluatorRegistry.register("text_aime2025")
class AIME2025Evaluator(BaseEvaluator):
    """
    Evaluates Text Projector on AIME 2025 (MathArena/aime_2025_I).
    Task: Math Competition (Integer Answer 0-999).
    Metric: Accuracy (Exact match of integer).
    """

    def __init__(self, model, tokenizer, device="cuda"):
        super().__init__(model, tokenizer, device)
        try:
            # Requires token? Maybe.
            self.dataset = load_dataset(
                "MathArena/aime_2025_I", split="train", streaming=True, trust_remote_code=True
            )
        except Exception as e:
            print(f"Warning: Failed to load AIME 2025: {e}")
            self.dataset = []

    def evaluate(self, limit: int = 50):
        print("Evaluating AIME 2025 (Math Competition)...")
        if not self.dataset:
            return {"accuracy": 0.0, "valid_count": 0}

        correct = 0
        count = 0

        for item in self.dataset:
            if limit and count >= limit:
                break
            try:
                # Keys: problem, answer (string/int)
                q = item["problem"]
                truth = str(item["answer"]).strip()

                prompt = f"Problem: {q}\nAnswer (Integer 0-999):"

                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                output_str = self.generate(inputs, max_new_tokens=10)  # Short answer
                generated = output_str.replace(prompt, "").strip()

                # Check if generated contains the truth number (simple check)

                if truth in generated:
                    correct += 1

                count += 1
            except Exception:
                continue

        acc = correct / count if count > 0 else 0.0
        return {"accuracy": acc, "valid_count": count}


import re


class IFEvalVerifier:
    """
    Native implementation of IFEval constraint checking.
    Validates generated text against kwargs constraints.
    """

    def check(self, text, kwargs):
        # kwargs is a list of dicts, each representing a constraint
        passes = True

        for constraint in kwargs:
            if not self._check_single(text, constraint):
                passes = False
                break  # Strict mode: one failure = fail
        return passes

    def _check_single(self, text, c):
        # Dispatch based on keys present in constraint dict

        # 1. Length Constraints (words)
        if c.get("num_words") is not None:
            count = len(text.split())
            limit = int(c["num_words"])
            relation = c.get("relation", "at least")

            if relation == "at least":
                if count < limit:
                    return False
            elif relation == "less than":
                if count >= limit:
                    return False

        # 2. Highlights
        if c.get("num_highlights") is not None:
            # Markdown highlights *text* or **text**
            # Heuristic regex
            matches = re.findall(r"\*+[^*]+\*+", text)
            count = len(matches)
            limit = int(c["num_highlights"])
            # Relation usually 'at least' for highlights? Or exact?
            # IFEval often specifies "at least X highlights"
            if count < limit:
                return False

        # 3. Keywords / Forbidden Words
        # Note: 'keywords' might be a list or 'keyword' a single string
        if c.get("keywords") is not None:
            for k in c["keywords"]:
                if k.lower() not in text.lower():
                    return False

        if c.get("forbidden_words") is not None:
            for k in c["forbidden_words"]:
                if k.lower() in text.lower():
                    return False

        # 4. Punctuation (e.g. no_comma)
        # Often encoded as instruction_id_list, but let's check keys if present
        # Based on sample, keys like 'punctuation' weren't explicit in kwargs dict,
        # but 'instruction_id_list' had 'punctuation:no_comma'.
        # However, verifying based on kwargs is safer if they map constraints there.
        # The sample showed 'kwargs': [{... 'num_highlights': None ...}] structure.
        # We will iterate all params that are NOT None.

        return True


@EvaluatorRegistry.register("text_ifeval")
class IFEvalEvaluator(BaseEvaluator):
    """
    Evaluates Text Projector on IFEval (google/IFEval).
    Task: Instruction Following.
    Metric: Strict Accuracy (All constraints met).
    """

    def __init__(self, model, tokenizer, device="cuda"):
        super().__init__(model, tokenizer, device)
        try:
            self.dataset = load_dataset(
                "google/IFEval", split="train", streaming=True, trust_remote_code=True
            )
            self.verifier = IFEvalVerifier()
        except Exception as e:
            print(f"Warning: Failed to load IFEval: {e}")
            self.dataset = []

    def evaluate(self, limit: int = 50):
        print("Evaluating IFEval (Instruction Following - Native)...")
        if not self.dataset:
            return {"accuracy": 0.0, "valid_count": 0}

        count = 0
        correct = 0

        for item in self.dataset:
            if limit and count >= limit:
                break
            try:
                p = item["prompt"]
                kwargs = item["kwargs"]  # List of constraints

                inputs = self.tokenizer(p, return_tensors="pt").to(self.device)

                # IFEval instructions can be long, allow more tokens
                output_str = self.generate(inputs, max_new_tokens=512)
                generated = output_str.replace(p, "").strip()

                if self.verifier.check(generated, kwargs):
                    correct += 1

                count += 1
            except Exception:
                continue

        acc = correct / count if count > 0 else 0.0
        return {"accuracy": acc, "valid_count": count}
