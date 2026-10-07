#!/usr/bin/env python3
"""Convert SciTS comparison text files into a single JSON file for LLM judging."""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from langchain_openai import ChatOpenAI

# BLEU score (optional, graceful fallback) - mirrors tools/universal_evaluator.py
try:
	from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

	HAS_NLTK = True
except ImportError:
	HAS_NLTK = False


SECTION_PATTERN = re.compile(
	r"--- PROMPT ---\s*(?P<prompt>.*?)\s*--- GROUND TRUTH ---\s*"
	r"(?P<ground_truth>.*?)\s*--- PREDICTION ---\s*(?P<prediction>.*?)\s*$",
	re.DOTALL,
)

SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")


def normalize_block(text: str) -> str:
	"""Strip leading/trailing whitespace while preserving internal newlines."""
	return text.strip()


def split_sentences(text: str) -> list[str]:
	"""Split text into sentences on '.', '!', or '?' followed by whitespace."""
	return [s for s in SENTENCE_SPLIT_PATTERN.split(text.strip()) if s]


def truncate_to_sentence_count(text: str, max_sentences: int) -> str:
	"""Truncate text to at most max_sentences sentences."""
	sentences = split_sentences(text)
	if max_sentences <= 0 or len(sentences) <= max_sentences:
		return text.strip()
	return " ".join(sentences[:max_sentences])


def compute_word_overlap(gt: str, pred: str) -> float:
	"""Compute Jaccard word overlap (intersection / union)."""
	gt_words = set(gt.lower().split())
	pred_words = set(pred.lower().split())
	if not gt_words or not pred_words:
		return 0.0
	intersection = gt_words & pred_words
	union = gt_words | pred_words
	return len(intersection) / len(union) if union else 0.0


def compute_bleu(gt: str, pred: str) -> float:
	"""Compute BLEU-4 score with smoothing."""
	if not HAS_NLTK:
		return 0.0
	reference = [gt.lower().split()]
	hypothesis = pred.lower().split()
	smoothie = SmoothingFunction().method1
	try:
		return sentence_bleu(reference, hypothesis, smoothing_function=smoothie)
	except Exception:
		return 0.0


def compute_rouge_l(gt: str, pred: str) -> float:
	"""Compute ROUGE-L F1 score (longest common subsequence)."""
	gt_tokens = gt.lower().split()
	pred_tokens = pred.lower().split()

	if not gt_tokens or not pred_tokens:
		return 0.0

	matcher = difflib.SequenceMatcher(None, gt_tokens, pred_tokens)
	lcs_length = sum(block.size for block in matcher.get_matching_blocks())

	precision = lcs_length / len(pred_tokens) if pred_tokens else 0
	recall = lcs_length / len(gt_tokens) if gt_tokens else 0

	if precision + recall == 0:
		return 0.0
	return 2 * precision * recall / (precision + recall)


def score_record(record: dict) -> dict:
	"""Attach the same metrics computed in verify_timeseries_scits to a record."""
	gt_text = record["ground_truth"]
	pred_text = record["prediction"]
	record["word_overlap"] = compute_word_overlap(gt_text, pred_text)
	record["rouge_l"] = compute_rouge_l(gt_text, pred_text)
	record["bleu"] = compute_bleu(gt_text, pred_text)
	record["char_similarity"] = difflib.SequenceMatcher(None, gt_text, pred_text).ratio()
	return record


def summarize_scores(records: list[dict]) -> str:
	"""Build the same summary text written by verify_timeseries_scits."""
	count = len(records)
	if count == 0:
		return "PRISM Time Series SciTS Verification Results\n" + "=" * 40 + "\nSamples: 0\n"

	avg_overlap = sum(r["word_overlap"] for r in records) / count
	avg_rouge = sum(r["rouge_l"] for r in records) / count
	avg_bleu = sum(r["bleu"] for r in records) / count
	avg_char = sum(r["char_similarity"] for r in records) / count

	return (
		"PRISM Time Series SciTS Verification Results\n"
		+ "=" * 40
		+ f"\nSamples: {count}\n\n"
		"Average Scores:\n"
		f"  Word Overlap (Jaccard): {avg_overlap:.4f}\n"
		f"  ROUGE-L (F1):           {avg_rouge:.4f}\n"
		f"  BLEU-4:                 {avg_bleu:.4f}\n"
		f"  Char Similarity:        {avg_char:.4f}\n"
	)


def extract_question(prompt_block: str) -> str:
	"""Return the question text without the wrapper labels used in the export."""
	question = normalize_block(prompt_block)

	if question.startswith("Question:"):
		question = question[len("Question:") :].lstrip()

	if question.endswith("Answer:"):
		question = question[: -len("Answer:")].rstrip()

	return question


def parse_comparison_file(file_path: Path) -> dict[str, str]:
	"""Parse one comparison file into the schema expected by the judge."""
	content = file_path.read_text(encoding="utf-8")
	match = SECTION_PATTERN.search(content)
	if match is None:
		raise ValueError(f"Unrecognized comparison file format: {file_path}")

	prompt = extract_question(match.group("prompt"))
	ground_truth = normalize_block(match.group("ground_truth"))
	prediction = normalize_block(match.group("prediction"))
	# Cap the prediction at the ground truth's sentence count so longer,
	# rambling completions aren't scored against a much shorter reference.
	prediction = truncate_to_sentence_count(prediction, len(split_sentences(ground_truth)))

	return {
		"question": prompt,
		"ground_truth": ground_truth,
		"prediction": prediction,
		"correct": "",
	}


def sort_key(file_path: Path) -> tuple[int, str]:
	"""Sort comparison files by sample number when present."""
	match = re.search(r"(\d+)", file_path.stem)
	if match is None:
		return (float("inf"), file_path.name)

	return (int(match.group(1)), file_path.name)


def iter_input_files(input_dir: Path) -> list[Path]:
	"""Return comparison text files in the input directory."""
	comparison_files = sorted(input_dir.glob("*_comparison.txt"), key=sort_key)
	if comparison_files:
		return comparison_files

	return sorted(input_dir.glob("*.txt"), key=sort_key)


def build_output_path(input_dir: Path, output_path: Path | None) -> Path:
	"""Choose an output path when one is not provided explicitly."""
	if output_path is not None:
		return output_path

	return input_dir / "results.json"


JUDGE_SYSTEM_PROMPT = """
You are an expert evaluator. You will be given a question, a ground truth answer, and a predicted answer.
Score the prediction as follows:
- 1   : correct — the prediction is semantically equivalent to the ground truth (same class and consistent index/explanation)
- 0.5 : partial — the prediction has the correct class but mismatched indices or a partially wrong explanation
- 0   : incorrect — the prediction has the wrong class

Respond with ONLY the numeric score (1, 0.5, or 0) and nothing else.
""".strip()


def build_llm() -> ChatOpenAI:
	# Imported lazily: langchain_openai is not a declared project dependency,
	# so a module-level import broke every caller of this file (including
	# the comparison-file parsing helpers that never touch the LLM judge)
	# whenever langchain_openai wasn't installed.
	from langchain_openai import ChatOpenAI

	api_key = os.getenv("OPENAI_API_KEY")
	base_url = os.getenv("OPENAI_BASE_URL")
	model_name = os.getenv("LITELLM_TEST_MODEL", "gpt-5.4")

	if not api_key or not base_url:
		raise OSError(
			"OPENAI_API_KEY and OPENAI_BASE_URL environment variables are required for the LLM judge."
		)

	return ChatOpenAI(
		model_name=model_name,
		api_key=api_key,
		base_url=base_url,
		temperature=0,
	)


def judge_record(llm: ChatOpenAI, record: dict) -> float:
	user_message = (
		f"Question: {record['question']}\n\n"
		f"Ground truth: {record['ground_truth']}\n\n"
		f"Prediction: {record['prediction']}"
	)
	response = llm.invoke([
		{"role": "system", "content": JUDGE_SYSTEM_PROMPT},
		{"role": "user", "content": user_message},
	])
	raw = response.content.strip()
	try:
		score = float(raw)
	except ValueError:
		print(f"Warning: unexpected judge response {raw!r}, defaulting to 0")
		score = 0.0
	return score


def run_judge(input_json: Path, output_json: Path) -> int:
	records = json.loads(input_json.read_text(encoding="utf-8"))
	llm = build_llm()

	for i, record in enumerate(records):
		score = judge_record(llm, record)
		record["correct"] = score
		print(f"[{i + 1}/{len(records)}] score={score}")

	output_json.parent.mkdir(parents=True, exist_ok=True)
	output_json.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
	print(f"Wrote judged records to {output_json}")

	total = len(records)
	if total == 0:
		print("No records to score.")
		return 0

	scores = [float(r["correct"]) for r in records]
	n_correct = sum(1 for s in scores if s == 1.0)
	n_partial = sum(1 for s in scores if s == 0.5)
	n_incorrect = sum(1 for s in scores if s == 0.0)
	avg_correct = n_correct / total
	avg_partial = n_partial / total
	avg_incorrect = n_incorrect / total
	summary = (
		f"Results (n={total}):\n"
		f"  Correct   (1.0): {n_correct:4d}  ({avg_correct:.2%})\n"
		f"  Partial   (0.5): {n_partial:4d}  ({avg_partial:.2%})\n"
		f"  Incorrect (0.0): {n_incorrect:4d}  ({avg_incorrect:.2%})\n"
	)
	print(f"\n{summary}")
	results_path = output_json.parent / "judge_results.txt"
	results_path.write_text(summary, encoding="utf-8")
	print(f"Wrote results to {results_path}")
	return 0


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Parse SciTS verification text files into a single JSON file."
	)
	parser.add_argument(
		"input_dir",
		type=Path,
		help="Directory containing comparison text files (parse mode) or a JSON file to judge (--judge mode).",
	)
	parser.add_argument(
		"-o",
		"--output",
		type=Path,
		default=None,
		help="Output JSON path. Defaults to <input_dir>.json.",
	)
	parser.add_argument(
		"--judge",
		action="store_true",
		default=False,
		help=(
			"Run the LLM-as-a-judge workflow on an existing JSON file produced by "
			"the parse step. Requires OPENAI_API_KEY and OPENAI_BASE_URL env vars."
		),
	)
	return parser.parse_args()


def main() -> int:
	args = parse_args()

	if args.judge:
		input_json = args.input_dir  # positional arg reused as the JSON path
		if not input_json.is_file():
			raise FileNotFoundError(f"Judge input file not found: {input_json}")
		output_json = args.output
		if output_json is None:
			stem = input_json.stem
			output_json = input_json.with_name(f"{stem}_judged.json")
		return run_judge(input_json, output_json)

	input_dir = args.input_dir
	output_path = build_output_path(input_dir, args.output)

	if not input_dir.is_dir():
		raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

	input_files = iter_input_files(input_dir)
	if not input_files:
		raise FileNotFoundError(f"No files found in input directory: {input_dir}")

	records = [parse_comparison_file(file_path) for file_path in input_files]

	output_path.parent.mkdir(parents=True, exist_ok=True)
	output_path.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

	print(f"Wrote {len(records)} records to {output_path}")

	# Re-score entry by entry using the same metrics as verify_timeseries_scits.
	scored_records = [score_record(dict(record)) for record in records]
	summary = summarize_scores(scored_records)
	print(f"\n{summary}")

	scored_path = output_path.with_name(f"{output_path.stem}_scored.json")
	scored_path.write_text(json.dumps(scored_records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
	print(f"Wrote scored records to {scored_path}")

	summary_path = output_path.parent / "results_summary.txt"
	summary_path.write_text(summary, encoding="utf-8")
	print(f"Wrote results to {summary_path}")

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
