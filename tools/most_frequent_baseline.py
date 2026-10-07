#!/usr/bin/env python3
"""Compute a "most frequent ground truth" baseline per time-series domain.

For each domain (inferred from the question text), find the most common
ground-truth answer among the samples in that domain, then score that single
answer against every ground truth in the domain using the same metrics as
tools/scits_f1.py (word overlap, ROUGE-L, BLEU-4, char similarity).
"""

from __future__ import annotations

import argparse
import difflib
import json
from collections import Counter, defaultdict
from pathlib import Path

from scits_f1 import compute_bleu, compute_rouge_l, compute_word_overlap

# Ordered (most-specific-first) substrings used to bucket questions into domains.
DOMAIN_PATTERNS = [
	("gravitational wave", "Gravitational Wave"),
	("earthquake event", "Earthquake"),
	("rainfall event", "Rainfall"),
	("traffic flow anomaly", "Traffic Flow Anomaly"),
	("weather anomaly", "Weather Anomaly"),
	("ecg anomaly", "ECG Anomaly"),
	("12-lead ecg", "12-Lead ECG Cardiac Condition"),
	("industrial machine anomaly", "Industrial Machine Anomaly"),
	("freezing anomaly", "Freezing Anomaly (Gait)"),
	("bearing vibration", "Bearing Vibration Fault"),
	("radar segment classification", "Radar Segment Classification"),
	("radar signal classification", "Radar Signal Classification"),
	("sleep stage classification", "Sleep Stage Classification"),
	("eeg waveform classification", "EEG Waveform Classification"),
	("movement imagination classification", "Movement Imagination (EEG)"),
	("depress", "EEG Depression Diagnosis"),
	("light curve classification", "Light Curve Classification"),
	("marmoset classification", "Marmoset Call Classification"),
	("birds sound classification", "Bird Sound Classification"),
	("activity classification", "Activity Classification"),
	("stock qa", "Stock QA"),
	("temperature qa", "Temperature QA"),
]


def infer_domain(question: str) -> str:
	q = question.lower()
	for substr, label in DOMAIN_PATTERNS:
		if substr in q:
			return label
	return "Unknown"


def score_pair(gt: str, pred: str) -> dict:
	return {
		"word_overlap": compute_word_overlap(gt, pred),
		"rouge_l": compute_rouge_l(gt, pred),
		"bleu": compute_bleu(gt, pred),
		"char_similarity": difflib.SequenceMatcher(None, gt, pred).ratio(),
	}


def main() -> int:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("input_json", type=Path, help="Path to results_scored.json")
	parser.add_argument("-o", "--output", type=Path, default=None,
		help="Output JSON path for per-domain baseline report (default: alongside input)")
	args = parser.parse_args()

	records = json.loads(args.input_json.read_text(encoding="utf-8"))

	by_domain: dict[str, list[dict]] = defaultdict(list)
	for r in records:
		by_domain[infer_domain(r["question"])].append(r)

	domain_reports = {}
	overall_metrics = defaultdict(list)

	for domain, recs in sorted(by_domain.items()):
		gts = [r["ground_truth"] for r in recs]
		most_common_gt, freq = Counter(gts).most_common(1)[0]

		metrics_sum = defaultdict(float)
		for r in recs:
			scores = score_pair(r["ground_truth"], most_common_gt)
			for k, v in scores.items():
				metrics_sum[k] += v
				overall_metrics[k].append(v)

		n = len(recs)
		domain_reports[domain] = {
			"n_samples": n,
			"most_frequent_ground_truth": most_common_gt,
			"most_frequent_count": freq,
			"most_frequent_frac": freq / n,
			"avg_word_overlap": metrics_sum["word_overlap"] / n,
			"avg_rouge_l": metrics_sum["rouge_l"] / n,
			"avg_bleu": metrics_sum["bleu"] / n,
			"avg_char_similarity": metrics_sum["char_similarity"] / n,
		}

	total_n = sum(len(v) for v in by_domain.values())
	overall = {
		"n_samples": total_n,
		"avg_word_overlap": sum(overall_metrics["word_overlap"]) / total_n,
		"avg_rouge_l": sum(overall_metrics["rouge_l"]) / total_n,
		"avg_bleu": sum(overall_metrics["bleu"]) / total_n,
		"avg_char_similarity": sum(overall_metrics["char_similarity"]) / total_n,
	}

	report = {"overall": overall, "by_domain": domain_reports}

	output_path = args.output or args.input_json.with_name("most_frequent_baseline.json")
	output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

	print(f"Most-Frequent-Class Baseline (n={total_n})")
	print("=" * 60)
	print(f"{'Domain':40s} {'n':>5s} {'WordOv':>7s} {'ROUGE-L':>8s} {'BLEU-4':>7s} {'CharSim':>8s}")
	for domain, rep in sorted(domain_reports.items(), key=lambda kv: -kv[1]["n_samples"]):
		print(
			f"{domain:40s} {rep['n_samples']:5d} "
			f"{rep['avg_word_overlap']:7.4f} {rep['avg_rouge_l']:8.4f} "
			f"{rep['avg_bleu']:7.4f} {rep['avg_char_similarity']:8.4f}"
		)
	print("-" * 60)
	print(
		f"{'OVERALL':40s} {overall['n_samples']:5d} "
		f"{overall['avg_word_overlap']:7.4f} {overall['avg_rouge_l']:8.4f} "
		f"{overall['avg_bleu']:7.4f} {overall['avg_char_similarity']:8.4f}"
	)
	print(f"\nWrote report to {output_path}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
