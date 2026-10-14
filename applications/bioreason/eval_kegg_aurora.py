"""KEGG accuracy eval for BioReason checkpoints on Aurora (XPU), single tile.

Stand-in until eval_kegg.py (PR #133) lands. It deliberately reuses the
training code paths instead of re-implementing them:

  * model:   src.train.build_model_config + UnifiedTransformer, so the
             architecture/special tokens match what src/train.py trained;
  * weights: load_model_weights_only (base + projector from model.safetensors),
             then merge_lora_into_base for the SFT adapter and, optionally,
             a GRPO adapter stacked on top (GRPO checkpoints hold only
             lora_adapter.pt -- see ZoneDTrainer.save_checkpoint);
  * prompt:  StreamingMultimodalDataset's own _process_dna_bioreason +
             _render_dna_prompt_text in GRPO mode (is_sft=False), i.e. the
             exact user turn + "<|im_start|>assistant\\n" generation prompt;
  * decode:  UnifiedTransformer.generate, as ZoneDTrainer._generate_completions.

Usage (on a compute node, after `module load frameworks` + venv):
  python applications/bioreason/eval_kegg_aurora.py \\
      --ckpt  $PRISM_OUTPUT_ROOT/BIOREASON-SFT-KEGG/<tag>/checkpoints/step_final \\
      [--grpo-adapter $PRISM_OUTPUT_ROOT/BIOREASON-GRPO-KEGG/<tag>/checkpoints/step_final/lora_adapter.pt] \\
      --data  $PRISM_DATA_ROOT/dna_local_shards_test/kegg \\
      --output results.json [--limit 50] [model=prism_olmo_1b ...extra Hydra overrides]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import torch  # noqa: E402

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def pick_device() -> torch.device:
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def compose_cfg(overrides: list[str]):
    """Hydra-compose src/conf/config.yaml exactly as src/train.py would."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    base = ["model=prism_olmo_1b", "training=bioreason_grpo"]
    with initialize_config_dir(config_dir=os.path.join(REPO, "src", "conf"), version_base=None):
        return compose(config_name="config", overrides=base + list(overrides))


def _adapter_alpha(path: str, fallback: float) -> float:
    """Alpha the adapter was trained with (sidecar written by save_lora_adapter)."""
    from src.utils.lora_utils import _load_adapter_meta

    meta = _load_adapter_meta(path) or {}
    return float(meta.get("alpha") or fallback)


def load_eval_model(
    ckpt: str | None,
    grpo_adapter: str | None = None,
    overrides: list[str] | None = None,
    device: torch.device | None = None,
):
    """Build the model the way src/train.py does and load trained weights.

    Returns (model, cfg). ``ckpt=None`` evaluates the untrained model (useful
    as a baseline / smoke test).
    """
    from src.model import UnifiedTransformer
    from src.train import build_model_config
    from src.training.distributed import load_model_weights_only
    from src.utils.lora_utils import merge_lora_into_base

    # Merged length = text + 2 x max_dna_length DNA slots (~2.8k for KEGG), above
    # the 2048 default of the merged-length guard (ModelConfig.max_merged_seq_length).
    os.environ.setdefault("MAX_SEQ_LENGTH", "4096")
    cfg = compose_cfg(overrides or [])
    model = UnifiedTransformer(build_model_config(cfg))

    if ckpt:
        load_model_weights_only(model, ckpt, device="cpu")
        sft_adapter = os.path.join(ckpt, "lora_adapter.pt")
        if os.path.exists(sft_adapter):
            alpha = _adapter_alpha(sft_adapter, cfg.training.get("resume_lora_alpha") or 64)
            merge_lora_into_base(model.backbone, sft_adapter, alpha=alpha)
            print(f"[eval] merged SFT adapter {sft_adapter} (alpha={alpha})")
    if grpo_adapter:
        alpha = _adapter_alpha(grpo_adapter, cfg.training.get("lora_alpha", 32))
        merge_lora_into_base(model.backbone, grpo_adapter, alpha=alpha)
        print(f"[eval] merged GRPO adapter {grpo_adapter} (alpha={alpha})")

    device = device or pick_device()
    model.to(device)
    if model.backbone is not None and device.type != "cpu":
        model.backbone.to(torch.bfloat16)
    model.eval()
    return model, cfg


DNA_MODES = ("real", "shuffled", "blank")
# Uninformative stand-in for --dna-mode blank. Non-empty on purpose: an empty string
# would give an all-zero DNA attention mask, which risks NaNs in the encoder softmax.
BLANK_DNA = "N" * 600


def make_prompt_builder(model, cfg, dna_mode: str = "real"):
    """Return ``build(item, donor=None) -> (prompt_batch, answer)`` using the dataset's renderer.

    dna_mode ablates only the DNA content (the prompt text and the answer stay real):
      real     -- the item's own reference/variant sequences
      shuffled -- another item's sequences (``donor``), so DNA is plausible but unrelated
      blank    -- a constant uninformative sequence
    """
    if dna_mode not in DNA_MODES:
        raise ValueError(f"dna_mode must be one of {DNA_MODES}, got {dna_mode!r}")
    from src.data.multimodal import StreamingMultimodalDataset
    from src.train import (
        _resolve_dataset_overrides,
        build_bio_dataset_kwargs,
        build_training_config,
    )

    mc = model.config
    tc = build_training_config(cfg, mc, output_dir="/tmp", seed=0)
    helper = StreamingMultimodalDataset(
        tokenizer=model.backbone_tokenizer,
        model_config=mc,
        dataset_overrides=_resolve_dataset_overrides(cfg),
        **build_bio_dataset_kwargs("bioreason_grpo", tc, mc),
    )
    tok = model.backbone_tokenizer

    def build(item, donor=None):
        d = helper._process_dna_bioreason(item)
        _, prompt_text = helper._render_dna_prompt_text(d["prompt"])
        ids = tok(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids
        if dna_mode == "shuffled":
            if donor is None:
                raise ValueError("dna_mode='shuffled' needs a donor item")
            ref, var = helper._process_dna_bioreason(donor)["dna_sequences"]
        elif dna_mode == "blank":
            ref, var = BLANK_DNA, BLANK_DNA
        else:
            ref, var = d["dna_sequences"]
        dna = {}
        for key, seq in (("dna_reference", ref), ("dna_variant", var)):
            enc = helper.dna_tokenizer(
                seq,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=mc.max_dna_length,
            )
            dna[key] = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
        batch = {"text": ids, "dna": dna, "_metadata": [f"{ids.shape[1]} 0"]}
        return batch, d["answer"]

    return build


def extract_answer(completion: str) -> str:
    from src.data.multimodal import clean_kegg_answer

    m = ANSWER_RE.search(completion)
    return clean_kegg_answer(m.group(1) if m else completion)


def _to(batch, device):
    if torch.is_tensor(batch):
        return batch.to(device)
    if isinstance(batch, dict):
        return {k: _to(v, device) for k, v in batch.items()}
    return batch


# Appended to a generation that never produced an <answer> tag, to force one (see _force_answer).
FORCE_SUFFIX = "\n</think>\n\nAnswer: <answer>"
FORCE_MAX_NEW_TOKENS = 32


def _force_answer(model, tok, device, batch, gen_ids, completion):
    """Make a generation that ended without <answer> produce one.

    Re-prompts with prompt + the model's own generation (trailing eos dropped) +
    FORCE_SUFFIX and decodes a few more tokens, so every example gets a scored
    prediction. Callers record this separately (``forced``) so natural completion
    stays visible in ``format_rate``.
    """
    eos = tok.eos_token_id
    gen = gen_ids.tolist()
    while gen and gen[-1] == eos:
        gen.pop()
    suffix = tok(FORCE_SUFFIX, add_special_tokens=False, return_tensors="pt").input_ids.to(batch["text"].device)
    ids = torch.cat([batch["text"], torch.tensor([gen], dtype=batch["text"].dtype, device=batch["text"].device),
                     suffix], dim=1)
    forced = dict(batch, text=ids, _metadata=[f"{ids.shape[1]} 0"])
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                         enabled=device.type != "cpu"):
        out = model.generate(_to(forced, device), max_new_tokens=FORCE_MAX_NEW_TOKENS, do_sample=False,
                             pad_token_id=tok.pad_token_id, eos_token_id=eos)
    cont = tok.decode(out[0, 1:], skip_special_tokens=True)
    cont = cont.split("</answer>")[0].split("\n")[0]
    return completion + FORCE_SUFFIX + cont + "</answer>"


def evaluate(model, cfg, dataset, limit: int | None = None, max_new_tokens: int = 1024, verbose=True,
             dna_mode: str = "real"):
    """Greedy-decode each KEGG item and score exact match on the <answer> span.

    Generation stops on OLMo's eos (<|endoftext|>, the token SFT supervises at the end of
    the assistant turn) or at max_new_tokens. KEGG reasoning traces are 331-649 tokens
    (+ answer), so 1024 leaves headroom; a smaller cap truncates before the answer.
    """
    build = make_prompt_builder(model, cfg, dna_mode=dna_mode)
    tok = model.backbone_tokenizer
    if "im_end" in (tok.eos_token or ""):
        raise ValueError(f"eos must be OLMo's <|endoftext|>, not a Qwen ChatML token: {tok.eos_token!r}")
    print(f"[eval] stop token: {tok.eos_token!r} (id {tok.eos_token_id}), pad id {tok.pad_token_id}, "
          f"max_new_tokens={max_new_tokens}", flush=True)
    device = next(model.parameters()).device
    n = len(dataset) if limit is None else min(limit, len(dataset))
    records = []
    t0 = time.time()
    for i in range(n):
        # shuffled: donor = next item in the full split (wraps), a deterministic derangement
        donor = dataset[(i + 1) % len(dataset)] if dna_mode == "shuffled" else None
        batch, answer = build(dataset[i], donor)
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                             enabled=device.type != "cpu"):
            out = model.generate(
                _to(batch, device),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
        completion = tok.decode(out[0, 1:], skip_special_tokens=True)  # strip BOS seed
        natural = bool(ANSWER_RE.search(completion))
        if not natural:
            completion = _force_answer(model, tok, device, batch, out[0, 1:], completion)
        pred = extract_answer(completion)
        records.append({
            "idx": i,
            "answer": answer,
            "pred": pred,
            "correct": pred == answer,
            "has_answer_tag": natural,  # the model's own generation, before any forcing
            "forced": not natural,
            "completion": completion,
        })
        if verbose:
            acc = sum(r["correct"] for r in records) / len(records)
            print(f"[eval] {i + 1}/{n} acc={acc:.3f} ({time.time() - t0:.0f}s) "
                  f"gt={answer!r} pred={pred[:60]!r}", flush=True)
    return summarize(records)


def _prf(records: list[dict]) -> dict:
    """Macro/weighted precision, recall, F1 over the ground-truth labels present."""
    support = Counter(r["answer"] for r in records)
    tp, fp = Counter(), Counter()
    for r in records:
        if r["correct"]:
            tp[r["answer"]] += 1
        else:
            fp[r["pred"]] += 1
    per, n = {}, max(len(records), 1)
    for lab, sup in support.items():
        p = tp[lab] / (tp[lab] + fp[lab]) if (tp[lab] + fp[lab]) else 0.0
        rc = tp[lab] / sup
        per[lab] = (p, rc, 2 * p * rc / (p + rc) if (p + rc) else 0.0, sup)
    k = max(len(per), 1)
    return {
        "precision_macro": sum(v[0] for v in per.values()) / k,
        "recall_macro": sum(v[1] for v in per.values()) / k,
        "f1_macro": sum(v[2] for v in per.values()) / k,
        "f1_weighted": sum(v[2] * v[3] for v in per.values()) / n,
    }


def summarize(records: list[dict]) -> dict:
    n = len(records)
    per_label = Counter(r["answer"] for r in records)
    per_label_correct = Counter(r["answer"] for r in records if r["correct"])
    return {
        "n": n,
        "accuracy": sum(r["correct"] for r in records) / max(n, 1),
        "format_rate": sum(r["has_answer_tag"] for r in records) / max(n, 1),
        "forced_answer_rate": sum(r.get("forced", False) for r in records) / max(n, 1),
        # accuracy of always predicting the most common label; real accuracy is only
        # meaningful relative to this (KEGG is heavily imbalanced)
        "majority_baseline": (per_label.most_common(1)[0][1] / n) if n else 0.0,
        **_prf(records),
        "per_label": {
            k: {"n": v, "correct": per_label_correct[k]} for k, v in per_label.most_common()
        },
        "records": records,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", help="SFT (or projector) checkpoint dir containing model.safetensors")
    ap.add_argument("--grpo-adapter", help="GRPO lora_adapter.pt to merge on top of --ckpt")
    ap.add_argument("--data", required=True, help="KEGG split saved with save_to_disk (helpers/download_bioreason_dna_datasets.py)")
    ap.add_argument("--output", required=True, help="JSON file for metrics + per-example records")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--dna-mode", choices=DNA_MODES, default="real",
                    help="DNA ablation: real, shuffled (another item's DNA), or blank (constant N)")
    args, overrides = ap.parse_known_args()

    from datasets import load_from_disk

    ds = load_from_disk(args.data)
    model, cfg = load_eval_model(args.ckpt, args.grpo_adapter, overrides)
    res = evaluate(model, cfg, ds, limit=args.limit, max_new_tokens=args.max_new_tokens,
                   dna_mode=args.dna_mode)
    res.update({"ckpt": args.ckpt, "grpo_adapter": args.grpo_adapter, "data": args.data,
                "dna_mode": args.dna_mode})
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(res, f, indent=1)
    print(f"[eval] dna={args.dna_mode} accuracy={res['accuracy']:.4f} f1_macro={res['f1_macro']:.4f} "
          f"majority={res['majority_baseline']:.4f} n={res['n']} "
          f"format_rate={res['format_rate']:.3f} forced={res['forced_answer_rate']:.3f} -> {args.output}")


if __name__ == "__main__":
    main()
