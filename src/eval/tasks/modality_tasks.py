import collections
import io
import os
import re
import string
import tarfile

import nltk
import torch
from datasets import load_dataset

from ..evaluator import BaseEvaluator, EvaluatorRegistry

try:
    from rdkit import Chem
except ImportError:
    Chem = None


# --- GRAPH EVALUATOR ---
@EvaluatorRegistry.register("graph_chebi")
class GraphEvaluator(BaseEvaluator):
    """
    Evaluates Graph Projector on ChEBI-20 (Molecule Captioning).
    Task: SMILES -> Description.
    Metric: BLEU (using nltk).
    """

    def __init__(self, model, tokenizer, device="cuda"):
        super().__init__(model, tokenizer, device)
        # Switched to liupf/ChEBI-20-MM due to OpenBioML validation.csv parsing errors
        self.dataset = load_dataset("liupf/ChEBI-20-MM", split="validation", streaming=True)

    def _featurize_graph(self, smiles):
        if not Chem:
            return {"x": torch.zeros(128, 32), "edge_index": torch.empty((2, 0), dtype=torch.long)}

        try:
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                return {
                    "x": torch.zeros(128, 32),
                    "edge_index": torch.empty((2, 0), dtype=torch.long),
                }

            # Node Features
            atoms = mol.GetAtoms()
            x_list = []
            for atom in atoms:
                feats = [
                    float(atom.GetAtomicNum()),
                    float(atom.GetDegree()),
                    float(atom.GetFormalCharge()),
                    float(atom.GetHybridization()),
                    float(atom.GetIsAromatic()),
                    float(atom.GetTotalNumHs()),
                ]
                feats += [0.0] * (32 - len(feats))
                x_list.append(feats)

            x_tensor = torch.tensor(x_list, dtype=torch.float)

            # Edges
            edges = []
            for bond in mol.GetBonds():
                u = bond.GetBeginAtomIdx()
                v = bond.GetEndAtomIdx()
                edges.append([u, v])
                edges.append([v, u])

            if edges:
                edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)

            # DEVICE & DTYPE FIX
            # We don't have access to model dtype easily here without 'self.model' context?
            # actually we do, self.model is in __init__.
            # But _featurize_graph is called inside evaluate loop.
            # Let's return raw tensors and move them in evaluate/generate helper.
            # OR better: cast here to safe defaults and let generate handle device?
            # Generate handles device. Dtype is the issue (Float32 vs BFloat16).
            return {"x": x_tensor, "edge_index": edge_index}
        except Exception:
            return {"x": torch.zeros(128, 32), "edge_index": torch.empty((2, 0), dtype=torch.long)}

    def evaluate(self, limit: int = 100):
        print("Evaluating ChEBI-20 (Graph)...")
        refs = []
        hyps = []

        count = 0
        for _i, item in enumerate(self.dataset):
            if limit and count >= limit:
                break
            try:
                # Robust Key Access
                smiles = item.get("SMILES", item.get("smiles", item.get("structure", "")))
                ground_truth = item.get("description", item.get("caption", item.get("text", "")))

                if not ground_truth or not smiles:
                    continue

                # Prepare Inputs
                prompt = f"Describe the following molecule/graph: {smiles}\nDescription:"
                tok_inputs = self.tokenizer(prompt, return_tensors="pt")

                inputs = {"text": tok_inputs.input_ids, "graph": self._featurize_graph(smiles)}

                output = self.generate(inputs, max_new_tokens=64)
                generated = output.replace(prompt, "").strip()

                refs.append([ground_truth.split()])
                hyps.append(generated.split())

                count += 1
            except Exception as e:
                print(f"Graph Eval Error: {e}")
                continue

        bleu = nltk.translate.bleu_score.corpus_bleu(refs, hyps)
        return {"bleu": bleu, "valid_count": count}


# --- TIME SERIES EVALUATOR ---
@EvaluatorRegistry.register("ts_timemmd")
class TimeEvaluator(BaseEvaluator):
    """
    Evaluates TS Projector on Time-MMD (QA).
    Task: QA Pair -> Answer.
    """

    def __init__(self, model, tokenizer, device="cuda"):
        super().__init__(model, tokenizer, device)
        self.dataset = load_dataset("mikeam/time-series-reasoning", split="test", streaming=True)

    def _featurize_ts(self, item):
        # Extract Series from dict/list keys
        vals = []
        # Try finding list values
        for _k, v in item.items():
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], int | float):
                vals = v
                break

        if not vals:
            return torch.randn(512, 1)

        # Tensorize
        tensor = torch.tensor(vals, dtype=torch.float).view(-1, 1)

        # Pad/Truncate to 512
        if tensor.shape[0] > 512:
            tensor = tensor[:512, :]
        elif tensor.shape[0] < 512:
            pad = torch.zeros(512 - tensor.shape[0], 1)
            tensor = torch.cat([tensor, pad], dim=0)

        # UNSQUEEZE BATCH DIM FIX
        # (S, D) -> (1, S, D)
        tensor = tensor.unsqueeze(0)

        return tensor

    def evaluate(self, limit: int = 100):
        print("Evaluating Time-MMD (Time Series)...")
        correct = 0
        count = 0

        for item in self.dataset:
            if limit and count >= limit:
                break
            try:
                q = item.get("question_text", item.get("question"))
                a = item.get("description", item.get("characteristics", item.get("answer", "")))

                if q:
                    prompt = f"User: {q}\nAssistant:"
                else:
                    prompt = "User: Describe the characteristics of this time series.\nAssistant:"

                tok_inputs = self.tokenizer(prompt, return_tensors="pt")

                inputs = {"text": tok_inputs.input_ids, "time_series": self._featurize_ts(item)}

                output = self.generate(inputs, max_new_tokens=32)
                generated = output.replace(prompt, "").strip().lower()

                if a.lower() in generated or generated in a.lower():
                    correct += 1

                count += 1
            except Exception as e:
                print(f"Time Eval Error: {e}")
                continue

        acc = correct / count if count > 0 else 0.0
        return {"accuracy": acc, "valid_count": count}


# --- SCITS EVALUATOR ---

class _SciTSShardsIterable:
    """Iterable over SciTS val shard *.tar files using stdlib tarfile only."""

    def __init__(self, val_shards_dir):
        # glob.glob() can hang on dfuse/DAOS mounts; os.listdir() + filter
        # is the safe pattern for shard directories that may live on DAOS.
        self._tars = sorted(
            os.path.join(val_shards_dir, f)
            for f in os.listdir(val_shards_dir)
            if f.endswith(".tar")
        )
        if not self._tars:
            raise ValueError(f"No *.tar shards found in {val_shards_dir}")

    def __iter__(self):
        import numpy as _np
        for tar_path in self._tars:
            with tarfile.open(tar_path) as tf:
                members = {m.name: m for m in tf.getmembers()}
                stems: dict = {}
                for name in members:
                    stem, _, ext = name.partition(".")
                    stems.setdefault(stem, {})[ext] = name
                for stem, exts in stems.items():
                    if "ts.npy" not in exts or "text" not in exts:
                        continue
                    npy_f = tf.extractfile(members[exts["ts.npy"]])
                    arr = _np.load(io.BytesIO(npy_f.read()))  # (T, V)
                    text_f = tf.extractfile(members[exts["text"]])
                    text = text_f.read().decode("utf-8")
                    if "Question:" in text and "Answer:" in text:
                        parts = text.split("Answer:", 1)
                        question = parts[0].replace("Question:", "", 1).strip()
                        answer = parts[1].strip()
                    else:
                        question = text.strip()
                        answer = ""
                    yield {
                        "__key__": stem,
                        "ts_array": arr,
                        "question": question,
                        "answer": answer,
                    }


@EvaluatorRegistry.register("ts_scits")
class SciTSEvaluator(BaseEvaluator):
    """
    Evaluates TS projector on SciTS validation shards (QA format).
    Reads *.tar shards from PRISM_VAL_SHARDS_DIR env var, falling back to the
    shards under the site's own PRISM_DATA_ROOT (see src/site_paths.py) rather
    than to one contributor's Lustre directory.
    Task: Question + time series -> Answer.
    Metric: normalized exact match (headline) plus token-level F1.
    """

    #: Resolved lazily, not at import: site_paths reads the environment and the
    #: .env file, and a module-level constant would freeze whichever values
    #: happened to be set when this module was first imported.
    _DEFAULT_VAL_SHARDS_TEMPLATE = "${PRISM_DATA_ROOT}/SciTS-processed/val_shards"

    @classmethod
    def _default_val_shards(cls):
        """The site's SciTS val-shard directory, or an "<unset:...>" marker."""
        from ...site_paths import expand

        return expand(cls._DEFAULT_VAL_SHARDS_TEMPLATE)

    def __init__(self, model, tokenizer, device="cuda"):
        from ...site_paths import require_resolved

        super().__init__(model, tokenizer, device)
        # os.environ.get(var, default) only substitutes the default when the
        # var is UNSET — an explicitly-empty PRISM_VAL_SHARDS_DIR="" passes
        # through and silently globs the current working directory.
        val_shards_dir = os.environ.get("PRISM_VAL_SHARDS_DIR") or self._default_val_shards()
        # Without this the "<unset:PRISM_DATA_ROOT>" marker reaches os.listdir
        # and surfaces as a bare FileNotFoundError naming a path nobody wrote.
        require_resolved(val_shards_dir, "SciTS validation shards")
        self.dataset = _SciTSShardsIterable(val_shards_dir)
        _cfg = getattr(model, "config", None)
        self._max_ts_len = getattr(_cfg, "max_ts_length", None) or 512
        # TimeOmni selects its patch size per sample and enforces its own token
        # budget, so it must receive the raw (T, V) series. Padding to
        # max_ts_length here would be actively harmful: under TimeOmni that
        # value is derived as max_stride * (max_patches - 1) — 407,552 with the
        # shipped config — so every short series would be zero-padded to
        # ~400k steps, defeating the dynamic patching and wasting the budget on
        # padding. Fixed-length padding only applies to linear/moirai.
        self._is_timeomni = getattr(_cfg, "ts_projector", None) == "timeomni"

    @staticmethod
    def _normalize_answer(s: str) -> str:
        """SQuAD-style normalization: lowercase, drop articles/punctuation/extra ws."""
        s = s.lower()
        s = "".join(ch for ch in s if ch not in set(string.punctuation))
        s = re.sub(r"\b(a|an|the)\b", " ", s)
        return " ".join(s.split())

    @classmethod
    def _token_f1(cls, pred: str, gold: str) -> float:
        p = cls._normalize_answer(pred).split()
        g = cls._normalize_answer(gold).split()
        if not p or not g:
            return float(p == g)
        common = collections.Counter(p) & collections.Counter(g)
        n_same = sum(common.values())
        if n_same == 0:
            return 0.0
        precision = n_same / len(p)
        recall = n_same / len(g)
        return 2 * precision * recall / (precision + recall)

    def _featurize_ts(self, item):
        import numpy as _np
        arr = item["ts_array"]  # numpy (T, V)
        if not isinstance(arr, _np.ndarray):
            arr = _np.array(arr, dtype=_np.float32)
        tensor = torch.from_numpy(arr).float()
        if tensor.dim() == 1:
            tensor = tensor.view(-1, 1)
        if self._is_timeomni:
            # Raw pass-through; the encoder patches dynamically and decimates
            # anything over budget.
            return tensor.unsqueeze(0)  # (1, T, V)
        if tensor.shape[0] > self._max_ts_len:
            tensor = tensor[: self._max_ts_len]
        elif tensor.shape[0] < self._max_ts_len:
            pad = torch.zeros(self._max_ts_len - tensor.shape[0], tensor.shape[1])
            tensor = torch.cat([tensor, pad], dim=0)
        return tensor.unsqueeze(0)  # (1, T, V)

    def evaluate(self, limit: int = 100):
        print("Evaluating SciTS (Time Series QA)...")
        em_total = 0.0
        f1_total = 0.0
        count = 0
        errors = 0
        for item in self.dataset:
            if limit and count >= limit:
                break
            try:
                question = item["question"]
                answer = item["answer"]
                prompt = f"Question: {question}\nAnswer:"
                tok_inputs = self.tokenizer(prompt, return_tensors="pt")
                inputs = {
                    "text": tok_inputs.input_ids,
                    "time_series": self._featurize_ts(item),
                }
                output = self.generate(inputs, max_new_tokens=64)
                generated = output.replace(prompt, "").strip()
                # Scored with normalized exact match + token F1. The previous
                # rule ("answer in generated or generated in answer") credited
                # any generation that was a *substring of the answer*: SciTS
                # answers are full sentences, so an untrained model emitting
                # "no" or "the" scored as correct, and the metric could not
                # distinguish a trained model from a broken one.
                em_total += float(
                    self._normalize_answer(generated) == self._normalize_answer(answer)
                )
                f1_total += self._token_f1(generated, answer)
                count += 1
            except Exception as e:
                print(f"SciTS Eval Error: {e}")
                errors += 1
                continue
        em = em_total / count if count > 0 else 0.0
        f1 = f1_total / count if count > 0 else 0.0
        # "accuracy" stays the headline key for registry/back-compat consumers,
        # but now means normalized exact match.
        return {
            "accuracy": em,
            "exact_match": em,
            "f1": f1,
            "valid_count": count,
            "errors": errors,
        }


# --- TABLE EVALUATOR ---
@EvaluatorRegistry.register("table_spider")
class TableEvaluator(BaseEvaluator):
    """
    Evaluates Table Projector on Spider (SQL Generation).
    """

    def __init__(self, model, tokenizer, device="cuda"):
        super().__init__(model, tokenizer, device)
        self.dataset = load_dataset("spider", split="validation", streaming=True)
        try:
            from transformers import TapasTokenizer

            self.tapas_tokenizer = TapasTokenizer.from_pretrained("google/tapas-base")
        except Exception:
            self.tapas_tokenizer = None

    def _featurize_table(self, item):
        # Dummy if no tokenizer
        if not self.tapas_tokenizer:
            return torch.randint(0, 30522, (128,))

        # Need to construct a DataFrame from Spider (which has 'query', 'question', 'db_id')
        # Spider doesn't have the table content in the main dictionary usually, it refers to a DB.
        # This is strictly a placeholder because loading the actual DB content is complex.
        # We will iterate and mock a table structure or skip.
        # Or check if 'table' exists? No.
        # For validation, we return a Dummy Tokenized Table to ensure pipeline works.
        # Real implementation needs spider_utils to fetch schema.

        import pandas as pd

        df = pd.DataFrame({"Col1": ["Val1"], "Col2": ["Val2"]})
        enc = self.tapas_tokenizer(
            table=df,
            queries="SQL?",
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "token_type_ids": enc["token_type_ids"],
        }

    def evaluate(self, limit: int = 100):
        print("Evaluating Spider (Table)...")
        correct = 0
        count = 0

        for item in self.dataset:
            if limit and count >= limit:
                break
            try:
                q = item["question"]
                db_id = item["db_id"]
                prompt = f"Generate SQL for Database {db_id}: {q}\nSQL:"

                truth = item["query"]

                tok_inputs = self.tokenizer(prompt, return_tensors="pt")
                inputs = {"text": tok_inputs.input_ids, "table": self._featurize_table(item)}

                output = self.generate(inputs, max_new_tokens=64)
                generated = output.replace(prompt, "").strip()

                if generated.lower() == truth.lower():
                    correct += 1

                count += 1
            except Exception:
                continue

        acc = correct / count if count > 0 else 0.0
        return {"accuracy": acc, "valid_count": count}
