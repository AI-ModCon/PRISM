"""Print a table from a submit_eval_sft_sweep.sh output directory of eval_kegg_aurora.py JSONs.

Usage: python applications/bioreason/helpers/summarize_eval_sweep.py <dir>
Stdlib only. Compare the real row against the shuffled/blank rows and the majority baseline.
"""

import glob
import json
import os
import re
import sys


def step_key(name: str):
    m = re.match(r"step_(\d+|final)_", name)
    return (1, 0) if m and m.group(1) == "final" else (0, int(m.group(1)) if m else -1)


def main(d: str):
    rows = []
    for p in glob.glob(os.path.join(d, "*.json")):
        r = json.load(open(p))
        rows.append((os.path.basename(p)[:-5], r))
    rows.sort(key=lambda x: (step_key(x[0]), x[0]))
    cols = ["acc", "f1_macro", "f1_wtd", "majority", "format", "n"]
    print(f"{'run':<26}{'dna':<10}" + "".join(f"{c:>10}" for c in cols))
    for name, r in rows:
        vals = [r["accuracy"], r.get("f1_macro", float("nan")), r.get("f1_weighted", float("nan")),
                r.get("majority_baseline", float("nan")), r["format_rate"], r["n"]]
        print(f"{name:<26}{r.get('dna_mode', 'real'):<10}"
              + "".join(f"{v:>10.4f}" if isinstance(v, float) else f"{v:>10}" for v in vals))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
