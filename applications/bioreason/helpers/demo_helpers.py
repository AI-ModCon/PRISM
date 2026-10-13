"""Display + PBS plumbing for bioreason_demo.ipynb.

Everything PRISM-specific (configs, model, data, tokens) lives in the notebook
itself so it can be read top to bottom; this module only holds the
presentation helpers (HTML renderers, plots, widgets) and the qsub/qstat
wrappers that would otherwise bury the interesting cells.
"""

from __future__ import annotations

import difflib
import html
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Palette (validated reference categorical slots 1-3 + neutral inks)
# ---------------------------------------------------------------------------
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # blue, orange, aqua
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
# Nucleotide colors (A/C/G/T) for sequence rendering -- identity only, always
# shown next to the letter itself so color is never the sole carrier.
BASE_COLORS = {"A": "#1baf7a", "C": "#2a78d6", "G": "#eda100", "T": "#e34948", "N": "#8a8984"}


def style_ax(ax, title=None, xlabel=None, ylabel=None):
    """Recessive axes: no top/right spines, light y-grid, muted tick ink."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, loc="left", fontsize=11, color=INK)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    return ax


# ---------------------------------------------------------------------------
# Source excerpts
# ---------------------------------------------------------------------------
def show_source(path: str | Path, start: int | None = None, end: int | None = None,
                pattern: str | None = None, context: int = 25, repo: str | Path = "."):
    """Render a numbered excerpt of a repo file as a Markdown code block.

    Either give explicit ``start``/``end`` lines or a regex ``pattern``: the
    excerpt then starts at its first match and spans ``context`` lines, so the
    notebook does not break when line numbers drift.
    """
    from IPython.display import Markdown, display

    p = Path(repo) / path
    lines = p.read_text().splitlines()
    if pattern is not None:
        rx = re.compile(pattern)
        hit = next((i for i, l in enumerate(lines) if rx.search(l)), None)
        if hit is None:
            raise ValueError(f"{pattern!r} not found in {path}")
        start, end = hit + 1, min(len(lines), hit + context)
    start = start or 1
    end = end or len(lines)
    body = "\n".join(f"{i:5d}  {lines[i - 1]}" for i in range(start, end + 1))
    lang = {".py": "python", ".json": "json", ".yaml": "yaml", ".sh": "bash"}.get(p.suffix, "")
    display(Markdown(f"`{path}:{start}-{end}`\n```{lang}\n{body}\n```"))


# ---------------------------------------------------------------------------
# DNA sample rendering
# ---------------------------------------------------------------------------
def diff_spans(ref: str, var: str):
    """Opcodes (tag, i1, i2, j1, j2) where ref and var differ (SNV / indel)."""
    sm = difflib.SequenceMatcher(None, ref, var, autojunk=False)
    return [op for op in sm.get_opcodes() if op[0] != "equal"]


def _bases_html(seq: str, offset: int, highlight: set[int]) -> str:
    out = []
    for k, ch in enumerate(seq):
        c = BASE_COLORS.get(ch.upper(), BASE_COLORS["N"])
        style = f"color:{c};"
        if offset + k in highlight:
            style += "background:#fde68a;font-weight:700;outline:1px solid #b45309;"
        out.append(f'<span style="{style}">{html.escape(ch)}</span>')
    return "".join(out)


def dna_pair_html(ref: str, var: str, window: int = 120) -> str:
    """Ref vs variant around the first difference, differing bases highlighted."""
    ops = diff_spans(ref, var)
    if ops:
        _, i1, i2, j1, j2 = ops[0]
    else:
        i1 = i2 = j1 = j2 = len(ref) // 2
    a0, a1 = max(0, i1 - window), min(len(ref), i2 + window)
    b0, b1 = max(0, j1 - window), min(len(var), j2 + window)
    ref_hl = {i for op in ops for i in range(op[1], op[2])}
    var_hl = {j for op in ops for j in range(op[3], op[4])}
    kinds = Counter(op[0] for op in ops)
    summary = ", ".join(f"{v} {k}" for k, v in kinds.items()) or "identical"
    mono = "font-family:ui-monospace,Menlo,monospace;font-size:12px;word-break:break-all;line-height:1.5"
    return (
        f'<div style="{mono}">'
        f'<div style="color:{INK_2}">ref[{a0}:{a1}] of {len(ref):,} bp  |  var[{b0}:{b1}] of {len(var):,} bp'
        f'  |  diff: {summary}</div>'
        f'<div><b>REF</b> {_bases_html(ref[a0:a1], a0, ref_hl)}</div>'
        f'<div><b>VAR</b> {_bases_html(var[b0:b1], b0, var_hl)}</div></div>'
    )


def sample_html(item: dict) -> str:
    ref = item.get("reference_sequence", "")
    var = item.get("variant_sequence") or item.get("mutated_sequence", "")
    parts = [dna_pair_html(ref, var)]
    for key in ("question", "reasoning", "answer"):
        if item.get(key):
            text = html.escape(str(item[key]))
            if key == "reasoning" and len(text) > 1500:
                text = text[:1500] + " …"
            parts.append(f'<p><b style="color:{INK_2}">{key}</b><br>{text}</p>')
    return "".join(parts)


def sample_browser(datasets: dict):
    """Dropdown (dataset) + slider (row) -> rendered sample."""
    import ipywidgets as w
    from IPython.display import HTML, display

    names = list(datasets)
    dd = w.Dropdown(options=names, description="dataset")
    sl = w.IntSlider(min=0, max=len(datasets[names[0]]) - 1, description="row",
                     continuous_update=False, layout=w.Layout(width="60%"))
    out = w.Output()

    def render(*_):
        sl.max = len(datasets[dd.value]) - 1
        with out:
            out.clear_output(wait=True)
            display(HTML(sample_html(datasets[dd.value][sl.value])))

    dd.observe(render, "value")
    sl.observe(render, "value")
    render()
    display(w.VBox([w.HBox([dd, sl]), out]))


# ---------------------------------------------------------------------------
# Dataset statistics
# ---------------------------------------------------------------------------
def length_frame(datasets: dict, llm_tok, dna_tok, max_rows: int = 2000):
    """Per-row lengths: bp, DNA tokens (6-mer NT), question/answer/reasoning LLM tokens."""
    import pandas as pd

    rows = []
    for name, ds in datasets.items():
        n = min(len(ds), max_rows)
        sub = ds.select(range(n))
        for item in sub:
            var = item.get("variant_sequence") or item.get("mutated_sequence", "")
            rows.append({
                "dataset": name,
                "ref_bp": len(item["reference_sequence"]),
                "var_bp": len(var),
                "ref_dna_tokens": len(dna_tok(item["reference_sequence"]).input_ids),
                "question_tokens": len(llm_tok(item.get("question", "")).input_ids),
                "answer_tokens": len(llm_tok(str(item.get("answer", ""))).input_ids),
                "reasoning_tokens": len(llm_tok(item.get("reasoning", "") or "").input_ids),
            })
    return pd.DataFrame(rows)


def plot_lengths(df, max_dna_length: int = 1024):
    """DNA span length vs the fixed padding, plus text-length histograms (one measure per panel)."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(17, 3.4), gridspec_kw={"width_ratios": [1.2, 1, 1, 1]})
    names = list(df["dataset"].unique())

    # Panel 1: DNA tokens are ~constant per dataset (fixed-length windows), so a
    # bar per dataset against the padded span length says more than a histogram.
    ax = axes[0]
    med = df.groupby("dataset", sort=False)["ref_dna_tokens"].max().reindex(names)
    ax.barh(names[::-1], med.values[::-1], color=SERIES[0], height=0.55)
    ax.axvline(max_dna_length, color=INK, linestyle="--", linewidth=1)
    for y, v in enumerate(med.values[::-1]):
        ax.text(v, y, f" {v:.0f} tok ({1 - v / max_dna_length:.0%} padding)", va="center",
                fontsize=8, color=INK)
    ax.set_xlim(0, max_dna_length * 1.08)
    style_ax(ax, title=f"DNA tokens per span (max) vs max_dna_length={max_dna_length}")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)

    panels = [
        ("question_tokens", "Question tokens (OLMo)"),
        ("reasoning_tokens", "Reasoning tokens (OLMo, KEGG only)"),
        ("answer_tokens", "Answer tokens (OLMo)"),
    ]
    for ax, (col, title) in zip(axes[1:], panels):
        for i, name in enumerate(names):
            vals = df.loc[df.dataset == name, col]
            if vals.max() == 0:
                continue
            ax.hist(vals, bins=40, alpha=0.6, color=SERIES[i % 3], label=name,
                    edgecolor="white", linewidth=0.5)
        style_ax(ax, title=title, ylabel="rows" if col == "question_tokens" else None)
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def plot_counts(counts: dict, title: str):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 0.45 * len(counts) + 0.8))
    names = list(counts)[::-1]
    vals = [counts[n] for n in names]
    ax.barh(names, vals, color=SERIES[0], height=0.6)
    for y, v in enumerate(vals):
        ax.text(v, y, f" {v:,}", va="center", fontsize=9, color=INK)
    style_ax(ax, title=title)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Model introspection
# ---------------------------------------------------------------------------
def module_tree(model, max_depth: int = 2) -> str:
    """Compact module tree: name, class, #params, down to ``max_depth``."""
    lines = []

    def walk(mod, name, depth):
        n = sum(p.numel() for p in mod.parameters())
        lines.append(f"{'  ' * depth}{name or '<root>'}: {type(mod).__name__} ({n / 1e6:,.1f}M)")
        if depth < max_depth:
            for child_name, child in mod.named_children():
                walk(child, child_name, depth + 1)

    walk(model, "", 0)
    return "\n".join(lines)


def _group(name: str) -> str:
    if "lora_" in name:
        return "LoRA adapters"
    if name.startswith("encoders.dna.proj"):
        return "DNA encoder output Linear (2560->d_dna)"
    if name.startswith("encoders.dna"):
        return "DNA encoder (NT)"
    if name.startswith("projectors.dna"):
        return "DNA projector (MLP)"
    if name.startswith("projectors"):
        return "other projectors"
    if "embed_tokens" in name or "lm_head" in name:
        return "LLM embeddings / lm_head"
    if name.startswith("backbone"):
        return "LLM backbone"
    return "other"


def param_table(model):
    """Total / trainable parameters per component."""
    import pandas as pd

    agg: dict[str, list[int]] = {}
    for n, p in model.named_parameters():
        g = agg.setdefault(_group(n), [0, 0])
        g[0] += p.numel()
        g[1] += p.numel() if p.requires_grad else 0
    df = pd.DataFrame(
        [(k, v[0], v[1]) for k, v in agg.items()], columns=["component", "params", "trainable"]
    ).set_index("component")
    df.loc["TOTAL"] = df.sum()
    df["trainable %"] = (100 * df.trainable / df.params.clip(lower=1)).round(2)
    return df


def side_by_side(*frames, titles=()):
    from IPython.display import HTML, display

    cells = []
    for i, f in enumerate(frames):
        t = titles[i] if i < len(titles) else ""
        tbl = f.style.format({"params": "{:,.0f}", "trainable": "{:,.0f}"}).to_html()
        cells.append(f'<div style="margin-right:2em"><b>{html.escape(t)}</b>{tbl}</div>')
    display(HTML(f'<div style="display:flex;flex-wrap:wrap">{"".join(cells)}</div>'))


# ---------------------------------------------------------------------------
# Tokenization view
# ---------------------------------------------------------------------------
def token_view(tokenizer, ids, prompt_len: int | None, special: dict[int, str],
               dna_tokens_per_span: int) -> str:
    """Render token ids: prompt (masked) vs target (loss) and DNA splice slots."""
    out = []
    for i, t in enumerate(ids):
        t = int(t)
        if t in special:
            label = html.escape(special[t])
            extra = (f' <span style="color:{INK_2}">[{dna_tokens_per_span} DNA embeddings]</span>'
                     if special[t].endswith("_start>") else "")
            out.append(f'<span style="background:#dbeafe;border:1px solid {SERIES[0]};'
                       f'border-radius:3px;padding:0 2px"><b>{label}</b></span>{extra}')
            continue
        piece = html.escape(tokenizer.decode([t]))
        in_target = prompt_len is not None and i >= prompt_len
        style = ("background:#dcfce7;" if in_target else f"color:{INK_2};")
        out.append(f'<span style="{style}">{piece}</span>')
    legend = (f'<div style="margin-bottom:6px;font-size:12px">'
              f'<span style="color:{INK_2}">grey = prompt (label -100, no loss)</span> &nbsp; '
              f'<span style="background:#dcfce7">green = target (loss)</span> &nbsp; '
              f'<span style="background:#dbeafe;border:1px solid {SERIES[0]}">blue = DNA splice tag</span></div>')
    return (legend + '<div style="font-family:ui-monospace,Menlo,monospace;font-size:12px;'
            'white-space:pre-wrap;line-height:1.6">' + "".join(out) + "</div>")


# ---------------------------------------------------------------------------
# PBS: patch, submit, monitor
# ---------------------------------------------------------------------------
def patch_stage_script(src: str | Path, dst: str | Path) -> Path:
    """Copy a submit_bioreason_{projector,sft}.sh with two notebook-side fixes.

    * Checkpoints go to ``$HYDRA_RUN_DIR`` (on /flare) instead of Hydra's
      default ``outputs/`` under the home-directory checkout, which hit
      "Disk quota exceeded" at the first save. ``$EXTRA_HYDRA`` is appended
      verbatim for any further overrides.
    * Drops the repo ``.venv`` activation (a CUDA torch build) that the
      projector script layers on top of the XPU venv.

    Idempotent: a script that already passes ``hydra.run.dir`` (and defines
    ``HYDRA_RUN_DIR``) is only stripped of the ``.venv`` line. The originals
    are left untouched.
    """
    text = Path(src).read_text()
    text = text.replace('source "${PROJ_DIR}/.venv/bin/activate"\n', "")
    if "hydra.run.dir=" not in text:
        last_arg = "    wandb.mode='\"${WANDB_MODE}\"'\n'"
        if last_arg not in text:
            raise ValueError(f"{src}: could not find the train.py argument list to extend")
        text = text.replace(
            last_arg,
            "    wandb.mode='\"${WANDB_MODE}\"' \\\n"
            "    hydra.run.dir='\"${HYDRA_RUN_DIR}\"' '\"${EXTRA_HYDRA:-}\"'\n'",
        )
    if not re.search(r"^HYDRA_RUN_DIR=", text, re.M):
        # Must exist in the outer shell before the mpiexec string is built.
        text = text.replace(
            'cd "${PROJ_DIR}"\n',
            'HYDRA_RUN_DIR="${HYDRA_RUN_DIR:-${PRISM_OUTPUT_ROOT}/${RUN_ID}/${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}}"\n'
            'cd "${PROJ_DIR}"\n',
            1,
        )
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(text)
    dst.chmod(0o755)
    return dst


def qsub(script, env: dict | None = None, depend: str | None = None, log: str | None = None,
         queue: str | None = None, walltime: str | None = None, name: str | None = None,
         cwd: str | None = None, dry_run: bool = False) -> str:
    """Submit ``script`` and return the job id. ``env`` is passed with -v."""
    cmd = ["qsub"]
    if env:
        bad = [k for k, v in env.items() if "," in str(v)]
        if bad:
            raise ValueError(f"qsub -v values cannot contain commas: {bad}")
        cmd += ["-v", ",".join(f"{k}={v}" for k, v in env.items())]
    if depend:
        cmd += ["-W", f"depend=afterok:{depend}"]
    if log:
        Path(log).parent.mkdir(parents=True, exist_ok=True)
        cmd += ["-o", str(log)]
    if queue:
        cmd += ["-q", queue]
    if walltime:
        cmd += ["-l", f"walltime={walltime}"]
    if name:
        cmd += ["-N", name]
    cmd.append(str(script))
    print(" ".join(cmd))
    if dry_run:
        return "DRYRUN"
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if res.returncode != 0:
        raise RuntimeError(res.stderr.strip() or res.stdout.strip())
    return res.stdout.strip()


def qstat_table(jobids):
    """State of each job (works for finished jobs too via -x)."""
    import pandas as pd

    ids = [j for j in jobids if j and j != "DRYRUN"]
    if not ids:
        return pd.DataFrame()
    res = subprocess.run(["qstat", "-x", "-f", "-F", "json", *ids], capture_output=True, text=True)
    try:
        jobs = json.loads(res.stdout).get("Jobs", {})
    except json.JSONDecodeError:
        print(res.stderr or res.stdout)
        return pd.DataFrame()
    state = {"Q": "queued", "R": "running", "H": "held", "F": "finished", "E": "exiting", "B": "array"}
    rows = []
    for jid, j in jobs.items():
        rows.append({
            "job": jid.split(".")[0],
            "name": j.get("Job_Name"),
            "queue": j.get("queue"),
            "state": state.get(j.get("job_state"), j.get("job_state")),
            "exit": j.get("Exit_status", ""),
            "walltime used": j.get("resources_used", {}).get("walltime", ""),
            "depends on": (j.get("depend") or "").replace("afterok:", "").split(".")[0],
        })
    return pd.DataFrame(rows).set_index("job")


def tail(path, n: int = 30) -> str:
    p = Path(path)
    if not p.exists():
        return f"(no log yet: {p})"
    return "\n".join(p.read_text(errors="replace").splitlines()[-n:])


_ZONE_A = re.compile(r"Step (\d+): Loss ([-\d.]+)")
_GRPO = re.compile(r"step=(\d+) loss=([-\d.]+) avg_reward=([-\d.]+)")


def parse_log(path) -> dict:
    """{'step': [...], 'loss': [...], 'reward': [...]} from a train.py log."""
    out = {"step": [], "loss": [], "reward": []}
    p = Path(path)
    if not p.exists():
        return out
    for line in p.read_text(errors="replace").splitlines():
        m = _GRPO.search(line)
        if m:
            out["step"].append(int(m[1]))
            out["loss"].append(float(m[2]))
            out["reward"].append(float(m[3]))
            continue
        m = _ZONE_A.search(line)
        if m:
            out["step"].append(int(m[1]))
            out["loss"].append(float(m[2]))
    return out


def plot_logs(logs: dict):
    """One small multiple per run (each its own y-scale; never a dual axis)."""
    import matplotlib.pyplot as plt

    runs = {k: parse_log(v) for k, v in logs.items()}
    panels = [(k, "loss", r) for k, r in runs.items()]
    panels += [(k, "reward", r) for k, r in runs.items() if r["reward"]]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3), squeeze=False)
    for ax, (name, key, r) in zip(axes[0], panels):
        if r[key]:
            ax.plot(r["step"], r[key], color=SERIES[0] if key == "loss" else SERIES[2], linewidth=2)
            ax.annotate(f"{r[key][-1]:.3f}", (r["step"][-1], r[key][-1]), fontsize=9, color=INK,
                        xytext=(4, 0), textcoords="offset points", va="center")
            style_ax(ax, title=f"{name}: {key}", xlabel="step")
        else:
            ax.set_axis_off()
            ax.set_title(f"{name}: {key}", loc="left", fontsize=11, color=INK)
            ax.text(0.5, 0.5, "no steps logged yet", ha="center", va="center",
                    transform=ax.transAxes, color=INK_2)
    fig.tight_layout()
    return fig


def monitor(jobids, logs: dict, n_tail: int = 15):
    """Refresh button: qstat table + log tails + loss curves."""
    import ipywidgets as w
    import matplotlib.pyplot as plt
    from IPython.display import display

    btn = w.Button(description="Refresh", icon="refresh")
    out = w.Output()

    def refresh(*_):
        with out:
            out.clear_output(wait=True)
            display(qstat_table(jobids))
            fig = plot_logs(logs)
            display(fig)
            plt.close(fig)
            for name, path in logs.items():
                print(f"--- {name}: {path}")
                print(tail(path, n_tail))

    btn.on_click(refresh)
    refresh()
    display(w.VBox([btn, out]))
