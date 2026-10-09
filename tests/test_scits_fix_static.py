"""Static regression guards for the PR #129 (SciTS/TimeOmni) fix PR.

No torch import (import cost ~43s on the Lustre venv) — these check source
text, launcher dry-run output, and shell syntax only. Keep this file fast.
"""

import ast
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.timeseries]

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Apostrophe-in-heredoc regression guard (PR #33/#116/#129 class of bug).
# A stray apostrophe in a comment inside the single-quoted mpiexec heredoc
# terminates the shell string early and silently breaks all but rank 0.
# ---------------------------------------------------------------------------


def _extract_mpiexec_heredoc(script_text: str) -> str:
    """Return the body of the single-quoted `mpiexec ... bash -lc '...'`
    region emitted by launch_aurora_web.py, from the opening `'` after
    `bash -lc` to the line that is exactly a closing `'`."""
    lines = script_text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if "bash -lc '" in line:
            start = i
            break
    assert start is not None, "could not find `bash -lc '` opening in generated script"

    end = None
    for i in range(start + 1, len(lines)):
        if lines[i].strip() == "'":
            end = i
            break
    assert end is not None, "could not find closing `'` for mpiexec heredoc"
    return "\n".join(lines[start:end])


def test_launch_aurora_web_source_has_no_apostrophe_in_mpiexec_heredoc():
    """Static guard on the launcher SOURCE (not a generated script): every
    comment line inside the single-quoted mpiexec heredoc region must be
    free of apostrophes. This is the exact bug class from PR #116 —
    `daos.py's` in a comment silently broke every rank but rank 0."""
    source = (REPO_ROOT / "tools" / "launch_aurora_web.py").read_text()
    body = _extract_mpiexec_heredoc(source)
    offending = [
        line for line in body.splitlines()
        if line.strip().startswith("#") and "'" in line
    ]
    assert not offending, (
        "Apostrophe found in a comment inside the single-quoted mpiexec "
        f"heredoc (breaks the shell string early): {offending}"
    )


@pytest.mark.launcher
@pytest.mark.aurora
def test_generated_scits_script_passes_bash_syntax_check(tmp_path):
    """bash -n on the FULL generated PBS script, plus explicit syntax-check
    of the isolated mpiexec heredoc body on its own (a syntax error inside
    a single-quoted heredoc is invisible to `bash -n` on the outer script,
    since the shell never has to parse the string's contents)."""
    if shutil.which("bash") is None:
        pytest.skip("bash not available")

    design_file = tmp_path / "design.yaml"
    design_file.write_text(
        "\n".join(
            [
                "experiments:",
                "  - id: TEST-SCITS-DESIGN",
                '    name: "test scits design"',
                "    resources:",
                "      ngpus: 1",
                '      walltime: "00:10:00"',
                "    common_overrides:",
                '      training.task: "vlm"',
                '      model.backbone_id: "allenai/OLMo-1B-0724-hf"',
            ]
        )
    )
    webdataset_root = tmp_path / "webdataset"
    (webdataset_root / "shards").mkdir(parents=True)
    (webdataset_root / "manifest.json").write_text(
        '{"num_shards": 1, "total_written": 1}'
    )
    shared_hf_home = tmp_path / "hf_hub"
    shared_hf_home.mkdir()

    launcher = REPO_ROOT / "tools" / "launch_aurora_web.py"
    cmd = [
        sys.executable,
        str(launcher),
        "--file",
        str(design_file),
        "--design",
        "TEST-SCITS-DESIGN",
        "--id",
        "SCITS_FIX_TEST",
        "--webdataset-dir",
        str(webdataset_root),
        "--webdataset-modality",
        "time_series",
        "--shared-hf-home",
        str(shared_hf_home),
        "--dry-run",
    ]
    result = subprocess.run(
        cmd, cwd=tmp_path, check=False, text=True, capture_output=True
    )
    assert result.returncode == 0, (
        f"launcher failed:\nstdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
    )

    generated_line = next(
        (
            line for line in result.stdout.splitlines()
            if line.startswith("Generated Run Script: ")
        ),
        None,
    )
    assert generated_line is not None, f"missing generated script line:\n{result.stdout}"
    script_path = tmp_path / generated_line.split(": ", 1)[1].strip()
    script_text = script_path.read_text()

    # 1. Full-script syntax check.
    full_check = subprocess.run(
        ["bash", "-n", str(script_path)], capture_output=True, text=True
    )
    assert full_check.returncode == 0, f"bash -n failed:\n{full_check.stderr}"

    # 2. Isolated heredoc body syntax check — write the body to its own file
    # (unwrapped from the outer single-quotes) and syntax-check it directly,
    # since bash -n on the outer script never actually parses the quoted
    # string's contents.
    heredoc_body = _extract_mpiexec_heredoc(script_text)
    inner_lines = heredoc_body.splitlines()[1:]  # drop the `... bash -lc '` line
    inner_script = tmp_path / "heredoc_body.sh"
    inner_script.write_text("\n".join(inner_lines))
    inner_check = subprocess.run(
        ["bash", "-n", str(inner_script)], capture_output=True, text=True
    )
    assert inner_check.returncode == 0, (
        f"bash -n on isolated mpiexec heredoc body failed:\n{inner_check.stderr}"
    )


@pytest.mark.launcher
@pytest.mark.aurora
def test_launcher_prints_validation_shard_count_for_scits_dir(tmp_path):
    """Defect A/B regression guard at the launcher layer: --webdataset-dir
    pointing at a SciTS-style directory (shards/ + val_shards/) must wire
    PRISM_VAL_SHARDS_DIR and print the shard count, so a training job
    actually gets a validation loader instead of silently training with
    none."""
    design_file = tmp_path / "design.yaml"
    design_file.write_text(
        "\n".join(
            [
                "experiments:",
                "  - id: TEST-SCITS-VAL-DESIGN",
                '    name: "test scits val design"',
                "    resources:",
                "      ngpus: 1",
                '      walltime: "00:10:00"',
                "    common_overrides:",
                '      training.task: "vlm"',
                '      model.backbone_id: "allenai/OLMo-1B-0724-hf"',
            ]
        )
    )
    webdataset_root = tmp_path / "webdataset"
    (webdataset_root / "shards").mkdir(parents=True)
    val_shards_dir = webdataset_root / "val_shards"
    val_shards_dir.mkdir()
    (val_shards_dir / "scits-000000.tar").write_bytes(b"")
    (val_shards_dir / "scits-000001.tar").write_bytes(b"")
    (webdataset_root / "manifest.json").write_text(
        '{"num_shards": 1, "total_written": 1}'
    )
    shared_hf_home = tmp_path / "hf_hub"
    shared_hf_home.mkdir()

    launcher = REPO_ROOT / "tools" / "launch_aurora_web.py"
    cmd = [
        sys.executable,
        str(launcher),
        "--file",
        str(design_file),
        "--design",
        "TEST-SCITS-VAL-DESIGN",
        "--id",
        "SCITS_VAL_TEST",
        "--webdataset-dir",
        str(webdataset_root),
        "--webdataset-modality",
        "time_series",
        "--shared-hf-home",
        str(shared_hf_home),
        "--dry-run",
    ]
    result = subprocess.run(
        cmd, cwd=tmp_path, check=False, text=True, capture_output=True
    )
    assert result.returncode == 0, (
        f"launcher failed:\nstdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
    )
    assert f"Validation: 2 shard(s) at {val_shards_dir}" in result.stdout, result.stdout

    generated_line = next(
        line for line in result.stdout.splitlines()
        if line.startswith("Generated Run Script: ")
    )
    script_text = (
        tmp_path / generated_line.split(": ", 1)[1].strip()
    ).read_text()
    assert f'export PRISM_VAL_SHARDS_DIR="{val_shards_dir}"' in script_text


# ---------------------------------------------------------------------------
# Defect D: run_full_eval's key -> headline-metric map must cover every
# evaluator it iterates, and must not rely on fragile substring matching
# (the "time" in "ts_scits" bug — "time" is not a substring of "ts_scits").
# ---------------------------------------------------------------------------


def test_run_full_eval_headline_metric_map_covers_every_evaluator_key():
    source = (REPO_ROOT / "tools" / "universal_evaluator.py").read_text()

    func_match = re.search(
        r"def run_full_eval\(.*?\n(?=\ndef )", source, re.DOTALL
    )
    assert func_match, "could not find run_full_eval function body"
    func_body = func_match.group(0)

    evaluators_match = re.search(
        r"evaluators\s*=\s*\[(.*?)\]", func_body, re.DOTALL
    )
    assert evaluators_match, "could not find `evaluators = [...]` in run_full_eval"
    keys = re.findall(r'"\s*,\s*"([a-zA-Z0-9_]+)"\s*\)', evaluators_match.group(1))
    assert keys, f"could not parse evaluator keys from: {evaluators_match.group(1)!r}"

    map_match = re.search(
        r"_HEADLINE_METRIC\s*=\s*\{(.*?)\}", source, re.DOTALL
    )
    assert map_match, "could not find _HEADLINE_METRIC map in universal_evaluator.py"
    mapped_keys = set(re.findall(r'"([a-zA-Z0-9_]+)"\s*:', map_match.group(1)))

    missing = [k for k in keys if k not in mapped_keys]
    assert not missing, (
        f"evaluator key(s) {missing} in run_full_eval's `evaluators` list have "
        "no entry in _HEADLINE_METRIC — they will silently report score='N/A'"
    )

    # The exact regression this guards: ts_scits contains no substring "time",
    # so a `"time" in key` check (the old code) always misses it.
    assert "ts_scits" in mapped_keys
    assert "time" not in "ts_scits"


# ---------------------------------------------------------------------------
# Defect E: every --mode choice in universal_evaluator.py's argparser must
# have a live dispatch branch (`args.mode == "..."`) — no stale mode names
# left behind after a rename, and no in-tree caller still passing a mode
# that was removed from `choices`.
# ---------------------------------------------------------------------------


def test_every_evaluator_mode_choice_has_a_dispatch_branch():
    source = (REPO_ROOT / "tools" / "universal_evaluator.py").read_text()

    choices_match = re.search(r'choices=\[(.*?)\]', source)
    assert choices_match, "could not find --mode choices=[...] in universal_evaluator.py"
    choices = re.findall(r'"([a-zA-Z0-9_]+)"', choices_match.group(1))
    assert choices, f"could not parse choices from: {choices_match.group(1)!r}"

    dispatched = set(re.findall(r'args\.mode\s*==\s*"([a-zA-Z0-9_]+)"', source))
    missing = [c for c in choices if c not in dispatched]
    assert not missing, (
        f"--mode choice(s) {missing} have no `args.mode == \"...\"` dispatch "
        "branch in universal_evaluator.py"
    )


def test_no_stale_verify_timeseries_caller_outside_choices():
    """Defect E: `verify_timeseries` (without a `_scits`/`_interleave` suffix)
    was removed from the --mode choices. No in-tree caller (scripts, docs,
    CLI wrapper) may still pass the bare removed name."""
    choices_source = (REPO_ROOT / "tools" / "universal_evaluator.py").read_text()
    choices_match = re.search(r'choices=\[(.*?)\]', choices_source)
    choices = set(re.findall(r'"([a-zA-Z0-9_]+)"', choices_match.group(1)))
    assert "verify_timeseries" not in choices  # sanity: confirms the bare name is gone

    callers = [
        REPO_ROOT / "scripts" / "perlmutter" / "run_eval_job_fixed.sh",
        REPO_ROOT / "src" / "cli" / "eval.py",
        REPO_ROOT / "docs" / "evaluation.md",
        REPO_ROOT / "docs" / "cli.md",
    ]
    pattern = re.compile(r"verify_timeseries(?!_scits|_interleave)\b")
    offenders = []
    for path in callers:
        if not path.exists():
            continue
        for line_no, line in enumerate(path.read_text().splitlines(), start=1):
            if pattern.search(line):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{line_no}: {line.strip()}")
    assert not offenders, (
        "found reference(s) to the removed bare `verify_timeseries` mode "
        f"(should be verify_timeseries_scits or verify_timeseries_interleave): {offenders}"
    )


# ---------------------------------------------------------------------------
# Defect K: no glob.glob() on shard directories across the PR's touched
# files — dfuse/DAOS-mounted directories can hang glob.glob().
# ---------------------------------------------------------------------------


def test_no_glob_glob_on_shard_directories_in_scits_files():
    touched_files = [
        REPO_ROOT / "src" / "training" / "trainer_native.py",
        REPO_ROOT / "src" / "eval" / "tasks" / "modality_tasks.py",
        REPO_ROOT / "tools" / "universal_evaluator.py",
        REPO_ROOT / "applications" / "timeseries" / "convert_scits_to_webdataset.py",
    ]
    offenders = []
    for path in touched_files:
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "glob"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in {"glob", "_glob"}
            ):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
    assert not offenders, (
        f"glob.glob() call(s) found on shard-directory code paths (can hang "
        f"on dfuse/DAOS mounts): {offenders}"
    )


# ---------------------------------------------------------------------------
# Defect L: os.environ.get(VAR, default) with an empty-string VAR must not
# silently glob the current working directory. Statically verify the fixed
# call sites use `or default` instead of the two-arg .get() form.
# ---------------------------------------------------------------------------


def test_prism_val_shards_dir_reads_use_or_fallback_not_get_default():
    touched_files = [
        REPO_ROOT / "src" / "eval" / "tasks" / "modality_tasks.py",
        REPO_ROOT / "tools" / "universal_evaluator.py",
    ]
    offenders = []
    for path in touched_files:
        for line_no, line in enumerate(path.read_text().splitlines(), start=1):
            if 'os.environ.get("PRISM_VAL_SHARDS_DIR",' in line:
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{line_no}: {line.strip()}")
    assert not offenders, (
        "PRISM_VAL_SHARDS_DIR read via os.environ.get(VAR, default) — an "
        "explicitly-empty env var bypasses the default and silently globs "
        f"the CWD. Use `os.environ.get(VAR) or default` instead: {offenders}"
    )


# ---------------------------------------------------------------------------
# langchain_openai must not be a module-level import in applications/timeseries/scits_f1.py —
# it's not a declared project dependency.
# ---------------------------------------------------------------------------


def test_langchain_openai_is_not_imported_at_module_level():
    source = (REPO_ROOT / "applications" / "timeseries" / "scits_f1.py").read_text()
    tree = ast.parse(source, filename="scits_f1.py")
    for node in tree.body:  # top-level statements only (skips guarded imports
        # nested inside `if TYPE_CHECKING:` or inside function bodies)
        if isinstance(node, ast.ImportFrom) and node.module == "langchain_openai":
            pytest.fail(
                f"module-level `from langchain_openai import ...` at line "
                f"{node.lineno} — langchain_openai is not a declared "
                "dependency; every importer of this module breaks without it"
            )


# ---------------------------------------------------------------------------
# Defect C: `_emit_families` must be assigned exactly once, before the
# training loop, so both the training-loss-proxy eval block and the
# held-out-validation eval block reference the same hoisted value instead
# of one of them computing it inline inside a try/except that can leave it
# referenced-before-assignment.
# ---------------------------------------------------------------------------


def test_emit_families_assigned_once_before_training_loop():
    source = (REPO_ROOT / "src" / "training" / "trainer_native.py").read_text()
    lines = source.splitlines()

    assignment_lines = [
        i + 1 for i, line in enumerate(lines)
        if re.match(r"\s*_emit_families\s*=", line)
    ]
    assert len(assignment_lines) == 1, (
        f"expected exactly one `_emit_families = ...` assignment, found at "
        f"lines {assignment_lines}"
    )

    usage_lines = [
        i + 1 for i, line in enumerate(lines)
        if re.search(r"[^#]*\b_emit_families\b", line)
        and not line.strip().startswith("#")
        and i + 1 not in assignment_lines
    ]
    assert len(usage_lines) >= 2, (
        f"expected >=2 usages of _emit_families, found at lines {usage_lines}"
    )

    loop_start = next(
        i + 1 for i, line in enumerate(lines)
        if re.match(r"\s*while step < max_steps\s*:", line)
    )
    assert assignment_lines[0] < loop_start, (
        "_emit_families must be assigned BEFORE the training loop starts "
        f"(assigned at line {assignment_lines[0]}, loop starts at {loop_start})"
    )
    for usage_line in usage_lines:
        assert usage_line > loop_start, (
            f"_emit_families usage at line {usage_line} is before the "
            "training loop — unexpected given the hoisted-before-loop fix"
        )


def test_no_user_specific_hub_paths_in_model_configs():
    """Post-merge regression: prism_qwen3_0_6b_timeomni_ts.yaml had its
    backbone_id/tokenizer_id pinned to a single user's HF snapshot
    (/flare/ModCon/pemami/hub/models--Qwen--Qwen3-0.6B/snapshots/<sha>),
    while its three ladder siblings all use the plain hub ID.

    Compute nodes are offline, so models are staged from the shared hub by
    the launcher (--shared-hf-home). Pinning one user's cache path makes the
    config unrunnable for anyone else and silently couples the run to a
    directory outside the repo.

    Tokenizer paths under a project dir are still allowed: the interleaved TS
    configs legitimately point at custom tokenizers checked in under
    /flare/ModCon/<user>/BaseMM_PRISM/tokenizers/.
    """
    offenders = []
    for cfg in sorted((REPO_ROOT / "src" / "conf" / "model").glob("*.yaml")):
        for lineno, line in enumerate(cfg.read_text().splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if not re.match(r"^(backbone_id|tokenizer_id)\s*:", stripped):
                continue
            # A per-user HF cache path: .../hub/models--*/snapshots/<sha>
            if re.search(r"/hub/models--[^/\s\"']+/snapshots/", stripped):
                offenders.append(f"{cfg.name}:{lineno}: {stripped}")

    assert not offenders, (
        "model configs must not pin a user-specific HF snapshot path; "
        "use the hub ID and let the launcher stage it:\n  "
        + "\n  ".join(offenders)
    )
