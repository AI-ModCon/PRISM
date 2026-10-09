# Contributing to PRISM

Thank you for your interest in contributing to PRISM — a unified multimodal
transformer for scientific data across text, images, time series, geometry,
graphs, and tables.

This project is developed by researchers at Argonne National Laboratory and
collaborating institutions. Contributions from the community are welcome.

## Code of Conduct

This project and everyone participating in it is governed by our
[Code of Conduct](./CODE_OF_CONDUCT.md). By participating, you are expected to
uphold this code.

## How to Contribute

### Reporting Bugs

Before creating a bug report, please search existing issues — you may find the
problem is already known. When filing a new report, use the
[bug report template](.github/ISSUE_TEMPLATE/bug_report.md) and include:

- A clear and descriptive title
- The exact steps that reproduce the problem
- What you observed, and what you expected instead
- Your environment: OS, Python version, platform (Aurora / Polaris /
  Perlmutter / CUDA workstation / CPU-only), and accelerator if relevant
- The relevant Hydra config, CLI invocation, or launcher flags
- Logs or tracebacks, with any institutional paths or credentials redacted

### Suggesting Enhancements

Use the [feature request template](.github/ISSUE_TEMPLATE/feature_request.md).
Describe the current behavior, the proposed behavior, and why the change would
be useful. For anything that touches a public interface, say so explicitly —
see *Compatibility* below.

### Pull Requests

Open or reference an issue first for anything beyond a small fix. Keep each PR
focused: do not combine cleanup with interface redesign.

Use the [pull request template](.github/pull_request_template.md), which asks
you to describe:

- the user-visible behavior;
- compatibility risks and mitigations;
- tests run, and tests intentionally skipped (with the reason);
- any new dependency, data source, model, tokenizer, or asset — with provenance;
- documentation changes.

Keep commits reviewable, and do not rewrite another contributor's work without
coordinating with them first.

## Compatibility

PRISM has downstream users running long training jobs on HPC systems. Unless an
approved migration is part of the issue, preserve:

- `src.*` import paths
- the `prism` CLI command surface
- Hydra config keys
- checkpoint keys
- launcher flags

Historical recipe IDs and their ordering are also stable, including existing
duplicate-ID first-match behavior. Retiring an ID requires a separate,
approved migration.

## What Not to Commit

Do not add secrets, credentials, private filesystem paths, non-redistributable
data, model weights, generated outputs, or third-party code without a
provenance record. See [SECURITY.md](./SECURITY.md) for the full policy and
what to do if a secret is committed.

Absolute paths under `/home/<user>`, `/flare`, `/lus`, `/eagle`, or
`/global/homes` should not appear in tracked files. Use configurable paths or
environment variables instead. This applies to new and modified files; a
cleanup of pre-existing occurrences, most of them in generated job scripts and
provenance records under `docs/`, is tracked as part of release readiness.

## Repository Layout

Put reusable behavior in the core, and keep operational tooling separate from
framework code:

| Location | Contents |
|---|---|
| `src/` | Framework implementation (encoders, model, training, CLI) |
| `src/conf/` | Hydra configuration tree |
| `tools/` | General-purpose operational utilities — `tools/ci/`, `tools/parity/`, launchers |
| `scripts/` | General-purpose data preparation and cluster setup |
| `applications/` | Scripts for one specific application, grouped by application |
| `experiments/` | Experiment and ablation catalogs (YAML) |
| `tests/` | Test suite |
| `docs/` | Documentation |

Keep framework implementation in `src/`. Experiment definitions belong in
`experiments/*.yaml`, not hard-coded in Python.

For an operational script, the question is **who can reuse it**:

- Useful across applications → `tools/` (launchers, evaluators, CI and
  parity harnesses, environment builders) or `scripts/` (data preparation,
  cluster setup).
- Useful only to one application → `applications/<name>/`.

The current groups are `bioreason/`, `timeseries/`, `vision_language/`,
`material_science/`, `text/` and `vla/`, matching the subjects in
[`docs/applications/`](./docs/applications/). Add a directory when a second
script for a new application appears, not for the first.

The test is what the script *serves*, not what it mentions.
`tools/universal_evaluator.py` names KEGG and DOCCI but evaluates every
benchmark, so it is general. `applications/bioreason/submit_bioreason_sft.sh`
runs one training job for one application, so it is not.

Note `tools/` and `scripts/` are not cleanly separated from each other —
data conversion lives in both. That is historical. Do not add to the
confusion; prefer the directory where similar work already sits.

Historical experiment IDs and their ordering are stable — see *Compatibility*.

## Development Setup

1. Fork the repository, then clone your fork **with submodules**:

   ```bash
   git clone --recursive https://github.com/<your-username>/BaseMM_PRISM.git
   cd BaseMM_PRISM
   ```

   If you already cloned without `--recursive`:
   `git submodule update --init --recursive`

2. Create a branch: `git checkout -b feature/my-feature`

3. Set up an environment. PRISM requires Python 3.10 or newer. The right path
   depends on your platform — see the
   [installation guide](./README.md#installation) for the full matrix.

   For a generic CUDA workstation or CPU-only machine:

   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements/base.txt
   pip install -e . --no-deps
   ```

   On Aurora, Polaris, or Perlmutter, **do not** use a plain `pip install` or
   `uv sync` — these resolve a fresh PyTorch wheel that shadows the
   system-provided, accelerator-aware build. Use the platform scripts in
   `tools/` instead, which install with `--no-deps`.

4. Make your changes and add tests.

5. Run the checks below.

6. Push and open a pull request.

## Development Checks

The quickest path is the helper script, which lints and type-checks only what
you changed:

```bash
bash tools/ci/local_quality.sh              # changed files only
bash tools/ci/local_quality.sh full         # full scope, as CI runs it
bash tools/ci/local_quality.sh --install    # also install requirements/ci.txt first
```

A `Makefile` wraps the same commands one at a time, for when you want just one
of them. `make help` lists every target; `make` on its own prints that help.

```bash
make lint          # ruff check src tests tools scripts examples
make type-check    # mypy, against your interpreter's version
make test          # the unit suite, with CI's marker exclusions
make multimodal    # the modality contracts, which `make test` excludes
make launcher      # the launcher contracts, which `make test` excludes
make test-cov      # the coverage gate, at CI's threshold
make docs          # tools/ci/check_doc_links.py
make format        # ruff format, scoped to the .py files you changed
```

Both paths are conveniences: every target is a thin alias for a command spelled
out below, so `make` never becomes the only way to reproduce a CI result. The
marker strings, path scopes, and coverage floor in the `Makefile` are copied
verbatim from `.github/workflows/pr_quality.yml`, and
`tests/test_makefile_ci_parity.py` compares each one against the specific job
it mirrors, so the two cannot drift apart.

`make install` / `make install-dev` refuse to run on a recognised HPC login node,
because they resolve `torch` from PyPI. Use the `tools/` platform scripts there.

`full` mode is exactly what CI runs:

```bash
ruff check src tests tools scripts examples
mypy src
```

CI runs the type check and the unit suite on **both Python 3.10 and 3.12** —
3.10 is the floor in `requires-python`, 3.12 is what Aurora's current frameworks
module ships. Its mypy invocation therefore passes the leg's own version,
`mypy --python-version 3.12 src`, because `python_version` is pinned to 3.10 in
`pyproject.toml` and otherwise governs how mypy parses installed third-party
stubs. Locally, `mypy src` against whichever interpreter you have is fine; the
matrix is there so the deployment interpreter is not the untested one.

For tests, the CI unit job runs:

```bash
pytest -q \
  -m "not multimodal and not launcher and not integration and not network and not slow and not gpu and not aurora and not perlmutter" \
  tests
```

The marker exclusions skip tests needing hardware, network, or a larger setup.
The available markers are `unit`, `multimodal`, `launcher`, `integration`,
`network`, `slow`, `gpu`, `aurora`, `perlmutter`, and `timeseries` (defined in
`pyproject.toml`). If you have the relevant platform, run the full suite.

CI additionally runs the multimodal contract and launcher contract suites:

```bash
pytest -q tests/multimodal/test_modality_contracts.py
pytest -q tests/test_launch_aurora_web_vla.py \
          tests/platform/test_perlmutter_launch_contract.py \
          tests/platform/test_hpc_bridge_contract.py \
          tests/platform/test_launcher_secrets_policy.py
```

Changes to launchers must also pass the dry-run and secret-policy contracts.
Changes to a modality must include tensor-shape, collation, forward, and
checkpoint coverage. Hardware-specific results may be reported separately in
the PR when the required platform is unavailable to you.

## Style Guidelines

### Python

- [PEP 8](https://www.python.org/dev/peps/pep-0008/), enforced with
  [Ruff](https://docs.astral.sh/ruff/): `ruff check src tests tools scripts examples`
- Type hints where they clarify intent; `mypy src` must pass
- Match the conventions of the surrounding code

### Docstrings

Document public functions, classes, and modules. Include parameter
descriptions and return types where they are not obvious from the signature.

### Commit Messages

- Present tense, imperative mood ("Add feature", not "Added feature")
- First line 72 characters or fewer
- Reference issues and PRs after the first line

### Documentation

- Markdown, kept current with the code
- No spaces in filenames — they break tooling and URLs
- Include runnable examples where they help
- Documentation lives under `docs/`, grouped by topic. Put a new page in the
  folder that matches its subject and add it to
  [`docs/index.md`](./docs/index.md).

  | Folder | Holds |
  |---|---|
  | `platforms/` | Per-machine environment build and job submission |
  | `training/` | Entry points, data pipeline, CLI, DeepSpeed |
  | `modalities/` | Encoders, projectors, output decoders |
  | `models/` | Backbone integrations |
  | `evaluation/` | Universal evaluator, vLLM inference |
  | `applications/` | Scientific applications (BioReason, KEGG, materials) |
  | `api/` | API reference for the public `src.*` surface |
  | `development/` | CI and contributor tooling |
  | `results/` | Dated experiment outcomes — a record, not a usage guide |
  | `reports/` | Dated pilot and implementation reports |
  | `skills/` | Task-oriented playbooks, one folder per skill |
  | `data/` | Dataset provenance and delivery records |
  | `assets/` | Figures and frozen run-provenance snapshots |

  `reports/`, `data/` and `assets/` are frozen material: write a new page
  there rather than editing an old one, and the hardcoded-path gate skips
  them, because in a provenance record the path *is* the content.
  `results/` is **not** skipped — its pages are read as reusable guidance, so
  a command in one must take a `${PRISM_*}` site variable or a
  `<placeholder>` like any other page (see `tools/ci/check_site_paths.py`).
- Relative links between Markdown files are checked in CI by
  `tools/ci/check_doc_links.py`; run it locally before pushing a docs change.

## Guidelines for AI/LLM-Assisted Contributions

- **Remain accountable for all your outputs and decisions.** Individuals remain
  fully responsible and accountable for the accuracy, quality, appropriateness,
  and consequences of their work. Use of AI does not transfer this
  responsibility to the AI model, agent, or other tool.
- **Understand your work.** Regardless of how code or a PR was produced, this
  project requires that authors demonstrate a thorough understanding of any
  proposed changes. You must review such code line-by-line; it is your
  responsibility to ensure that it is correct and that it does not breach
  copyright. Always critically engage with AI outputs — do not trust them
  implicitly. AI-assisted code, analysis, and artifacts must be tested and
  validated at a level appropriate to their impact. Authors are responsible for
  ensuring that generated code is correct, secure, maintainable,
  non-obfuscated, appropriately scoped, documented, and reproducible where
  relevant.
- **Disclose AI-generated or AI-assisted work.** If AI/LLM tools were primarily
  used to generate code or artifacts, indicate this clearly in the PR.
- **Use of AI to review PRs.** All PRs must be reviewed by a human reviewer. An
  LLM review may be used in addition to a human reviewer, since it can help
  spot issues a human may have missed, but it must not be the sole reviewer.
  The human reviewer is fully accountable for the review feedback. This is a
  project norm rather than a mechanical gate today: `main` runs with
  `required_approving_review_count: 0` while the release gates land, and the
  rule becomes an enforced branch-protection setting at the public release.
- **Proprietary or personal information.** For this project, proprietary or
  personal information must never be sent to code generators or AI tools. This
  includes unpublished data, credentials, and internal hostnames or paths.
- **Be transparent, assume goodwill, and share what you learn.** Be open about
  relevant AI use, engage constructively with colleagues, and share experiences
  and lessons learned with the project.

## Questions?

Open an issue with the `question` label.

Thank you for contributing!
