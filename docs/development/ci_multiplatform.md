# CI: Multiplatform Foundation

This repository uses a tiered CI strategy built for multimodal development and HPC portability.

## Required PR checks

These are the **unprefixed job names** from `pr_quality.yml`, the only workflow
that runs on pull requests. GitHub only prefixes a status-check context with
`Workflow / ` for jobs in a reusable workflow invoked through `workflow_call`;
this one is not, so a prefixed context would never be satisfied and merges would
deadlock. `tools/ci/configure_branch_protection.py` holds the same list and
`tests/platform/test_branch_protection_contract.py` keeps the two in step.

`types` and `unit` run a Python matrix, and a matrixed job reports one
context per leg -- `types (3.10)`, `types (3.12)` -- and never its bare name.
Requiring the bare name of a matrixed job requires something that will never
report, which, with `enforce_admins` on, blocks every pull request with no way
out through the UI. Change the matrix and this list together.

- `lint`
- `doc_links`
- `types (3.10)`
- `types (3.12)`
- `unit (3.10)`
- `unit (3.12)`
- `multimodal_contract`
- `launcher_contract`
- `coverage_gate`

## Test marker taxonomy

- `unit`: fast, local-only checks
- `multimodal`: tensor shape and modality contracts
- `launcher`: launcher and CI-bridge contracts
- `integration`: broader integration tests
- `network`: tests that require external downloads/APIs
- `slow`: long-running tests
- `gpu`: tests that need GPU hardware
- `aurora`: Aurora/XPU-specific tests
- `perlmutter`: Perlmutter/CUDA-specific tests

## Local parity commands

```bash
# Changed-files mode (recommended for day-to-day development)
tools/ci/local_quality.sh changed --python /path/to/python

# Full mode (legacy backlog may fail until full cleanup)
tools/ci/local_quality.sh full --python /path/to/python

# Equivalent raw commands:
ruff check src tests tools scripts examples
mypy src
pytest -m "not integration and not network and not slow and not gpu and not aurora and not perlmutter" tests
pytest tests/multimodal/test_modality_contracts.py
pytest tests/test_launch_aurora_web_vla.py tests/platform/test_hpc_bridge_contract.py tests/platform/test_launcher_secrets_policy.py
```

## HPC scheduler bridge scripts

The scripts under `tools/ci/` submit and track real scheduler jobs:

- `dispatch_hpc_job.py` — `qsub` (PBS, Aurora) or `sbatch` (Slurm, Perlmutter)
- `poll_hpc_job.py`
- `fetch_hpc_artifacts.py`
- `validate_hpc_result.py`

They are run by hand from a machine that can reach the scheduler, and are
covered by `tests/platform/test_hpc_bridge_contract.py`.

No workflow invokes them. `pr_hpc_smoke_contract.yml` used to, on every pull
request, and was removed — see the note below before reinstating anything like
it.

### Why the HPC CI workflow was removed

It ran on every pull request but only ever in dry-run mode: real submissions
were gated on `push` to a branch that no longer exists, behind repository
variables that were never set. Across its lifetime it produced 380 deployments,
all of them to `hpc-dryrun`; the `aurora-ci` and `perlmutter-ci` environments
recorded zero. It was verifying that the dry-run path still parsed, which is
what the bridge contract tests already do, faster and without a runner.

Reinstating it needs three things fixed first, none of which were true:

- **Artifacts leak scheduler identifiers.** `dispatch_hpc_job.py` writes
  `scheduler_host`, `scheduler_user` and `account` into its dispatch JSON, and
  the workflow uploaded that directory with `actions/upload-artifact`. Artifacts
  are **not** secret-redacted, and in a public repository they are downloadable
  by anyone. Redact at the writer before any upload step exists again.
- **The environments had no protection rules.** `aurora-ci`, `perlmutter-ci` and
  `hpc-dryrun` each had zero required reviewers and no branch restrictions, so
  the "opt-in real mode" gate was a workflow condition and nothing more.
- **A self-hosted ALCF runner in a public repository.** `aurora_contract_self_hosted`
  targeted a runner inside the ALCF network. GitHub advises against self-hosted
  runners on public repositories: a fork PR can run code on them.

## Branch protection automation

Use `tools/ci/configure_branch_protection.py` to configure required checks on `main`.

Dry-run example:

```bash
python tools/ci/configure_branch_protection.py --repo <OWNER/REPO> --branch main --dry-run
```

Apply example:

```bash
GITHUB_TOKEN=<ADMIN_TOKEN> \
python tools/ci/configure_branch_protection.py --repo <OWNER/REPO> --branch main
```

Run it locally with an admin token, as above. There is deliberately no
workflow wrapper: a dispatchable job that applies branch protection would
let anyone with write access reconfigure the protection on `main`.

### The preflight guard

Applying protection is refused unless every required context above has reported
a `success` conclusion on the head commit of the branch being protected. Two
failure modes wedge a protected branch, and `enforce_admins` leaves no way out
through the UI:

- A context nothing produces -- a disabled workflow, or a name that does not
  match the job -- never reports, so every pull request waits on it forever.
- A context that reports red. A job failing on the branch head will fail on pull
  request heads too, and a required context that fails is a merge that cannot
  happen.

`skipped`, `neutral`, and a run still in progress are all refused: none is
evidence the job ran and passed. When a name reports twice (a re-run, or two
apps publishing it), the unsuccessful outcome wins.

So the order is: enable the workflows, push to the branch, get a fully green
run, then protect. `--skip-preflight` overrides the guard when the branch is
known green by other means.
