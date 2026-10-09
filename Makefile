# Developer convenience wrappers around the commands CI already runs.
# Nothing here is required to build or test PRISM -- every target is a thin
# alias, so `make` never becomes the only way to reproduce a CI result.
#
# NOT for Aurora, Polaris, or Perlmutter login nodes: `install` / `install-dev`
# resolve a PyPI torch wheel that shadows the platform's accelerator-aware
# build (undefined symbol: __kmpc_fork_call). Use the platform scripts in
# tools/ there, as requirements/base.txt and CONTRIBUTING.md describe.

.DEFAULT_GOAL := help

# Same override knob as tools/ci/local_quality.sh, so one env var steers both.
PYTHON ?= $(or $(PRISM_CI_PYTHON),python3)

# `=`, not `:=`: only the type-check recipe needs this, so no other target pays
# for the subprocess. It mirrors CI's `mypy --python-version <matrix leg> src`.
# pyproject.toml pins mypy's python_version to 3.10; left at that, mypy parses
# the installed stubs under 3.10 rules and dies on numpy's PEP 695 `type`
# statements before reaching src, on any interpreter newer than the floor.
PY_VERSION = $(shell $(PYTHON) -c 'import sys; print("%d.%d" % sys.version_info[:2])')

# Anchor every path to this file's directory so `make -C`, a symlinked checkout,
# or a recursive invocation cannot change what `clean` deletes.
ROOT := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))

# CI exports PYTHONPATH=<workspace> for every job (pr_quality.yml `env:`); that
# is what makes `import src` work without an editable install, since the only
# conftest.py lives in tests/. Prepend rather than clobber a caller's value.
export PYTHONPATH := $(ROOT)$(if $(PYTHONPATH),:$(PYTHONPATH))

# Marker selectors copied verbatim from the `unit` and `coverage_gate` jobs in
# .github/workflows/pr_quality.yml. pytest does not error on an unknown marker
# inside a `not` clause -- it deselects nothing and exits 0 -- so a typo here
# would leave the target green while running the wrong set.
# tests/test_makefile_ci_parity.py binds each variable to its own job, so the
# two sides cannot drift apart unnoticed; edit both together.
UNIT_MARKERS := not multimodal and not launcher and not integration and not network and not slow and not gpu and not aurora and not perlmutter
COV_MARKERS := not integration and not network and not slow and not gpu and not aurora and not perlmutter

.PHONY: help install install-dev test multimodal launcher test-cov lint format type-check clean docs site-paths

help:  ## Show this help
	@echo 'PRISM make targets (PYTHON=$(PYTHON)):'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-12s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# requirements/base.txt and ci.txt both list bare `torch`, so these two targets
# RESOLVE torch from PyPI. On an HPC login node that wheel shadows the
# platform's accelerator-aware build and every PyG C++ extension then dies at
# import with `undefined symbol: __kmpc_fork_call`. A comment is not a guard, so
# refuse outright where a platform module is loaded. ALLOW_HPC_PIP=1 overrides.
# Detect by hostname (the idiom tools/build_aurora_env.sh:103 already uses) and
# by marker directories, because the env vars are absent in a plain login shell.
HPC_HOST := $(shell hostname 2>/dev/null | grep -qiE 'aurora-uan|polaris|^x[0-9]|login[0-9]*\.(alcf|nersc)|perlmutter|nid[0-9]' && echo yes)
HPC_MARKER := $(shell { [ -d /opt/aurora ] || [ -d /opt/cray ] || [ -n "$$NERSC_HOST" ] || [ -n "$$LMOD_SYSTEM_NAME" ]; } && echo yes)

define hpc_guard
	@if [ -z "$(ALLOW_HPC_PIP)" ] && { [ -n "$(HPC_HOST)" ] || [ -n "$(HPC_MARKER)" ]; }; then \
	  echo "REFUSING: this looks like an HPC system (host $$(hostname))."; \
	  echo "  \`make $(1)\` resolves torch from PyPI, which shadows the platform build"; \
	  echo "  and breaks the PyG extensions (undefined symbol: __kmpc_fork_call)."; \
	  echo "  Use the platform scripts instead: tools/build_aurora_env.sh,"; \
	  echo "  tools/setup_polaris_env.sh -- they install with --no-deps throughout."; \
	  echo "  To override anyway: make $(1) ALLOW_HPC_PIP=1"; \
	  exit 1; \
	fi
endef

install:  ## Install runtime deps + PRISM (editable). NOT for HPC login nodes
	$(call hpc_guard,install)
	$(PYTHON) -m pip install -r requirements/base.txt
# --no-deps: the line above already fixed the dependency set. Letting pip
# resolve for the editable install re-fetches torch over a local build.
	$(PYTHON) -m pip install -e . --no-deps

install-dev:  ## Install the CI toolchain (ruff/mypy/pytest) + PRISM editable
	$(call hpc_guard,install-dev)
	$(PYTHON) -m pip install -r requirements/ci.txt
	$(PYTHON) -m pip install -e . --no-deps

test:  ## Run the unit suite as CI's `unit` job does
	$(PYTHON) -m pytest -q -m "$(UNIT_MARKERS)" tests

multimodal:  ## Run the contract suite as CI's `multimodal_contract` job does
	$(PYTHON) -m pytest -q -m "not network" tests/multimodal/test_modality_contracts.py

# No `-m` selector: CI's launcher_contract job names these four files on the
# command line, and naming a file does not apply a marker filter. They carry
# the `launcher` marker, which UNIT_MARKERS excludes, so `make test` runs
# none of them -- this target is the only make path that does.
launcher:  ## Run the launcher contracts as CI's `launcher_contract` job does
	$(PYTHON) -m pytest -q \
	  tests/test_launch_aurora_web_vla.py \
	  tests/platform/test_perlmutter_launch_contract.py \
	  tests/platform/test_hpc_bridge_contract.py \
	  tests/platform/test_launcher_secrets_policy.py

test-cov:  ## Run the coverage gate as CI's `coverage_gate` job does
	$(PYTHON) -m pytest -m "$(COV_MARKERS)" --cov=src --cov-report=xml --cov-fail-under=34 tests

lint:  ## Ruff check, same scope as CI
	$(PYTHON) -m ruff check src tests tools scripts applications examples

# Scoped to files you actually changed, NOT the whole tree. The ruff-format
# pre-commit hook only ever sees staged files, so the tree has drifted from
# what a full `ruff format` would produce (CI runs `ruff check`, never
# `ruff format --check`). Formatting everything would bury a one-line change
# in a several-hundred-file diff. Override to widen:
#     make format FORMAT_PATHS="src tests tools"
FORMAT_PATHS ?= $(shell cd $(ROOT) && git diff --name-only --diff-filter=ACMR HEAD -- '*.py' 2>/dev/null)
format:  ## Ruff format your changed .py files (FORMAT_PATHS=... to override)
	@if [ -z "$(strip $(FORMAT_PATHS))" ]; then \
	  echo "No changed .py files vs HEAD. Pass FORMAT_PATHS=... to format explicitly."; \
	else \
	  echo "Formatting: $(FORMAT_PATHS)"; \
	  $(PYTHON) -m ruff format $(FORMAT_PATHS); \
	fi

type-check:  ## Mypy over src, same scope as CI
	$(PYTHON) -m mypy --python-version $(PY_VERSION) src

# Enforced in CI by tests/test_site_path_gate.py, which the required `unit`
# job runs -- this target is the same check, spelled for a human, plus the
# --detail listing the test does not print.
site-paths:  ## List hardcoded site paths and check the ratchet
	$(PYTHON) tools/ci/check_site_paths.py --detail

# There is no docs build system here -- docs/ is plain Markdown read on GitHub,
# and nothing in the repo depends on sphinx or mkdocs. Link resolution is the
# only thing there is to build-check, and it is what CI's doc_links job runs.
docs:  ## Check relative Markdown links, as CI's `doc_links` job does
	$(PYTHON) tools/ci/check_doc_links.py

# Scoped deliberately: a repo-wide `find` would reach into a local .venv and
# delete the *.egg-info directories that make its installed packages importable.
clean:  ## Remove build, test, and bytecode artifacts
	@test -f '$(ROOT)/pyproject.toml' || \
		{ echo 'clean: $(ROOT) is not a PRISM checkout; refusing to delete' >&2; exit 1; }
	rm -rf '$(ROOT)/build' '$(ROOT)/dist' '$(ROOT)/coverage.xml' '$(ROOT)/.pytest_cache'
	rm -rf '$(ROOT)'/*.egg-info
	find '$(ROOT)/src' '$(ROOT)/tests' '$(ROOT)/tools' \
		-name '__pycache__' -type d -prune -exec rm -rf {} +
