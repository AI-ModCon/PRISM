# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

<!-- TODO(gate-4): before tagging v0.1.0, promote the relevant entries below
     into a dated release section. The pre-v0.1.0 history is large; this file
     starts recording changes from the open-source readiness work rather than
     reconstructing the full development history retroactively. -->

Nothing here has shipped in a tagged release yet; these are the changes that
will make up v0.1.0.

### Added

- Apache-2.0 `LICENSE` and `NOTICE` with third-party attributions (#186)
- Genesis Mission acknowledgment and a License section in the README (#186)
- `CONTRIBUTING.md`, including a stated policy on AI/LLM-assisted contributions
- `CODE_OF_CONDUCT.md` (Contributor Covenant 2.1)
- `SECURITY.md` with a private vulnerability reporting path and a
  supported-versions table
- This changelog
- GitHub issue templates (bug report, feature request) and a pull request
  template
- README: a portable quickstart that does not require an HPC allocation, and a
  per-modality support-status table
- `tools/ci/check_doc_links.py`, a link checker for relative Markdown links, run
  as a required `doc_links` CI job

### Changed

- Corrected the `src/libs/walrus` submodule URL to the upstream
  `PolymathicAI/walrus` organization (#186)
- `requirements/base.txt` now installs `h5py`, which `src/data/multimodal.py`
  imports at module level — without it `pytest` aborts during collection
- Documentation is now grouped by topic under `docs/` — `platforms/`,
  `training/`, `modalities/`, `models/`, `evaluation/`, `results/`,
  `applications/`, `architecture/`, and `development/` — rather than 33 flat
  files. `docs/index.md` is a categorised entry point covering every page, and
  `docs/getting-started.md` has been rewritten around the four real install
  paths. `docs/VLM Ablations.md` is now `docs/results/vlm_ablations.md`; the
  space in the old filename broke tooling and URLs.

### Removed

- `docs/plans/` (9 files) and `docs/historical/` (5 files) — dated design plans
  and bring-up debug journals whose work has all landed. Plans are tracking
  artifacts and belong in the issue tracker rather than the source tree.
  `git log` retains them. Module docstrings in `src/decoders/` and the decoder
  test suite that cited the output-decoder plan by section are now
  self-contained: the design rationale each one needed is stated inline.
- `changes.txt`, a 412-line free-form log superseded by this changelog
- `scaling-study/deck/` — a slide-deck builder whose three inputs are not
  tracked, so it cannot run, plus its 314 KB generated HTML output
- `scaling-study/legacy/investigation/` (38 files) — an Aurora HSDP scaling
  investigation that concluded 2026-06-21; the production launcher
  configuration it produced is unchanged and lives in `tools/`

<!-- TODO(gate-4): at the v0.1.0 tag, rename the section above to
     "## [0.1.0] - YYYY-MM-DD" and open a fresh [Unreleased] above it.
     Candidate release summary, to be confirmed: the initial public release of
     PRISM — a unified multimodal transformer for scientific data spanning
     text, images, time series, DNA, geometry, graphs, and tables, on a
     pretrained causal-LM backbone, with validated training paths on Aurora,
     Polaris, and Perlmutter. -->

[Unreleased]: https://github.com/AI-ModCon/BaseMM_PRISM/commits/main
