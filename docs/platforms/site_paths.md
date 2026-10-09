# Site paths

PRISM's tracked configuration names filesystem roots as variables rather than
as one contributor's directories. A path like
`/lus/flare/projects/<project>/<user>/data/zone_a` is correct for exactly one
account on one allocation; for everyone else it is a `FileNotFoundError`, and
in a public repository it also publishes the project and account names of an
internal system.

Five variables cover every root the shipped configs need:

| Variable | What it points at | Who reads it |
|---|---|---|
| `PRISM_HF_HUB` | A Hugging Face **hub** directory, containing `models--<org>--<name>/snapshots/<sha>` | `src/conf/image_generation/*.json` |
| `PRISM_DATA_ROOT` | Root for prepared WebDataset/Arrow shards | `src/data/datasets_config.json`, `src/conf/data/*.yaml`, the SciTS evaluator |
| `PRISM_TOKENIZERS` | Interleaved tokenizers; the repo ships two under `tokenizers/` | four of the seven `src/conf/model/prism_olmo*_ts.yaml` configs |
| `PRISM_ASSETS` | Staged third-party generator assets, e.g. the pinned OmniGen2 snapshot | `src/conf/image_generation/*.json` |
| `PRISM_OUTPUT_ROOT` | Training outputs: run directories, checkpoints, resume sources | `src/conf/training/bioreason_sft.yaml` |

They are resolved by [`src/site_paths.py`](../../src/site_paths.py).

Two caveats on that table. The three `prism_olmo*_linear_interleaved_ts.yaml`
configs still carry a sanitized literal tokenizer path rather than
`${PRISM_TOKENIZERS}`; they are on the hygiene gate's allowlist and convert as
they are next touched. And four further site-ish variables are read outside
this set — `PRISM_CALVIN_ROOT`, `PRISM_AURORAGPT_2B_CHECKPOINT`,
`PRISM_OLMO1B_INTERLEAVED_TOKENIZER`, `PRISM_WALRUS_WEIGHTS_PATH`. They are
documented in [`.env.template`](../../.env.template) but are **not** part of
`SITE_VARIABLES`, so they get no `<unset:NAME>` placeholder and no
`require_resolved()` error — an unset one fails wherever it is used.

## Where the values come from

Highest precedence first:

1. **The process environment.** `PRISM_DATA_ROOT=/tmp/$USER/shards prism train …`
   overrides one root for one run without editing anything.
2. **The file named by `PRISM_SITE_ENV`**, if it is readable. This is how a team
   shares one set of real values without committing them — see below.
3. **The repository's `.env`**, which is gitignored and already the convention
   for `HF_HOME` and `PRISM_DIR` (see [`.env.template`](../../.env.template)).
4. **A placeholder**, `<unset:NAME>`, when nothing supplies a value.

The placeholder is deliberate. Expanding an unset variable to `""` would turn
`${PRISM_DATA_ROOT}/shards` into `/shards` — a real, wrong path — and expanding
it to `.` would silently glob the working directory. With the marker, a config
still *parses* (so `--help`, schema checks, and tests need no filesystem), and
any code about to touch the disk calls `require_resolved()` and raises naming
the variable you have to set.

> [!NOTE]
> That guarantee covers the **JSON** configs, which are expanded by
> `site_paths.expand_tree()` — `src/data/datasets_config.json` and
> `src/conf/image_generation/*.json`. The **Hydra YAML** configs resolve
> `${oc.env:PRISM_*}` through OmegaConf instead, and most carry no default, so
> an unset variable raises an OmegaConf resolution error rather than producing
> a placeholder. `src/conf/training/bioreason_sft.yaml` is the exception: it
> supplies its own `<set-PRISM_OUTPUT_ROOT>` default.

## Setting up: one user

Copy `.env.template` to `.env` and fill in the five values, or copy the
assignments out of [`.env.aurora.example`](../../.env.aurora.example). `.env`
is gitignored, as is every `.env.*` variant except the committed `*.template`
and `*.example` files.

```bash
cp .env.template .env
$EDITOR .env      # set the PRISM_* roots for your account
```

## Setting up: a shared allocation

Keep **one** file of real values outside every git checkout, in a directory the
allocation's group can read, and point each member at it:

```bash
# once, by whoever maintains the allocation
cat > /lus/flare/projects/<project>/prism-site.env <<'EOF'
PRISM_HF_HUB=/lus/flare/projects/<project>/shared/huggingface/hub
PRISM_DATA_ROOT=/lus/flare/projects/<project>/shared/data
PRISM_TOKENIZERS=/lus/flare/projects/<project>/shared/tokenizers
PRISM_ASSETS=/lus/flare/projects/<project>/shared/assets
PRISM_OUTPUT_ROOT=/lus/flare/projects/<project>/$USER/outputs
EOF
chmod 640 /lus/flare/projects/<project>/prism-site.env

# in each member's shell profile or job script
export PRISM_SITE_ENV=/lus/flare/projects/<project>/prism-site.env
```

`$USER` and `${VAR}` inside the file are expanded against the process
environment when it is read, which is what lets a single shared file give every
member their own `PRISM_OUTPUT_ROOT` while sharing the read-only roots.

The file must live outside any checkout. Putting it inside the repository —
even gitignored — is how these paths ended up in the history in the first
place: one `git add -f`, one `.gitignore` edit, and the real values are
published permanently.

## Reading a value from code

```python
from src.site_paths import expand, expand_tree, require_resolved

path = expand("${PRISM_DATA_ROOT}/SciTS-processed/val_shards")
require_resolved(path, "SciTS validation shards")   # raises, naming the variable
```

`expand_tree()` is the same thing over decoded JSON, and is what
`DatasetManager` applies to `datasets_config.json` at load.

`expand()` substitutes **only** the five names above. Any other `${...}` is
left exactly as written, so Hydra's `${oc.env:VAR}` and `${hydra:run.dir}` pass
through untouched — in a YAML config, prefer Hydra's own `${oc.env:PRISM_*}`
form, which OmegaConf resolves.

## The hygiene gate

[`tools/ci/check_site_paths.py`](../../tools/ci/check_site_paths.py) scans every
tracked file under `src/`, `tools/`, `scripts/`, `tests/`, `examples/`, and
`docs/` — minus the four `docs/` exemptions listed below — for absolute paths
under an HPC filesystem root, skipping anything containing a `<placeholder>`,
a `$VAR`, or a `...` elision.

It is a **ratchet**, not a clean gate. `tools/ci/site_path_allowlist.json`
records the per-file count as of the day it landed; the check fails when a file
exceeds its recorded count or an unlisted file gains one. It *also* fails when a
count goes down without the allowlist being regenerated, so the numbers cannot
quietly become a permanent excuse.

```bash
make site-paths                                   # list every occurrence
python tools/ci/check_site_paths.py --write       # after removing some, re-record
```

`src/` has no allowlist entries and must keep none —
`tests/test_site_path_gate.py` asserts that directly. The remaining occurrences
are spread across four trees, and come out as each file is next touched:

| Tree | Occurrences | What they are |
|---|---|---|
| `scripts/` | 119 | Operational conversion and staging scripts |
| `tools/` | 69 | Launcher defaults |
| `docs/` | 55 | Worked examples in prose, plus observational paths in `results/` |
| `tests/` | 16 | Fixture paths |

Do not re-derive these counts by hand; `make site-paths` prints the current
breakdown, and the allowlist is the source of truth.

Prose is scanned, with three exemptions: `docs/assets/`, `docs/reports/`,
and `docs/data/public_image_sources/`. Those are frozen artifacts where the
path *is* the content — a provenance snapshot that must stay byte-identical,
or a date-named report whose value is the exact command someone ran on one
machine — so rewriting one destroys the record it exists to keep.

`docs/results/` was exempt alongside them and no longer is. Its pages are
undated and the index advertises them as reusable — "how to run it", "launch
commands" — so a reader copies out of them, which is the thing this gate
exists to stop. Every runnable command there now takes a site variable or a
`<placeholder>`; what remains on the ratchet is observational (the inputs one
sweep actually read, an error message quoted verbatim), and that stays.

Everything else under `docs/` is in scope, including this page. Living
documentation is where a "here is how our team sets this up" paragraph gets
written, and a real shared path pasted into a guide reaches a public
repository exactly as a config would — while being far likelier to be copied,
since a guide is read as instructions. `tests/test_site_path_gate.py` asserts
this page in particular names no real path, because it carries the
shared-allocation recipe.
