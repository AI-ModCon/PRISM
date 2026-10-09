"""Resolve site-specific filesystem roots from the environment.

Tracked configuration must not name a real user's directories: the paths that
worked on one Aurora allocation are meaningless to anyone else, and committing
them leaks project and account names into a public repository. Configs instead
reference the variables below, and each site supplies the values.

Precedence, highest first:

1. the process environment;
2. ``PRISM_SITE_ENV``, if it points at a readable ``KEY=value`` file;
3. the repository's ``.env`` (already the convention for ``HF_HOME`` and
   friends -- see ``.env.template``);
4. the documented placeholder, which is deliberately not a usable path.

The placeholders keep every config loadable -- tests, ``--help`` output, and
schema checks parse a config without touching a filesystem -- while failing
loudly if a real run is attempted against an unconfigured site.

Teams that share an allocation can keep one file of real values outside the
repository and point ``PRISM_SITE_ENV`` at it; see
``docs/platforms/site_paths.md``.

Stdlib-only, importing nothing from ``src``, so configuration loaders and the
light CLI modules can use it without pulling in the model stack.
"""

import os
from pathlib import Path

#: Marker for a value no site has configured. Substituted into paths so a
#: misconfiguration surfaces as a path naming the variable that is missing,
#: rather than as a confusing empty string or a silent fallback to ".".
UNSET = "<unset:{name}>"

#: Site variables, mapped to the placeholder used when nothing supplies them.
#: Every entry is a directory root; configs join their own suffixes onto it.
SITE_VARIABLES = {
    # Hugging Face hub cache holding `models--<org>--<name>/snapshots/<sha>`.
    "PRISM_HF_HUB": UNSET.format(name="PRISM_HF_HUB"),
    # Root under which prepared WebDataset/Arrow shards live.
    "PRISM_DATA_ROOT": UNSET.format(name="PRISM_DATA_ROOT"),
    # Directory holding the interleaved tokenizers (see tokenizers/ in the repo).
    "PRISM_TOKENIZERS": UNSET.format(name="PRISM_TOKENIZERS"),
    # Staged third-party generator assets (e.g. the pinned OmniGen2 snapshot).
    "PRISM_ASSETS": UNSET.format(name="PRISM_ASSETS"),
    # Training outputs: checkpoints, resume sources, run directories.
    "PRISM_OUTPUT_ROOT": UNSET.format(name="PRISM_OUTPUT_ROOT"),
}

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _read_env_file(path):
    """Parse ``KEY=value`` lines, ignoring comments, blanks, and quotes.

    ``$VAR``/``${VAR}`` in a value is expanded against the process
    environment, so a shared team file can write ``$USER`` once instead of
    one line per account. An undefined name is left as written rather than
    silently collapsing to an empty path segment.
    """
    values: dict[str, str] = {}
    try:
        text = Path(path).read_text()
    except OSError:
        return values
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        # Trailing inline comments are not stripped: a path may legitimately
        # contain "#", and every consumer here writes one value per line.
        value = os.path.expandvars(value.strip().strip('"').strip("'"))
        if key:
            values[key] = value
    return values


def site_values():
    """Return every site variable, resolved by the documented precedence.

    Returns:
        A ``dict`` mapping each name in ``SITE_VARIABLES`` to its resolved
        value. Names that no source supplies map to their ``<unset:NAME>``
        placeholder, so the result is always complete.
    """
    resolved = dict(SITE_VARIABLES)
    for source in (_REPO_ROOT / ".env", os.environ.get("PRISM_SITE_ENV")):
        if not source:
            continue
        for key, value in _read_env_file(source).items():
            if key in resolved and value:
                resolved[key] = value
    for key in resolved:
        value = os.environ.get(key)
        if value:
            resolved[key] = value
    return resolved


def expand(value):
    """Substitute ``${PRISM_*}`` site variables into one config value.

    Only the names in ``SITE_VARIABLES`` are substituted. Any other
    ``${...}`` is left untouched, so Hydra's own interpolation
    (``${oc.env:...}``, ``${hydra:run.dir}``) passes through unharmed.

    Args:
        value: The string to expand. Non-strings are returned unchanged, so
            this is safe to map over arbitrary decoded JSON.

    Returns:
        ``value`` with every known ``${NAME}`` replaced by its resolved site
        value, or by ``<unset:NAME>`` when the site has not configured it.
    """
    if not isinstance(value, str) or "${" not in value:
        return value
    for key, resolved in site_values().items():
        value = value.replace("${" + key + "}", resolved)
    return value


def expand_tree(node):
    """Apply :func:`expand` to every string in a nested config structure.

    Args:
        node: Decoded JSON -- any nesting of dicts, lists, and scalars.

    Returns:
        A structure of the same shape with site variables substituted.
        Dict keys are left alone; only values are expanded.
    """
    if isinstance(node, dict):
        return {key: expand_tree(value) for key, value in node.items()}
    if isinstance(node, list):
        return [expand_tree(item) for item in node]
    return expand(node)


def unresolved(value):
    """Return the site variables that ``value`` still needs, if any.

    Args:
        value: A string that has already been through :func:`expand`.

    Returns:
        A sorted list of variable names left unset, empty when the value is
        fully resolved. Callers about to touch the filesystem use this to
        report the variable to set instead of an opaque ``FileNotFoundError``.
    """
    if not isinstance(value, str):
        return []
    return sorted(name for name in SITE_VARIABLES if UNSET.format(name=name) in value)


def require_resolved(value, context):
    """Raise unless ``value`` has every site variable resolved.

    Args:
        value: The expanded path to check.
        context: What is being resolved, named in the error message.

    Raises:
        ValueError: If any site variable is still unset, naming each one and
            where to set it.
    """
    missing = unresolved(value)
    if not missing:
        return
    names = ", ".join(missing)
    raise ValueError(
        f"{context} needs site path(s) not configured here: {names}.\n"
        f"Set them in the repository .env, in the file named by PRISM_SITE_ENV, "
        f"or in the environment. See docs/platforms/site_paths.md."
    )
