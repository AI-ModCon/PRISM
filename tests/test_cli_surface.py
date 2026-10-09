"""Drive the shipped `prism` CLI the way a newcomer does, and assert it exits 0.

Nothing exercised `src/cli/` before this file. That is how three defects reached
a release branch at once, each of them in the exact invocation the CLI's own
docstrings advertise:

* `prism analyze model --preset <a timeseries preset>` printed
  `config.max_instances_per_seq`, a field that exists nowhere in the repo, and
  died with a raw AttributeError traceback after half-writing its output. Four
  of the fourteen shipped presets were affected.
* `prism train model=...` and `prism launch ... -- model=...` forwarded Hydra
  overrides through `ctx.args`, which a Typer *group* never populates -- it tries
  to resolve the first override as a subcommand instead, so the documented
  invocation died with `No such command 'model=prism_olmo3_7b'`.
* Every example in those docstrings named `model=prism-olmo3-7b`, a Hydra config
  group that does not exist; the real file is `src/conf/model/prism_olmo3_7b.yaml`.

The tests below are deliberately written to sweep rather than to spot-check.
Asserting on a fixed list of expected field names would reproduce the original
bug class: the list drifts out of step with `ModelConfig`, and a test that names
`ts_projector` cannot notice the *next* nonexistent attribute. So the preset test
enumerates `PRISM_CONFIGS` at runtime and only asserts "no preset crashes", which
is the property that was actually violated.

Kept import-light on purpose: `src/cli/__init__.py` imports only `typer` at
module scope and every subcommand defers `from src.config import ...` into its
function body, so building the command tree never pulls in torch. `prism train`
is only ever exercised through a stubbed `subprocess.run`. No test here builds a
model, touches the network, or spawns a real training process.
"""

import shlex

import pytest

typer = pytest.importorskip("typer", reason="typer is the CLI runtime (pyproject.toml:11)")
from typer.testing import CliRunner  # noqa: E402

runner = CliRunner()


@pytest.fixture
def cli():
    """The real assembled app, with every subcommand registered."""
    from src.cli import app

    return app


def _run(app, argv):
    return runner.invoke(app, argv)


# --------------------------------------------------------------------------
# `prism analyze model` -- every shipped preset must render
# --------------------------------------------------------------------------


def _preset_names():
    from src.config import PRISM_CONFIGS

    return list(PRISM_CONFIGS)


def test_there_are_presets_to_sweep():
    # Guards the sweep below against silently becoming a no-op: a
    # parametrization over an empty list passes without running anything.
    assert len(_preset_names()) >= 10


@pytest.mark.parametrize("preset", _preset_names())
def test_analyze_model_renders_every_shipped_preset(cli, preset):
    """`prism analyze model --preset X` must exit 0 for every X we ship.

    This is the assertion that would have caught `max_instances_per_seq`: the
    four timeseries presets exited 1 with an AttributeError, and no test noticed.
    """
    result = _run(cli, ["analyze", "model", "--preset", preset])
    assert result.exit_code == 0, (
        f"`prism analyze model --preset {preset}` exited {result.exit_code}.\n"
        f"output:\n{result.output}\n"
        f"exception: {result.exception!r}"
    )
    # A crash after partial output still writes the header, so "exit 0" alone is
    # weak. Require the section the renderer ends with.
    assert preset in result.output


def test_analyze_model_rejects_an_unknown_preset(cli):
    """The negative control: if this ever passes, the sweep above proves nothing."""
    result = _run(cli, ["analyze", "model", "--preset", "definitely-not-a-preset"])
    assert result.exit_code != 0


# --------------------------------------------------------------------------
# Hydra override passthrough
# --------------------------------------------------------------------------


@pytest.fixture
def captured_train(monkeypatch):
    """Run `prism train` without launching training; capture the argv it builds."""
    import src.cli.train as train_mod

    seen = {}

    class _Completed:
        returncode = 0

    def _fake_run(cmd, **kwargs):
        seen["cmd"] = list(cmd)
        seen["kwargs"] = kwargs
        return _Completed()

    monkeypatch.setattr(train_mod.subprocess, "run", _fake_run)
    return seen


def _is_title_line(args: str) -> bool:
    """Is this the module docstring's own first line, not an example?

    Every CLI module opens with `prism <cmd> -- <summary>`, which starts with the
    same prefix the scrapers match on. Left in, it becomes a parametrized case
    that invokes the CLI with the summary text as arguments. For `prism train`
    that happens to parse as a Hydra override and exits 0, so the train scraper
    was passing by accident; for `prism launch` it exits 2 on a missing required
    option. Recognised by the em dash, which is what separates the command from
    its summary and never appears in an invocation.
    """
    return args.startswith("\u2014")


def _documented_train_invocations():
    """Scrape `prism train ...` examples out of the CLI's own docstrings.

    Deliberately *not* a hardcoded list. A literal list drifts the moment someone
    edits a docstring, and the test then certifies an invocation nobody ships
    while the advertised one stays broken -- which is how `model=prism-olmo3-7b`
    survived from the CLI's first commit to a release branch. Reading the
    docstrings means editing them is what updates the test.
    """
    import src.cli
    import src.cli.train

    found = []
    for mod in (src.cli, src.cli.train):
        for raw in (mod.__doc__ or "").splitlines():
            line = raw.strip()
            if not line.startswith("prism train "):
                continue
            args = line[len("prism train ") :].strip()
            if _is_title_line(args):
                continue
            if args and args not in found:
                found.append(args)
    return found


DOCUMENTED_TRAIN_INVOCATIONS = _documented_train_invocations()


def test_docstrings_actually_advertise_train_invocations():
    """Guards the scraper: an empty parametrize list passes without testing anything."""
    assert len(DOCUMENTED_TRAIN_INVOCATIONS) >= 3, DOCUMENTED_TRAIN_INVOCATIONS


@pytest.mark.parametrize("invocation", DOCUMENTED_TRAIN_INVOCATIONS)
def test_documented_train_invocations_reach_the_trainer(cli, captured_train, invocation):
    argv = shlex.split(invocation)
    result = _run(cli, ["train", *argv])

    assert result.exit_code == 0, (
        f"`prism train {invocation}` exited {result.exit_code}.\n"
        f"output:\n{result.output}\nexception: {result.exception!r}"
    )
    built = captured_train.get("cmd")
    assert built is not None, "subprocess.run was never reached"
    assert built[1].endswith("src/train.py")

    # Every non-flag token in the documented invocation must survive into the
    # command actually executed. `--multirun` is consumed by Typer and re-emitted,
    # so check it separately.
    for token in argv:
        if token == "--multirun":
            assert "--multirun" in built
        else:
            assert token in built, f"override {token!r} was dropped; built={built}"


def test_train_forwards_hydra_append_and_delete_syntax(cli, captured_train):
    """Hydra's `+key=v` and `~key` forms start with characters an arg parser may eat."""
    result = _run(cli, ["train", "+extra.thing=1", "~unwanted"])
    assert result.exit_code == 0, result.output
    assert "+extra.thing=1" in captured_train["cmd"]
    assert "~unwanted" in captured_train["cmd"]


def test_train_without_overrides_still_runs(cli, captured_train):
    result = _run(cli, ["train"])
    assert result.exit_code == 0, result.output
    assert captured_train["cmd"][1].endswith("src/train.py")


def test_train_list_presets_does_not_launch_training(cli, captured_train):
    result = _run(cli, ["train", "--list-presets"])
    assert result.exit_code == 0, result.output
    assert "cmd" not in captured_train, "--list-presets must not start training"
    assert "Available model presets" in result.output


def test_train_rejects_an_unknown_flag(cli, captured_train):
    """Negative control for the passthrough.

    The override capture must not be so permissive that a typo'd flag is silently
    swallowed as a Hydra override. `--multirunn` should be an error, not an
    override named `--multirunn`.
    """
    result = _run(cli, ["train", "--multirunn"])
    assert result.exit_code != 0
    assert "cmd" not in captured_train


# --------------------------------------------------------------------------
# `prism launch` forwards its overrides too
#
# Kept as its own section rather than folded into the train tests: `launch` has
# the identical Typer-group defect and the identical fix, but it is a separate
# callback, and a revert of only its half leaves the train tests entirely green.
# --------------------------------------------------------------------------


@pytest.fixture
def captured_launch(monkeypatch, tmp_path):
    """Run `prism launch` without spawning a launcher; capture the argv it builds.

    Also points every entry of `_LAUNCHER_MAP` at a real file in `tmp_path`: the
    callback refuses with exit 1 if the launcher script is missing, which on a
    fresh clone would make this test pass for the wrong reason.
    """
    import src.cli.launch as launch_mod

    seen = {}

    class _Completed:
        returncode = 0

    def _fake_run(cmd, **kwargs):
        seen["cmd"] = list(cmd)
        seen["kwargs"] = kwargs
        return _Completed()

    stub = tmp_path / "launcher.py"
    stub.write_text("")
    monkeypatch.setattr(
        launch_mod, "_LAUNCHER_MAP", {k: str(stub) for k in launch_mod._LAUNCHER_MAP}
    )
    monkeypatch.setattr(launch_mod.subprocess, "run", _fake_run)
    return seen


def _documented_launch_invocations():
    """Scrape `prism launch ...` examples out of the CLI's own docstrings.

    Same contract as the train scraper: reading the docstring is what keeps the
    test in step with what is advertised.
    """
    import src.cli
    import src.cli.launch

    found = []
    for mod in (src.cli, src.cli.launch):
        for raw in (mod.__doc__ or "").splitlines():
            line = raw.strip()
            if not line.startswith("prism launch "):
                continue
            args = line[len("prism launch ") :].strip()
            if _is_title_line(args):
                continue
            if args and args not in found:
                found.append(args)
    return found


DOCUMENTED_LAUNCH_INVOCATIONS = _documented_launch_invocations()


def test_docstrings_actually_advertise_launch_invocations():
    """Guards the scraper: an empty parametrize list passes without testing anything."""
    assert len(DOCUMENTED_LAUNCH_INVOCATIONS) >= 3, DOCUMENTED_LAUNCH_INVOCATIONS


@pytest.mark.parametrize("invocation", DOCUMENTED_LAUNCH_INVOCATIONS)
def test_documented_launch_invocations_reach_the_launcher(cli, captured_launch, invocation):
    """The `-- model=...` form in the docstring is the one that used to exit 2."""
    argv = shlex.split(invocation)
    result = _run(cli, ["launch", *argv])

    assert result.exit_code == 0, (
        f"`prism launch {invocation}` exited {result.exit_code}.\n"
        f"output:\n{result.output}\nexception: {result.exception!r}"
    )
    built = captured_launch.get("cmd")
    assert built is not None, "subprocess.run was never reached"

    # `--` is consumed by the parser; every other non-flag token after it is a
    # Hydra override and must survive into the command actually executed.
    if "--" in argv:
        for token in argv[argv.index("--") + 1 :]:
            assert token in built, f"override {token!r} was dropped; built={built}"


def test_launch_forwards_hydra_append_and_delete_syntax(cli, captured_launch):
    """`+key=v` and `~key` start with characters an arg parser may eat."""
    result = _run(
        cli,
        ["launch", "--platform", "baremetal", "--id", "x", "--", "+extra.thing=1", "~unwanted"],
    )
    assert result.exit_code == 0, result.output
    assert "+extra.thing=1" in captured_launch["cmd"]
    assert "~unwanted" in captured_launch["cmd"]


def test_launch_without_overrides_still_runs(cli, captured_launch):
    result = _run(cli, ["launch", "--platform", "baremetal", "--id", "x"])
    assert result.exit_code == 0, result.output
    assert captured_launch.get("cmd") is not None


def test_launch_rejects_an_unknown_flag(cli, captured_launch):
    """Negative control, same as train's: a typo'd flag must not become an override."""
    result = _run(cli, ["launch", "--platform", "baremetal", "--id", "x", "--nodez", "4"])
    assert result.exit_code != 0
    assert "cmd" not in captured_launch


# --------------------------------------------------------------------------
# The advertised Hydra config groups must exist on disk
# --------------------------------------------------------------------------


def _advertised_group_overrides():
    """Every `<group>=<value>` token PRISM advertises, with where it was advertised.

    Covers the CLI docstrings *and* `docs/training/cli.md`, because the broken
    `model=prism-olmo3-7b` name appeared in both and fixing only one leaves the
    other lying to users.
    """
    from pathlib import Path

    import src
    import src.cli
    import src.cli.launch
    import src.cli.train

    sources = {
        "src/cli/__init__.py": src.cli.__doc__ or "",
        "src/cli/train.py": src.cli.train.__doc__ or "",
        "src/cli/launch.py": src.cli.launch.__doc__ or "",
    }
    doc = Path(src.__file__).resolve().parents[1] / "docs" / "training" / "cli.md"
    if doc.is_file():
        sources[str(doc.name)] = doc.read_text()

    out = []
    for origin, text in sources.items():
        for raw in text.splitlines():
            if "prism train" not in raw and "prism launch" not in raw:
                continue
            # Markdown wraps commands in backticks and table pipes; strip both so
            # a doc line yields the same tokens a shell would see.
            cleaned = raw.strip().strip("|").replace("`", " ").rstrip("\\")
            try:
                tokens = shlex.split(cleaned)
            except ValueError:
                continue
            for token in tokens:
                if token.startswith("-") or "=" not in token or "." in token.split("=")[0]:
                    continue
                out.append((origin, token))
    return out


def test_documented_config_groups_exist():
    """`model=prism-olmo3-7b` was advertised for as long as the CLI has existed and
    never resolved: Hydra group names come from filenames, which use underscores.

    Reads both the advertised tokens and the available configs off disk, so a
    renamed config or an edited docstring reddens this rather than silently
    re-breaking the documented invocation.
    """
    from pathlib import Path

    import src

    conf = Path(src.__file__).resolve().parent / "conf"
    advertised = _advertised_group_overrides()
    assert advertised, "scraper found no `group=value` overrides to check"

    checked = 0
    for origin, token in advertised:
        key, _, value = token.partition("=")
        group = conf / key
        if not group.is_dir():
            continue  # not a Hydra config group (e.g. an --option=value form)
        available = sorted(p.stem for p in group.glob("*.yaml"))
        checked += 1
        assert value in available, (
            f"{origin} advertises `{token}` but src/conf/{key}/ has no {value}.yaml.\n"
            f"available: {available}"
        )
    assert checked, "no advertised token resolved to a real src/conf group directory"


# --------------------------------------------------------------------------
# The command tree itself
# --------------------------------------------------------------------------


def test_help_works_for_every_registered_subcommand(cli):
    """`--help` must not crash anywhere in the tree.

    A module-scope import error in any `src/cli/*.py` shows up here, which is the
    cheapest possible guard on the console script the docs tell users to install.
    """
    top = _run(cli, ["--help"])
    assert top.exit_code == 0, top.output

    for name in ("train", "launch", "data", "eval", "analyze", "serve"):
        result = _run(cli, [name, "--help"])
        assert result.exit_code == 0, (
            f"`prism {name} --help` exited {result.exit_code}\n{result.output}"
        )


def test_cli_imports_without_torch(monkeypatch):
    """The command tree must build even where torch is missing.

    Not because any documented install leaves torch absent -- none currently
    does. README:34-35 runs `pip install -r requirements/base.txt` before
    `pip install -e . --no-deps`, and that requirements file pins torch
    (`requirements/base.txt:7`); docs/training/cli.md:9 is a plain
    `pip install -e .`. The case this pins is the one that is easy to reach by
    accident and impossible to see in review: a partial or torch-less
    environment where `prism --help` is the first thing anyone types.

    What makes that work is the deferred-import structure, not a guard inside
    `src/config.py`: on this branch `src/config.py:5` is a bare `import torch`,
    and the CLI survives only because nothing imports `src.config` until a
    command body runs.

    So this pins the property that actually holds today -- `prism --help` and
    every `<cmd> --help` work with no torch -- and deliberately stops there. The
    commands that do reach `src.config` (`prism train --list-presets`,
    `prism analyze model`) still raise ImportError without torch; #223 fixes that
    by wrapping the import, and this test is written not to contradict either
    state.
    """
    import builtins
    import importlib
    import sys

    real_import = builtins.__import__

    def _no_torch(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("torch is not installed (simulated)")
        return real_import(name, *args, **kwargs)

    for mod in [m for m in sys.modules if m == "src.cli" or m.startswith("src.cli.")]:
        monkeypatch.delitem(sys.modules, mod, raising=False)
    monkeypatch.delitem(sys.modules, "src.config", raising=False)
    monkeypatch.setattr(builtins, "__import__", _no_torch)

    cli = importlib.import_module("src.cli")
    result = runner.invoke(cli.app, ["--help"])
    assert result.exit_code == 0, result.output

    # Every subcommand's help too: `add_typer` only registers the module, so a
    # top-level `--help` that passes proves less than it looks. A module-scope
    # `from src.config import ...` added to any one of these would redden here.
    for name in ("train", "launch", "data", "eval", "analyze", "serve"):
        sub = runner.invoke(cli.app, [name, "--help"])
        assert sub.exit_code == 0, f"`prism {name} --help` needs torch\n{sub.output}"
