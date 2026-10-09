"""Tests for src/site_paths.py — the site-variable resolver.

Two properties matter beyond the happy path, and both have bitten before:

* An unset variable must not vanish. Expanding to "" would turn
  "${PRISM_DATA_ROOT}/shards" into "/shards", which is a real, wrong, and
  usually unreadable path; expanding to "." would silently glob the CWD. The
  "<unset:NAME>" marker fails loudly and names what to set.
* Hydra's own "${oc.env:...}" and "${hydra:run.dir}" interpolations live in the
  same YAML files. expand() must leave every "${...}" it does not own alone.
"""

from __future__ import annotations

import pytest
from src import site_paths

pytestmark = [pytest.mark.unit]


@pytest.fixture(autouse=True)
def isolated_site(monkeypatch, tmp_path):
    """Detach the resolver from the developer's own environment and .env.

    Every site variable is cleared, PRISM_SITE_ENV is unset, and the repo-root
    .env lookup is pointed at an empty tmp dir — otherwise these tests would
    pass or fail depending on whose machine ran them.
    """
    for name in site_paths.SITE_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("PRISM_SITE_ENV", raising=False)
    monkeypatch.setattr(site_paths, "_REPO_ROOT", tmp_path)
    return tmp_path


def test_unset_variable_expands_to_a_named_marker():
    assert site_paths.expand("${PRISM_DATA_ROOT}/shards") == "<unset:PRISM_DATA_ROOT>/shards"


def test_unresolved_reports_the_variable_to_set():
    expanded = site_paths.expand("${PRISM_DATA_ROOT}/x/${PRISM_HF_HUB}")
    assert site_paths.unresolved(expanded) == ["PRISM_DATA_ROOT", "PRISM_HF_HUB"]


def test_unresolved_is_empty_once_configured(monkeypatch):
    monkeypatch.setenv("PRISM_DATA_ROOT", "/somewhere/data")
    assert site_paths.unresolved(site_paths.expand("${PRISM_DATA_ROOT}/shards")) == []


def test_require_resolved_names_every_missing_variable():
    value = site_paths.expand("${PRISM_HF_HUB}/m/${PRISM_ASSETS}")
    with pytest.raises(ValueError) as excinfo:
        site_paths.require_resolved(value, "Test asset")
    message = str(excinfo.value)
    assert "Test asset" in message
    assert "PRISM_HF_HUB" in message
    assert "PRISM_ASSETS" in message
    assert "docs/platforms/site_paths.md" in message


def test_require_resolved_passes_a_fully_resolved_value(monkeypatch):
    monkeypatch.setenv("PRISM_HF_HUB", "/hub")
    site_paths.require_resolved(site_paths.expand("${PRISM_HF_HUB}/m"), "Test asset")


# --------------------------------------------------------------------------
# Precedence: process environment > $PRISM_SITE_ENV file > repo .env
# --------------------------------------------------------------------------


def test_repo_dotenv_supplies_a_value(isolated_site):
    (isolated_site / ".env").write_text('PRISM_DATA_ROOT="/from/dotenv"\n')
    assert site_paths.site_values()["PRISM_DATA_ROOT"] == "/from/dotenv"


def test_site_env_file_beats_repo_dotenv(isolated_site, tmp_path, monkeypatch):
    (isolated_site / ".env").write_text("PRISM_DATA_ROOT=/from/dotenv\n")
    team = tmp_path / "team.env"
    team.write_text("PRISM_DATA_ROOT=/from/team\n")
    monkeypatch.setenv("PRISM_SITE_ENV", str(team))
    assert site_paths.site_values()["PRISM_DATA_ROOT"] == "/from/team"


def test_process_environment_beats_every_file(isolated_site, tmp_path, monkeypatch):
    (isolated_site / ".env").write_text("PRISM_DATA_ROOT=/from/dotenv\n")
    team = tmp_path / "team.env"
    team.write_text("PRISM_DATA_ROOT=/from/team\n")
    monkeypatch.setenv("PRISM_SITE_ENV", str(team))
    monkeypatch.setenv("PRISM_DATA_ROOT", "/from/environ")
    assert site_paths.site_values()["PRISM_DATA_ROOT"] == "/from/environ"


def test_missing_site_env_file_is_not_an_error(monkeypatch, tmp_path):
    monkeypatch.setenv("PRISM_SITE_ENV", str(tmp_path / "absent.env"))
    assert site_paths.site_values()["PRISM_DATA_ROOT"] == "<unset:PRISM_DATA_ROOT>"


def test_env_file_ignores_comments_blanks_and_quotes(isolated_site):
    (isolated_site / ".env").write_text(
        "\n".join(
            [
                "# a comment",
                "",
                "   ",
                "not_a_pair",
                "PRISM_HF_HUB = '/quoted/hub'  ",
                'PRISM_ASSETS="/quoted/assets"',
            ]
        )
    )
    values = site_paths.site_values()
    assert values["PRISM_HF_HUB"] == "/quoted/hub"
    assert values["PRISM_ASSETS"] == "/quoted/assets"


def test_env_file_expands_shell_variables(isolated_site, monkeypatch):
    """One shared team file serves every account via $USER."""
    monkeypatch.setenv("USER", "someone")
    (isolated_site / ".env").write_text("PRISM_DATA_ROOT=/projects/$USER/data\n")
    assert site_paths.site_values()["PRISM_DATA_ROOT"] == "/projects/someone/data"


def test_env_file_ignores_names_that_are_not_site_variables(isolated_site):
    (isolated_site / ".env").write_text("HF_TOKEN=hf_secret\nPRISM_DATA_ROOT=/d\n")
    values = site_paths.site_values()
    assert "HF_TOKEN" not in values
    assert values["PRISM_DATA_ROOT"] == "/d"


def test_empty_assignment_does_not_shadow_the_placeholder(isolated_site):
    (isolated_site / ".env").write_text("PRISM_DATA_ROOT=\n")
    assert site_paths.site_values()["PRISM_DATA_ROOT"] == "<unset:PRISM_DATA_ROOT>"


# --------------------------------------------------------------------------
# Interpolation belonging to other systems must survive untouched
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        "${oc.env:PRISM_DATA_ROOT}/SciTS-processed",
        "${hydra:run.dir}",
        "${oc.env:PRISM_AURORAGPT_2B_CHECKPOINT,/lus/flare/projects/<project>/x}",
        "${UNKNOWN_NAME}/data",
    ],
)
def test_foreign_interpolation_passes_through(value):
    assert site_paths.expand(value) == value


def test_non_strings_pass_through_unchanged():
    for value in (None, 7, 1.5, True):
        assert site_paths.expand(value) is value


# --------------------------------------------------------------------------
# expand_tree over decoded JSON
# --------------------------------------------------------------------------


def test_expand_tree_recurses_values_but_not_keys(monkeypatch):
    monkeypatch.setenv("PRISM_DATA_ROOT", "/root")
    tree = {
        "${PRISM_DATA_ROOT}": "keys are left alone",
        "zone_a": {
            "datasets": [
                {"local_path": "${PRISM_DATA_ROOT}/a", "weight": 0.5},
                {"local_path": "${PRISM_DATA_ROOT}/b", "skip": False},
            ]
        },
    }
    out = site_paths.expand_tree(tree)
    assert "${PRISM_DATA_ROOT}" in out
    entries = out["zone_a"]["datasets"]
    assert entries[0]["local_path"] == "/root/a"
    assert entries[1]["local_path"] == "/root/b"
    assert entries[0]["weight"] == 0.5
    assert entries[1]["skip"] is False


def test_expand_tree_preserves_a_pinned_snapshot_sha(monkeypatch):
    """The image-generation configs join a pinned SHA onto the hub root.

    Substituting a bare hub id for these would silently drop the revision —
    which is why they keep the full snapshot path.
    """
    monkeypatch.setenv("PRISM_HF_HUB", "/hub")
    sha = "70d244cc0e5b7b1b2b1d1f0f4e0a0d9e2c3b4a59"
    value = "${PRISM_HF_HUB}/models--Qwen--Qwen3-1.7B/snapshots/" + sha
    assert site_paths.expand_tree(value) == f"/hub/models--Qwen--Qwen3-1.7B/snapshots/{sha}"
