"""Guards against silent drift between the places that declare which
transformers version each Aurora build variant gets.

Aurora builds two mutually exclusive variants, because Intern-S2 Preview needs
``transformers>=5.2.0`` and the system vLLM 0.15.0+xpu declares
``transformers<5,>=4.56.0`` — an empty intersection:

- default      system transformers (4.57.6), Intern-S2 encoders unavailable
- --intern-s2  overlays transformers==5.2.0, vLLM outside its declared range

Three files have to agree on that split:

- requirements/aurora-py3.12.intern-s2.nodeps.txt (exact pins, --no-deps)
- requirements/aurora-py3.12.nodeps.txt (must NOT carry the overlay pins)
- tools/build_aurora_env.sh (both verification floors, and the flag itself)

Each is checked against independent, hardcoded ground truth below — comparing
the files only against *each other* would pass even if they regressed in
lockstep (e.g. all stuck at 4.57.6, or all bumped past what Intern-S2 needs).
"""

import ast
import re
from pathlib import Path

import pytest
from packaging.version import Version

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parent.parent
NODEPS_FILE = REPO_ROOT / "requirements" / "aurora-py3.12.nodeps.txt"
INTERN_S2_FILE = REPO_ROOT / "requirements" / "aurora-py3.12.intern-s2.nodeps.txt"
BUILD_SCRIPT = REPO_ROOT / "tools" / "build_aurora_env.sh"

# Independent source of truth for this test — not derived from any file under
# test. Keep in sync manually with Intern-S2 Preview's actual requirement: its
# vendored config imports RopeParameters from transformers.modeling_rope_utils,
# added in transformers 5.2.0.
REQUIRED_TRANSFORMERS_FOR_INTERN_S2 = "5.2.0"

# OLMo-3, the default backbone, needs >=4.57.0. The default variant must clear
# this without an overlay — frameworks/2025.3.1 ships 4.57.6.
REQUIRED_TRANSFORMERS_FOR_OLMO3 = "4.57.0"

# vLLM 0.15.0+xpu's declared ceiling (exclusive). Anything at or above this
# puts the system vLLM outside its own metadata, which is exactly the trade
# the --intern-s2 flag exists to make explicit.
VLLM_TRANSFORMERS_CEILING = "5"

# The three packages that move together in the overlay. Installed --no-deps,
# so nothing re-resolves them into a working combination later.
INTERN_S2_OVERLAY_PINS = {
    "transformers": "5.2.0",
    "huggingface-hub": "1.32.0",
    "hf_xet": "1.6.0",
}


def _parse_exact_pins(text):
    """Map every exact ``name==X.Y.Z`` pin in a requirements file.

    Deliberately only matches exact pins: an open floor like
    ``transformers>=5.2.0`` is absent from the result rather than silently
    accepted, and the callers assert on what they expect to find.
    """
    return dict(re.findall(r"^([A-Za-z0-9._-]+)==([0-9][0-9A-Za-z.]*)$", text, re.MULTILINE))


def _parse_build_script_floors(text):
    """Extract both transformers floors from tools/build_aurora_env.sh.

    Returns ``(default_floor, intern_s2_floor)``. The default lives in the
    ``required`` dict literal; the overlay's is assigned conditionally after
    it. Both are read with ``ast`` rather than regex over the values, so
    quoting and formatting changes cannot silently break parsing.
    """
    match = re.search(r"^required = \{.*?^\}", text, re.DOTALL | re.MULTILINE)
    assert match is not None, f"could not locate `required = {{...}}` dict in {BUILD_SCRIPT.name}"
    dict_node = ast.parse(match.group(0), mode="exec").body[0].value
    assert isinstance(dict_node, ast.Dict)

    default_floor = None
    for key_node, value_node in zip(dict_node.keys, dict_node.values, strict=True):
        if isinstance(key_node, ast.Constant) and key_node.value == "transformers":
            assert isinstance(value_node, ast.Constant), (
                "expected required['transformers'] to be a string literal"
            )
            default_floor = value_node.value
    assert default_floor is not None, (
        f"no 'transformers' entry in the required dict in {BUILD_SCRIPT.name}"
    )

    # `required["transformers"] = "5.2.0"` inside the `if intern_s2:` block.
    overlay = re.search(
        r'^\s*required\["transformers"\]\s*=\s*"([0-9][0-9A-Za-z.]*)"',
        text,
        re.MULTILINE,
    )
    assert overlay is not None, (
        f"no conditional required['transformers'] assignment in {BUILD_SCRIPT.name} — "
        "the --intern-s2 variant would verify against the default floor"
    )
    return default_floor, overlay.group(1)


def test_overlay_file_pins_every_coupled_package_exactly():
    """All three move together; an open floor here would let --no-deps pick up
    an untested trio."""
    pins = _parse_exact_pins(INTERN_S2_FILE.read_text())
    for name, expected in INTERN_S2_OVERLAY_PINS.items():
        assert pins.get(name) == expected, (
            f"{INTERN_S2_FILE.name} pins {name}=={pins.get(name)}, expected "
            f"{expected}. These three are validated together — bump them as a "
            "set after re-validating, rather than editing this test."
        )


def test_overlay_transformers_pin_meets_intern_s2_requirement():
    pin = _parse_exact_pins(INTERN_S2_FILE.read_text())["transformers"]
    assert Version(pin) >= Version(REQUIRED_TRANSFORMERS_FOR_INTERN_S2), (
        f"{INTERN_S2_FILE.name} pins transformers=={pin}, but Intern-S2 Preview "
        f"needs >={REQUIRED_TRANSFORMERS_FOR_INTERN_S2}."
    )


def test_default_nodeps_file_does_not_carry_the_overlay():
    """The default build must leave transformers at the system version.

    If the overlay pins leak back into the default manifest, every default
    build silently breaks vLLM again — which is the thing the split fixed.
    """
    pins = _parse_exact_pins(NODEPS_FILE.read_text())
    leaked = sorted(set(pins) & set(INTERN_S2_OVERLAY_PINS))
    assert not leaked, (
        f"{NODEPS_FILE.name} pins {leaked}, which belongs in "
        f"{INTERN_S2_FILE.name}. The default variant installs this file and "
        "must not override the system transformers."
    )


def test_build_script_default_floor_covers_olmo3_but_not_intern_s2():
    """The default floor has to be high enough for OLMo-3 and low enough that
    a default build (system 4.57.6, no overlay) actually passes verification."""
    default_floor, _ = _parse_build_script_floors(BUILD_SCRIPT.read_text())
    assert Version(default_floor) >= Version(REQUIRED_TRANSFORMERS_FOR_OLMO3), (
        f"default required['transformers'] floor is {default_floor}, below "
        f"OLMo-3's {REQUIRED_TRANSFORMERS_FOR_OLMO3}."
    )
    assert Version(default_floor) < Version(VLLM_TRANSFORMERS_CEILING), (
        f"default required['transformers'] floor is {default_floor}, at or "
        f"above vLLM's declared ceiling ({VLLM_TRANSFORMERS_CEILING}). A "
        "default build installs no overlay, so this floor would fail against "
        "the system transformers and the default variant could never build."
    )


def test_build_script_intern_s2_floor_covers_intern_s2_requirement():
    _, intern_s2_floor = _parse_build_script_floors(BUILD_SCRIPT.read_text())
    assert Version(intern_s2_floor) >= Version(REQUIRED_TRANSFORMERS_FOR_INTERN_S2), (
        f"{BUILD_SCRIPT.name}'s --intern-s2 required['transformers'] floor is "
        f"{intern_s2_floor}, below the {REQUIRED_TRANSFORMERS_FOR_INTERN_S2} "
        "Intern-S2 Preview needs — a build with a stale floor would pass "
        "verification against an inadequate transformers version."
    )


def test_build_script_intern_s2_floor_matches_the_overlay_pin():
    """The floor verifies what the overlay installs; a mismatch means the
    build could pass verification against a version it never installs."""
    _, intern_s2_floor = _parse_build_script_floors(BUILD_SCRIPT.read_text())
    pin = _parse_exact_pins(INTERN_S2_FILE.read_text())["transformers"]
    assert intern_s2_floor == pin, (
        f"{BUILD_SCRIPT.name} verifies >={intern_s2_floor} but {INTERN_S2_FILE.name} installs {pin}"
    )


def test_build_script_installs_the_overlay_only_behind_the_flag():
    """The overlay manifest must be referenced, and guarded."""
    text = BUILD_SCRIPT.read_text()
    assert "--intern-s2" in text, "no --intern-s2 flag in the build script"
    assert INTERN_S2_FILE.name in text, (
        f"{BUILD_SCRIPT.name} never references {INTERN_S2_FILE.name}, so the "
        "overlay can never be installed"
    )
    install = re.search(
        r'if \[ "\$INTERN_S2" = "1" \]; then\n\s*echo[^\n]*\n\s*'
        r'uv pip install --no-deps -r "\$INTERN_S2_FILE"',
        text,
    )
    assert install is not None, (
        "the overlay install is not guarded by an INTERN_S2 test — a default "
        "build would install transformers 5.x and break vLLM"
    )


@pytest.mark.parametrize("stale_floor", ["4.57.6", "5.0.0", "5.1.9"])
def test_stale_floor_is_correctly_rejected(stale_floor):
    """Negative fixture: proves the comparison above can actually fail, rather
    than being a vacuously-true no-op. Uses synthetic strings, not the real
    files.
    """
    assert not Version(stale_floor) >= Version(REQUIRED_TRANSFORMERS_FOR_INTERN_S2)


@pytest.mark.parametrize("bad_default", ["4.56.0", "5.2.0"])
def test_out_of_range_default_floor_is_correctly_rejected(bad_default):
    """Negative fixture for the two-sided default check: too low for OLMo-3,
    or at/above vLLM's ceiling."""
    in_range = Version(bad_default) >= Version(REQUIRED_TRANSFORMERS_FOR_OLMO3) and Version(
        bad_default
    ) < Version(VLLM_TRANSFORMERS_CEILING)
    assert not in_range
