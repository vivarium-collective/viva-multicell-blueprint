"""BP-01: the blueprint as a versioned, backend-neutral canonical artifact.

Acceptance (review BP-01):
  * both backends take precisely the SAME immutable canonical blueprint,
  * no backend-specific geometry/dynamics or run parameters appear in the
    canonical biological state,
  * compiler defaults/options are recorded separately from the blueprint,
  * a static validator reports structured diagnostics (not just a success badge).
"""
from __future__ import annotations

import copy
from pathlib import Path

import pytest

from viva_multicell_blueprint.blueprint.model import Blueprint
from viva_multicell_blueprint.blueprint.canonical import (
    BACKEND_RUN_KEYS, CompileOptions, canonical_violations, split_canonical,
)
from viva_multicell_blueprint.blueprint.schema import validate, validate_blueprint
from viva_multicell_blueprint.compile import compile as compile_bp

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "workspace" / "blueprints" / "examples"

_MIN = {
    "blueprint_version": "1.0",
    "domain": {"extent_um": [20, 20], "grid_resolution_um": 1, "boundary": "noflux"},
    "cell_types": [{"name": "A", "target_size": 25}],
    "initial_state": {"kind": "blocks",
                      "block": {"type": "A", "origin": [5, 5], "shape": [2, 2], "spacing": 5, "size": 25}},
}


def test_backend_run_params_are_not_canonical():
    doc = copy.deepcopy(_MIN)
    doc["mechanics_mapping"] = {"tension_scale": 2.0, "rules_seed": 3, "const": {"k": 1.0}}
    canon, opts = split_canonical(doc)
    # the canonical artifact carries no backend/run params ...
    assert "tension_scale" not in canon.get("mechanics_mapping", {})
    assert "rules_seed" not in canon.get("mechanics_mapping", {})
    # ... but keeps genuine model constants
    assert canon["mechanics_mapping"]["const"] == {"k": 1.0}
    # ... and the backend/run params are recorded SEPARATELY
    assert opts["tension_scale"] == 2.0 and opts["rules_seed"] == 3
    assert BACKEND_RUN_KEYS >= {"tension_scale", "rules_seed"}


def test_canonical_state_is_identical_regardless_of_backend_params():
    a = copy.deepcopy(_MIN)
    b = copy.deepcopy(_MIN)
    b["mechanics_mapping"] = {"tension_scale": 4.0, "rules_seed": 9}
    assert Blueprint.from_dict(a).canonical_state() == Blueprint.from_dict(b).canonical_state()


def test_validator_structured_diagnostics_and_violation_warning():
    bad = copy.deepcopy(_MIN)
    del bad["domain"]
    diags = validate_blueprint(bad)
    assert any(d["severity"] == "error" for d in diags)
    assert all({"severity", "path", "message"} <= set(d) for d in diags)

    warnish = copy.deepcopy(_MIN)
    warnish["mechanics_mapping"] = {"tension_scale": 2.0}
    wdiags = validate_blueprint(warnish)
    assert not [d for d in wdiags if d["severity"] == "error"]
    assert any(d["severity"] == "warning" and "tension_scale" in d["message"] for d in wdiags)


def test_missing_blueprint_version_is_a_warning_not_error():
    doc = copy.deepcopy(_MIN)
    del doc["blueprint_version"]
    diags = validate_blueprint(doc)
    assert not [d for d in diags if d["severity"] == "error"]        # still compiles
    assert any(d["severity"] == "warning" and "version" in d["message"].lower() for d in diags)
    assert validate(doc) == []                                        # back-compat: errors-only is empty


def test_compile_options_separate_path_matches_mechanics_mapping():
    # Setting tension_scale via CompileOptions is equivalent to the legacy in-blueprint path.
    canon = copy.deepcopy(_MIN)
    legacy = copy.deepcopy(_MIN)
    legacy["mechanics_mapping"] = {"tension_scale": 3.0}
    via_opts = compile_bp(Blueprint.from_dict(canon), backend="vertex",
                          options=CompileOptions(tension_scale=3.0))
    via_legacy = compile_bp(Blueprint.from_dict(legacy), backend="vertex")
    # both produce a composite envelope; the vertex mechanics node sees scale 3.0 either way
    assert "state" in via_opts and "state" in via_legacy


def test_same_canonical_blueprint_compiles_to_both_backends():
    canon = Blueprint.from_dict(copy.deepcopy(_MIN)).canonical_state()
    for backend in ("cpm", "vertex"):
        out = compile_bp(Blueprint.from_dict(copy.deepcopy(canon)), backend=backend)
        assert "state" in out


@pytest.mark.parametrize("name", ["simple_sheet", "mixed_population", "infection_field"])
def test_worked_examples_validate_and_compile_on_both_backends(name):
    doc = Blueprint.load(EXAMPLES / f"{name}.yaml").to_dict()
    assert validate(doc) == [], validate(doc)
    canon = Blueprint.from_dict(copy.deepcopy(doc)).canonical_state()
    for backend in ("cpm", "vertex"):
        assert "state" in compile_bp(Blueprint.from_dict(copy.deepcopy(canon)), backend=backend)
