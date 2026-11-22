import copy
from pathlib import Path

import pytest

from viva_multicell_blueprint.blueprint.model import Blueprint
from viva_multicell_blueprint.blueprint.schema import validate

ROOT = Path(__file__).resolve().parents[1]

_DOMAIN = {"extent_um": [100, 100], "grid_resolution_um": 1.0}
_CT = [{"name": "epithelial", "target_size": 50}]
_IS = {"kind": "blocks"}
_FIELDS = [{"name": "v", "D": 1.0, "decay": 0.1}]
_SYS = {"model_ref": "x"}
_SUB = {"epithelial": {"module": "tellurium"}}


def _minimal_v1_dict():
    return {"domain": copy.deepcopy(_DOMAIN), "cell_types": copy.deepcopy(_CT), "initial_state": dict(_IS)}


def _v1_equiv_dict():
    d = _minimal_v1_dict()
    d["adhesion"] = [{"a": "epithelial", "b": "epithelial", "strength": 1.0}]
    d.update(fields=copy.deepcopy(_FIELDS), systemic_ode=dict(_SYS))
    return d


def _v2_dict():
    return {
        "initial_conditions": {"domain": copy.deepcopy(_DOMAIN), "cell_types": copy.deepcopy(_CT),
                               "adhesion": [{"a": "epithelial", "b": "epithelial", "strength": 1.0}],
                               "initial_state": dict(_IS)},
        "model_definition": {"fields": copy.deepcopy(_FIELDS), "systemic": dict(_SYS)},
    }


def _v2_with_subcellular():
    d = _v2_dict()
    d["model_definition"]["subcellular"] = copy.deepcopy(_SUB)
    return d


def test_v1_doc_loads_flat_unchanged():
    bp = Blueprint.from_dict(_minimal_v1_dict())
    assert bp.domain and bp.cell_types and bp.subcellular == {}


def test_v2_doc_flattens_to_same_object():
    assert Blueprint.from_dict(_v2_dict()).to_dict() == Blueprint.from_dict(_v1_equiv_dict()).to_dict()


def test_v2_subcellular_is_captured():
    bp = Blueprint.from_dict(_v2_with_subcellular())
    assert "epithelial" in bp.subcellular and bp.subcellular["epithelial"]["module"] == "tellurium"


def test_mixed_shape_rejected():
    with pytest.raises(ValueError, match="mixed"):
        Blueprint.from_dict({"domain": copy.deepcopy(_DOMAIN), "model_definition": {"fields": copy.deepcopy(_FIELDS)}})


def test_v2_spec_layout_loads():
    bp = Blueprint.from_dict(_v2_dict())
    assert bp.cell_types and bp.adhesion and bp.systemic_ode == _SYS


def test_v2_spec_layout_schema_valid():
    assert validate(_v2_dict()) == []


def test_v2_cell_types_under_model_definition_schema_invalid():
    d = _v2_dict()
    d["model_definition"]["cell_types"] = d["initial_conditions"].pop("cell_types")
    assert validate(d)


def test_v2_cell_types_under_model_definition_rejected():
    d = _v2_dict()
    d["model_definition"]["cell_types"] = d["initial_conditions"].pop("cell_types")
    with pytest.raises(ValueError):
        Blueprint.from_dict(d)


def test_unknown_key_still_rejected_after_flatten():
    d = _v2_dict()
    d["model_definition"]["bogus"] = 1
    with pytest.raises(ValueError):
        Blueprint.from_dict(d)


def test_existing_blueprint_files_still_load():
    paths = list((ROOT / "workspace" / "blueprints").glob("*.yaml"))
    assert paths
    for p in paths:
        Blueprint.load(p)
