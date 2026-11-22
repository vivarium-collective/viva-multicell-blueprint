import pytest

from viva_multicell_blueprint.blueprint.model import Blueprint
from viva_multicell_blueprint.compile import builders as b

CELLS = [{"id": 1, "type": "A", "centroid": [1.0, 1.0], "size": 1.0},
         {"id": 2, "type": "B", "centroid": [3.0, 1.0], "size": 1.0}]


def bp(**kw):
    base = dict(
        domain={"extent_um": [10, 10], "grid_resolution_um": 1.0},
        cell_types=[{"name": "A"}, {"name": "B"}],
        initial_state={"cells": CELLS},
        adhesion=[{"a": "A", "b": "B", "J": 1.0}],
    )
    base.update(kw)
    return Blueprint(**base)


SUB = {"A": {"module": "test_linear", "model": {"x0": 1.0}, "params": {"rate": 1.0},
             "coupling": {"inputs": {}, "outputs": {}}, "division": "copy", "seed": 3}}


@pytest.mark.parametrize("backend,addr", [
    ("cpm", "local:CPMBackend"), ("vertex", "local:VertexBackend"), ("munk", "local:MunkBackend")])
def test_build_mechanics(backend, addr):
    node = b.build_mechanics(bp(), backend)
    assert node["address"] == addr


def test_build_mechanics_unknown_backend():
    with pytest.raises(ValueError):
        b.build_mechanics(bp(), "nope")


def test_fields_rules_absent_none():
    assert b.build_fields(bp()) is None
    assert b.build_rules(bp()) is None
    assert b.build_systemic(bp()) is None
    assert b.build_immune(bp()) is None
    assert b.build_video(bp(), None) is None


def test_fields_rules_present():
    p = bp(fields=[{"name": "f", "D": 1.0}],
           rules=[{"type": "kill", "rate_expr": "0.1"}])
    assert b.build_fields(p)["address"] == "local:FieldProcess"
    assert b.build_rules(p)["address"] == "local:RulesProcess"


def test_systemic_and_immune():
    p = bp(systemic_ode={"model_ref": "price", "params": {"c": 1}},
           immune={"types": {"macrophage_inflow": "M"}})
    s = b.build_systemic(p)
    assert s["ode"]["address"] == "local:SystemicODEProcess"
    assert s["counts"]["address"] == "local:CellCountStep"
    assert b.build_immune(p)["address"] == "local:ImmuneRecruitmentProcess"


def test_video():
    assert b.build_video(bp(), "x", "cpm")["address"] == "local:SpatialVideoStep"


def test_subcellular_and_merge():
    p = bp(subcellular=SUB)
    nodes = b.build_subcellular(p)
    assert len(nodes) == 1
    assert nodes[0]["address"] == "local:SubcellularProcess"
    assert nodes[0]["config"]["cell_type"] == "A"
    assert nodes[0]["config"]["seed"] == 3
    assert b.build_merge(p)["address"] == "local:CellActuationMergeStep"
    assert b.build_merge(p)["config"]["arbitration"] == "rules_first"
    assert b.build_subcellular(bp()) == []
    assert b.build_merge(bp()) is None


def test_subcellular_dangling_type():
    with pytest.raises(ValueError, match="Ghost"):
        b.build_subcellular(bp(subcellular={"Ghost": {"module": "test_linear"}}))
