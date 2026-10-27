from viva_multicell_blueprint.backends.fake_backend import FakeBackend
from viva_multicell_blueprint.compile.base import MechanicsBackend, OUTPUT_PORTS
from viva_multicell_blueprint.core import build_core

CELLS = [
    {"id": 1, "type": "A", "centroid": (0.0, 0.0), "size": 4.0},
    {"id": 2, "type": "B", "centroid": (1.0, 0.0), "size": 5.0},
    {"id": 3, "type": "A", "centroid": (10.0, 10.0), "size": 6.0},
]


def make():
    return FakeBackend({"initial_state": CELLS}, core=build_core())


def test_fake_backend_emits_port_shape():
    b = make()
    assert isinstance(b, MechanicsBackend)
    assert set(b.outputs()) == set(OUTPUT_PORTS)
    assert set(b.inputs()) == {"fates", "field_grid", "divisions", "deaths"}
    out = b.update({}, 1.0)
    assert set(out) == set(OUTPUT_PORTS)
    assert out["positions"]["1"] == [0.0, 0.0]
    assert out["sizes"]["2"] == 5.0
    assert out["types"] == {"1": "A", "2": "B", "3": "A"}
    assert out["alive"] == {"1": True, "2": True, "3": True}
    assert out["neighbors"]["1"] == ["2"] and out["neighbors"]["3"] == []
    assert out["shared_surface_by_type"]["1"] == {"B": 1.0}


def test_fake_backend_applies_fates():
    b = make()
    out = b.update({"fates": {"1": "C"}, "deaths": [{"cell_id": 3}]}, 1.0)
    assert out["types"]["1"] == "C"
    assert out["positions"]["1"] == [0.0, 0.0]  # fixed
    assert out["alive"]["3"] is False
    assert out["shared_surface_by_type"]["2"] == {"C": 1.0}
