from viva_multicell_blueprint.backends.fake_backend import FakeBackend
from viva_multicell_blueprint.compile.base import MechanicsBackend, OUTPUT_PORTS
from viva_multicell_blueprint.core import build_core

import pytest

from viva_multicell_blueprint.blueprint.data_loading import from_cell_table

TABLE = "id,type,x,y,size\n1,A,0,0,4\n2,B,1,0,5\n3,A,10,10,6\n"


@pytest.fixture
def cells(tmp_path):
    p = tmp_path / "cells.csv"
    p.write_text(TABLE)
    return from_cell_table(p)  # real loader output (Cell shape)


@pytest.fixture
def make(cells):
    return lambda: FakeBackend({"initial_state": cells}, core=build_core())


def test_fake_backend_without_backend_params(cells):
    b = FakeBackend({"initial_state": cells}, core=build_core())
    b.config.pop("backend_params", None)
    assert b.update({}, 1.0)["neighbors"]["1"] == ["2"]


def test_fake_backend_emits_port_shape(make):
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


def test_fake_backend_applies_fates(make):
    b = make()
    out = b.update({"fates": {"1": "C"}, "deaths": [{"cell_id": 3}]}, 1.0)
    assert out["types"]["1"] == "C"
    assert out["positions"]["1"] == [0.0, 0.0]  # fixed
    assert out["alive"]["3"] is False
    assert out["shared_surface_by_type"]["2"] == {"C": 1.0}
