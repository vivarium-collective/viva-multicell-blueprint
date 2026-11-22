import pytest

pytest.importorskip("viva_tellurium")

from viva_multicell_blueprint.core import build_core
from viva_multicell_blueprint.mechanism.subcellular import MODULES, SubcellularProcess

GROWTH = """
model growth
  species S;
  S = 1.0;
  mu = 0.5;
  drive = 0.0;
  J0: -> S; (mu + drive) * S;
end
"""


def make(coupling=None, division="copy"):
    cfg = {"cell_type": "T", "module": "tellurium", "model": {"source": GROWTH},
           "params": {}, "division": division,
           "coupling": coupling or {"inputs": {}, "outputs": {"target_size": "S"}}}
    return SubcellularProcess(cfg, core=build_core())


def world(ids=("1", "2"), **extra):
    d = {"types": {i: "T" for i in ids}, "sizes": {i: 2.0 for i in ids},
         "positions": {i: [1.0, 1.0] for i in ids}, "neighbors": {i: [] for i in ids},
         "alive": {i: True for i in ids}, "field_grid": {}, "divisions": [], "deaths": [],
         "lineage": {}}
    d.update(extra)
    return d


def test_registered():
    assert "tellurium" in MODULES


def test_species_grows_and_feeds_target_size():
    p = make()
    o1 = p.update(world(), 1)
    o2 = p.update(world(), 1)
    for c in ("1", "2"):
        assert o2["sub_scalar"][c]["S"] > o1["sub_scalar"][c]["S"] > 1.0
        assert o2["req_target_size"][c] == pytest.approx(o2["sub_scalar"][c]["S"])
    assert o2["req_target_size"]["1"] > o1["req_target_size"]["1"]


def test_mapped_input_changes_trajectory():
    base = make({"inputs": {"drive": "scalar.d"}, "outputs": {"target_size": "S"}})
    w = world(scalar={"1": {"d": 0.0}, "2": {"d": 1.0}})
    o = base.update(w, 1)
    assert o["req_target_size"]["2"] > o["req_target_size"]["1"]


def test_vars_and_unknown_var_rejected():
    with pytest.raises(ValueError):
        make({"inputs": {"nope": "size"}, "outputs": {}})
    m = MODULES["tellurium"]({"source": GROWTH}, {})
    assert {"S", "mu", "drive"} <= m.input_vars()
    assert "S" in m.output_vars()


def test_split_halves_species_and_daughter_independent():
    p = make(division="split")
    p.update(world(("1",)), 1)
    o = p.update(world(("1", "2"), lineage={"2": "1"}), 1)
    assert o["sub_scalar"]["1"]["S"] == pytest.approx(o["sub_scalar"]["2"]["S"])
