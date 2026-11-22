import math

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


def test_mapped_input_matches_analytic():
    # dS/dt = (mu + drive) * S, mu = 0.5, S0 = 1  ->  S(1) = exp(mu + drive)
    p = make({"inputs": {"drive": "scalar.d"}, "outputs": {"target_size": "S"}})
    o = p.update(world(scalar={"1": {"d": 0.0}, "2": {"d": 1.0}}), 1)
    assert o["req_target_size"]["1"] == pytest.approx(math.exp(0.5), rel=1e-5)
    assert o["req_target_size"]["2"] == pytest.approx(math.exp(1.5), rel=1e-5)


def test_interleaved_cells_with_different_initial_species_stay_isolated():
    m = MODULES["tellurium"]({"source": GROWTH}, {})
    a = m.init_cell("a", {})
    b = m.init_cell("b", {})
    a["species"]["S"] = 1.0
    b["species"]["S"] = 5.0
    ea, eb = 1.0, 5.0
    for _ in range(4):  # A, B, A, B ... ; B also gets a drive, A none
        a = m.step(a, {}, 0.5)
        b = m.step(b, {"drive": 0.5}, 0.5)
        ea *= math.exp(0.5 * 0.5)
        eb *= math.exp(1.0 * 0.5)
        assert m.read(a)["S"] == pytest.approx(ea, rel=1e-5)
        assert m.read(b)["S"] == pytest.approx(eb, rel=1e-5)
        assert m.read(a)["drive"] == 0.0 and m.read(b)["drive"] == 0.5


RULED = """
model ruled
  species S;
  S = 1.0;
  k = 0.5;
  r := k * 2;
  J0: -> S; r * S;
end
"""


def test_assignment_rule_param_not_settable_and_steps():
    m = MODULES["tellurium"]({"source": RULED}, {})
    assert "r" not in m.input_vars() and "k" in m.input_vars()
    st = m.step(m.init_cell("c", {}), {"k": 1.0}, 1.0)   # r = 2 -> S = e^2
    assert m.read(st)["S"] == pytest.approx(math.exp(2.0), rel=1e-5)
    assert m.read(st)["r"] == pytest.approx(2.0)


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
