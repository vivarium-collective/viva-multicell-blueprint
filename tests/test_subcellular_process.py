import numpy as np
import pytest

from viva_multicell_blueprint.core import build_core
from viva_multicell_blueprint.mechanism.subcellular import (
    SubcellularProcess, resolve_inputs, resolve_outputs)


def make(division="copy", coupling=None, params=None, cell_type="T"):
    cfg = {"cell_type": cell_type, "module": "test_linear", "model": {"x0": 1.0},
           "params": params or {"rate": 1.0}, "division": division,
           "coupling": coupling or {"inputs": {}, "outputs": {"target_size": "x"}}}
    return SubcellularProcess(cfg, core=build_core())


def world(ids=(1, 2), **extra):
    ids = [str(i) for i in ids]
    d = {"types": {i: "T" for i in ids}, "sizes": {i: 2.0 for i in ids},
         "positions": {i: [1.0, 1.0] for i in ids}, "neighbors": {i: [] for i in ids},
         "alive": {i: True for i in ids}, "field_grid": {}, "divisions": [], "deaths": [],
         "lineage": {}}
    d.update(extra)
    return d


def test_holds_and_steps_per_cell_state():
    p = make()
    o1 = p.update(world(), 1)
    o2 = p.update(world(), 1)
    assert set(o2["sub_scalar"]) == {"1", "2"}
    assert o2["sub_scalar"]["1"]["x"] > o1["sub_scalar"]["1"]["x"] > 1.0
    assert o2["req_target_size"] == {"1": o2["sub_scalar"]["1"]["x"],
                                     "2": o2["sub_scalar"]["2"]["x"]}


def test_other_type_cells_ignored():
    w = world()
    w["types"]["2"] = "Z"
    assert set(make().update(w, 1)["sub_scalar"]) == {"1"}


def _divide(division):
    p = make(division)
    p.update(world(ids=(1,)), 1)
    mx = p.update(world(ids=(1,)), 1)["sub_scalar"]["1"]["x"]
    w = world(ids=(1, 5), lineage={"5": "1"}, divisions=[{"cell_id": "1"}])
    return mx, p.update(w, 1)["sub_scalar"]


def test_division_copy_split_reset():
    mx, s = _divide("copy")
    assert s["5"]["x"] == pytest.approx(s["1"]["x"])
    assert s["5"]["x"] > mx
    mx, s = _divide("split")
    assert s["5"]["x"] == pytest.approx(s["1"]["x"]) and s["5"]["x"] < mx
    mx, s = _divide("reset")
    assert s["5"]["x"] == pytest.approx(2.0)  # fresh x0=1 + one step of rate 1
    assert s["1"]["x"] > 2.5


def test_copy_does_not_alias():
    p = make()
    p.update(world(ids=(1,)), 1)
    p.update(world(ids=(1, 5), lineage={"5": "1"}), 1)
    assert p._cells["1"] is not p._cells["5"]


def test_death_drops_state():
    p = make()
    p.update(world(), 1)
    out = p.update(world(deaths=[{"cell_id": "2"}]), 1)
    assert set(out["sub_scalar"]) == {"1"} and "2" not in p._cells
    w = world(); w["alive"]["1"] = False
    out = p.update(w, 1)
    assert "1" not in p._cells and set(out["req_target_size"]) == {"2"}


def test_coupling_inputs_resolve_field_at_centroid():
    grid = np.zeros((5, 5)); grid[1, 1] = 7.0
    maps = {"types": {"a": "T"}, "sizes": {"a": 3.0}, "positions": {"a": [1.2, 0.8]},
            "neighbors": {"a": ["b", "c"]}, "alive": {"a": True, "b": True, "c": True},
            "field_grid": {"f": grid}, "scalar": {"a": {"s": 4.0}},
            "shared_surface_by_type": {"a": {"B": 2.5}}}
    maps["types"].update({"b": "B", "c": "B"})
    got = resolve_inputs({"u": "field.f@centroid", "sz": "size", "n": "neighbors.count",
                          "sh": "neighbors.shared_surface.B", "ty": "type",
                          "sc": "scalar.s", "k": "const.kk"}, "a", maps, const={"kk": 9.0})
    assert got == {"u": 7.0, "sz": 3.0, "n": 2, "sh": 2.5, "ty": "T", "sc": 4.0, "k": 9.0}


def test_coupling_inputs_drive_module():
    grid = np.full((4, 4), 3.0)
    p = make(coupling={"inputs": {"drive": "field.f@centroid"}, "outputs": {"target_size": "x"}})
    out = p.update(world(ids=(1,), field_grid={"f": grid}), 1)
    assert out["sub_scalar"]["1"]["x"] == pytest.approx(1.0 + 1.0 + 3.0)


def test_coupling_outputs_map_to_requests():
    got = resolve_outputs({"target_size": "x", "divide_when": "d", "die_when": "k",
                           "become.M": "b", "secrete.f": "s", "motility": "m", "adhesion": "a"},
                          {"x": 5.0, "d": 1.0, "k": 0.0, "b": 1.0, "s": 0.5, "m": 2.0, "a": 3.0})
    assert got == {"target_size": 5.0, "divide_when": True, "die_when": False,
                   "become.M": True, "secrete.f": 0.5, "motility": 2.0, "adhesion": 3.0}
    p = make(coupling={"inputs": {}, "outputs": {"divide_when": "gt1", "become.M": "gt1",
                                                 "secrete.f": "x", "motility": "x"}})
    out = p.update(world(ids=(1,)), 1)
    assert out["req_divide"] == [{"cell_id": "1"}]
    assert out["req_become"] == {"1": "M"}
    assert out["req_secrete"] == {"1": {"f": out["sub_scalar"]["1"]["x"]}}
    assert set(out["req_motility"]) == {"1"} and out["req_die"] == []


def test_unknown_coupling_var_errors():
    with pytest.raises(ValueError, match="unknown"):
        resolve_outputs({"target_size": "nope"}, {"x": 1.0})
    with pytest.raises(ValueError, match="unknown"):
        resolve_inputs({"u": "bogus.thing"}, "a", {"types": {"a": "T"}})
    # validated at construction, not first update
    with pytest.raises(ValueError, match="nope"):
        make(coupling={"inputs": {}, "outputs": {"target_size": "nope"}})
    with pytest.raises(ValueError, match="nope"):
        make(coupling={"inputs": {}, "outputs": {"divide_when": {"var": "nope", "gt": 1}}})


def test_unknown_input_var_errors_at_construction():
    with pytest.raises(ValueError, match="bogus_in"):
        make(coupling={"inputs": {"bogus_in": "size"}, "outputs": {}})


def test_threshold_grammar():
    ro = lambda spec, v, const=None: resolve_outputs(spec, {"B": v}, const)
    assert ro({"divide_when": {"var": "B", "gt": 2.0}}, 2.5) == {"divide_when": True}
    assert ro({"divide_when": {"var": "B", "gt": 2.0}}, 1.5) == {"divide_when": False}
    assert ro({"die_when": {"var": "B", "lt": 0.1}}, 0.05) == {"die_when": True}
    assert ro({"divide_when": {"var": "B", "gt": "const.Bdiv"}}, 3.0, {"Bdiv": 2.0}) == {
        "divide_when": True}
    with pytest.raises(ValueError, match="Bdiv"):
        ro({"divide_when": {"var": "B", "gt": "const.Bdiv"}}, 3.0)
    assert ro({"divide_when": "B"}, 0.9) == {"divide_when": True}  # bare form kept
    # end to end: x starts 1, +1/step; x=2,3,4; divides once x > Bdiv=3.5 (3rd step)
    p = make(coupling={"inputs": {}, "const": {"Bdiv": 3.5},
                       "outputs": {"divide_when": {"var": "x", "gt": "const.Bdiv"}}})
    fired = [bool(p.update(world(ids=(1,)), 1)["req_divide"]) for _ in range(3)]
    assert fired == [False, False, True]


def test_split_missing_module_method_and_single_halving(monkeypatch):
    from viva_multicell_blueprint.mechanism import subcellular as sc
    class NoSplit:
        def __init__(self, model=None, params=None): pass
        def input_vars(self): return set()
        def output_vars(self): return set()
    monkeypatch.setitem(sc.MODULES, "nosplit", NoSplit)
    with pytest.raises(ValueError, match="split"):
        SubcellularProcess({"cell_type": "T", "module": "nosplit", "division": "split"},
                           core=build_core())
    p = make("split")
    p.update(world(ids=(1,)), 1)           # x = 2
    out = p.update(world(ids=(1, 5, 6), lineage={"5": "1", "6": "1"}), 1)
    assert out["sub_scalar"]["1"]["x"] == pytest.approx(2.0)  # halved once (2/2+1); halving twice would give 1.5


def test_split_halves_at_request_so_one_crossing_requests_once():
    """division='split': the dividing cell's state is halved the SAME tick it requests
    division, so a single threshold crossing yields one request, not one per tick."""
    import pytest as _pt
    _pt.importorskip("viva_tellurium")
    from viva_multicell_blueprint.core import build_core
    from viva_multicell_blueprint.mechanism.subcellular import SubcellularProcess
    src = "model g\n species S; S = 30.0; mu = 0.3;\n J0: -> S; mu * S;\nend\n"
    cfg = {"cell_type": "T", "module": "tellurium", "model": {"source": src}, "params": {},
           "division": "split",
           "coupling": {"const": {"t": 40.0}, "inputs": {},
                        "outputs": {"target_size": "S", "divide_when": {"var": "S", "gt": "const.t"}}}}
    p = SubcellularProcess(cfg, core=build_core())
    w = {"types": {"1": "T"}, "sizes": {"1": 2.0}, "positions": {"1": [1.0, 1.0]},
         "neighbors": {"1": []}, "alive": {"1": True}, "field_grid": {}, "divisions": [],
         "deaths": [], "lineage": {}}
    reqs = [len(p.update(w, 1)["req_divide"]) for _ in range(3)]
    assert reqs == [1, 0, 0], reqs          # 30 -> 40.5 (fires, halves) -> ~27 -> ~36 (no)


def test_split_done_cleared_when_mother_leaves_live_set():
    """A mother that requested division but then died (daughter never appeared) must not
    linger in _split_done, or a later re-request would skip its halving."""
    import pytest as _pt
    _pt.importorskip("viva_tellurium")
    from viva_multicell_blueprint.core import build_core
    from viva_multicell_blueprint.mechanism.subcellular import SubcellularProcess
    src = "model g\n species S; S = 30.0; mu = 0.3;\n J0: -> S; mu * S;\nend\n"
    cfg = {"cell_type": "T", "module": "tellurium", "model": {"source": src}, "params": {},
           "division": "split",
           "coupling": {"const": {"t": 40.0}, "inputs": {},
                        "outputs": {"divide_when": {"var": "S", "gt": "const.t"}}}}
    p = SubcellularProcess(cfg, core=build_core())
    w = {"types": {"1": "T"}, "sizes": {"1": 2.0}, "positions": {"1": [1.0, 1.0]},
         "neighbors": {"1": []}, "alive": {"1": True}, "field_grid": {}, "divisions": [],
         "deaths": [], "lineage": {}}
    assert len(p.update(w, 1)["req_divide"]) == 1 and p._split_done == {"1"}
    w["alive"] = {"1": False}
    p.update(w, 1)
    assert p._split_done == set()
