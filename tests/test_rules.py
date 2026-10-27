import math

import numpy as np

from viva_multicell_blueprint.backends.fake_backend import FakeBackend
from viva_multicell_blueprint.core import build_core
from viva_multicell_blueprint.mechanism.rules import RulesProcess


def cells(*spec):
    return [{"id": i, "type": t, "centroid": list(c), "size": s} for i, t, c, s in spec]


def backend(init):
    return FakeBackend({"initial_state": init, "backend_params": {"neighbor_radius": 1.5}},
                       core=build_core())


def rules_proc(rules, seed=0, const=None):
    return RulesProcess({"rules": rules, "seed": seed, "const": const or {}}, core=build_core())


def step(proc, bk_out, grid=None):
    return proc.update({**bk_out, "field_grid": grid or {}}, 1)


def virus_grid(level, shape=(10, 10)):
    return {"virus": np.full(shape, level, dtype=np.float32)}


INFECT = {"kind": "transition", "from": "H", "to": "I", "rate_expr": "g_hv * field['virus']"}


def test_infection_from_virus_field():
    init = cells((1, "H", (2, 2), 1.0), (2, "H", (7, 7), 1.0), (3, "D", (5, 5), 1.0))
    out0 = backend(init).update({}, 1)
    # virus only near cell 1: exposure is the field at the centroid
    grid = {"virus": np.zeros((10, 10), dtype=np.float32)}
    grid["virus"][2, 2] = 1000.0
    p = rules_proc([INFECT], const={"g_hv": 1.0})
    out = step(p, out0, grid)
    assert out["fates"] == {"1": "I"}  # huge rate -> p~1; cell 2 zero virus; D untouched
    # statistics: Pr = 1 - exp(-rate) (Sego ul_rate_to_prob), not rate directly
    rate, n, hits = 0.5, 4000, 0
    init = [{"id": i, "type": "H", "centroid": [1.0, 1.0], "size": 1.0} for i in range(n)]
    o = backend(init).update({}, 1)
    p = rules_proc([INFECT], seed=3, const={"g_hv": rate})
    hits = len(step(p, o, virus_grid(1.0))["fates"])
    assert abs(hits / n - (1 - math.exp(-rate))) < 0.03
    assert abs(hits / n - rate) > 0.05  # clearly not the raw-rate convention
    # deterministic given the seed
    p2 = rules_proc([INFECT], seed=3, const={"g_hv": rate})
    assert len(step(p2, o, virus_grid(1.0))["fates"]) == hits


RESIST = {"kind": "resistance", "scalar": "resist", "rate_expr": "hill(field['ifn'], a_rf)"}
IDEATH = {"kind": "transition", "from": "I", "to": "D",
          "rate_expr": "mu_i * (1 - scalar['resist'])"}


def test_resistance_scalar_then_death():
    init = cells((1, "I", (2, 2), 1.0), (2, "I", (7, 7), 1.0))
    out0 = backend(init).update({}, 1)
    ifn = np.zeros((10, 10), dtype=np.float32)
    ifn[2, 2] = 3.0
    ifn[7, 7] = 0.0
    p = rules_proc([RESIST, IDEATH], seed=1, const={"a_rf": 1.0, "mu_i": 1e6})
    out = step(p, out0, {"ifn": ifn})
    assert out["scalar"]["1"]["resist"] == 3.0 / (1.0 + 3.0)  # hill(x,k)=x/(k+x)
    assert out["scalar"]["2"]["resist"] == 0.0
    # later rule reads earlier scalar: unresistant cell dies, resistant one has lower rate
    assert out["fates"].get("2") == "D"
    # fully-resistant (resist -> 1) cell never dies
    ifn[2, 2] = 1e12
    p = rules_proc([RESIST, IDEATH], seed=1, const={"a_rf": 1.0, "mu_i": 50.0})
    assert "1" not in step(p, out0, {"ifn": ifn})["fates"]


KILL = {"kind": "kill", "killer": "K", "target": "I", "to": "D",
        "rate_expr": "g_ik * neighbors.shared_surface['K'] * scalar['resist'] / size"}


def test_contact_kill_uses_shared_surface():
    # cell 1 touches two K cells, cell 2 touches none, cell 3 touches one but size is huge
    init = cells((1, "I", (2, 2), 1.0), (10, "K", (2, 3), 1.0), (11, "K", (3, 2), 1.0),
                 (2, "I", (8, 8), 1.0),
                 (3, "I", (5, 0), 1e12), (12, "K", (5, 1), 1.0))
    out0 = backend(init).update({}, 1)
    assert out0["shared_surface_by_type"]["1"]["K"] == 2.0
    resist = {"kind": "resistance", "scalar": "resist", "rate_expr": "1"}
    p = rules_proc([resist, KILL], seed=2, const={"g_ik": 1e6})
    out = step(p, out0, {})
    assert out["fates"] == {"1": "D"}  # no contact -> no kill; huge size -> ~0 rate
    assert out["scalar"]["1"]["resist"] == 1.0


ALLEE = {"kind": "allee", "from": "H", "to": "D", "rate_expr":
         "b_h * (1 - scalar['resist']) * neighbors.shared_surface['D'] * "
         "max(0.5 * tot - neighbors.shared_surface['H'], 0) / max(tot, 1e-9) ** 2"}
RECOV = {"kind": "recovery", "from": "D", "to": "H", "rate_expr":
         "b_h * (1 - scalar['resist']) * neighbors.shared_surface['H'] * "
         "max(neighbors.shared_surface['H'] - 0.5 * tot, 0) / max(tot, 1e-9) ** 2"}


def _with_tot(rule):
    r = dict(rule)
    tot = "(neighbors.shared_surface['H'] + neighbors.shared_surface['I'] + neighbors.shared_surface['D'])"
    r["rate_expr"] = r["rate_expr"].replace("tot", tot)
    return r


def test_recovery_and_allee():
    # H cell (1) among dead cells -> allee death; D cell (5) among healthy -> recovery;
    # isolated H (9) is untouched (rate 0, srf_total == 0)
    init = cells((1, "H", (2, 2), 1.0), (2, "D", (2, 3), 1.0), (3, "D", (3, 2), 1.0),
                 (5, "D", (7, 7), 1.0), (6, "H", (7, 8), 1.0), (7, "H", (8, 7), 1.0),
                 (9, "H", (0, 9), 1.0))
    out0 = backend(init).update({}, 1)
    resist0 = {"kind": "resistance", "scalar": "resist", "rate_expr": "0"}
    p = rules_proc([resist0, _with_tot(ALLEE), _with_tot(RECOV)], seed=4, const={"b_h": 1e6})
    out = step(p, out0, {})
    assert out["fates"]["1"] == "D"
    assert out["fates"]["5"] == "H"
    assert "9" not in out["fates"]
    # one fate per cell per step, even though later rules could also match
    assert set(out["fates"]) == {"1", "5"}


def test_dead_cells_skipped_and_removal_events():
    init = cells((1, "H", (2, 2), 1.0), (2, "H", (7, 7), 1.0))
    bk = backend(init)
    out0 = bk.update({"deaths": [{"cell_id": "2"}]}, 1)
    assert out0["alive"]["2"] is False
    rem = {"kind": "kill", "target": "H", "remove": True, "rate_expr": "1e9"}
    div = {"kind": "divide", "type": "H", "rate_expr": "1e9"}
    p = rules_proc([rem], seed=0)
    out = step(p, out0, {})
    assert out["deaths"] == [{"cell_id": "1"}] and out["fates"] == {}
    assert "2" not in out["scalar"]
    p = rules_proc([div], seed=0)
    out = step(p, out0, {})
    assert out["divisions"] == [{"cell_id": "1"}]
