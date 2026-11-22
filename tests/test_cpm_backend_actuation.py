"""CPMBackend honors the per-cell actuation ports and emits real division lineage."""
from __future__ import annotations

import copy
import warnings

import pytest

pytest.importorskip("viva_cpm")

from viva_multicell_blueprint.core import build_core  # noqa: E402


def _make(cells, backend_params=None):
    from viva_multicell_blueprint.backends.cpm_backend import CPMBackend

    bp = {"dims": [28, 14, 1], "mcs_per_update": 5}
    bp.update(backend_params or {})
    return CPMBackend(
        {"initial_state": cells, "cell_types": [], "adhesion": [], "domain": {},
         "backend_params": bp},
        core=build_core(),
    )


def _cells():
    return [
        {"id": 1, "type": "A", "centroid": [6.0, 6.0], "size": 36.0},
        {"id": 2, "type": "A", "centroid": [18.0, 6.0], "size": 16.0},
    ]


def _run(inputs_seq, **kw):
    b = _make(_cells(), **kw)
    return [b.update(i, 1.0) for i in inputs_seq]


def test_parity_empty_new_ports_unchanged():
    base = _run([{}, {}, {}])
    with_empty = _run([{"target_size": {}, "motility": {}, "adhesion": {}, "divisions": []}] * 3)
    assert base == with_empty
    assert all(o["lineage"] == {} for o in base)


def test_target_size_changes_emitted_size():
    outs = _run([{}] + [{"target_size": {"1": 80.0}}] * 12)
    base = _run([{}] * 13)
    assert outs[-1]["sizes"]["1"] > 55.0
    assert outs[-1]["sizes"]["1"] > base[-1]["sizes"]["1"] + 15


def test_motility_and_adhesion_accepted_without_change_when_absent():
    b = _make(_cells())
    out = b.update({"motility": {"99": 2.0}, "adhesion": {"99": 1.0}}, 1.0)  # unknown id: no-op
    assert set(out["alive"]) == {"1", "2"}


def test_adhesion_sets_type_pair_contact_energy():
    b = _make(_cells())
    calls = []

    class Spy:
        def __init__(self, w):
            self._w = w

        def set_contact(self, a, c, j):
            calls.append((a, c, j))
            return self._w.set_contact(a, c, j)

        def __getattr__(self, n):
            return getattr(self._w, n)

    b._world = Spy(b._world)
    b.update({"adhesion": {"1": {"A": 3.0}}}, 1.0)
    assert (b._type_ids["A"], b._type_ids["A"], 3.0) in calls


def test_division_emits_daughter_and_lineage():
    b = _make(_cells(), {"mcs_per_update": 0})
    out0 = b.update({}, 1.0)
    out = b.update({"divisions": [{"cell_id": 1}]}, 1.0)
    daughters = set(out["lineage"])
    assert len(daughters) == 1
    (d,) = daughters
    assert out["lineage"][d] == "1"
    assert d not in out0["alive"] and d != "2"
    for port in ("positions", "sizes", "types", "alive", "neighbors",
                 "shared_surface_by_type", "footprints"):
        assert d in out[port], port
    assert out["alive"][d] is True and out["types"][d] == "A"
    assert out["sizes"]["1"] < 36.0 and out["sizes"][d] > 0
    # daughter persists alive next step; lineage is per-event (not re-reported)
    out2 = b.update({}, 1.0)
    assert out2["alive"][d] is True and out2["lineage"] == {}


def test_division_ids_do_not_collide():
    b = _make(_cells(), {"mcs_per_update": 0})
    b.update({"divisions": [{"cell_id": 1}]}, 1.0)
    out = b.update({"divisions": [{"cell_id": 1}]}, 1.0)
    ids = list(b._all_ids)
    assert len(ids) == len(set(ids))
    assert all(v == "1" or v in b._cpm_id for v in out["lineage"].values())
    assert set(out["lineage"]).isdisjoint({"1", "2"})


def test_daughter_adhesion_not_aliased():
    b = _make(_cells(), {"mcs_per_update": 0})
    b.update({"adhesion": {"1": {"A": 3.0}}}, 1.0)
    out = b.update({"divisions": [{"cell_id": 1}]}, 1.0)
    (d,) = out["lineage"]
    assert b._cell_state[d]["adhesion"] == b._cell_state["1"]["adhesion"]
    assert b._cell_state[d]["adhesion"] is not b._cell_state["1"]["adhesion"]


def test_motility_warns_unsupported_once():
    b = _make(_cells())
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        b.update({"motility": {"1": 2.0}}, 1.0)
        b.update({"motility": {"1": 3.0}}, 1.0)
    assert sum(issubclass(x.category, RuntimeWarning) for x in w) == 1
    assert b._cell_state["1"]["motility"] == 3.0
