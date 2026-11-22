"""MunkBackend honors the per-cell actuation ports and emits real division lineage."""
from __future__ import annotations

import math
import warnings

import pytest

pytest.importorskip("viva_munk")

from viva_multicell_blueprint.core import build_core  # noqa: E402


def _cells():
    return [
        {"id": 1, "type": "A", "centroid": [20.0, 30.0], "size": 12.0},
        {"id": 2, "type": "A", "centroid": [26.0, 30.0], "size": 12.0},
        {"id": 3, "type": "A", "centroid": [45.0, 10.0], "size": 12.0},
    ]


def _make(backend_params=None, adhesion=None):
    from viva_multicell_blueprint.backends.munk_backend import MunkBackend

    bp = {"substeps": 2, "jitter_per_second": 0.0, "mechanics_seed": 1}
    bp.update(backend_params or {})
    return MunkBackend({"initial_state": _cells(), "adhesion": adhesion or [],
                        "domain": {"extent_um": [60.0, 60.0]}, "backend_params": bp},
                       core=build_core())


def _run(inputs_seq, **kw):
    b = _make(**kw)
    return [b.update(i, 1.0) for i in inputs_seq]


def test_parity_empty_new_ports_unchanged():
    base = _run([{}, {}, {}])
    with_empty = _run([{"target_size": {}, "motility": {}, "adhesion": {}, "divisions": []}] * 3)
    assert base == with_empty
    assert all(o["lineage"] == {} for o in base)


def test_target_size_changes_emitted_size_and_footprint():
    outs = _run([{}, {"target_size": {"1": 30.0}}, {}])
    assert outs[0]["sizes"]["1"] == 12.0
    assert math.isclose(outs[-1]["sizes"]["1"], 30.0)
    assert outs[-1]["sizes"]["2"] == 12.0


def test_unknown_ids_are_noop():
    b = _make()
    out = b.update({"target_size": {"99": 2.0}, "motility": {"99": 2.0},
                    "adhesion": {"99": 1.0}}, 1.0)
    assert len(out["alive"]) == 3


def test_motility_per_cell_moves_only_that_cell_without_warning():
    base = _run([{}] * 4)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        moved = _run([{"motility": {"3": 2.0}}] * 4)
    assert moved[-1]["positions"]["3"] != base[-1]["positions"]["3"]
    assert moved[-1]["positions"]["1"] == base[-1]["positions"]["1"]


def test_adhesion_per_cell_pulls_pair_together_without_warning():
    def gap(o):
        return abs(o["positions"]["2"][0] - o["positions"]["1"][0])
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        base = _run([{}] * 5)
        adh = _run([{"adhesion": {"1": 40.0, "2": 40.0}}] * 5)
    assert gap(adh[-1]) < gap(base[-1])
    # a non-adhesive far cell is untouched
    assert adh[-1]["positions"]["3"] == base[-1]["positions"]["3"]


def test_adhesion_dict_form_by_partner_type():
    base = _run([{}] * 5)
    adh = _run([{"adhesion": {"1": {"A": 40.0}, "2": {"A": 40.0}}}] * 5)
    assert adh[-1]["positions"]["1"] != base[-1]["positions"]["1"]


def test_division_emits_daughter_and_lineage():
    b = _make()
    out0 = b.update({}, 1.0)
    out = b.update({"divisions": [{"cell_id": 2}]}, 1.0)
    assert out["lineage"] == {"2.1": "2"}
    assert "2.1" not in out0["alive"]
    for port in ("positions", "sizes", "types", "alive", "neighbors",
                 "shared_surface_by_type", "footprints"):
        assert "2.1" in out[port], port
    assert out["alive"]["2.1"] is True and out["types"]["2.1"] == "A"
    out2 = b.update({}, 1.0)
    assert out2["alive"]["2.1"] is True and out2["lineage"] == {}


def test_division_ids_do_not_collide_and_chain():
    b = _make()
    b.update({"divisions": [{"cell_id": 2}]}, 1.0)
    out = b.update({"divisions": [{"cell_id": 2}, {"cell_id": 1}, {"cell_id": "2.1"}]}, 1.0)
    assert out["lineage"] == {"2.2": "2", "1.1": "1", "2.1.1": "2.1"}
    assert len(b._all_ids) == len(set(b._all_ids)) == 7


def test_runtime_state_survives_division_and_not_aliased():
    b = _make()
    b.update({"target_size": {"2": 20.0}, "adhesion": {"2": {"A": 5.0}}, "motility": {"2": 1.0}}, 1.0)
    out = b.update({"divisions": [{"cell_id": 2}]}, 1.0)
    assert math.isclose(out["sizes"]["2.1"], 20.0)
    assert b._cell_act["2.1"] == b._cell_act["2"]
    assert b._cell_act["2.1"] is not b._cell_act["2"]
    assert b._cell_act["2.1"]["adhesion"] is not b._cell_act["2"]["adhesion"]
