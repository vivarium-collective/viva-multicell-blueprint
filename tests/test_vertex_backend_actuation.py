"""VertexBackend honors the per-cell actuation ports and emits real division lineage."""
from __future__ import annotations

import warnings

import numpy as np
import pytest

pytest.importorskip("vivarium_tyssue")

from viva_multicell_blueprint.core import build_core  # noqa: E402


def _cells():
    pts = [(x, y) for y in range(3) for x in range(3)]
    return [{"id": i + 1, "type": "A", "centroid": [float(x), float(y)], "size": 1.0}
            for i, (x, y) in enumerate(pts)]


def _make(backend_params=None):
    from viva_multicell_blueprint.backends.vertex_backend import VertexBackend

    return VertexBackend({"initial_state": _cells(), "backend_params": backend_params or {},
                          "domain": {}}, core=build_core())


def _run(inputs_seq, **kw):
    b = _make(**kw)
    return [b.update(i, 1.0) for i in inputs_seq]


def test_parity_empty_new_ports_unchanged():
    base = _run([{}, {}, {}])
    with_empty = _run([{"target_size": {}, "motility": {}, "adhesion": {}, "divisions": []}] * 3)
    assert base == with_empty
    assert all(o["lineage"] == {} for o in base)


def test_target_size_changes_emitted_size():
    outs = _run([{}] + [{"target_size": {"5": 2.0}}] * 8)
    base = _run([{}] * 9)
    assert outs[-1]["sizes"]["5"] > base[-1]["sizes"]["5"] + 0.2
    assert outs[-1]["sizes"]["5"] > 1.2


def test_unknown_ids_are_noop():
    b = _make()
    out = b.update({"target_size": {"99": 2.0}, "motility": {"99": 2.0},
                    "adhesion": {"99": 1.0}}, 1.0)
    assert len(out["alive"]) == 9


def test_motility_per_cell_takes_effect_without_warning():
    kw = {"backend_params": {"motility": 0.0, "sheet_motility": 0.0, "seed": 3}}
    base = _run([{}] * 6, **kw)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        moved = _run([{"motility": {"5": 3.0}}] * 6, **kw)
    assert moved[-1]["positions"]["5"] != base[-1]["positions"]["5"]


def test_adhesion_per_cell_takes_effect_and_is_not_global():
    kw = {"backend_params": {"line_tension": 0.01}}
    base = _run([{}] * 4, **kw)
    hi = _run([{"adhesion": {"5": 0.5}}] * 4, **kw)
    assert hi[-1]["sizes"]["5"] != base[-1]["sizes"]["5"]
    # a cell with no override keeps the pair-table tension: far corner is closer to base
    assert abs(hi[-1]["sizes"]["5"] - base[-1]["sizes"]["5"]) > \
        abs(hi[-1]["sizes"]["1"] - base[-1]["sizes"]["1"])


def test_adhesion_shared_junction_conflict_warns_once():
    b = _make()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        b.update({"adhesion": {"4": 0.1, "5": 0.5}}, 1.0)
        b.update({"adhesion": {"4": 0.1, "5": 0.5}}, 1.0)
    msgs = [x for x in w if issubclass(x.category, RuntimeWarning)
            and "adhesion" in str(x.message)]
    assert len(msgs) == 1


def test_division_emits_daughter_and_lineage():
    b = _make()
    out0 = b.update({}, 1.0)
    out = b.update({"divisions": [{"cell_id": 5}]}, 1.0)
    assert len(out["lineage"]) == 1
    (d,) = out["lineage"]
    assert out["lineage"][d] == "5"
    assert d not in out0["alive"] and d not in {str(i) for i in range(1, 10)}
    for port in ("positions", "sizes", "types", "alive", "neighbors",
                 "shared_surface_by_type", "footprints"):
        assert d in out[port], port
    assert out["alive"][d] is True and out["types"][d] == "A"
    assert out["sizes"][d] > 0
    out2 = b.update({}, 1.0)
    assert out2["alive"][d] is True and out2["lineage"] == {}
    assert all(out2["alive"][k] for k in ("5", d))


def test_division_ids_do_not_collide_and_chain():
    b = _make()
    b.update({"divisions": [{"cell_id": 5}]}, 1.0)
    out = b.update({"divisions": [{"cell_id": 5}, {"cell_id": 1}]}, 1.0)
    assert len(out["lineage"]) == 2
    assert len(b._all_ids) == len(set(b._all_ids)) == 12
    assert set(out["lineage"]).isdisjoint({str(i) for i in range(1, 10)})
    assert out["lineage"]["5.2"] == "5" and out["lineage"]["1.1"] == "1"
    d1, d2 = sorted(out["lineage"])
    assert d1 != d2 and b._uid[d1] != b._uid[d2]


def test_runtime_target_size_survives_division_and_not_aliased():
    b = _make()
    b.update({"target_size": {"5": 2.0}}, 1.0)
    out = b.update({"divisions": [{"cell_id": 5}], "adhesion": {"5": {"A": 0.2}}}, 1.0)
    (d,) = out["lineage"]
    fd = b._solver.eptm.face_df
    t = {u: fd.loc[b._face_label(u), "prefered_area"] for u in (b._uid["5"], b._uid[d])}
    assert all(np.isclose(v, 2.0) for v in t.values())
    assert b._cell_act[d]["adhesion"] == b._cell_act["5"]["adhesion"]
    assert b._cell_act[d]["adhesion"] is not b._cell_act["5"]["adhesion"]
