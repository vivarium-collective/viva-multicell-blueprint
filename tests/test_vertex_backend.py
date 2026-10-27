"""VertexBackend: the tyssue vertex model behind the MechanicsBackend port.

Small fixtures (a 3x3 cluster of centroids). These cover the four review-focus
seams: mesh identity (interior 1:1 + residual), port-shape emission, field-driven
migration via the ``behaviors`` port, and the all-one-type degenerate case.
"""
from __future__ import annotations

import numpy as np
import pytest

from viva_multicell_blueprint.compile.base import MechanicsBackend, OUTPUT_PORTS
from viva_multicell_blueprint.core import build_core


def _grid_cells(types=None, spacing=1.0):
    """A clean 3x3 grid of centroids (well-separated -> clean Voronoi interior)."""
    xs = [0, 1, 2]
    pts = [(x * spacing, y * spacing) for y in xs for x in xs]
    if types is None:
        types = ["A", "B"] * 5
    return [
        {"id": i + 1, "type": types[i], "centroid": [float(px), float(py)], "size": 1.0}
        for i, (px, py) in enumerate(pts)
    ]


def _make(cells, backend_params=None, domain=None):
    from viva_multicell_blueprint.backends.vertex_backend import VertexBackend

    return VertexBackend(
        {
            "initial_state": cells,
            "backend_params": backend_params or {},
            "domain": domain or {},
        },
        core=build_core(),
    )


def test_vertex_mesh_from_centroids_interior_1to1():
    cells = _grid_cells()
    b = _make(cells)
    rep = b.residual_report
    assert rep["n_input"] == 9
    # a clean, well-separated cluster -> every centroid gets its own interior face
    assert rep["n_clean"] == 9
    assert rep["n_collision"] == 0
    assert rep["fraction_clean"] == 1.0
    # identity registry maps every blueprint id to a distinct tyssue face unique_id
    assert len(b.identity) == 9
    assert len(set(b.identity.values())) == 9
    # guard/collar faces are tagged and excluded, so there ARE extra faces
    assert rep["n_guard_faces"] >= 1


def test_vertex_emits_port_shape():
    cells = _grid_cells()
    b = _make(cells)
    assert isinstance(b, MechanicsBackend)
    assert set(b.inputs()) == {"fates", "field_grid", "divisions", "deaths"}
    assert set(b.outputs()) == set(OUTPUT_PORTS)

    out = b.update({}, 1.0)
    assert set(out) == set(OUTPUT_PORTS)
    ids = {str(c["id"]) for c in cells}
    # every blueprint cell is present in every output map
    for port in OUTPUT_PORTS:
        assert set(out[port]) == ids
    # positions are 2D, sizes are positive face areas
    assert all(len(p) == 2 for p in out["positions"].values())
    assert all(s > 0 for s in out["sizes"].values())
    assert all(out["alive"].values())
    # shared_surface_by_type is map[cell -> map[type -> edge length>0]]; guard excluded
    shared = out["shared_surface_by_type"]
    assert all(isinstance(v, dict) for v in shared.values())
    assert "guard" not in {t for m in shared.values() for t in m}
    assert all(length > 0 for m in shared.values() for length in m.values())
    # neighbors reference only blueprint ids
    for nbrs in out["neighbors"].values():
        assert all(n in ids for n in nbrs)
    # the center cell (id 5) has interior neighbors on all sides
    assert len(out["neighbors"]["5"]) >= 3

    # shared-edge-by-type is load-bearing (feeds the kill rule), so check it for
    # real. The 3x3 grid alternates A/B; the center cell (id 5) is type A and its
    # four orthogonal neighbors (ids 2,4,6,8) are all B. So its shared surface is
    # keyed ONLY by "B" (it touches no A), and that B total is ~its own perimeter
    # (an interior cell is fully surrounded, so every edge is a shared edge).
    assert set(out["types"]["5"]) and out["types"]["5"] == "A"
    center_shared = out["shared_surface_by_type"]["5"]
    assert set(center_shared) == {"B"}  # no "A" entry: it touches no same-type cell
    import math as _m
    # reconstruct the center face perimeter from the solver and compare
    fd = b._solver.eptm.face_df
    lab = fd.index[fd["unique_id"] == b.identity["5"]][0]
    perim = float(fd.loc[lab, "perimeter"])
    assert center_shared["B"] == pytest.approx(perim, rel=1e-6)
    assert center_shared["B"] > 0


def test_contested_centroid_still_emitted_as_dead():
    # two centroids close enough to collide onto one Voronoi face (plus a few
    # spread-out cells so the mesh is well-formed). The loser is dropped from the
    # identity map but MUST still appear in every port, alive=False.
    cells = [
        {"id": 1, "type": "A", "centroid": [0.0, 0.0], "size": 1.0},
        {"id": 2, "type": "A", "centroid": [0.02, 0.0], "size": 1.0},  # collides w/ 1
        {"id": 3, "type": "A", "centroid": [3.0, 0.0], "size": 1.0},
        {"id": 4, "type": "A", "centroid": [0.0, 3.0], "size": 1.0},
        {"id": 5, "type": "A", "centroid": [3.0, 3.0], "size": 1.0},
    ]
    b = _make(cells)
    rep = b.residual_report
    assert rep["n_collision"] >= 1
    assert rep["unmapped_ids"]  # non-empty
    unmapped = set(rep["unmapped_ids"])
    out = b.update({}, 1.0)
    ids = {str(c["id"]) for c in cells}
    # contract holds despite a non-zero residual: every cell in every port
    for port in OUTPUT_PORTS:
        assert set(out[port]) == ids
    for cid in unmapped:
        assert out["alive"][cid] is False
        assert out["neighbors"][cid] == []
        assert out["shared_surface_by_type"][cid] == {}
        assert len(out["positions"][cid]) == 2  # real placeholder, not missing
    # a mapped cell is still alive
    mapped = ids - unmapped
    assert any(out["alive"][c] for c in mapped)


def test_migration_director_moves_cell_up_gradient():
    bp = {
        "chemotaxis": [{"field": "chemo", "strength": 1.0}],
        "migration_scale": 0.4,
        "max_displacement": 0.1,
    }
    # a field that increases in +x: gradient points +x everywhere
    nx, ny = 4, 4
    ramp = np.tile(np.arange(nx, dtype=float)[:, None], (1, ny))
    field_grid = {"chemo": ramp}

    def _run(fg):
        b = _make(_grid_cells(), backend_params=bp)
        p0 = b.update({}, 1.0)["positions"]
        x0 = float(np.mean([p[0] for p in p0.values()]))
        y0 = float(np.mean([p[1] for p in p0.values()]))
        out = None
        for _ in range(20):
            out = b.update(({"field_grid": fg} if fg else {}), 1.0)
        x1 = float(np.mean([p[0] for p in out["positions"].values()]))
        y1 = float(np.mean([p[1] for p in out["positions"].values()]))
        return x1 - x0, y1 - y0

    dx_field, dy_field = _run(field_grid)
    dx_ctrl, dy_ctrl = _run({})  # control: no field -> no active migration

    assert np.isfinite(dx_field) and np.isfinite(dy_field)
    # cells migrate up-gradient (+x) by a clear, non-trivial amount
    assert dx_field > 0.3
    # the +x drift is attributable to ActiveMigration, not relaxation/asymmetry:
    # no field -> near-zero x-drift, and under a pure +x ramp y stays near zero
    assert abs(dx_ctrl) < 0.05
    assert abs(dy_field) < 0.1
    assert dx_field > 5 * abs(dx_ctrl) + 0.2


def test_all_one_type_degenerate():
    cells = _grid_cells(types=["A"] * 9)
    b = _make(cells)
    out = b.update({}, 1.0)
    # every cell is type A
    assert set(out["types"].values()) == {"A"}
    # shared-surface-by-type only ever mentions the single real type
    seen_types = {t for m in out["shared_surface_by_type"].values() for t in m}
    assert seen_types <= {"A"}
    # the center cell has a real, perimeter-sized contact with its A neighbors
    center_shared = out["shared_surface_by_type"]["5"]
    assert set(center_shared) == {"A"}
    fd = b._solver.eptm.face_df
    lab = fd.index[fd["unique_id"] == b.identity["5"]][0]
    perim = float(fd.loc[lab, "perimeter"])
    assert center_shared["A"] == pytest.approx(perim, rel=1e-6)
    assert center_shared["A"] > 0
    assert len(out["positions"]) == 9
