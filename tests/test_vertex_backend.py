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


def test_migration_director_moves_cell_up_gradient():
    cells = _grid_cells()
    b = _make(
        cells,
        backend_params={
            "chemotaxis": [{"field": "chemo", "strength": 1.0}],
            "migration_scale": 0.4,
            "max_displacement": 0.1,
        },
    )
    # a field that increases in +x: gradient points +x everywhere
    nx, ny = 4, 4
    ramp = np.tile(np.arange(nx, dtype=float)[:, None], (1, ny))
    field_grid = {"chemo": ramp}

    x0 = float(np.mean([p[0] for p in b.update({}, 1.0)["positions"].values()]))
    out = None
    for t in range(20):
        out = b.update({"field_grid": field_grid}, 1.0)
    x1 = float(np.mean([p[0] for p in out["positions"].values()]))
    assert np.isfinite(x1)
    # cells migrate up-gradient (+x) by a clear, non-trivial amount
    assert x1 - x0 > 0.3


def test_all_one_type_degenerate():
    cells = _grid_cells(types=["A"] * 9)
    b = _make(cells)
    out = b.update({}, 1.0)
    # every cell is type A
    assert set(out["types"].values()) == {"A"}
    # shared-surface-by-type only ever mentions the single real type
    seen_types = {t for m in out["shared_surface_by_type"].values() for t in m}
    assert seen_types <= {"A"}
    # the center cell still has real contacts
    assert out["shared_surface_by_type"]["5"] == {"A": pytest.approx(
        sum(out["shared_surface_by_type"]["5"].values()))}
    assert len(out["positions"]) == 9
