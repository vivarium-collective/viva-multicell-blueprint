"""VertexBackend: the tyssue vertex model behind the MechanicsBackend port.

Small fixtures (a 3x3 cluster of centroids). These cover the four review-focus
seams: the hexagonal one-polygon-per-cell mesh, overlay islands, per-pair junction
tension, reconnection, port-shape emission, field-driven migration via the
``behaviors`` port, and the all-one-type degenerate case.
"""
from __future__ import annotations

import numpy as np
import pytest

from viva_multicell_blueprint.compile.base import MechanicsBackend, OUTPUT_PORTS
from viva_multicell_blueprint.core import build_core


def _grid_cells(types=None, spacing=1.0):
    """A clean 3x3 grid of centroids (mapped 1:1 onto a hexagonal sheet)."""
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


def test_vertex_mesh_is_hexagonal_one_polygon_per_cell():
    cells = _grid_cells()
    b = _make(cells)
    rep = b.residual_report
    assert rep["mesh"] == "hex"
    assert rep["n_input"] == rep["n_clean"] == 9
    assert rep["n_collision"] == 0 and rep["fraction_clean"] == 1.0
    # no guard ring / collar / medium polygons: exactly one face per blueprint cell
    assert rep["n_guard_faces"] == 0
    eptm = b._solver.eptm
    assert len(eptm.face_df) == 9
    # identity registry maps every blueprint id to a distinct tyssue face unique_id
    assert len(b.identity) == 9
    assert len(set(b.identity.values())) == 9
    # every cell starts as a regular hexagon at its target area
    sides = eptm.edge_df.groupby("face").size()
    assert set(sides.tolist()) == {6}
    assert np.allclose(eptm.face_df["area"], 1.0)
    # the hex block is centred on the blueprint cloud (sub-cell placement residual)
    assert rep["sheet_centroid_max"] < 1.0


def test_vertex_emits_port_shape():
    cells = _grid_cells()
    b = _make(cells)
    assert isinstance(b, MechanicsBackend)
    assert set(b.inputs()) == {"fates", "field_grid", "divisions", "deaths", "activations"}
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
    # real. The centre cell (id 5, type A) is fully surrounded on the hex sheet, so
    # its shared surface (all types) sums to its own perimeter; its 4 orthogonal
    # blueprint neighbours (ids 2,4,6,8) are all B, so B contact is present.
    assert out["types"]["5"] == "A"
    center_shared = out["shared_surface_by_type"]["5"]
    assert set(center_shared) <= {"A", "B"} and center_shared["B"] > 0
    fd = b._solver.eptm.face_df
    lab = fd.index[fd["unique_id"] == b.identity["5"]][0]
    perim = float(fd.loc[lab, "perimeter"])
    assert sum(center_shared.values()) == pytest.approx(perim, rel=1e-6)
    assert len(out["neighbors"]["5"]) == 6


def test_coincident_centroids_get_separate_polygons():
    # two centroids close enough to have shared ONE Voronoi face before. The hex
    # assignment is 1:1, so they are split onto neighbouring sites: every cell has
    # its own polygon, is alive, and touches its twin.
    cells = [
        {"id": 1, "type": "A", "centroid": [0.0, 0.0], "size": 1.0},
        {"id": 2, "type": "A", "centroid": [0.02, 0.0], "size": 1.0},  # ~coincident w/ 1
        {"id": 3, "type": "A", "centroid": [3.0, 0.0], "size": 1.0},
        {"id": 4, "type": "A", "centroid": [0.0, 3.0], "size": 1.0},
        {"id": 5, "type": "A", "centroid": [3.0, 3.0], "size": 1.0},
    ]
    b = _make(cells)
    rep = b.residual_report
    assert rep["n_collision"] == 0 and rep["unmapped_ids"] == []
    assert len(set(b.identity.values())) == 5
    out = b.update({}, 1.0)
    ids = {str(c["id"]) for c in cells}
    for port in OUTPUT_PORTS:
        assert set(out[port]) == ids
    assert all(out["alive"].values())
    assert "2" in out["neighbors"]["1"]


def test_overlay_cells_are_islands_with_overlap_contact():
    """Overlay (immune-layer) cells are free polygons ON TOP of the sheet: no shared
    vertices, and their contact with the epithelium is the overlap area."""
    cells = _grid_cells(types=["H"] * 9)
    cells.append({"id": 10, "type": "M", "centroid": [1.0, 1.0], "size": 1.0})   # over the centre
    cells.append({"id": 11, "type": "M", "centroid": [9.0, 9.0], "size": 1.0})   # far away
    b = _make(cells, backend_params={"overlay_types": ["M"]})
    rep = b.residual_report
    assert rep["n_sheet"] == 9 and rep["n_overlay"] == 2
    # the island centred exactly on its blueprint centroid
    out = b._emit()
    assert out["positions"]["10"] == pytest.approx([1.0, 1.0])
    # it shares no vertex with the sheet
    ed = b._solver.eptm.edge_df
    fd = b._solver.eptm.face_df
    island = fd.index[fd["unique_id"] == b.identity["10"]][0]
    sheet_faces = [fd.index[fd["unique_id"] == b.identity[str(i)]][0] for i in range(1, 10)]
    island_verts = set(ed.loc[ed["face"] == island, "srce"])
    sheet_verts = set(ed.loc[ed["face"].isin(sheet_faces), "srce"])
    assert island_verts.isdisjoint(sheet_verts)
    # contact = overlap area, credited both ways; total overlap ~ the island's area
    m_contact = out["shared_surface_by_type"]["10"]
    assert set(m_contact) == {"H"}
    assert m_contact["H"] == pytest.approx(1.0, rel=0.05)
    assert out["shared_surface_by_type"]["5"]["M"] > 0
    assert "10" in out["neighbors"]["5"] and "5" in out["neighbors"]["10"]
    # the far island touches nothing
    assert out["shared_surface_by_type"]["11"] == {} and out["neighbors"]["11"] == []


def test_junction_tension_follows_type_pairs():
    """Per-pair adhesion is applied junction by junction (interior = J_ab, border =
    2 x J_(a,medium) per half-edge), not collapsed to one mean tension."""
    cells = _grid_cells()   # alternating A/B
    tensions = {"A|A": 1.0, "A|B": 3.0, "B|B": 1.0, "A|medium": 5.0, "B|medium": 5.0}
    b = _make(cells, backend_params={"tension_by_pair": tensions})
    eptm = b._solver.eptm
    ed, fd = eptm.edge_df, eptm.face_df
    from viva_multicell_blueprint.backends.vertex_coupling import _opposite_positions
    opp = _opposite_positions(ed)
    own = fd["cell_type"].reindex(ed["face"]).to_numpy()
    lt = ed["line_tension"].to_numpy()
    for i in range(len(ed)):
        if opp[i] < 0:
            assert lt[i] == pytest.approx(10.0)
        else:
            want = 3.0 if own[i] != own[opp[i]] else 1.0
            assert lt[i] == pytest.approx(want)


def test_reconnect_enabled_and_scaled_to_cell_size():
    b = _make(_grid_cells())
    s = b._solver
    assert any(fn.__name__ == "reconnect" for fn, _ in s.manager.next)
    # merge threshold is a fraction of the hexagon side, not tyssue's 1e-3 default
    side = b._side
    assert s.eptm.settings["threshold_length"] == pytest.approx(0.1 * side)
    off = _make(_grid_cells(), backend_params={"auto_reconnect": False})
    assert not any(fn.__name__ == "reconnect" for fn, _ in off._solver.manager.next)


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


def test_vertex_activation_changes_latent_to_active():
    """Task 20: the ``activations`` input activates a pre-placed LATENT pool on the
    vertex backend -- ``{M: 3}`` turns exactly 3 ``M_latent`` faces into ``M`` via the
    same differentiation behavior fates use, and the pool is never over-activated."""
    cells = _grid_cells(types=["M_latent"] * 9)
    b = _make(cells)
    out0 = b.update({}, 1.0)
    assert sum(t == "M_latent" for t in out0["types"].values()) == 9
    assert sum(t == "M" for t in out0["types"].values()) == 0

    # activate 3 -> exactly 3 become M (type change), 6 latent remain.
    out1 = b.update({"activations": {"M": 3}}, 1.0)
    assert sum(t == "M" for t in out1["types"].values()) == 3
    assert sum(t == "M_latent" for t in out1["types"].values()) == 6
    # deterministic selection: the first three ids in sorted order activate (matching
    # the CPM test -> the CPM-vs-vertex reproducibility claim is pinned, not assumed).
    assert out1["types"]["1"] == "M" and out1["types"]["2"] == "M" and out1["types"]["3"] == "M"
    assert all(out1["types"][str(i)] == "M_latent" for i in range(4, 10))

    # pool respected: request far more than remain -> clamp to the 6 left.
    out2 = b.update({"activations": {"M": 100}}, 1.0)
    assert sum(t == "M" for t in out2["types"].values()) == 9
    assert sum(t == "M_latent" for t in out2["types"].values()) == 0


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
