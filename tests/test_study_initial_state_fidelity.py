"""Behavior test for study `initial-state-fidelity` (spec §9.2).

Both backends must reproduce the blueprint's t=0 centroids / sizes / types. We
read each backend's t0 emission (``n_steps=0``) and compare to the resolved
blueprint cells; we also report the vertex 1:1 residual (every cell its own
polygon on a hexagonal sheet, immune cells as overlay islands).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from viva_multicell_blueprint.blueprint.model import Blueprint
from viva_multicell_blueprint.compile._shared import resolve_initial_state
from viva_multicell_blueprint.compare.runner import run_backend

SEGO = Path(__file__).resolve().parents[1] / "workspace" / "blueprints" / "influenza_sego2022.yaml"


def _fidelity(backend):
    bp = Blueprint.load(SEGO)
    by = {str(c["id"]): c for c in resolve_initial_state(bp)}
    run = run_backend(bp, backend, 0)       # t0 frame only
    f = run.frames[0]
    res, type_ok = [], 0
    for cid, c in by.items():
        if not f["alive"].get(cid):
            continue
        p = f["positions"][cid]
        res.append(np.hypot(p[0] - c["centroid"][0], p[1] - c["centroid"][1]))
        type_ok += int(str(f["types"][cid]) == str(c["type"]))
    res = np.array(res)
    return run, len(by), len(res), float(np.sqrt((res ** 2).mean())), type_ok


def test_cpm_reproduces_t0_exactly():
    run, n_total, n_alive, rms, type_ok = _fidelity("cpm")
    assert n_alive == n_total == 48          # every blueprint cell present
    assert type_ok == n_alive                # all types match
    assert rms < 1e-6                        # CPM seeds cells exactly at their centroid


def test_vertex_reproduces_t0_within_residual():
    run, n_total, n_alive, rms, type_ok = _fidelity("vertex")
    assert n_alive == n_total == 48
    assert type_ok == n_alive                # all types match
    # hexagonal sheet re-tiles the square block: centroid residual is sub-cell
    # (cell radius ~2.8 um); overlay islands sit exactly on their centroids
    assert rms < 2.0
    r = run.residual
    assert r["mesh"] == "hex"
    assert r["fraction_clean"] == 1.0 and len(r["unmapped_ids"]) == 0   # 1:1, nothing contested
    assert r["n_clean"] == 48
    # every polygon is exactly one Sego cell: no guard ring / medium polygons
    assert r["n_guard_faces"] == 0
    assert r["n_sheet"] == 36 and r["n_overlay"] == 12   # epithelium + immune overlay


def test_vertex_t0_tissue_is_one_hexagonal_sheet():
    """The epithelium is ONE confluent sheet of hexagons at the target area; the
    immune cells are separate overlay islands (fields are overlaid, not meshed)."""
    from collections import deque

    run, *_ = _fidelity("vertex")
    eptm = run.instance._solver.eptm
    fd, ed = eptm.face_df, eptm.edge_df
    assert len(fd) == 48
    sheet = fd[fd["layer"] == "sheet"]
    assert set(sheet["cell_type"]) <= {"H", "I"}
    assert set(ed[ed["face"].isin(sheet.index)].groupby("face").size()) == {6}
    assert np.allclose(sheet["area"], 25.0)
    # sheet faces form a single connected component through shared vertices
    verts = {f: set(ed.loc[ed["face"] == f, "srce"]) for f in sheet.index}
    seen, todo = set(), deque([sheet.index[0]])
    while todo:
        f = todo.popleft()
        if f in seen:
            continue
        seen.add(f)
        todo.extend(g for g in verts if g not in seen and verts[f] & verts[g])
    assert seen == set(sheet.index)
