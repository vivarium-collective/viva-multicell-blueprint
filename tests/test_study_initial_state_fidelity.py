"""Behavior test for study `initial-state-fidelity` (spec §9.2).

Both backends must reproduce the blueprint's t=0 centroids / sizes / types. We
read each backend's t0 emission (``n_steps=0``) and compare to the resolved
blueprint cells; we also report the vertex 1:1 residual (fraction of cells that
get a clean, uncontested interior Voronoi face).
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
    # Voronoi + guard collar: centroid residual is sub-cell (cell radius ~2.8 um)
    assert rms < 2.0
    r = run.residual
    assert r["fraction_clean"] == 1.0 and len(r["unmapped_ids"]) == 0   # 1:1 residual clean
    assert r["n_clean"] == 48 and r.get("n_guard_faces", 0) > 0
