"""Behavior test for study `adhesion-calibration` (spec §9.1).

A mechanics-only sheet (no rules/fields/ODE) runs on BOTH backends; the neutral
adhesion lowers to CPM contact J (direct) and to vertex line tension (scaled).
We sweep the vertex ``tension_scale`` and check both substrates reach comparable
cell-shape statistics (mean area ~ the size-25 target; low dispersion). The
mapping is DIRECTIONAL, not pinned -- the test asserts comparability + records
the best scale, not an exact tension.
"""
from __future__ import annotations

import copy

import numpy as np

from viva_multicell_blueprint.blueprint.model import Blueprint
from viva_multicell_blueprint.compare.runner import run_backend

MECH = {
    "domain": {"extent_um": [30, 30], "grid_resolution_um": 1, "boundary": "noflux"},
    "cell_types": [{"name": "H", "target_size": 25}],
    "adhesion": [{"a": "H", "b": "H", "strength": 5}, {"a": "H", "b": "medium", "strength": 25}],
    "initial_state": {"kind": "blocks",
                      "block": {"type": "H", "origin": [6, 6], "shape": [4, 4], "spacing": 5, "size": 25}},
}


def _area_stats(run):
    f = run.frames[-1]
    a = np.array([s for cid, s in f["sizes"].items() if f["alive"].get(cid)], float)
    return a.mean(), (a.std() / a.mean() if a.mean() else float("nan")), len(a)


def test_adhesion_calibration_matches_shape_stats_across_backends():
    cpm = run_backend(Blueprint.from_dict(copy.deepcopy(MECH)), "cpm", 6)
    c_mean, c_cv, c_n = _area_stats(cpm)
    assert c_n == 16 and 12 < c_mean < 40 and c_cv < 0.3   # CPM sheet holds the size-25 target

    best = None
    swept = {}
    for scale in (0.5, 1.0, 2.0, 4.0):
        d = copy.deepcopy(MECH)
        d["mechanics_mapping"] = {"tension_scale": scale}
        v = run_backend(Blueprint.from_dict(d), "vertex", 6)
        v_mean, v_cv, v_n = _area_stats(v)
        swept[scale] = (v_mean, v_cv)
        assert v_n == 16 and 12 < v_mean < 60 and v_cv < 0.3   # vertex sheet comparable
        diff = abs(v_cv - c_cv)
        if best is None or diff < best[1]:
            best = (scale, diff)
    # comparable shape statistics: mean areas same order of magnitude, CVs both small
    for scale, (v_mean, v_cv) in swept.items():
        assert 0.4 < v_mean / c_mean < 2.5, (scale, v_mean, c_mean)
    # the mapping is only directional at this scale (shape stats weakly depend on scale):
    # the spread of vertex CV across the sweep is small -> calibration is not sharply pinned.
    cvs = [cv for _, cv in swept.values()]
    assert max(cvs) - min(cvs) < 0.1
    assert best is not None   # a best-matching scale exists (recorded in the study)
