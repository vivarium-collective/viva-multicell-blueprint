"""Behavior test for study `adhesion-calibration` (spec §9.1).

A mechanics-only sheet (no rules/fields/ODE) runs on BOTH backends; the neutral
adhesion lowers to CPM contact J (direct) and to vertex line tension (scaled).
We sweep the vertex ``tension_scale`` and check both substrates reach comparable
cell-shape statistics (mean area ~ the size-25 target; low dispersion), and that
the scale is identifiable: per-pair junction tension contracts cells, so the vertex
mean area falls with the scale and brackets the CPM area.
"""
from __future__ import annotations

import copy

import numpy as np

from viva_multicell_blueprint.blueprint.model import Blueprint
from viva_multicell_blueprint.compile.adhesion import to_vertex_tension
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


SWEEP = (0.5, 2.0, 4.0)   # scale 8 collapses this small sheet (area ~2)
N_STEPS = 60


def test_adhesion_calibration_matches_shape_stats_across_backends():
    cpm = run_backend(Blueprint.from_dict(copy.deepcopy(MECH)), "cpm", N_STEPS)
    c_mean, c_cv, c_n = _area_stats(cpm)
    assert c_n == 16 and 12 < c_mean < 40 and c_cv < 0.3   # CPM sheet holds the size-25 target

    swept = {}
    for scale in SWEEP:
        d = copy.deepcopy(MECH)
        d["mechanics_mapping"] = {"tension_scale": scale}
        v = run_backend(Blueprint.from_dict(d), "vertex", N_STEPS)
        v_mean, v_cv, v_n = _area_stats(v)
        swept[scale] = (v_mean, v_cv)
        assert v_n == 16 and 12 < v_mean < 60 and v_cv < 0.3   # vertex sheet comparable
    # comparable shape statistics: mean areas same order of magnitude on both substrates
    for scale, (v_mean, _) in swept.items():
        assert 0.4 < v_mean / c_mean < 2.5, (scale, v_mean, c_mean)

    # With per-pair junction tension applied, tension contracts cells against their
    # area elasticity: the relaxed mean area falls strictly with the tension scale, so
    # the scale is identifiable from area (and the CPM area is bracketed by the sweep).
    areas = [swept[s][0] for s in SWEEP]
    assert areas[0] > areas[1] > areas[2], areas
    assert areas[-1] < c_mean < areas[0], (areas, c_mean)


def test_j_tension_mapping_is_monotone():
    """The neutral-strength -> vertex line-tension lowering is strictly monotone in BOTH
    the strength and the scale (the J<->tension direction the study records)."""
    adh = MECH["adhesion"]
    # monotone in scale: a larger scale gives a larger tension for every pair
    t1 = to_vertex_tension(adh, scale=1.0)
    t2 = to_vertex_tension(adh, scale=2.0)
    assert set(t1) == set(t2)
    for pair in t1:
        assert t2[pair] > t1[pair]
    # monotone in strength: the stronger neutral pair maps to the larger tension
    hh = to_vertex_tension([{"a": "H", "b": "H", "strength": 5}], scale=1.0)[("H", "H")]
    hm = to_vertex_tension([{"a": "H", "b": "medium", "strength": 25}], scale=1.0)[("H", "medium")]
    assert hm > hh
