"""Behavior test for study `substrate-divergence-counterfactual` (adversarial).

The positive control: in a mechanics-dominated regime the SPATIAL equivalence test
must REJECT (the two backends place cells differently). With the vertex chemotaxis
calibrated to the CPM's saturating drift, the divergence is NOT a directed-migration
difference: neither population travels far up the gradient.
"""
from __future__ import annotations

import copy
import itertools
from pathlib import Path

import numpy as np

from viva_multicell_blueprint.blueprint.model import Blueprint
from viva_multicell_blueprint.compile._shared import resolve_initial_state
from viva_multicell_blueprint.compare import metrics
from viva_multicell_blueprint.compare.identity import IdentityRegistry
from viva_multicell_blueprint.compare.runner import run_backend

BP = Path(__file__).resolve().parents[1] / "workspace" / "blueprints" / "mechanics_divergence.yaml"
N = 50
SEEDS = (0, 1, 2)


def _run(backend, seed):
    d = copy.deepcopy(Blueprint.load(BP).to_dict())
    d.setdefault("mechanics_mapping", {})["rules_seed"] = seed
    return run_backend(Blueprint.from_dict(d), backend, N)


def _mean_A_y(run, ids):
    f = run.frames[-1]
    ys = [f["positions"][i][1] for i in ids if f["types"].get(i) == "A" and f["alive"].get(i)]
    return float(np.mean(ys)) if ys else float("nan")


def _mean_A_y_at(run, ids, k):
    f = run.frames[k]
    return float(np.mean([f["positions"][i][1] for i in ids if f["types"].get(i) == "A"]))


def test_spatial_equivalence_is_rejected_without_directed_migration_gap():
    ids = [str(c["id"]) for c in resolve_initial_state(Blueprint.load(BP))]
    cpm = {s: _run("cpm", s) for s in SEEDS}
    vtx = {s: _run("vertex", s) for s in SEEDS}

    def spatial(a, b):
        reg = IdentityRegistry.from_backends(ids, a.identity, b.identity, getattr(b, "residual", {}) or {})
        return metrics.per_cell_spatial_agreement(a.frames, b.frames, reg, tol_um=5.0)["mean"]

    same = ([spatial(cpm[i], cpm[j]) for i, j in itertools.combinations(SEEDS, 2)]
            + [spatial(vtx[i], vtx[j]) for i, j in itertools.combinations(SEEDS, 2)])
    cross = [spatial(cpm[i], vtx[j]) for i in SEEDS for j in SEEDS]
    tost = metrics.equivalence_tost(same, cross, margin=0.10)
    assert tost["equivalent"] is False, tost          # the test has power
    assert tost["mean_cross"] < tost["mean_same"]
    # calibrated chemotaxis: no substrate carries the A population far up the
    # gradient (the old vertex "climb" came from the uncalibrated linear mapping)
    y0 = _mean_A_y_at(cpm[0], ids, 0)
    for run in (cpm[0], vtx[0]):
        assert abs(_mean_A_y(run, ids) - y0) < 3.0
