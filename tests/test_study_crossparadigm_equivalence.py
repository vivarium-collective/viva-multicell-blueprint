"""Behavior test for study `crossparadigm-equivalence`.

A fast replicate of the equivalence grid: run the Sego blueprint on both backends
over a few seeds and assert that cross-backend FATE agreement is TOST-equivalent to
the same-backend baseline within the pre-declared margin.
"""
from __future__ import annotations

import copy
import itertools
from pathlib import Path

from viva_multicell_blueprint.blueprint.model import Blueprint
from viva_multicell_blueprint.compile._shared import resolve_initial_state
from viva_multicell_blueprint.compare import metrics
from viva_multicell_blueprint.compare.identity import IdentityRegistry
from viva_multicell_blueprint.compare.runner import run_backend

SEGO = Path(__file__).resolve().parents[1] / "workspace" / "blueprints" / "influenza_sego2022.yaml"
N = 40
SEEDS = (0, 1, 2)


def _run(backend, seed):
    d = copy.deepcopy(Blueprint.load(SEGO).to_dict())
    d.setdefault("mechanics_mapping", {})["rules_seed"] = seed
    return run_backend(Blueprint.from_dict(d), backend, N)


def test_fate_is_tost_equivalent_across_backends():
    ids = [str(c["id"]) for c in resolve_initial_state(Blueprint.load(SEGO))]
    cpm = {s: _run("cpm", s) for s in SEEDS}
    vtx = {s: _run("vertex", s) for s in SEEDS}

    def fate(a, b):
        reg = IdentityRegistry.from_backends(ids, a.identity, b.identity, getattr(b, "residual", {}) or {})
        return metrics.per_cell_agreement(a.frames, b.frames, reg)["mean"]

    same = ([fate(cpm[i], cpm[j]) for i, j in itertools.combinations(SEEDS, 2)]
            + [fate(vtx[i], vtx[j]) for i, j in itertools.combinations(SEEDS, 2)])
    cross = [fate(cpm[i], vtx[j]) for i in SEEDS for j in SEEDS]
    tost = metrics.equivalence_tost(same, cross, margin=0.05)
    assert tost["equivalent"] is True, tost
    assert tost["ci_low"] > -0.05 and tost["ci_high"] < 0.05
