"""Behavior test for the STRETCH study `dose-response-crossparadigm` (spec §9, Fig 5/7).

Sweeps the initial viral load (the number of initially-infected epithelial cells,
via ``infected_seed.fraction``) across THREE doses and MULTIPLE rules-seeds, and
asks whether BOTH backends agree on the DIRECTION: a higher initial load yields
STRICTLY more infection at the end. This is a directional (sign) check between the
lowest and highest dose -- intermediate-dose monotonicity is seed-dependent and is
NOT claimed.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from viva_multicell_blueprint.blueprint.model import Blueprint
from viva_multicell_blueprint.compile._shared import resolve_initial_state
from viva_multicell_blueprint.compare import metrics
from viva_multicell_blueprint.compare.runner import run_backend

SEGO = Path(__file__).resolve().parents[1] / "workspace" / "blueprints" / "influenza_sego2022.yaml"
N = 120
FRACTIONS = (0.03, 0.08, 0.15)
SEEDS = (0, 1)


def _final_I(run):
    tc = metrics.type_counts(run.frames)
    return int(np.array(tc.get("I", [0]))[-1]) if "I" in tc else 0


def _dose(doc, frac, backend, seed):
    d = copy.deepcopy(doc)
    d["initial_state"]["infected_seed"]["fraction"] = frac
    d.setdefault("mechanics_mapping", {})["rules_seed"] = seed
    b = Blueprint.from_dict(d)
    seed_I = sum(c["type"] == "I" for c in resolve_initial_state(b))
    return seed_I, _final_I(run_backend(b, backend, N))


def test_both_backends_agree_on_strict_dose_direction():
    doc = Blueprint.load(SEGO).to_dict()
    for backend in ("cpm", "vertex"):
        for seed in SEEDS:
            rows = [_dose(doc, f, backend, seed) for f in FRACTIONS]
            seed_I = [r[0] for r in rows]
            final_I = [r[1] for r in rows]
            # the higher dose really does seed more infected cells
            assert seed_I[0] < seed_I[-1], (backend, seed, seed_I)
            # STRICT directional agreement: highest dose ends with MORE infection than
            # the lowest, on every backend and seed (intermediate monotonicity NOT claimed).
            assert final_I[0] < final_I[-1], (backend, seed, final_I)
