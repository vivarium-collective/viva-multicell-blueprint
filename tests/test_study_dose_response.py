"""Behavior test for the STRETCH study `dose-response-crossparadigm` (spec §9, Fig 5/7).

Sweeps the initial viral load (the number of initially-infected epithelial cells,
via ``infected_seed.fraction``) and asks: do BOTH backends agree on the MONOTONE
DIRECTION -- more initial load -> at-least-as-much infection at the end? This is a
directional (sign) agreement check, not a magnitude match.
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
N = 90


def _final_I(run):
    tc = metrics.type_counts(run.frames)
    return int(np.array(tc.get("I", [0]))[-1]) if "I" in tc else 0


def _dose(bp_doc, frac, backend):
    d = copy.deepcopy(bp_doc)
    d["initial_state"]["infected_seed"]["fraction"] = frac
    b = Blueprint.from_dict(d)
    seed_I = sum(c["type"] == "I" for c in resolve_initial_state(b))
    return seed_I, _final_I(run_backend(b, backend, N))


def test_both_backends_agree_on_monotone_dose_direction():
    doc = Blueprint.load(SEGO).to_dict()
    fracs = (0.03, 0.15)                 # low vs high initial viral load
    cpm = [_dose(doc, f, "cpm") for f in fracs]
    vtx = [_dose(doc, f, "vertex") for f in fracs]
    # the higher dose really does seed more infected cells
    assert cpm[0][0] < cpm[1][0]
    # both backends: final infected count is non-decreasing in initial load (same sign)
    assert cpm[0][1] <= cpm[1][1], cpm
    assert vtx[0][1] <= vtx[1][1], vtx
