"""Rate contract: one canonical mass-action rate, translated per engine.

The canonical model is a reaction network with mass-action ``kf`` / ``kb`` (see
``ModelSpec.source``). The two engines do not read those numbers the same way:

* Smoldyn (particles, 2D) takes ``kf`` as a bimolecular rate in area/time and ``kb``
  as 1/time, acting on molecules in a polygon. ``to_particle`` passes them as-is.
* The ODE adapter integrates ``dC/dt = kf*A*B - kb*C`` on molecule COUNTS, so the
  bimolecular constant must be divided by an effective volume V: ``kf_ode = kf / V``.
  ``kb`` is unimolecular and unchanged.

V is NOT the polygon area. Smoldyn's finite diffusion, reaction radius and
discretization make the particle cell react less than ideal mixing predicts, so the
ODE needs V smaller than the geometric area for the equilibria to agree. Empirical
calibration (A=B=40..., kf=8, kb=2, difc 1.6/1.6/0.9, smol_dt 0.02, 4 seeds each):
V_eff/area = 0.66 (area 80), 0.55 (36), 0.64 (120), 0.63 (80, n=20). We use
``EFFECTIVE_VOLUME_FRACTION = 0.65``. Valid near the calibrated regime (moderate
conversion, ~0.2-0.7 of reagent bound); expect drift outside it.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from .registry import ModelSpec

EFFECTIVE_VOLUME_FRACTION = 0.65
DEFAULT_DIFC = 1.6
_DEFAULT_DIFC_BY_SPECIES = {"C": 0.9}


def polygon_area(poly) -> float:
    a = np.asarray(poly, float)
    return float(0.5 * abs(np.dot(a[:, 0], np.roll(a[:, 1], -1)) - np.dot(a[:, 1], np.roll(a[:, 0], -1))))


def effective_volume(poly) -> float:
    """ODE effective volume equivalent to the particle cell `poly`."""
    return EFFECTIVE_VOLUME_FRACTION * polygon_area(poly)


def _net(spec: ModelSpec) -> dict[str, Any]:
    if not isinstance(spec.source, dict):
        raise ValueError("rate translation needs a reaction-network dict source")
    return spec.source


def to_particle(spec: ModelSpec) -> dict[str, Any]:
    """Smoldyn `step_cell` spec from the canonical network (single A+B<->C reaction)."""
    net = _net(spec)
    rxns = net.get("reactions", [])
    if len(rxns) != 1 or len(rxns[0]["subs"]) != 2 or len(rxns[0]["prds"]) != 1:
        raise ValueError("particle adapter supports one bimolecular A+B<->C reaction")
    r = rxns[0]
    difc = spec.params.get("difc", {})
    species = {s: float(difc.get(s, _DEFAULT_DIFC_BY_SPECIES.get(s, DEFAULT_DIFC))) for s in net["species"]}
    return {
        "species": species,
        "reaction": {"subs": list(r["subs"]), "prd": r["prds"][0],
                     "kf": float(r["kf"]), "kb": float(r.get("kb", 0.0))},
    }


def to_ode(spec: ModelSpec, *, volume: float) -> ModelSpec:
    """Copy of `spec` whose bimolecular kf is divided by `volume` (count-based ODE)."""
    net = _net(spec)
    rxns = []
    for r in net.get("reactions", []):
        r = dict(r)
        order = len(r["subs"])
        if order == 2:
            r["kf"] = float(r["kf"]) / volume
        if len(r["prds"]) == 2:
            r["kb"] = float(r.get("kb", 0.0)) / volume
        rxns.append(r)
    return replace(spec, source={**net, "reactions": rxns})
