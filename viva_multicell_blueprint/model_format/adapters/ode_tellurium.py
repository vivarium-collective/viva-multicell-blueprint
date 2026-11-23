"""SBML / reaction-network ODE adapter backed by tellurium (libRoadRunner)."""

from __future__ import annotations

from typing import Any

from ..registry import ModelSpec
from .base import IntracellularAdapter

try:  # tellurium is an optional engine
    import tellurium as te

    _AVAILABLE = True
except Exception:  # pragma: no cover - depends on environment
    te = None
    _AVAILABLE = False


def _antimony(net: dict[str, Any]) -> str:
    """Build mass-action antimony text from a reaction-network dict."""
    lines = []
    params = []
    for i, rxn in enumerate(net.get("reactions", [])):
        subs, prds = list(rxn["subs"]), list(rxn["prds"])
        lhs, rhs = " + ".join(subs), " + ".join(prds)
        kf, kb = float(rxn.get("kf", 0.0)), float(rxn.get("kb", 0.0))
        fwd = "*".join([f"kf{i}"] + subs)
        lines.append(f"J{i}f: {lhs} -> {rhs}; {fwd}")
        params.append(f"kf{i} = {kf}")
        if kb:
            rev = "*".join([f"kb{i}"] + prds)
            lines.append(f"J{i}r: {rhs} -> {lhs}; {rev}")
            params.append(f"kb{i} = {kb}")
    species = [f"{s} = {float(v)}" for s, v in net.get("species", {}).items()]
    return "model m\n" + "\n".join(lines + params + species) + "\nend"


class ODEAdapter(IntracellularAdapter):
    def __init__(self, model_spec: ModelSpec, initial_levels: dict[str, float]):
        self.model_spec = model_spec
        self._rr = None
        if not _AVAILABLE:
            self._fallback = dict(initial_levels)
            return
        src = model_spec.source
        if isinstance(src, dict):
            self._rr = te.loada(_antimony(src))
        else:  # SBML string or path
            self._rr = te.loadSBMLModel(src)
        for name, val in initial_levels.items():
            if name in self._rr.getFloatingSpeciesIds():
                self._rr[name] = float(val)

    def step(self, dt: float, inputs: dict[str, Any]) -> None:
        if self._rr is not None:
            self._rr.simulate(0, dt, 2)

    def amounts(self) -> dict[str, float]:
        if self._rr is None:
            return dict(self._fallback)
        return {s: float(self._rr[s]) for s in self._rr.getFloatingSpeciesIds()}

    def crossed(self) -> dict[str, float]:
        return {}

    def available(self) -> bool:
        return _AVAILABLE
