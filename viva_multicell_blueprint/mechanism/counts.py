"""CellCountStep: backend `types` map -> the spatial->ODE cell-type counts (Task 21).

Closes the last wiring gap: SystemicODEProcess consumes scalar counts
H, I, M, K, E, DH (Price-2015 spatialized inputs); this Step produces them each
tick from the backend's per-cell ``types`` map.

Semantics (source: viva-cpm influenza/run.py ``_counts`` + the DH-input fix):
  H, I, M, K, E  count of cells whose type is in the configured type list
  DH             dead-from-HEALTHY: currently-dead cells NOT known to have died
                 from infection. A cell that transitions X -> dead is dead-from-
                 healthy iff X is a healthy type, else dead-from-infected. A cell
                 first seen already dead has unknown origin and counts as DH, as
                 in Sego (``DH = D_total - len(dead_from_infected_ids)``).
The ODE's nearby surrogates (M_nb/K_nb/E_nb) are not added here; the Process holds
them constant (see systemic_ode.NEARBY_SURROGATES).
"""
from __future__ import annotations

from process_bigraph import Step

_ODE_KEYS = ("H", "I", "M", "K", "E", "DH")
_DEFAULT_MAP = {"H": ["H"], "I": ["I"], "M": ["M"], "K": ["NK"], "E": ["CD8"]}


class CellCountStep(Step):
    """Config: ``count_map`` {ODE input -> [cell types]}, ``healthy_types`` (types
    whose death counts as dead-from-healthy; default = ``count_map['H']``),
    ``dead_type`` (default ``"D"``)."""

    config_schema = {
        "count_map": {"_type": "node", "_default": _DEFAULT_MAP},
        "healthy_types": {"_type": "list[string]", "_default": []},
        "dead_type": {"_type": "string", "_default": "D"},
    }

    def initialize(self, config):
        cm = dict(_DEFAULT_MAP)
        cm.update({k: list(v) for k, v in (config.get("count_map") or {}).items()})
        self._map = {k: set(cm[k]) for k in ("H", "I", "M", "K", "E") if k in cm}
        self._healthy = set(config.get("healthy_types") or cm["H"])
        self._dead = config.get("dead_type", "D")
        self._prev: dict = {}
        self._dead_from_infected: set = set()

    def inputs(self):
        return {"types": "map[string]"}

    def outputs(self):
        return {k: "overwrite[float]" for k in _ODE_KEYS}

    def update(self, inputs):
        types = dict(inputs.get("types") or {})
        for cid, t in types.items():
            prev = self._prev.get(cid)
            if t == self._dead:
                if prev is not None and prev != self._dead and prev not in self._healthy:
                    self._dead_from_infected.add(cid)
            else:
                self._dead_from_infected.discard(cid)
        self._dead_from_infected &= set(types)
        out = {k: float(sum(1 for t in types.values() if t in ts)) for k, ts in self._map.items()}
        n_dead = sum(1 for t in types.values() if t == self._dead)
        out["DH"] = float(n_dead - len(self._dead_from_infected))
        self._prev = types
        return out
