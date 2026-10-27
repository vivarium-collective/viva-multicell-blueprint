"""FakeBackend: trivial in-memory MechanicsBackend for testing shared mechanism.

Cells stay at their seeded centroids; no mechanics. ``fates`` change a cell's
type; ``deaths`` mark cells not alive (divisions are ignored). Neighbors are
live cells within ``backend_params.neighbor_radius`` (default 1.5); shared
surface per type is 1.0 per neighbor of that type -- enough for rules to have
something to read.
"""
from __future__ import annotations

import math

from viva_multicell_blueprint.compile.base import MechanicsBackend


class FakeBackend(MechanicsBackend):
    def initialize(self, config):
        self._cells = {}
        for c in config.get("initial_state") or []:
            self._cells[str(c["id"])] = {
                "type": c["type"],
                "pos": [float(v) for v in c["centroid"]],
                "size": float(c.get("size", 1.0)),
                "alive": True,
            }

    def _emit(self):
        radius = float((self.config.get("backend_params") or {}).get("neighbor_radius", 1.5))
        live = {i: c for i, c in self._cells.items() if c["alive"]}
        neighbors, shared = {}, {}
        for i, c in self._cells.items():
            nb = [j for j, d in live.items()
                  if j != i and c["alive"] and math.dist(c["pos"], d["pos"]) <= radius]
            neighbors[i] = nb
            by_type = {}
            for j in nb:
                by_type[live[j]["type"]] = by_type.get(live[j]["type"], 0.0) + 1.0
            shared[i] = by_type
        return {
            "positions": {i: list(c["pos"]) for i, c in self._cells.items()},
            "sizes": {i: c["size"] for i, c in self._cells.items()},
            "types": {i: c["type"] for i, c in self._cells.items()},
            "shared_surface_by_type": shared,
            "neighbors": neighbors,
            "alive": {i: c["alive"] for i, c in self._cells.items()},
        }

    def update(self, inputs, interval):
        for cid, new_type in (inputs.get("fates") or {}).items():
            if str(cid) in self._cells:
                self._cells[str(cid)]["type"] = new_type
        for ev in inputs.get("deaths") or []:
            cid = str(ev["cell_id"] if isinstance(ev, dict) else ev)
            if cid in self._cells:
                self._cells[cid]["alive"] = False
        return self._emit()
