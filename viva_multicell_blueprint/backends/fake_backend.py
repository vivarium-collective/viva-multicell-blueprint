"""FakeBackend: trivial in-memory MechanicsBackend for testing shared mechanism.

Cells stay at their seeded centroids; no mechanics. ``fates`` change a cell's
type; ``deaths`` mark cells not alive (divisions are ignored). Neighbors are
live cells within ``backend_params.neighbor_radius`` (default 1.5); shared
surface per type is 1.0 per neighbor of that type -- enough for rules to have
something to read.
"""
from __future__ import annotations

import math

from viva_multicell_blueprint.compile.base import (
    GENERIC_LATENT,
    MechanicsBackend,
    latent_pool_types,
    no_footprint,
    sorted_cell_ids,
)


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
            "footprints": {i: no_footprint() for i in self._cells},
        }

    def update(self, inputs, interval):
        for cid, new_type in (inputs.get("fates") or {}).items():
            if str(cid) in self._cells:
                self._cells[str(cid)]["type"] = new_type
        # activations (Task 20): activate N latent cells per active type, deterministic.
        bp = self.config.get("backend_params") or {}
        latent_map = {str(k): str(v) for k, v in (bp.get("latent_types") or {}).items()}
        generic = str(bp.get("latent_generic", GENERIC_LATENT))
        consumed: set = set()
        activations = inputs.get("activations") or {}
        for active_type in sorted(activations):
            count = int(activations.get(active_type, 0) or 0)
            pool = latent_pool_types(active_type, latent_map, generic)
            picked = 0
            for cid in sorted_cell_ids(self._cells):
                if picked >= count:
                    break
                c = self._cells[cid]
                if cid in consumed or not c["alive"] or c["type"] not in pool:
                    continue
                c["type"] = active_type
                consumed.add(cid)
                picked += 1
        for ev in inputs.get("deaths") or []:
            cid = str(ev["cell_id"] if isinstance(ev, dict) else ev)
            if cid in self._cells:
                self._cells[cid]["alive"] = False
        return self._emit()
