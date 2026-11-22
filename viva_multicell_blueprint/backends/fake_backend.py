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
                "motility": 1.0,
                "adhesion": None,
            }
        self._div_count = {}
        self._lineage = {}

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
            "lineage": dict(self._lineage),
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
        # per-cell runtime actuators (input ports; NOT the init-time config keys).
        for cid, v in (inputs.get("target_size") or {}).items():
            if str(cid) in self._cells:
                self._cells[str(cid)]["size"] = float(v)
        for cid, v in (inputs.get("motility") or {}).items():
            if str(cid) in self._cells:
                self._cells[str(cid)]["motility"] = float(v)
        for cid, v in (inputs.get("adhesion") or {}).items():
            if str(cid) in self._cells:
                self._cells[str(cid)]["adhesion"] = v
        # divisions: mint a daughter beside a live mother; report via lineage.
        self._lineage = {}
        for ev in inputs.get("divisions") or []:
            mid = str(ev["cell_id"] if isinstance(ev, dict) else ev)
            m = self._cells.get(mid)
            if m is None or not m["alive"]:
                continue
            k = self._div_count.get(mid, 0) + 1
            did = f"{mid}.{k}"
            while did in self._cells:
                k += 1
                did = f"{mid}.{k}"
            self._div_count[mid] = k
            half = m["size"] / 2.0
            m["size"] = half
            d = dict(m)
            d["pos"] = [m["pos"][0] + 0.1] + list(m["pos"][1:])
            self._cells[did] = d
            self._lineage[did] = mid
        return self._emit()
