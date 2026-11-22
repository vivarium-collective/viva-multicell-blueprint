"""FakeBackend: trivial in-memory MechanicsBackend for testing shared mechanism.

Cells stay at their seeded centroids; no mechanics. ``fates`` change a cell's
type; ``deaths`` mark cells not alive (divisions are ignored). Neighbors are
live cells within ``backend_params.neighbor_radius`` (default 1.5); shared
surface per type is 1.0 per neighbor of that type -- enough for rules to have
something to read.
"""
from __future__ import annotations

import math

from viva_multicell_blueprint.compile.base import MechanicsBackend, no_footprint

from ._common import event_cid, latent_config, mint_daughter_id, select_activations


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
        latent_map, generic = latent_config(bp)
        for cid, active_type in select_activations(
                inputs.get("activations") or {}, self._cells,
                lambda b: self._cells[b]["type"] if self._cells[b]["alive"] else None,
                latent_map, generic):
            self._cells[cid]["type"] = active_type
        for ev in inputs.get("deaths") or []:
            cid = event_cid(ev)
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
            mid = event_cid(ev)
            m = self._cells.get(mid)
            if m is None or not m["alive"]:
                continue
            did = mint_daughter_id(self._cells, mid, self._div_count)
            half = m["size"] / 2.0
            m["size"] = half
            d = dict(m)
            d["pos"] = [m["pos"][0] + 0.1] + list(m["pos"][1:])
            self._cells[did] = d
            self._lineage[did] = mid
        return self._emit()
