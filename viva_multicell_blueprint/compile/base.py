"""The MechanicsBackend contract (spec section 6) -- the single seam between the
shared mechanism (fields, rules, immune, ODE) and the mechanical substrates
(CPM, vertex).

Design choice: a process-bigraph ``Process`` subclass, matching how
``viva_cpm`` / ``vivarium_tyssue`` declare processes (``config_schema``,
``inputs()``, ``outputs()``, ``update()``), so real backends can subclass it and
be wired into a composite unchanged. Cell ids are strings on the ports (map
keys); backends convert from the blueprint's int ids.
"""
from __future__ import annotations

from process_bigraph import Process

INPUT_PORTS = {
    "fates": "map[string]",           # cell_id -> new_type (from RulesProcess)
    "field_grid": "map[node]",        # name -> grid (from FieldProcess)
    "divisions": "list[node]",        # lineage events (from RulesProcess)
    "deaths": "list[node]",           # lineage events (from RulesProcess)
}

# Each port is a per-step SNAPSHOT of the whole cell cohort, so it is wrapped in
# ``overwrite[...]``: emitting the full map atomically REPLACES the shared store.
# A bare ``map[..]`` apply merges by key and DROPS keys absent from the prior
# (empty, at t=0) store, so the mechanics outputs would never reach the shared
# stores the field/rules read (viva-cpm's CPMProcess wraps its map outputs for the
# identical reason). Readers wire plain ``map[..]`` inputs; the two reconcile.
OUTPUT_PORTS = {
    "positions": "overwrite[map[list[float]]]",             # centroid [x,y(,z)]
    "sizes": "overwrite[map[float]]",                       # volume (CPM) / area (vertex)
    "types": "overwrite[map[string]]",
    "shared_surface_by_type": "overwrite[map[map[float]]]",  # cell_id -> {type -> float}
    "neighbors": "overwrite[map[list[string]]]",
    "alive": "overwrite[map[boolean]]",
    # cell_id -> {"kind": "occupancy"|"polygon"|"none", "xy": [[x, y], ...]} in the
    # backend's NATIVE coords. occupancy (CPM): lattice voxel indices the cell owns.
    # polygon (vertex): the face's vertices in boundary order. none/[]: dead or no
    # footprint (rasterize falls back to the area-equivalent disc). Every cell id
    # gets an entry. ``compare.raster.frame_from_backend`` splits this into the
    # ``polygons`` / ``occupancy`` keys ``rasterize`` consumes.
    "footprints": "overwrite[map[node]]",
}


def no_footprint() -> dict:
    """A FRESH empty footprint (never share the xy list between cells)."""
    return {"kind": "none", "xy": []}


class MechanicsBackend(Process):
    """Process contract both CPMBackend and VertexBackend satisfy.

    config: initial_state (list of Cell dicts: id/type/centroid/size),
    cell_types, adhesion (lowered), domain, backend_params.
    Lineage events (``divisions`` / ``deaths``) are dicts of the shape
    ``{"cell_id": <id>}`` (a bare id is also accepted).
    Subclasses implement ``update(inputs, interval)`` returning a dict with
    every key in ``OUTPUT_PORTS``.
    """

    config_schema = {
        "initial_state": {"_type": "list[node]", "_default": []},
        "cell_types": {"_type": "list[node]", "_default": []},
        "adhesion": {"_type": "list[node]", "_default": []},
        "domain": {"_type": "node", "_default": {}},
        "backend_params": {"_type": "node", "_default": {}},
    }

    def inputs(self):
        return dict(INPUT_PORTS)

    def outputs(self):
        return dict(OUTPUT_PORTS)

    def update(self, inputs, interval):
        raise NotImplementedError("MechanicsBackend subclasses must implement update()")
