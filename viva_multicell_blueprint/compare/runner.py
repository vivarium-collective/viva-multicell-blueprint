"""Side-by-side run harness (spec §9): compile a blueprint onto a backend, run it
for N coupled steps, and collect per-step frames shaped exactly as the
:mod:`compare` metrics and :mod:`viz` figures consume.

A *frame* is the :func:`compare.raster.frame_from_backend` output (geometry +
split ``polygons``/``occupancy`` footprints + ``frame`` meta) with two extras:

* ``field`` -- one field grid (``underlay``, default ``virus``) as a 2-D array,
  for the spatial-playback heat underlay;
* ``field_integrals`` -- ``{field_name: total}`` over every shared field, for the
  dynamics overlay.

Frame 0 is the backend's own t0 emission (pre-step): both backends are identical
at t0 by construction (shared ``initial_state``), so per-cell agreement starts at
1.0 -- a built-in sanity anchor. Frames 1..N are the post-step shared-store
snapshots read straight off the composite.
"""
from __future__ import annotations

import numpy as np
from process_bigraph import Composite

from ..compile import compile as compile_blueprint
from ..compile._shared import grid_shape_from_domain, resolve_initial_state
from ..core import build_core
from .identity import IdentityRegistry
from .raster import frame_from_backend

# Shared geometry stores the mechanics node writes each step.
_GEOM_KEYS = ("positions", "sizes", "types", "alive", "neighbors",
              "shared_surface_by_type", "footprints")

BACKENDS = ("cpm", "vertex")


def _meta() -> dict:
    """Native->um mapping for the Sego patch: 1 grid unit == 1 um (grid_resolution_um
    == 1), so both backends' native coords are already in grid units."""
    return {"scale": 1.0, "origin": [0.0, 0.0]}


def _field_array(field_grid, name):
    g = (field_grid or {}).get(name)
    return np.asarray(g, dtype=float) if g is not None else None


def _integrals(field_grid, field_names) -> dict:
    out = {k: float(np.asarray(v, dtype=float).sum()) for k, v in (field_grid or {}).items()}
    for name in field_names:
        out.setdefault(name, 0.0)   # keep keys stable across frames (incl. t0)
    return out


def _frame(out, field_grid, underlay, field_names, zeros_shape) -> dict:
    fr = frame_from_backend(out, _meta())
    field = _field_array(field_grid, underlay)
    fr["field"] = field if field is not None else np.zeros(zeros_shape)
    fr["field_integrals"] = _integrals(field_grid, field_names)
    return fr


class BackendRun:
    """One backend's run: the per-step ``frames`` plus the identity it exposes."""

    def __init__(self, backend: str, frames: list, instance):
        self.backend = backend
        self.frames = frames
        self.instance = instance
        self.identity = dict(getattr(instance, "identity", {}) or {})
        self.residual = dict(getattr(instance, "residual_report", {}) or {})

    def type_counts(self):
        from .metrics import type_counts
        return type_counts(self.frames)


def run_backend(blueprint, backend: str, n_steps: int, *, underlay: str = "virus",
                interval: float = 1.0) -> BackendRun:
    """Compile ``blueprint`` onto ``backend`` and run ``n_steps`` coupled steps."""
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; expected one of {BACKENDS}")
    doc = compile_blueprint(blueprint, backend=backend)
    comp = Composite({"state": doc["state"]}, core=build_core())
    inst = comp.state["mechanics"]["instance"]
    field_names = [f["name"] for f in (blueprint.fields or [])]
    shape = tuple(grid_shape_from_domain(blueprint.domain)[:2])   # (nx, ny) field layout

    frames = [_frame(inst._emit(), {}, underlay, field_names, shape)]   # t0 (pre-step)
    for _ in range(n_steps):
        comp.run(interval)
        out = {k: (comp.state.get(k) or {}) for k in _GEOM_KEYS}
        frames.append(_frame(out, comp.state.get("field_grid", {}), underlay, field_names, shape))
    return BackendRun(backend, frames, inst)


def run_both(blueprint, n_steps: int, *, underlay: str = "virus", interval: float = 1.0):
    """Run both backends and build the shared :class:`IdentityRegistry`.

    Returns ``(cpm_run, vertex_run, registry)``.
    """
    cpm = run_backend(blueprint, "cpm", n_steps, underlay=underlay, interval=interval)
    vtx = run_backend(blueprint, "vertex", n_steps, underlay=underlay, interval=interval)
    ids = [str(c["id"]) for c in resolve_initial_state(blueprint)]
    reg = IdentityRegistry.from_backends(ids, cpm.identity, vtx.identity, vtx.residual)
    return cpm, vtx, reg
