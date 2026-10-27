"""FieldProcess: the single authoritative reaction-diffusion field engine.

Both mechanical backends (CPM, vertex) read this engine's ``field_grid``; no
second diffusion engine exists. Fields live on a fixed Eulerian grid; secretion
and uptake are applied at cell CENTROIDS by cell type; the field gradient is
exposed for chemotaxis via ``field_gradient_at``.

The numerical routines below (``_diffuse_step``, ``_check_stability``) are vendored from viva-cpm and kept faithful.
"""
from __future__ import annotations

import numpy as np
from process_bigraph import Process

# --------------------------------------------------------------------------
# Vendored from viva-cpm viva_cpm_studies/influenza/fields.py @ commit b952eec2 — pure field math, kept verbatim for cross-paradigm fidelity.
#
# Stability rule and the dt/substeps convention are verbatim from that file's
# per-field ``assert dt * diffusion * 2 * 2 < 1.0`` / ``assert substeps * dt ==
# 1.0`` (generalised from ndim=2 to ndim). ``_diffuse_step`` is the numpy transcription of the engine routine
# it drives (crates/cpm-core/src/field.rs ``Field::diffuse_step``, which the
# python file only configures): explicit forward Euler on the face-neighbour
# Laplacian, ``c += dt * (D * sum_j(c_j - c_i) - decay * c_i)``, clamped >= 0,
# no-flux walls (missing neighbours contribute nothing). Secretion is added as
# ``rate * dt`` exactly ONCE per MCS, AFTER the diffusion substeps (engine
# quirk). Boundary: only no-flux is implemented; boundary='periodic' raises
# NotImplementedError (fail loud rather than silently treat as no-flux).
# --------------------------------------------------------------------------


def _check_stability(name: str, d: float, dt: float, substeps: int, ndim: int = 2) -> None:
    if not dt * d * 2 * ndim < 1.0:
        raise ValueError(f"{name} field diffusion dt is unstable")
    if substeps * dt != 1.0:
        raise ValueError(f"{name} field substeps*dt must equal one MCS")


def _diffuse_step(conc: np.ndarray, d: float, decay: float, dt: float) -> np.ndarray:
    """One explicit forward-Euler diffusion+decay sub-step (no-flux walls)."""
    lap = np.zeros(conc.shape, dtype=np.float64)
    c64 = conc.astype(np.float64)
    for ax in range(conc.ndim):
        lo = np.moveaxis(c64, ax, 0)
        l = np.moveaxis(lap, ax, 0)
        l[1:] += lo[:-1] - lo[1:]    # neighbour at i-1
        l[:-1] += lo[1:] - lo[:-1]   # neighbour at i+1
    updated = c64 + dt * (d * lap - decay * c64)
    return np.maximum(updated, 0.0).astype(np.float32)  # concentrations stay non-negative


# --------------------------------------------------------------------------


class FieldProcess(Process):
    """Reaction-diffusion fields on a common grid.

    config:
      grid_shape: [nx, ny(, nz)] sites (1 site = 1 length unit; positions index
        the grid by rounding).
      fields: list of {name, d, decay, secretion:[{type,rate}], uptake:[{type,rate}],
        chemotaxis:{...metadata}, dynamics:{dt, substeps}}.
      sites_per_cell: multiplier on secretion (cell footprint in sites, default 1).

    ``update(inputs, interval)`` runs ``interval`` MCS (integer). Uptake is a
    first-order sink at the centroid site (blueprint extension, not in the
    vendored source): ``c -= min(c, rate * c * dt)``.
    """

    config_schema = {
        "grid_shape": {"_type": "list[integer]", "_default": [50, 50]},
        "fields": {"_type": "list[node]", "_default": []},
        "sites_per_cell": {"_type": "float", "_default": 1.0},
        "boundary": {"_type": "string", "_default": "noflux"},
    }

    def initialize(self, config):
        if config.get("boundary", "noflux") != "noflux":
            raise NotImplementedError("periodic boundary not yet supported")
        self._shape = tuple(int(n) for n in config["grid_shape"])
        self._fields = []
        for f in config.get("fields") or []:
            dyn = f.get("dynamics") or {}
            dt, substeps = float(dyn.get("dt", 1.0)), int(dyn.get("substeps", 1))
            d = float(f.get("d", 0.0))
            _check_stability(f["name"], d, dt, substeps, ndim=len(self._shape))
            self._fields.append({
                "name": f["name"], "d": d, "decay": float(f.get("decay", 0.0)),
                "dt": dt, "substeps": substeps,
                "secretion": {s["type"]: float(s["rate"]) for s in f.get("secretion") or []},
                "uptake": {u["type"]: float(u["rate"]) for u in f.get("uptake") or []},
                "chemotaxis": f.get("chemotaxis") or {},
            })
        self._grids = {f["name"]: np.zeros(self._shape, dtype=np.float32) for f in self._fields}
        self._positions = {}

    def inputs(self):
        return {"positions": "map[list[float]]", "types": "map[string]"}

    def outputs(self):
        return {"field_grid": "map[node]"}

    def _site(self, pos):
        return tuple(int(min(max(round(float(v)), 0), n - 1)) for v, n in zip(pos, self._shape))

    def update(self, inputs, interval=1):
        positions = inputs.get("positions") or {}
        types = inputs.get("types") or {}
        self._positions = {str(k): list(v) for k, v in positions.items()}
        n_mcs = max(int(round(float(interval))), 1)
        scale = float(self.config.get("sites_per_cell", 1.0))
        for f in self._fields:
            g = self._grids[f["name"]]
            for _ in range(n_mcs):
                for _ in range(f["substeps"]):
                    g = _diffuse_step(g, f["d"], f["decay"], f["dt"])
                for cid, pos in positions.items():
                    t = types.get(cid)
                    site = self._site(pos)
                    rate = f["secretion"].get(t, 0.0)
                    if rate:
                        g[site] += np.float32(rate * scale * f["dt"])
                    up = f["uptake"].get(t, 0.0)
                    if up:
                        g[site] -= min(g[site], np.float32(up * g[site] * f["dt"]))
            self._grids[f["name"]] = g
        return {"field_grid": {k: v.copy() for k, v in self._grids.items()}}

    def field_gradient_at(self, cell, name=None):
        """Gradient (central difference, one-sided at walls) at a cell centroid.

        ``cell`` is a cell id (last-seen position) or a position. Returns a
        list per axis for ``name``, or {name: list} for all fields if None.
        """
        pos = self._positions[str(cell)] if isinstance(cell, (str, int)) else cell
        site = self._site(pos)
        out = {}
        for nm, g in self._grids.items():
            grad = []
            for ax, n in enumerate(self._shape):
                lo, hi = list(site), list(site)
                lo[ax], hi[ax] = max(site[ax] - 1, 0), min(site[ax] + 1, n - 1)
                span = hi[ax] - lo[ax]
                grad.append(float(g[tuple(hi)] - g[tuple(lo)]) / span if span else 0.0)
            out[nm] = grad
        return out if name is None else out[name]
