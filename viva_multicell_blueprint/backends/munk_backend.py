"""MunkBackend: the viva-munk center-based model behind the ``MechanicsBackend`` port.

This wraps ``viva_munk``'s :class:`~viva_munk.processes.multibody.PymunkProcess`
(2D rigid-body physics on a ``pymunk`` space) without changing it, as ONE
self-contained, swappable mechanics node -- the *center-based* paradigm alongside
the CPM (voxel) and vertex (polygon) substrates.

Cells are **circles** (the center-based primitive: a cell is a point-center with a
radius), NOT the rod/capsule ``segment`` cells viva-munk calibrates to *E. coli*.
The blueprint's backend-neutral cell list is materialized **exactly 1:1** -- each
blueprint cell becomes one circular body at its centroid with area == ``size`` (so
radius = sqrt(size / pi)) -- and an identity map records
``blueprint_cell_id <-> munk_id`` (here the munk id IS the blueprint id string) so
the comparison keeps a shared identity across backends.

The drive protocol. ``PymunkProcess`` is a process-bigraph Process driven through
its ``circle_particles`` port: each call teleports every body to the passed
``location``/``velocity``, integrates ``substeps`` sub-steps, and returns per-body
*deltas* (new - passed). So this backend holds the AUTHORITATIVE per-cell physics
state (location, velocity, angle), passes it in each step, and applies the returned
delta back -- the standard "caller owns the state" pattern.

The field seam (as in CPMBackend). The authoritative reaction-diffusion field lives
in the shared ``FieldProcess``, NOT here. viva-munk's own field processes
(DiffusionAdvection / CellFieldExchange) are NOT run; each step the shared
``field_grid`` is sampled at each cell center and its local gradient drives a
**chemotactic velocity** (per-type strength, default 0) injected on the body -- so
the center-based cells bias their motion off the authoritative shared field, never
a second field engine. (``thrust``/``motile_speed`` in viva-munk only act on
``segment`` cells, so chemotaxis for circles is applied as an injected velocity,
which the engine honors.)

Per step: apply ``fates`` (type relabel), ``activations`` (latent->active immune
recruitment, exactly as CPM), ``deaths`` (drop the body, mark not-alive),
``divisions`` (spawn a lineage daughter body adjacent to the mother -- no blueprint
id, tracked by lineage, not folded into the t=0 cohort, as on the CPM/vertex side),
then sample the shared field for chemotaxis, step the physics, and read bodies back.
``shared_surface_by_type`` is the pairwise contact between cohort cells: the chord
length of the overlap lens of two circles (a center-based analog of the CPM contact
area / vertex junction length), grouped by the neighbour's type NAME.
"""
from __future__ import annotations

import math

from viva_munk import core_import
from viva_munk.processes.multibody import PymunkProcess

from viva_multicell_blueprint.compile.base import (
    GENERIC_LATENT,
    MechanicsBackend,
    latent_pool_types,
    no_footprint,
    sorted_cell_ids,
)

_FOOTPRINT_SEGMENTS = 16  # polygon points approximating a cell's circular outline


def _circle_outline(cx: float, cy: float, r: float, n: int = _FOOTPRINT_SEGMENTS) -> list:
    """A closed circular outline (boundary-order [x, y] points) for ``rasterize``.

    The contract's ``footprints`` accepts ``kind="polygon"`` with the face's
    vertices in boundary order; a center-based cell's face is its disc, so we
    hand back a regular n-gon inscribed on the circle of radius ``r``.
    """
    return [[cx + r * math.cos(2.0 * math.pi * k / n),
             cy + r * math.sin(2.0 * math.pi * k / n)] for k in range(n)]


class MunkBackend(MechanicsBackend):
    """viva-munk center-based (pymunk) substrate satisfying the ``MechanicsBackend`` contract."""

    def initialize(self, config):
        bp = config.get("backend_params") or {}
        self._substeps = int(bp.get("substeps", 10))

        # Immune-recruitment-as-activation (Task 20): which latent pool each active
        # immune type draws from -- identical semantics to CPMBackend.
        self._latent_map = {str(k): str(v) for k, v in (bp.get("latent_types") or {}).items()}
        self._latent_generic = str(bp.get("latent_generic", GENERIC_LATENT))
        self._warned_activation_types: set = set()

        # touching tolerance (um): two cells whose circles come within this gap are
        # neighbours even without hard overlap (center-based cells seldom interpenetrate).
        self._contact_eps = float(bp.get("contact_eps", 0.75))

        # chemotaxis off the AUTHORITATIVE shared field: field_name -> {type_name: strength}.
        self._chemotaxis = self._parse_chemotaxis(bp)

        cells = config.get("initial_state") or []
        self._all_ids = [str(c["id"]) for c in cells]

        self._meta = {}        # bid -> {"type", "size", "radius"}
        self._alive = {}
        self._cache = {}       # last-known emit for dead cells
        self.identity = {}     # blueprint id (str) -> munk id (str); 1:1 here
        self._agents = {}      # munk_id -> authoritative physics agent dict
        self._daughter_types = {}  # munk_id -> type NAME for non-cohort daughters
        self._div_counter = 0

        # chamber walls sized from the domain extent (fallback: bounding box + margin).
        domain = config.get("domain") or {}
        env_w, env_h = self._env_size(domain, cells)
        self._munk_core = core_import()
        self._pymunk = PymunkProcess(
            {
                "env_size": env_w,
                "env_height": env_h,
                "substeps": self._substeps,
                "damping_per_second": float(bp.get("damping_per_second", 0.1)),
                "jitter_per_second": float(bp.get("jitter_per_second", 0.0)),
                "gravity": 0.0,
                "friction": float(bp.get("friction", 0.8)),
                "elasticity": float(bp.get("elasticity", 0.0)),
                "wall_thickness": float(bp.get("wall_thickness", 5.0)),
            },
            core=self._munk_core,
        )

        for c in cells:
            bid = str(c["id"])
            pos = [float(v) for v in c["centroid"]][:2]
            while len(pos) < 2:
                pos.append(0.0)
            size = float(c.get("size", 1.0))
            r = self._radius(size)
            self._meta[bid] = {"type": str(c["type"]), "size": size, "radius": r}
            self._alive[bid] = True
            self.identity[bid] = bid
            self._agents[bid] = self._circle_agent(pos, r, size)

    # -- construction helpers ------------------------------------------------ #
    @staticmethod
    def _radius(size: float) -> float:
        """Radius of a disc whose AREA equals ``size`` (so emitted sizes round-trip)."""
        return math.sqrt(max(float(size), 1e-9) / math.pi)

    def _circle_agent(self, pos, r, size) -> dict:
        """A fresh authoritative agent dict for the ``circle_particles`` port."""
        return {
            "type": "circle",
            "location": (float(pos[0]), float(pos[1])),
            "velocity": (0.0, 0.0),
            "angle": 0.0,
            "radius": float(r),
            "mass": max(float(size), 1.0),
            "inertia": 0.0,
        }

    def _parse_chemotaxis(self, bp) -> dict:
        """field_name -> {type_name: strength}. Mirrors the CPM ``fields[].chemotaxis``
        shape so one blueprint lowering feeds every backend."""
        out = {}
        for f in (bp.get("fields") or []):
            if not isinstance(f, dict):
                continue
            name = f.get("name")
            if not name:
                continue
            per_type = {}
            for c in (f.get("chemotaxis") or []):
                if isinstance(c, dict) and "type" in c:
                    per_type[str(c["type"])] = float(c.get("strength", c.get("lambda", 0.0)))
            if per_type:
                out[str(name)] = per_type
        return out

    def _env_size(self, domain, cells) -> tuple:
        """Chamber (width, height) in um: the domain extent if given, else a
        bounding box around the seeded cells plus a one-cell margin."""
        extent = domain.get("extent_um")
        if extent:
            w = float(extent[0])
            h = float(extent[1]) if len(extent) > 1 else w
            return max(w, 1.0), max(h, 1.0)
        if not cells:
            return 50.0, 50.0
        xs = [float(c["centroid"][0]) for c in cells]
        ys = [float(c["centroid"][1]) for c in cells]
        margin = 2.0 * max(self._radius(float(c.get("size", 1.0))) for c in cells) + 2.0
        return max(max(xs) + margin, 1.0), max(max(ys) + margin, 1.0)

    # -- per-step ------------------------------------------------------------ #
    def update(self, inputs, interval):
        fates = inputs.get("fates") or {}
        field_grid = inputs.get("field_grid") or {}
        divisions = inputs.get("divisions") or []
        deaths = inputs.get("deaths") or []
        activations = inputs.get("activations") or {}

        # (a) fates: hard type switches (names on the port). Geometry is unchanged
        #     -- a center-based cell keeps its disc; only its label moves.
        for cid, new_type in fates.items():
            bid = str(cid)
            if bid in self._meta and self._alive.get(bid, False):
                self._meta[bid]["type"] = str(new_type)

        # (a2) activations (immune recruitment): activate N currently-LATENT cells of
        #      each requested active type, deterministic id order, clamped to the pool.
        self._apply_activations(activations)

        # (b) deaths: drop the body (it leaves the engine next sync) and mark not-alive.
        for ev in deaths:
            bid = str(ev["cell_id"] if isinstance(ev, dict) else ev)
            if bid in self._agents and self._alive.get(bid, False):
                self._alive[bid] = False
                self._agents.pop(bid, None)

        # (c) divisions: spawn a lineage DAUGHTER body adjacent to the mother. The
        #     daughter has no blueprint id (tracked by lineage, not in the t=0 cohort)
        #     -- it participates in the physics (collisions) but is never emitted.
        for ev in divisions:
            bid = str(ev["cell_id"] if isinstance(ev, dict) else ev)
            self._spawn_daughter(bid)

        # (d) chemotaxis off the AUTHORITATIVE shared field: inject a per-cell velocity
        #     up the local field gradient (per-type strength). No field is re-diffused.
        self._apply_chemotaxis(field_grid)

        # (e) step the physics: pass the authoritative state, apply the returned deltas.
        circle_in = {mid: dict(agent) for mid, agent in self._agents.items()}
        out = self._pymunk.update({"circle_particles": circle_in}, interval)
        deltas = out.get("circle_particles", {}) or {}
        for mid, rec in deltas.items():
            agent = self._agents.get(mid)
            if agent is None:
                continue
            dloc = rec.get("location", (0.0, 0.0))
            dvel = rec.get("velocity", (0.0, 0.0))
            lx, ly = agent["location"]
            vx, vy = agent["velocity"]
            agent["location"] = (lx + dloc[0], ly + dloc[1])
            agent["velocity"] = (vx + dvel[0], vy + dvel[1])
            agent["angle"] = float(agent.get("angle", 0.0)) + float(rec.get("angle", 0.0))

        return self._emit()

    def _spawn_daughter(self, bid: str) -> None:
        """Split a cohort mother: place a same-type, same-size daughter one radius
        away along a deterministic direction. The daughter is a new engine body with
        a non-cohort munk id; collisions then relax the pair apart."""
        if bid not in self._meta or not self._alive.get(bid, False):
            return
        mother = self._agents.get(bid)
        if mother is None:
            return
        meta = self._meta[bid]
        r = float(meta["radius"])
        self._div_counter += 1
        ang = (self._div_counter * 2.399963)  # golden-angle spread so daughters fan out
        mx, my = mother["location"]
        dx, dy = r * math.cos(ang), r * math.sin(ang)
        # nudge the mother one way, the daughter the other, so they do not coincide.
        mother["location"] = (mx - 0.5 * dx, my - 0.5 * dy)
        did = f"{bid}:d{self._div_counter}"
        daughter = self._circle_agent((mx + 0.5 * dx, my + 0.5 * dy), r, meta["size"])
        self._agents[did] = daughter
        self._daughter_types[did] = meta["type"]

    def _apply_activations(self, activations) -> None:
        """Activate latent cells into active immune types (Task 20) -- same contract
        as CPMBackend: for each ``active_type -> count`` pick up to ``count`` alive
        cohort cells whose current type is in that type's latent pool (deterministic
        id order) and relabel them. Clamped to the pool; a cell is consumed once."""
        if not activations:
            return
        import warnings

        declared = {self._meta[b]["type"] for b in self._all_ids}
        declared |= set(self._latent_map)
        consumed: set = set()

        def current_name(bid):
            if not self._alive.get(bid, False):
                return None
            return self._meta.get(bid, {}).get("type")

        for active_type in sorted(activations):
            count = int(activations.get(active_type, 0) or 0)
            if count <= 0:
                continue
            pool = latent_pool_types(active_type, self._latent_map, self._latent_generic)
            # an active type that is nowhere declared/derivable is a config bug: WARN once.
            if active_type not in declared and not any(
                current_name(b) in pool for b in self._all_ids
            ):
                if active_type not in self._warned_activation_types:
                    self._warned_activation_types.add(active_type)
                    warnings.warn(
                        f"MunkBackend: activation requested for unknown active type "
                        f"{active_type!r} (no latent cells in its pool) -- no cells "
                        f"activated.",
                        RuntimeWarning, stacklevel=2,
                    )
                continue
            picked = 0
            for bid in sorted_cell_ids(self._all_ids):
                if picked >= count:
                    break
                if bid in consumed or current_name(bid) not in pool:
                    continue
                self._meta[bid]["type"] = active_type
                consumed.add(bid)
                picked += 1

    def _apply_chemotaxis(self, field_grid) -> None:
        """Inject a per-cell velocity up the local gradient of each shared field,
        scaled by the cell type's strength. Zero strength (the default) is a no-op,
        so without a chemotaxis lowering the center-based cells move only by
        collisions/relaxation -- never by re-diffusing a field here."""
        if not self._chemotaxis or not field_grid:
            return
        for name, per_type in self._chemotaxis.items():
            grid = field_grid.get(name)
            if grid is None:
                continue
            for bid, agent in self._agents.items():
                tname = self._type_of(bid)
                strength = per_type.get(tname, 0.0)
                if not strength:
                    continue
                gx, gy = self._sample_gradient(grid, agent["location"])
                norm = math.hypot(gx, gy)
                if norm < 1e-12:
                    continue
                vx, vy = agent["velocity"]
                agent["velocity"] = (vx + strength * gx / norm, vy + strength * gy / norm)

    @staticmethod
    def _sample_gradient(grid, loc):
        """Central-difference gradient of a 2D field (Fortran [nx, ny] or nested
        list) at integer-rounded cell coords. Returns (dC/dx, dC/dy)."""
        import numpy as np

        arr = np.asarray(grid, dtype=float)
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        if arr.ndim != 2:
            return 0.0, 0.0
        nx, ny = arr.shape
        x = int(round(loc[0]))
        y = int(round(loc[1]))
        x = max(0, min(x, nx - 1))
        y = max(0, min(y, ny - 1))
        xm, xp = max(0, x - 1), min(nx - 1, x + 1)
        ym, yp = max(0, y - 1), min(ny - 1, y + 1)
        dx = (arr[xp, y] - arr[xm, y]) / max(xp - xm, 1)
        dy = (arr[x, yp] - arr[x, ym]) / max(yp - ym, 1)
        return float(dx), float(dy)

    def _type_of(self, mid: str) -> str:
        if mid in self._meta:
            return self._meta[mid]["type"]
        return self._daughter_types.get(mid, "")

    # -- read-back ----------------------------------------------------------- #
    def _emit(self):
        positions, sizes, out_types, alive, neighbors, shared, footprints = (
            {}, {}, {}, {}, {}, {}, {},
        )

        # live cohort geometry (daughters excluded from the emitted cohort, as on
        # the CPM/vertex side) for the pairwise contact scan.
        live = {}
        for bid in self._all_ids:
            if self._alive.get(bid, False) and bid in self._agents:
                a = self._agents[bid]
                live[bid] = (a["location"][0], a["location"][1], float(self._meta[bid]["radius"]))

        for bid in self._all_ids:
            if bid not in live:
                cached = self._cache.get(bid) or {}
                meta = self._meta.get(bid, {})
                positions[bid] = list(cached.get("pos", [0.0, 0.0]))
                sizes[bid] = float(cached.get("size", 0.0))
                out_types[bid] = cached.get("type", meta.get("type", ""))
                alive[bid] = False
                neighbors[bid] = []
                shared[bid] = {}
                footprints[bid] = no_footprint()
                continue

            cx, cy, r = live[bid]
            size = float(self._meta[bid]["size"])
            tname = self._meta[bid]["type"]
            positions[bid] = [cx, cy]
            sizes[bid] = size
            out_types[bid] = tname
            alive[bid] = True
            footprints[bid] = {"kind": "polygon", "xy": _circle_outline(cx, cy, r)}
            self._cache[bid] = {"pos": [cx, cy], "size": size, "type": tname}

            nbrs, by_type = [], {}
            for other, (ox, oy, orr) in live.items():
                if other == bid:
                    continue
                contact = self._contact_length(cx, cy, r, ox, oy, orr)
                if contact is None:
                    continue
                nbrs.append(other)
                otype = self._meta[other]["type"]
                by_type[otype] = by_type.get(otype, 0.0) + contact
            neighbors[bid] = nbrs
            shared[bid] = by_type

        return {
            "positions": positions,
            "sizes": sizes,
            "types": out_types,
            "shared_surface_by_type": shared,
            "neighbors": neighbors,
            "alive": alive,
            "footprints": footprints,
        }

    def _contact_length(self, cx, cy, r, ox, oy, orr):
        """Center-based contact measure between two discs, or ``None`` if they are
        farther apart than the touching tolerance. When the discs overlap, this is
        the chord length of the overlap lens (the shared interface); when they are
        merely within ``contact_eps`` of touching, a small gap-scaled proxy so a
        just-touching pair still registers a contact (analogous to CPM sharing a
        face / a vertex junction of nonzero length)."""
        d = math.hypot(cx - ox, cy - oy)
        if d > r + orr + self._contact_eps:
            return None
        if d <= 1e-9:
            return 2.0 * min(r, orr)  # concentric: full overlap
        if d < r + orr:
            # chord of the circle-circle intersection lens
            a = (d * d - orr * orr + r * r) / (2.0 * d)
            h2 = r * r - a * a
            if h2 <= 0.0:
                return self._contact_eps
            return 2.0 * math.sqrt(h2)
        # within eps but not overlapping: a small contact that fades with the gap.
        gap = d - (r + orr)
        return max(self._contact_eps - gap, 1e-3)
