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

        # -- motility + coupling knobs (the "make it come alive" seam) ----------- #
        # Hard-core collisions alone relax one tiny overlap and then freeze. These
        # knobs add the three mechanisms the blueprint already declares so the
        # center-based cells keep moving and rearranging:
        #   * adhesion   -- a pairwise distance-based pull/push keyed by the blueprint
        #                   adhesion matrix (differential-adhesion-hypothesis form),
        #   * jitter     -- active-Brownian wobble so passive circles explore + resort,
        #   * damping<1  -- momentum persists across steps instead of being bled away.
        # max cell speed (um/s): a hard clamp applied every step so no drive
        # (adhesion, chemotaxis, a stiff collision) can fling a cell across the
        # chamber or diverge -- the stability guarantee for the whole backend.
        self._max_speed = float(bp.get("max_speed", 2.0))
        # adhesion coupling: interaction reach beyond hard contact (um) and the
        # velocity gained per unit of differential adhesivity per step (um/s). A
        # cohesive epithelial pair (adhesivity ~20) then feels ~0.4 um/s of pull at
        # contact -- enough to hold the sheet together against jitter, far below the
        # clamp so it never destabilizes.
        self._adhesion_range = float(bp.get("adhesion_range", 3.0))
        self._adh_scale = float(bp.get("adhesion_velocity_scale", 0.02))
        self._adhesivity = self._parse_adhesivity(config.get("adhesion") or [])
        # chemotaxis gain: converts the blueprint `strength` (a CPM Hamiltonian
        # lambda, e.g. 5000 -- NOT a velocity) into a sane up-gradient drift
        # (um/s per unit strength). With the clamp above, strength 5000 saturates
        # at max_speed instead of teleporting the cell into the wall, while small
        # strengths stay dose-dependent.
        self._chemo_gain = float(bp.get("chemotaxis_gain", 0.08))

        # Deterministic jitter: seed the engine's global RNG from the run's
        # mechanics seed so replicate runs differ in draws, not in determinism.
        seed = bp.get("mechanics_seed")
        if seed is not None:
            import random as _random
            _random.seed(int(seed))

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
                # damping < 1 so injected motion (jitter/adhesion/chemotaxis) and
                # collision momentum persist across steps instead of being fully
                # bled away in one relaxation step (the old 0.1 froze everything).
                "damping_per_second": float(bp.get("damping_per_second", 0.85)),
                # nonzero jitter (active-Brownian wobble) by default so passive
                # circles explore and rearrange; the engine applies ~0.2x of this
                # to circle particles. Tuned to give visible rearrangement of a
                # size-25 cell (~0.5-1 um/step) without approaching the clamp.
                "jitter_per_second": float(bp.get("jitter_per_second", 50.0)),
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

    def _parse_adhesivity(self, adhesion) -> dict:
        """Lower the blueprint adhesion matrix into a per-type-pair *adhesivity*.

        The blueprint ``strength`` is a CPM contact energy ``J`` (``compile.adhesion``:
        higher = LESS adhesive). Two cells prefer to stick together rather than face
        the medium exactly when the Differential-Adhesion-Hypothesis surface energy is
        favorable, so the net adhesivity of a pair is

            A(a,b) = 1/2 (J(a,medium) + J(b,medium)) - J(a,b)

        -- positive when the (a,b) interface costs less than the two medium interfaces
        it replaces (they attract), negative when the medium is preferred (they stay
        apart). This is the center-based analog of CPM cell sorting and uses ONLY the
        declared matrix -- no invented biology. Pairs with no declared energy (e.g. the
        latent immune reservoir carries no adhesion rows) get adhesivity 0 (neutral),
        matching the blueprint's "latent falls back to default contact" intent.
        """
        contact: dict = {}
        medium: dict = {}
        for e in (adhesion or []):
            a, b, s = str(e["a"]), str(e["b"]), float(e["strength"])
            if a == "medium" or b == "medium":
                medium[b if a == "medium" else a] = s
            else:
                contact[tuple(sorted((a, b)))] = s
        out: dict = {}
        for (a, b), j in contact.items():
            if a in medium and b in medium:
                out[(a, b)] = 0.5 * (medium[a] + medium[b]) - j
            else:
                # No medium reference for one side: fall back to the monotone rule
                # (lower contact energy = more adhesive) centered on the mean declared
                # contact energy, so the sign still orders pairs sensibly.
                ref = sum(contact.values()) / max(len(contact), 1)
                out[(a, b)] = ref - j
        return out

    def _adhesivity_for(self, ta: str, tb: str) -> float:
        return self._adhesivity.get(tuple(sorted((str(ta), str(tb)))), 0.0)

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

        # (d2) adhesion: pairwise cell-cell coupling off the blueprint adhesion matrix.
        #      This is what makes cohorts actually INTERACT over time -- cohesive pairs
        #      pull together, medium-preferring pairs ease apart (hard-core collisions
        #      still supply the short-range repulsion).
        self._apply_adhesion()

        # (d3) clamp every cell's speed before integrating: the stability guarantee so
        #      no injected drive or stiff collision can blow positions up.
        self._clamp_speeds()

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

        # keep the carried-forward velocity bounded (collisions can spike it).
        self._clamp_speeds()
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
                # Convert the blueprint strength (a CPM Hamiltonian lambda) into a
                # bounded up-gradient DRIFT speed (um/s), NOT a raw velocity: without
                # this a strength of 5000 was added straight to velocity each step and
                # teleported the cell into the wall. Clamp to max_speed so even a huge
                # lambda saturates at a sane migration speed.
                drift = min(strength * self._chemo_gain, self._max_speed)
                vx, vy = agent["velocity"]
                agent["velocity"] = (vx + drift * gx / norm, vy + drift * gy / norm)

    def _apply_adhesion(self) -> None:
        """Pairwise cell-cell coupling from the blueprint adhesion matrix.

        For every pair of physical bodies within the adhesion reach, add a velocity
        along their center line proportional to the type-pair *adhesivity* (see
        :meth:`_parse_adhesivity`): positive pulls them together, negative eases them
        apart, faded smoothly to zero at the reach edge. The short-range hard-core
        repulsion is left to the engine's collision solver; this term supplies the
        longer-range attraction/sorting the matrix declares, so the cohort's contact
        graph evolves over the run instead of freezing. A no-op when no matrix is
        given (adhesivity empty)."""
        if not self._adhesivity:
            return
        items = []
        for mid, agent in self._agents.items():
            x, y = agent["location"]
            items.append((mid, agent, float(x), float(y), float(agent["radius"]),
                          self._type_of(mid)))
        n = len(items)
        for i in range(n):
            _mi, ai, xi, yi, ri, ti = items[i]
            for j in range(i + 1, n):
                _mj, aj, xj, yj, rj, tj = items[j]
                a = self._adhesivity_for(ti, tj)
                if a == 0.0:
                    continue
                dx, dy = xj - xi, yj - yi
                d = math.hypot(dx, dy)
                if d < 1e-9:
                    continue
                gap = d - (ri + rj)
                if gap > self._adhesion_range:
                    continue
                # window: full strength at/within contact, linearly to 0 at the reach.
                w = 1.0 if gap <= 0.0 else max(0.0, 1.0 - gap / self._adhesion_range)
                mag = self._adh_scale * a * w
                ux, uy = dx / d, dy / d
                vix, viy = ai["velocity"]
                vjx, vjy = aj["velocity"]
                # a>0 attractive: i gains +u (toward j), j gains -u (toward i).
                ai["velocity"] = (vix + mag * ux, viy + mag * uy)
                aj["velocity"] = (vjx - mag * ux, vjy - mag * uy)

    def _clamp_speeds(self) -> None:
        """Clamp every body's speed to ``max_speed`` (um/s). Also scrubs any
        non-finite velocity to zero, so a pathological drive/collision can never
        propagate a NaN into a position -- the backend's stability guarantee."""
        vmax = self._max_speed
        for agent in self._agents.values():
            vx, vy = agent["velocity"]
            if not (math.isfinite(vx) and math.isfinite(vy)):
                agent["velocity"] = (0.0, 0.0)
                continue
            s = math.hypot(vx, vy)
            if s > vmax > 0.0:
                f = vmax / s
                agent["velocity"] = (vx * f, vy * f)

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
