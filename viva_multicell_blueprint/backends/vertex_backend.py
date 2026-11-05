"""VertexBackend: the tyssue vertex model behind the ``MechanicsBackend`` port.

This wraps ``vivarium_tyssue``'s :class:`EulerSolver` (spec section 5, the vertex
substrate) without changing it, as ONE self-contained, swappable mechanics node.

Tissue (see :mod:`.vertex_mesh`): every polygon is exactly one blueprint cell.

* The confluent **epithelial sheet** is a regular hexagonal tiling (cell area =
  blueprint size); each blueprint cell gets its own lattice site by an optimal
  1:1 assignment, so no polygon ever stands for two cells and no guard ring /
  collar / medium polygon exists.
* **Overlay** cell types (``backend_params.overlay_types`` -- the immune layer,
  Sego's z=1 over the z=0 epithelium) are free hexagonal islands at their exact
  blueprint centroid; their contact with the sheet is the polygon overlap area.
* The shared **fields are overlaid** on the tissue (sampled per face into
  ``field_<name>`` columns), never meshed.

Mechanics (the vertex energy; full definition in ``docs/mechanics.md``)::

    E = sum_faces K_t/2 (A - A0)^2  +  sum_junctions Lambda_ab * l  (+ Gamma_t/2 (P - P0)^2)
    eta dr/dt = -grad E + F_migration,   Lambda_ab(t) = kappa * J_ab + OU noise

with per-type ``K_t`` / ``Gamma_t`` from the blueprint ``cell_types`` (``2 *
lambda_volume`` / ``2 * lambda_surface`` -- the CPM's own per-type knobs, energy-
matched), per-type-PAIR ``Lambda_ab`` from the lowered adhesion table, T1/T3
neighbour exchange via ``auto_reconnect``, and active motility (a persistent random
walk, plus an optional Ornstein-Uhlenbeck junction-tension fluctuation) as the
counterpart of the CPM temperature. Each coupled step is integrated in ``substeps``
solver updates (no vertex outruns the T1 threshold), and the shared field domain is a
hard wall. The calibrated sheet motility is capped so the mesh never folds
(docs/mechanics.md section 4).

Coupling (see :mod:`.vertex_coupling`): fields, fates, activations and lineage
events are turned into vivarium-tyssue **behavior nodes** (``apply_gradient``,
``differentiation``, ``split_cells``, plus the ``junction_tension`` /
``overlay_fields`` behaviors registered here) and passed to the solver's
``behaviors`` port -- the same path the Gillespie regulator uses -- so the solver's
EventManager applies them to the live epithelium before each integration step.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from viva_multicell_blueprint.compile.base import (
    GENERIC_LATENT,
    MechanicsBackend,
    latent_pool_types,
    no_footprint,
    sorted_cell_ids,
)

from . import vertex_coupling as coupling
from .vertex_mesh import (
    assign_to_lattice,
    hex_lattice_sites,
    hex_polygon,
    hex_side,
    overlap_area,
    sheet_from_polygons,
)

MEDIUM = "medium"
SHEET, OVERLAY = "sheet", "overlay"
_MIGRATION_COLS = coupling.MIGRATION_COLS


class VertexBackend(MechanicsBackend):
    """tyssue vertex substrate satisfying the ``MechanicsBackend`` contract."""

    def initialize(self, config):
        bp = config.get("backend_params") or {}
        self._geom = str(bp.get("geom", "SheetGeometry"))
        self._factory = str(bp.get("factory", "model_factory"))
        self._area_elasticity = float(bp.get("area_elasticity", 2.0))
        self._viscosity = bp.get("viscosity")
        self._migration_scale = float(bp.get("migration_scale", 0.4))
        self._chemotaxis_saturation = float(bp.get("chemotaxis_saturation", 0.0))
        self._chemotaxis = list(bp.get("chemotaxis") or [])
        self._overlay_types = {str(t) for t in (bp.get("overlay_types") or [])}
        # per-pair junction tension (the lowered blueprint adhesion); a pair not
        # in the table falls back to the scalar line_tension.
        self._tensions = {str(k): float(v) for k, v in (bp.get("tension_by_pair") or {}).items()}
        self._default_tension = float(bp.get("line_tension", 0.01))
        # junction-tension fluctuation (the CPM-temperature counterpart)
        self._sigma = float(bp.get("tension_noise", 0.0))
        self._tau = float(bp.get("tension_noise_tau", 1.0))
        self._reconnect = bool(bp.get("auto_reconnect", True))
        # persistent random walk (active-Brownian polarity) -- the counterpart of
        # a CPM cell wandering under Metropolis fluctuations; force per layer.
        self._motility_force = {OVERLAY: float(bp.get("motility", 0.0)),
                                SHEET: float(bp.get("sheet_motility", 0.0))}
        self._seed = int(bp.get("seed", 0))
        self._rng = np.random.default_rng(self._seed)
        # one polarity stream per layer, so the overlay's random walk does not
        # depend on how many sheet cells draw before it (and vice versa)
        persistence = {OVERLAY: float(bp.get("motility_persistence", 10.0)),
                       SHEET: float(bp.get("sheet_motility_persistence",
                                           bp.get("motility_persistence", 10.0)))}
        self._motility = {layer: coupling.RandomMotility(persistence[layer], seed=self._seed + k)
                          for k, layer in ((1, SHEET), (2, OVERLAY))}
        self._threshold_length = bp.get("threshold_length")
        self._substeps = max(1, int(bp.get("substeps", 4)))
        # rosette detachment rates (per unit time) for the 4-way / 5+-way vertices a
        # merge leaves behind; unset -> tyssue's defaults (p_4 = 0.1, p_5p = 0.01)
        self._detach_rates = {k: float(bp[k]) for k in ("p_4", "p_5p") if bp.get(k) is not None}
        # the shared field domain is a hard wall, as the CPM lattice edge is
        domain = config.get("domain") or {}
        extent, res = domain.get("extent_um"), domain.get("grid_resolution_um")
        self._bounds = ([0.0, 0.0], [float(e) / float(res) for e in extent[:2]]) \
            if extent and res else None
        self._max_disp = bp.get("max_displacement")
        self._type_mech = self._per_type_mechanics(config, bp)
        # Shape stiffness 1/2 Gamma (P - P0)^2 with P0 = shape_index * sqrt(A0) for
        # cells whose type sets no perimeter term. The CPM has no counterpart
        # (lambda_S = 0); on the vertex side it is the regularization that keeps
        # actively-sheared polygons simple (non-self-intersecting).
        self._shape_gamma = float(bp.get("perimeter_elasticity", 0.0))
        self._shape_index = float(bp.get("shape_index", 3.9))
        # Immune-recruitment-as-activation (Task 20): latent pool resolution.
        self._latent_map = {str(k): str(v) for k, v in (bp.get("latent_types") or {}).items()}
        self._latent_generic = str(bp.get("latent_generic", GENERIC_LATENT))

        self._t = 0.0
        self._alive: dict[str, bool] = {}
        self._cache: dict[str, dict] = {}  # last-known emit per blueprint id
        self._build(config.get("initial_state") or [])

    # -- construction -------------------------------------------------------- #
    def _per_type_mechanics(self, config, bp) -> dict:
        """Per-type face moduli, energy-matched to the CPM's per-type knobs:
        ``K = 2 * lambda_volume`` and ``Gamma = 2 * lambda_surface`` (CPM charges
        ``lambda (V - V_T)^2``, tyssue ``K/2 (A - A0)^2``). An explicit
        ``mechanics.area_elasticity`` / ``perimeter_elasticity`` /
        ``prefered_perimeter`` on the cell type wins."""
        out = {}
        for ct in (config.get("cell_types") or []):
            if not isinstance(ct, dict):
                continue
            mech = dict(ct.get("mechanics") or {})
            m = {"area_elasticity": self._area_elasticity,
                 "perimeter_elasticity": 0.0, "prefered_perimeter": 0.0}
            lam_v = mech.get("lambda_volume", ct.get("lambda_volume"))
            if lam_v is not None:
                m["area_elasticity"] = 2.0 * float(lam_v)
            lam_s = mech.get("lambda_surface", ct.get("lambda_surface"))
            if lam_s:
                m["perimeter_elasticity"] = 2.0 * float(lam_s)
                m["prefered_perimeter"] = float(mech.get("target_surface", ct.get("target_surface", 0.0)))
            for k in m:
                if k in mech:
                    m[k] = float(mech[k])
            out[str(ct["name"])] = m
        return out

    def _build(self, cells):
        from tyssue import SheetGeometry
        from tyssue.io.hdf5 import save_datasets

        ids = [str(c["id"]) for c in cells]
        types = [str(c["type"]) for c in cells]
        sizes = [float(c.get("size", 1.0)) for c in cells]
        cents = np.array([[float(c["centroid"][0]), float(c["centroid"][1])] for c in cells],
                         dtype=float).reshape(-1, 2)
        layer = [OVERLAY if t in self._overlay_types else SHEET for t in types]

        sheet_ix = [i for i, l in enumerate(layer) if l == SHEET]
        over_ix = [i for i, l in enumerate(layer) if l == OVERLAY]
        polys, owner, placed = [], [], {}
        if sheet_ix:
            area = float(np.median([sizes[i] for i in sheet_ix]))
            sites = hex_lattice_sites(cents[sheet_ix], area)
            # alternate 1:1 assignment and rigid recentring of the (still regular)
            # lattice on the blueprint cloud until the assignment stops changing
            site_of = None
            for _ in range(10):
                new = assign_to_lattice(cents[sheet_ix], sites)
                sites = sites + (cents[sheet_ix].mean(0) - sites[new].mean(0))
                if site_of is not None and np.array_equal(new, site_of):
                    break
                site_of = new
            for k, i in enumerate(sheet_ix):
                center = sites[site_of[k]]
                polys.append(hex_polygon(center, area))
                owner.append(i)
                placed[i] = center
        for i in over_ix:
            polys.append(hex_polygon(cents[i], sizes[i]))
            owner.append(i)
            placed[i] = cents[i]

        side = hex_side(float(np.median(sizes))) if sizes else 1.0
        self._side = side
        sheet = sheet_from_polygons(polys)
        SheetGeometry.update_all(sheet)

        fd = sheet.face_df
        fd["cell_type"] = [types[i] for i in owner]
        fd["layer"] = [layer[i] for i in owner]
        fd["prefered_area"] = [sizes[i] for i in owner]
        for col in ("area_elasticity", "perimeter_elasticity", "prefered_perimeter"):
            fd[col] = [self._mech(types[i])[col] for i in owner]
        if self._shape_gamma > 0:
            unset = fd["perimeter_elasticity"].to_numpy() <= 0
            p0 = self._shape_index * np.sqrt(np.asarray([sizes[i] for i in owner], dtype=float))
            fd.loc[unset, "perimeter_elasticity"] = self._shape_gamma
            fd.loc[unset, "prefered_perimeter"] = p0[unset]
        for col, val in zip(_MIGRATION_COLS, (1.0, 0.0, 0.0, 0.0)):
            fd[col] = val
        # Explicit Euler (dt = one step) is stable only if the vertex drag beats
        # the area stiffness ~K*A0; eta = 2*K*A0 keeps a step well inside it.
        if self._viscosity is None:
            k_max = max([self._mech(t)["area_elasticity"] for t in types] or [self._area_elasticity])
            self._viscosity = 2.0 * k_max * (float(np.median(sizes)) if sizes else 1.0)
        sheet.vert_df["viscosity"] = float(self._viscosity)

        self._all_ids = list(ids)
        self._cell_meta = {ids[i]: {"pos": [float(cents[i, 0]), float(cents[i, 1])],
                                    "size": sizes[i], "type": types[i]} for i in range(len(ids))}
        uids = fd["unique_id"].to_numpy() if "unique_id" in fd else fd.index.to_numpy()
        self._uid = {ids[i]: int(uids[f]) for f, i in enumerate(owner)}
        self._uid_to_bid = {u: b for b, u in self._uid.items()}
        self._layer_of = {ids[i]: layer[i] for i in range(len(ids))}
        for b in ids:
            self._alive[b] = True
        self.identity = dict(self._uid)

        disp = np.array([np.hypot(*(placed[i] - cents[i])) for i in sheet_ix]) if sheet_ix else np.zeros(0)
        self.residual_report = {
            "mesh": "hex",
            "n_input": len(ids),
            "n_clean": len(ids),            # 1:1 by construction: one polygon per cell
            "n_collision": 0,
            "fraction_clean": 1.0 if ids else 0.0,
            "contested_faces": [],
            "unmapped_ids": [],
            "n_guard_faces": 0,
            "n_sheet": len(sheet_ix),
            "n_overlay": len(over_ix),
            "sheet_centroid_rms": float(np.sqrt(np.mean(disp ** 2))) if len(disp) else 0.0,
            "sheet_centroid_max": float(disp.max()) if len(disp) else 0.0,
        }

        # Held on the instance so its finalizer removes the scratch mesh dir when
        # the backend is collected / at interpreter exit (no orphaned temp dirs).
        self._tmpdir_obj = tempfile.TemporaryDirectory(prefix="vertex_backend_")
        hf5 = str(Path(self._tmpdir_obj.name) / "mesh.hf5")
        save_datasets(hf5, sheet)
        self._solver = self._make_solver(hf5)
        # Junction tensions are owned by the junction_tension behavior from step 1;
        # seed them now so the t0 mesh already carries its type-pair tensions.
        coupling.junction_tension(self._solver.eptm, None, tensions=self._tensions,
                                  default_tension=self._default_tension)

    def _mech(self, ctype: str) -> dict:
        return self._type_mech.get(ctype) or {
            "area_elasticity": self._area_elasticity,
            "perimeter_elasticity": 0.0, "prefered_perimeter": 0.0}

    def _make_solver(self, hf5):
        from process_bigraph import allocate_core
        from vivarium_tyssue.data_types import register_types
        from vivarium_tyssue.maps import BEHAVIOR_MAP
        from vivarium_tyssue.processes import register_processes
        from vivarium_tyssue.processes.eulersolver import EulerSolver

        effectors = ["FaceAreaElasticity", "LineTension", "ActiveMigration"]
        if self._shape_gamma > 0 or any(m["perimeter_elasticity"] > 0 for m in self._type_mech.values()):
            effectors.insert(1, "PerimeterElasticity")
        threshold = (float(self._threshold_length) if self._threshold_length is not None
                     else 0.1 * self._side)
        max_disp = float(self._max_disp) if self._max_disp is not None else 0.5 * self._side

        # tyssue's reconnect (merge/detach) draws from the global numpy RNG.
        np.random.seed(self._seed)
        core = allocate_core()
        core = register_types(core)
        core = register_processes(core)
        cfg = {
            "name": "vertex",
            "eptm": hf5,
            "tissue_type": "Sheet",
            "geom": self._geom,
            "effectors": effectors,
            "ref_effector": "FaceAreaElasticity",
            "factory": self._factory,
            # per-type face moduli + tensions are already on the mesh; no global
            # parameter overrides.
            "parameters": {},
            "settings": {"threshold_length": threshold},
            "maps": {},
            "backend": "python",
            # T1/T3 neighbour exchange: short junctions (< threshold_length) merge
            # into a 4-way vertex that detaches along the other axis.
            "auto_reconnect": {"enabled": self._reconnect, **self._detach_rates},
            "bounds": {},
            "output_columns": {},
            "substeps": 1,
            # explicit-Euler safety clamp: bounds a strong transient so it relaxes
            # over steps instead of exploding to NaN in one.
            "max_displacement": max_disp,
        }
        solver = EulerSolver(cfg, core=core)
        # This backend's behaviors, on a PRIVATE copy of the maps (EulerSolver's
        # maps dict is module-global; never mutate it for other solvers).
        solver.maps = dict(solver.maps)
        solver.maps["BEHAVIOR_MAP"] = {**BEHAVIOR_MAP, **coupling.EXTRA_BEHAVIORS}
        return solver

    # -- per-step ------------------------------------------------------------ #
    def _face_label(self, uid):
        """Current face-index label for a stable unique_id (face_df reindexes on
        topology changes), or None if the face is gone (extruded / T2)."""
        fd = self._solver.eptm.face_df
        m = fd.index[fd["unique_id"] == uid]
        return m[0] if len(m) else None

    def _activation_behaviors(self, activations) -> list:
        """Activate latent cells into active immune types (Task 20). For each
        ``active_type -> count``, pick up to ``count`` currently-alive cells whose
        current face ``cell_type`` is in that type's latent pool (deterministic id
        order) and emit a ``differentiation`` behavior to the active type. Clamped to
        the pool; each latent cell is consumed at most once per step."""
        if not activations:
            return []
        fd = self._solver.eptm.face_df
        out = []
        consumed: set = set()

        def current_name(bid):
            uid = self._uid.get(bid)
            if uid is None or not self._alive.get(bid, False):
                return None
            lab = self._face_label(uid)
            if lab is None:
                return None
            return str(fd.loc[lab, "cell_type"])

        for active_type in sorted(activations):
            count = int(activations.get(active_type, 0) or 0)
            if count <= 0:
                continue
            pool = latent_pool_types(active_type, self._latent_map, self._latent_generic)
            picked = 0
            for bid in sorted_cell_ids(self._all_ids):
                if picked >= count:
                    break
                if bid in consumed or current_name(bid) not in pool:
                    continue
                out.append({
                    "func": "differentiation",
                    "cell_uid": self._uid[bid],
                    "new_type": str(active_type),
                })
                consumed.add(bid)
                picked += 1
        return out

    def update(self, inputs, interval):
        fates = inputs.get("fates") or {}
        field_grid = inputs.get("field_grid") or {}
        divisions = inputs.get("divisions") or []
        deaths = inputs.get("deaths") or []
        activations = inputs.get("activations") or {}
        fd = self._solver.eptm.face_df

        # deaths first, so a cell dying this step stops migrating this step
        dead_uids = []
        for ev in deaths:
            cid = str(ev["cell_id"] if isinstance(ev, dict) else ev)
            uid = self._uid.get(cid)
            if uid is not None:
                self._alive[cid] = False
                dead_uids.append(uid)
        live = [u for b, u in self._uid.items() if self._alive.get(b, True)]

        behaviors = []
        behaviors += coupling.fate_behaviors(fates, self._uid)
        behaviors += self._activation_behaviors(activations)
        behaviors += coupling.field_overlay_behaviors(fd, field_grid)
        random_force = {}
        for layer, walker in self._motility.items():
            force_of = {u: self._motility_force[layer] for u in live
                        if self._layer_of[self._uid_to_bid[u]] == layer}
            random_force.update(walker.step(force_of, float(interval)))
        behaviors += coupling.migration_behaviors(
            fd, field_grid, self._chemotaxis, self._migration_scale, live,
            random_force=random_force,
            saturation=self._chemotaxis_saturation)
        behaviors += coupling.death_behaviors(dead_uids)
        div_uids = []
        for ev in divisions:
            cid = str(ev["cell_id"] if isinstance(ev, dict) else ev)
            if self._uid.get(cid) is not None:
                div_uids.append(self._uid[cid])
        behaviors += coupling.division_behaviors(div_uids, self._geom)

        # Integrate the coupled step in `substeps` solver updates so no vertex moves
        # more than a fraction of the T1 threshold per update -- otherwise a short
        # junction can invert (fold the mesh) before reconnect gets to merge it.
        # The coupling behaviors go in with the first update; junction tension (the
        # topology can change in every update) and domain confinement go in with
        # each one. Forces and directors persist on the mesh between updates.
        dt = float(interval) / self._substeps
        for k in range(self._substeps):
            step = list(behaviors) if k == 0 else []
            step += coupling.junction_tension_behavior(
                self._tensions, self._default_tension, self._sigma, self._tau,
                dt, int(self._rng.integers(2 ** 31 - 1)))
            step += coupling.confinement_behavior(self._bounds)
            self._solver.update({"behaviors": step, "global_time": float(self._t)}, dt)
            self._t += dt
        return self._emit()

    # -- read-back ----------------------------------------------------------- #
    def _emit(self):
        eptm = self._solver.eptm
        fd = eptm.face_df
        ed = eptm.edge_df
        label_by_uid = {int(u): lab for lab, u in fd["unique_id"].items()}
        opp = coupling._opposite_positions(ed)
        ed_face = ed["face"].to_numpy()
        ed_len = ed["length"].to_numpy(dtype=float)

        positions, sizes, types, alive, neighbors, shared = {}, {}, {}, {}, {}, {}
        footprints = {}
        live_labels = [label_by_uid[u] for u in self._uid.values() if u in label_by_uid]
        polys = self._face_polygons(live_labels)

        for bid in self._all_ids:
            uid = self._uid.get(bid)
            lab = label_by_uid.get(uid) if uid is not None else None
            if lab is None:
                # face vanished (extrusion / T2): report last-known state, not alive
                cached = self._cache.get(bid) or self._cell_meta.get(bid, {})
                positions[bid] = list(cached.get("pos", [0.0, 0.0]))
                sizes[bid] = cached.get("size", 0.0)
                types[bid] = cached.get("type", "")
                alive[bid] = False
                neighbors[bid] = []
                shared[bid] = {}
                footprints[bid] = no_footprint()
                continue
            footprints[bid] = {"kind": "polygon", "xy": polys.get(lab, [])}
            pos = [float(fd.loc[lab, "x"]), float(fd.loc[lab, "y"])]
            size = float(fd.loc[lab, "area"])
            ctype = str(fd.loc[lab, "cell_type"])
            positions[bid] = pos
            sizes[bid] = size
            types[bid] = ctype
            alive[bid] = bool(self._alive.get(bid, True)) and bool(fd.loc[lab, "is_alive"])
            self._cache[bid] = {"pos": pos, "size": size, "type": ctype}
            neighbors[bid] = []
            shared[bid] = {}

        # in-sheet contact: shared junction length by neighbour type (the vertex
        # analog of CPM contact area). A border half-edge (no opposite) touches
        # medium, which -- like CPM's medium tag -- is not reported.
        bid_of_label = {lab: self._uid_to_bid.get(int(u)) for lab, u in fd["unique_id"].items()}
        for pos_i in np.flatnonzero(opp >= 0):
            a = bid_of_label.get(ed_face[pos_i])
            b = bid_of_label.get(ed_face[opp[pos_i]])
            if a is None or b is None:
                continue
            btype = types[b]
            shared[a][btype] = shared[a].get(btype, 0.0) + float(ed_len[pos_i])
            if b not in neighbors[a]:
                neighbors[a].append(b)

        # overlay contact: polygon overlap area between an overlay cell and any
        # other cell (the 2-layer contact surface; Sego's immune layer sits on the
        # epithelium), credited symmetrically to both cells.
        others = [b for b in self._all_ids if footprints[b]["xy"]]   # faces still in the mesh
        over = [b for b in others if self._layer_of.get(b) == OVERLAY]
        for a in over:
            for b in others:
                if b == a or (self._layer_of.get(b) == OVERLAY and b < a and b in over):
                    continue  # each overlay-overlay pair once
                ov = overlap_area(footprints[b]["xy"], footprints[a]["xy"])
                if ov <= 1e-9:
                    continue
                shared[a][types[b]] = shared[a].get(types[b], 0.0) + ov
                shared[b][types[a]] = shared[b].get(types[a], 0.0) + ov
                if b not in neighbors[a]:
                    neighbors[a].append(b)
                if a not in neighbors[b]:
                    neighbors[b].append(a)

        return {
            "positions": positions,
            "sizes": sizes,
            "types": types,
            "shared_surface_by_type": shared,
            "neighbors": neighbors,
            "alive": alive,
            "footprints": footprints,
        }

    def _face_polygons(self, labels) -> dict:
        """face label -> ordered [[x, y], ...] boundary vertices for the given
        faces, walking half-edges srce -> trgt. Raises if a ring does not close
        into >= 3 vertices rather than emit a partial one."""
        eptm = self._solver.eptm
        ed = eptm.edge_df
        vx = eptm.vert_df["x"].to_dict()
        vy = eptm.vert_df["y"].to_dict()
        sel = ed[ed["face"].isin(list(labels))]
        out = {}
        for lab, grp in sel.groupby("face"):
            nxt = dict(zip(grp["srce"].tolist(), grp["trgt"].tolist()))
            start = next(iter(nxt))
            ring, v = [], start
            while True:
                ring.append([float(vx[v]), float(vy[v])])
                v = nxt.get(v)
                if v is None or v == start or len(ring) > len(nxt):
                    break
            if v != start or len(ring) < 3 or len(ring) != len(nxt):
                raise RuntimeError(f"face {lab}: half-edge ring did not close "
                                   f"({len(ring)} of {len(nxt)} vertices)")
            out[lab] = ring
        return out
