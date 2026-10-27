"""VertexBackend: the tyssue vertex model behind the ``MechanicsBackend`` port.

This wraps ``vivarium_tyssue``'s :class:`EulerSolver` (spec section 5, the vertex
substrate) without changing it. The blueprint's backend-neutral cell list is
materialized as a **guard-ring-bounded Voronoi tessellation** of the cell centroids,
written to the ``.hf5`` the solver loads, with per-face ``cell_type`` and
``prefered_area = size``. Vertex *motility* comes from the fork's
``ActiveMigration`` effector: each step the field gradient is sampled at every
face centroid and the per-face migration director (``mx``/``my``/``mz`` +
``migration_strength``) is rewritten through the solver's ``behaviors`` port
(the ``apply_gradient`` behavior) -- no change to ``EulerSolver``.

Mesh fidelity is honest, not assumed. The fork's Voronoi builders add a boundary
**guard collar** and weld coincident vertices, so faces are *not* guaranteed 1:1
with the input centroids. Identity is built by **nearest-centroid assignment over
interior faces**; guard/collar faces are tagged and excluded; and the residual
(how many centroids got a clean, uncontested interior face) is measured and
reported in :attr:`residual_report` -- this feeds the section 9.2
``initial-state-fidelity`` study.

Division / apoptosis / differentiation enter through the same ``behaviors`` port
(``split_cells`` / ``apoptosis`` is available via ``is_alive``; ``differentiation``).
Lineage events arrive on the ``divisions`` / ``deaths`` input ports shaped
``{"cell_id": <id>}`` (a bare id is also accepted), matching the port contract;
their effect is reflected in the emitted ``alive`` / ``types`` / geometry.
"""
from __future__ import annotations

import contextlib
import io
import math
import tempfile
from pathlib import Path

import numpy as np

from viva_multicell_blueprint.compile.base import MechanicsBackend

# Default sheet mechanics. ActiveMigration is the motility effector the field
# drives; FaceAreaElasticity + LineTension keep cells bounded so a strong
# migration transient relaxes instead of exploding (see max_displacement).
_DEFAULT_EFFECTORS = ["FaceAreaElasticity", "LineTension", "ActiveMigration"]
_MIGRATION_COLS = ("mx", "my", "mz", "migration_strength")


def _voronoi_sheet(centroids: np.ndarray, guard_factor: float = 2.0):
    """Build a flat-3D tyssue :class:`Sheet` from a guard-ring-bounded Voronoi
    tessellation of ``centroids``.

    A rectangular **guard ring** of dummy generators ~``guard_factor`` cell-widths
    outside the cloud bounds every real cell to a finite size (the frame absorbs
    the unbounded exterior Voronoi regions); the resulting sheet is promoted to a
    flat 3D sheet (z=0) so the standard ``SheetGeometry`` + rust kernels apply,
    matching the fork's ``hra/loaders`` pattern.
    """
    from scipy.spatial import Voronoi, cKDTree
    from tyssue import Sheet
    from tyssue.generation import from_2d_voronoi

    centroids = np.asarray(centroids, dtype=float)
    d, _ = cKDTree(centroids).query(centroids, k=min(2, len(centroids)))
    step = float(np.median(d[:, 1])) if d.ndim == 2 and d.shape[1] > 1 else 1.0
    if not (step > 0):
        step = 1.0
    lo = centroids.min(0) - guard_factor * step
    hi = centroids.max(0) + guard_factor * step
    xs = np.arange(lo[0], hi[0] + step, step)
    ys = np.arange(lo[1], hi[1] + step, step)
    guard = np.vstack([
        np.c_[xs, np.full(len(xs), lo[1])],
        np.c_[xs, np.full(len(xs), hi[1])],
        np.c_[np.full(len(ys), lo[0]), ys],
        np.c_[np.full(len(ys), hi[0]), ys],
    ])
    vor = Voronoi(np.vstack([centroids, guard]))
    with contextlib.redirect_stdout(io.StringIO()):
        dsets = from_2d_voronoi(vor)
        sheet = Sheet("vertex", dsets, coords=["x", "y"])
    return _promote_to_flat_3d(sheet)


def _promote_to_flat_3d(sheet, name: str = "vertex"):
    """Rebuild a 2D sheet as a flat 3D sheet (z=0). tyssue derives dcoords/ncoords
    at construction, so a genuine 3D sheet must be *reconstructed* (setting coords
    alone leaves stale 2D bookkeeping). face_df columns carry over."""
    from tyssue import Sheet

    vert = sheet.vert_df.copy()
    vert["z"] = 0.0
    face = sheet.face_df.copy()
    if "z" not in face:
        face["z"] = 0.0
    with contextlib.redirect_stdout(io.StringIO()):
        s3 = Sheet(name, {"vert": vert, "edge": sheet.edge_df.copy(), "face": face},
                   coords=["x", "y", "z"])
        s3.reset_index()
        s3.reset_topo()
    return s3


def _assign_identity(sheet, centroids: np.ndarray):
    """Nearest-centroid assignment of each input centroid to an *interior* face.

    Faces whose centroid is non-finite (unbounded Voronoi regions seeded by the
    guard ring) are excluded. Returns ``(face_of_cell, residual_report)``:
    ``face_of_cell[i]`` is the face-index label assigned to centroid ``i`` (``-1``
    if none), and the report quantifies how many centroids got a *clean*
    (uncontested) interior face -- the section 9.2 fidelity residual.
    """
    from scipy.spatial import cKDTree

    fx = sheet.face_df[["x", "y"]].to_numpy(dtype=float)
    finite = np.isfinite(fx).all(axis=1)
    finite_labels = sheet.face_df.index.to_numpy()[finite]
    _, loc = cKDTree(fx[finite]).query(centroids)
    face_of_cell = finite_labels[loc]

    uniq, counts = np.unique(face_of_cell, return_counts=True)
    contested = set(uniq[counts > 1].tolist())
    n_collision = int(sum(counts[counts > 1]))
    n_clean = int(len(centroids) - n_collision)
    report = {
        "n_input": int(len(centroids)),
        "n_clean": n_clean,
        "n_collision": n_collision,
        "fraction_clean": (n_clean / len(centroids)) if len(centroids) else 0.0,
        "contested_faces": sorted(int(f) for f in contested),
    }
    return face_of_cell.astype(int), report


class VertexBackend(MechanicsBackend):
    """tyssue vertex substrate satisfying the ``MechanicsBackend`` contract."""

    def initialize(self, config):
        bp = config.get("backend_params") or {}
        self._geom = str(bp.get("geom", "SheetGeometry"))
        self._factory = str(bp.get("factory", "model_factory"))
        self._effectors = list(bp.get("effectors", _DEFAULT_EFFECTORS))
        self._ref_effector = str(bp.get("ref_effector", "FaceAreaElasticity"))
        self._area_elasticity = float(bp.get("area_elasticity", 1.0))
        self._line_tension = float(bp.get("line_tension", 0.01))
        self._max_disp = float(bp.get("max_displacement", 0.1))
        self._migration_scale = float(bp.get("migration_scale", 0.4))
        self._chemotaxis = list(bp.get("chemotaxis") or [])
        self._guard_factor = float(bp.get("guard_factor", 2.0))

        self._t = 0.0
        self._alive: dict[str, bool] = {}
        self._cache: dict[str, dict] = {}  # last-known emit per blueprint id
        self._build(config.get("initial_state") or [])

    # -- construction -------------------------------------------------------- #
    def _build(self, cells):
        from tyssue import SheetGeometry
        from tyssue.io.hdf5 import save_datasets

        centroids = np.array(
            [[float(c["centroid"][0]), float(c["centroid"][1])] for c in cells],
            dtype=float,
        )
        ids = [str(c["id"]) for c in cells]
        types = [str(c["type"]) for c in cells]
        sizes = [float(c.get("size", 1.0)) for c in cells]

        sheet = _voronoi_sheet(centroids, guard_factor=self._guard_factor)
        SheetGeometry.update_all(sheet)
        face_of_cell, report = _assign_identity(sheet, centroids)

        # Tag every face guard by default; paint the assigned interior faces with
        # their blueprint cell_type + prefered_area. prefered_area is the TARGET
        # (= blueprint size); the actual relaxed area is what the port emits.
        sheet.face_df["cell_type"] = "guard"
        sheet.face_df["prefered_area"] = sheet.face_df["area"].to_numpy()

        # Every blueprint id is a port citizen regardless of mesh fidelity. Keep
        # its blueprint pos/size/type so an unmapped cell still emits a real
        # placeholder (never dropped -> the "every cell in every port" contract
        # holds even when the section 9.2 residual is non-zero).
        self._all_ids = list(ids)
        self._cell_meta = {
            ids[i]: {
                "pos": [float(centroids[i, 0]), float(centroids[i, 1])],
                "size": sizes[i],
                "type": types[i],
            }
            for i in range(len(ids))
        }
        self._uid: dict[str, int] = {}  # blueprint id -> stable tyssue unique_id
        unmapped = []
        assigned_faces = set()
        for i, lab in enumerate(face_of_cell):
            if lab < 0 or lab in assigned_faces:
                # unassigned or contested (a later centroid colliding onto an
                # already-claimed face). Record it; it still emits as not-alive.
                unmapped.append(ids[i])
                self._alive[ids[i]] = False
                continue
            assigned_faces.add(int(lab))
            sheet.face_df.loc[lab, "cell_type"] = types[i]
            sheet.face_df.loc[lab, "prefered_area"] = sizes[i]
            self._uid[ids[i]] = int(sheet.face_df.loc[lab, "unique_id"])
            self._alive[ids[i]] = True

        report["n_guard_faces"] = int((sheet.face_df["cell_type"] == "guard").sum())
        report["unmapped_ids"] = unmapped
        self.residual_report = report
        self.identity = dict(self._uid)
        self._uid_to_bid = {u: b for b, u in self._uid.items()}

        for col, val in zip(_MIGRATION_COLS, (1.0, 0.0, 0.0, 0.0)):
            sheet.face_df[col] = val
        if "viscosity" not in sheet.vert_df.columns:
            sheet.vert_df["viscosity"] = 1.0

        # Held on the instance so its finalizer removes the scratch mesh dir when
        # the backend is collected / at interpreter exit (no orphaned temp dirs).
        self._tmpdir_obj = tempfile.TemporaryDirectory(prefix="vertex_backend_")
        hf5 = str(Path(self._tmpdir_obj.name) / "mesh.hf5")
        save_datasets(hf5, sheet)
        self._solver = self._make_solver(hf5)

    def _make_solver(self, hf5):
        from process_bigraph import allocate_core
        from vivarium_tyssue.data_types import register_types
        from vivarium_tyssue.processes import register_processes
        from vivarium_tyssue.processes.eulersolver import EulerSolver

        core = allocate_core()
        core = register_types(core)
        core = register_processes(core)
        cfg = {
            "name": "vertex",
            "eptm": hf5,
            "tissue_type": "Sheet",
            "geom": self._geom,
            "effectors": self._effectors,
            "ref_effector": self._ref_effector,
            "factory": self._factory,
            # keys are the eptm DataFrame attribute names (face_df / edge_df).
            "parameters": {
                "face_df": {"area_elasticity": self._area_elasticity},
                "edge_df": {"line_tension": self._line_tension},
            },
            "settings": {},
            "maps": {},
            "backend": "python",
            "auto_reconnect": False,
            "bounds": {},
            "output_columns": {},
            "substeps": 1,
            # explicit-Euler safety clamp: bounds a strong migration transient so
            # it relaxes over steps instead of exploding to NaN in one.
            "max_displacement": self._max_disp,
        }
        return EulerSolver(cfg, core=core)

    # -- per-step ------------------------------------------------------------ #
    def _face_label(self, uid):
        """Current face-index label for a stable unique_id (face_df reindexes on
        topology changes), or None if the face is gone (extruded / T2)."""
        fd = self._solver.eptm.face_df
        m = fd.index[fd["unique_id"] == uid]
        return m[0] if len(m) else None

    def _migration_behaviors(self, field_grid):
        """Rewrite each interior face's migration director from the sampled field
        gradient at its centroid. Returns a list with one ``apply_gradient``
        behavior node (or empty). Chemotaxis wiring (``[{field, type?, strength}]``)
        selects which field(s) drive which cell types; absent -> no migration."""
        if not field_grid or not self._chemotaxis:
            return []
        fd = self._solver.eptm.face_df
        grads = {}
        for name, grid in field_grid.items():
            arr = np.asarray(grid, dtype=float)
            if arr.ndim >= 2:
                gx, gy = np.gradient(arr)[:2]
                grads[name] = (gx, gy, arr.shape)
        if not grads:
            return []

        upd = {c: {} for c in _MIGRATION_COLS}
        for bid, uid in self._uid.items():
            if not self._alive.get(bid, True):
                continue
            lab = self._face_label(uid)
            if lab is None:
                continue
            ctype = str(fd.loc[lab, "cell_type"])
            x = float(fd.loc[lab, "x"])
            y = float(fd.loc[lab, "y"])
            vx = vy = 0.0
            for ch in self._chemotaxis:
                want = ch.get("type")
                if want is not None and want != ctype:
                    continue
                name = ch.get("field")
                if name not in grads:
                    continue
                gx, gy, shp = grads[name]
                ix = int(min(max(round(x), 0), shp[0] - 1))
                iy = int(min(max(round(y), 0), shp[1] - 1))
                s = float(ch.get("strength", 1.0))
                vx += s * float(gx[ix, iy])
                vy += s * float(gy[ix, iy])
            norm = math.hypot(vx, vy)
            # ActiveMigration normalizes the director internally; strength is the
            # force magnitude, scaled from the gradient magnitude.
            upd["mx"][uid] = vx
            upd["my"][uid] = vy
            upd["mz"][uid] = 0.0
            upd["migration_strength"][uid] = self._migration_scale * norm
        if not upd["migration_strength"]:
            return []
        return [{
            "func": "apply_gradient",
            "parameter_updates": {
                col: {"dataframe": "face", "update": dict(vals)}
                for col, vals in upd.items()
            },
        }]

    def update(self, inputs, interval):
        fates = inputs.get("fates") or {}
        field_grid = inputs.get("field_grid") or {}
        divisions = inputs.get("divisions") or []
        deaths = inputs.get("deaths") or []

        behaviors = []
        # (a) fates -> relabel cell_type via the differentiation behavior
        for cid, new_type in fates.items():
            uid = self._uid.get(str(cid))
            if uid is not None:
                behaviors.append({
                    "func": "differentiation", "cell_uid": uid, "new_type": str(new_type),
                })
        # (b) field gradient -> migration directors via apply_gradient
        behaviors.extend(self._migration_behaviors(field_grid))
        # (c) deaths -> mark not-alive (is_alive=0 stops migration + flags the port)
        dead_updates = {}
        for ev in deaths:
            cid = str(ev["cell_id"] if isinstance(ev, dict) else ev)
            uid = self._uid.get(cid)
            if uid is not None:
                self._alive[cid] = False
                dead_updates[uid] = 0
        if dead_updates:
            behaviors.append({
                "func": "apply_gradient",
                "parameter_updates": {
                    "is_alive": {"dataframe": "face", "update": dead_updates},
                },
            })
        # (d) divisions -> split the mother face now (daughter gets a fresh uid,
        #     no blueprint id -> tracked by lineage, not emitted in the cohort)
        for ev in divisions:
            cid = str(ev["cell_id"] if isinstance(ev, dict) else ev)
            uid = self._uid.get(cid)
            if uid is not None:
                behaviors.append({
                    "func": "split_cells", "geom": self._geom, "cell_uids": [uid],
                })

        self._solver.update({"behaviors": behaviors, "global_time": float(self._t)}, interval)
        self._t += float(interval)
        return self._emit()

    # -- read-back ----------------------------------------------------------- #
    def _emit(self):
        fd = self._solver.eptm.face_df
        ed = self._solver.eptm.edge_df
        # current unique_id -> face label, for O(1) neighbor lookups
        label_by_uid = {int(u): lab for lab, u in fd["unique_id"].items()}

        positions, sizes, types, alive, neighbors, shared = {}, {}, {}, {}, {}, {}
        for bid in self._all_ids:
            uid = self._uid.get(bid)
            if uid is None:
                # unmapped at build (contested / no interior face): still a port
                # citizen -> blueprint-placeholder geometry, not alive, no contacts
                meta = self._cell_meta.get(bid, {})
                positions[bid] = list(meta.get("pos", [0.0, 0.0]))
                sizes[bid] = meta.get("size", 0.0)
                types[bid] = meta.get("type", "")
                alive[bid] = False
                neighbors[bid] = []
                shared[bid] = {}
                continue
            lab = label_by_uid.get(uid)
            if lab is None:
                # face vanished (extrusion / T2): report last-known state, not alive
                cached = self._cache.get(bid, {})
                positions[bid] = cached.get("pos", [0.0, 0.0])
                sizes[bid] = cached.get("size", 0.0)
                types[bid] = cached.get("type", "")
                alive[bid] = False
                neighbors[bid] = []
                shared[bid] = {}
                continue
            pos = [float(fd.loc[lab, "x"]), float(fd.loc[lab, "y"])]
            size = float(fd.loc[lab, "area"])
            ctype = str(fd.loc[lab, "cell_type"])
            positions[bid] = pos
            sizes[bid] = size
            types[bid] = ctype
            alive[bid] = bool(self._alive.get(bid, True)) and bool(fd.loc[lab, "is_alive"])
            self._cache[bid] = {"pos": pos, "size": size, "type": ctype}

            nbrs, by_type = [], {}
            rows = ed[ed["face"] == lab]
            for _, r in rows.iterrows():
                opp = r.get("opposite", -1)
                if opp is None or opp < 0 or opp not in ed.index:
                    continue  # mesh boundary (no neighbor across this edge)
                nlab = ed.loc[opp, "face"]
                ntype = str(fd.loc[nlab, "cell_type"])
                length = float(r["length"])
                # shared-edge-length by neighbor cell_type (the vertex analog of
                # CPM contact area); the guard collar is not a real cell type.
                if ntype != "guard":
                    by_type[ntype] = by_type.get(ntype, 0.0) + length
                nuid = int(fd.loc[nlab, "unique_id"])
                nb_bid = self._uid_to_bid.get(nuid)
                if nb_bid is not None and nb_bid not in nbrs:
                    nbrs.append(nb_bid)
            neighbors[bid] = nbrs
            shared[bid] = by_type

        return {
            "positions": positions,
            "sizes": sizes,
            "types": types,
            "shared_surface_by_type": shared,
            "neighbors": neighbors,
            "alive": alive,
        }
