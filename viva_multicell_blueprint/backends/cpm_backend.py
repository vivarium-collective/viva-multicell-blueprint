"""CPMBackend: the viva-cpm Cellular Potts model behind the ``MechanicsBackend`` port.

This wraps ``viva_cpm``'s Rust ``World`` (built via ``viva_cpm.schema.load_world``)
without changing it (spec section 5, the CPM substrate). The blueprint's
backend-neutral cell list is materialized **exactly 1:1** -- each blueprint cell
becomes one CPM cell at its centroid with ``target_volume = size`` -- and an
identity map records ``blueprint_cell_id <-> cpm_cell_id`` so the comparison can
keep a shared identity across backends.

The field seam (spec section 5). The authoritative reaction-diffusion field lives
in the shared ``FieldProcess``, NOT here. So the CPM World's native field dynamics
are **zeroed** at build (``d=0``, ``decay=0``, no secretion); each step the shared
``field_grid`` is written straight into the engine via ``World.set_field(idx, grid)``
(a bulk overwrite in the same flat layout ``field_conc`` returns), and ``step()``
cannot re-diffuse it. CPM's native *chemotaxis* term is kept, so its Metropolis
flips are biased off the authoritative shared field -- CPM contributes only its
stochastic flip mechanics, never a second field engine.

Per step: apply ``fates`` (``set_cell_type``), write the shared field
(``set_field`` per field), run ``mcs_per_update`` sweeps (``step``), then read back
``cell_coms``/``cell_volumes``/``cell_types`` and ``cell_contact_area_by_type`` and
emit the port. ``shared_surface_by_type`` is the contact-area readback regrouped
from integer type tags into blueprint type NAMES (medium / tag 0 excluded, the CPM
analog of the vertex backend dropping its ``guard`` collar). Deaths remove the cell
(``remove_cells``) and mark it not-alive; divisions split it (``divide_cells``) --
the daughter has no blueprint id, so, as on the vertex side, it is tracked by
lineage and not folded into the t=0 cohort. Lineage events arrive shaped
``{"cell_id": <id>}`` (a bare id is also accepted).
"""
from __future__ import annotations

import math

import numpy as np

from viva_cpm.schema import load_world

from viva_multicell_blueprint.compile.adhesion import to_cpm_contact
from viva_multicell_blueprint.compile.base import NO_FOOTPRINT, MechanicsBackend

MEDIUM = "medium"

# Default per-type CPM mechanics. A volume constraint keeps cells near their
# target size; the surface constraint is off by default (lambda_surface=0) so a
# cell needs no per-type perimeter tuning to stay well-formed.
_DEFAULT_MECH = {
    "lambda_volume": 1.0,
    "target_surface": 0.0,
    "lambda_surface": 0.0,
}
# Default contact energies when the blueprint supplies no adhesion: cohesive
# (cell-medium costs more than cell-cell) so seeded blocks hold together.
_DEFAULT_J_MEDIUM = 16.0
_DEFAULT_J_CELL = 6.0


class CPMBackend(MechanicsBackend):
    """viva-cpm Cellular Potts substrate satisfying the ``MechanicsBackend`` contract."""

    def initialize(self, config):
        bp = config.get("backend_params") or {}
        self._mcs = int(bp.get("mcs_per_update", 10))

        cells = config.get("initial_state") or []
        self._all_ids = [str(c["id"]) for c in cells]

        self._type_ids = self._build_type_ids(config, cells, bp)
        self._type_name = {v: k for k, v in self._type_ids.items()}
        self._dims = self._resolve_dims(config, cells, bp)

        # per-blueprint-cell bookkeeping; a cell stays a port citizen for life.
        self._cell_meta = {}
        self._alive = {}
        self._cache = {}
        for c in cells:
            bid = str(c["id"])
            pos = [float(v) for v in c["centroid"]]
            self._cell_meta[bid] = {
                "pos": pos,
                "size": float(c.get("size", 1.0)),
                "type": str(c["type"]),
            }
            self._alive[bid] = True

        spec, self._field_idx, self._cpm_id = self._build_spec(config, cells, bp)
        self._world = load_world(spec)
        self.identity = dict(self._cpm_id)             # blueprint id (str) -> cpm id (int)
        self._cpm_to_bid = {v: k for k, v in self._cpm_id.items()}

    # -- construction -------------------------------------------------------- #
    def _build_type_ids(self, config, cells, bp) -> dict:
        """Name -> integer tag. ``medium`` is tag 0 (CPM background); other names
        are numbered 1.. in first-appearance order. An explicit ``type_ids`` map
        (from the Task 12 ``to_cpm_contact`` lowering) wins."""
        if bp.get("type_ids"):
            tids = {str(k): int(v) for k, v in bp["type_ids"].items()}
            tids.setdefault(MEDIUM, 0)
            return tids
        tids = {MEDIUM: 0}
        names = [ct["name"] if isinstance(ct, dict) else str(ct)
                 for ct in (config.get("cell_types") or [])]
        names += [str(c["type"]) for c in cells]
        for n in names:
            if n not in tids:
                tids[n] = len(tids)
        return tids

    def _resolve_dims(self, config, cells, bp) -> tuple:
        """Lattice dims (nx, ny, nz). Explicit ``backend_params.dims`` wins; else
        derive from the domain extent/resolution; else infer a bounding box around
        the centroids with a margin for each cell's seeded footprint."""
        if bp.get("dims"):
            d = list(bp["dims"])
            while len(d) < 3:
                d.append(1)
            return tuple(int(v) for v in d[:3])
        domain = config.get("domain") or {}
        extent = domain.get("extent_um")
        res = domain.get("grid_resolution_um")
        if extent and res:
            d = [max(1, int(math.ceil(float(e) / float(res)))) for e in extent]
            while len(d) < 3:
                d.append(1)
            return tuple(d[:3])
        # infer from centroids + the largest seeded half-width + a margin.
        if not cells:
            return (10, 10, 1)
        xs = [float(c["centroid"][0]) for c in cells]
        ys = [float(c["centroid"][1]) for c in cells]
        zs = [float(c["centroid"][2]) if len(c["centroid"]) > 2 else 0.0 for c in cells]
        half = max(self._side(float(c.get("size", 1.0))) for c in cells)
        margin = int(half) + 2
        nx = int(math.ceil(max(xs))) + margin
        ny = int(math.ceil(max(ys))) + margin
        nz = int(math.ceil(max(zs))) + (margin if any(z > 0 for z in zs) else 1)
        return (nx, ny, max(1, nz))

    @staticmethod
    def _side(size: float) -> int:
        """Square (2D) side length whose area ~ ``size``; at least 1."""
        return max(1, int(round(math.sqrt(max(size, 1.0)))))

    def _mech_for(self, config) -> dict:
        """Per-type-name mechanics knobs (lambda_volume / target_surface /
        lambda_surface), merged over the defaults from each cell_type's
        ``mechanics`` block."""
        out = {}
        for ct in (config.get("cell_types") or []):
            if not isinstance(ct, dict):
                continue
            mech = dict(_DEFAULT_MECH)
            mech.update({k: ct[k] for k in _DEFAULT_MECH if k in ct})
            mech.update({k: v for k, v in (ct.get("mechanics") or {}).items()
                         if k in _DEFAULT_MECH})
            out[str(ct["name"])] = mech
        return out

    def _seed_block(self, centroid, size):
        """Axis-aligned block [x0,y0,z0,x1,y1,z1) centered on ``centroid`` whose
        voxel count ~ ``size``, clamped inside the lattice (x1/y1/z1 exclusive)."""
        nx, ny, nz = self._dims
        cx = float(centroid[0])
        cy = float(centroid[1])
        cz = float(centroid[2]) if len(centroid) > 2 else 0.0
        if nz > 1:
            s = max(1, int(round(size ** (1.0 / 3.0))))
            ext = [(cx, s, nx), (cy, s, ny), (cz, s, nz)]
        else:
            s = self._side(size)
            ext = [(cx, s, nx), (cy, s, ny), (0.0, 1, 1)]
        lo, hi = [], []
        for c, span, n in ext:
            x0 = int(round(c)) - span // 2
            x0 = max(0, min(x0, n - span if n - span > 0 else 0))
            x1 = min(n, x0 + span)
            x0 = max(0, min(x0, x1 - 1))
            lo.append(x0)
            hi.append(x1)
        return [lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]]

    def _build_spec(self, config, cells, bp):
        """Assemble the ``load_world`` spec. Returns (spec, field_name->idx, bid->cpm_id)."""
        nx, ny, nz = self._dims
        potts = dict(bp.get("potts") or {})
        boundary = potts.get("boundary", (config.get("domain") or {}).get("boundary", "noflux"))
        spec = {
            "potts": {
                "dims": [nx, ny, nz],
                "boundary": "periodic" if boundary == "periodic" else "noflux",
                "neighbor_order": int(potts.get("neighbor_order", 2)),
                "temperature": float(potts.get("temperature", 10.0)),
                "seed": int(potts.get("seed", 0)),
            },
            "contact": self._contact(config, bp),
        }

        label_image = bp.get("label_image")
        if label_image is not None:
            cpm_id = self._seed_labels_spec(spec, cells, bp, label_image)
        else:
            cpm_id = self._seed_cells_spec(spec, config, cells)

        # Fields with NATIVE DYNAMICS ZEROED: d=0, decay=0, no secretion -> the
        # engine only READS the grid written each step via set_field. Chemotaxis
        # is kept so CPM biases its flips off the authoritative shared field.
        field_idx = {}
        fields = []
        for i, f in enumerate(bp.get("fields") or []):
            name = f["name"] if isinstance(f, dict) else str(f)
            fields.append({
                "name": name,
                "d": 0.0,
                "decay": 0.0,
                "secretion": [],
                "chemotaxis": [
                    {"type": self._tag(c["type"]), "lambda": float(c.get("strength", c.get("lambda", 0.0)))}
                    for c in (f.get("chemotaxis") or []) if isinstance(f, dict)
                ],
            })
            field_idx[name] = i  # declaration order IS the engine field index
        if fields:
            spec["fields"] = fields
        return spec, field_idx, cpm_id

    def _seed_cells_spec(self, spec, config, cells):
        """Explicit per-cell placement. load_world assigns cpm ids 1.. in list
        order, so the i-th blueprint cell maps to cpm id i+1 (exact 1:1)."""
        mech = self._mech_for(config)
        cell_specs = []
        cpm_id = {}
        for i, c in enumerate(cells):
            bid = str(c["id"])
            tname = str(c["type"])
            m = mech.get(tname, _DEFAULT_MECH)
            cell_specs.append({
                "type": self._tag(tname),
                "target_volume": float(c.get("size", 1.0)),
                "lambda_volume": float(m["lambda_volume"]),
                "target_surface": float(m["target_surface"]),
                "lambda_surface": float(m["lambda_surface"]),
                "seed_block": self._seed_block(c["centroid"], float(c.get("size", 1.0))),
            })
            cpm_id[bid] = i + 1  # load_world add_cell returns contiguous ids from 1
        spec["cells"] = cell_specs
        return cpm_id

    def _seed_labels_spec(self, spec, cells, bp, label_image):
        """Label-image placement via ``seed_from_labels``. Label value == blueprint
        cell id; the engine compacts labels to contiguous cpm ids 1..K in order of
        FIRST APPEARANCE in the flat label list (NOT ascending label order)."""
        labels = np.asarray(label_image)
        flat = labels.flatten(order="F").astype(int).tolist()
        types = {int(c["id"]): self._tag(str(c["type"])) for c in cells}
        spec["seed_labels"] = {
            "labels": flat,
            "types": types,
            "default_type": 0,
            "target_volume": float(bp.get("target_volume", 1.0)),
            "lambda_volume": float(bp.get("lambda_volume", _DEFAULT_MECH["lambda_volume"])),
        }
        wanted = {int(c["id"]) for c in cells}
        seen: list = []
        seen_set: set = set()
        for v in flat:
            if v in wanted and v not in seen_set:
                seen_set.add(v)
                seen.append(v)
        return {str(lbl): rank + 1 for rank, lbl in enumerate(seen)}

    def _contact(self, config, bp) -> list:
        """CPM contact (J) matrix. A pre-lowered ``backend_params.contact`` wins;
        else lower the neutral blueprint ``adhesion`` via ``to_cpm_contact``; else
        a cohesive default over every type pair so seeded blocks hold shape."""
        if bp.get("contact"):
            return [{"a": int(e["a"]), "b": int(e["b"]), "j": float(e["j"])}
                    for e in bp["contact"]]
        adhesion = config.get("adhesion") or []
        if adhesion:
            return to_cpm_contact(adhesion, dict(self._type_ids))
        tags = sorted(set(self._type_ids.values()))
        out = []
        for i, a in enumerate(tags):
            for b in tags[i:]:
                if a == 0 and b == 0:
                    continue
                j = _DEFAULT_J_MEDIUM if (a == 0 or b == 0) else _DEFAULT_J_CELL
                out.append({"a": a, "b": b, "j": j})
        return out

    def _tag(self, name) -> int:
        if isinstance(name, (int, np.integer)):
            return int(name)
        return self._type_ids[str(name)]

    # -- per-step ------------------------------------------------------------ #
    def update(self, inputs, interval):
        fates = inputs.get("fates") or {}
        field_grid = inputs.get("field_grid") or {}
        divisions = inputs.get("divisions") or []
        deaths = inputs.get("deaths") or []

        # (a) fates: hard cell-type switches (port values are type NAMES).
        for cid, new_type in fates.items():
            cpm = self._cpm_id.get(str(cid))
            if cpm is not None and self._alive.get(str(cid), False):
                self._world.set_cell_type(cpm, self._tag(new_type))
                self._cell_meta[str(cid)]["type"] = (
                    new_type if isinstance(new_type, str) else self._type_name.get(int(new_type), str(new_type))
                )

        # (b) deaths: remove the cell from the lattice, mark it not-alive.
        for ev in deaths:
            bid = str(ev["cell_id"] if isinstance(ev, dict) else ev)
            cpm = self._cpm_id.get(bid)
            if cpm is not None and self._alive.get(bid, False):
                self._world.remove_cells([cpm])
                self._alive[bid] = False

        # (c) divisions: split the mother now; the daughter has no blueprint id
        #     (tracked by lineage, not folded into the t=0 cohort).
        div_ids = [self._cpm_id.get(str(ev["cell_id"] if isinstance(ev, dict) else ev))
                   for ev in divisions]
        div_ids = [c for c in div_ids if c is not None and self._alive.get(self._cpm_to_bid.get(c, ""), False)]
        if div_ids:
            self._world.divide_cells(div_ids, True)

        # (d) write the AUTHORITATIVE shared field into the engine (native
        #     dynamics zeroed -> step() reads it, never re-diffuses it).
        for name, grid in field_grid.items():
            idx = self._field_idx.get(name)
            if idx is None:
                continue
            arr = np.asarray(grid, dtype=float)
            flat = arr.ravel() if arr.ndim == 1 else arr.flatten(order="F")
            self._world.set_field(idx, flat)

        # (e) advance the Metropolis flips.
        self._world.step(self._mcs)
        return self._emit()

    # -- read-back ----------------------------------------------------------- #
    def _emit(self):
        nz = self._dims[2]
        volumes = list(self._world.cell_volumes())
        types = list(self._world.cell_types())
        coms = [list(c) for c in self._world.cell_coms()]

        positions, sizes, out_types, alive, neighbors, shared = {}, {}, {}, {}, {}, {}
        nbr_ids = self._neighbor_ids()
        footprints = {}
        occ = self._occupancy()
        for bid in self._all_ids:
            cpm = self._cpm_id.get(bid)
            is_alive = bool(self._alive.get(bid, False)) and cpm is not None \
                and cpm < len(volumes) and volumes[cpm] > 0
            if not is_alive:
                # dead / removed: emit last-known (or seeded) placeholder, not-alive.
                cached = self._cache.get(bid) or self._cell_meta.get(bid, {})
                positions[bid] = list(cached.get("pos", [0.0, 0.0]))
                sizes[bid] = float(cached.get("size", 0.0))
                out_types[bid] = cached.get("type", "")
                alive[bid] = False
                neighbors[bid] = []
                shared[bid] = {}
                footprints[bid] = dict(NO_FOOTPRINT)
                continue
            footprints[bid] = {"kind": "occupancy", "xy": occ.get(cpm, [])}
            com = coms[cpm]
            pos = [float(com[0]), float(com[1])] if nz <= 1 else [float(v) for v in com]
            size = float(volumes[cpm])
            tname = self._type_name.get(int(types[cpm]), str(types[cpm]))
            positions[bid] = pos
            sizes[bid] = size
            out_types[bid] = tname
            alive[bid] = True
            self._cache[bid] = {"pos": pos, "size": size, "type": tname}

            # contact-area-by-type: regroup integer tags -> type NAMES, drop medium.
            by_type = {}
            for tag, area in self._world.cell_contact_area_by_type(cpm).items():
                if int(tag) == 0:
                    continue  # medium (CPM background) is not a cell type
                by_type[self._type_name.get(int(tag), str(tag))] = float(area)
            shared[bid] = by_type
            neighbors[bid] = [self._cpm_to_bid[n] for n in nbr_ids.get(cpm, ())
                              if n in self._cpm_to_bid]

        return {
            "positions": positions,
            "sizes": sizes,
            "types": out_types,
            "shared_surface_by_type": shared,
            "neighbors": neighbors,
            "alive": alive,
            "footprints": footprints,
        }

    def _occupancy(self) -> dict:
        """cpm id -> list of [x, y(, z)] voxel indices it owns, in ONE pass over
        the lattice snapshot (stable argsort groups voxels by owner)."""
        nx, ny, nz = self._dims
        lab = np.asarray(self._world.snapshot()).reshape((nx, ny, nz), order="F")
        flat = lab.ravel()
        idx = np.flatnonzero(flat)
        order = idx[np.argsort(flat[idx], kind="stable")]
        owners = flat[order]
        coords = np.stack(np.unravel_index(order, lab.shape), axis=1)
        if nz <= 1:
            coords = coords[:, :2]
        cuts = np.flatnonzero(np.diff(owners)) + 1
        out = {}
        for o, grp in zip(np.split(owners, cuts), np.split(coords, cuts)):
            if len(o):
                out[int(o[0])] = grp.astype(float).tolist()
        return out

    def _neighbor_ids(self) -> dict:
        """Map cpm cell id -> set of adjacent cpm cell ids, from the lattice
        snapshot (Moore neighborhood, matching the contact-area readback).

        CPM exposes contact area BY TYPE but not neighbor ids, so we scan the
        label grid once: wherever two distinct non-medium cells abut, each is the
        other's neighbor.
        """
        nx, ny, nz = self._dims
        lab = np.asarray(self._world.snapshot()).reshape((nx, ny, nz), order="F")
        nbrs: dict = {}

        def _link(a, b):
            if a == b or a == 0 or b == 0:
                return
            nbrs.setdefault(int(a), set()).add(int(b))
            nbrs.setdefault(int(b), set()).add(int(a))

        # compare against all forward Moore offsets; each unordered pair once.
        for dx in (0, 1):
            for dy in (-1, 0, 1):
                for dz in ((-1, 0, 1) if nz > 1 else (0,)):
                    if (dx, dy, dz) <= (0, 0, 0):
                        continue
                    a = lab[
                        max(0, -dx):nx - max(0, dx),
                        max(0, -dy):ny - max(0, dy),
                        max(0, -dz):nz - max(0, dz),
                    ]
                    b = lab[
                        max(0, dx):nx - max(0, -dx),
                        max(0, dy):ny - max(0, -dy),
                        max(0, dz):nz - max(0, -dz),
                    ]
                    mask = (a != b) & (a != 0) & (b != 0)
                    for av, bv in zip(a[mask].tolist(), b[mask].tolist()):
                        _link(av, bv)
        return nbrs
