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
from integer type tags into blueprint type NAMES (medium / tag 0 excluded, as the
vertex backend excludes a junction on the free tissue border). Deaths remove the cell
(``remove_cells``) and mark it not-alive; divisions split it (``divide_cells``) --
the daughter is given a fresh blueprint id (``<mother>.<k>``), joins the emitted
cohort the step it is born, and is reported in the ``lineage`` map
(``daughter -> mother``) for that step only. Lineage events arrive shaped
``{"cell_id": <id>}`` (a bare id is also accepted).

Per-cell actuation ports (read from ``inputs`` each step, never from config; an
absent/empty map leaves the init-time value): ``target_size`` -> ``set_target_volume``
(native, per cell). ``adhesion`` (float | {type -> float}) -> ``set_contact`` on the
cell's TYPE pair(s) -- viva-cpm's contact matrix is per type pair, so cells of one type
share the last (id-ordered) value written. ``motility`` has NO per-cell viva-cpm knob
(temperature is global and fixed at finalize), so it is recorded and a RuntimeWarning
is raised once; it does not alter dynamics.
"""
from __future__ import annotations

import copy
import math

import numpy as np

from viva_cpm.schema import load_world

from viva_multicell_blueprint.compile.adhesion import to_cpm_contact
from viva_multicell_blueprint.compile.base import MechanicsBackend, sorted_cell_ids

from ._common import (
    dead_entry,
    event_cid,
    latent_config,
    mint_daughter_id,
    select_activations,
)

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

        # Immune-recruitment-as-activation (Task 20): which latent pool each active
        # immune type draws from. Explicit per-type map + generic latent name.
        self._latent_map, self._latent_generic = latent_config(bp)
        self._warned: set = set()

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

        # runtime actuator state (per blueprint id); daughters get a deep copy.
        self._cell_state = {bid: {"motility": 1.0, "adhesion": None, "target": None}
                            for bid in self._all_ids}
        self._div_count = {}
        self._lineage = {}

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
        activations = inputs.get("activations") or {}
        self._lineage = {}

        # (a) fates: hard cell-type switches (port values are type NAMES).
        for cid, new_type in fates.items():
            cpm = self._cpm_id.get(str(cid))
            if cpm is not None and self._alive.get(str(cid), False):
                self._world.set_cell_type(cpm, self._tag(new_type))
                self._cell_meta[str(cid)]["type"] = (
                    new_type if isinstance(new_type, str) else self._type_name.get(int(new_type), str(new_type))
                )

        # (a2) activations (immune recruitment): activate N currently-LATENT cells of
        #      each requested active type, in deterministic id order, via the SAME
        #      set_cell_type path as fates. Clamped to the available latent pool.
        self._apply_activations(activations)

        # (b) deaths: remove the cell from the lattice, mark it not-alive.
        for ev in deaths:
            bid = event_cid(ev)
            cpm = self._cpm_id.get(bid)
            if cpm is not None and self._alive.get(bid, False):
                self._world.remove_cells([cpm])
                self._alive[bid] = False

        # (b2) per-cell runtime actuators (input ports; NOT the init-time config).
        self._apply_actuators(inputs)

        # (c) divisions: viva-cpm mitosis is VOLUME-THRESHOLD based. The binding is
        #     ``World.divide_cells(threshold: float, reset_target: float)`` -- it splits
        #     EVERY non-medium cell with volume >= ``threshold`` (the mother keeps its id;
        #     a NEW cell is the other daughter; both daughters' target_volume is set to
        #     ``reset_target``). There is NO per-id split in viva-cpm (0.1.3 or main), so
        #     we map the lineage division EVENTS onto it by dividing at a threshold equal
        #     to the SMALLEST current volume among the requested, still-alive mother cells
        #     -- so every requested mother divides -- and reset the daughters to the
        #     mother's blueprint target size so they regrow as normal cells. Daughters
        #     (new cpm ids) have no blueprint id: tracked by lineage, not folded into the
        #     t=0 cohort (unchanged behaviour). Caveat: because the API thresholds on
        #     volume, any non-requested cell at least as large also divides; at the
        #     near-uniform cell volumes here divisions are rare and daughters are never
        #     cohort-scored, so this is an accepted substrate limitation, not a fate effect.
        #     Daughters are identified from a lattice snapshot (each new cpm id's voxels
        #     were owned by its mother pre-split), given a fresh blueprint id and added
        #     to the cohort + ``lineage`` (see ``_register_daughters``).
        #     (The previous ``divide_cells(div_ids, True)`` call passed a list + bool and
        #     raised TypeError against this signature on every division -- a latent bug,
        #     since no shipped study fires a CPM division.)
        div_cpm = [self._cpm_id.get(event_cid(ev))
                   for ev in divisions]
        div_cpm = [c for c in div_cpm
                   if c is not None and self._alive.get(self._cpm_to_bid.get(c, ""), False)]
        if div_cpm:
            vols = list(self._world.cell_volumes())
            req_vols = [vols[c] for c in div_cpm if c < len(vols) and vols[c] > 0]
            if req_vols:
                threshold = float(min(req_vols))
                req_sizes = [float(self._cell_meta[self._cpm_to_bid[c]]["size"])
                             for c in div_cpm if c in self._cpm_to_bid]
                reset_target = float(min(req_sizes)) if req_sizes else threshold
                n_before = self._world.n_cells()
                pre = np.asarray(self._world.snapshot()).copy()
                self._world.divide_cells(threshold, reset_target)
                self._register_daughters(n_before, pre)

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

    def _apply_actuators(self, inputs) -> None:
        """target_size / adhesion / motility input maps (see module docstring)."""
        for cid, v in (inputs.get("target_size") or {}).items():
            bid = str(cid)
            cpm = self._cpm_id.get(bid)
            if cpm is not None and self._alive.get(bid, False):
                self._cell_state[bid]["target"] = float(v)
                self._world.set_target_volume(cpm, float(v))
        if inputs.get("adhesion"):
            self.warn_once(
                "adhesion",
                "CPMBackend: adhesion is applied per TYPE PAIR (viva-cpm's contact matrix "
                "is global, last write wins), not per cell; same-type cells with differing "
                "adhesion values collapse to one value, and dict keys naming undeclared "
                "types are discarded.",
            )
        for cid in sorted_cell_ids((inputs.get("adhesion") or {}).keys()):
            adh = inputs["adhesion"][cid] if cid in inputs["adhesion"] else None
            bid = str(cid)
            if bid not in self._cell_state or not self._alive.get(bid, False) or adh is None:
                continue
            self._cell_state[bid]["adhesion"] = copy.deepcopy(adh)
            tag = self._tag(self._cell_meta[bid]["type"])
            if isinstance(adh, dict):
                for other, j in adh.items():
                    if str(other) in self._type_ids:
                        self._world.set_contact(tag, self._tag(other), float(j))
            else:
                self._world.set_contact(tag, tag, float(adh))
        for cid, v in (inputs.get("motility") or {}).items():
            bid = str(cid)
            if bid not in self._cell_state:
                continue
            self._cell_state[bid]["motility"] = float(v)
            self.warn_once(
                "motility",
                "CPMBackend: viva-cpm has no per-cell temperature/motility knob "
                "(temperature is global, fixed at finalize); the motility input is "
                "recorded but does not alter CPM dynamics.",
            )

    def _register_daughters(self, n_before: int, pre) -> None:
        """Give every cpm cell created by ``divide_cells`` a blueprint id, add it to
        the cohort, and record ``lineage[daughter] = mother``. New cpm ids are
        ``n_before+1 ..``; the mother is whoever owned the daughter's voxels in the
        pre-split snapshot."""
        post = np.asarray(self._world.snapshot())
        types = list(self._world.cell_types())
        for new in range(n_before + 1, self._world.n_cells() + 1):
            # BEST-EFFORT attribution: the mother is the majority owner (bincount argmax) of
            # the daughter's voxels in the pre-split snapshot. Under heavy simultaneous splits
            # a daughter can be mis-attributed; unattributable ones are warn-guarded + dropped.
            owned = pre[post == new]
            owned = owned[owned > 0]
            mcpm = int(np.bincount(owned.astype(int)).argmax()) if owned.size else None
            mother = self._cpm_to_bid.get(mcpm) if mcpm is not None else None
            if mother is None:
                self.warn_once(
                    "daughter",
                    f"CPMBackend: viva-cpm created daughter cell {new} whose mother "
                    f"could not be attributed; it has no blueprint id and is absent "
                    f"from the emitted maps and lineage.",
                )
                continue
            did = mint_daughter_id(self._cpm_id, mother, self._div_count)
            self._cpm_id[did] = new
            self._cpm_to_bid[new] = did
            self.identity[did] = new
            self._all_ids.append(did)
            self._alive[did] = True
            meta = dict(self._cell_meta[mother])
            meta["pos"] = list(meta["pos"])
            if new < len(types):
                meta["type"] = self._type_name.get(int(types[new]), meta["type"])
            self._cell_meta[did] = meta
            self._cell_state[did] = copy.deepcopy(self._cell_state[mother])
            tgt = self._cell_state[mother]["target"]
            if tgt is not None:  # runtime target_size survives the division reset
                self._world.set_target_volume(new, tgt)
                self._world.set_target_volume(self._cpm_id[mother], tgt)
            self._lineage[did] = mother

    def _apply_activations(self, activations) -> None:
        """Activate latent cells into active immune types (Task 20). For each
        ``active_type -> count``, pick up to ``count`` currently-alive cells whose
        type is in that type's latent pool (deterministic id order) and ``set_cell_type``
        them to the active type. Clamped to the pool; a cell is consumed at most once
        per step (so two active types cannot both claim the same latent cell)."""
        if not activations:
            return
        types_now = list(self._world.cell_types())

        def current_name(bid):
            cpm = self._cpm_id.get(bid)
            if cpm is None or not self._alive.get(bid, False) or cpm >= len(types_now):
                return None
            return self._type_name.get(int(types_now[cpm]), None)

        def skip(active_type, pool):
            if active_type in self._type_ids:
                return False
            # Fail LOUD(ish), not silent: an undeclared active type would make
            # recruitment a silent no-op (the Task-20 silent-failure trap).
            self.warn_once(
                ("activation", active_type),
                f"CPMBackend: activation requested for unknown active type "
                f"{active_type!r} (not in cell_types/type_ids) -- no cells "
                f"activated. Declare it in the blueprint cell_types.",
            )
            return True

        for bid, active_type in select_activations(
                activations, self._all_ids, current_name,
                self._latent_map, self._latent_generic, skip):
            cpm = self._cpm_id.get(bid)
            if cpm is None:
                continue
            self._world.set_cell_type(cpm, self._tag(active_type))
            self._cell_meta[bid]["type"] = active_type

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
                d = dead_entry(self._cache.get(bid) or self._cell_meta.get(bid, {}))
                positions[bid], sizes[bid], out_types[bid] = d["pos"], d["size"], d["type"]
                alive[bid], neighbors[bid] = d["alive"], d["neighbors"]
                shared[bid], footprints[bid] = d["shared"], d["footprint"]
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
            "lineage": dict(self._lineage),
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
