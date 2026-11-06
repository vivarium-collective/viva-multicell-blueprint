"""Shared-identity registry (spec section 8).

Backend-agnostic: consumes plain identity info, never imports a backend engine.
Generalized to **N paradigms** (CPM, vertex, center-based/munk, ...) -- the cohort
is the set of cells cleanly mapped in EVERY registered backend, and the comparison
statistics run PAIRWISE over backend pairs.

Inputs (preferred, N-ary)
- ``maps``: ``{backend: {blueprint_id: native_id}}`` -- one 1:1 map per backend
  (CPM tag / tyssue face id / munk body id). Only mapped cells need appear.
- ``residuals``: ``{backend: residual_dict}`` -- each backend's ``residual_report``;
  its ``unmapped_ids`` (blueprint ids: guard/contested/dropped cells) are unioned
  across backends and excluded from the shared cohort. CPM and munk are exact 1:1
  (no unmapped); vertex may drop cells on an imperfect mesh assignment.

Legacy 2-backend input (``from_backends(ids, cpm_map, vertex_map, vertex_residual)``)
is still supported as a thin shim over the N-ary constructor.

Scoring rule (section 8): primary per-cell cohort = t0 clean shared set; one-sided
death of a cohort cell is a disagreement; division-born cells are scored by lineage
separately; a cell vanishing from one backend's bookkeeping (cause "vanished") is
flagged, counted, and excluded from the per-cell score.
"""
from __future__ import annotations

VANISHED = "vanished"
BACKENDS = ("cpm", "vertex")   # legacy default pair (the registry itself is N-ary)


class IdentityRegistry:
    def __init__(self, blueprint_ids, maps, residuals=None):
        self.blueprint_ids = [str(b) for b in blueprint_ids]
        self.backends = tuple(maps.keys())
        self._maps = {bk: {str(b): v for b, v in (m or {}).items()} for bk, m in maps.items()}
        self._rev = {bk: self._invert(m, f"{bk}_map") for bk, m in self._maps.items()}
        self.residuals = {bk: dict(r or {}) for bk, r in (residuals or {}).items()}
        self._unmapped = set()
        for r in self.residuals.values():
            self._unmapped |= {str(i) for i in r.get("unmapped_ids", [])}
        # back-compat: vertex was historically the only backend with a residual.
        self.residual = self.residuals.get("vertex", {})
        self._parent: dict[str, str] = {}
        self._children: dict[str, dict[str, set]] = {}
        self._losses: dict[str, dict[str, str]] = {}  # id -> {backend: cause}

    @staticmethod
    def _invert(m, name):
        rev = {}
        for b, v in m.items():
            if v in rev:
                raise ValueError(
                    f"{name} is not 1:1: value {v!r} maps from both {rev[v]!r} and {b!r}"
                )
            rev[v] = b
        return rev

    @classmethod
    def from_maps(cls, blueprint_ids, maps, residuals=None):
        """N-ary constructor: ``maps`` / ``residuals`` keyed by backend name."""
        return cls(blueprint_ids, maps, residuals)

    @classmethod
    def from_backends(cls, blueprint_ids, cpm_map, vertex_map, vertex_residual=None):
        """Legacy 2-backend shim (CPM + vertex)."""
        return cls(blueprint_ids, {"cpm": cpm_map, "vertex": vertex_map},
                   {"vertex": vertex_residual or {}})

    # -- maps (generic + legacy accessors) ----------------------------------- #
    def native_id(self, backend, bid):
        return self._maps.get(backend, {}).get(bid)

    def blueprint_from(self, backend, native):
        return self._rev.get(backend, {}).get(native)

    def cpm_id(self, bid):
        return self.native_id("cpm", bid)

    def face_id(self, bid):
        return self.native_id("vertex", bid)

    def blueprint_from_cpm(self, cid):
        return self.blueprint_from("cpm", cid)

    def blueprint_from_face(self, fid):
        return self.blueprint_from("vertex", fid)

    # -- cohort -------------------------------------------------------------- #
    def t0_cohort(self) -> set:
        """Blueprint ids cleanly mapped in EVERY registered backend (unmapped out)."""
        return {
            b for b in self.blueprint_ids
            if all(b in self._maps[bk] for bk in self.backends)
            and b not in self._unmapped
        }

    # -- lineage ------------------------------------------------------------- #
    def on_division(self, parent_id, child_id, backend):
        self._check(backend)
        parent_id, child_id = str(parent_id), str(child_id)
        self._parent[child_id] = parent_id
        self._children.setdefault(parent_id, {}).setdefault(backend, set()).add(child_id)

    def parent_of(self, cid):
        return self._parent.get(cid)

    def root_of(self, cid):
        while cid in self._parent:
            cid = self._parent[cid]
        return cid

    def lineage_children(self, parent_id):
        return {b: set(s) for b, s in self._children.get(parent_id, {}).items()}

    def born_cells(self) -> set:
        return set(self._parent)

    def lineage_counts(self) -> dict:
        """Per t0 root: {backend: number of division-born descendants}. Separate metric."""
        out: dict[str, dict[str, int]] = {}
        for parent, by_backend in self._children.items():
            root = self.root_of(parent)
            d = out.setdefault(root, {b: 0 for b in self.backends})
            for b, kids in by_backend.items():
                d[b] = d.get(b, 0) + len(kids)
        return out

    # -- loss ---------------------------------------------------------------- #
    def on_loss(self, cell_id, backend, cause):
        self._check(backend)
        self._losses.setdefault(str(cell_id), {})[backend] = cause

    def vanished(self) -> dict:
        """{id: backend} for cells that vanished from one backend's bookkeeping."""
        return {
            i: next(b for b, c in l.items() if c == VANISHED)
            for i, l in self._losses.items() if VANISHED in l.values()
        }

    def n_vanished(self) -> int:
        return len(self.vanished())

    def disagreements(self) -> dict:
        """Cohort cells dead on exactly one side (real death, not vanished): {id: (backend, cause)}."""
        cohort, out = self.t0_cohort(), {}
        for i, l in self._losses.items():
            if i not in cohort or VANISHED in l.values() or len(l) != 1:
                continue
            (b, cause), = l.items()
            out[i] = (b, cause)
        return out

    def per_cell_scored_ids(self) -> set:
        """Primary per-cell cohort: t0 shared set minus vanished cells."""
        return self.t0_cohort() - set(self.vanished())

    def _check(self, backend):
        if backend not in self.backends:
            raise ValueError(f"backend must be one of {self.backends}, got {backend!r}")
