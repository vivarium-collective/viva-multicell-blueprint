"""Shared-identity registry (spec section 8).

Backend-agnostic: consumes plain identity info, never imports a backend engine.

Inputs
- ``cpm_map``: ``{blueprint_id: cpm_id}`` (exact 1:1 tag per cell).
- ``vertex_map``: ``{blueprint_id: tyssue_face_id}`` -- only mapped cells
  (``VertexBackend.identity``).
- ``vertex_residual``: ``VertexBackend.residual_report``; uses ``unmapped_ids``
  (blueprint ids, which already include contested cells; the single source for
  cohort exclusion).

Scoring rule (section 8): primary per-cell cohort = t0 clean shared set; one-sided
death of a cohort cell is a disagreement; division-born cells are scored by lineage
separately; a cell vanishing from one backend's bookkeeping (cause "vanished") is
flagged, counted, and excluded from the per-cell score.
"""
from __future__ import annotations

VANISHED = "vanished"
BACKENDS = ("cpm", "vertex")


class IdentityRegistry:
    def __init__(self, blueprint_ids, cpm_map, vertex_map, residual=None):
        self.blueprint_ids = [str(b) for b in blueprint_ids]
        self._cpm = {str(b): c for b, c in cpm_map.items()}
        self._face = {str(b): f for b, f in vertex_map.items()}
        self._cpm_rev = self._invert(self._cpm, "cpm_map")
        self._face_rev = self._invert(self._face, "vertex_map")
        self.residual = dict(residual or {})
        self._unmapped = {str(i) for i in self.residual.get("unmapped_ids", [])}
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
    def from_backends(cls, blueprint_ids, cpm_map, vertex_map, vertex_residual=None):
        return cls(blueprint_ids, cpm_map, vertex_map, vertex_residual)

    # -- maps ---------------------------------------------------------------- #
    def cpm_id(self, bid):
        return self._cpm.get(bid)

    def face_id(self, bid):
        return self._face.get(bid)

    def blueprint_from_cpm(self, cid):
        return self._cpm_rev.get(cid)

    def blueprint_from_face(self, fid):
        return self._face_rev.get(fid)

    # -- cohort -------------------------------------------------------------- #
    def t0_cohort(self) -> set:
        """Blueprint ids cleanly mapped in both backends (guard/unmapped/contested out)."""
        return {
            b for b in self.blueprint_ids
            if b in self._cpm and b in self._face
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
            d = out.setdefault(root, {b: 0 for b in BACKENDS})
            for b, kids in by_backend.items():
                d[b] += len(kids)
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

    @staticmethod
    def _check(backend):
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
