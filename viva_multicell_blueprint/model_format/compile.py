"""Task 7 — assemble a scene + registry into ONE runnable process-bigraph composite.

``compile_scene`` is the integration point for the model-definition format: it binds
each scene cell to its per-type :class:`~..registry.ModelSpec` (Tasks 1-3), chooses an
intracellular adapter per cell by ``ModelSpec.formalism`` (Task 4), reconciles the
shared extracellular field + membrane contract (Task 6), and wires the whole thing to
this repo's tissue-mechanics backend (``..compile.compile``).

Rather than emit a literal process-bigraph node per cell, the composite holds a single
orchestrating :class:`SceneModelProcess` (mirroring
:class:`~...mechanism.bounded_particles.BoundedParticlesProcess`): it builds the
mechanics backend from the scene, keeps one adapter per cell and one shared
:class:`~..extracellular.FieldExtracellular`, and steps them all internally each
``update()`` under the membrane contract. One process, many cells — the same shape
``bounded_particles`` uses to confine a Smoldyn sim inside each moving tyssue cell.

Error timing contract:
  * an **unbound** cell type raises :class:`~..registry.UnboundCellType` at *compile*
    (``compile_scene`` calls :func:`~..binding.bind` before building the document);
  * a bound cell whose permeable (cross/transporter) species has no home in the
    extracellular field raises :class:`~..membrane.NoExtracellularHome` at *compile*;
  * an unavailable *engine* (tellurium / Smoldyn / MaBoSS missing) does NOT error —
    the adapter degrades to holding its levels, so the composite still runs.
"""
from __future__ import annotations

import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

from process_bigraph import Process

from .adapters.base import IntracellularAdapter
from .adapters.boolean_maboss import BooleanAdapter
from .adapters.coupled import CoupledCell
from .adapters.ode_tellurium import ODEAdapter
from .adapters.particle_smoldyn import ParticleAdapter
from .binding import bind
from .extracellular import FieldExtracellular
from .membrane import PERMEABLE, NoExtracellularHome
from .membrane import apply as membrane_apply
from .registry import ModelSpec, Registry
from .scene import Scene

# formalism token (incl. obvious engine aliases) -> adapter class
_ADAPTER_BY_FORMALISM: dict[str, type[IntracellularAdapter]] = {
    "ode": ODEAdapter, "sbml": ODEAdapter, "tellurium": ODEAdapter,
    "particle": ParticleAdapter, "particles": ParticleAdapter, "smoldyn": ParticleAdapter,
    "boolean": BooleanAdapter, "bool": BooleanAdapter, "maboss": BooleanAdapter,
    "coupled": CoupledCell, "coupled_cell": CoupledCell,
}


def _adapter_class(formalism: str) -> type[IntracellularAdapter]:
    key = str(formalism).strip().lower()
    try:
        return _ADAPTER_BY_FORMALISM[key]
    except KeyError:
        raise ValueError(
            f"unknown formalism {formalism!r}; "
            f"have {sorted(set(_ADAPTER_BY_FORMALISM))}"
        ) from None


def _square(pos, half: float = 4.0) -> list[list[float]]:
    """A static axis-aligned square polygon centred on ``pos`` (mechanics fallback)."""
    x, y = float(pos[0]), float(pos[1])
    return [[x - half, y - half], [x + half, y - half],
            [x + half, y + half], [x - half, y + half]]


def _extent(scene: Scene) -> tuple[float, float]:
    d = scene.domain or {}
    for key in ("extent_um", "size", "extent"):
        v = d.get(key)
        if v and len(v) >= 2:
            return float(v[0]), float(v[1])
    xs = [c.position[0] for c in scene.cells.values()] or [10.0]
    ys = [c.position[1] for c in scene.cells.values()] or [10.0]
    return float(max(xs)) + 10.0, float(max(ys)) + 10.0


def _permeable_species(bound: dict[str, ModelSpec]) -> set[str]:
    out: set[str] = set()
    for spec in bound.values():
        for sp, tr in (spec.membrane or {}).items():
            if tr.kind in PERMEABLE:
                out.add(sp)
    return out


def _build_default_field(scene: Scene, bound: dict[str, ModelSpec]) -> FieldExtracellular:
    """A field covering every permeable species (one diffusing grid each)."""
    species = sorted(_permeable_species(bound))
    w, h = _extent(scene)
    nx, ny = int(max(4, round(w))), int(max(4, round(h)))
    return FieldExtracellular(shape=(ny, nx), species=species, extent=(w, h))


def _footprints(fp_dict: dict) -> dict[str, list[list[float]]]:
    """Mechanics ``_emit()`` footprints -> ordered polygons keyed by (str) cell id.

    Vertex/munk already give ordered polygons; CPM gives occupancy voxels that need
    tracing to an outline (same handling as ``bounded_particles``).
    """
    out: dict[str, list[list[float]]] = {}
    for bid, fp in (fp_dict or {}).items():
        if not isinstance(fp, dict):
            continue
        xy = fp.get("xy")
        if not xy:
            continue
        if fp.get("kind") == "occupancy":
            from ..particles.smoldyn_cell import occupancy_outline
            poly = occupancy_outline(xy)
        else:
            poly = [[float(x), float(y)] for x, y in xy] if len(xy) >= 3 else None
        if poly and len(poly) >= 3:
            out[str(bid)] = poly
    return out


class SceneModelProcess(Process):
    """One orchestrating process for a whole scene.

    Holds the tissue mechanics backend + one :class:`IntracellularAdapter` per cell
    (by ``ModelSpec.formalism``) + one shared :class:`FieldExtracellular`. Each
    ``update()`` advances the mechanics, reads every cell's footprint/position, steps
    each adapter (particle cells get their polygon), and routes any crossed molecules
    into the field via the membrane contract.

    Config carries live Python objects (scene/registry/field) opaquely via the
    ``quote`` schema — the document is built by :func:`compile_scene`, never
    round-tripped through JSON.
    """

    config_schema = {
        "scene": "quote",
        "registry": "quote",
        "mechanics": {"_type": "string", "_default": "vertex"},
        "extracellular": "quote",
        "intracellular_overrides": "quote",
    }

    def __init__(self, config=None, core=None):
        super().__init__(config, core)
        scene: Scene = self.config["scene"]
        registry: Registry = self.config["registry"]
        overrides = self.config.get("intracellular_overrides") or {}
        self._scene = scene
        self._bound = bind(scene, registry)  # raises UnboundCellType for an unknown type
        self._backend_name = self.config.get("mechanics") or "vertex"
        self._cell_ids = sorted(scene.cells)
        # deterministic scene id (str) -> mechanics id (str "1".."N") in sorted order
        self._id_map = {sid: str(i + 1) for i, sid in enumerate(self._cell_ids)}

        self._mech = None
        self._mech_error: str | None = None
        self._static_fp: dict[str, list[list[float]]] = {}
        self._static_pos: dict[str, list[float]] = {}
        self._build_mechanics(scene)

        self._field: FieldExtracellular | None = self.config.get("extracellular")
        if self._field is None:
            self._field = _build_default_field(scene, self._bound)

        fp, _pos = self._mechanics_emit()
        self._adapters: dict[str, IntracellularAdapter] = {}
        for seed, sid in enumerate(self._cell_ids):
            spec = self._bound[sid]
            cell = scene.cells[sid]
            formalism = overrides.get(cell.type, spec.formalism)
            cls = _adapter_class(formalism)
            levels = dict(cell.molecule_levels)
            if cls is ParticleAdapter:
                poly = fp.get(sid) or _square(cell.position)
                spec_poly = replace(spec, params={**(spec.params or {}), "polygon": poly})
                self._adapters[sid] = cls(spec_poly, levels, seed=seed)
            else:
                self._adapters[sid] = cls(spec, levels)
        self._last_amounts = {sid: self._adapters[sid].amounts() for sid in self._cell_ids}

    # ---- mechanics backend ---------------------------------------------------
    def _build_mechanics(self, scene: Scene) -> None:
        """Build the tissue mechanics backend from the scene (real backend preferred).

        Writes a minimal cell table (ids = the scene->mechanics map) and compiles it
        onto ``self._backend_name``. If the backend cannot be built (engine missing or
        a compile error), fall back to a static square polygon per cell derived from
        its position — no tissue dynamics, but the composite still runs.
        """
        tmp_path: Path | None = None
        try:
            from process_bigraph import Composite

            from ..blueprint.model import Blueprint
            from ..compile import compile as _compile
            from ..core import build_core

            types = sorted({c.type for c in scene.cells.values()})
            w, h = _extent(scene)
            tmp = tempfile.NamedTemporaryFile(
                "w", suffix=".csv", delete=False, newline="", encoding="utf-8")
            tmp_path = Path(tmp.name)
            tmp.write("id,type,x,y,size\n")
            for sid in self._cell_ids:
                c = scene.cells[sid]
                tmp.write(f"{self._id_map[sid]},{c.type},"
                          f"{float(c.position[0])},{float(c.position[1])},25.0\n")
            tmp.close()

            bp = Blueprint.from_dict({
                "blueprint_version": "1.0",
                "domain": {"extent_um": [w, h], "grid_resolution_um": 1, "boundary": "noflux"},
                "cell_types": [{"name": t, "target_size": 25} for t in types],
                "initial_state": {"from_data": {"cell_table": str(tmp_path)}},
            })
            doc = _compile(bp, backend=self._backend_name)
            self._mech = Composite(
                {"state": doc["state"]}, core=build_core()).state["mechanics"]["instance"]
        except Exception as exc:  # pragma: no cover - env-dependent fallback
            self._mech = None
            self._mech_error = repr(exc)
            for sid in self._cell_ids:
                pos = scene.cells[sid].position
                self._static_pos[sid] = [float(pos[0]), float(pos[1])]
                self._static_fp[sid] = _square(pos)
        finally:
            if tmp_path is not None:
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

    def _mechanics_emit(self) -> tuple[dict[str, list[list[float]]], dict[str, list[float]]]:
        """Per-scene-cell (polygon, position), from the real backend or the fallback."""
        if self._mech is None:
            return dict(self._static_fp), dict(self._static_pos)
        try:
            g = self._mech._emit() or {}
        except Exception as exc:
            # Record (don't swallow): a backend emit failure must be observable
            # even though we fall back to static squares for this step.
            self._mech_error = f"emit: {exc!r}"
            g = {}
        fp_by_mech = _footprints(g.get("footprints", {}))
        pos_by_mech = g.get("positions", {}) or {}
        fp, pos = {}, {}
        for sid in self._cell_ids:
            mid = self._id_map[sid]
            fp[sid] = fp_by_mech.get(mid) or _square(self._scene.cells[sid].position)
            p = pos_by_mech.get(mid)
            pos[sid] = ([float(p[0]), float(p[1])] if p
                        else [float(v) for v in self._scene.cells[sid].position])
        return fp, pos

    def _advance_mechanics(self) -> None:
        if self._mech is None:
            return
        try:
            self._mech.update({"fates": {}, "field_grid": {}, "divisions": []}, 1.0)
        except Exception as exc:
            # Record (don't swallow): a runtime mechanics-step failure must be
            # observable via self._mech_error, not masked by the static fallback.
            self._mech_error = f"advance: {exc!r}"

    # ---- process-bigraph contract -------------------------------------------
    def inputs(self):
        return {}

    def outputs(self):
        return {"n_cells": "overwrite[integer]", "total_molecules": "overwrite[float]"}

    def update(self, inputs, interval=1):
        self._advance_mechanics()
        fp, pos = self._mechanics_emit()
        dt = float(interval)
        total = 0.0
        for sid in self._cell_ids:
            adapter = self._adapters[sid]
            membrane_spec = self._bound[sid].membrane or {}
            cell_pos = pos.get(sid) or [float(v) for v in self._scene.cells[sid].position]

            # read extracellular concentrations (validates the membrane contract)
            uptake: dict[str, Any] = (
                membrane_apply({}, cell_pos, self._field, membrane_spec)
                if membrane_spec else {})
            # inject the cell polygon unconditionally: a ParticleAdapter (or a
            # CoupledCell holding a particle sub-model) confines to it; ODE/Boolean
            # adapters ignore the key.
            step_inputs = dict(uptake)
            step_inputs["polygon"] = fp.get(sid)

            adapter.step(dt, step_inputs)

            # NOTE: adapters return crossed() == {} this milestone, so the shared
            # field stays inert here until per-adapter efflux lands (future step).
            crossed = adapter.crossed()
            if crossed and membrane_spec:
                membrane_apply(crossed, cell_pos, self._field, membrane_spec)

            total += sum(adapter.amounts().values())

        if self._field is not None:
            self._field.diffuse()
        self._last_amounts = {sid: self._adapters[sid].amounts() for sid in self._cell_ids}
        return {"n_cells": len(self._cell_ids), "total_molecules": float(total)}


def compile_scene(
    scene: Scene,
    registry: Registry,
    *,
    mechanics: str = "vertex",
    extracellular: FieldExtracellular | None = None,
    intracellular_overrides: dict[str, str] | None = None,
) -> dict:
    """Assemble ``scene`` + ``registry`` into one runnable composite document.

    Returns ``{"state": {...}}`` with a single ``SceneModelProcess`` node. Build it
    with ``Composite({"state": doc["state"]}, core=build_core())`` and ``run(n)``.

    Parameters
    ----------
    mechanics:
        Tissue backend name (``"vertex"`` | ``"cpm"`` | ``"munk"``).
    extracellular:
        A prebuilt :class:`FieldExtracellular`; if ``None``, one covering every
        permeable species is built from the scene.
    intracellular_overrides:
        ``{cell_type: formalism}`` forcing a type onto a specific adapter.

    Raises
    ------
    UnboundCellType
        at compile, if a scene cell's type has no ModelSpec in ``registry``.
    NoExtracellularHome
        at compile, if a bound cell declares a cross/transporter species with no
        home in the (given or default) extracellular field.
    """
    bound = bind(scene, registry)  # compile-time: unbound type raises here

    field = extracellular if extracellular is not None else _build_default_field(scene, bound)
    for cid, spec in bound.items():
        for sp, tr in (spec.membrane or {}).items():
            if tr.kind in PERMEABLE and not field.has(sp):
                raise NoExtracellularHome(
                    f"cell {cid!r} species {sp!r} is {tr.kind!r} but the extracellular "
                    f"field has no home for it")

    return {"state": {
        "scene_model": {
            "_type": "process",
            "address": "local:SceneModelProcess",
            "config": {
                "scene": scene,
                "registry": registry,
                "mechanics": mechanics,
                "extracellular": field,
                "intracellular_overrides": dict(intracellular_overrides or {}),
            },
            "inputs": {},
            "outputs": {"n_cells": ["n_cells"], "total_molecules": ["total_molecules"]},
        },
        "n_cells": 0,
        "total_molecules": 0.0,
    }}


__all__ = ["compile_scene", "SceneModelProcess"]
