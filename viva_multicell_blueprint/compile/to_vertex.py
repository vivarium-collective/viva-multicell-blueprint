"""Compile a blueprint onto the vertex (vivarium-tyssue) mechanics backend (spec §5).

``compile(blueprint, backend="vertex")`` returns a composite document whose shared
mechanism (field/rules/ODE/immune) is byte-identical to the CPM compile and whose
single mechanics node is a :class:`VertexBackend`. The neutral type-pair
``adhesion`` is lowered to per-pair tyssue junction tension via Task-12's
``to_vertex_tension``; field chemotaxis is flattened to the per-field
``{field, type, strength}`` list the vertex backend turns into ActiveMigration
directors each step; immune types (and their latent pools) are declared IMMUNE
types -- smaller polygons attached to the margin of the same sheet as the
epithelium (an explicit ``vertex_overlay_types`` list moves types to a free
overlay layer instead); and the calibrated fluctuation/mobility knobs (:data:`CALIBRATED_DEFAULTS`)
are supplied unless the blueprint overrides them.
"""
from __future__ import annotations

from viva_multicell_blueprint.compile.adhesion import to_vertex_tension
from viva_multicell_blueprint.compile.base import INPUT_PORTS, OUTPUT_PORTS, latent_pool_types

from ._shared import _as_cell_nodes, assemble, mechanics_seed, resolve_initial_state

VERTEX_ADDR = "local:VertexBackend"


def _vertex_chemotaxis(blueprint) -> list[dict]:
    """Flatten each field's chemotaxis into [{field, type?, strength}] entries."""
    out = []
    for f in blueprint.fields or []:
        for c in f.get("chemotaxis") or []:
            entry = {"field": f["name"], "strength": float(c.get("strength", c.get("lambda", 1.0)))}
            if c.get("type") is not None:
                entry["type"] = c["type"]
            out.append(entry)
    return out


def _immune_types(blueprint) -> list[str]:
    """The recruited immune types plus their latent pools. On the vertex substrate
    they are members of the epithelial sheet (same plane, as in the CPM lattice):
    smaller polygons attached to the tissue margin that move through it by
    neighbour exchange."""
    imm = blueprint.immune if isinstance(blueprint.immune, dict) else {}
    active = [str(t) for t in (imm.get("types") or {}).values()]
    declared = {ct["name"] if isinstance(ct, dict) else str(ct) for ct in (blueprint.cell_types or [])}
    out = set(active)
    for t in active:
        out |= latent_pool_types(t) & declared
    return sorted(out)


def _overlay_types(blueprint) -> list[str]:
    """Cell types put on a free layer ABOVE the sheet (overlap contact) instead of
    in it: opt-in via ``mechanics_mapping.vertex_overlay_types``; none by default."""
    mm = blueprint.mechanics_mapping or {}
    return [str(t) for t in (mm.get("vertex_overlay_types") or [])]


# Vertex-only mechanics knobs; a blueprint may override any of them under
# ``mechanics_mapping.vertex`` (backend/run parameters, not canonical biology).
_VERTEX_KNOBS = ("migration_scale", "chemotaxis_saturation", "line_tension", "area_elasticity",
                 "viscosity", "max_displacement", "threshold_length", "auto_reconnect",
                 "tension_noise", "tension_noise_tau", "motility",
                 "sheet_motility", "motility_persistence", "sheet_motility_persistence", "substeps",
                 "perimeter_elasticity", "shape_index", "p_4", "p_5p", "seed",
                 "immune_area_scale", "immune_perimeter_elasticity")

# The vertex substrate's free fluctuation / mobility knobs, CALIBRATED against the
# CPM (temperature 10, neighbour order 2, size-25 cells, 2 MCS per step) on the Sego
# initial state by scripts/calibrate_vertex_mechanics.py; provenance and fit quality
# in workspace/studies/adhesion-calibration/results/vertex_mechanics_calibration.json
# (and docs/mechanics.md). Valid for target_size ~25 cells (viscosity = 2 K A0 = 100).
CALIBRATED_DEFAULTS = {
    "tension_scale": 1.683,          # Lambda_ab = scale * J_ab; mean cell area (CPM 23.0)
    "sheet_motility": 26.6,          # FOLD-FREE CAP: the MSD fit (81) folds the mesh, so the
                                     # epithelium wanders less than CPM (MSD 2.2 vs 9.5 um^2)
    "motility": 56.1,                # FOLD-FREE CAP of the in-sheet immune random walk: the
                                     # displacement fit (214) folds the mesh (4.6 vs CPM 9.4 um / 60 steps)
    "motility_persistence": 10.0,    # not fitted (polarity persistence, steps)
    "tension_noise": 5.0,            # largest fold-free step toward the CPM area CV (0.038 vs 0.093)
    "migration_scale": 3.54,         # chemotactic drift slope   } fitted to the CPM
    "chemotaxis_saturation": 105.0,  # chemotactic drift plateau } drift-vs-gradient curve
}


# Vertex representation choices (not fitted to the CPM; docs/mechanics.md section 2):
# * in-sheet immune polygons relax to half the blueprint size, so they are small
#   enough to squeeze along and between epithelial cells by neighbour exchange, as
#   a CPM immune cell does;
# * they carry a shape stiffness (P0 = shape_index * sqrt(A0)) -- a latent cell has
#   no tension, so it wets the tissue border as in the CPM, and without it the
#   polygon thins into a sliver that folds;
# * a 4-way vertex left by a T1 merge resolves at p_4 = 1 per unit time (tyssue's
#   0.1 left merged rosettes standing for ~10 steps, so neighbour exchange stalled).
REPRESENTATION_DEFAULTS = {
    "immune_area_scale": 0.5,
    "immune_perimeter_elasticity": 20.0,
    "p_4": 1.0,
}


def vertex_mechanics_node(blueprint) -> dict:
    cells = _as_cell_nodes(resolve_initial_state(blueprint))
    mm = blueprint.mechanics_mapping or {}

    backend_params = {
        "chemotaxis": _vertex_chemotaxis(blueprint),
        "overlay_types": _overlay_types(blueprint),
        "immune_types": _immune_types(blueprint),
    }
    backend_params["seed"] = mechanics_seed(blueprint)
    vertex_mm = dict(mm.get("vertex") or {})
    for key in _VERTEX_KNOBS:
        val = vertex_mm.get(key, mm.get(key, CALIBRATED_DEFAULTS.get(key, REPRESENTATION_DEFAULTS.get(key))))
        if val is not None:
            backend_params[key] = val

    if blueprint.adhesion:
        scale = float(mm.get("tension_scale", CALIBRATED_DEFAULTS["tension_scale"]))
        tensions = to_vertex_tension(blueprint.adhesion, scale=scale)
        # JSON-friendly keys ("a|b"); the full map is kept for provenance (the
        # adhesion-calibration study fits the J<->tension relationship).
        backend_params["tension_by_pair"] = {f"{a}|{b}": v for (a, b), v in tensions.items()}
        # a pair absent from the adhesion table is J = 0 on the CPM (viva-cpm's
        # default contact), so it is tension-free here too.
        backend_params.setdefault("line_tension", 0.0)

    return {
        "_type": "process",
        "address": VERTEX_ADDR,
        "config": {
            "initial_state": cells,
            "cell_types": [dict(ct) for ct in (blueprint.cell_types or [])],
            "adhesion": [dict(a) for a in (blueprint.adhesion or [])],
            "domain": dict(blueprint.domain or {}),
            "backend_params": backend_params,
        },
        "inputs": {p: [p] for p in INPUT_PORTS},
        "outputs": {p: [p] for p in OUTPUT_PORTS},
    }


def compile(blueprint, backend: str = "vertex", video_out=None) -> dict:
    if backend != "vertex":
        raise ValueError(f"to_vertex.compile only builds the vertex backend, got backend={backend!r}")
    return assemble(blueprint, vertex_mechanics_node(blueprint), video_out=video_out, backend="vertex")
