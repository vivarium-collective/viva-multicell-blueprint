"""Compile a blueprint onto the vertex (vivarium-tyssue) mechanics backend (spec §5).

``compile(blueprint, backend="vertex")`` returns a composite document whose shared
mechanism (field/rules/ODE/immune) is byte-identical to the CPM compile and whose
single mechanics node is a :class:`VertexBackend`. The neutral type-pair
``adhesion`` is lowered to tyssue line tension via Task-12's ``to_vertex_tension``;
field chemotaxis is flattened to the per-field ``{field, type, strength}`` list the
vertex backend rewrites into ActiveMigration directors each step.
"""
from __future__ import annotations

from viva_multicell_blueprint.compile.adhesion import to_vertex_tension
from viva_multicell_blueprint.compile.base import INPUT_PORTS, OUTPUT_PORTS

from ._shared import _as_cell_nodes, assemble, resolve_initial_state

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


def vertex_mechanics_node(blueprint) -> dict:
    cells = _as_cell_nodes(resolve_initial_state(blueprint))
    mm = blueprint.mechanics_mapping or {}

    backend_params = {
        "chemotaxis": _vertex_chemotaxis(blueprint),
    }
    for key in ("migration_scale", "line_tension", "area_elasticity",
                "max_displacement", "guard_factor"):
        if mm.get(key) is not None:
            backend_params[key] = mm[key]

    if blueprint.adhesion:
        scale = float(mm.get("tension_scale", 1.0))
        tensions = to_vertex_tension(blueprint.adhesion, scale=scale)
        # JSON-friendly keys ("a|b"); the full map is kept for provenance (the
        # adhesion-calibration study fits the J<->tension relationship).
        backend_params["tension_by_pair"] = {f"{a}|{b}": v for (a, b), v in tensions.items()}
        backend_params.setdefault(
            "line_tension", sum(tensions.values()) / len(tensions) if tensions else 0.01)

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


def compile(blueprint, backend: str = "vertex") -> dict:
    if backend != "vertex":
        raise ValueError(f"to_vertex.compile only builds the vertex backend, got backend={backend!r}")
    return assemble(blueprint, vertex_mechanics_node(blueprint))
