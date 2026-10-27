"""Compile a blueprint onto the CPM (viva-cpm) mechanics backend (spec section 5).

``compile(blueprint, backend="cpm")`` returns a process-bigraph composite document
whose shared mechanism (field/rules/ODE/immune) is byte-identical to the vertex
compile and whose single mechanics node is a :class:`CPMBackend`. The neutral
type-pair ``adhesion`` is lowered to the CPM contact (J) matrix via Task-12's
``to_cpm_contact``, with an EXPLICIT ``type_ids`` map derived from the blueprint
``cell_types`` declaration order so the integer tags are stable.
"""
from __future__ import annotations

from viva_multicell_blueprint.compile.adhesion import to_cpm_contact
from viva_multicell_blueprint.compile.base import INPUT_PORTS, OUTPUT_PORTS

from ._shared import (
    _as_cell_nodes,
    assemble,
    field_names,
    resolve_initial_state,
    type_ids_from_cell_types,
)

CPM_ADDR = "local:CPMBackend"


def _cpm_fields(blueprint) -> list[dict]:
    """Field entries for the CPM World: name + chemotaxis (type NAMES + strength).

    Native field DYNAMICS are zeroed inside the backend (it only READS the shared
    grid written via set_field); only the chemotaxis wiring is carried so CPM biases
    its Metropolis flips off the authoritative shared field.
    """
    out = []
    for f in blueprint.fields or []:
        out.append({
            "name": f["name"],
            "chemotaxis": [dict(c) for c in (f.get("chemotaxis") or [])],
        })
    return out


def cpm_mechanics_node(blueprint) -> dict:
    cells = _as_cell_nodes(resolve_initial_state(blueprint))
    type_ids = type_ids_from_cell_types(blueprint.cell_types)
    mm = blueprint.mechanics_mapping or {}

    backend_params = {
        "mcs_per_update": int(mm.get("mcs_per_update", 2)),
        "type_ids": type_ids,
        "fields": _cpm_fields(blueprint),
    }
    if blueprint.adhesion:
        backend_params["contact"] = to_cpm_contact(blueprint.adhesion, dict(type_ids))

    return {
        "_type": "process",
        "address": CPM_ADDR,
        "config": {
            "initial_state": cells,
            "cell_types": [dict(ct) for ct in (blueprint.cell_types or [])],
            "adhesion": [dict(a) for a in (blueprint.adhesion or [])],
            "domain": dict(blueprint.domain or {}),
            "backend_params": backend_params,
        },
        # the mechanics node reads the rule/field outputs and writes the shared
        # geometry stores the field + rules read back.
        "inputs": {p: [p] for p in INPUT_PORTS},
        "outputs": {p: [p] for p in OUTPUT_PORTS},
    }


def compile(blueprint, backend: str = "cpm") -> dict:
    if backend != "cpm":
        raise ValueError(f"to_cpm.compile only builds the cpm backend, got backend={backend!r}")
    return assemble(blueprint, cpm_mechanics_node(blueprint))
