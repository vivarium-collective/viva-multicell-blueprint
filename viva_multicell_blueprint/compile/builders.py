"""Per-concern pipeline builder units.

Each builder emits ONE part of the blueprint composite document (a node spec),
or ``None`` / ``[]`` when its blueprint config slice is absent. They delegate to
the existing ``_shared`` / ``to_*`` node builders; wiring them into a composite
is the pipeline's job.
"""
from __future__ import annotations

from ._shared import (
    _counts_node,
    _field_node,
    _immune_node,
    _ode_node,
    _rules_node,
    _video_node,
)

MERGE_ADDR = "local:CellActuationMergeStep"
SUBCELLULAR_ADDR = "local:SubcellularProcess"
_BACKENDS = ("cpm", "vertex", "munk")


def build_mechanics(blueprint, backend: str) -> dict:
    """The backend-specific mechanics node (incl. adhesion lowering)."""
    if backend == "cpm":
        from .to_cpm import cpm_mechanics_node as fn
    elif backend == "vertex":
        from .to_vertex import vertex_mechanics_node as fn
    elif backend == "munk":
        from .to_munk import munk_mechanics_node as fn
    else:
        raise ValueError(f"unknown backend {backend!r}; expected one of {sorted(_BACKENDS)}")
    return fn(blueprint)


def build_fields(blueprint) -> dict | None:
    return _field_node(blueprint) if blueprint.fields else None


def build_rules(blueprint) -> dict | None:
    return _rules_node(blueprint) if blueprint.rules else None


def build_systemic(blueprint) -> dict | None:
    """``{"ode": SystemicODEProcess node, "counts": CellCountStep node}``."""
    if blueprint.systemic_ode is None:
        return None
    return {"ode": _ode_node(blueprint), "counts": _counts_node(blueprint)}


def build_immune(blueprint) -> dict | None:
    return _immune_node(blueprint) if blueprint.immune else None


def build_subcellular(blueprint) -> list[dict]:
    """One SubcellularProcess node per ``blueprint.subcellular`` entry."""
    sub = blueprint.subcellular or {}
    names = {ct["name"] if isinstance(ct, dict) else str(ct) for ct in (blueprint.cell_types or [])}
    nodes = []
    for ctype, entry in sub.items():
        if ctype not in names:
            raise ValueError(
                f"subcellular entry for cell type {ctype!r}, which is not in cell_types "
                f"{sorted(names)}"
            )
        nodes.append({
            "_type": "process",
            "address": SUBCELLULAR_ADDR,
            "config": {**(entry or {}), "cell_type": ctype},
        })
    return nodes


def build_merge(blueprint) -> dict | None:
    """CellActuationMergeStep, present iff the blueprint has subcellular models."""
    if not blueprint.subcellular:
        return None
    arb = (blueprint.mechanics_mapping or {}).get("arbitration", "rules_first")
    return {"_type": "step", "address": MERGE_ADDR, "config": {"arbitration": arb}}


def build_video(blueprint, video_out, backend: str = "cpm") -> dict | None:
    return _video_node(blueprint, backend, video_out) if video_out else None
