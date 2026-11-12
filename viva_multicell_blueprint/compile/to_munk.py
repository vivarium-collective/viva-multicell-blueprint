"""Compile a blueprint onto the center-based (viva-munk) mechanics backend.

``compile(blueprint, backend="munk")`` returns a process-bigraph composite document
whose shared mechanism (field/rules/ODE/immune) is byte-identical to the CPM and
vertex compiles and whose single mechanics node is a :class:`MunkBackend` -- the
center-based paradigm (``pymunk`` circular bodies). As on the CPM side, field
chemotaxis is carried as type-name + strength so the center-based cells bias their
motion off the authoritative shared field; the field itself is never re-diffused
in the backend.
"""
from __future__ import annotations

from viva_multicell_blueprint.compile.base import INPUT_PORTS, OUTPUT_PORTS

from ._shared import (
    _as_cell_nodes,
    assemble,
    mechanics_seed,
    resolve_initial_state,
)

MUNK_ADDR = "local:MunkBackend"


def _munk_fields(blueprint) -> list[dict]:
    """Field entries for the center-based backend: name + chemotaxis (type NAMES +
    strength), mirroring the CPM lowering so one blueprint feeds every paradigm."""
    out = []
    for f in blueprint.fields or []:
        out.append({
            "name": f["name"],
            "chemotaxis": [dict(c) for c in (f.get("chemotaxis") or [])],
        })
    return out


def munk_mechanics_node(blueprint) -> dict:
    cells = _as_cell_nodes(resolve_initial_state(blueprint))
    mm = blueprint.mechanics_mapping or {}

    backend_params = {
        "substeps": int(mm.get("substeps", 10)),
        # momentum persists across steps (was 0.1, which bled all motion away in one
        # step and froze the cohort); below 1.0 so it stays stable.
        "damping_per_second": float(mm.get("damping_per_second", 0.85)),
        # jitter (active-Brownian wobble) follows the run seed so replicate runs
        # differ in mechanics, not only in rule draws. Default ON so passive
        # center-based cells actually explore + rearrange; a blueprint can still
        # set it to 0 for a frozen control.
        "jitter_per_second": float(mm.get("jitter_per_second", 50.0)),
        "friction": float(mm.get("friction", 0.8)),
        "elasticity": float(mm.get("elasticity", 0.0)),
        "contact_eps": float(mm.get("contact_eps", 0.75)),
        # cell-cell adhesion coupling off the blueprint adhesion matrix (the knobs
        # the backend turns the matrix into pairwise forces with) + the speed clamp
        # that keeps the whole thing stable, and the chemotaxis unit conversion.
        "adhesion_range": float(mm.get("adhesion_range", 3.0)),
        "adhesion_velocity_scale": float(mm.get("adhesion_velocity_scale", 0.02)),
        "max_speed": float(mm.get("max_speed", 2.0)),
        "chemotaxis_gain": float(mm.get("chemotaxis_gain", 0.08)),
        "fields": _munk_fields(blueprint),
        "mechanics_seed": mechanics_seed(blueprint),
    }

    return {
        "_type": "process",
        "address": MUNK_ADDR,
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


def compile(blueprint, backend: str = "munk", video_out=None) -> dict:
    if backend != "munk":
        raise ValueError(f"to_munk.compile only builds the munk backend, got backend={backend!r}")
    return assemble(blueprint, munk_mechanics_node(blueprint), video_out=video_out, backend="munk")
