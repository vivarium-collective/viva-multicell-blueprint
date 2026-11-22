"""Engine-independent helpers shared by every MechanicsBackend implementation.

Each backend keeps its NATIVE application logic (how a type is set, a daughter is
spawned, an actuator applied); only the bits that do not touch an engine live here.
"""
from __future__ import annotations

from viva_multicell_blueprint.compile.base import (
    GENERIC_LATENT,
    latent_pool_types,
    no_footprint,
    sorted_cell_ids,
)


def event_cid(ev) -> str:
    """Cell id of a lineage event (``{"cell_id": id}`` dict or a bare id)."""
    return str(ev["cell_id"] if isinstance(ev, dict) else ev)


def mint_daughter_id(existing_ids, mother: str, counter: dict) -> str:
    """Collision-guarded ``<mother>.<k>`` daughter id.

    ``counter`` is the per-mother daughters-minted-so-far map (mutated);
    ``existing_ids`` is anything supporting ``in`` over ids already taken.
    """
    k = counter.get(mother, 0) + 1
    did = f"{mother}.{k}"
    while did in existing_ids:
        k += 1
        did = f"{mother}.{k}"
    counter[mother] = k
    return did


def latent_config(backend_params) -> tuple:
    """``(latent_types map, generic latent name)`` from ``backend_params``."""
    bp = backend_params or {}
    latent_map = {str(k): str(v) for k, v in (bp.get("latent_types") or {}).items()}
    return latent_map, str(bp.get("latent_generic", GENERIC_LATENT))


def select_activations(activations, ids, current_type, latent_map, generic, skip=None) -> list:
    """Pick which latent cells to activate (immune recruitment, Task 20).

    For each ``active_type -> count`` (sorted), take up to ``count`` cells -- in
    deterministic id order -- whose ``current_type(bid)`` is in that type's latent
    pool. A cell is consumed at most once per step. Returns ``[(cell_id,
    active_type), ...]``; the backend APPLIES them through its own set-type call.

    ``skip(active_type, pool)`` (optional) is consulted before selecting for a type;
    return True to skip it (the backend's unknown-type guard; the warning is emitted by the callback).
    """
    pairs: list = []
    if not activations:
        return pairs
    ordered = sorted_cell_ids(ids)
    consumed: set = set()
    for active_type in sorted(activations):
        count = int(activations.get(active_type, 0) or 0)
        if count <= 0:
            continue
        pool = latent_pool_types(active_type, latent_map, generic)
        if skip is not None and skip(active_type, pool):
            continue
        picked = 0
        for bid in ordered:
            if picked >= count:
                break
            if bid in consumed or current_type(bid) not in pool:
                continue
            pairs.append((bid, active_type))
            consumed.add(bid)
            picked += 1
    return pairs


def dead_entry(cached) -> dict:
    """Emit placeholder for a dead / vanished cell from its last-known ``cached``
    ``{pos, size, type}`` (missing keys fall back to origin / 0 / "")."""
    return {
        "pos": list(cached.get("pos", [0.0, 0.0])),
        "size": float(cached.get("size", 0.0)),
        "type": cached.get("type", ""),
        "alive": False,
        "neighbors": [],
        "shared": {},
        "footprint": no_footprint(),
    }
