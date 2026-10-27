"""Shared compiler internals (spec section 5): the ONE authoritative mechanism.

The compiler lowers a backend-neutral :class:`Blueprint` onto each mechanical
substrate. Everything in this module is **backend-independent** — it builds the
shared ``FieldProcess`` / ``RulesProcess`` / ``SystemicODEProcess`` /
``ImmuneRecruitmentProcess`` nodes and the shared stores they read and write.
``to_cpm`` / ``to_vertex`` add exactly ONE more node (the mechanics backend) and
nothing else, so the two compiled composites are byte-identical everywhere except
that single node (the invariant the tests assert).

Addresses use the ``local:<ClassName>`` form; the classes are registered into the
workspace core by :func:`viva_multicell_blueprint.core.build_core` (the
``backends`` / ``mechanism`` subpackages export them so the core's registration
walk finds them).
"""
from __future__ import annotations

import math

from viva_multicell_blueprint.blueprint.data_loading import from_cell_table, from_label_image

MEDIUM = "medium"

# --- process addresses (resolved via build_core's link registry) --------------
FIELD_ADDR = "local:FieldProcess"
RULES_ADDR = "local:RulesProcess"
ODE_ADDR = "local:SystemicODEProcess"
IMMUNE_ADDR = "local:ImmuneRecruitmentProcess"

# Rule kinds RulesProcess (Task 7) accepts. The blueprint ``rules`` section is
# untyped (the JSON schema only enumerates the structural-declarative kinds); the
# validation hook below normalizes each rule into this closed set, failing loud on
# anything else and on a missing rate_expr.
_RULE_KINDS = {"transition", "allee", "recovery", "kill", "resistance", "divide"}
# Fields carried through onto the RulesProcess rule dict verbatim when present.
_RULE_PASSTHROUGH = ("from", "to", "target", "killer", "scalar", "remove", "types")


# --------------------------------------------------------------------------- #
# Rule validation / normalization hook
# --------------------------------------------------------------------------- #
def normalize_rules(rules) -> list[dict]:
    """Validate + normalize the untyped blueprint ``rules`` into RulesProcess shape.

    The blueprint uses ``type`` for the rule kind (JSON-schema enum); RulesProcess
    reads ``kind``. This hook maps ``type`` -> ``kind``, keeps the source/target/
    scalar/remove fields, and requires a ``rate_expr``. Fails loud (``ValueError``)
    on an unknown kind or a missing rate_expr so a malformed blueprint never
    reaches the engine as a silent no-op.
    """
    out: list[dict] = []
    for i, r in enumerate(rules or []):
        kind = r.get("kind", r.get("type"))
        if kind not in _RULE_KINDS:
            raise ValueError(
                f"rule[{i}]: unknown rule kind {kind!r} "
                f"(expected one of {sorted(_RULE_KINDS)})"
            )
        expr = r.get("rate_expr")
        if not isinstance(expr, str) or not expr.strip():
            raise ValueError(f"rule[{i}] ({kind}): missing/empty rate_expr")
        norm = {"kind": kind, "rate_expr": expr}
        for key in _RULE_PASSTHROUGH:
            if r.get(key) is not None:
                norm[key] = r[key]
        out.append(norm)
    return out


# --------------------------------------------------------------------------- #
# Structural helpers (type tags, grid, fields, initial state)
# --------------------------------------------------------------------------- #
def type_ids_from_cell_types(cell_types) -> dict:
    """Name -> integer tag from the blueprint ``cell_types`` DECLARATION ORDER.

    ``medium`` is tag 0 (the CPM background); the declared cell types follow 1..
    in order. Deriving the map from the declared order (not first-appearance in
    the cell list) makes CPM's integer tags STABLE across datasets/runs — the
    Task 12 contract for ``to_cpm_contact``.
    """
    tids = {MEDIUM: 0}
    for ct in cell_types or []:
        name = ct["name"] if isinstance(ct, dict) else str(ct)
        if name not in tids:
            tids[name] = len(tids)
    return tids


def grid_shape_from_domain(domain) -> list[int]:
    """Shared field grid shape [nx, ny(, nz)] = ceil(extent / resolution)."""
    domain = domain or {}
    extent = domain.get("extent_um")
    res = domain.get("grid_resolution_um")
    if not extent or not res:
        raise ValueError("domain needs extent_um + grid_resolution_um to size the field grid")
    return [max(1, int(math.ceil(float(e) / float(res)))) for e in extent]


def field_process_fields(fields) -> list[dict]:
    """Map blueprint fields (``D``) onto the FieldProcess config shape (``d``).

    Secretion/uptake pass through unchanged ({type, rate}); chemotaxis is backend
    (mechanics) metadata, so it is NOT carried into the field engine's config.
    """
    out = []
    for f in fields or []:
        out.append({
            "name": f["name"],
            "d": float(f.get("D", f.get("d", 0.0))),
            "decay": float(f.get("decay", 0.0)),
            "secretion": [dict(s) for s in (f.get("secretion") or [])],
            "uptake": [dict(u) for u in (f.get("uptake") or [])],
            "dynamics": dict(f.get("dynamics") or {}),
        })
    return out


def field_names(fields) -> list[str]:
    return [f["name"] for f in (fields or [])]


def resolve_initial_state(blueprint) -> list[dict]:
    """Produce the canonical cell list ``[{id, type, centroid, size}]``.

    Supports ``from_data`` (cell table / label image, via the Task-2 data loader)
    and an explicit ``cells`` list (for blueprints built in-memory). Generative
    ``kind: blocks|voronoi`` is not a Task-13 concern — it fails loud so a caller
    knows to supply data rather than get a silent empty tissue.
    """
    init = blueprint.initial_state or {}
    if init.get("cells"):
        return [dict(c) for c in init["cells"]]
    from_data = init.get("from_data")
    if from_data:
        if "cell_table" in from_data:
            return from_cell_table(from_data["cell_table"])
        if "label_image" in from_data:
            return from_label_image(from_data["label_image"], from_data.get("type_map"))
        raise ValueError(f"from_data needs cell_table or label_image, got {sorted(from_data)}")
    if init.get("kind") == "blocks" and isinstance(init.get("block"), dict):
        return _generate_blocks(init)
    raise ValueError(
        f"cannot resolve initial_state {sorted(init)}: supply from_data "
        "(cell_table/label_image) or an explicit cells list"
    )


def _generate_blocks(init: dict) -> list[dict]:
    """Generative ``kind: blocks``: a ``shape`` grid of ``type`` cells at ``spacing`` pitch
    from ``origin`` (centroids in grid units), plus an optional ``infected_seed``
    ``{type, fraction, seed}`` that retypes ``max(1, round(fraction * n))`` of them
    (seeded, reproducible)."""
    import numpy as np

    b = init["block"]
    nx, ny = (int(v) for v in b["shape"])
    ox, oy = (float(v) for v in b.get("origin", [0, 0]))
    pitch = float(b["spacing"])
    cells = [
        {"id": 1 + j * nx + i, "type": b["type"],
         "centroid": [ox + i * pitch, oy + j * pitch], "size": float(b.get("size", 1.0))}
        for j in range(ny) for i in range(nx)
    ]
    seed = init.get("infected_seed")
    if seed:
        n = max(1, int(round(float(seed["fraction"]) * len(cells))))
        rng = np.random.default_rng(int(seed.get("seed", 0)))
        for k in rng.choice(len(cells), size=n, replace=False):
            cells[int(k)]["type"] = seed["type"]
    return cells


def _as_cell_nodes(cells) -> list[dict]:
    """Normalize Cell dicts for a composite document: str/float-friendly, tuple->list."""
    out = []
    for c in cells:
        out.append({
            "id": c["id"],
            "type": str(c["type"]),
            "centroid": [float(v) for v in c["centroid"]],
            "size": float(c.get("size", 1.0)),
        })
    return out


# --------------------------------------------------------------------------- #
# Shared store defaults (identical across backends — part of the invariant)
# --------------------------------------------------------------------------- #
def _shared_store_defaults(blueprint) -> dict:
    """Empty-but-typed starting values for every shared store.

    Pre-seeding keeps the store set explicit and identical across backends. On
    step 1 the mechanism processes read these empties; by step 2 the mechanics
    node has populated positions/types/... — ordinary dynamical-system wiring.
    """
    stores = {
        # mechanics -> field/rules
        "positions": {}, "sizes": {}, "types": {},
        "shared_surface_by_type": {}, "neighbors": {}, "alive": {},
        # field -> rules/mechanics
        "field_grid": {},
        # rules -> mechanics
        "fates": {}, "scalar": {}, "divisions": [], "deaths": [],
        # immune recruitment -> mechanics (Task 20): active_type -> count to activate.
        # Always present: the MechanicsBackend ``activations`` input is wired on every
        # compile (via INPUT_PORTS), so the store must exist even with no immune node.
        "activations": {},
    }
    if blueprint.systemic_ode is not None:
        stores.update({"ode_state": {}, "recruit_drivers": {}, "sig_1": 0.0})
        # scalar counts the ODE consumes (fed by a later counting process / study).
        for k in ("H", "I", "M", "K", "E", "DH", "V", "L", "C", "F", "G_ki", "B_ei"):
            stores[f"ode_in_{k}"] = 0.0
    if blueprint.systemic_ode is not None or blueprint.immune:
        stores["spawns"] = []
    return stores


# --------------------------------------------------------------------------- #
# The shared mechanism nodes
# --------------------------------------------------------------------------- #
def _field_node(blueprint) -> dict:
    return {
        "_type": "process",
        "address": FIELD_ADDR,
        "config": {
            "grid_shape": grid_shape_from_domain(blueprint.domain),
            "fields": field_process_fields(blueprint.fields),
            "boundary": (blueprint.domain or {}).get("boundary", "noflux"),
        },
        "inputs": {"positions": ["positions"], "types": ["types"]},
        "outputs": {"field_grid": ["field_grid"]},
    }


def _rules_node(blueprint) -> dict:
    const = {}
    # blueprint constants may live under mechanics_mapping.const or rules_const.
    mm = blueprint.mechanics_mapping or {}
    for src in (mm.get("const"), mm.get("rules_const")):
        if isinstance(src, dict):
            const.update({k: float(v) for k, v in src.items()})
    return {
        "_type": "process",
        "address": RULES_ADDR,
        "config": {
            "rules": normalize_rules(blueprint.rules),
            "const": const,
            "seed": int(mm.get("rules_seed", 0)),
        },
        "inputs": {
            "positions": ["positions"], "sizes": ["sizes"], "types": ["types"],
            "shared_surface_by_type": ["shared_surface_by_type"],
            "neighbors": ["neighbors"], "alive": ["alive"],
            "field_grid": ["field_grid"],
        },
        "outputs": {
            "fates": ["fates"], "scalar": ["scalar"],
            "divisions": ["divisions"], "deaths": ["deaths"],
        },
    }


def _ode_node(blueprint) -> dict:
    so = blueprint.systemic_ode or {}
    coupling = so.get("coupling") or {}
    # The blueprint SELECTS + PARAMETERIZES the registered ODE component (spec §7):
    # it must supply the price_ode params block (constants/homeostatic_pops_ode/
    # hill_exponents), which SystemicODEProcess resolves into the per-MCS consts.
    # A model_ref with no params would build a broken ODE (KeyError '_s_t' deep in
    # GlobalODE), so fail loud here instead. (A model_ref -> packaged-params registry
    # is a follow-up; v1 requires explicit params.)
    params = so.get("params") or coupling.get("params")
    if not params:
        raise ValueError(
            f"systemic_ode (model_ref={so.get('model_ref')!r}) needs a 'params' block "
            "(the price_ode constants/homeostatic_pops_ode/hill_exponents); none found"
        )
    config = {
        "consts": params,
        "num_epithelial": int(coupling.get("num_epithelial", 250000)),
        "dt_seconds": float(coupling.get("dt_seconds", 60.0)),
        "v0": float(coupling.get("v0", 0.0)),
    }
    keys = ("H", "I", "M", "K", "E", "DH", "V", "L", "C", "F", "G_ki", "B_ei")
    return {
        "_type": "process",
        "address": ODE_ADDR,
        "config": config,
        # FOLLOW-UP (not wired in v1): the spatialized counts H/I/M/... the ODE
        # consumes have no producer yet (no counting Step among Tasks 2-12), so
        # these ode_in_* stores sit at their 0.0 defaults until that Step lands.
        "inputs": {k: [f"ode_in_{k}"] for k in keys},
        "outputs": {
            "ode_state": ["ode_state"],
            "recruit_drivers": ["recruit_drivers"],
            "sig_1": ["sig_1"],
        },
    }


def _immune_node(blueprint) -> dict:
    imm = blueprint.immune if isinstance(blueprint.immune, dict) else {}
    config = {}
    if imm.get("types"):
        config["types"] = dict(imm["types"])
    for k in ("id_prefix", "start_index", "max_per_step", "seed"):
        if imm.get(k) is not None:
            config[k] = imm[k]
    return {
        "_type": "process",
        "address": IMMUNE_ADDR,
        "config": config,
        "inputs": {"recruit_drivers": ["recruit_drivers"]},
        # ``spawns`` stays for provenance/lineage; ``activations`` (Task 20) is the
        # per-type activation-count map the MechanicsBackend ``activations`` input
        # consumes to activate pre-placed LATENT cells -- this closes the T13-GAP where
        # spawns were emitted but no backend port consumed them.
        "outputs": {"spawns": ["spawns"], "activations": ["activations"]},
    }


def build_mechanism_state(blueprint) -> dict:
    """The backend-independent part of the composite state: shared stores + the
    shared mechanism process nodes. IDENTICAL for CPM and vertex."""
    state: dict = dict(_shared_store_defaults(blueprint))
    state["field"] = _field_node(blueprint)
    state["rules"] = _rules_node(blueprint)
    if blueprint.systemic_ode is not None:
        state["systemic_ode"] = _ode_node(blueprint)
    if blueprint.systemic_ode is not None or blueprint.immune:
        state["immune"] = _immune_node(blueprint)
    return state


def assemble(blueprint, mechanics_node: dict) -> dict:
    """Combine the shared mechanism with the ONE backend-specific mechanics node."""
    state = build_mechanism_state(blueprint)
    state["mechanics"] = mechanics_node
    return {"state": state}
