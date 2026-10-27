"""Task 13 — the compilers (blueprint -> runnable composite per backend).

The compiler lowers ONE backend-neutral blueprint onto each mechanical substrate.
The shared mechanism (FieldProcess + RulesProcess + SystemicODEProcess + immune
recruitment) is wired IDENTICALLY through shared stores on both backends; only the
single mechanics node (CPMBackend vs VertexBackend) differs. These tests pin that
invariant (spec section 5) and prove each compiled composite actually builds + runs.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from process_bigraph import Composite

from viva_multicell_blueprint.blueprint.model import Blueprint
from viva_multicell_blueprint.compile import compile as compile_blueprint
from viva_multicell_blueprint.core import build_core

# The canonical Price-2015 params block (copied from viva-cpm), reused so the full
# blueprint's systemic_ode node is a REAL, buildable ODE rather than a stub.
_PRICE_PARAMS = yaml.safe_load(
    (Path(__file__).parent / "fixtures" / "price_ode_params.yaml").read_text()
)["price_ode"]

# A small CSV cell table: 4 non-collinear cells, two types (H, I) -> a 2x2 block
# both backends can materialize (Voronoi needs a non-degenerate point cloud). Cells
# are size 25 (5x5): small 9-voxel CPM cells are energetically unstable under the
# default contact energies (a cell fully ringed by medium LOWERS its energy by
# vanishing), so they dissolve before anything can be observed. 25-voxel cells are
# stable, so the backend simulates LIVING cells (fine J/lambda tuning is the
# adhesion-calibration study's job, spec sec 9.1).
_CELL_TABLE = "id,type,x,y,size\n1,H,6,6,25\n2,H,15,6,25\n3,H,6,15,25\n4,I,15,15,25\n"


def _cell_table(tmp_path, text: str = _CELL_TABLE):
    p = tmp_path / "cells.csv"
    p.write_text(text)
    return str(p)


def _min_blueprint(tmp_path):
    """A few cells, one field, one rule — small enough to run fast on both backends."""
    return Blueprint.from_dict({
        "domain": {"extent_um": [22, 22], "grid_resolution_um": 1, "boundary": "noflux"},
        "cell_types": [
            {"name": "H", "target_size": 25},
            {"name": "I", "target_size": 25},
        ],
        "initial_state": {"from_data": {"cell_table": _cell_table(tmp_path)}},
        "fields": [{
            "name": "virus", "D": 0.1, "decay": 0.01,
            "secretion": [{"type": "I", "rate": 1.0}],
            "chemotaxis": [{"type": "H", "strength": 0.1}],
        }],
        "rules": [
            {"type": "transition", "from": "H", "to": "I", "rate_expr": "0.2 + field['virus']"},
        ],
    })


def _full_blueprint(tmp_path):
    """Adds adhesion + a systemic ODE + immune component so the identical-stores
    invariant is exercised over ALL four shared mechanism nodes, and the adhesion
    lowering (to_cpm_contact / to_vertex_tension) is actually called."""
    bp = _min_blueprint(tmp_path)
    bp.adhesion = [
        {"a": "H", "b": "H", "strength": 5.0},
        {"a": "H", "b": "I", "strength": 8.0},
    ]
    bp.rules.append({"type": "resistance", "scalar": "resist", "rate_expr": "hill(field['virus'], 2.0)"})
    bp.systemic_ode = {
        "model_ref": "price2015",
        "params": _PRICE_PARAMS,
        "coupling": {"num_epithelial": 250000},
    }
    bp.immune = {"types": {"macrophage_inflow": "M"}}
    return bp


# --------------------------------------------------------------------------- #
# Step 1 tests
# --------------------------------------------------------------------------- #
def test_compile_cpm_runs_n_steps(tmp_path):
    bp = _min_blueprint(tmp_path)
    doc = compile_blueprint(bp, backend="cpm")
    assert doc["state"]["mechanics"]["address"].endswith("CPMBackend")

    comp = Composite({"state": doc["state"]}, core=build_core())
    comp.run(3.0)

    pos = comp.state["positions"]
    assert set(pos) == {"1", "2", "3", "4"}
    assert all(len(p) == 2 for p in pos.values())
    # the shared field engine produced the virus grid, read by both rules + mechanics
    assert "virus" in comp.state["field_grid"]
    # cells are stable (size 25) -> the backend simulates LIVING cells, not a lattice
    # that instantly dissolved to medium.
    assert all(comp.state["alive"].values())


def test_compile_vertex_runs_n_steps(tmp_path):
    bp = _min_blueprint(tmp_path)
    doc = compile_blueprint(bp, backend="vertex")
    assert doc["state"]["mechanics"]["address"].endswith("VertexBackend")

    comp = Composite({"state": doc["state"]}, core=build_core())
    comp.run(3.0)

    pos = comp.state["positions"]
    assert set(pos) == {"1", "2", "3", "4"}
    assert all(len(p) == 2 for p in pos.values())
    assert "virus" in comp.state["field_grid"]


def test_mechanism_stores_identical_across_backends(tmp_path):
    bp = _full_blueprint(tmp_path)
    cpm = compile_blueprint(bp, backend="cpm")["state"]
    vtx = compile_blueprint(bp, backend="vertex")["state"]

    # exactly one node differs: the mechanics store.
    assert "mechanics" in cpm and "mechanics" in vtx
    rest_cpm = {k: v for k, v in cpm.items() if k != "mechanics"}
    rest_vtx = {k: v for k, v in vtx.items() if k != "mechanics"}
    assert json.dumps(rest_cpm, sort_keys=True) == json.dumps(rest_vtx, sort_keys=True), (
        "mechanism stores (field/rules/ode/immune + shared stores) must be "
        "BYTE-IDENTICAL across backends"
    )

    # all four shared mechanism processes are present + identical
    for node in ("field", "rules", "systemic_ode", "immune"):
        assert node in rest_cpm
        assert rest_cpm[node] == rest_vtx[node]

    # the mechanics nodes DO differ (that is the whole point).
    assert cpm["mechanics"]["address"] != vtx["mechanics"]["address"]

    # adhesion was lowered per backend (type_ids from cell_type order -> stable tags)
    cpm_bp = cpm["mechanics"]["config"]["backend_params"]
    vtx_bp = vtx["mechanics"]["config"]["backend_params"]
    assert cpm_bp["type_ids"] == {"medium": 0, "H": 1, "I": 2}
    assert any(isinstance(e["a"], int) for e in cpm_bp["contact"])
    assert "tension_by_pair" in vtx_bp and vtx_bp["tension_by_pair"]


@pytest.mark.parametrize("backend", ["cpm", "vertex"])
def test_full_blueprint_builds_and_runs(tmp_path, backend):
    """The FULL blueprint (adds the ODE + immune nodes) must actually BUILD and
    advance on both backends -- a port-type/config mismatch in those nodes would
    slip past the dict-only identical-stores check and only fail here."""
    bp = _full_blueprint(tmp_path)
    doc = compile_blueprint(bp, backend=backend)
    comp = Composite({"state": doc["state"]}, core=build_core())
    comp.run(1.0)

    # the systemic ODE really integrated its 10 states into the shared store
    ode = comp.state["ode_state"]
    assert isinstance(ode, dict) and {"P", "T", "O"} <= set(ode)
    # the immune recruitment node produced a (possibly empty) spawns list
    assert isinstance(comp.state["spawns"], list)
    assert float(comp.state["global_time"]) == 1.0


# EVERY cell is type H -- so observing ANY "I" proves a real H->I transition flowed
# through the stores (not a pre-seeded I, which made the first version vacuous).
_ALL_H_TABLE = "id,type,x,y,size\n1,H,6,6,25\n2,H,15,6,25\n3,H,6,15,25\n4,H,15,15,25\n"


def test_fates_propagate_through_overwrite_map_store(tmp_path):
    """Pin the t=0 key-drop fix with a REAL transition: every cell starts as H, a
    certain-to-fire ``transition {H->I}`` must flow rules.fates -> mechanics.fates
    (applied via set_cell_type) -> the shared `types` store (overwrite[map]) so the
    I-count goes 0 -> >0. If the overwrite wrapping were dropped the map outputs
    would be swallowed on the empty t=0 store and no cell would ever flip."""
    bp = Blueprint.from_dict({
        "domain": {"extent_um": [22, 22], "grid_resolution_um": 1, "boundary": "noflux"},
        "cell_types": [{"name": "H", "target_size": 25}, {"name": "I", "target_size": 25}],
        "initial_state": {"from_data": {"cell_table": _cell_table(tmp_path, _ALL_H_TABLE)}},
        # rate ~100 -> Pr = 1 - exp(-100) ~ 1: every H fires, deterministically.
        "rules": [{"type": "transition", "from": "H", "to": "I", "rate_expr": "100"}],
    })
    doc = compile_blueprint(bp, backend="cpm")
    comp = Composite({"state": doc["state"]}, core=build_core())

    # start: not a single I anywhere.
    assert "I" not in (comp.state.get("types") or {}).values()

    # run enough ticks for fates to round-trip the concurrent stores:
    # t1 mechanics fills `types`; t2 rules reads `types`, emits `fates`;
    # t3 mechanics reads `fates`, applies set_cell_type, re-emits `types`.
    comp.run(3.0)

    types = comp.state["types"]
    assert set(types) == {"1", "2", "3", "4"}
    i_cells = [cid for cid, t in types.items() if t == "I"]
    assert i_cells, (
        "no H->I transition reached the shared `types` store -- fates were dropped "
        "crossing the overwrite[map] store (the t=0 key-drop regression)"
    )
    # specific-cell flip + count went 0 -> >0, and the flipped cell is alive.
    assert types["1"] == "I"
    assert comp.state["alive"]["1"] is True


def test_compiler_wires_recruitment_to_activation(tmp_path):
    """Task 20: the ImmuneRecruitmentProcess ``activations`` output must reach the
    MechanicsBackend ``activations`` input store on both backends (closing the
    T13-GAP where ``spawns`` was emitted but never consumed), and a compiled
    composite must build + run with that wiring live."""
    from viva_multicell_blueprint.core import build_core as _bc
    from viva_multicell_blueprint.mechanism.immune import ImmuneRecruitmentProcess

    bp = _full_blueprint(tmp_path)
    for backend in ("cpm", "vertex"):
        state = compile_blueprint(bp, backend=backend)["state"]
        # immune node emits per-type activation COUNTS into the shared store ...
        assert state["immune"]["outputs"]["activations"] == ["activations"]
        # ... and the mechanics node reads them back as its activation input.
        assert state["mechanics"]["inputs"]["activations"] == ["activations"]
        # the shared store is pre-seeded (identical across backends).
        assert "activations" in state

        comp = Composite({"state": state}, core=build_core())
        comp.run(1.0)
        assert isinstance(comp.state["activations"], dict)

    # the activation path is real: non-zero recruit drivers -> non-zero activation
    # counts, and those counts equal the number of spawn events for each type.
    proc = ImmuneRecruitmentProcess(
        {"seed": 1, "types": {"macrophage_inflow": "M", "cd8_inflow": "CD8"},
         "max_per_step": 50}, core=_bc())
    out = proc.update({"recruit_drivers": {"macrophage_inflow": 4.0, "cd8_inflow": 3.0}}, 1)
    acts = out["activations"]
    assert sum(acts.values()) > 0
    from collections import Counter
    by_type = Counter(e["type"] for e in out["spawns"])
    assert {k: v for k, v in acts.items() if v} == dict(by_type)


# --------------------------------------------------------------------------- #
# Rule validation hook
# --------------------------------------------------------------------------- #
def test_rule_validation_hook_normalizes_type_to_kind(tmp_path):
    from viva_multicell_blueprint.compile._shared import normalize_rules

    out = normalize_rules([
        {"type": "transition", "from": "H", "to": "I", "rate_expr": "0.1"},
        {"type": "kill", "target": "I", "killer": "CD8", "rate_expr": "0.2"},
        {"type": "resistance", "scalar": "resist", "rate_expr": "0.3"},
    ])
    assert [r["kind"] for r in out] == ["transition", "kill", "resistance"]
    assert out[0]["from"] == "H" and out[0]["to"] == "I"
    assert out[1]["target"] == "I"


def test_rule_validation_hook_fails_loud_on_unknown_kind():
    from viva_multicell_blueprint.compile._shared import normalize_rules

    with pytest.raises(ValueError, match="unknown rule kind"):
        normalize_rules([{"type": "teleport", "rate_expr": "0.1"}])


def test_rule_validation_hook_requires_rate_expr():
    from viva_multicell_blueprint.compile._shared import normalize_rules

    with pytest.raises(ValueError, match="rate_expr"):
        normalize_rules([{"type": "transition", "from": "H", "to": "I"}])


def test_compile_rejects_unknown_backend(tmp_path):
    bp = _min_blueprint(tmp_path)
    with pytest.raises(ValueError, match="backend"):
        compile_blueprint(bp, backend="delaunay")


# --------------------------------------------------------------------------- #
# Registration: local:<Backend> / local:<mechanism process> must resolve
# --------------------------------------------------------------------------- #
def test_backends_and_mechanism_resolve_in_core():
    core = build_core()
    for name in ("CPMBackend", "VertexBackend", "FieldProcess", "RulesProcess",
                 "SystemicODEProcess", "ImmuneRecruitmentProcess"):
        assert core.link_registry.get(name) is not None, f"local:{name} does not resolve"
