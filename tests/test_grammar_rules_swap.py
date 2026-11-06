"""Stage 2: viva-grammar's BlueprintGrammarProcess as a drop-in for RulesProcess.

Proves the swap on the FakeBackend (deterministic): the grammar engine reads the
backend readouts + a field and drives fates/deaths that the backend applies; and
that it occupies the SAME rules slot with the SAME four output ports (single-slot
swap). A guarded CPM smoke test shows it composing + running on a real ABM.
"""
from __future__ import annotations

import numpy as np
import pytest

from viva_multicell_blueprint.backends.fake_backend import FakeBackend
from viva_multicell_blueprint.blueprint.model import Blueprint
from viva_multicell_blueprint.compile._shared import build_mechanism_state
from viva_multicell_blueprint.core import build_core

pytest.importorskip("viva_grammar")
from viva_grammar.steps.adapters.blueprint import BlueprintGrammarProcess  # noqa: E402
from viva_multicell_blueprint.composites.grammar_demo import (  # noqa: E402
    DEMO_CELL_DEFINITIONS, DEMO_RULES_CSV, grammar_rules_demo, swap_rules_with_grammar,
)

# Grammar-shaped rules; saturated values are large so per_step Pr ~ 1 (deterministic).
GRAMMAR_CSV = (
    "A,signal,increases,transform to B,50,1.0,2,0\n"
    "A,contact with K,increases,necrosis,50,1.0,2,0\n"
)


def _cells(*spec):
    return [{"id": i, "type": t, "centroid": list(c), "size": s} for i, t, c, s in spec]


def _backend(init):
    return FakeBackend({"initial_state": init, "backend_params": {"neighbor_radius": 1.5}},
                       core=build_core())


def _grammar(csv=GRAMMAR_CSV, cell_defs=None, seed=0):
    return BlueprintGrammarProcess(
        {"rules_csv": csv, "cell_definitions": cell_defs or {}, "seed": seed}, core=build_core())


def test_grammar_process_is_registered_via_import():
    # build_core chains viva_grammar.core.build_core from workspace.yaml imports
    assert "BlueprintGrammarProcess" in build_core().link_registry


def test_grammar_defaults_to_per_step_parity():
    assert _grammar().config["rate_mode"] == "per_step"
    assert set(_grammar().outputs()) == {"fates", "scalar", "divisions", "deaths"}


def test_grammar_drives_fates_and_deaths_from_readouts():
    init = _cells((1, "A", (2, 2), 1.0), (2, "A", (7, 7), 1.0), (3, "K", (7, 8), 1.0))
    out0 = _backend(init).update({}, 1)
    grid = {"signal": np.zeros((10, 10), dtype=np.float32)}
    grid["signal"][2, 2] = 1000.0  # high signal only at cell 1's centroid

    out = _grammar().update({**out0, "field_grid": grid}, 1)
    # cell 1: high signal, no K contact -> transform to B
    assert out["fates"] == {"1": "B"}
    # cell 2: touches K (3), zero signal -> necrosis death (carries cause)
    assert {"cell_id": "2", "cause": "necrosis"} in out["deaths"]
    # cell 3 (K) untouched by both rules
    assert "3" not in out["fates"]


def test_closed_loop_backend_applies_grammar_events():
    init = _cells((1, "A", (2, 2), 1.0), (2, "A", (7, 7), 1.0), (3, "K", (7, 8), 1.0))
    bk = _backend(init)
    out0 = bk.update({}, 1)
    grid = {"signal": np.zeros((10, 10), dtype=np.float32)}
    grid["signal"][2, 2] = 1000.0
    ev = _grammar().update({**out0, "field_grid": grid}, 1)
    applied = bk.update({"fates": ev["fates"], "deaths": ev["deaths"]}, 1)
    assert applied["types"]["1"] == "B"       # transformed
    assert applied["alive"]["2"] is False     # died
    assert applied["alive"]["1"] is True


def test_single_slot_swap_invariant():
    """Grammar occupies the same rules slot; every other node is byte-identical,
    and the rules slot writes the same four output ports."""
    bp = Blueprint.from_dict({
        "domain": {"extent_um": [10, 10], "grid_resolution_um": 1, "boundary": "noflux"},
        "cell_types": [{"name": "A", "target_size": 4}, {"name": "B", "target_size": 4}],
        "initial_state": {"kind": "blocks",
                          "block": {"type": "A", "origin": [2, 2], "shape": [2, 2], "spacing": 2, "size": 4}},
    })
    stock = build_mechanism_state(bp)
    grammar = build_mechanism_state(bp)
    swap_rules_with_grammar({"state": grammar}, GRAMMAR_CSV, {}, seed=0)

    assert set(stock) == set(grammar)
    for key in stock:
        if key == "rules":
            continue
        assert stock[key] == grammar[key], f"node {key!r} changed by the swap"
    four = {"fates": ["fates"], "scalar": ["scalar"],
            "divisions": ["divisions"], "deaths": ["deaths"]}
    assert stock["rules"]["outputs"] == four
    assert grammar["rules"]["outputs"] == four
    assert stock["rules"]["address"] == "local:RulesProcess"
    assert grammar["rules"]["address"] == "local:BlueprintGrammarProcess"
    # same readout inputs the rules slot consumes
    assert stock["rules"]["inputs"] == grammar["rules"]["inputs"]


def test_demo_generator_builds_and_runs_on_cpm():
    """The grammar_rules_demo composite builds + runs on the CPM backend (real ABM)."""
    pytest.importorskip("viva_cpm")
    from process_bigraph import Composite
    core = build_core()
    doc = grammar_rules_demo(core=core, backend="cpm", seed=0)
    assert doc["state"]["rules"]["address"] == "local:BlueprintGrammarProcess"
    sim = Composite(doc, core=core)
    sim.run(6)  # a few shared-mechanism steps; must not raise
