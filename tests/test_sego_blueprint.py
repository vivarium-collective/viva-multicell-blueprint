"""Task 17 -- the full Sego 2022 influenza model as ONE backend-neutral blueprint.

This is the template's expressiveness acceptance test (spec sections 7, 9): if the
blueprint document cannot SAY every Sego rate as data, the template is wrong.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from process_bigraph import Composite

from viva_multicell_blueprint.blueprint.model import Blueprint
from viva_multicell_blueprint.compile import compile as compile_blueprint
from viva_multicell_blueprint.core import build_core
from viva_multicell_blueprint.mechanism.rate_expr import parse

SEGO = Path(__file__).resolve().parents[1] / "workspace" / "blueprints" / "influenza_sego2022.yaml"


@pytest.fixture(scope="module")
def bp():
    return Blueprint.load(SEGO)


def test_sego_blueprint_validates(bp):
    names = {c["name"] for c in bp.cell_types}
    assert {"H", "I", "D", "M", "NK", "CD8"} <= names
    # size-25 epithelium: a size-9 CPM cell dissolves to volume 0 (Task 13 finding)
    assert all(c["target_size"] == 25 for c in bp.cell_types if c["name"] in {"H", "I", "D"})
    assert bp.domain["boundary"] == "noflux" and bp.domain["grid_resolution_um"] > 0
    assert {f["name"] for f in bp.fields} == {"virus", "ifn", "chemokine", "il10"}
    assert bp.systemic_ode["model_ref"] == "price2015" and bp.systemic_ode["params"]
    # adhesion carries the collapsed Sego J values
    adh = {frozenset((a["a"], a["b"])): a["strength"] for a in bp.adhesion}
    assert adh[frozenset(("H", "H"))] == 5 and adh[frozenset(("H", "medium"))] == 25
    # every rate_expr is inside the closed grammar
    for r in bp.rules:
        parse(r["rate_expr"])


@pytest.mark.parametrize("backend", ["cpm", "vertex"])
def test_sego_compiles_to_both_backends(bp, backend):
    doc = compile_blueprint(bp, backend=backend)
    comp = Composite({"state": doc["state"]}, core=build_core())
    comp.run(2.0)
    assert len(comp.state["types"]) > 0
    assert {"virus", "ifn", "chemokine", "il10"} <= set(comp.state["field_grid"])


def _rule(bp, **match):
    hits = [r for r in bp.rules if all(r.get(k) == v for k, v in match.items())]
    assert hits, f"no rule matching {match}"
    return hits[0]


def test_expresses_every_sego_rate(bp):
    """Each Sego rate is PRESENT as blueprint data (a rule or a field entry)."""
    infection = _rule(bp, type="transition", **{"from": "H", "to": "I"})
    assert "field['virus']" in infection["rate_expr"] and "g_hv" in infection["rate_expr"]
    resistance = _rule(bp, type="resistance", scalar="resist")
    assert "field['ifn']" in resistance["rate_expr"] and "a_rf" in resistance["rate_expr"]
    idie = _rule(bp, type="transition", **{"from": "I", "to": "D"})
    assert "mu_i" in idie["rate_expr"] and "scalar['resist']" in idie["rate_expr"]
    allee = _rule(bp, type="allee", **{"from": "H", "to": "D"})
    assert "shared_surface" in allee["rate_expr"] and "b_h" in allee["rate_expr"]
    recov = _rule(bp, type="recovery", **{"from": "D", "to": "H"})
    assert "shared_surface" in recov["rate_expr"] and "b_h" in recov["rate_expr"]
    for killer, g in (("NK", "g_ik"), ("CD8", "g_ie")):
        k = _rule(bp, type="kill", killer=killer, target="I")
        assert g in k["rate_expr"] and f"shared_surface['{killer}']" in k["rate_expr"]
        assert "scalar['resist']" in k["rate_expr"] and "size" in k["rate_expr"]
    # IL-10 secretion (macrophage + uninfected epithelium) lives on the il10 field
    il10 = next(f for f in bp.fields if f["name"] == "il10")
    assert {s["type"] for s in il10["secretion"]} >= {"M", "H"}
    # virus/ifn secreted by I; IFN taken up; chemotaxis declared on the fields
    by = {f["name"]: f for f in bp.fields}
    assert any(s["type"] == "I" for s in by["virus"]["secretion"])
    assert any(s["type"] == "I" for s in by["ifn"]["secretion"]) and by["ifn"]["uptake"]
    assert any(s["type"] == "M" for s in by["chemokine"]["secretion"])
    chemo = {(f["name"], c["type"]): c["strength"] for f in bp.fields for c in f.get("chemotaxis", [])}
    assert chemo[("virus", "M")] == 5000 and chemo[("chemokine", "NK")] == 5000
    assert chemo[("chemokine", "CD8")] == 10000
    # every rule constant is declared as data (none hidden in code)
    const = bp.mechanics_mapping["const"]
    for r in bp.rules:
        ex = parse(r["rate_expr"])
        for name in ("g_hv", "a_rf", "mu_i", "b_h", "theta", "g_ik", "g_ie", "tot_ec"):
            if name in r["rate_expr"]:
                assert name in const, name
    assert bp.immune and bp.systemic_ode["coupling"]


def test_generative_blocks_initial_state(bp):
    from viva_multicell_blueprint.compile._shared import resolve_initial_state

    cells = resolve_initial_state(bp)
    assert len(cells) == 36 and all(c["size"] == 25 for c in cells)
    n_i = sum(c["type"] == "I" for c in cells)
    assert n_i == 2 and sum(c["type"] == "H" for c in cells) == 34  # 5% seed of 36
    assert resolve_initial_state(bp) == cells  # seeded -> reproducible
