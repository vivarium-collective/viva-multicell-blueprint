"""composition-atlas: the four CoupledCell composites build, register and run."""
from __future__ import annotations

import pytest
from process_bigraph import Composite
from process_bigraph import composite_spec as _cs

import viva_multicell_blueprint.composites  # noqa: F401 (registers generators)
from viva_multicell_blueprint.core import build_core

ATLAS = {
    "coupled_fate_oscillator": "viva_multicell_blueprint.composites.coupled_subcellular.coupled_fate_oscillator",
    "coupled_particle_catalysis": "viva_multicell_blueprint.composites.coupled_subcellular.coupled_particle_catalysis",
    "coupled_triple_cell": "viva_multicell_blueprint.composites.coupled_subcellular.coupled_triple_cell",
    "fateful_tissue": "viva_multicell_blueprint.composites.fateful_tissue.fateful_tissue",
}


def _build(cid, **params):
    spec = _cs.get(cid)
    doc = spec.to_document(params, core=build_core())
    return Composite({"state": doc["state"]}, core=build_core())


@pytest.mark.parametrize("name,cid", list(ATLAS.items()))
def test_atlas_generators_registered(name, cid):
    spec = _cs.get(cid)
    assert spec is not None and spec.kind == "generator"


def test_fate_oscillator_oscillates():
    pytest.importorskip("tellurium")
    comp = _build(ATLAS["coupled_fate_oscillator"], n_cells=1)
    proc = comp.state["scene_model"]["instance"]
    sid = proc._cell_ids[0]
    if not proc._adapters[sid].available():
        pytest.skip("tellurium/maboss unavailable")
    divides = []
    for _ in range(24):
        comp.run(1)
        divides.append(proc._adapters[sid].amounts()["fate/divide"])
    transitions = sum(1 for a, b in zip(divides, divides[1:]) if (a > 0.5) != (b > 0.5))
    assert transitions >= 2, f"fate did not oscillate: {divides}"


def test_triple_cell_runs_all_three_formalisms():
    pytest.importorskip("tellurium")
    comp = _build(ATLAS["coupled_triple_cell"], n_cells=1)
    proc = comp.state["scene_model"]["instance"]
    sid = proc._cell_ids[0]
    ad = proc._adapters[sid]
    if not ad.available():
        pytest.skip("an engine (tellurium/maboss/smoldyn) unavailable")
    assert set(ad.subs) == {"sensor", "metab", "fate"}
    comp.run(6)
    am = ad.amounts()
    assert {"sensor/C", "metab/C", "fate/divide"} <= set(am)


def test_fateful_tissue_fate_drives_division_and_conserves_particles():
    pytest.importorskip("tellurium")
    comp = _build(ATLAS["fateful_tissue"], seed_per_cell=30)
    proc = comp.state["fateful_tissue"]["instance"]
    if not proc._runner.available():
        pytest.skip("smoldyn bridge unavailable")
    comp.run(12)
    st = comp.state
    assert st["n_divisions"] >= 1          # fate drove at least one division
    assert st["n_cells"] >= 4              # tissue sustained (grew, did not collapse)
    assert st["n_particles"] > 0           # particles survive splits
