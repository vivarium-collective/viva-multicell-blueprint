"""Task 8 — the registered ``scene_model`` generator + mixed-formalism demo."""
from __future__ import annotations

import pytest
from process_bigraph import Composite
from process_bigraph import composite_spec as _cs

import viva_multicell_blueprint.composites  # noqa: F401 (registers generators)
from viva_multicell_blueprint.core import build_core
from viva_multicell_blueprint.model_format.adapters.boolean_maboss import BooleanAdapter
from viva_multicell_blueprint.model_format.adapters.ode_tellurium import ODEAdapter
from viva_multicell_blueprint.model_format.adapters.particle_smoldyn import ParticleAdapter


_ID = "viva_multicell_blueprint.composites.scene_model.scene_model"


def _build():
    spec = _cs.get(_ID)
    doc = spec.to_document({}, core=build_core())
    return Composite({"state": doc["state"]}, core=build_core())


def test_scene_model_generator_is_registered():
    spec = _cs.get(_ID)
    assert spec is not None and spec.kind == "generator"
    assert {"scene", "mechanics", "overlay_out"} <= set(spec.parameters)


def test_demo_runs_and_emits_n_cells():
    comp = _build()
    comp.run(3.0)
    assert comp.state["n_cells"] == 5


def test_three_cell_types_use_their_declared_formalisms():
    proc = _build().state["scene_model"]["instance"]
    assert isinstance(proc._adapters["m1"], ODEAdapter)
    assert isinstance(proc._adapters["s1"], ParticleAdapter)
    assert isinstance(proc._adapters["f1"], BooleanAdapter)
    assert proc._adapters["m1"].available(), "ODE cell must be exercised"
