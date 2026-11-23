"""Task 7 — compile_scene(scene, registry) -> one runnable process-bigraph composite.

`compile_scene` assembles the scene + registry + per-type model specs (Tasks 1-6)
into a SINGLE orchestrating process (`SceneModelProcess`) that holds the tissue
mechanics backend, one intracellular adapter per cell (chosen by the bound
`ModelSpec.formalism`), and one shared extracellular field, stepping them all
internally each `update()` under the membrane contract.

These tests pin the integration contract: a mixed ODE+particle scene compiles to a
document that `Composite(...)` instantiates and `run(3)`s without error, each cell's
adapter is the declared formalism, and an unbound cell type errors at compile (not
at run).
"""
from __future__ import annotations

import pytest
from process_bigraph import Composite

from viva_multicell_blueprint.core import build_core
from viva_multicell_blueprint.model_format.adapters.ode_tellurium import ODEAdapter
from viva_multicell_blueprint.model_format.adapters.particle_smoldyn import ParticleAdapter
from viva_multicell_blueprint.model_format.compile import compile_scene
from viva_multicell_blueprint.model_format.extracellular import FieldExtracellular
from viva_multicell_blueprint.model_format.membrane import NoExtracellularHome
from viva_multicell_blueprint.model_format.registry import (
    ModelSpec,
    Registry,
    Transport,
    UnboundCellType,
)
from viva_multicell_blueprint.model_format.scene import CellSpec, Scene

# One canonical A + B <-> C mass-action network, read by BOTH engines (Task 5 rates).
_NET = {
    "species": {"A": 4.0, "B": 4.0, "C": 0.0},
    "reactions": [{"subs": ["A", "B"], "prds": ["C"], "kf": 0.05, "kb": 0.5}],
}


def _ode_spec() -> ModelSpec:
    # C is permeable: exercises the membrane contract + shared extracellular field.
    return ModelSpec(formalism="ode", source=_NET, params={},
                     membrane={"C": Transport("cross", 0.1)})


def _particle_spec() -> ModelSpec:
    return ModelSpec(formalism="particle", source=_NET, params={}, membrane={})


def _scene() -> Scene:
    return Scene(
        domain={"extent_um": [30.0, 30.0]},
        cells={
            "e1": CellSpec(type="E", position=[9.0, 14.0],
                           molecule_levels={"A": 4.0, "B": 4.0, "C": 0.0}),
            "s1": CellSpec(type="S", position=[20.0, 14.0],
                           molecule_levels={"A": 4.0, "B": 4.0, "C": 0.0}),
        },
    )


def _registry() -> Registry:
    return Registry({"E": _ode_spec(), "S": _particle_spec()})


def test_compile_scene_builds_runnable_composite():
    doc = compile_scene(_scene(), _registry())

    assert "state" in doc
    assert doc["state"]["scene_model"]["address"].endswith("SceneModelProcess")

    comp = Composite({"state": doc["state"]}, core=build_core())
    comp.run(3.0)  # must complete without error

    assert comp.state["n_cells"] == 2


def test_adapters_are_the_declared_formalisms():
    doc = compile_scene(_scene(), _registry())
    comp = Composite({"state": doc["state"]}, core=build_core())
    proc = comp.state["scene_model"]["instance"]

    assert isinstance(proc._adapters["e1"], ODEAdapter)

    s1 = proc._adapters["s1"]
    if not s1.available():
        pytest.skip("Smoldyn unavailable; ODE cell still exercised above")
    assert isinstance(s1, ParticleAdapter)


def test_particle_scene_runs_three_steps():
    if not ParticleAdapter(_particle_spec(), {"A": 4.0}).available():
        pytest.skip("Smoldyn unavailable")
    doc = compile_scene(_scene(), _registry())
    comp = Composite({"state": doc["state"]}, core=build_core())
    comp.run(3.0)
    assert comp.state["n_cells"] == 2


def test_real_mechanics_backend_is_used_not_the_fallback():
    # Locks in that the REAL tissue backend ran, not the static-square degrade path:
    # a VertexBackend regression (build/compile throwing) would otherwise stay green.
    doc = compile_scene(_scene(), _registry())
    comp = Composite({"state": doc["state"]}, core=build_core())
    proc = comp.state["scene_model"]["instance"]

    assert proc._mech is not None
    assert proc._mech_error is None
    comp.run(3.0)
    # no runtime mechanics-step failure was recorded either
    assert proc._mech_error is None


def test_permeable_species_without_extracellular_home_errors_at_compile():
    # ODE cell declares Z permeable, but the prebuilt field has no Z grid -> the
    # membrane contract has no home for it, so compile_scene must raise AT COMPILE
    # (before any Composite is built).
    reg = Registry({
        "E": ModelSpec(formalism="ode", source=_NET, params={},
                       membrane={"Z": Transport("cross", 0.2)}),
    })
    scene = Scene(
        domain={"extent_um": [30.0, 30.0]},
        cells={"e1": CellSpec(type="E", position=[9.0, 14.0], molecule_levels={"A": 4.0})},
    )
    field = FieldExtracellular(shape=(30, 30), species=[], extent=(30.0, 30.0))  # no Z
    assert not field.has("Z")

    with pytest.raises(NoExtracellularHome, match="Z"):
        compile_scene(scene, reg, extracellular=field)


def test_unbound_cell_type_errors_at_compile():
    scene = _scene()
    registry = Registry({"E": _ode_spec()})  # no spec for type "S"

    with pytest.raises(UnboundCellType, match="S"):
        compile_scene(scene, registry)


def test_intracellular_override_forces_formalism():
    # Force the "E" (ode) type onto the Boolean adapter via a per-type override.
    from viva_multicell_blueprint.model_format.adapters.boolean_maboss import BooleanAdapter

    bnet = {"X": "Y", "Y": "X"}
    reg = Registry({
        "E": ModelSpec(formalism="ode", source=bnet, params={}, membrane={}),
        "S": _particle_spec(),
    })
    scene = Scene(
        domain={"extent_um": [30.0, 30.0]},
        cells={"e1": CellSpec(type="E", position=[9.0, 14.0], molecule_levels={"X": 1.0})},
    )
    doc = compile_scene(scene, reg, intracellular_overrides={"E": "boolean"})
    comp = Composite({"state": doc["state"]}, core=build_core())
    proc = comp.state["scene_model"]["instance"]
    assert isinstance(proc._adapters["e1"], BooleanAdapter)
