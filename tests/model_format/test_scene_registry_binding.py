"""Scene / registry / binding data model for the model-definition format."""

import pytest

from viva_multicell_blueprint.model_format.binding import bind
from viva_multicell_blueprint.model_format.registry import (
    ModelSpec,
    Registry,
    Transport,
    UnboundCellType,
)
from viva_multicell_blueprint.model_format.scene import CellSpec, Scene


def _spec(formalism: str) -> ModelSpec:
    return ModelSpec(
        formalism=formalism,
        source={"model": formalism},
        params={"k": 1.0},
        membrane={"S": Transport(kind="cross", rate=0.5)},
    )


def _three_cell_scene() -> Scene:
    return Scene(
        domain={"size": [10.0, 10.0]},
        cells={
            "c1": CellSpec(type="A", position=[1.0, 2.0], molecule_levels={"S": 3.0}),
            "c2": CellSpec(type="B", position=[4.0, 5.0]),
            "c3": CellSpec(type="C", position=[7.0, 8.0], molecule_levels={"G": 0.1}),
        },
    )


def _three_type_registry() -> Registry:
    return Registry({"A": _spec("sbml"), "B": _spec("smoldyn"), "C": _spec("maboss")})


def test_bind_resolves_each_cell_to_its_type_spec():
    registry = _three_type_registry()

    bound = bind(_three_cell_scene(), registry)

    assert set(bound) == {"c1", "c2", "c3"}
    assert bound["c1"] is registry.get("A")
    assert bound["c2"] is registry.get("B")
    assert bound["c3"] is registry.get("C")
    assert bound["c1"].formalism == "sbml"
    assert bound["c2"].formalism == "smoldyn"
    assert bound["c3"].formalism == "maboss"


def test_unbound_cell_type_raises():
    scene = _three_cell_scene()
    registry = Registry({"A": _spec("sbml"), "B": _spec("smoldyn")})

    with pytest.raises(UnboundCellType, match="C"):
        bind(scene, registry)


def test_registry_get_unknown_type_raises():
    with pytest.raises(UnboundCellType):
        Registry({}).get("missing")


def test_model_spec_defaults():
    spec = ModelSpec(formalism="sbml", source="model.xml")

    assert spec.params == {}
    assert spec.membrane == {}


def test_scene_round_trips_through_yaml():
    scene = _three_cell_scene()

    text = scene.to_yaml()
    restored = Scene.from_yaml(text)

    assert restored == scene
    assert restored.cells["c1"].molecule_levels == {"S": 3.0}
    assert restored.cells["c2"].molecule_levels == {}
