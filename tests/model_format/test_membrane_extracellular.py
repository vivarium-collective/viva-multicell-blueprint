import pytest

from viva_multicell_blueprint.model_format.extracellular import FieldExtracellular
from viva_multicell_blueprint.model_format.membrane import NoExtracellularHome, apply
from viva_multicell_blueprint.model_format.registry import Transport

SPEC = {"S": Transport("cross", 0.5), "A": Transport("reflect")}


def test_crossed_species_reaches_neighbour():
    field = FieldExtracellular(shape=(20, 20), species=["S"])
    sender, neighbour = (5.5, 5.5), (8.5, 5.5)
    assert field.concentration_at(neighbour, "S") == 0.0
    apply({"S": 10.0, "A": 4.0}, sender, field, SPEC)
    for _ in range(5):
        field.diffuse()
    assert field.concentration_at(neighbour, "S") > 0.0
    inputs = apply({}, neighbour, field, SPEC)
    assert inputs["S"] > 0.0 and "A" not in inputs


def test_cross_without_home_raises():
    field = FieldExtracellular(shape=(10, 10), species=[])
    with pytest.raises(NoExtracellularHome):
        apply({"S": 1.0}, (1, 1), field, SPEC)
    with pytest.raises(NoExtracellularHome):
        apply({}, (1, 1), None, SPEC)


def test_invalid_kind_rejected():
    field = FieldExtracellular(shape=(5, 5), species=["S"])
    with pytest.raises(ValueError):
        apply({}, (1, 1), field, {"S": Transport("bogus")})


@pytest.mark.skip(reason="particle extracellular deferred: viva-pde-particle")
def test_particle_species_extracellular():
    raise AssertionError("not implemented")
