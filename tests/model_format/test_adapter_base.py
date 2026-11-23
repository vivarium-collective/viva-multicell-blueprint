import pytest

from viva_multicell_blueprint.model_format.adapters.base import (
    IntracellularAdapter,
    NullAdapter,
)
from viva_multicell_blueprint.model_format.registry import ModelSpec


def test_abc_not_instantiable():
    with pytest.raises(TypeError):
        IntracellularAdapter(ModelSpec("null", "x"), {})


def test_null_adapter_contract():
    levels = {"A": 1.5, "B": 0.0}
    a = NullAdapter(ModelSpec("null", "x"), levels)
    assert a.amounts() == levels
    a.step(0.1, {"glucose": 5.0})
    assert a.amounts() == levels
    assert a.crossed() == {}
    assert a.available() is True


def test_null_adapter_copies_initial_levels():
    levels = {"A": 1.0}
    a = NullAdapter(ModelSpec("null", "x"), levels)
    levels["A"] = 9.0
    assert a.amounts() == {"A": 1.0}
