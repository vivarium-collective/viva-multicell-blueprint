import pytest

from viva_multicell_blueprint.model_format.adapters.boolean_maboss import (
    BooleanAdapter,
)
from viva_multicell_blueprint.model_format.registry import ModelSpec

BNET = "input, input\nmid, input\nout, mid\n"
INIT = {"input": 0.0, "mid": 0.0, "out": 0.0}


def _mk():
    return BooleanAdapter(ModelSpec(formalism="boolean", source=BNET), INIT)


def test_construct_and_flags():
    a = _mk()
    assert isinstance(a.available(), bool)
    assert a.crossed() == {}
    assert set(a.amounts()) == {"input", "mid", "out"}


def test_input_on_drives_output_high():
    a = _mk()
    if not a.available():
        pytest.skip("maboss engine unavailable")
    for _ in range(10):
        a.step(2.0, {"input": 1.0})
    am = a.amounts()
    assert am["out"] > 0.9 and am["mid"] > 0.9


def test_pinning_input_off_flips_output():
    a = _mk()
    if not a.available():
        pytest.skip("maboss engine unavailable")
    for _ in range(10):
        a.step(2.0, {"input": 1.0})
    for _ in range(10):
        a.step(2.0, {"input": 0.0})
    assert a.amounts()["out"] < 0.1


def test_dict_source():
    a = BooleanAdapter(
        ModelSpec(formalism="boolean", source={"i": "i", "o": "!i"}),
        {"i": 1.0},
    )
    if not a.available():
        pytest.skip("maboss engine unavailable")
    for _ in range(10):
        a.step(2.0, {"i": 1.0})
    assert a.amounts()["o"] < 0.1
