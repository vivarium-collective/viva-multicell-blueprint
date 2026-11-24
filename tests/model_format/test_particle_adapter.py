import pytest

from viva_multicell_blueprint.model_format.adapters.particle_smoldyn import ParticleAdapter
from viva_multicell_blueprint.model_format.registry import ModelSpec

NET = {
    "species": {"A": 40.0, "B": 40.0, "C": 0.0, "D": 15.0},  # D does not react
    "reactions": [{"subs": ["A", "B"], "prds": ["C"], "kf": 8.0, "kb": 2.0}],
}
SQUARE = [[0, 0], [8, 0], [8, 10], [0, 10]]


def _adapter():
    spec = ModelSpec(formalism="particle", source=NET, params={"polygon": SQUARE})
    a = ParticleAdapter(spec, {"A": 40.0, "B": 40.0, "C": 0.0, "D": 15.0})
    if not a.available():
        pytest.skip("Smoldyn unavailable")
    return a


def test_seeds_integer_counts():
    a = _adapter()
    assert a.amounts() == {"A": 40, "B": 40, "C": 0, "D": 15}
    assert a.crossed() == {}


def test_conserves_nonreacting_and_forms_c():
    a = _adapter()
    for i in range(15):
        a.step(1.0, {"polygon": SQUARE})
    am = a.amounts()
    assert am["D"] == 15
    assert am["C"] > 5
    assert am["A"] + am["C"] == 40 and am["B"] + am["C"] == 40
