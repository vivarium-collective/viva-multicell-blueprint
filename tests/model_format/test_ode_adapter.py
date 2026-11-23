import pytest

from viva_multicell_blueprint.model_format.adapters.ode_tellurium import ODEAdapter
from viva_multicell_blueprint.model_format.registry import ModelSpec

NET = {
    "species": {"A": 40.0, "B": 40.0, "C": 0.0},
    "reactions": [{"subs": ["A", "B"], "prds": ["C"], "kf": 0.01, "kb": 0.5}],
}


def _spec(source):
    return ModelSpec(formalism="ode", source=source, params={}, membrane={})


def test_unavailable_flag_matches_import():
    a = ODEAdapter(_spec(NET), {"A": 40.0, "B": 40.0, "C": 0.0})
    assert isinstance(a.available(), bool)
    assert a.crossed() == {}


def test_reaches_mass_action_equilibrium():
    pytest.importorskip("tellurium")
    a = ODEAdapter(_spec(NET), {"A": 40.0, "B": 40.0, "C": 0.0})
    assert a.available()
    for _ in range(200):
        a.step(1.0, {})
    am = a.amounts()
    # x = C at equilibrium: kf (40-x)^2 = kb x
    kf, kb = 0.01, 0.5
    # solve kf x^2 - (80kf + kb) x + 1600kf = 0
    b = 80 * kf + kb
    x = (b - (b * b - 4 * kf * 1600 * kf) ** 0.5) / (2 * kf)
    assert am["C"] == pytest.approx(x, rel=1e-2)
    assert am["A"] == pytest.approx(40 - x, rel=1e-2)
