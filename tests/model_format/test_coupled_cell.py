"""CoupledCell: several coupled sub-models inside one cell (modes A, B, C)."""
import pytest

from viva_multicell_blueprint.model_format.adapters.coupled import CoupledCell
from viva_multicell_blueprint.model_format.registry import ModelSpec

# --- mode A + C oscillator: ODE metabolism <-> MaBoSS fate (spike 1) ---
_OSC = {
    "models": {
        "metab": {"formalism": "ode",
                  "source": {"species": {"A": 1.2, "B": 1.2, "C": 0.6},
                             "reactions": [{"subs": ["A", "B"], "prds": ["C"],
                                            "kf": 0.01, "kb": 0.3}]},
                  "initial": {"A": 40.0, "B": 40.0, "C": 0.0}},
        "fate": {"formalism": "boolean",
                 "source": {"high_C": "high_C", "divide": "high_C"},
                 "initial": {"high_C": 0.0, "divide": 0.0}},
    },
    "couplings": [
        {"mode": "A", "from": "metab", "species": "C", "threshold": 12.0,
         "to": "fate", "input": "high_C"},
        {"mode": "C", "from": "fate", "node": "divide", "to": "metab", "action": "reset"},
    ],
}

# --- mode B catalysis: particle count -> ODE rate (spike 2) ---
_CAT = {
    "models": {
        "sensor": {"formalism": "particle",
                   "source": {"species": {"A": 1, "B": 1, "C": 0},
                              "reactions": [{"subs": ["A", "B"], "prds": ["C"],
                                             "kf": 0.35, "kb": 0.015}]},
                   "params": {"polygon": [[0, 0], [10, 0], [10, 10], [0, 10]]},
                   "initial": {"A": 45.0, "B": 45.0, "C": 0.0}, "seed": 7},
        "metab": {"formalism": "ode",
                  "source": {"species": {"S": 400.0, "P": 0.0},
                             "reactions": [{"subs": ["S"], "prds": ["P"],
                                            "kf": 0.0, "kb": 0.0}]},
                  "initial": {"S": 400.0, "P": 0.0}},
    },
    "couplings": [
        {"mode": "B", "from": "sensor", "species": "C",
         "to": "metab", "param": "kf0", "base": 0.001},
    ],
}


def _mk(src):
    return CoupledCell(ModelSpec(formalism="coupled", source=src, params={}, membrane={}), {})


def test_construct_and_namespaced_amounts():
    cell = _mk(_OSC)
    am = cell.amounts()
    assert "metab/C" in am and "fate/divide" in am
    assert isinstance(cell.available(), bool)
    assert cell.crossed() == {}


def test_initial_override_namespaced():
    cell = CoupledCell(
        ModelSpec(formalism="coupled", source=_OSC, params={}, membrane={}),
        {"metab/C": 5.0})
    assert cell.amounts()["metab/C"] == pytest.approx(5.0)


def test_mode_a_c_oscillator():
    """C builds -> pins high_C -> divide fires -> division resets metabolism -> repeat."""
    pytest.importorskip("tellurium")
    cell = _mk(_OSC)
    if not cell.available():
        pytest.skip("tellurium/maboss engine unavailable")
    divides = []
    for _ in range(40):
        cell.step(1.0, {})
        divides.append(cell.amounts()["fate/divide"])
    # oscillation: divide must both rise and fall at least once (not stuck)
    transitions = sum(1 for a, b in zip(divides, divides[1:]) if (a > 0.5) != (b > 0.5))
    assert transitions >= 2, f"divide did not oscillate: {divides}"


def test_mode_b_catalysis_rate_tracks_count():
    """Particle C count sets the ODE rate; P accumulates and the rate responded to C."""
    pytest.importorskip("tellurium")
    cell = _mk(_CAT)
    if not cell.available():
        pytest.skip("tellurium/smoldyn engine unavailable")
    p_series, c_series = [], []
    for _ in range(14):
        cell.step(0.12, {})
        am = cell.amounts()
        p_series.append(am["metab/P"])
        c_series.append(am["sensor/C"])
    # product accumulates (P[0]==0: step 0's pre-step reads the sensor's initial C=0)
    assert p_series[-1] > 0.0
    assert all(b >= a for a, b in zip(p_series, p_series[1:]))  # monotone non-decreasing
    assert c_series[-1] > c_series[0]                 # particle signal built up
    # coupling: the live ODE rate constant tracks the particle count. Pre-step B reads
    # the sensor's PREVIOUS amounts, so kf0 after the last step reflects c_series[-2].
    assert cell.subs["metab"]._rr["kf0"] == pytest.approx(0.001 * c_series[-2], rel=1e-6)
    assert cell.subs["metab"]._rr["kf0"] > 0.0
