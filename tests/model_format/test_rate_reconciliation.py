import numpy as np
import pytest

from viva_multicell_blueprint.model_format import rates
from viva_multicell_blueprint.model_format.adapters.ode_tellurium import ODEAdapter
from viva_multicell_blueprint.model_format.adapters.particle_smoldyn import ParticleAdapter
from viva_multicell_blueprint.model_format.registry import ModelSpec

pytest.importorskip("tellurium")

SQUARE = [[0, 0], [8, 0], [8, 10], [0, 10]]  # area 80
NET = {
    "species": {"A": 40.0, "B": 40.0, "C": 0.0},
    "reactions": [{"subs": ["A", "B"], "prds": ["C"], "kf": 8.0, "kb": 2.0}],
}
INIT = {"A": 40.0, "B": 40.0, "C": 0.0}
SPEC = ModelSpec(formalism="particle", source=NET, params={"polygon": SQUARE})
TOL = 0.12  # relative; particle noise on a time-averaged C is ~4%


def _ode_equilibrium_c(spec):
    a = ODEAdapter(spec, INIT)
    for _ in range(300):
        a.step(1.0, {})
    return a.amounts()["C"]


@pytest.fixture(scope="module")
def particle_c():
    vals = []
    for seed in (0, 1):
        a = ParticleAdapter(SPEC, INIT, seed=seed)
        if not a.available():
            pytest.skip("Smoldyn unavailable")
        for i in range(60):
            a.step(1.0, {"polygon": SQUARE})
            if i >= 20:
                vals.append(a.amounts()["C"])
    return float(np.mean(vals))


def test_translated_ode_matches_particle(particle_c):
    ode_spec = rates.to_ode(SPEC, volume=rates.effective_volume(SQUARE))
    assert _ode_equilibrium_c(ode_spec) == pytest.approx(particle_c, rel=TOL)


def test_untranslated_controls_disagree(particle_c):
    # raw canonical kf in a unit volume
    raw = _ode_equilibrium_c(SPEC)
    assert abs(raw - particle_c) / particle_c > 0.5
