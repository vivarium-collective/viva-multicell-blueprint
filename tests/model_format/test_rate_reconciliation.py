import numpy as np
import pytest

from viva_multicell_blueprint.model_format import rates
from viva_multicell_blueprint.model_format.adapters.ode_tellurium import ODEAdapter
from viva_multicell_blueprint.model_format.adapters.particle_smoldyn import ParticleAdapter
from viva_multicell_blueprint.model_format.registry import ModelSpec

pytest.importorskip("tellurium")


def _poly(w, h):
    return [[0, 0], [w, 0], [w, h], [0, h]]


def _setup(w, h, n):
    net = {
        "species": {"A": float(n), "B": float(n), "C": 0.0},
        "reactions": [{"subs": ["A", "B"], "prds": ["C"], "kf": 8.0, "kb": 2.0}],
    }
    init = {"A": float(n), "B": float(n), "C": 0.0}
    return ModelSpec(formalism="particle", source=net, params={"polygon": _poly(w, h)}), init


def _ode_c(spec, init):
    a = ODEAdapter(spec, init)
    for _ in range(300):
        a.step(1.0, {})
    return a.amounts()["C"]


def _particle_c(spec, init, poly):
    vals = []
    for seed in (0, 1):
        a = ParticleAdapter(spec, init, seed=seed)
        if not a.available():
            pytest.skip("Smoldyn unavailable")
        for i in range(60):
            a.step(1.0, {"polygon": poly})
            if i >= 20:
                vals.append(a.amounts()["C"])
    return float(np.mean(vals))


def _rel(x, ref):
    return abs(x - ref) / ref


# At the calibration point (80 area, 40+40): observed agreement 1-3%, so 8% is safe, and it
# is tight enough to reject the obvious V = polygon area (13% off).
TOL_FIT = 0.08
# Off the calibration point V_eff/area was measured at 0.55-0.66 (vs the 0.65 used), so
# the translation drifts; 12% allows for that without being meaningless.
TOL_OFF = 0.12


@pytest.fixture(scope="module")
def fit():
    spec, init = _setup(8, 10, 40)
    return spec, init, _particle_c(spec, init, _poly(8, 10))


def test_translated_ode_matches_particle(fit):
    spec, init, pc = fit
    ode = _ode_c(rates.to_ode(spec, volume=rates.effective_volume(_poly(8, 10))), init)
    assert _rel(ode, pc) < TOL_FIT


def test_untranslated_raw_kf_disagrees(fit):
    spec, init, pc = fit
    assert _rel(_ode_c(spec, init), pc) > 0.5


def test_volume_equals_area_control_disagrees(fit):
    # the obvious alternative (no 0.65 calibration) must NOT pass the fit-point tolerance
    spec, init, pc = fit
    ode = _ode_c(rates.to_ode(spec, volume=rates.polygon_area(_poly(8, 10))), init)
    assert _rel(ode, pc) > TOL_FIT


@pytest.mark.parametrize("w,h,n", [(10, 12, 40), (6, 6, 40), (8, 10, 20)])
def test_translation_generalizes_off_calibration(w, h, n):
    spec, init = _setup(w, h, n)
    pc = _particle_c(spec, init, _poly(w, h))
    ode = _ode_c(rates.to_ode(spec, volume=rates.effective_volume(_poly(w, h))), init)
    assert _rel(ode, pc) < TOL_OFF
