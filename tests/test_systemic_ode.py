from pathlib import Path

import pytest
import yaml

from viva_multicell_blueprint.core import build_core
from viva_multicell_blueprint.mechanism.systemic_ode import (
    INTEGRATED_STATES, SystemicODEProcess)

SRC = Path("/Users/eranagmon/code/viva-cpm/viva_cpm_studies/influenza/params.yaml")


def price_params():
    if not SRC.exists():
        pytest.skip("viva-cpm params.yaml not available")
    return yaml.safe_load(SRC.read_text())["price_ode"]


INP = {"H": 200000.0, "I": 20000.0, "M": 5.0, "K": 2.0, "E": 0.0, "DH": 0.0,
       "V": 50.0, "L": 0.5, "C": 3.0, "F": 0.0}


def proc(**kw):
    cfg = {"consts": price_params(), "num_epithelial": 250000, "dt_seconds": 60.0, "v0": 0.0}
    cfg.update(kw)
    return SystemicODEProcess(cfg, core=build_core())


def test_price_ode_integrates_states():
    p = proc()
    out0 = p.update(INP, 1)
    assert set(out0["ode_state"]) >= set(INTEGRATED_STATES)
    assert len(INTEGRATED_STATES) == 10
    assert all(v >= 0 for v in out0["ode_state"].values())
    # infection with M>0 drives TNF-like T up from 0, monotonically over steps
    ts = [out0["ode_state"]["T"]]
    for _ in range(5):
        ts.append(p.update(INP, 1)["ode_state"]["T"])
    assert ts[0] > 0
    assert all(b > a for a, b in zip(ts, ts[1:]))


def test_sig_1_is_a11_T_plus_a12_D():
    p = proc()
    out = p.update(INP, 1)
    c = p._consts
    # sig_1 is the frozen Sigma1 used for the step: computed from step-start T (=0)
    D = 250000 - INP["H"] - INP["I"]
    assert out["sig_1"] == pytest.approx(c["a_11"] * 0.0 + c["a_12"] * D)
    out2 = p.update(INP, 1)
    assert out2["sig_1"] == pytest.approx(c["a_11"] * out["ode_state"]["T"] + c["a_12"] * D)


def test_recruit_drivers_emitted():
    p = proc()
    out = p.update(INP, 1)
    rd = out["recruit_drivers"]
    for k in ("macrophage_inflow", "nk_inflow", "cd8_inflow",
              "macrophage_outflow", "nk_outflow", "cd8_outflow", "P", "C"):
        assert k in rd
    assert rd["macrophage_inflow"] > 0 and rd["nk_inflow"] > 0 and rd["cd8_inflow"] > 0
    assert rd["P"] == pytest.approx(out["ode_state"]["P"])
    # CD8 inflow has no homeostatic baseline: zero when APC P is zero is a pure function
    from viva_multicell_blueprint.mechanism.immune import cd8_inflow
    assert cd8_inflow(0.0, p._consts) == 0.0
