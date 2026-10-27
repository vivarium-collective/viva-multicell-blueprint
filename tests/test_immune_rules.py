import numpy as np
import pytest

from viva_multicell_blueprint.core import build_core
from viva_multicell_blueprint.mechanism.immune import (
    ImmuneRecruitmentProcess, macrophage_secretion_scale, poisson_inflow_count,
    uninfected_il10_scale)


def test_il10_hill_secretion_scale():
    g1, g2, d2, sig = 498.2, 5462.06, 428.25, 2.44
    s0 = macrophage_secretion_scale(0.0, sig, g1, g2, d2)
    vals = [macrophage_secretion_scale(L, sig, g1, g2, d2) for L in (0.0, 1.0, 10.0, 1e3)]
    assert s0 == pytest.approx(sig / (sig + g2 / d2))
    assert all(a > b for a, b in zip(vals, vals[1:]))  # self-limiting
    assert macrophage_secretion_scale(-5.0, sig, g1, g2, d2) == s0
    assert macrophage_secretion_scale(1.0, 0.0, g1, g2, d2) == 0.0
    assert uninfected_il10_scale(0.25) == 0.75 and uninfected_il10_scale(2.0) == 0.0


def test_recruitment_emits_spawn_events():
    p = ImmuneRecruitmentProcess(
        {"seed": 1, "types": {"macrophage_inflow": "M", "nk_inflow": "NK", "cd8_inflow": "CD8"},
         "id_prefix": "rec", "start_index": 100, "max_per_step": 50}, core=build_core())
    out = p.update({"recruit_drivers": {"macrophage_inflow": 3.0, "nk_inflow": 0.0,
                                         "cd8_inflow": 2.0}}, 1)
    ev = out["spawns"]
    assert ev and all(set(e) >= {"cell_id", "type", "parent"} for e in ev)
    assert {e["type"] for e in ev} <= {"M", "CD8"}
    ids = [e["cell_id"] for e in ev]
    assert len(set(ids)) == len(ids)
    out2 = p.update({"recruit_drivers": {"macrophage_inflow": 3.0}}, 1)
    assert not set(e["cell_id"] for e in out2["spawns"]) & set(ids)  # ids never reused


def test_recruitment_zero_rate_none_and_deterministic():
    cfg = {"seed": 7, "types": {"macrophage_inflow": "M"}}
    a = ImmuneRecruitmentProcess(cfg, core=build_core())
    b = ImmuneRecruitmentProcess(cfg, core=build_core())
    assert a.update({"recruit_drivers": {"macrophage_inflow": 0.0}}, 1)["spawns"] == []
    r = {"recruit_drivers": {"macrophage_inflow": 4.0}}
    assert a.update(r, 1) == b.update(r, 1)
    assert poisson_inflow_count(0.0, np.random.default_rng(0)) == 0
