import numpy as np

from viva_multicell_blueprint.compare.identity import IdentityRegistry
from viva_multicell_blueprint.compare.metrics import (
    dynamics_distance, field_match, iou, lineage_agreement, per_cell_agreement,
    spatial_pattern, ssim)
from viva_multicell_blueprint.compare.raster import rasterize, type_codes

GRID = {"extent_um": [40.0, 40.0], "grid_resolution_um": 1.0}
IDS = ["a", "b", "c"]


def frame(types, alive=None, shift=0.0):
    pos = {"a": [10 + shift, 10], "b": [20 + shift, 20], "c": [30 + shift, 30]}
    return {"positions": pos, "sizes": {k: 40.0 for k in IDS}, "types": dict(types),
            "alive": alive or {k: True for k in IDS}}


def reg():
    return IdentityRegistry(IDS, {i: n for n, i in enumerate(IDS)}, {i: n for n, i in enumerate(IDS)})


def test_identical_runs_agreement_one():
    run = [frame({"a": "H", "b": "I", "c": "H"}), frame({"a": "I", "b": "I", "c": "D"})]
    assert per_cell_agreement(run, run, reg())["mean"] == 1.0
    codes = type_codes(run)
    m = rasterize(run[1], dict(GRID, type_codes=codes))
    assert iou(m, m) == 1.0 and ssim(m, m) == 1.0
    assert dynamics_distance(run, run)["type_counts_total"] == 0.0


def test_one_side_death_counts_as_disagreement():
    t = {"a": "H", "b": "I", "c": "H"}
    A = [frame(t, alive={"a": False, "b": True, "c": True})]
    B = [frame(t)]
    r = per_cell_agreement(A, B, reg())
    assert r["per_step"] == [2 / 3]
    # both dead -> agree
    assert per_cell_agreement(A, A, reg())["mean"] == 1.0


def test_shifted_pattern_metrics():
    t = {"a": "I", "b": "H", "c": "H"}
    codes = type_codes([frame(t)])
    g = dict(GRID, type_codes=codes)
    mA, mB = rasterize(frame(t), g), rasterize(frame(t, shift=4.0), g)
    pA, pB = spatial_pattern(mA, codes)["I"], spatial_pattern(mB, codes)["I"]
    assert pA["area"] == pB["area"]
    assert abs((pB["centroid"][1] - pA["centroid"][1]) - 4.0) < 1e-9
    assert abs(pA["rg"] - pB["rg"]) < 1e-9
    assert spatial_pattern(mA, codes)["D"] is None
    assert 0.0 < iou(mA, mB) < 1.0 and ssim(mA, mB) < 1.0


def test_lineage_and_field_match():
    r = reg()
    r.on_division("a", "a1", "cpm"); r.on_division("a", "a2", "vertex")
    r.on_division("b", "b1", "cpm")
    la = lineage_agreement(r)
    assert abs(la["root_agreement"] - 2 / 3) < 1e-9 and la["n_born"] == {"cpm": 2, "vertex": 1}
    f = np.random.default_rng(0).random((8, 8))
    assert field_match(f, f)["match"] and not field_match(f, f + 0.1)["match"]
