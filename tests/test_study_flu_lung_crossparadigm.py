"""Behavior test for the CAPSTONE study `flu-lung-crossparadigm` (spec §9.3).

Runs the FULL Sego 2022 influenza blueprint on BOTH backends at a tractable
reduced scale and computes the three comparison families -- but ALWAYS against a
NULL CONTROL: the same-backend seed-to-seed spread. The scientific question is not
"do the backends differ?" (they will, stochastically) but "do they differ MORE
than two runs of the SAME backend with different RNG seeds?". So we compute:

  * same-backend baseline: CPM-vs-CPM fate agreement over >=3 rules-seeds;
  * cross-backend: CPM-vs-vertex fate agreement;
  * and assert the cross-backend number is NOT meaningfully worse than the
    same-backend baseline -- i.e. the shared-mechanism architecture makes the
    model backend-neutral to within seed noise.

It also (2) checks the final-frame cell areas match across backends (so the
spatial comparison is NOT representation-confounded -- the vertex Voronoi faces
relax to the target by the end), (3) checks the vertex trajectory lies within the
CPM stochastic I-count envelope (a non-circular dynamics check), and renders the
three figures under the inline cap.
"""
from __future__ import annotations

import copy
import itertools
from pathlib import Path

import numpy as np

from viva_multicell_blueprint.blueprint.model import Blueprint
from viva_multicell_blueprint.compile._shared import resolve_initial_state
from viva_multicell_blueprint.compare import metrics
from viva_multicell_blueprint.compare.identity import IdentityRegistry
from viva_multicell_blueprint.compare.raster import grid_shape, rasterize, type_codes
from viva_multicell_blueprint.compare.runner import run_backend
from viva_multicell_blueprint.viz import divergence, dynamics_overlay, spatial_playback
from viva_multicell_blueprint.viz._common import HTML_CAP, MAX_FRAMES

SEGO = Path(__file__).resolve().parents[1] / "workspace" / "blueprints" / "influenza_sego2022.yaml"
ASSETS = Path(__file__).resolve().parents[1] / "workspace" / "reports" / "assets"
N = 150                      # CI-reduced scale (the regen script reports the N=400 figures)
CPM_SEEDS = (0, 1, 2)       # >=3 seeds for the stochastic CPM side (rigor spine)
GRID = {"extent_um": [50, 50], "grid_resolution_um": 1}


def _run(backend, seed, n):
    d = copy.deepcopy(Blueprint.load(SEGO).to_dict())
    d.setdefault("mechanics_mapping", {})["rules_seed"] = seed
    return run_backend(Blueprint.from_dict(d), backend, n)


def _agree(a, b, ids):
    reg = IdentityRegistry.from_backends(ids, a.identity, b.identity, getattr(b, "residual", {}) or {})
    return metrics.per_cell_agreement(a.frames, b.frames, reg), reg


def _final_mean_area(run):
    f = run.frames[-1]
    return float(np.mean([s for cid, s in f["sizes"].items() if f["alive"].get(cid)]))


def test_capstone_backend_neutral_within_seed_noise_and_figures():
    ids = [str(c["id"]) for c in resolve_initial_state(Blueprint.load(SEGO))]
    cpm = {s: _run("cpm", s, N) for s in CPM_SEEDS}
    vtx = _run("vertex", 0, N)
    for r in list(cpm.values()) + [vtx]:
        assert len(r.frames) == N + 1          # both backends run the full model end-to-end

    # --- NULL CONTROL: same-backend seed-to-seed fate agreement ---
    same = [_agree(cpm[i], cpm[j], ids)[0]["mean"] for i, j in itertools.combinations(CPM_SEEDS, 2)]
    same_mean = float(np.mean(same))
    # --- family 1: cross-backend fate agreement, judged RELATIVE to the null control ---
    cross_pca, reg_cross = _agree(cpm[0], vtx, ids)
    cross_mean = cross_pca["mean"]
    assert cross_pca["n_cohort"] == 48 and cross_pca["per_step"][0] == 1.0
    # HEADLINE: cross-backend fate agreement is NOT meaningfully worse than the
    # same-backend seed baseline -> backend-neutral to within seed noise (not a
    # substrate-driven divergence). (No lower bound claim beyond the baseline.)
    assert cross_mean >= same_mean - 0.05, (cross_mean, same_mean)
    assert metrics.lineage_agreement(reg_cross)["n_roots"] == 48   # no division rule

    # --- family 2: dynamics vs the CPM stochastic I-count envelope (non-circular) ---
    tc = lambda r: np.array(metrics.type_counts(r.frames).get("I", [0] * (N + 1)), float)
    M = np.vstack([tc(cpm[s]) for s in CPM_SEEDS])
    lo, hi = M.min(0), M.max(0)
    vI = tc(vtx)
    inband = float(np.mean([lo[t] - 1e-9 <= vI[t] <= hi[t] + 1e-9 for t in range(N + 1)]))
    assert inband >= 0.8, inband          # vertex trajectory lies within the CPM envelope
    band = {"I cells": [(float(t), float(M.mean(0)[t]), float(lo[t]), float(hi[t]))
                        for t in range(N + 1)]}
    dyn = metrics.dynamics_distance(cpm[0].frames, vtx.frames)
    assert "type_counts_total" in dyn and "virus" in dyn.get("field_integrals_rms", {})

    # --- family 3: spatial pattern, with the representation confound CONTROLLED ---
    # final-frame mean cell areas must match (vertex Voronoi faces relax to target),
    # so the occupancy comparison is a fair morphology comparison, not representation.
    area_ratio = _final_mean_area(vtx) / _final_mean_area(cpm[0])
    assert 0.6 < area_ratio < 1.6, area_ratio
    codes = type_codes(*[cpm[s].frames for s in CPM_SEEDS], vtx.frames)
    grid = dict(GRID, type_codes=codes)
    a, b = rasterize(cpm[0].frames[-1], grid), rasterize(vtx.frames[-1], grid)
    assert a.shape == grid_shape(grid)
    sp_c, sp_v = metrics.spatial_pattern(a, codes), metrics.spatial_pattern(b, codes)
    assert sp_c["I"] is not None and sp_v["I"] is not None   # both have an infected patch
    iou_I = metrics.iou(a, b, codes, ["I"])
    assert 0.0 <= iou_I <= 1.0 and -1.0 <= metrics.ssim(a, b) <= 1.0

    # --- figures: render all three, validate under the inline cap, save artifacts ---
    figs = {
        "capstone_spatial_playback": spatial_playback(cpm[0].frames, vtx.frames, GRID),
        "capstone_dynamics_overlay": dynamics_overlay(
            {"cpm": cpm[0].frames, "vertex": vtx.frames}, band),
        "capstone_divergence": divergence(cpm[0].frames, vtx.frames, reg_cross, GRID),
    }
    ASSETS.mkdir(parents=True, exist_ok=True)
    for name, fig in figs.items():
        assert len(fig.frames) <= MAX_FRAMES
        html = fig.to_html(include_plotlyjs="cdn")
        assert len(html.encode()) < HTML_CAP
        (ASSETS / f"{name}.html").write_text(html, encoding="utf-8")
