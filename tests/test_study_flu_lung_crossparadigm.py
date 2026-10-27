"""Behavior test for the CAPSTONE study `flu-lung-crossparadigm` (spec §9.3).

Runs the FULL Sego 2022 influenza blueprint on BOTH backends at a tractable
reduced scale and computes the three comparison families:

  1. per-cell agreement over the t0 cohort (identity registry),
  2. dynamics trajectories (type counts + field integrals) with a Sego-side
     acceptance band = the >=3-seed CPM stochastic ensemble envelope (rigor spine),
  3. spatial pattern (patch area/centroid/Rg/front) + occupancy IoU/SSIM on the
     real footprints.

It renders the three figures (spatial_playback, dynamics_overlay, divergence) and
checks they are valid and under the workbench inline cap. Divergence between the
backends is a FINDING (identical shared mechanism, different mechanical substrate),
not a failure -- the test asserts the pipeline completes and the metrics are
well-formed, and that the substrates DO diverge spatially (the headline result).
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from viva_multicell_blueprint.blueprint.model import Blueprint
from viva_multicell_blueprint.compare import metrics
from viva_multicell_blueprint.compare.raster import grid_shape, rasterize, type_codes
from viva_multicell_blueprint.compare.runner import run_backend, run_both
from viva_multicell_blueprint.viz import divergence, dynamics_overlay, spatial_playback
from viva_multicell_blueprint.viz._common import HTML_CAP, MAX_FRAMES

SEGO = Path(__file__).resolve().parents[1] / "workspace" / "blueprints" / "influenza_sego2022.yaml"
ASSETS = Path(__file__).resolve().parents[1] / "workspace" / "reports" / "assets"
N = 80          # CI-reduced scale (the demonstration run in the study uses more steps)
GRID = {"extent_um": [50, 50], "grid_resolution_um": 1}


def _cpm_I_ensemble(bp, seeds, n):
    """>=3-seed CPM stochastic ensemble of the infected-cell count -> (mean,min,max)."""
    series = []
    for s in seeds:
        d = copy.deepcopy(bp.to_dict())
        d.setdefault("mechanics_mapping", {})["rules_seed"] = s
        r = run_backend(Blueprint.from_dict(d), "cpm", n)
        tc = metrics.type_counts(r.frames)
        series.append(np.array(tc.get("I", [0] * (n + 1)), float))
    M = np.vstack(series)
    return M.mean(0), M.min(0), M.max(0)


def test_capstone_three_families_and_figures():
    bp = Blueprint.load(SEGO)
    cpm, vtx, reg = run_both(bp, n_steps=N)
    assert len(cpm.frames) == len(vtx.frames) == N + 1

    # --- family 1: per-cell agreement over the t0 cohort ---
    pca = metrics.per_cell_agreement(cpm.frames, vtx.frames, reg)
    assert pca["n_cohort"] == 48 and pca["n_vanished"] == 0
    assert pca["per_step"][0] == 1.0                 # identical at t0 by construction
    assert 0.8 <= pca["mean"] <= 1.0
    lin = metrics.lineage_agreement(reg)
    assert lin["n_roots"] == 48                      # no divisions in this model

    # --- family 2: dynamics + Sego-side (CPM ensemble) acceptance band ---
    dyn = metrics.dynamics_distance(cpm.frames, vtx.frames)
    assert "type_counts_total" in dyn and "field_integrals_rms" in dyn
    assert "virus" in dyn["field_integrals_rms"]
    mean, lo, hi = _cpm_I_ensemble(bp, (0, 1, 2), N)
    band = {"I cells": [(float(t), float(mean[t]), float(lo[t]), float(hi[t]))
                        for t in range(N + 1)]}
    assert all(lo[t] <= hi[t] for t in range(N + 1))  # well-formed band

    # --- family 3: spatial pattern + occupancy IoU/SSIM on real footprints ---
    codes = type_codes(cpm.frames, vtx.frames)
    grid = dict(GRID, type_codes=codes)
    a, b = rasterize(cpm.frames[-1], grid), rasterize(vtx.frames[-1], grid)
    assert a.shape == grid_shape(grid)
    sp_c, sp_v = metrics.spatial_pattern(a, codes), metrics.spatial_pattern(b, codes)
    assert sp_c["I"] is not None and sp_v["I"] is not None   # both have an infected patch
    iou_all = metrics.iou(a, b)
    iou_I = metrics.iou(a, b, codes, ["I"])
    ssim = metrics.ssim(a, b)
    assert 0.0 <= iou_all <= 1.0 and 0.0 <= iou_I <= 1.0 and -1.0 <= ssim <= 1.0
    # HEADLINE finding: the two substrates DIVERGE spatially even from the shared
    # mechanism -- the infected-patch footprints do not coincide pixel-for-pixel.
    assert iou_I < 0.9

    # --- figures: render all three, validate under the inline cap ---
    figs = {
        "capstone_spatial_playback": spatial_playback(cpm.frames, vtx.frames, GRID),
        "capstone_dynamics_overlay": dynamics_overlay(
            {"cpm": cpm.frames, "vertex": vtx.frames}, band),
        "capstone_divergence": divergence(cpm.frames, vtx.frames, reg, GRID),
    }
    ASSETS.mkdir(parents=True, exist_ok=True)
    for name, fig in figs.items():
        assert len(fig.frames) <= MAX_FRAMES
        html = fig.to_html(include_plotlyjs="cdn")
        assert len(html.encode()) < HTML_CAP
        (ASSETS / f"{name}.html").write_text(html, encoding="utf-8")   # produced artifact
