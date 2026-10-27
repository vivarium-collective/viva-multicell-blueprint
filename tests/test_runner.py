"""Task 18: the side-by-side run harness (compare.runner).

A thin, tested seam the studies call: compile a blueprint onto a backend, run it
for N coupled steps, and collect per-step frames shaped exactly as the
``compare`` metrics + ``viz`` figures consume (positions/sizes/types/alive +
split footprints + a field underlay + field integrals). ``run_both`` also builds
the shared :class:`IdentityRegistry` from the two backends.
"""
from __future__ import annotations

import numpy as np

from viva_multicell_blueprint.blueprint.model import Blueprint
from viva_multicell_blueprint.compare import metrics
from viva_multicell_blueprint.compare.runner import BackendRun, run_backend, run_both
from viva_multicell_blueprint.viz import divergence, dynamics_overlay, spatial_playback
from viva_multicell_blueprint.viz._common import HTML_CAP, MAX_FRAMES

# Minimal self-contained blueprint: a 2x2 epithelial sheet + one diffusing field,
# no rules/ODE -- keeps the harness unit test fast and backend-focused.
BP = {
    "domain": {"extent_um": [14, 14], "grid_resolution_um": 1, "boundary": "noflux"},
    "cell_types": [{"name": "H", "target_size": 25}, {"name": "I", "target_size": 25}],
    "initial_state": {"kind": "blocks",
                      "block": {"type": "H", "origin": [4, 4], "shape": [2, 2], "spacing": 5, "size": 25}},
    "fields": [{"name": "virus", "D": 0.1, "decay": 0.01,
                "dynamics": {"dt": 0.5, "substeps": 2},
                "secretion": [{"type": "H", "rate": 0.2}]}],
}


def _bp():
    return Blueprint.from_dict(dict(BP))


def test_run_backend_frames_shaped_for_compare_and_viz():
    run = run_backend(_bp(), "cpm", n_steps=4)
    assert isinstance(run, BackendRun) and run.backend == "cpm"
    assert len(run.frames) == 5                 # t0 + 4 coupled steps
    f = run.frames[1]
    for key in ("positions", "sizes", "types", "alive", "polygons", "occupancy",
                "field", "field_integrals", "frame"):
        assert key in f, key
    assert "virus" in f["field_integrals"]
    # the CPM footprints split into occupancy (not polygons)
    assert f["occupancy"] and not f["polygons"]
    assert np.asarray(f["field"]).ndim == 2
    # t0 frame reproduces the blueprint's initial types
    assert set(run.frames[0]["types"].values()) == {"H"}


def test_run_both_builds_registry_and_metrics_compute():
    cpm, vtx, reg = run_both(_bp(), n_steps=4)
    assert cpm.backend == "cpm" and vtx.backend == "vertex"
    assert reg.t0_cohort() <= set(cpm.identity)          # cohort is a subset of mapped ids
    agree = metrics.per_cell_agreement(cpm.frames, vtx.frames, reg)
    assert 0.0 <= agree["mean"] <= 1.0 and agree["n_cohort"] >= 0
    dyn = metrics.dynamics_distance(cpm.frames, vtx.frames)
    assert "type_counts_total" in dyn and dyn["type_counts_total"] >= 0.0
    # vertex footprints are polygons -> real occupancy IoU is computable
    from viva_multicell_blueprint.compare.raster import grid_shape, rasterize, type_codes
    grid = {"extent_um": [14, 14], "grid_resolution_um": 1,
            "type_codes": type_codes(cpm.frames, vtx.frames)}
    a = rasterize(cpm.frames[-1], grid)
    b = rasterize(vtx.frames[-1], grid)
    assert a.shape == grid_shape(grid)
    assert 0.0 <= metrics.iou(a, b) <= 1.0


def test_run_both_figures_render_under_inline_cap():
    cpm, vtx, reg = run_both(_bp(), n_steps=6)
    grid = {"extent_um": [14, 14], "grid_resolution_um": 1}
    figs = [
        spatial_playback(cpm.frames, vtx.frames, grid),
        dynamics_overlay({"cpm": cpm.frames, "vertex": vtx.frames}),
        divergence(cpm.frames, vtx.frames, reg, grid),
    ]
    for fig in figs:
        assert len(fig.frames) <= MAX_FRAMES
        assert len(fig.to_html(include_plotlyjs="cdn").encode()) < HTML_CAP
