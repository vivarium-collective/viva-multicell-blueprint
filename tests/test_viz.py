import numpy as np
import plotly.graph_objects as go

from viva_multicell_blueprint.compare.identity import IdentityRegistry
from viva_multicell_blueprint.viz import divergence, dynamics_overlay, spatial_playback
from viva_multicell_blueprint.viz._common import HTML_CAP, MAX_FRAMES

GRID = {"extent_um": [60.0, 60.0], "grid_resolution_um": 1.0}
IDS = [f"c{i}" for i in range(12)]


def frame(t, shift=0.0):
    pos = {c: [8 + 12 * (k % 4) + shift, 10 + 14 * (k // 4)] for k, c in enumerate(IDS)}
    types = {c: ("I" if (k + t) % 3 == 0 else "H") for k, c in enumerate(IDS)}
    alive = {c: not (k == 0 and t > 5) for k, c in enumerate(IDS)}
    return {"positions": pos, "sizes": {c: 60.0 for c in IDS}, "types": types, "alive": alive,
            "field": np.full((60, 60), 0.1 * t), "field_integrals": {"ifn": 0.5 * t}}


def run(n, shift=0.0):
    return [frame(t, shift) for t in range(n)]


def reg():
    return IdentityRegistry(IDS, {c: i for i, c in enumerate(IDS)}, {c: i for i, c in enumerate(IDS)})


def test_playback_has_two_synced_panels():
    fig = spatial_playback(run(6), run(6, 2.0), GRID)
    assert isinstance(fig, go.Figure)
    assert len({tr.xaxis for tr in fig.data if tr.type == "heatmap"}) == 2  # two panels
    assert len(fig.layout.sliders) == 1 and len(fig.frames) == 6
    # every frame drives BOTH panels' traces
    assert all(len(f.traces) >= 2 and len(f.data) == len(f.traces) for f in fig.frames)
    assert len(fig.layout.sliders[0].steps) == len(fig.frames)


def test_dynamics_overlay_has_band_ribbons():
    runs = {"cpm": run(10), "vertex": run(10, 2.0)}
    bands = {"I cells": [(t, 4.0, 2.0, 6.0) for t in range(10)]}
    fig = dynamics_overlay(runs, bands)
    fills = [i for i, t in enumerate(fig.data) if t.fill == "tonexty"]
    lines = [i for i, t in enumerate(fig.data) if t.name in ("cpm", "vertex")]
    assert fills
    assert fig.data[fills[0]].name == "Sego acceptance band"
    assert fills[0] < min(i for i in lines if fig.data[i].xaxis == fig.data[fills[0]].xaxis)  # behind
    assert len(dynamics_overlay(runs).data) < len(fig.data)


def test_divergence_curve_and_map():
    fig = divergence(run(8), run(8, 3.0), reg(), GRID)
    assert len(fig.frames) == 8 and any(t.type == "heatmap" for t in fig.data)


def test_figure_under_inline_cap():
    n = 400  # long run -> must decimate
    a, b = run(n), run(n, 2.0)
    for fig in (spatial_playback(a, b, GRID), divergence(a, b, reg(), GRID),
                dynamics_overlay({"cpm": a, "vertex": b})):
        assert len(fig.frames) <= MAX_FRAMES
        size = len(fig.to_html(include_plotlyjs="cdn").encode())
        print("size", size)
        assert size < HTML_CAP


def test_legend_entries_exactly_once_regardless_of_row():
    runs = {"cpm": run(10), "vertex": run(10, 2.0)}
    bands = {"I cells": [(t, 4.0, 2.0, 6.0) for t in range(10)]}  # band on a non-first row
    fig = dynamics_overlay(runs, bands)
    for nm in ("Sego acceptance band", "cpm", "vertex"):
        assert sum(1 for t in fig.data if t.name == nm and t.showlegend) == 1, nm


def test_playback_halving_fallback_and_bottom_out_warns():
    import warnings
    a, b = run(40), run(40, 2.0)
    full = spatial_playback(a, b, GRID, budget=10 ** 9)
    small = spatial_playback(a, b, GRID, budget=700_000)
    assert len(small.frames) < len(full.frames) <= MAX_FRAMES
    assert len(small.to_json()) < 700_000
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        spatial_playback(a, b, GRID, budget=1000)
    assert any(issubclass(x.category, RuntimeWarning) for x in w)
