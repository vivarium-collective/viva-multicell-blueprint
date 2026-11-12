import base64
import io

import numpy as np
from PIL import Image

from viva_multicell_blueprint.compare.identity import IdentityRegistry
from viva_multicell_blueprint.viz import (
    divergence, dynamics_overlay, single_playback, spatial_playback, spatial_playback_multi,
)
from viva_multicell_blueprint.viz._common import HTML_CAP, MAX_FRAMES
from viva_multicell_blueprint.viz.flipbook import Flipbook

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


def vrun(n, shift=0.0):
    """A run that also carries per-cell polygons (vertex / center-based shape)."""
    out = []
    for t in range(n):
        f = frame(t, shift)
        f["polygons"] = {c: [[cx - 3, cy - 3], [cx + 3, cy - 3], [cx + 3, cy + 3], [cx - 3, cy + 3]]
                         for c in IDS for cx, cy in [f["positions"][c]]}
        out.append(f)
    return out


def reg():
    return IdentityRegistry.from_backends(IDS, {c: i for i, c in enumerate(IDS)}, {c: i for i, c in enumerate(IDS)})


def _png(uri):
    """Decode a flipbook frame ``data:`` URI to a PIL image."""
    assert uri.startswith("data:image/png;base64,"), uri[:32]
    return Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1])))


# --- spatial playback: server-rendered image flipbooks (not Plotly) ----------
def test_playback_is_a_standalone_two_panel_flipbook():
    fb = spatial_playback(run(6), run(6, 2.0), GRID)
    assert isinstance(fb, Flipbook)
    assert len(fb.frames) == 6
    html = fb.to_html(include_plotlyjs="cdn")  # kwarg accepted + ignored (drop-in)
    # self-contained: our flipbook marker, no Plotly, and no external fetch at all
    # (every frame is an embedded data: URI, so no CDN / http(s) reference).
    assert "vmb-flipbook" in html and "Plotly.newPlot" not in html
    assert "http://" not in html and "https://" not in html
    # every frame is a real PNG of identical pixel size, so the <img> never resizes
    sizes = {_png(u).size for u in fb.frames}
    assert len(sizes) == 1


def test_multi_panel_flipbook_is_wider_than_single():
    runs = {"cpm": run(6), "vertex": vrun(6, 2.0), "munk": vrun(6, -2.0)}
    fb3 = spatial_playback_multi(runs, GRID)
    fb1 = single_playback(run(6), GRID, backend="cpm")
    assert isinstance(fb3, Flipbook) and len(fb3.frames) == 6
    # three side-by-side panels render strictly wider than a single panel
    assert _png(fb3.frames[0]).size[0] > _png(fb1.frames[0]).size[0]


def test_flipbook_frames_draw_colored_cells():
    # CPM (voxel) + vertex (polygon) panels must both paint non-background colour.
    fb = spatial_playback(run(6), vrun(6, 2.0), GRID)
    arr = np.asarray(_png(fb.frames[-1]).convert("RGB"))
    uniq = np.unique(arr.reshape(-1, 3), axis=0)
    assert len(uniq) > 5, "frame looks blank"


def test_playback_halving_fallback_and_bottom_out_warns():
    import warnings
    a, b = run(40), run(40, 2.0)
    full = spatial_playback(a, b, GRID, budget=10 ** 9)
    # a budget well under the full size forces the frame count to halve down
    budget = len(full.to_html().encode()) // 3
    small = spatial_playback(a, b, GRID, budget=budget)
    assert len(small.frames) < len(full.frames) <= MAX_FRAMES
    assert len(small.to_html().encode()) < budget
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        spatial_playback(a, b, GRID, budget=1000)  # unsatisfiable -> bottoms out, warns
    assert any(issubclass(x.category, RuntimeWarning) for x in w)


# --- divergence + dynamics overlays stay interactive Plotly figures ----------
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
    n = 400  # long run -> must decimate (flipbook frames AND Plotly frames)
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
