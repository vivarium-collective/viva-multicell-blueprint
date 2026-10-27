"""Synchronized two-panel playback: CPM | vertex state maps, one shared time slider."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..compare.raster import grid_shape, rasterize, type_codes
from ._common import (MAX_FRAMES, decimate_indices, discrete_colorscale, rnd, state_color,
                      times)


def _field_underlay(frame, shape):
    """Optional frame['field'] heat underlay. FieldProcess grids are [nx, ny];
    the raster is (ny, nx), so transpose and assert orientation."""
    f = frame.get("field")
    if f is None:
        return None
    f = np.asarray(f, float)
    if f.shape == shape[::-1]:      # (nx, ny) -> (ny, nx)
        f = f.T
    assert f.shape == shape, f"field {f.shape} incompatible with raster {shape}"
    return rnd(f, 3).astype(np.float32)


def _state_z(frame, grid):
    # int8 (compact bdata); 0 = empty -> transparent in the colorscale (field shows through)
    return rasterize(frame, grid).astype(np.int8)


def _spatial_playback(run_cpm, run_vertex, grid_spec, *, dt_days=1.0, max_frames=MAX_FRAMES,
                     title="Spatial playback: CPM vs vertex"):
    codes = type_codes(run_cpm, run_vertex)
    grid = dict(grid_spec, type_codes=codes)
    shape = grid_shape(grid)
    n = min(len(run_cpm), len(run_vertex))
    idx = decimate_indices(n, max_frames)
    ts = times(n, dt_days, idx)
    colors = {t: state_color(t, i) for i, t in enumerate(codes)}
    cs = discrete_colorscale(codes, colors)
    res = float(grid["grid_resolution_um"])
    xs = rnd(np.arange(shape[1]) * res + res / 2, 2).tolist()
    ys = rnd(np.arange(shape[0]) * res + res / 2, 2).tolist()
    has_field = any(r[0].get("field") is not None for r in (run_cpm, run_vertex))

    fig = make_subplots(rows=1, cols=2, subplot_titles=("CPM (voxel map)", "Vertex (polygon map)"),
                        horizontal_spacing=0.08)
    runs = (run_cpm, run_vertex)
    anim = []   # trace indices updated by frames
    for c, run in enumerate(runs, start=1):
        if has_field:
            fz = _field_underlay(run[idx[0]], shape)
            fig.add_trace(go.Heatmap(x=xs, y=ys, z=None if fz is None else fz,
                                     colorscale="Greys", opacity=0.6, showscale=False,
                                     hoverinfo="skip", name="field"), row=1, col=c)
            anim.append(len(fig.data) - 1)
        fig.add_trace(go.Heatmap(x=xs, y=ys, z=_state_z(run[idx[0]], grid),
                                 colorscale=cs, zmin=0, zmax=len(codes),
                                 showscale=False, xgap=0, ygap=0, name="state",
                                 hovertemplate="x=%{x} um<br>y=%{y} um<extra></extra>"), row=1, col=c)
        anim.append(len(fig.data) - 1)
        fig.update_xaxes(title_text="x (um)", row=1, col=c)
        fig.update_yaxes(title_text="y (um)", row=1, col=c, scaleanchor=f"x{'' if c == 1 else c}",
                         constrain="domain")
    # legend-only entries (static, not animated)
    for t in codes:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=t,
                                 marker=dict(size=10, color=colors[t], symbol="square")),
                      row=1, col=1)

    frames = []
    for k, i in enumerate(idx):
        data = []
        for run in runs:
            if has_field:
                fz = _field_underlay(run[i], shape)
                data.append(go.Heatmap(z=None if fz is None else fz))
            data.append(go.Heatmap(z=_state_z(run[i], grid)))
        frames.append(go.Frame(name=str(k), data=data, traces=anim))
    fig.frames = frames
    steps = [dict(method="animate", label=f"{t:g}",
                  args=[[str(k)], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                       transition=dict(duration=0))])
             for k, t in enumerate(ts)]
    fig.update_layout(
        title=title,
        sliders=[dict(active=0, steps=steps, currentvalue=dict(prefix="time (days): "),
                      pad=dict(t=40))],
        updatemenus=[dict(type="buttons", direction="left", x=0.0, y=-0.05, showactive=False,
                          buttons=[dict(label="Play", method="animate",
                                        args=[None, dict(frame=dict(duration=250, redraw=True),
                                                         fromcurrent=True, transition=dict(duration=0))]),
                                   dict(label="Pause", method="animate",
                                        args=[[None], dict(mode="immediate",
                                                           frame=dict(duration=0, redraw=False),
                                                           transition=dict(duration=0))])])],
        legend=dict(title="Cell state"), height=520, template="plotly_white")
    return fig


def spatial_playback(run_cpm, run_vertex, grid_spec, *, max_frames=MAX_FRAMES, budget=2_200_000, **kw):
    """Two synchronized panels (CPM | vertex), shared slider. Frames are decimated to
    <= max_frames and further halved until the serialized figure fits ``budget`` bytes
    (headroom under the workbench's 2.5 MB per-figure inline cap)."""
    mf = max_frames
    while True:
        fig = _spatial_playback(run_cpm, run_vertex, grid_spec, max_frames=mf, **kw)
        if mf <= 2 or len(fig.to_json()) < budget:
            return fig
        mf = max(2, mf // 2)
