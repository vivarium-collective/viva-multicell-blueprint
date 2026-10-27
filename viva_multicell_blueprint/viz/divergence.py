"""Per-cell agreement curve over time + spatial divergence map per checkpoint."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..compare.metrics import per_cell_agreement
from ..compare.raster import grid_shape, rasterize, type_codes
from ._common import MAX_FRAMES, decimate_indices, rnd, times


def _diff_map(fa, fb, grid):
    """int8: 0 = empty in both, 1 = same state, 2 = backends disagree (incl. one-sided)."""
    a, b = rasterize(fa, grid), rasterize(fb, grid)
    z = np.where(a == b, 1, 2).astype(np.int8)
    z[(a == 0) & (b == 0)] = 0
    return z


def divergence(run_cpm, run_vertex, registry, grid_spec, *, dt_days=1.0, max_frames=MAX_FRAMES,
               title="Divergence: where CPM and vertex disagree"):
    n = min(len(run_cpm), len(run_vertex))
    agree = per_cell_agreement(run_cpm, run_vertex, registry)["per_step"]
    idx = decimate_indices(n, max_frames)
    ts = times(n, dt_days, idx)
    grid = dict(grid_spec, type_codes=type_codes(run_cpm, run_vertex))
    shape = grid_shape(grid)
    res = float(grid["grid_resolution_um"])
    xs = rnd(np.arange(shape[1]) * res + res / 2, 2).tolist()
    ys = rnd(np.arange(shape[0]) * res + res / 2, 2).tolist()

    fig = make_subplots(rows=1, cols=2, column_widths=[0.5, 0.5], horizontal_spacing=0.1,
                        subplot_titles=("Per-cell agreement (t0 cohort)", "Spatial divergence at checkpoint"))
    fig.add_trace(go.Scatter(x=ts, y=rnd([agree[i] for i in idx], 4), mode="lines+markers",
                             name="agreement", line=dict(color="#0072B2")), row=1, col=1)
    fig.add_trace(go.Scatter(x=[ts[0]], y=[rnd(agree[idx[0]], 4)], mode="markers", name="checkpoint",
                             marker=dict(size=13, color="#D55E00", symbol="diamond")), row=1, col=1)
    cur = len(fig.data) - 1
    zfun = lambda i: _diff_map(run_cpm[i], run_vertex[i], grid)
    fig.add_trace(go.Heatmap(x=xs, y=ys, z=zfun(idx[0]), zmin=0, zmax=2, showscale=False,
                             colorscale=[[0, "rgba(0,0,0,0)"], [0.25, "rgba(0,0,0,0)"], [0.25, "#56B4E9"], [0.75, "#56B4E9"], [0.75, "#D55E00"], [1, "#D55E00"]],
                             hovertemplate="x=%{x} um<br>y=%{y} um<extra></extra>", name="map"),
                  row=1, col=2)
    mapi = len(fig.data) - 1
    for lab, col in (("backends agree", "#56B4E9"), ("backends disagree", "#D55E00")):
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=lab,
                                 marker=dict(size=10, color=col, symbol="square")), row=1, col=2)
    frames = [go.Frame(name=str(k), traces=[cur, mapi],
                       data=[go.Scatter(x=[ts[k]], y=[rnd(agree[i], 4)]), go.Heatmap(z=zfun(i))])
              for k, i in enumerate(idx)]
    fig.frames = frames
    steps = [dict(method="animate", label=f"{t:g}",
                  args=[[str(k)], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                       transition=dict(duration=0))]) for k, t in enumerate(ts)]
    fig.update_xaxes(title_text="time (days)", row=1, col=1)
    fig.update_yaxes(title_text="fraction of cells in same state", range=[0, 1.05], row=1, col=1)
    fig.update_xaxes(title_text="x (um)", row=1, col=2)
    fig.update_yaxes(title_text="y (um)", scaleanchor="x2", constrain="domain", row=1, col=2)
    fig.update_layout(title=title, template="plotly_white", height=520,
                      sliders=[dict(active=0, steps=steps, currentvalue=dict(prefix="checkpoint (days): "),
                                    pad=dict(t=40))])
    return fig
