"""CPM vs vertex trajectories per observable with acceptance-band ribbons behind."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..compare.metrics import type_counts
from ._common import BACKEND_COLORS, MAX_FRAMES, decimate_indices, rnd


def _observables(run):
    out = {f"{t} cells": v for t, v in type_counts(run).items()}
    names = sorted(run[0].get("field_integrals", {})) if run else []
    for k in names:
        out[k] = np.array([f.get("field_integrals", {}).get(k, np.nan) for f in run], float)
    return out


def dynamics_overlay(runs, bands=None, *, dt_days=1.0, max_points=200, title="Dynamics: CPM vs vertex"):
    """runs = {"cpm": run, "vertex": run}. bands = {observable: [(t_days, value, lo, hi)]},
    keyed by the observable label ("I cells", a field name, ...); a bare type name also matches."""
    bands = bands or {}
    obs = {b: _observables(r) for b, r in runs.items()}
    names = sorted(set().union(*[set(o) for o in obs.values()]))
    fig = make_subplots(rows=len(names), cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        subplot_titles=names)
    for row, name in enumerate(names, start=1):
        bkey = name if name in bands else name.replace(" cells", "")
        if bkey in bands:
            b = sorted(bands[bkey])
            t = rnd([p[0] for p in b], 3)
            lo, hi = rnd([p[2] for p in b], 4), rnd([p[3] for p in b], 4)
            fig.add_trace(go.Scatter(x=t, y=hi, mode="lines", line=dict(width=0), hoverinfo="skip",
                                     showlegend=False, legendgroup="band"), row=row, col=1)
            fig.add_trace(go.Scatter(x=t, y=lo, mode="lines", line=dict(width=0), fill="tonexty",
                                     fillcolor="rgba(120,120,120,0.25)", name="Sego acceptance band",
                                     legendgroup="band", showlegend=(row == 1),
                                     hoverinfo="skip"), row=row, col=1)
        for backend, o in obs.items():
            if name not in o:
                continue
            y = o[name]
            idx = decimate_indices(len(y), max_points)
            fig.add_trace(go.Scatter(x=[round(i * dt_days, 4) for i in idx], y=rnd(y[idx], 4),
                                     mode="lines", name=backend, legendgroup=backend,
                                     showlegend=(row == 1),
                                     line=dict(color=BACKEND_COLORS.get(backend, "#333"))),
                          row=row, col=1)
        fig.update_yaxes(title_text="count" if name.endswith("cells") else "integral (a.u.)",
                         row=row, col=1)
    fig.update_xaxes(title_text="time (days)", row=len(names), col=1)
    fig.update_layout(title=title, height=260 * max(1, len(names)) + 80, template="plotly_white",
                      legend=dict(title="Backend"))
    return fig
