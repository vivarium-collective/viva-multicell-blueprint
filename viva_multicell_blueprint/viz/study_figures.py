"""Per-study summary figures (spec §10) for the three non-capstone studies.

The capstone (``flu-lung-crossparadigm``) uses the cross-paradigm trio
(:func:`spatial_playback`, :func:`dynamics_overlay`, :func:`divergence`). These
builders cover the other three studies' headline results:

* :func:`adhesion_sweep`  -- J<->tension calibration (adhesion-calibration)
* :func:`dose_response`   -- initial-load dose response (dose-response-crossparadigm)
* :func:`t0_fidelity`     -- t0 centroid residuals (initial-state-fidelity)

Each takes already-computed arrays (not runs), so it is pure and cheaply tested;
``scripts/render_study_figures.py`` does the simulation and feeds these.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ._common import BACKEND_COLORS, rnd

_CPM = BACKEND_COLORS["cpm"]        # blue
_VERTEX = BACKEND_COLORS["vertex"]  # vermillion
_GREY = "#999999"


def adhesion_sweep(scales, vertex_mean, vertex_cv, cpm_mean, cpm_cv, *,
                   target_size=25.0, title="Adhesion calibration: J ↔ vertex tension"):
    """Vertex cell-shape statistics across a ``tension_scale`` sweep, against the
    CPM reference (direct J mapping) and the size target.

    scales/vertex_mean/vertex_cv are equal-length sequences; cpm_mean/cpm_cv are
    scalars (CPM has no tension knob). Two stacked panels: mean cell area and
    area CV vs the vertex tension scale. The mapping is directional, not pinned —
    the figure shows the vertex statistics barely move, bracketing the CPM value.
    """
    scales = [float(s) for s in scales]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("Mean cell area", "Area dispersion (CV)"))

    # Panel 1: mean area.
    fig.add_hrect(y0=target_size * 0.9, y1=target_size * 1.1, line_width=0,
                  fillcolor="rgba(150,150,150,0.12)", row=1, col=1)
    fig.add_hline(y=target_size, line=dict(color=_GREY, dash="dot"), row=1, col=1,
                  annotation_text=f"target size {target_size:g}", annotation_position="top left")
    fig.add_hline(y=float(cpm_mean), line=dict(color=_CPM, dash="dash"), row=1, col=1,
                  annotation_text=f"CPM {float(cpm_mean):.1f}", annotation_position="bottom right")
    fig.add_trace(go.Scatter(x=scales, y=rnd(vertex_mean, 2), mode="lines+markers",
                             name="vertex", legendgroup="vertex", showlegend=True,
                             line=dict(color=_VERTEX), marker=dict(size=9),
                             hovertemplate="scale %{x}<br>mean area %{y:.1f} µm²<extra></extra>"),
                  row=1, col=1)
    fig.update_yaxes(title_text="area (µm²)", row=1, col=1)

    # Panel 2: CV.
    fig.add_hline(y=float(cpm_cv), line=dict(color=_CPM, dash="dash"), row=2, col=1,
                  annotation_text=f"CPM CV {float(cpm_cv):.3f}", annotation_position="top right")
    fig.add_trace(go.Scatter(x=scales, y=rnd(vertex_cv, 4), mode="lines+markers",
                             name="vertex", legendgroup="vertex", showlegend=False,
                             line=dict(color=_VERTEX), marker=dict(size=9),
                             hovertemplate="scale %{x}<br>CV %{y:.3f}<extra></extra>"),
                  row=2, col=1)
    fig.update_yaxes(title_text="area CV", row=2, col=1)
    fig.update_xaxes(title_text="vertex tension scale (× J mapping)", type="log", row=2, col=1)
    fig.update_layout(title=title, height=620, template="plotly_white",
                      legend=dict(title="Backend"))
    return fig


def dose_response(doses, series, *, title="Dose response: initial viral load → final infection"):
    """Final infected-cell count vs initial dose, for each (backend, seed).

    doses: the dose axis (e.g. seeded infected fractions). series: a list of
    ``{"backend", "seed", "final_I"}`` dicts, each ``final_I`` aligned with
    ``doses``. Color encodes backend, line dash encodes seed; the shared upward
    trend is the directional result (higher load → more infection on both
    substrates).
    """
    doses = [float(d) for d in doses]
    dashes = ("solid", "dash", "dot", "dashdot")
    seeds = sorted({int(s["seed"]) for s in series})
    seed_dash = {s: dashes[i % len(dashes)] for i, s in enumerate(seeds)}
    seen_backend: set = set()
    fig = go.Figure()
    for s in series:
        backend, seed = s["backend"], int(s["seed"])
        color = BACKEND_COLORS.get(backend, "#333")
        show = backend not in seen_backend
        seen_backend.add(backend)
        fig.add_trace(go.Scatter(
            x=doses, y=[int(v) for v in s["final_I"]], mode="lines+markers",
            name=backend, legendgroup=backend, showlegend=show,
            line=dict(color=color, dash=seed_dash[seed]), marker=dict(size=8),
            hovertemplate=(f"{backend} · seed {seed}<br>dose %{{x}}"
                           "<br>final I %{y}<extra></extra>")))
    # A second legend for the seed→dash encoding (invisible anchor traces).
    for seed in seeds:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
                                 line=dict(color=_GREY, dash=seed_dash[seed]),
                                 name=f"seed {seed}", legendgroup=f"seed{seed}"))
    fig.update_layout(
        title=title, template="plotly_white",
        xaxis_title="initial infected fraction (dose)", yaxis_title="final infected-cell count",
        height=460, legend=dict(title="Backend · seed"))
    return fig


def t0_fidelity(residuals, *, type_ok=None, n_cells=None, cell_radius_um=2.8,
                title="Initial-state fidelity: t0 centroid residuals"):
    """Per-backend distribution of each cell's t0 centroid residual (µm).

    residuals: ``{"cpm": [...], "vertex": [...]}`` per-cell distances between the
    backend's emitted centroid and the blueprint's. CPM seeds cells exactly
    (≈0); the vertex Voronoi+collar is sub-cell. A reference line marks the cell
    radius so "sub-cell" is visually obvious; RMS (and optional type-match) are
    annotated per backend.
    """
    order = [b for b in ("cpm", "vertex") if b in residuals]
    fig = go.Figure()
    for b in order:
        r = np.asarray(residuals[b], float)
        rms = float(np.sqrt(np.mean(r ** 2))) if r.size else 0.0
        label = b
        if type_ok is not None and n_cells:
            label = f"{b}<br>RMS {rms:.3g} µm · types {int(type_ok.get(b, 0))}/{int(n_cells)}"
        else:
            label = f"{b}<br>RMS {rms:.3g} µm"
        fig.add_trace(go.Box(
            y=rnd(r, 4), name=label, legendgroup=b, showlegend=False,
            boxpoints="all", jitter=0.5, pointpos=0, marker=dict(color=BACKEND_COLORS.get(b, "#333")),
            line=dict(color=BACKEND_COLORS.get(b, "#333")),
            hovertemplate="%{y:.3g} µm<extra></extra>"))
    fig.add_hline(y=float(cell_radius_um), line=dict(color=_GREY, dash="dot"),
                  annotation_text=f"cell radius ≈ {cell_radius_um:g} µm",
                  annotation_position="top right")
    fig.update_layout(title=title, template="plotly_white",
                      yaxis_title="centroid residual (µm)", xaxis_title="backend",
                      height=460, boxgap=0.4)
    return fig


__all__ = ["adhesion_sweep", "dose_response", "t0_fidelity"]
