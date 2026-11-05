"""Synchronized two-panel playback: CPM voxel map | vertex polygon map, one shared
time slider.

The two backends have genuinely different spatial representations, and the figure
shows each in its native form — **entirely with scatter traces**, so the whole
animation runs with ``redraw=False``. That matters: an animated ``Heatmap`` can
only update under a full per-frame ``redraw``, which clears the plot to white
between frames (a disorienting strobe). Everything here is scatter, so Plotly
transitions the traces in place with no flash.

* **CPM** — a lattice model, drawn as filled voxel rectangles (``_type_fill_paths``
  run-length-encodes the type raster into one ``fill="toself"`` scatter per cell
  type) with a per-cell outline overlay (``_boundary_xy`` traces the edges of the
  cell-id raster, so touching same-type cells still read as distinct cells).
* **Vertex** — a polygon model, drawn as the actual cell polygons (the hexagonal
  sheet faces and overlay islands from ``frame['polygons']``), filled by cell state
  with visible cell edges.

Falls back to scatter voxel fills for the vertex panel when a run carries no
polygon footprints (e.g. the synthetic position-only frames used in unit tests).
The optional virus-field heat underlay was dropped from these videos: as an
animated heatmap it reintroduced the redraw strobe, and as a static t0 backdrop it
clashed with the (grey) M_latent cells.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..compare.raster import cell_id_raster, grid_shape, rasterize, type_codes
from ._common import MAX_FRAMES, decimate_indices, state_color, times


def _boundary_xy(frame, grid):
    """NaN-separated line segments tracing every individual-cell boundary in the
    CPM voxel map: edges where the per-voxel cell index changes (cell-vs-cell and
    cell-vs-empty, so each cell gets a closed outline). Coordinates are in um."""
    idr = cell_id_raster(frame, grid)
    ny, nx = idr.shape
    res = float(grid["grid_resolution_um"])
    xs: list = []
    ys: list = []

    def seg(x0, y0, x1, y1):
        xs.extend((x0, x1, None)); ys.extend((y0, y1, None))

    # interior vertical edges (between column c and c+1) -> edge at x=(c+1)*res
    rr, cc = np.nonzero(idr[:, 1:] != idr[:, :-1])
    for r, c in zip(rr.tolist(), cc.tolist()):
        seg((c + 1) * res, r * res, (c + 1) * res, (r + 1) * res)
    # interior horizontal edges (between row r and r+1) -> edge at y=(r+1)*res
    rr, cc = np.nonzero(idr[1:, :] != idr[:-1, :])
    for r, c in zip(rr.tolist(), cc.tolist()):
        seg(c * res, (r + 1) * res, (c + 1) * res, (r + 1) * res)
    # outer domain-border edges for cells touching the wall
    for r in np.nonzero(idr[:, 0])[0].tolist():
        seg(0, r * res, 0, (r + 1) * res)
    for r in np.nonzero(idr[:, -1])[0].tolist():
        seg(nx * res, r * res, nx * res, (r + 1) * res)
    for c in np.nonzero(idr[0, :])[0].tolist():
        seg(c * res, 0, (c + 1) * res, 0)
    for c in np.nonzero(idr[-1, :])[0].tolist():
        seg(c * res, ny * res, (c + 1) * res, ny * res)
    return xs, ys


def _boundary_trace(xs, ys):
    return go.Scatter(x=xs or [None], y=ys or [None], mode="lines", name="cell edges",
                      legendgroup="cell edges", showlegend=False, hoverinfo="skip",
                      line=dict(color="rgba(30,30,30,0.55)", width=0.6))


def _type_fill_paths(frame, grid, codes):
    """Filled rectangles per cell type, built by run-length-encoding each row of
    the type raster. A Scatter ``fill="toself"`` fills each rectangle, so the CPM
    voxel map is drawn with *scatter* traces (not a heatmap) and therefore animates
    smoothly with ``redraw=False`` — no per-frame white-flash strobe. Returns
    ``{type: (xs, ys)}`` with NaN-separated rectangles."""
    z = rasterize(frame, grid)
    res = float(grid["grid_resolution_um"])
    code_to_type = {v: k for k, v in codes.items()}
    out = {t: ([], []) for t in codes}
    ny, nx = z.shape
    for r in range(ny):
        row = z[r]
        c = 0
        while c < nx:
            v = int(row[c])
            if v == 0:
                c += 1
                continue
            c0 = c
            while c < nx and int(row[c]) == v:
                c += 1
            t = code_to_type.get(v)
            if t not in out:
                continue
            xs, ys = out[t]
            x0, x1, y0, y1 = c0 * res, c * res, r * res, (r + 1) * res
            xs.extend((x0, x1, x1, x0, x0, None))
            ys.extend((y0, y0, y1, y1, y0, None))
    return out


def _fill_trace(xs, ys, color, name):
    return go.Scatter(x=xs or [None], y=ys or [None], mode="lines", name=name,
                      legendgroup=name, showlegend=False, fill="toself",
                      fillcolor=color, line=dict(width=0), hoverinfo="skip")


def _voxel_traces(frame, grid, codes, colors):
    """The per-type fill scatters + the cell-outline scatter for one CPM frame,
    in a fixed order (types sorted as in ``codes``, then outlines)."""
    paths = _type_fill_paths(frame, grid, codes)
    traces = [_fill_trace(*paths[t], colors[t], t) for t in codes]
    bx, by = _boundary_xy(frame, grid)
    traces.append(_boundary_trace(bx, by))
    return traces


def _has_polygons(run) -> bool:
    return any((f.get("polygons") or {}) for f in run)


def _hex_to_rgba(hexc: str, a: float) -> str:
    h = hexc.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{a})"


def _poly_paths_by_type(frame, codes):
    """Group a frame's alive cell polygons by state into NaN-separated paths.

    Returns ``{type: (xs, ys)}`` where each ``(xs, ys)`` concatenates every
    polygon of that type, each path closed and separated by ``None`` so a single
    ``fill="toself"`` Scatter fills them independently and the edge line does not
    connect separate cells.
    """
    polys = frame.get("polygons") or {}
    types = frame.get("types", {})
    alive = frame.get("alive", {})
    out = {t: ([], []) for t in codes}
    for cid, verts in polys.items():
        if not alive.get(cid, True) or not verts:
            continue
        t = types.get(cid)
        if t not in out:
            continue
        xs, ys = out[t]
        for x, y in verts:
            xs.append(float(x)); ys.append(float(y))
        xs.append(float(verts[0][0])); ys.append(float(verts[0][1]))  # close
        xs.append(None); ys.append(None)                               # separate
    return out


def _poly_trace(xs, ys, color, name):
    return go.Scatter(
        x=xs or [None], y=ys or [None], mode="lines", name=name, legendgroup=name,
        showlegend=False, fill="toself", fillcolor=_hex_to_rgba(color, 0.85),
        line=dict(color="rgba(40,40,40,0.55)", width=0.7), hoverinfo="skip")


def _spatial_playback(run_cpm, run_vertex, grid_spec, *, dt_days=1.0, max_frames=MAX_FRAMES,
                     title="Spatial playback: CPM vs vertex"):
    codes = type_codes(run_cpm, run_vertex)
    grid = dict(grid_spec, type_codes=codes)
    shape = grid_shape(grid)
    n = min(len(run_cpm), len(run_vertex))
    idx = decimate_indices(n, max_frames)
    ts = times(n, dt_days, idx)
    colors = {t: state_color(t, i) for i, t in enumerate(codes)}
    res = float(grid["grid_resolution_um"])
    vtx_poly = _has_polygons(run_vertex)   # real vertex runs -> draw polygons

    fig = make_subplots(rows=1, cols=2, subplot_titles=("CPM (voxel map)", "Vertex (polygon map)"),
                        horizontal_spacing=0.08)
    anim = []   # trace indices updated by frames, in frame-data order

    def _vertex_traces(frame):
        if vtx_poly:
            paths = _poly_paths_by_type(frame, codes)
            return [_poly_trace(*paths[t], colors[t], t) for t in codes]
        return _voxel_traces(frame, grid, codes, colors)

    # Everything is scatter-based (voxel fills + outlines, or polygons) so the
    # animation runs with redraw=False — no per-frame heatmap redraw, no strobe.
    # --- CPM panel (scatter voxel fills + per-cell outlines) ---
    for tr in _voxel_traces(run_cpm[idx[0]], grid, codes, colors):
        fig.add_trace(tr, row=1, col=1); anim.append(len(fig.data) - 1)
    # --- Vertex panel (polygons, or scatter-voxel fallback for synthetic runs) ---
    for tr in _vertex_traces(run_vertex[idx[0]]):
        fig.add_trace(tr, row=1, col=2); anim.append(len(fig.data) - 1)

    for c in (1, 2):
        fig.update_xaxes(title_text="x (um)", row=1, col=c, range=[0, shape[1] * res])
        fig.update_yaxes(title_text="y (um)", row=1, col=c, scaleanchor=f"x{'' if c == 1 else c}",
                         constrain="domain", range=[0, shape[0] * res])

    # legend-only entries (static, not animated): one square per state
    for t in codes:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=t,
                                 marker=dict(size=10, color=colors[t], symbol="square")),
                      row=1, col=1)

    frames = []
    for k, i in enumerate(idx):
        data = list(_voxel_traces(run_cpm[i], grid, codes, colors))
        data += _vertex_traces(run_vertex[i])
        frames.append(go.Frame(name=str(k), data=data, traces=anim))
    fig.frames = frames
    steps = [dict(method="animate", label=f"{t:g}",
                  args=[[str(k)], dict(mode="immediate", frame=dict(duration=0, redraw=False),
                                       transition=dict(duration=0))])
             for k, t in enumerate(ts)]
    fig.update_layout(
        title=title,
        sliders=[dict(active=0, steps=steps, currentvalue=dict(prefix="time (days): "),
                      pad=dict(t=40))],
        updatemenus=[dict(type="buttons", direction="left", x=0.0, y=-0.05, showactive=False,
                          buttons=[dict(label="Play", method="animate",
                                        args=[None, dict(frame=dict(duration=250, redraw=False),
                                                         fromcurrent=True, transition=dict(duration=0))]),
                                   dict(label="Pause", method="animate",
                                        args=[[None], dict(mode="immediate",
                                                           frame=dict(duration=0, redraw=False),
                                                           transition=dict(duration=0))])])],
        legend=dict(title="Cell state"), height=520, template="plotly_white")
    return fig


def spatial_playback(run_cpm, run_vertex, grid_spec, *, max_frames=MAX_FRAMES, budget=2_200_000, **kw):
    """Two synchronized panels (CPM voxel map | vertex polygon map), shared slider.
    Frames are decimated to <= max_frames and further halved until the serialized
    figure fits ``budget`` bytes (headroom under the workbench's 2.5 MB per-figure
    inline cap)."""
    mf = max_frames
    while True:
        fig = _spatial_playback(run_cpm, run_vertex, grid_spec, max_frames=mf, **kw)
        size = len(fig.to_json())
        if size < budget:
            return fig
        if mf <= 2:
            import warnings
            warnings.warn(f"spatial_playback still {size} B at 2 frames (budget {budget}); "
                          "figure may exceed the workbench inline cap", RuntimeWarning)
            return fig
        mf = max(2, mf // 2)


def _single_playback(run, grid_spec, *, backend="cpm", dt_days=1.0, max_frames=MAX_FRAMES,
                     title=None):
    """One animated panel of a SINGLE backend run (for the Blueprint Studio per-run
    video). Voxel heatmap when the frames carry occupancy (CPM); filled cell
    polygons when they carry polygons (vertex); a field underlay when present.

    ``run`` is a list of frames (``compare.runner`` / ``frame_from_backend`` shape).
    """
    codes = type_codes(run)
    grid = dict(grid_spec, type_codes=codes)
    shape = grid_shape(grid)
    n = len(run)
    idx = decimate_indices(n, max_frames)
    ts = times(n, dt_days, idx)
    colors = {t: state_color(t, i) for i, t in enumerate(codes)}
    res = float(grid["grid_resolution_um"])
    use_poly = _has_polygons(run)
    title = title or f"Spatial playback — {backend}"

    def _frame_traces(frame):
        if use_poly:
            paths = _poly_paths_by_type(frame, codes)
            return [_poly_trace(*paths[t], colors[t], t) for t in codes]
        return _voxel_traces(frame, grid, codes, colors)

    # Scatter-based (voxel fills + outlines, or polygons) so the animation runs
    # with redraw=False — no per-frame heatmap redraw, no white-flash strobe.
    fig = go.Figure()
    anim = []
    for tr in _frame_traces(run[idx[0]]):
        fig.add_trace(tr)
        anim.append(len(fig.data) - 1)
    fig.update_xaxes(title_text="x (um)", range=[0, shape[1] * res])
    fig.update_yaxes(title_text="y (um)", scaleanchor="x", constrain="domain",
                     range=[0, shape[0] * res])
    for t in codes:   # legend
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=t,
                                 marker=dict(size=10, color=colors[t], symbol="square")))

    frames = [go.Frame(name=str(k), data=_frame_traces(run[i]), traces=anim)
              for k, i in enumerate(idx)]
    fig.frames = frames
    steps = [dict(method="animate", label=f"{t:g}",
                  args=[[str(k)], dict(mode="immediate", frame=dict(duration=0, redraw=False),
                                       transition=dict(duration=0))]) for k, t in enumerate(ts)]
    fig.update_layout(
        title=title,
        sliders=[dict(active=0, steps=steps, currentvalue=dict(prefix="time (days): "), pad=dict(t=40))],
        updatemenus=[dict(type="buttons", direction="left", x=0.0, y=-0.05, showactive=False,
                          buttons=[dict(label="Play", method="animate",
                                        args=[None, dict(frame=dict(duration=250, redraw=False),
                                                         fromcurrent=True, transition=dict(duration=0))]),
                                   dict(label="Pause", method="animate",
                                        args=[[None], dict(mode="immediate", frame=dict(duration=0, redraw=False),
                                                           transition=dict(duration=0))])])],
        legend=dict(title="Cell state"), height=560, template="plotly_white")
    return fig


def single_playback(run, grid_spec, *, backend="cpm", max_frames=MAX_FRAMES,
                    budget=2_200_000, **kw):
    """Single-backend animated playback, decimated until it fits the inline cap."""
    mf = max_frames
    while True:
        fig = _single_playback(run, grid_spec, backend=backend, max_frames=mf, **kw)
        if len(fig.to_json()) < budget or mf <= 2:
            return fig
        mf = max(2, mf // 2)
