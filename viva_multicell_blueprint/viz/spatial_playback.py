"""Synchronized two-panel playback: CPM voxel map | vertex polygon map, one shared
time slider.

The two backends have genuinely different spatial representations, and the figure
now shows each in its native form:

* **CPM** — a lattice model, drawn as a voxel heatmap (``rasterize`` of the
  per-cell occupancy; empty lattice sites are transparent so the field underlay
  shows through).
* **Vertex** — a polygon model, drawn as the actual cell polygons (the hexagonal
  sheet faces and overlay islands from ``frame['polygons']``), filled by cell state with visible cell
  edges. Rasterizing those polygons (the old behaviour) filled the domain into an
  edgeless blob that did not read as "vertex" at all.

Falls back to a voxel heatmap for the vertex panel when a run carries no polygon
footprints (e.g. the synthetic position-only frames used in unit tests).
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..compare.raster import cell_id_raster, grid_shape, rasterize, type_codes
from ._common import (MAX_FRAMES, decimate_indices, discrete_colorscale, rnd, state_color,
                      times)


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
    cs = discrete_colorscale(codes, colors)
    res = float(grid["grid_resolution_um"])
    xs = rnd(np.arange(shape[1]) * res + res / 2, 2).tolist()
    ys = rnd(np.arange(shape[0]) * res + res / 2, 2).tolist()
    has_field = any(r[0].get("field") is not None for r in (run_cpm, run_vertex))
    vtx_poly = _has_polygons(run_vertex)   # real vertex runs -> draw polygons

    fig = make_subplots(rows=1, cols=2, subplot_titles=("CPM (voxel map)", "Vertex (polygon map)"),
                        horizontal_spacing=0.08)
    anim = []   # trace indices updated by frames, in frame-data order

    def _add_field(run, col):
        fz = _field_underlay(run[idx[0]], shape)
        fig.add_trace(go.Heatmap(x=xs, y=ys, z=None if fz is None else fz,
                                 colorscale="Greys", opacity=0.6, showscale=False,
                                 hoverinfo="skip", name="field"), row=1, col=col)
        anim.append(len(fig.data) - 1)

    # --- CPM panel (voxel heatmap) ---
    if has_field:
        _add_field(run_cpm, 1)
    fig.add_trace(go.Heatmap(x=xs, y=ys, z=_state_z(run_cpm[idx[0]], grid),
                             colorscale=cs, zmin=0, zmax=len(codes), showscale=False,
                             xgap=0, ygap=0, name="state",
                             hovertemplate="x=%{x} um<br>y=%{y} um<extra></extra>"), row=1, col=1)
    anim.append(len(fig.data) - 1)
    bx, by = _boundary_xy(run_cpm[idx[0]], grid)   # per-cell outlines over the voxels
    fig.add_trace(_boundary_trace(bx, by), row=1, col=1)
    anim.append(len(fig.data) - 1)

    # --- Vertex panel (polygons, or heatmap fallback) ---
    if has_field:
        _add_field(run_vertex, 2)
    if vtx_poly:
        paths = _poly_paths_by_type(run_vertex[idx[0]], codes)
        for t in codes:
            px, py = paths[t]
            fig.add_trace(_poly_trace(px, py, colors[t], t), row=1, col=2)
            anim.append(len(fig.data) - 1)
    else:
        fig.add_trace(go.Heatmap(x=xs, y=ys, z=_state_z(run_vertex[idx[0]], grid),
                                 colorscale=cs, zmin=0, zmax=len(codes), showscale=False,
                                 xgap=0, ygap=0, name="state",
                                 hovertemplate="x=%{x} um<br>y=%{y} um<extra></extra>"), row=1, col=2)
        anim.append(len(fig.data) - 1)

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
        data = []
        if has_field:
            fz = _field_underlay(run_cpm[i], shape)
            data.append(go.Heatmap(z=None if fz is None else fz))
        data.append(go.Heatmap(z=_state_z(run_cpm[i], grid)))
        bx, by = _boundary_xy(run_cpm[i], grid)
        data.append(go.Scatter(x=bx or [None], y=by or [None]))
        if has_field:
            fz = _field_underlay(run_vertex[i], shape)
            data.append(go.Heatmap(z=None if fz is None else fz))
        if vtx_poly:
            paths = _poly_paths_by_type(run_vertex[i], codes)
            for t in codes:
                px, py = paths[t]
                data.append(go.Scatter(x=px or [None], y=py or [None]))
        else:
            data.append(go.Heatmap(z=_state_z(run_vertex[i], grid)))
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
    cs = discrete_colorscale(codes, colors)
    res = float(grid["grid_resolution_um"])
    xs = rnd(np.arange(shape[1]) * res + res / 2, 2).tolist()
    ys = rnd(np.arange(shape[0]) * res + res / 2, 2).tolist()
    has_field = run and run[0].get("field") is not None
    use_poly = _has_polygons(run)
    title = title or f"Spatial playback — {backend}"

    fig = go.Figure()
    anim = []
    if has_field:
        fz = _field_underlay(run[idx[0]], shape)
        fig.add_trace(go.Heatmap(x=xs, y=ys, z=None if fz is None else fz, colorscale="Greys",
                                 opacity=0.6, showscale=False, hoverinfo="skip", name="field"))
        anim.append(len(fig.data) - 1)
    if use_poly:
        paths = _poly_paths_by_type(run[idx[0]], codes)
        for t in codes:
            px, py = paths[t]
            fig.add_trace(_poly_trace(px, py, colors[t], t))
            anim.append(len(fig.data) - 1)
    else:
        fig.add_trace(go.Heatmap(x=xs, y=ys, z=_state_z(run[idx[0]], grid), colorscale=cs,
                                 zmin=0, zmax=len(codes), showscale=False, xgap=0, ygap=0,
                                 name="state", hovertemplate="x=%{x} um<br>y=%{y} um<extra></extra>"))
        anim.append(len(fig.data) - 1)
        bx, by = _boundary_xy(run[idx[0]], grid)   # per-cell outlines over the voxels
        fig.add_trace(_boundary_trace(bx, by))
        anim.append(len(fig.data) - 1)
    fig.update_xaxes(title_text="x (um)", range=[0, shape[1] * res])
    fig.update_yaxes(title_text="y (um)", scaleanchor="x", constrain="domain",
                     range=[0, shape[0] * res])
    for t in codes:   # legend
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=t,
                                 marker=dict(size=10, color=colors[t], symbol="square")))

    frames = []
    for k, i in enumerate(idx):
        data = []
        if has_field:
            fz = _field_underlay(run[i], shape)
            data.append(go.Heatmap(z=None if fz is None else fz))
        if use_poly:
            paths = _poly_paths_by_type(run[i], codes)
            for t in codes:
                px, py = paths[t]
                data.append(go.Scatter(x=px or [None], y=py or [None]))
        else:
            data.append(go.Heatmap(z=_state_z(run[i], grid)))
            bx, by = _boundary_xy(run[i], grid)
            data.append(go.Scatter(x=bx or [None], y=by or [None]))
        frames.append(go.Frame(name=str(k), data=data, traces=anim))
    fig.frames = frames
    steps = [dict(method="animate", label=f"{t:g}",
                  args=[[str(k)], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                       transition=dict(duration=0))]) for k, t in enumerate(ts)]
    fig.update_layout(
        title=title,
        sliders=[dict(active=0, steps=steps, currentvalue=dict(prefix="time (days): "), pad=dict(t=40))],
        updatemenus=[dict(type="buttons", direction="left", x=0.0, y=-0.05, showactive=False,
                          buttons=[dict(label="Play", method="animate",
                                        args=[None, dict(frame=dict(duration=250, redraw=True),
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
