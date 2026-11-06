"""Synchronized two-panel playback: CPM voxel map | vertex polygon map, one shared
time slider.

The two backends have genuinely different spatial representations, and the figure
shows each in its native form — **entirely with scatter traces, no Heatmap**. That
is what makes the animation legible:

* An animated ``Heatmap`` can only update under a full per-frame ``redraw``, which
  clears the plot's canvas to white between frames — a disorienting strobe. With
  no Heatmap there is no canvas to clear, so a full redraw of these SVG traces is
  flash-free.
* The animation therefore uses ``redraw=True`` (a discrete redraw per frame), NOT
  ``redraw=False``. ``redraw=False`` takes Plotly's transition path, which tries to
  interpolate each scatter between frames; because cells grow and the point counts
  change frame-to-frame, that makes the coloured regions morph and jump around.
  ``redraw=True`` snaps cleanly to each frame.

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

from ..compare.metrics import neighbour_exchanges
from ..compare.raster import cell_id_raster, grid_shape, rasterize, type_codes
from ._common import MAX_FRAMES, decimate_indices, state_color, times


def _boundary_xy(idr, grid):
    """NaN-separated line segments tracing every individual-cell boundary in the
    CPM voxel map: edges where the per-voxel cell index changes (cell-vs-cell and
    cell-vs-empty, so each cell gets a closed outline). ``idr`` is a cell-id raster
    (possibly temporally smoothed). Coordinates are in um."""
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


def _type_fill_paths(z, grid, codes):
    """Filled rectangles per cell type, built by run-length-encoding each row of
    the type raster ``z`` (possibly temporally smoothed). A Scatter ``fill="toself"``
    fills each rectangle, so the CPM voxel map is drawn with *scatter* traces (not a
    heatmap). Returns ``{type: (xs, ys)}`` with NaN-separated rectangles."""
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


def _voxel_traces(type_raster, id_raster, grid, codes, colors):
    """The per-type fill scatters + the cell-outline scatter for one CPM frame,
    in a fixed order (types sorted as in ``codes``, then outlines). Takes the
    already-computed type/id rasters so a caller can pass temporally-smoothed ones."""
    paths = _type_fill_paths(type_raster, grid, codes)
    traces = [_fill_trace(*paths[t], colors[t], t) for t in codes]
    bx, by = _boundary_xy(id_raster, grid)
    traces.append(_boundary_trace(bx, by))
    return traces


def _has_polygons(run) -> bool:
    return any((f.get("polygons") or {}) for f in run)


def _smooth_run(run, window=2):
    """Temporally smooth per-cell GEOMETRY over a centered +/-``window`` so the
    per-step model noise -- the vertex Ornstein-Uhlenbeck junction wiggle, the CPM
    Metropolis boundary fluctuation, and the immune random walk -- is damped while
    the coherent dynamics (growth, infection spread, migration trends) survive.

    Only geometry is averaged; ``types`` / ``alive`` / ``field`` are taken from the
    centre frame unchanged, so a cell's colour and life/death are never blurred.

      * polygons (vertex): per-vertex mean over the window, where the vertex count
        matches the centre frame (topology changes fall back to the raw polygon).
      * positions: windowed centroid mean (used by the disc fallback).

    CPM occupancy is NOT smoothed here — voting voxel sets punches holes in the
    tissue. CPM smoothing is done on the rasters at render time (per-voxel temporal
    mode, :func:`_cpm_rasters`), which keeps the interior solid.

    ``window=0`` (or a run shorter than 3 frames) is a no-op."""
    n = len(run)
    if n < 3 or window < 1:
        return run
    out = []
    for i in range(n):
        win = run[max(0, i - window):min(n, i + window + 1)]
        f = run[i]
        nf = dict(f)
        raw_pos = f.get("positions") or {}
        nf["positions"] = {
            cid: (np.mean([w["positions"][cid] for w in win if cid in (w.get("positions") or {})],
                          axis=0).tolist())
            for cid in raw_pos}
        if f.get("polygons"):
            polys = {}
            for cid, verts in f["polygons"].items():
                v0 = np.asarray(verts, float)
                stack = [np.asarray(w["polygons"][cid], float) for w in win
                         if (w.get("polygons") or {}).get(cid) is not None
                         and np.asarray(w["polygons"][cid], float).shape == v0.shape]
                polys[cid] = np.mean(stack, axis=0).tolist() if len(stack) >= 2 else verts
            nf["polygons"] = polys
        out.append(nf)
    return out


def _mode_raster(stack, kmax):
    """Per-voxel most-common integer value in ``0..kmax`` across a ``(W, ny, nx)``
    stack (ties broken toward the lower value). Vectorised over the small value
    range, so it stays cheap for the type raster (a handful of codes) and the
    cell-id raster (one plane per cell)."""
    counts = np.zeros((kmax + 1,) + stack.shape[1:], dtype=np.int32)
    for k in range(kmax + 1):
        counts[k] = (stack == k).sum(axis=0)
    return counts.argmax(axis=0).astype(stack.dtype)


def _cpm_rasters(run, i, grid, window):
    """The (type_raster, cell_id_raster) for shown frame ``i``, temporally smoothed
    over +/-``window`` by per-voxel mode (``window=0`` -> the single frame). Mode on
    the TYPE raster stabilises the Metropolis boundary without hollowing the solid
    interior; mode on the id raster steadies the per-cell outlines."""
    if window < 1:
        return rasterize(run[i], grid), cell_id_raster(run[i], grid)
    win = run[max(0, i - window):min(len(run), i + window + 1)]
    tstack = np.stack([rasterize(f, grid) for f in win])
    istack = np.stack([cell_id_raster(f, grid) for f in win])
    return (_mode_raster(tstack, len(grid.get("type_codes") or {})),
            _mode_raster(istack, int(istack.max()) if istack.size else 0))


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


# Human labels for each paradigm's panel (the native representation it draws).
_PANEL_LABEL = {
    "cpm": "CPM (voxel map)",
    "vertex": "Vertex (polygon map)",
    "munk": "Center-based (disc map)",
}


def _panel_traces(run, i, grid, codes, colors, smooth_window, has_poly):
    """One frame of one backend panel: filled cell polygons when the run carries
    polygon footprints (vertex faces, or munk discs), else scatter voxel fills +
    per-cell outlines (CPM, or a synthetic position-only fallback)."""
    if has_poly:
        paths = _poly_paths_by_type(run[i], codes)
        return [_poly_trace(*paths[t], colors[t], t) for t in codes]
    zr, idr = _cpm_rasters(run, i, grid, smooth_window)
    return _voxel_traces(zr, idr, grid, codes, colors)


def _spatial_playback_multi(runs, grid_spec, *, dt_days=1.0, max_frames=MAX_FRAMES,
                            title=None, smooth_window=0):
    """N synchronized panels (one per paradigm), one shared time slider. ``runs`` is
    an ordered ``{backend: run}`` mapping. Each panel draws that backend in its
    native form (voxel fills for CPM, polygons for vertex/center-based); the panel
    subtitle shows that backend's running neighbour-exchange count."""
    backends = list(runs)
    codes = type_codes(*runs.values())
    grid = dict(grid_spec, type_codes=codes)
    shape = grid_shape(grid)
    n = min(len(runs[b]) for b in backends)
    idx = decimate_indices(n, max_frames)
    ts = times(n, dt_days, idx)
    colors = {t: state_color(t, i) for i, t in enumerate(codes)}
    res = float(grid["grid_resolution_um"])
    ncols = len(backends)
    has_poly = {b: _has_polygons(runs[b]) for b in backends}
    # per-backend running neighbour-exchange count (CPM boundary flips / vertex T1-T3
    # auto_reconnect / center-based collisions), surfaced in each panel's subtitle.
    swaps = {b: neighbour_exchanges(runs[b]) for b in backends}
    if title is None:
        title = "Spatial playback: " + " vs ".join(b for b in backends)

    def _titles(i):
        return tuple(f"{_PANEL_LABEL.get(b, b)} — neighbour exchanges: {swaps[b][i]}"
                     for b in backends)

    fig = make_subplots(rows=1, cols=ncols, subplot_titles=_titles(idx[0]),
                        horizontal_spacing=0.06 if ncols > 2 else 0.08)
    title_annotations = [a.to_plotly_json() for a in fig.layout.annotations]
    anim = []   # trace indices updated by frames, in frame-data order

    def _traces_for(b, i):
        return _panel_traces(runs[b], i, grid, codes, colors, smooth_window, has_poly[b])

    # Everything is scatter-based (voxel fills + outlines, or polygons) so the
    # animation runs with redraw=True — discrete frames, no heatmap, no strobe.
    for c, b in enumerate(backends, start=1):
        for tr in _traces_for(b, idx[0]):
            fig.add_trace(tr, row=1, col=c); anim.append(len(fig.data) - 1)

    for c in range(1, ncols + 1):
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
        for b in backends:
            data += list(_traces_for(b, i))
        notes = [dict(a, text=text) for a, text in zip(title_annotations, _titles(i))]
        frames.append(go.Frame(name=str(k), data=data, traces=anim,
                               layout=go.Layout(annotations=notes)))
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


def spatial_playback_multi(runs, grid_spec, *, max_frames=MAX_FRAMES, budget=2_200_000,
                           smooth_window=4, **kw):
    """N synchronized panels (one per paradigm), shared slider. ``runs`` is an
    ordered ``{backend: run}`` mapping (e.g. ``{"cpm": ..., "vertex": ..., "munk": ...}``).
    Per-cell geometry is temporally smoothed over +/-``smooth_window`` sim steps and
    frames decimated/halved until the serialized figure fits ``budget`` bytes."""
    runs = {b: _smooth_run(r, smooth_window) for b, r in runs.items()}
    mf = max_frames
    while True:
        fig = _spatial_playback_multi(runs, grid_spec, max_frames=mf,
                                      smooth_window=smooth_window, **kw)
        size = len(fig.to_json())
        if size < budget:
            return fig
        if mf <= 2:
            import warnings
            warnings.warn(f"spatial_playback still {size} B at 2 frames (budget {budget}); "
                          "figure may exceed the workbench inline cap", RuntimeWarning)
            return fig
        mf = max(2, mf // 2)


def spatial_playback(run_cpm, run_vertex, grid_spec, **kw):
    """Two synchronized panels (CPM voxel map | vertex polygon map), shared slider --
    the legacy 2-backend entry point. For three paradigms use
    :func:`spatial_playback_multi` with a ``{backend: run}`` mapping."""
    return spatial_playback_multi({"cpm": run_cpm, "vertex": run_vertex}, grid_spec, **kw)


def _single_playback(run, grid_spec, *, backend="cpm", dt_days=1.0, max_frames=MAX_FRAMES,
                     title=None, smooth_window=0):
    """One animated panel of a SINGLE backend run (for the Blueprint Studio per-run
    video). Filled scatter voxels + per-cell outlines when the frames carry
    occupancy (CPM); filled cell polygons when they carry polygons (vertex). The CPM
    rasters are temporally smoothed over +/-``smooth_window`` (vertex polygons are
    pre-smoothed by the public wrapper).

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

    def _frame_traces(i):
        if use_poly:
            paths = _poly_paths_by_type(run[i], codes)
            return [_poly_trace(*paths[t], colors[t], t) for t in codes]
        zr, idr = _cpm_rasters(run, i, grid, smooth_window)
        return _voxel_traces(zr, idr, grid, codes, colors)

    # Scatter-based (voxel fills + outlines, or polygons) so the animation runs
    # with redraw=True giving clean discrete frames and no white-flash strobe.
    fig = go.Figure()
    anim = []
    for tr in _frame_traces(idx[0]):
        fig.add_trace(tr)
        anim.append(len(fig.data) - 1)
    fig.update_xaxes(title_text="x (um)", range=[0, shape[1] * res])
    fig.update_yaxes(title_text="y (um)", scaleanchor="x", constrain="domain",
                     range=[0, shape[0] * res])
    for t in codes:   # legend
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=t,
                                 marker=dict(size=10, color=colors[t], symbol="square")))

    frames = [go.Frame(name=str(k), data=_frame_traces(i), traces=anim)
              for k, i in enumerate(idx)]
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
                    budget=2_200_000, smooth_window=4, **kw):
    """Single-backend animated playback, decimated until it fits the inline cap.
    Per-cell geometry is temporally smoothed over +/-``smooth_window`` sim steps
    (:func:`_smooth_run`, ``0`` disables) to damp per-step model noise."""
    run = _smooth_run(run, smooth_window)
    mf = max_frames
    while True:
        fig = _single_playback(run, grid_spec, backend=backend, max_frames=mf,
                               smooth_window=smooth_window, **kw)
        if len(fig.to_json()) < budget or mf <= 2:
            return fig
        mf = max(2, mf // 2)
