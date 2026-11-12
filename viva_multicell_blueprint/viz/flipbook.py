"""Server-rendered image flipbook for spatial playback.

This replaces the Plotly animation path for the Blueprint Studio spatial videos.
Each timeframe is rendered **server-side to a crisp PNG** (matplotlib, Agg):

* **CPM** — ``imshow`` of the type raster (true voxels, nearest-neighbour, no
  interpolation) with a cell-id edge overlay (``LineCollection``), so touching
  same-type cells still read as distinct cells.
* **Vertex / center-based (munk)** — the actual cell polygons, filled by state
  with faint per-cell edges (``PolyCollection``).

The frames are embedded as base64 PNG ``data:`` URIs in a tiny standalone HTML
flipbook (a pure-JS slider + Play button, **no Plotly, no CDN**). What the browser
shows is EXACTLY what the server drew — one static image per frame — so the whole
class of client-side animation bugs that plagued the Plotly path is gone:

* no ``Plotly.animate`` redraw strobe / vanishing Heatmap colour on replay,
* no ``fill="toself"`` even-odd holes punched through the tissue on re-render,
* no scatter comet-tail streaks when a cell's point count changes between frames.

Public entry points (drop-in for the former Plotly figures — each returns a
:class:`Flipbook` whose ``.to_html(include_plotlyjs=...)`` signature matches a
``go.Figure``):

* :func:`single_playback`      — one backend panel (the per-run Studio video).
* :func:`spatial_playback`     — two synced panels (CPM | vertex).
* :func:`spatial_playback_multi` — N synced panels, one per paradigm.
"""
from __future__ import annotations

import base64
import io
import json

import matplotlib
matplotlib.use("Agg")  # headless: these render inside the server worker / tests
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection, PolyCollection  # noqa: E402
from matplotlib.colors import to_rgba  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from PIL import Image  # noqa: E402

from ..compare.metrics import neighbour_exchanges  # noqa: E402
from ..compare.raster import cell_id_raster, grid_shape, rasterize, type_codes  # noqa: E402
from ._common import HTML_CAP, MAX_FRAMES, decimate_indices, state_color, times  # noqa: E402

_DPI = 100
_PALETTE_COLORS = 128   # adaptive-palette quantization: flat cell regions compress
                        # hugely as an indexed PNG with no visible quality loss, so
                        # many more frames fit under the inline cap.

# Human labels for each paradigm's panel (the native representation it draws).
_PANEL_LABEL = {
    "cpm": "CPM (voxel map)",
    "vertex": "Vertex (polygon map)",
    "munk": "Center-based (disc map)",
}


# --------------------------------------------------------------------------- #
# temporal smoothing (shared with the retired Plotly path's semantics)
# --------------------------------------------------------------------------- #
def _has_polygons(run) -> bool:
    return any((f.get("polygons") or {}) for f in run)


def _smooth_run(run, window=2):
    """Temporally smooth per-cell GEOMETRY over a centered +/-``window`` so per-step
    model noise (vertex OU junction wiggle, CPM Metropolis boundary fluctuation,
    immune random walk) is damped while the coherent dynamics survive.

    Only geometry is averaged; ``types`` / ``alive`` / ``field`` come from the centre
    frame unchanged, so colour and life/death are never blurred. CPM occupancy is
    NOT smoothed here (voting voxel sets punches holes); CPM smoothing happens on the
    rasters at render time (:func:`_cpm_rasters`). ``window=0`` is a no-op."""
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
    stack (ties toward the lower value)."""
    counts = np.zeros((kmax + 1,) + stack.shape[1:], dtype=np.int32)
    for k in range(kmax + 1):
        counts[k] = (stack == k).sum(axis=0)
    return counts.argmax(axis=0).astype(stack.dtype)


def _cpm_rasters(run, i, grid, window):
    """The (type_raster, cell_id_raster) for frame ``i``, temporally smoothed over
    +/-``window`` by per-voxel mode (``window=0`` -> the single frame). Mode on the
    type raster stabilises the Metropolis boundary without hollowing the interior;
    mode on the id raster steadies the per-cell outlines."""
    if window < 1:
        return rasterize(run[i], grid), cell_id_raster(run[i], grid)
    win = run[max(0, i - window):min(len(run), i + window + 1)]
    tstack = np.stack([rasterize(f, grid) for f in win])
    istack = np.stack([cell_id_raster(f, grid) for f in win])
    return (_mode_raster(tstack, len(grid.get("type_codes") or {})),
            _mode_raster(istack, int(istack.max()) if istack.size else 0))


# --------------------------------------------------------------------------- #
# per-panel drawing
# --------------------------------------------------------------------------- #
def _boundary_segments(idr, res):
    """Line segments tracing every individual-cell boundary in a cell-id raster:
    edges where the per-voxel cell index changes (cell-vs-cell and cell-vs-empty).
    Returned as a list of ``[(x0,y0),(x1,y1)]`` for a ``LineCollection`` (um)."""
    segs = []
    rr, cc = np.nonzero(idr[:, 1:] != idr[:, :-1])
    for r, c in zip(rr.tolist(), cc.tolist()):
        segs.append([((c + 1) * res, r * res), ((c + 1) * res, (r + 1) * res)])
    rr, cc = np.nonzero(idr[1:, :] != idr[:-1, :])
    for r, c in zip(rr.tolist(), cc.tolist()):
        segs.append([(c * res, (r + 1) * res), ((c + 1) * res, (r + 1) * res)])
    return segs


def _to_um(verts, meta):
    """Map native polygon vertices to um via the frame's optional scale/origin
    (identity when absent — matches the raster's ``_to_grid`` convention)."""
    scale = float((meta or {}).get("scale", 1.0))
    origin = np.asarray((meta or {}).get("origin", (0.0, 0.0)), float)[:2]
    return origin + scale * np.asarray(verts, float)[:, :2]


def _draw_voxels(ax, zr, idr, codes, colors, extent, res):
    """CPM panel: a crisp RGBA voxel raster + per-cell edge outlines."""
    ny, nx = zr.shape
    rgba = np.zeros((ny, nx, 4))
    for t, k in codes.items():
        rgba[zr == k] = to_rgba(colors[t])
    ax.imshow(rgba, origin="lower", extent=extent, interpolation="nearest", zorder=1)
    segs = _boundary_segments(idr, res)
    if segs:
        ax.add_collection(LineCollection(segs, colors="#222222", linewidths=0.35,
                                         alpha=0.5, zorder=2))


def _draw_polys(ax, frame, codes, colors):
    """Vertex / center-based panel: the actual cell polygons, filled by state with
    faint per-cell edges."""
    polys = frame.get("polygons") or {}
    types = frame.get("types", {})
    alive = frame.get("alive", {})
    meta = frame.get("frame") or {}
    by_type = {t: [] for t in codes}
    for cid, verts in polys.items():
        if not alive.get(cid, True) or not verts:
            continue
        t = types.get(cid)
        if t in by_type:
            by_type[t].append(_to_um(verts, meta))
    for t in codes:
        if by_type[t]:
            ax.add_collection(PolyCollection(
                by_type[t], facecolors=to_rgba(colors[t], 0.9),
                edgecolors="#28282899", linewidths=0.5, zorder=1))


def _draw_panel(ax, run, i, grid, codes, colors, smooth_window, has_poly, extent, res):
    ax.clear()
    if has_poly:
        _draw_polys(ax, run[i], codes, colors)
    else:
        zr, idr = _cpm_rasters(run, i, grid, smooth_window)
        _draw_voxels(ax, zr, idr, codes, colors, extent, res)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.set_xlabel("x (µm)", fontsize=9)
    ax.set_ylabel("y (µm)", fontsize=9)
    ax.tick_params(labelsize=8)


# --------------------------------------------------------------------------- #
# the Flipbook object + HTML
# --------------------------------------------------------------------------- #
class Flipbook:
    """A sequence of server-rendered PNG frames with a standalone HTML player.

    Drop-in for a ``go.Figure`` where the spatial-playback code used one: it exposes
    ``to_html(include_plotlyjs=...)`` (the kwarg is accepted and ignored — there is
    no Plotly) and ``frames`` (the list of ``data:`` URIs, so ``len(fb.frames)`` is
    the frame count)."""

    def __init__(self, frames, labels, title, *, slider_prefix="time (days): "):
        self.frames = list(frames)      # list[str] data-URIs
        self.labels = list(labels)      # per-frame slider label (e.g. "0", "1.5")
        self.title = title
        self.slider_prefix = slider_prefix

    def __len__(self):
        return len(self.frames)

    def to_html(self, include_plotlyjs=None, **_):  # noqa: D401 — drop-in signature
        """Standalone HTML flipbook (pure JS; ``include_plotlyjs`` is ignored)."""
        return _flipbook_html(self.frames, self.labels, self.title, self.slider_prefix)

    def to_json(self) -> str:
        """JSON of the frame payload — used only to size the figure in tests."""
        return json.dumps({"title": self.title, "labels": self.labels, "frames": self.frames})


def _flipbook_html(frames, labels, title, slider_prefix):
    imgs = ",".join(json.dumps(u) for u in frames)
    labs = ",".join(json.dumps(x) for x in labels)
    n = len(frames)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(title)}</title>
<!-- vmb-flipbook: server-rendered spatial playback (no Plotly, no CDN) -->
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
         margin:12px; color:#1a1a1a; background:#fff; }}
  @media (prefers-color-scheme:dark) {{ body {{ color:#e8e8e8; background:#161616; }} }}
  h3 {{ margin:0 0 8px; font-size:15px; font-weight:600; }}
  #vmb-flipbook {{ max-width:960px; }}
  #vmb-img {{ width:100%; height:auto; display:block; border:1px solid rgba(128,128,128,.3);
             border-radius:4px; background:#fff; }}
  .ctl {{ margin-top:8px; display:flex; gap:10px; align-items:center; }}
  .ctl button {{ padding:4px 14px; border:1px solid rgba(128,128,128,.45);
                 border-radius:4px; background:transparent; color:inherit; cursor:pointer;
                 font-size:13px; }}
  .ctl button:hover {{ background:rgba(128,128,128,.12); }}
  .ctl input[type=range] {{ flex:1; min-width:120px; }}
  .lbl {{ font-variant-numeric:tabular-nums; font-size:12px; opacity:.8; min-width:150px; }}
</style></head><body>
<div id="vmb-flipbook">
  <h3>{_esc(title)}</h3>
  <img id="vmb-img" alt="spatial playback frame">
  <div class="ctl">
    <button id="vmb-play" type="button">► Play</button>
    <input id="vmb-sl" type="range" min="0" max="{n - 1}" value="0" step="1"
           aria-label="frame">
    <span class="lbl" id="vmb-lbl"></span>
  </div>
</div>
<script>
(function() {{
  var F=[{imgs}], L=[{labs}], PFX={json.dumps(slider_prefix)};
  var img=document.getElementById('vmb-img'), sl=document.getElementById('vmb-sl'),
      lbl=document.getElementById('vmb-lbl'), btn=document.getElementById('vmb-play');
  var i=0, timer=null;
  function show(k) {{ i=((k%F.length)+F.length)%F.length; img.src=F[i]; sl.value=i;
    lbl.textContent='frame '+(i+1)+'/'+F.length+'  ·  '+PFX+L[i]; }}
  function stop() {{ if(timer){{clearInterval(timer);timer=null;btn.textContent='► Play';}} }}
  function play() {{ if(F.length<2)return; timer=setInterval(function(){{show(i+1);}},220);
    btn.textContent='❚❚ Pause'; }}
  sl.addEventListener('input', function(e){{ stop(); show(+e.target.value); }});
  btn.addEventListener('click', function(){{ if(timer)stop(); else play(); }});
  show(0);
}})();
</script></body></html>"""


def _esc(s: str) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


# --------------------------------------------------------------------------- #
# figure assembly + budget-aware public entry points
# --------------------------------------------------------------------------- #
def _fig_to_datauri(fig) -> str:
    raw = io.BytesIO()
    # NOTE: no bbox_inches="tight" — a fixed canvas keeps every frame the SAME pixel
    # size, so the <img> never jumps/resizes between frames.
    fig.savefig(raw, format="png", dpi=_DPI)
    raw.seek(0)
    # Quantize to an adaptive palette: the frames are mostly flat cell-colour regions
    # (plus thin edges), so an indexed PNG is several times smaller than RGBA with no
    # visible loss — which is what lets a full-length run keep ~MAX_FRAMES frames
    # under the workbench inline cap.
    img = Image.open(raw).convert("RGB").quantize(
        colors=_PALETTE_COLORS, method=Image.FASTOCTREE, dither=Image.Dither.NONE)
    out = io.BytesIO()
    img.save(out, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(out.getvalue()).decode("ascii")


def _flipbook(runs, grid_spec, *, dt_days=1.0, max_frames=MAX_FRAMES, title=None,
              smooth_window=0, single=False):
    """Render an ordered ``{backend: run}`` mapping to a :class:`Flipbook`. Each
    shown frame is one PNG with ``len(runs)`` side-by-side panels (CPM voxel map,
    vertex/munk polygons), a shared cell-state legend, and a per-frame time label."""
    backends = list(runs)
    codes = type_codes(*runs.values())
    grid = dict(grid_spec, type_codes=codes)
    ny, nx = grid_shape(grid)
    res = float(grid["grid_resolution_um"])
    extent = [0.0, nx * res, 0.0, ny * res]
    n = min(len(runs[b]) for b in backends)
    idx = decimate_indices(n, max_frames)
    ts = times(n, dt_days, idx)
    colors = {t: state_color(t, i) for i, t in enumerate(codes)}
    has_poly = {b: _has_polygons(runs[b]) for b in backends}
    # running neighbour-exchange count (CPM boundary flips / vertex T1-T3 / munk
    # collisions) per panel, shown in multi-panel subtitles.
    swaps = None if single else {b: neighbour_exchanges(runs[b]) for b in backends}
    if title is None:
        title = (f"Spatial playback — {backends[0]}" if single
                 else "Spatial playback: " + " vs ".join(backends))

    ncols = len(backends)
    fig, axes = plt.subplots(1, ncols, figsize=(max(5.8, 4.3 * ncols), 5.3),
                             dpi=_DPI, squeeze=False)
    axes = axes[0]
    fig.subplots_adjust(left=0.08 if ncols == 1 else 0.055, right=0.985,
                        top=0.9, bottom=0.17, wspace=0.2)
    fig.suptitle(title, fontsize=12, y=0.975)
    time_text = fig.text(0.985, 0.955, "", ha="right", va="top", fontsize=9.5,
                         fontstyle="italic")
    # shared, static cell-state legend along the bottom
    handles = [Patch(facecolor=colors[t], edgecolor="none", label=t) for t in codes]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(codes) or 1, 6),
               frameon=False, fontsize=9, title="Cell state", title_fontsize=9,
               bbox_to_anchor=(0.5, 0.0))

    try:
        data_uris = []
        for k, i in enumerate(idx):
            for ax, b in zip(axes, backends):
                _draw_panel(ax, runs[b], i, grid, codes, colors, smooth_window,
                            has_poly[b], extent, res)
                if single:
                    ax.set_title(_PANEL_LABEL.get(b, b), fontsize=10.5, loc="left")
                else:
                    ax.set_title(f"{_PANEL_LABEL.get(b, b)}\nneighbour exchanges: {swaps[b][i]}",
                                 fontsize=10, loc="left")
            time_text.set_text(f"t = {ts[k]:g} days")
            data_uris.append(_fig_to_datauri(fig))
    finally:
        plt.close(fig)
    return Flipbook(data_uris, [f"{t:g}" for t in ts], title)


def _flipbook_budgeted(runs, grid_spec, *, max_frames=MAX_FRAMES, budget=2_200_000,
                       smooth_window=4, single=False, **kw):
    """Pre-smooth geometry, then decimate/halve frames until the rendered HTML fits
    ``budget`` bytes (bottoming out at 2 frames with a RuntimeWarning)."""
    runs = {b: _smooth_run(r, smooth_window) for b, r in runs.items()}
    mf = max_frames
    while True:
        fb = _flipbook(runs, grid_spec, max_frames=mf, smooth_window=smooth_window,
                       single=single, **kw)
        size = len(fb.to_html().encode())
        if size < budget:
            return fb
        if mf <= 2:
            import warnings
            warnings.warn(f"spatial playback flipbook still {size} B at 2 frames "
                          f"(budget {budget}); may exceed the workbench inline cap "
                          f"({HTML_CAP})", RuntimeWarning)
            return fb
        mf = max(2, mf // 2)


def single_playback(run, grid_spec, *, backend="cpm", max_frames=MAX_FRAMES,
                    budget=2_200_000, smooth_window=4, title=None, **kw):
    """Single-backend spatial playback flipbook (the Blueprint Studio per-run video).
    Geometry is temporally smoothed over +/-``smooth_window`` sim steps, then frames
    are decimated to fit the inline cap. Returns a :class:`Flipbook`."""
    return _flipbook_budgeted({backend: run}, grid_spec, max_frames=max_frames,
                              budget=budget, smooth_window=smooth_window, single=True,
                              title=title or f"Spatial playback — {backend}", **kw)


def spatial_playback_multi(runs, grid_spec, *, max_frames=MAX_FRAMES, budget=2_200_000,
                           smooth_window=4, **kw):
    """N synchronized panels (one per paradigm), one shared time slider. ``runs`` is
    an ordered ``{backend: run}`` mapping. Each panel draws that backend in its
    native form (voxels for CPM, polygons for vertex/center-based). Returns a
    :class:`Flipbook`."""
    return _flipbook_budgeted(runs, grid_spec, max_frames=max_frames, budget=budget,
                              smooth_window=smooth_window, single=False, **kw)


def spatial_playback(run_cpm, run_vertex, grid_spec, **kw):
    """Two synchronized panels (CPM voxel map | vertex polygon map), shared slider —
    the legacy 2-backend entry point. For three paradigms use
    :func:`spatial_playback_multi` with a ``{backend: run}`` mapping."""
    return spatial_playback_multi({"cpm": run_cpm, "vertex": run_vertex}, grid_spec, **kw)
