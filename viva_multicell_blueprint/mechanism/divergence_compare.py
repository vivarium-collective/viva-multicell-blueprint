"""``DivergenceCompareProcess``: the same blueprint run four ways, rendered as a
3-column x 2-row animation to show how the mechanics paradigms diverge from one
shared initial state.

Each step advances FOUR sub-simulations seeded from the SAME blueprint:
  * CPM alone, vertex alone, center-based (munk) alone — each stepped UNCOUPLED
    (``TriCoupledProcess`` in ``solo`` mode), and
  * the full tri-coupled consensus.

The frame is a 3x2 grid:
  top row    — CPM · alone | Vertex · alone | Center-based · alone
  bottom row — the SAME three paradigms, but as the coupled sim sees them
               (CPM · coupled | Vertex · coupled | Center-based · coupled).

So each COLUMN is one paradigm, standalone (top) vs in-consensus (bottom): the
top row drifts apart, the bottom row stays mutually registered. Written as a
flipbook + embedded GIF to ``reports/assets/<overlay_out>.html``.
"""
from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
from process_bigraph import Process

from .tri_coupled import TriCoupledProcess, _emit


class DivergenceCompareProcess(Process):
    config_schema = {
        "blueprint": "string",
        "cpm_res": {"_type": "integer", "_default": 3},
        "cpm_temp": {"_type": "float", "_default": 1.0},
        "munk_squish": {"_type": "float", "_default": 0.8},
        "overlay_out": {"_type": "string", "_default": "divergence_compare"},
        # Render (keep) only every Nth step in the gif, so a long run stays a
        # sane-sized animation. All steps are still simulated; 1 = every step.
        "render_every": {"_type": "integer", "_default": 1},
    }

    def __init__(self, config=None, core=None):
        super().__init__(config, core)
        base = {k: self.config.get(k) for k in ("blueprint", "cpm_res", "cpm_temp", "munk_squish")
                if self.config.get(k) is not None}
        # Four sub-sims from the SAME blueprint/init; render=False (we render here).
        self._solo = {
            b: TriCoupledProcess({**base, "solo": b, "render": False}, core=core)
            for b in ("cpm", "vertex", "munk")
        }
        self._tri = TriCoupledProcess({**base, "render": False}, core=core)
        self._render_every = max(1, int(self.config.get("render_every", 1) or 1))
        t = self._tri
        self.W, self.H, self.nx, self.ny = t.W, t.H, t.nx, t.ny
        # bid -> cell-type name (from the shared init's vertex footprints; bids are
        # assigned identically across the four sims, same blueprint).
        v0 = _emit(t._insts["vertex"])
        self._bid2type = {str(b): typ for b, typ in (v0.get("types", {}) or {}).items()}
        # type -> colour, exactly as viz._common.state_color over the sorted types
        # (same palette as every other spatial render in the repo).
        from ..viz._common import state_color
        names = sorted({c["name"] for c in t._bp_dict["cell_types"]})
        self._colors = {n: state_color(n, i) for i, n in enumerate(names)}
        self._out = Path("workspace/reports/assets") / (
            str(self.config.get("overlay_out", "divergence_compare")) + ".html")
        self._frame_uris: list[str] = []
        self._gif_frames: list = []
        self._step = 0

    def inputs(self):
        return {}

    def outputs(self):
        return {"cpm_vs_coupled_drift": "overwrite[float]", "n_cells": "overwrite[integer]"}

    # ---- rendering -----------------------------------------------------------
    def _col(self, type_name):
        return self._colors.get(type_name, "#999999")

    def _draw_cpm(self, ax, geom, cpm_to_bid):
        from matplotlib.colors import to_rgb
        lab = np.asarray(geom.get("_cpm_labels", []), int)
        if lab.size < self.nx * self.ny:
            return
        grid = lab[: self.nx * self.ny].reshape(self.ny, self.nx)
        rgb = np.ones((self.ny, self.nx, 3))
        for cpmid in np.unique(grid):
            if cpmid == 0:
                continue
            bid = cpm_to_bid.get(int(cpmid))
            rgb[grid == cpmid] = to_rgb(self._col(self._bid2type.get(str(bid))))
        ax.imshow(rgb, origin="lower", extent=[0, self.W, 0, self.H],
                  interpolation="nearest", zorder=0)

    def _draw_polys(self, ax, geom, backend):
        from matplotlib.colors import to_rgb
        from matplotlib.patches import Polygon as MplPoly
        types = geom.get(backend, {}).get("types", {}) or {}
        for bid, fp in (geom.get(backend, {}).get("footprints", {}) or {}).items():
            xy = fp.get("xy") if isinstance(fp, dict) else None
            if xy and len(xy) >= 3:
                col = self._col(types.get(str(bid)) or self._bid2type.get(str(bid)))
                ax.add_patch(MplPoly(np.array(xy, float), closed=True,
                                     facecolor=(*to_rgb(col), 0.9),
                                     edgecolor="#28282899", lw=0.5))

    def _render_frame(self, solo_geom, tri_geom):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        from PIL import Image
        fig = plt.figure(figsize=(10.2, 7.1), dpi=128)
        gs = GridSpec(2, 3, hspace=0.20, wspace=0.08)
        axes = [[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(2)]
        for row in axes:
            for ax in row:
                ax.set_xlim(0, self.W); ax.set_ylim(0, self.H)
                ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        titles = [["CPM · alone", "Vertex · alone", "Center-based · alone"],
                  ["CPM · coupled", "Vertex · coupled", "Center-based · coupled"]]
        for r in range(2):
            for c in range(3):
                axes[r][c].set_title(titles[r][c], fontsize=9.5, color="#333")
        # top row: the three standalone sims
        self._draw_cpm(axes[0][0], solo_geom["cpm"], self._solo["cpm"]._cpm_to_bid)
        self._draw_polys(axes[0][1], solo_geom["vertex"], "vertex")
        self._draw_polys(axes[0][2], solo_geom["munk"], "munk")
        # bottom row: the SAME three paradigms from the one coupled sim
        self._draw_cpm(axes[1][0], tri_geom, self._tri._cpm_to_bid)
        self._draw_polys(axes[1][1], tri_geom, "vertex")
        self._draw_polys(axes[1][2], tri_geom, "munk")
        fig.suptitle(f"Same initial state, four ways · step {self._step}  "
                     f"(top: each paradigm alone — bottom: the coupled consensus)",
                     fontsize=10.5, y=0.99)
        raw = io.BytesIO(); fig.savefig(raw, format="png", bbox_inches="tight", dpi=128)
        plt.close(fig)
        self._frame_uris.append("data:image/png;base64," + base64.b64encode(raw.getvalue()).decode())
        raw.seek(0); im = Image.open(raw).convert("RGB"); w, h = im.size
        if w > 980:
            im = im.resize((980, int(h * 980 / w)), Image.LANCZOS)
        self._gif_frames.append(im)

    def _write_overlay(self):
        frames_js = ",".join(__import__("json").dumps(u) for u in self._frame_uris)
        gif_uri = ""
        if self._gif_frames:
            gbuf = io.BytesIO()
            self._gif_frames[0].save(gbuf, format="GIF", save_all=True,
                                     append_images=self._gif_frames[1:], duration=240,
                                     loop=0, optimize=True)
            gif_uri = "data:image/gif;base64," + base64.b64encode(gbuf.getvalue()).decode()
        legend = "".join(
            f'<span class="k"><span class="d" style="background:{self._colors[t]}"></span>{t}</span>'
            for t in sorted(self._colors))
        self._out.parent.mkdir(parents=True, exist_ok=True)
        self._out.write_text(f"""<!doctype html><html><head><meta charset="utf-8"><title>Divergence — same init, four ways</title>
<style>body{{font-family:-apple-system,sans-serif;margin:16px;max-width:1040px;color:#1b1b1f}}
#img{{width:min(1000px,100%);border:1px solid #ddd;border-radius:6px}}
.banner{{background:#f4f6ff;border:1px solid #d6ddf5;border-radius:8px;padding:8px 12px;font-size:14px;margin-bottom:10px}}
.k{{display:inline-block;margin-right:14px;font-size:13px}}.d{{display:inline-block;width:11px;height:11px;border-radius:2px;vertical-align:middle;margin-right:4px}}
button{{padding:5px 14px;border:1px solid #ccc;border-radius:6px;background:#f6f6f8;cursor:pointer}}</style></head>
<body><h2>Same initial state, four ways — how the paradigms diverge</h2>
<div class="banner"><b>Top row</b>: CPM, vertex and center-based each simulated <b>alone</b> (uncoupled). <b>Bottom row</b>: the same three paradigms inside the <b>tri-coupled consensus</b>. Each column is one paradigm — standalone vs coupled.</div>
<p>All six panels start from the identical imaged/initial cells (coloured by cell type). Watch the top row drift apart while the bottom row stays mutually registered.</p>
<p><b>Cell types:</b> {legend}</p>
<img id="img"><div style="margin-top:8px"><button id="play">Play</button>
<input id="sl" type="range" min="0" max="{max(len(self._frame_uris)-1,0)}" value="0" style="width:420px;vertical-align:middle">
<a href="{gif_uri}" download="divergence-compare.gif" style="margin-left:14px;padding:5px 14px;border:1px solid #ccc;border-radius:6px;background:#f6f6f8;text-decoration:none;color:inherit;font-size:14px">⬇ GIF</a></div>
<script>var F=[{frames_js}],i=0,t=null,im=document.getElementById('img'),sl=document.getElementById('sl'),b=document.getElementById('play');
function show(k){{i=((k%F.length)+F.length)%F.length;im.src=F[i];sl.value=i;}}
sl.oninput=e=>{{stop();show(+e.target.value);}};function stop(){{if(t){{clearInterval(t);t=null;b.textContent='Play';}}}}
b.onclick=()=>{{if(t)stop();else{{t=setInterval(()=>show(i+1),240);b.textContent='Pause';}}}};show(0);</script>
</body></html>""", encoding="utf-8")

    # ---- step -----------------------------------------------------------------
    def update(self, inputs, interval=1):
        for b in ("cpm", "vertex", "munk"):
            self._solo[b].update({}, 1)
        self._tri.update({}, 1)
        solo_geom = {b: self._solo[b]._prev or {} for b in ("cpm", "vertex", "munk")}
        tri_geom = self._tri._prev or {}
        if self._step % self._render_every == 0:
            try:
                self._render_frame(solo_geom, tri_geom)
                self._write_overlay()
            except Exception as e:  # pragma: no cover
                print(f"  [divergence] render step {self._step}: {e}")
        # drift readout: mean CPM-alone vs CPM-coupled centroid distance
        drift = self._cpm_drift(solo_geom["cpm"], tri_geom)
        self._step += 1
        return {"cpm_vs_coupled_drift": drift, "n_cells": len(self._tri._cpm_to_bid)}

    def _cpm_drift(self, solo_cpm, tri_geom):
        import math
        a = (solo_cpm.get("cpm", {}) or {}).get("positions", {}) or {}
        b = (tri_geom.get("cpm", {}) or {}).get("positions", {}) or {}
        ks = set(a) & set(b)
        if not ks:
            return 0.0
        return float(np.mean([math.hypot(a[k][0] - b[k][0], a[k][1] - b[k][1]) for k in ks]))
