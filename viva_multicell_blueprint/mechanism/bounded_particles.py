"""BoundedParticlesProcess — Smoldyn particles confined inside tyssue cells.

Composes a particle simulator with a multicellular one: the vertex (tyssue) backend
provides each cell as a moving/deforming polygon, and a Smoldyn sim runs *inside* each
cell (reflective boundary + reversible A+B<->C), via :class:`..particles.SmoldynRunner`
(in-process on Linux/3.12, else a 3.11 subprocess bridge). When a cell divides, its
molecules are partitioned into the two daughter polygons by containment.

A self-contained :class:`~process_bigraph.Process` (like ``TriCoupledProcess``): it
builds the vertex backend in ``__init__``, advances it each step with an optional
built-in division schedule, steps the confined particles, emits molecule counts, and
writes a cell-coloured overlay (flipbook + embedded shareable GIF) to
``reports/assets/<overlay_out>.html``.
"""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import numpy as np
from matplotlib.path import Path as MplPath

from process_bigraph import Process

from ..blueprint.model import Blueprint
from ..compile import compile as _compile
from ..core import build_core
from ..particles import SmoldynRunner

_CELL_PAL = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
             "#e377c2", "#17becf", "#bcbd22", "#7f7f7f", "#393b79", "#637939",
             "#8c6d31", "#843c39", "#7b4173", "#5254a3"]


def _resolve_blueprint(src):
    if isinstance(src, Blueprint):
        return src
    if isinstance(src, dict):
        return Blueprint.from_dict(src)
    import yaml
    text = str(src)
    p = Path(text)
    if len(text) < 512 and p.exists():
        return Blueprint.load(p)
    doc = yaml.safe_load(text)
    if not isinstance(doc, dict):
        raise ValueError("blueprint must be a path, inline YAML mapping, or dict")
    return Blueprint.from_dict(doc)


def _rand_in_poly(poly, n, rng):
    arr = np.asarray(poly, float)
    lo, hi = arr.min(0), arr.max(0)
    path = MplPath(arr)
    out = []
    guard = 0
    while len(out) < n and guard < n * 200:
        guard += 1
        p = [rng.uniform(lo[0], hi[0]), rng.uniform(lo[1], hi[1])]
        if path.contains_point(p):
            out.append(["A" if len(out) % 2 == 0 else "B", p[0], p[1]])
    return out


class BoundedParticlesProcess(Process):
    config_schema = {
        "blueprint": "string",
        "backend": {"_type": "string", "_default": "vertex"},   # cpm | vertex | munk
        "n_steps_hint": {"_type": "integer", "_default": 20},
        "seed_per_cell": {"_type": "integer", "_default": 60},    # total A+B seeded per cell
        "divide_every": {"_type": "integer", "_default": 8},       # 0 = never; else divide one cell every N steps
        "overlay_out": {"_type": "string", "_default": "bounded_particles"},
    }

    def __init__(self, config=None, core=None):
        super().__init__(config, core)
        import random
        self._rng = random.Random(0)
        bp = _resolve_blueprint(self.config.get("blueprint"))
        ext = dict(bp.domain)["extent_um"]
        self.W, self.H = float(ext[0]), float(ext[1])
        from process_bigraph import Composite
        c = core or build_core()
        self._backend = str(self.config.get("backend", "vertex"))
        doc = _compile(bp, backend=self._backend)
        self._mech = Composite({"state": doc["state"]}, core=c).state["mechanics"]["instance"]
        if self._backend == "munk":                 # oblong footprints read better under crowding
            try:
                self._mech._squish = 0.6
            except Exception:
                pass
        self._runner = SmoldynRunner(spec=None)
        self._particles = {}       # cid -> [[species, x, y], ...]
        self._cell_color = {}
        self._frames = []
        self._frame_imgs = []
        self._gif_frames = []
        self._step = 0
        self._divide_every = int(self.config.get("divide_every", 8))
        self._seed_n = int(self.config.get("seed_per_cell", 60))
        self._out = Path("workspace/reports/assets") / (str(self.config.get("overlay_out", "bounded_particles")) + ".html")
        # seed particles into the starting cells
        g0 = self._emit()
        for cid, poly in self._footprints(g0).items():
            self._particles[cid] = _rand_in_poly(poly, self._seed_n, self._rng)

    def _emit(self):
        try:
            return self._mech._emit() or {}
        except Exception:
            return {}

    @staticmethod
    def _footprints(g):
        from ..particles.smoldyn_cell import occupancy_outline
        out = {}
        for bid, fp in (g.get("footprints", {}) or {}).items():
            if not isinstance(fp, dict):
                continue
            xy = fp.get("xy")
            if not xy:
                continue
            # CPM footprints are a list of occupied voxels, not an ordered boundary —
            # trace the true cell outline; vertex/munk already give ordered polygons.
            if fp.get("kind") == "occupancy":
                poly = occupancy_outline(xy)
            else:
                poly = [[float(x), float(y)] for x, y in xy] if len(xy) >= 3 else None
            if poly and len(poly) >= 3:
                out[str(bid)] = poly
        return out

    def inputs(self):
        return {}

    def outputs(self):
        return {"n_cells": "overwrite[integer]", "n_particles": "overwrite[integer]",
                "smoldyn_mode": "overwrite[string]"}

    def _color(self, cid):
        if cid not in self._cell_color:
            self._cell_color[cid] = _CELL_PAL[len(self._cell_color) % len(_CELL_PAL)]
        return self._cell_color[cid]

    def update(self, inputs, interval=1):
        inp = {"fates": {}, "field_grid": {}, "divisions": []}
        g0 = self._emit()
        live = sorted(self._footprints(g0).keys())
        # built-in division schedule: divide the first live cell every N steps
        if self._divide_every and self._step > 0 and self._step % self._divide_every == 0 and live:
            inp["divisions"] = [live[0]]
        try:
            self._mech.update(inp, 1.0)
        except Exception:
            pass
        g = self._emit()
        foot = self._footprints(g)
        lineage = {str(k): str(v) for k, v in (g.get("lineage", {}) or {}).items()}

        # division: partition a mother's particles into the two daughter polygons
        for daughter, mother in lineage.items():
            if daughter in self._particles or daughter not in foot:
                continue
            dpath = MplPath(np.asarray(foot[daughter]))
            keep, give = [], []
            for p in self._particles.get(mother, []):
                (give if dpath.contains_point([p[1], p[2]]) else keep).append(p)
            self._particles[mother] = keep
            self._particles[daughter] = give

        # seed particles for any new cell that still has none
        for cid, poly in foot.items():
            if cid not in self._particles:
                self._particles[cid] = _rand_in_poly(poly, self._seed_n, self._rng)

        # advance the confined particles one step (only cells that are alive now)
        cells = {cid: {"poly": foot[cid], "parts": self._particles.get(cid, [])}
                 for cid in foot}
        if self._runner.available() and cells:
            try:
                advanced = self._runner.step(cells, self._step)
                for cid, parts in advanced.items():
                    self._particles[cid] = parts
            except Exception as e:
                print(f"  [smoldyn] step {self._step}: {e}")

        snapshot = {cid: {"poly": foot[cid], "parts": self._particles.get(cid, [])} for cid in foot}
        self._frames.append(snapshot)
        self._step += 1
        try:
            self._write_overlay()
        except Exception as e:
            print(f"  [overlay] {e}")

        npart = sum(len(s["parts"]) for s in snapshot.values())
        return {"n_cells": len(foot), "n_particles": npart, "smoldyn_mode": self._runner.mode}

    # ---- overlay: cells + particles, coloured by cell ------------------------
    def _write_overlay(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon as MplPoly
        from PIL import Image
        cache, gcache = self._frame_imgs, self._gif_frames
        base = len(cache)
        for j, snapshot in enumerate(self._frames[base:]):
            step_idx = base + j
            fig, ax = plt.subplots(figsize=(6.4, 6.4), dpi=150)
            ax.set_xlim(0, self.W); ax.set_ylim(0, self.H); ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_facecolor("#fbfbfd")
            for sp in ax.spines.values():
                sp.set_edgecolor("#e3e3ea")
            npart = 0
            for cid, snap in snapshot.items():
                cc = self._color(cid)
                poly = np.asarray(snap["poly"])
                ax.add_patch(MplPoly(poly, closed=True, facecolor=cc, alpha=0.09,
                                     edgecolor="none", zorder=0))          # subtle cell region
                ax.add_patch(MplPoly(poly, closed=True, fill=False,
                                     edgecolor=cc, lw=2.1, zorder=1))       # crisp boundary
                if snap["parts"]:
                    P = np.array([[p[1], p[2]] for p in snap["parts"]])
                    ax.scatter(P[:, 0], P[:, 1], s=17, c=cc, zorder=2,
                               edgecolors="white", linewidths=0.4)          # molecules
                    npart += len(snap["parts"])
            ax.text(0.015, 0.985,
                    f"Smoldyn in {self._backend}  ·  step {step_idx}  ·  "
                    f"{len(snapshot)} cells  ·  {npart} molecules",
                    transform=ax.transAxes, va="top", ha="left", fontsize=9.5, color="#333",
                    bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#ddd", alpha=0.9))
            raw = io.BytesIO(); fig.savefig(raw, format="png", bbox_inches="tight", dpi=150)
            plt.close(fig)
            cache.append("data:image/png;base64," + base64.b64encode(raw.getvalue()).decode())
            raw.seek(0)
            im = Image.open(raw).convert("RGB")
            w, h = im.size
            if w > 620:
                im = im.resize((620, int(h * 620 / w)), Image.LANCZOS)
            gcache.append(im)
        frames_js = ",".join(json.dumps(u) for u in cache)
        gif_uri = ""
        if gcache:
            gbuf = io.BytesIO()
            gcache[0].save(gbuf, format="GIF", save_all=True, append_images=gcache[1:],
                           duration=220, loop=0, optimize=True)
            gif_uri = "data:image/gif;base64," + base64.b64encode(gbuf.getvalue()).decode()
        legend = "".join(f'<span class="k"><span class="d" style="background:{self._cell_color[c]}"></span>cell {c}</span>'
                         for c in self._cell_color)
        self._out.parent.mkdir(parents=True, exist_ok=True)
        self._out.write_text(f"""<!doctype html><html><head><meta charset="utf-8"><title>Bounded particles</title>
<style>body{{font-family:-apple-system,sans-serif;margin:16px;background:#fff;color:#1b1b1f}}
#img{{width:min(640px,100%);height:auto;border:1px solid #ddd;border-radius:6px}}
.k{{display:inline-block;margin-right:14px;font-size:13px}}.d{{display:inline-block;width:11px;height:11px;border-radius:50%;vertical-align:middle;margin-right:4px}}
button{{padding:5px 14px;border:1px solid #ccc;border-radius:6px;background:#f6f6f8;cursor:pointer}}</style></head>
<body><h2>Smoldyn particles bounded by {self._backend} cells</h2>
<p>A particle simulator composed with a multicellular one: each <b>{self._backend}</b> cell is a
moving boundary, and a <b>Smoldyn</b> sim runs <b>inside each one</b> — particles reflect off
the cell boundary and react (A + B ⇌ C). Particles are <b>coloured by their cell</b> (matching
the outline); on division a cell's molecules split into the two daughters. {legend}</p>
<img id="img"><div style="margin-top:8px"><button id="play">Play</button>
<input id="sl" type="range" min="0" max="{max(len(cache)-1,0)}" value="0" style="width:420px;vertical-align:middle">
<a href="{gif_uri}" download="bounded-particles.gif" style="margin-left:14px;padding:5px 14px;border:1px solid #ccc;border-radius:6px;background:#f6f6f8;text-decoration:none;color:inherit;font-size:14px">⬇ Download GIF</a></div>
<script>var F=[{frames_js}],i=0,t=null,im=document.getElementById('img'),sl=document.getElementById('sl'),b=document.getElementById('play');
function show(k){{i=((k%F.length)+F.length)%F.length;im.src=F[i];sl.value=i;}}
sl.oninput=e=>{{stop();show(+e.target.value);}};function stop(){{if(t){{clearInterval(t);t=null;b.textContent='Play';}}}}
b.onclick=()=>{{if(t)stop();else{{t=setInterval(()=>show(i+1),220);b.textContent='Pause';}}}};show(0);</script>
</body></html>""", encoding="utf-8")
