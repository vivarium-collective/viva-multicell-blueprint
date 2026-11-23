"""Smoldyn particles confined inside a tri-coupled CONSENSUS tissue.

``ParticleConsensusProcess`` runs the full tri-coupled consensus tissue (CPM + vertex + munk)
and, inside each consensus cell, a Smoldyn particle sim (A+B<->C) confined to that cell's
boundary. The confining boundary is itself a composition: its SHAPE is the vertex polygon, its
POSITION is the munk-weighted consensus, and its AREA is the CPM-owned area. So the molecules
are bounded by a boundary that three tissue simulators agree on — spatially-resolved chemistry
inside a multi-paradigm tissue.
"""
from __future__ import annotations

import base64
import io
import json

import numpy as np
from matplotlib.path import Path as MplPath

from ..particles import SmoldynRunner
from .bounded_particles import _rand_in_poly
from .tri_coupled import TriCoupledProcess


class ParticleConsensusProcess(TriCoupledProcess):
    """Tri-coupled consensus tissue + a Smoldyn sim confined inside each consensus cell."""

    def __init__(self, config=None, core=None):
        super().__init__(config, core)
        import random
        self._prng = random.Random(0)
        self._seed_n = int(self.config.get("seed_per_cell", 50))
        self._runner = SmoldynRunner(spec=None)
        self._particles: dict[str, list] = {}
        self._pframes: list = []
        self._pstep = 0
        self._pimgs, self._pgif = [], []
        # seed into the starting vertex (consensus-shape) polygons
        for cid, poly in self._vertex_polys().items():
            self._particles[cid] = _rand_in_poly(poly, self._seed_n, self._prng)

    def _vertex_polys(self) -> dict[str, list]:
        """Per-cell consensus boundary polygons (vertex shape), from the latest geom."""
        geom = self._prev or {}
        if not geom:
            from .tri_coupled import _emit
            geom = {"vertex": _emit(self._insts["vertex"])}
        out = {}
        for cid, fp in ((geom.get("vertex", {}) or {}).get("footprints", {}) or {}).items():
            xy = fp.get("xy")
            if xy and len(xy) >= 3:
                out[str(cid)] = [[float(x), float(y)] for x, y in xy]
        return out

    def outputs(self):
        return {**super().outputs(),
                "n_particles": "overwrite[integer]", "confined_fraction": "overwrite[float]"}

    def update(self, inputs, interval=1):
        out = super().update(inputs, interval)   # runs the consensus; updates self._prev (geom)
        polys = self._vertex_polys()
        # seed any new cell
        for cid, poly in polys.items():
            if cid not in self._particles:
                self._particles[cid] = _rand_in_poly(poly, self._seed_n, self._prng)
        # advance the confined Smoldyn sim in each consensus cell
        cells = {cid: {"poly": polys[cid], "parts": self._particles.get(cid, [])} for cid in polys}
        if self._runner.available() and cells:
            try:
                advanced = self._runner.step(cells, self._pstep)
                for cid, parts in advanced.items():
                    self._particles[cid] = parts
            except Exception as e:  # pragma: no cover
                print(f"  [smoldyn] consensus step {self._pstep}: {e}")
        snap = {cid: {"poly": polys[cid], "parts": self._particles.get(cid, [])} for cid in polys}
        self._pframes.append(snap)
        self._pstep += 1
        # confinement + count
        inside = total = 0
        for cid, s in snap.items():
            if not s["parts"]:
                continue
            path = MplPath(np.asarray(s["poly"]))
            for p in s["parts"]:
                total += 1
                inside += int(path.contains_point([p[1], p[2]]))
        try:
            self._write_particle_overlay()
        except Exception:  # pragma: no cover
            pass
        out["n_particles"] = int(sum(len(s["parts"]) for s in snap.values()))
        out["confined_fraction"] = float(inside / total) if total else 1.0
        return out

    def _write_particle_overlay(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon as MplPoly
        from PIL import Image
        _PAL = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
                "#e377c2", "#17becf", "#bcbd22", "#7f7f7f"]
        cache = self._pimgs
        base = len(cache)
        out = self._out.parent / "particle_consensus.html"
        for j, snap in enumerate(self._pframes[base:]):
            fig, ax = plt.subplots(figsize=(6.6, 6.6), dpi=140)
            ax.set_xlim(0, self.W); ax.set_ylim(0, self.H); ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor("#fbfbfd")
            for sp in ax.spines.values():
                sp.set_edgecolor("#e3e3ea")
            npart = 0
            for k, (cid, s) in enumerate(snap.items()):
                col = _PAL[k % len(_PAL)]
                poly = np.asarray(s["poly"])
                ax.add_patch(MplPoly(poly, closed=True, facecolor=col, alpha=0.08, edgecolor="none"))
                ax.add_patch(MplPoly(poly, closed=True, fill=False, edgecolor=col, lw=1.8))
                if s["parts"]:
                    P = np.array([[p[1], p[2]] for p in s["parts"]])
                    ax.scatter(P[:, 0], P[:, 1], s=13, c=col, edgecolors="white", linewidths=0.3)
                    npart += len(s["parts"])
            ax.text(0.015, 0.985, f"particles in consensus · step {base + j} · {len(snap)} cells · "
                    f"{npart} molecules", transform=ax.transAxes, va="top", ha="left", fontsize=9.5,
                    color="#333", bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#ddd", alpha=0.9))
            raw = io.BytesIO(); fig.savefig(raw, format="png", bbox_inches="tight", dpi=140); plt.close(fig)
            cache.append("data:image/png;base64," + base64.b64encode(raw.getvalue()).decode())
            raw.seek(0); im = Image.open(raw).convert("RGB"); w, h = im.size
            if w > 620:
                im = im.resize((620, int(h * 620 / w)), Image.LANCZOS)
            self._pgif.append(im)
        frames_js = ",".join(json.dumps(u) for u in cache)
        gif_uri = ""
        if self._pgif:
            gbuf = io.BytesIO()
            self._pgif[0].save(gbuf, format="GIF", save_all=True, append_images=self._pgif[1:],
                               duration=220, loop=0, optimize=True)
            gif_uri = "data:image/gif;base64," + base64.b64encode(gbuf.getvalue()).decode()
        out.write_text(f"""<!doctype html><html><head><meta charset="utf-8"><title>Particles in consensus</title>
<style>body{{font-family:-apple-system,sans-serif;margin:16px;max-width:900px;color:#1b1b1f}}
#img{{width:min(660px,100%);border:1px solid #ddd;border-radius:6px}}
.banner{{background:#f4f6ff;border:1px solid #d6ddf5;border-radius:8px;padding:8px 12px;font-size:14px;margin-bottom:10px}}
button{{padding:5px 14px;border:1px solid #ccc;border-radius:6px;background:#f6f6f8;cursor:pointer}}</style></head>
<body><h2>Smoldyn particles in a consensus tissue</h2>
<div class="banner"><b>Composing  CPM &oplus; vertex &oplus; munk  (consensus boundary)  &otimes;  Smoldyn particles</b></div>
<p>A Smoldyn sim (A+B⇌C) runs inside each cell, confined to a boundary three tissue simulators agree
on: vertex SHAPE, munk-consensus POSITION, CPM AREA. Molecules reflect off that composed boundary.</p>
<img id="img"><div style="margin-top:8px"><button id="play">Play</button>
<input id="sl" type="range" min="0" max="{max(len(cache)-1,0)}" value="0" style="width:420px;vertical-align:middle">
<a href="{gif_uri}" download="particle-consensus.gif" style="margin-left:14px;padding:5px 14px;border:1px solid #ccc;border-radius:6px;background:#f6f6f8;text-decoration:none;color:inherit;font-size:14px">⬇ GIF</a></div>
<script>var F=[{frames_js}],i=0,t=null,im=document.getElementById('img'),sl=document.getElementById('sl'),b=document.getElementById('play');
function show(k){{i=((k%F.length)+F.length)%F.length;im.src=F[i];sl.value=i;}}
sl.oninput=e=>{{stop();show(+e.target.value);}};function stop(){{if(t){{clearInterval(t);t=null;b.textContent='Play';}}}}
b.onclick=()=>{{if(t)stop();else{{t=setInterval(()=>show(i+1),220);b.textContent='Pause';}}}};show(0);</script>
</body></html>""", encoding="utf-8")


__all__ = ["ParticleConsensusProcess"]
