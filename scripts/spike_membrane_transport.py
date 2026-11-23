"""Spike (throwaway) — membrane transport between cells, via a swappable adapter.

Question this spike answers: once a molecule can CROSS the membrane, do per-cell-isolated
Smoldyn sims still work, or does crossing force a shared extracellular representation —
and what does an "adapter" between modes have to do?

Setup: 3 static cells in a row. Each runs its own confined Smoldyn sim (A+B<->C). One
species, **S** (a signal), is declared MEMBRANE-PERMEABLE: its surface action is
`transmit`, not `reflect`, so it leaves the cell. Cell 0 is seeded with S; cells 1,2 are
not. Each step:
  1. step each cell; molecules that ended up OUTSIDE the polygon have crossed the membrane.
  2. ADAPTER (extracellular): take the crossed S and put it into a shared representation.
     Here: a diffusing FIELD (deposit at crossing point, diffuse). A particle-space adapter
     would drop in at the same seam.
  3. ADAPTER (uptake): each cell samples the extracellular S over its footprint and takes
     a fraction back IN as interior S molecules.
So S made/seeded in cell 0 should cross out, diffuse, and appear inside cells 1 and 2.

Run (viva-smoldyn 3.11 venv):
  ~/code/viva-smoldyn/.venv/bin/python scripts/spike_membrane_transport.py
"""
from __future__ import annotations

import base64
import contextlib
import io
import json
import os
import random
from pathlib import Path

import numpy as np
from matplotlib.path import Path as MplPath

import smoldyn

OUT = Path("workspace/reports/assets/membrane_transport_spike.html")
W = H = 30
NX = NY = 30
N_STEPS = 18
SPECIES = {"A": 1.4, "B": 1.4, "C": 0.9, "S": 2.2}      # difc µm²/s
PERMEABLE = {"S"}                                        # declared to cross the membrane
KF, KB = 8.0, 2.0
SPCOL = {"A": "#b9c0c9", "B": "#9aa3ad", "C": "#7fae7f", "S": "#d1495b"}
_rng = random.Random(0)

CELLS = {
    "0": [[4, 11], [11, 11], [11, 19], [4, 19]],       # sender (seeded with S)
    "1": [[12, 11], [19, 11], [19, 19], [12, 19]],
    "2": [[20, 11], [27, 11], [27, 19], [20, 19]],
}


@contextlib.contextmanager
def _suppress():
    saved = os.dup(1); dn = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(dn, 1); yield
    finally:
        os.dup2(saved, 1); os.close(saved); os.close(dn)


def _rand_in(poly, n, sp):
    arr = np.asarray(poly, float); lo, hi = arr.min(0), arr.max(0); path = MplPath(arr)
    out = []
    while len(out) < n:
        p = [_rng.uniform(lo[0], hi[0]), _rng.uniform(lo[1], hi[1])]
        if path.contains_point(p):
            out.append([sp, p[0], p[1]])
    return out


def step_cell(poly, parts, seed):
    """Advance one cell; return (interior_parts, crossed_parts). PERMEABLE species get a
    `transmit` surface action, so after the step they can be OUTSIDE the polygon."""
    arr = np.asarray(poly, float); path = MplPath(arr)
    lo = [float(arr[:, 0].min() - 2), float(arr[:, 1].min() - 2)]
    hi = [float(arr[:, 0].max() + 2), float(arr[:, 1].max() + 2)]
    with _suppress():
        s = smoldyn.Simulation(low=lo, high=hi, boundary_type="r", seed=int(seed))
        objs = {n: s.addSpecies(n, difc=d, display_size=2) for n, d in SPECIES.items()}
        s.addBidirectionalReaction("r", subs=[objs["A"], objs["B"]], prds=[objs["C"]], kf=KF, kb=KB)
        panels = [smoldyn.Triangle(vertices=[list(arr[i]), list(arr[(i + 1) % len(arr)])])
                  for i in range(len(arr))]
        srf = s.addSurface("mem", panels=panels)
        srf.setAction("both", [objs[n] for n in SPECIES if n not in PERMEABLE], "reflect")
        srf.setAction("both", [objs[n] for n in PERMEABLE], "trans")        # <-- the membrane spec (permeable)
        for sp, x, y in parts:
            objs[sp].addToSolution(1, pos=[float(x), float(y)])
        s.addOutputData("molpos"); s.addCommand(cmd="listmols2 molpos", cmd_type="E")
        s.setGraphics("none"); s.updateSim(); s.run(stop=1.0, dt=0.05)
        data = s.getOutputData("molpos", True) or []
    if not data:
        return parts, []
    name = {i + 1: n for i, n in enumerate(SPECIES)}
    last = data[-1][0]
    interior, crossed = [], []
    for r in data:
        if r[0] != last or int(r[1]) not in name:
            continue
        sp, x, y = name[int(r[1])], float(r[3]), float(r[4])
        (interior if path.contains_point([x, y]) else crossed).append([sp, x, y])
    return interior, crossed


# ---- extracellular adapter: FIELD mode (a particle-space mode drops in here) --------
class FieldExtracellular:
    mode = "field"

    def __init__(self):
        self.grid = np.zeros((NY, NX), float)

    def deposit(self, crossed):          # crossed: list of [sp,x,y] that left a cell
        for sp, x, y in crossed:
            if sp == "S":
                self.grid[int(np.clip(y, 0, NY - 1)), int(np.clip(x, 0, NX - 1))] += 1.0

    def diffuse(self, d=0.18, decay=0.01):
        g = self.grid
        lap = (-4 * g + np.roll(g, 1, 0) + np.roll(g, -1, 0) + np.roll(g, 1, 1) + np.roll(g, -1, 1))
        self.grid = np.clip(g + d * lap - decay * g, 0, None)

    def uptake(self, poly, rate=0.25):   # cell pulls a fraction of the S over its footprint back IN
        ys, xs = np.mgrid[0:NY, 0:NX]
        mask = MplPath(np.asarray(poly)).contains_points(np.c_[xs.ravel() + 0.5, ys.ravel() + 0.5]).reshape(NY, NX)
        avail = float(self.grid[mask].sum())
        self.grid[mask] *= (1 - rate)
        return int(round(avail * rate))


def main():
    particles = {cid: _rand_in(poly, 30, "A") + _rand_in(poly, 30, "B") for cid, poly in CELLS.items()}
    particles["0"] += _rand_in(CELLS["0"], 45, "S")            # only the sender has S
    ext = FieldExtracellular()
    frames = []
    for step in range(N_STEPS):
        crossed_all = []
        for cid, poly in CELLS.items():
            interior, crossed = step_cell(poly, particles[cid], seed=step * 131 + int(cid) * 17)
            particles[cid] = interior
            crossed_all += crossed
        ext.deposit(crossed_all)                               # adapter: membrane -> extracellular
        ext.diffuse()
        for cid, poly in CELLS.items():
            n = ext.uptake(poly)                               # adapter: extracellular -> membrane
            if n:
                particles[cid] += _rand_in(poly, n, "S")       # S re-enters the cell
        frames.append({"field": ext.grid.copy(),
                       "cells": {cid: {"poly": CELLS[cid], "parts": list(particles[cid])} for cid in CELLS}})
        per_cell_S = {cid: sum(1 for p in particles[cid] if p[0] == "S") for cid in CELLS}
        print(f"  step {step:2d}: S per cell {per_cell_S}  field S total {ext.grid.sum():.0f}")
    _render(frames)


def _render(frames):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPoly
    from PIL import Image
    imgs, gifs = [], []
    vmax = max(f["field"].max() for f in frames) or 1.0
    for fr in frames:
        fig, ax = plt.subplots(figsize=(6.2, 6.2), dpi=150)
        ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(fr["field"], origin="lower", extent=[0, W, 0, H], cmap="Reds",
                  vmin=0, vmax=vmax, alpha=0.65, zorder=0, interpolation="bilinear")
        for cid, c in fr["cells"].items():
            ax.add_patch(MplPoly(np.asarray(c["poly"]), closed=True, fill=False, edgecolor="#333", lw=2.0, zorder=1))
            if c["parts"]:
                P = np.array([[p[1], p[2]] for p in c["parts"]])
                col = [SPCOL[p[0]] for p in c["parts"]]
                ax.scatter(P[:, 0], P[:, 1], s=16, c=col, zorder=2, edgecolors="white", linewidths=0.3)
        ax.text(0.015, 0.985, "membrane transport · extracellular adapter = FIELD · S (red) crosses out",
                transform=ax.transAxes, va="top", fontsize=9, color="#333",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ddd", alpha=0.9))
        raw = io.BytesIO(); fig.savefig(raw, format="png", bbox_inches="tight", dpi=150); plt.close(fig)
        imgs.append("data:image/png;base64," + base64.b64encode(raw.getvalue()).decode())
        raw.seek(0); im = Image.open(raw).convert("RGB"); w, h = im.size
        if w > 620:
            im = im.resize((620, int(h * 620 / w)), Image.LANCZOS)
        gifs.append(im)
    g = io.BytesIO(); gifs[0].save(g, format="GIF", save_all=True, append_images=gifs[1:], duration=240, loop=0, optimize=True)
    gif = "data:image/gif;base64," + base64.b64encode(g.getvalue()).decode()
    fjs = ",".join(json.dumps(u) for u in imgs)
    leg = "".join(f'<span style="margin-right:12px"><span style="display:inline-block;width:11px;height:11px;border-radius:50%;background:{SPCOL[s]};vertical-align:middle;margin-right:4px"></span>{s}</span>' for s in SPECIES)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(f"""<!doctype html><html><head><meta charset="utf-8"><title>Membrane transport</title>
<style>body{{font-family:-apple-system,sans-serif;margin:16px}}#img{{width:min(640px,100%);border:1px solid #ddd;border-radius:6px}}
button{{padding:5px 14px;border:1px solid #ccc;border-radius:6px;background:#f6f6f8;cursor:pointer}}</style></head>
<body><h2>Membrane transport between cells (spike)</h2>
<p><b>S</b> (red) is declared membrane-permeable (Smoldyn <code>transmit</code>); it is made/seeded in the left cell,
crosses the membrane, diffuses in the extracellular <b>field</b> (red haze), and is taken up by the neighbor cells.
A/B/C stay confined (<code>reflect</code>). The extracellular representation is a swappable <b>adapter</b>. {leg}</p>
<img id="img"><div style="margin-top:8px"><button id="play">Play</button>
<input id="sl" type="range" min="0" max="{len(imgs)-1}" value="0" style="width:420px;vertical-align:middle">
<a href="{gif}" download="membrane-transport.gif" style="margin-left:14px;padding:5px 14px;border:1px solid #ccc;border-radius:6px;background:#f6f6f8;text-decoration:none;color:inherit">⬇ GIF</a></div>
<script>var F=[{fjs}],i=0,t=null,im=document.getElementById('img'),sl=document.getElementById('sl'),b=document.getElementById('play');
function show(k){{i=((k%F.length)+F.length)%F.length;im.src=F[i];sl.value=i}}sl.oninput=e=>{{stop();show(+e.target.value)}}
function stop(){{if(t){{clearInterval(t);t=null;b.textContent='Play'}}}}b.onclick=()=>{{if(t)stop();else{{t=setInterval(()=>show(i+1),240);b.textContent='Pause'}}}};show(0);</script>
</body></html>""", encoding="utf-8")
    print(f"wrote {OUT} ({len(imgs)} frames)")


if __name__ == "__main__":
    main()
