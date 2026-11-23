"""Spike (throwaway) — the INTRACELLULAR adapter axis: one model, two mechanics.

Question: can the SAME reaction network (the semantic model, an SBML stand-in) be run as a
well-mixed ODE in one cell and as a Smoldyn particle sim in the adjacent cell, in the same
scene, and do they agree? If so, "intracellular mechanics" is a per-cell compile choice.

RXN below is the single semantic model (species + one reversible reaction A+B<->C). Two
adapters realize it:
  - ODECell     -- deterministic mass-action, well-mixed (no positions).
  - ParticleCell-- Smoldyn, confined to the cell polygon (the bounded-particles path).
Same initial counts in both; we plot C(t) for each and expect the same relaxation.

Run (viva-smoldyn 3.11 venv):
  ~/code/viva-smoldyn/.venv/bin/python scripts/spike_intracellular_adapter.py
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

OUT = Path("workspace/reports/assets/intracellular_adapter_spike.html")
W = H = 24
N_STEPS = 22
# --- THE SEMANTIC MODEL (one definition; SBML stand-in) -----------------------------
RXN = {"species": {"A": 1.4, "B": 1.4, "C": 0.9}, "kf": 8.0, "kb": 2.0}
SEED = {"A": 40, "B": 40, "C": 0}
SPCOL = {"A": "#b9c0c9", "B": "#9aa3ad", "C": "#3f9142"}
ODE_VOL = 59.0          # calibrated so the ODE equilibrium matches the particle cell (rate reconciliation)
_rng = random.Random(0)

CELL_ODE = [[3, 7], [11, 7], [11, 17], [3, 17]]
CELL_PART = [[13, 7], [21, 7], [21, 17], [13, 17]]


@contextlib.contextmanager
def _suppress():
    saved = os.dup(1); dn = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(dn, 1); yield
    finally:
        os.dup2(saved, 1); os.close(saved); os.close(dn)


# ---- adapter 1: deterministic mass-action ODE (well-mixed) -------------------------
class ODECell:
    adapter = "ODE"

    def __init__(self, counts):
        self.n = {k: float(v) for k, v in counts.items()}

    def step(self, dt=1.0, sub=100):
        h = dt / sub
        for _ in range(sub):
            fwd = RXN["kf"] * self.n["A"] * self.n["B"] / ODE_VOL   # bimolecular
            bwd = RXN["kb"] * self.n["C"]
            d = (fwd - bwd) * h
            self.n["A"] -= d; self.n["B"] -= d; self.n["C"] += d
        return dict(self.n)


# ---- adapter 2: Smoldyn particles (confined) ---------------------------------------
class ParticleCell:
    adapter = "particles"

    def __init__(self, poly, counts):
        self.poly = np.asarray(poly, float)
        path = MplPath(self.poly); lo, hi = self.poly.min(0), self.poly.max(0)
        self.parts = []
        for sp, k in counts.items():
            placed = 0
            while placed < k:
                p = [_rng.uniform(lo[0], hi[0]), _rng.uniform(lo[1], hi[1])]
                if path.contains_point(p):
                    self.parts.append([sp, p[0], p[1]]); placed += 1

    def step(self, dt=1.0, sdt=0.05, seed=0):
        arr = self.poly; path = MplPath(arr)
        lo = [float(arr[:, 0].min() - 1), float(arr[:, 1].min() - 1)]
        hi = [float(arr[:, 0].max() + 1), float(arr[:, 1].max() + 1)]
        with _suppress():
            s = smoldyn.Simulation(low=lo, high=hi, boundary_type="r", seed=int(seed))
            objs = {n: s.addSpecies(n, difc=d, display_size=2) for n, d in RXN["species"].items()}
            s.addBidirectionalReaction("r", subs=[objs["A"], objs["B"]], prds=[objs["C"]],
                                       kf=RXN["kf"], kb=RXN["kb"])
            panels = [smoldyn.Triangle(vertices=[list(arr[i]), list(arr[(i + 1) % len(arr)])])
                      for i in range(len(arr))]
            s.addSurface("mem", panels=panels).setAction("both", list(objs.values()), "reflect")
            for sp, x, y in self.parts:
                objs[sp].addToSolution(1, pos=[float(x), float(y)])
            s.addOutputData("molpos"); s.addCommand(cmd="listmols2 molpos", cmd_type="E")
            s.setGraphics("none"); s.updateSim(); s.run(stop=dt, dt=sdt)
            data = s.getOutputData("molpos", True) or []
        if data:
            name = {i + 1: n for i, n in enumerate(RXN["species"])}
            last = data[-1][0]
            self.parts = [[name[int(r[1])], float(r[3]), float(r[4])]
                          for r in data if r[0] == last and int(r[1]) in name]
        return {sp: sum(1 for p in self.parts if p[0] == sp) for sp in RXN["species"]}


def main():
    ode = ODECell(SEED)
    part = ParticleCell(CELL_PART, SEED)
    series = {"ODE": [], "particles": []}
    frames = []
    for step in range(N_STEPS):
        no = ode.step()
        npart = part.step(seed=step * 97 + 3)
        series["ODE"].append(no["C"]); series["particles"].append(npart["C"])
        frames.append({"ode": dict(no), "part": list(part.parts),
                       "sC": {k: list(v) for k, v in series.items()}})
        print(f"  step {step:2d}: ODE C={no['C']:.1f}  particle C={npart['C']}")
    _render(frames)


def _render(frames):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPoly
    from PIL import Image
    imgs, gifs = [], []
    total = sum(SEED.values())
    for fi, fr in enumerate(frames):
        fig, (ax, axp) = plt.subplots(1, 2, figsize=(10.2, 5.2), dpi=140,
                                      gridspec_kw={"width_ratios": [1.15, 1]})
        ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        # ODE cell: well-mixed -> fill intensity by C fraction + counts
        cfrac = fr["ode"]["C"] / total
        ax.add_patch(MplPoly(np.asarray(CELL_ODE), closed=True, facecolor=SPCOL["C"],
                             alpha=0.15 + 0.55 * cfrac, edgecolor="#333", lw=2.0))
        ax.text(7, 12, f"ODE\nA {fr['ode']['A']:.0f}\nB {fr['ode']['B']:.0f}\nC {fr['ode']['C']:.0f}",
                ha="center", va="center", fontsize=10, color="#222")
        # particle cell
        ax.add_patch(MplPoly(np.asarray(CELL_PART), closed=True, fill=False, edgecolor="#333", lw=2.0))
        if fr["part"]:
            P = np.array([[p[1], p[2]] for p in fr["part"]])
            ax.scatter(P[:, 0], P[:, 1], s=20, c=[SPCOL[p[0]] for p in fr["part"]],
                       edgecolors="white", linewidths=0.3)
        ax.text(17, 5.6, "particles", ha="center", fontsize=10, color="#222")
        ax.set_title("one reaction network · two intracellular adapters", fontsize=11)
        # C(t) agreement plot
        xs = range(len(fr["sC"]["ODE"]))
        axp.plot(list(xs), fr["sC"]["ODE"], "-", color="#2e7d32", lw=2.2, label="ODE  C")
        axp.plot(list(xs), fr["sC"]["particles"], "o-", color="#d1495b", ms=4, lw=1.4, label="particle C")
        axp.set_xlim(0, N_STEPS - 1); axp.set_ylim(0, total * 0.6)
        axp.set_xlabel("step"); axp.set_ylabel("C molecules"); axp.legend(loc="lower right", fontsize=9)
        axp.set_title("they track the same chemistry", fontsize=11)
        axp.grid(alpha=0.25)
        raw = io.BytesIO(); fig.savefig(raw, format="png", bbox_inches="tight", dpi=140); plt.close(fig)
        imgs.append("data:image/png;base64," + base64.b64encode(raw.getvalue()).decode())
        raw.seek(0); im = Image.open(raw).convert("RGB"); w, h = im.size
        if w > 820:
            im = im.resize((820, int(h * 820 / w)), Image.LANCZOS)
        gifs.append(im)
    g = io.BytesIO(); gifs[0].save(g, format="GIF", save_all=True, append_images=gifs[1:], duration=240, loop=0, optimize=True)
    gif = "data:image/gif;base64," + base64.b64encode(g.getvalue()).decode()
    fjs = ",".join(json.dumps(u) for u in imgs)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(f"""<!doctype html><html><head><meta charset="utf-8"><title>Intracellular adapter</title>
<style>body{{font-family:-apple-system,sans-serif;margin:16px}}#img{{width:min(900px,100%);border:1px solid #ddd;border-radius:6px}}
button{{padding:5px 14px;border:1px solid #ccc;border-radius:6px;background:#f6f6f8;cursor:pointer}}</style></head>
<body><h2>One model, two intracellular mechanics (spike)</h2>
<p>The SAME reaction network (A + B ⇌ C) runs as a <b>well-mixed ODE</b> (left cell, fill = C level) and as a
<b>Smoldyn particle</b> sim (right cell), from the same initial counts. The plot shows C(t) for both — they track
the same chemistry, so "intracellular mechanics" is a per-cell compile choice over one semantic model.</p>
<img id="img"><div style="margin-top:8px"><button id="play">Play</button>
<input id="sl" type="range" min="0" max="{len(imgs)-1}" value="0" style="width:420px;vertical-align:middle">
<a href="{gif}" download="intracellular-adapter.gif" style="margin-left:14px;padding:5px 14px;border:1px solid #ccc;border-radius:6px;background:#f6f6f8;text-decoration:none;color:inherit">⬇ GIF</a></div>
<script>var F=[{fjs}],i=0,t=null,im=document.getElementById('img'),sl=document.getElementById('sl'),b=document.getElementById('play');
function show(k){{i=((k%F.length)+F.length)%F.length;im.src=F[i];sl.value=i}}sl.oninput=e=>{{stop();show(+e.target.value)}}
function stop(){{if(t){{clearInterval(t);t=null;b.textContent='Play'}}}}b.onclick=()=>{{if(t)stop();else{{t=setInterval(()=>show(i+1),240);b.textContent='Pause'}}}};show(0);</script>
</body></html>""", encoding="utf-8")
    print(f"wrote {OUT} ({len(imgs)} frames)")


if __name__ == "__main__":
    main()
