"""Spike stage 2 — Smoldyn particles confined inside each tyssue cell, split on division.

Reads the tyssue trajectory from stage 1 (per-step cell polygons + lineage) and, for
each cell each step, runs a Smoldyn sim confined to that cell's polygon (reflective
line panels) with a reversible reaction A+B<->C. The cell's polygon moves/deforms
every step, so the sim is rebuilt each step from the particles' last positions (the
same re-seed pattern the tri-coupled CPM uses). On division the mother's particles
are partitioned into the two daughter polygons by containment.

Run (viva-smoldyn's Python 3.11 venv, which has a working Smoldyn):
  ~/code/viva-smoldyn/.venv/bin/python scripts/spike_bounded_particles_stage2_smoldyn.py
"""
from __future__ import annotations

import base64
import io
import json
import random
from pathlib import Path

import numpy as np
from matplotlib.path import Path as MplPath

import smoldyn

IN = Path("workspace/reports/assets/_bounded_particles_tyssue.json")
OUT = Path("workspace/reports/assets/bounded_particles_spike.html")
SEED_PER_CELL = {"A": 35, "B": 35}     # initial molecules per cell
DIFC = {"A": 1.6, "B": 1.6, "C": 0.9}  # µm²/s
KF, KB = 8.0, 2.0                      # A+B -> C (kf), C -> A+B (kb)
FRAME_DT = 1.0                         # sim time advanced per trajectory step
SMOL_DT = 0.02
SPCOL = {"A": "#d1495b", "B": "#2e6fb0", "C": "#3f9142"}
_rng = random.Random(0)


def _rand_in_poly(poly, n):
    """n random points uniformly inside a polygon (rejection sampling)."""
    arr = np.asarray(poly)
    lo, hi = arr.min(0), arr.max(0)
    path = MplPath(arr)
    out = []
    while len(out) < n:
        p = [_rng.uniform(lo[0], hi[0]), _rng.uniform(lo[1], hi[1])]
        if path.contains_point(p):
            out.append(p)
    return out


def _clamp_inside(xy, poly_path, centroid):
    """Pull a point toward the centroid until it lands inside the polygon."""
    if poly_path.contains_point(xy):
        return xy
    p = np.asarray(xy, float)
    c = np.asarray(centroid, float)
    for f in (0.3, 0.6, 0.85, 0.97):
        q = p + (c - p) * f
        if poly_path.contains_point(q):
            return [float(q[0]), float(q[1])]
    return [float(c[0]), float(c[1])]


def _step_cell(poly, parts, seed):
    """Advance one cell's particles by FRAME_DT inside its polygon. `parts` is a list
    of (species, x, y); returns the updated list after diffusion + A+B<->C + reflection."""
    arr = np.asarray(poly, float)
    lo = [float(arr[:, 0].min() - 1.0), float(arr[:, 1].min() - 1.0)]
    hi = [float(arr[:, 0].max() + 1.0), float(arr[:, 1].max() + 1.0)]
    path = MplPath(arr)
    cen = arr.mean(0)
    s = smoldyn.Simulation(low=lo, high=hi, boundary_type="r", seed=int(seed))
    objs = {name: s.addSpecies(name, difc=DIFC[name], color=SPCOL[name], display_size=2)
            for name in ("A", "B", "C")}
    s.addBidirectionalReaction("dim", subs=[objs["A"], objs["B"]], prds=[objs["C"]], kf=KF, kb=KB)
    panels = [smoldyn.Triangle(vertices=[list(arr[i]), list(arr[(i + 1) % len(arr)])])
              for i in range(len(arr))]
    srf = s.addSurface("cell", panels=panels)
    srf.setAction("both", [objs["A"], objs["B"], objs["C"]], "reflect")
    for sp, x, y in parts:
        xy = _clamp_inside([x, y], path, cen)
        objs[sp].addToSolution(1, pos=xy)
    s.addOutputData("molpos")
    s.addCommand(cmd="listmols2 molpos", cmd_type="E")
    s.setGraphics("none")
    s.updateSim()
    s.run(stop=FRAME_DT, dt=SMOL_DT)
    data = s.getOutputData("molpos", True) or []
    name_map = {1: "A", 2: "B", 3: "C"}
    if not data:
        return parts
    last = data[-1][0]
    out = []
    for row in data:
        if row[0] != last:
            continue
        sp = name_map.get(int(row[1]))
        if sp:
            out.append((sp, float(row[3]), float(row[4])))
    return out


def main():
    traj = json.loads(IN.read_text())
    frames = traj["frames"]
    W, H = traj["extent_um"]

    # seed particles in each cell present at t0
    particles = {}   # cell_id -> [(species, x, y), ...]
    f0 = frames[0]
    for cid, poly in f0["footprints"].items():
        parts = []
        for sp, n in SEED_PER_CELL.items():
            parts += [(sp, x, y) for x, y in _rand_in_poly(poly, n)]
        particles[cid] = parts

    render_frames = []
    for fi, fr in enumerate(frames):
        foot = fr["footprints"]
        lineage = fr["lineage"]
        # division: a daughter cell appears with lineage[daughter]=mother. Split the
        # mother's particles into the two daughter polygons by containment.
        for daughter, mother in lineage.items():
            if daughter in particles or daughter not in foot:
                continue
            mother_parts = particles.get(mother, [])
            dpath = MplPath(np.asarray(foot[daughter]))
            keep, give = [], []
            for p in mother_parts:
                (give if dpath.contains_point([p[1], p[2]]) else keep).append(p)
            particles[mother] = keep
            particles[daughter] = give
            print(f"  step {fr['step']}: split {mother} -> {mother}({len(keep)}) + {daughter}({len(give)})")

        # advance every live cell's confined particles
        snapshot = {}
        for cid, poly in foot.items():
            parts = particles.get(cid)
            if parts is None:
                parts = []
                for sp, n in SEED_PER_CELL.items():
                    parts += [(sp, x, y) for x, y in _rand_in_poly(poly, n)]
            particles[cid] = _step_cell(poly, parts, seed=fi * 997 + hash(cid) % 9973)
            snapshot[cid] = {"poly": poly, "parts": particles[cid]}
        render_frames.append(snapshot)
        counts = {s: sum(1 for snap in snapshot.values() for p in snap["parts"] if p[0] == s)
                  for s in ("A", "B", "C")}
        print(f"  frame {fi}: cells={len(snapshot)}  A/B/C={counts['A']}/{counts['B']}/{counts['C']}")

    _render(render_frames, W, H)


def _render(render_frames, W, H):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPoly
    from PIL import Image
    # one stable colour per cell id (assigned in first-appearance order), so particles
    # can be tracked by which cell they belong to — and a division shows as a new colour.
    _PAL = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
            "#e377c2", "#17becf", "#bcbd22", "#7f7f7f", "#393b79", "#637939"]
    cell_color, order = {}, []
    for snapshot in render_frames:
        for cid in snapshot:
            if cid not in cell_color:
                cell_color[cid] = _PAL[len(cell_color) % len(_PAL)]
                order.append(cid)
    imgs, gif_frames = [], []
    for snapshot in render_frames:
        fig, ax = plt.subplots(figsize=(5.6, 5.6), dpi=140)
        ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for cid, snap in snapshot.items():
            cc = cell_color[cid]
            ax.add_patch(MplPoly(np.asarray(snap["poly"]), closed=True, fill=False,
                                 edgecolor=cc, lw=1.8, zorder=1))
            if snap["parts"]:
                P = np.array([[p[1], p[2]] for p in snap["parts"]])
                ax.scatter(P[:, 0], P[:, 1], s=13, c=cc, zorder=2, edgecolors="none")
        raw = io.BytesIO(); fig.savefig(raw, format="png", bbox_inches="tight", dpi=140)
        plt.close(fig)
        imgs.append("data:image/png;base64," + base64.b64encode(raw.getvalue()).decode())
        raw.seek(0)
        im = Image.open(raw).convert("RGB")
        w, h = im.size
        if w > 560:
            im = im.resize((560, int(h * 560 / w)), Image.LANCZOS)
        gif_frames.append(im)
    gbuf = io.BytesIO()
    gif_frames[0].save(gbuf, format="GIF", save_all=True, append_images=gif_frames[1:],
                       duration=220, loop=0, optimize=True)
    gif_uri = "data:image/gif;base64," + base64.b64encode(gbuf.getvalue()).decode()
    frames_js = ",".join(json.dumps(u) for u in imgs)
    legend = "".join(f'<span class="k"><span class="d" style="background:{cell_color[c]}"></span>cell {c}</span>'
                     for c in order)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(f"""<!doctype html><html><head><meta charset="utf-8"><title>Bounded particles</title>
<style>body{{font-family:-apple-system,sans-serif;margin:16px;background:#fff;color:#1b1b1f}}
#img{{width:min(640px,100%);height:auto;border:1px solid #ddd;border-radius:6px}}
.k{{display:inline-block;margin-right:14px;font-size:13px}}.d{{display:inline-block;width:11px;height:11px;border-radius:50%;vertical-align:middle;margin-right:4px}}
button{{padding:5px 14px;border:1px solid #ccc;border-radius:6px;background:#f6f6f8;cursor:pointer}}</style></head>
<body><h2>Smoldyn particles bounded by tyssue cells</h2>
<p>Each tyssue (vertex) cell is a moving polygon; a <b>Smoldyn</b> particle sim runs
<b>inside each one</b>, reflecting off the cell boundary, with a reversible reaction
<b>A + B ⇌ C</b>. Particles are <b>coloured by the cell they belong to</b> (matching the
cell outline) so you can track them as cells move — when a cell divides, its molecules
are partitioned into the two daughters, which get their own colours: {legend}</p>
<img id="img"><div style="margin-top:8px"><button id="play">Play</button>
<input id="sl" type="range" min="0" max="{len(imgs)-1}" value="0" style="width:420px;vertical-align:middle">
<a href="{gif_uri}" download="bounded-particles.gif" style="margin-left:14px;padding:5px 14px;border:1px solid #ccc;border-radius:6px;background:#f6f6f8;text-decoration:none;color:inherit;font-size:14px">⬇ Download GIF</a></div>
<script>var F=[{frames_js}],i=0,t=null,im=document.getElementById('img'),sl=document.getElementById('sl'),b=document.getElementById('play');
function show(k){{i=((k%F.length)+F.length)%F.length;im.src=F[i];sl.value=i;}}
sl.oninput=e=>{{stop();show(+e.target.value);}};function stop(){{if(t){{clearInterval(t);t=null;b.textContent='Play';}}}}
b.onclick=()=>{{if(t)stop();else{{t=setInterval(()=>show(i+1),220);b.textContent='Pause';}}}};show(0);</script>
</body></html>""", encoding="utf-8")
    print(f"wrote {OUT}  ({len(imgs)} frames)")


if __name__ == "__main__":
    main()
