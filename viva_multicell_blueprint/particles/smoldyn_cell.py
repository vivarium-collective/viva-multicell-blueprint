"""Per-cell Smoldyn stepper — the shared physics for the bounded-particles coupling.

This module is deliberately **standalone**: it imports only ``smoldyn`` + ``numpy`` +
``matplotlib.path`` and nothing from ``viva_multicell_blueprint``. That lets it run

* **in-process** on Linux/Python 3.12 (where ``smoldyn`` has a binary wheel), imported
  via the package, and
* **in a Python 3.11 subprocess worker** (macOS, where Smoldyn's only binary is 3.11),
  imported flat by adding this directory to ``sys.path`` — so it never triggers the
  3.12-only package ``__init__``.

A "cell" is one tyssue polygon. Its particles are confined to that polygon by a
reflective Smoldyn surface built from one line-segment panel per edge; a reversible
reaction ``A + B <-> C`` runs inside. Because the polygon moves/deforms every step, the
sim is rebuilt each step from the particles' last positions (particles that fell outside
the new boundary are pulled back in).
"""
from __future__ import annotations

import contextlib
import os
import random

import numpy as np
from matplotlib.path import Path as MplPath


@contextlib.contextmanager
def _suppress_c_stdout():
    """Redirect the C-level stdout (fd 1) to /dev/null. Smoldyn writes notifications
    ("Simulation complete") straight to fd 1, which would corrupt the worker's
    JSON-over-stdout protocol — so every Smoldyn call is wrapped in this."""
    saved = os.dup(1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        yield
    finally:
        os.dup2(saved, 1)
        os.close(saved)
        os.close(devnull)

# default coupling parameters (override via the `spec` arg)
DEFAULTS = {
    "species": {"A": 1.6, "B": 1.6, "C": 0.9},   # name -> diffusion coeff (µm²/s)
    "reaction": {"subs": ["A", "B"], "prd": "C", "kf": 8.0, "kb": 2.0},
    "frame_dt": 1.0,       # sim time advanced per trajectory step
    "smol_dt": 0.02,       # Smoldyn internal timestep
    "colors": {"A": "#d1495b", "B": "#2e6fb0", "C": "#3f9142"},
}


def _convex_hull(points):
    """Andrew's monotone chain hull (dependency-free fallback)."""
    pts = sorted(set((float(x), float(y)) for x, y in points))
    if len(pts) < 3:
        return pts
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def occupancy_outline(pixels):
    """Ordered boundary polygon of a set of occupied unit voxels — the CPM
    ``kind="occupancy"`` footprint (a list of integer voxel coords, NOT a boundary).

    Each pixel ``[x, y]`` is the unit square ``[x, x+1] x [y, y+1]``; an edge shared by
    two occupied voxels is interior, so edges that belong to exactly ONE voxel form the
    cell boundary. Those edges are chained into an ordered loop (handles concave cells).
    Falls back to the convex hull of the voxel corners if the boundary self-touches.
    """
    occ = {(int(round(x)), int(round(y))) for x, y in pixels}
    if not occ:
        return None
    edge_count = {}
    for (x, y) in occ:
        c = [(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)]
        for i in range(4):
            a, b = c[i], c[(i + 1) % 4]
            key = (a, b) if a <= b else (b, a)
            edge_count[key] = edge_count.get(key, 0) + 1
    bedges = [e for e, n in edge_count.items() if n == 1]
    if not bedges:
        return None
    from collections import defaultdict
    adj = defaultdict(list)
    for a, b in bedges:
        adj[a].append(b); adj[b].append(a)
    # a clean (non-self-touching) boundary has every vertex at degree 2
    if any(len(v) != 2 for v in adj.values()):
        hull = _convex_hull([pt for e in bedges for pt in e])
        return [[float(x), float(y)] for x, y in hull] if len(hull) >= 3 else None
    start = bedges[0][0]
    loop = [start]
    prev, cur = None, start
    while True:
        a, b = adj[cur]
        nxt = b if a == prev else a
        if nxt == start:
            break
        loop.append(nxt)
        prev, cur = cur, nxt
        if len(loop) > len(bedges) + 2:
            break
    return [[float(x), float(y)] for x, y in loop] if len(loop) >= 3 else None


def sanitize_poly(poly, min_area=1.0, eps=1e-3):
    """Clean a polygon so Smoldyn's surface builder can't crash on it: drop
    consecutive-duplicate vertices (a zero-length line panel segfaults Smoldyn), close
    it, and reject anything with < 3 vertices or negligible area. Returns a list of
    [x, y] or ``None`` (degenerate — e.g. a 1-voxel CPM cell at division)."""
    arr = np.asarray(poly, float)
    if arr.ndim != 2 or len(arr) < 3:
        return None
    keep = [arr[0]]
    for p in arr[1:]:
        if float(np.hypot(p[0] - keep[-1][0], p[1] - keep[-1][1])) > eps:
            keep.append(p)
    arr = np.asarray(keep)
    if len(arr) >= 2 and float(np.hypot(arr[0][0] - arr[-1][0], arr[0][1] - arr[-1][1])) < eps:
        arr = arr[:-1]
    if len(arr) < 3:
        return None
    area = 0.5 * abs(np.dot(arr[:, 0], np.roll(arr[:, 1], -1)) -
                     np.dot(arr[:, 1], np.roll(arr[:, 0], -1)))
    if area < min_area:
        return None
    return arr.tolist()


def rand_in_poly(poly, n, rng=None):
    """n points uniformly inside a polygon (rejection sampling)."""
    rng = rng or random.Random(0)
    arr = np.asarray(poly, float)
    lo, hi = arr.min(0), arr.max(0)
    path = MplPath(arr)
    out = []
    while len(out) < n:
        p = [rng.uniform(lo[0], hi[0]), rng.uniform(lo[1], hi[1])]
        if path.contains_point(p):
            out.append(p)
    return out


def clamp_inside(xy, path, centroid):
    """Pull a point toward the centroid until it lands inside the polygon."""
    if path.contains_point(xy):
        return [float(xy[0]), float(xy[1])]
    p = np.asarray(xy, float)
    c = np.asarray(centroid, float)
    for f in (0.3, 0.6, 0.85, 0.97):
        q = p + (c - p) * f
        if path.contains_point(q):
            return [float(q[0]), float(q[1])]
    return [float(c[0]), float(c[1])]


def step_cell(poly, parts, seed, spec=None):
    """Advance one cell's particles by ``frame_dt`` inside ``poly``.

    ``parts`` is a list of ``[species, x, y]``; returns the updated list after
    diffusion + the reversible reaction + reflection off the (current) polygon.
    """
    import smoldyn

    spec = {**DEFAULTS, **(spec or {})}
    species = spec["species"]
    rxn = spec["reaction"]
    clean = sanitize_poly(poly)
    if clean is None:                 # degenerate cell this step — hold particles as-is
        return parts
    arr = np.asarray(clean, float)
    lo = [float(arr[:, 0].min() - 1.0), float(arr[:, 1].min() - 1.0)]
    hi = [float(arr[:, 0].max() + 1.0), float(arr[:, 1].max() + 1.0)]
    path = MplPath(arr)
    cen = arr.mean(0)

    with _suppress_c_stdout():
        s = smoldyn.Simulation(low=lo, high=hi, boundary_type="r", seed=int(seed))
        objs = {name: s.addSpecies(name, difc=float(d), display_size=2)
                for name, d in species.items()}
        if rxn:
            s.addBidirectionalReaction(
                "rxn", subs=[objs[x] for x in rxn["subs"]], prds=[objs[rxn["prd"]]],
                kf=float(rxn["kf"]), kb=float(rxn["kb"]))
        panels = [smoldyn.Triangle(vertices=[list(arr[i]), list(arr[(i + 1) % len(arr)])])
                  for i in range(len(arr))]
        srf = s.addSurface("cell", panels=panels)
        srf.setAction("both", list(objs.values()), "reflect")
        for sp, x, y in parts:
            if sp in objs:
                objs[sp].addToSolution(1, pos=clamp_inside([x, y], path, cen))
        s.addOutputData("molpos")
        s.addCommand(cmd="listmols2 molpos", cmd_type="E")
        s.setGraphics("none")
        s.updateSim()
        s.run(stop=float(spec["frame_dt"]), dt=float(spec["smol_dt"]))
        data = s.getOutputData("molpos", True) or []
    if not data:
        return parts
    name_map = {i + 1: name for i, name in enumerate(species)}   # 0=empty, 1=first…
    last = data[-1][0]
    return [[name_map[int(r[1])], float(r[3]), float(r[4])]
            for r in data if r[0] == last and int(r[1]) in name_map]


def smoldyn_available():
    try:
        import smoldyn  # noqa: F401
        return True
    except Exception:
        return False
