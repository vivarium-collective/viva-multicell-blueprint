"""SPIKE (throwaway): three mechanics paradigms running together, coupled by cell id.

CPM (voxel), vertex (polygon) and center-based (munk disc) each hold a representation
of the SAME cells (shared blueprint id). Between steps we apply:
  A. CPM target area  <-  the cell's vertex-twin area   (vertex topology -> CPM size)
  B. munk center      <-  the cell's vertex centroid     (tug; munk chases vertex)
and render every cell all three ways in one overlay, so you watch them converge.

Run:  PYTHONPATH=. python scripts/spike_tri_coupled.py  ->  reports/assets/tri_coupled_spike.html
"""
from __future__ import annotations
import base64, io, math, json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPoly
from matplotlib.path import Path as MplPath
from matplotlib.colors import to_rgb

from process_bigraph import Composite
from viva_multicell_blueprint.blueprint.model import Blueprint
from viva_multicell_blueprint.compile import compile as compile_bp
from viva_multicell_blueprint.core import build_core

BACKENDS = ("cpm", "vertex", "munk")
TYPE_COLOR = {"A": "#2e7d32", "B": "#D55E00", "H": "#2b6cb0", "I": "#c7761b"}
N_STEPS = 24

# A small, fast, visually legible blueprint: two adhesive types on a 24x24 patch.
BP = {
    "blueprint_version": "1.0",
    "domain": {"extent_um": [24, 24], "grid_resolution_um": 1, "boundary": "noflux"},
    "cell_types": [{"name": "A", "target_size": 16}, {"name": "B", "target_size": 16}],
    "adhesion": [
        {"a": "A", "b": "A", "strength": 4}, {"a": "B", "b": "B", "strength": 4},
        {"a": "A", "b": "B", "strength": 12},
        {"a": "A", "b": "medium", "strength": 16}, {"a": "B", "b": "medium", "strength": 16},
    ],
    "initial_state": {
        "kind": "blocks",
        "block": {"type": "A", "origin": [6, 6], "shape": [3, 3], "spacing": 4, "size": 16},
        "extra_blocks": [{"type": "B", "origin": [8, 8], "shape": [2, 2], "spacing": 4, "size": 16}],
    },
}


def make(bp, backend):
    doc = compile_bp(bp, backend=backend)
    comp = Composite({"state": doc["state"]}, core=build_core())
    return comp.state["mechanics"]["instance"]


def emit(inst):
    """Geometry dict: positions (centroid), sizes (area/volume), footprints (shape)."""
    try:
        return inst._emit() or {}
    except Exception:
        return {}


import viva_cpm.schema as cpm_schema

CPM_MCS = 1          # one light sweep -> voxel/membrane character without sprawl
CPM_TEMP = 1.0       # low T: the CPM holds the re-seeded vertex shape (no adhesion here)
CPM_BLEND = 0.9      # the vertex shape dominates: CPM tracks it tightly
CPM_RES = 3          # voxels per µm: finer lattice -> sharper shape match
# position is a three-way consensus; munk OWNS position so it dominates the average.
W_MUNK, W_VERTEX, W_CPM = 0.6, 0.25, 0.15


def cpm_world(nx, ny, labels, types_map, tvol, contacts, seed):
    """A fresh CPM world seeded from a label grid (the legal re-seed path:
    load_world's `seed_labels` runs set_from_labels BEFORE finalize). Contact
    energies (surface tension) keep the cells compact + bounded so they hold the
    re-seeded vertex shape instead of dissolving into the medium."""
    spec = {
        "potts": {"dims": [int(nx), int(ny), 1], "boundary": "noflux",
                  "neighbor_order": 2, "temperature": CPM_TEMP, "seed": int(seed)},
        "contact": contacts,
        "seed_labels": {"labels": [int(x) for x in labels],
                        "types": {int(k): int(v) for k, v in types_map.items()},
                        "default_type": 0, "target_volume": float(tvol), "lambda_volume": 1.0},
    }
    return cpm_schema.load_world(spec)


def rasterize(vfoot, cpm_id, pts, nx, ny, offsets=None):
    """Flat (nx*ny) label grid: each voxel -> the cpm id of the vertex polygon that
    contains it (nearest-centroid tiebreak). 0 where no polygon covers it.

    `offsets[bid]` translates a cell's vertex polygon before rasterizing, so the
    CPM cell takes the vertex SHAPE but is CENTERED at the consensus position
    (vertex owns shape; munk/position consensus owns where it sits)."""
    desired = np.zeros(nx * ny, dtype=np.int64)
    best = np.full(nx * ny, 1e18)
    for bid, fp in (vfoot or {}).items():
        cid = cpm_id.get(str(bid))
        if cid is None or not isinstance(fp, dict):
            continue
        xy = fp.get("xy") or []
        if len(xy) < 3:
            continue
        poly = np.array(xy, float)
        if offsets and str(bid) in offsets:
            poly = poly + np.asarray(offsets[str(bid)], float)
        inside = MplPath(poly).contains_points(pts)
        if not inside.any():
            continue
        cx, cy = poly[:, 0].mean(), poly[:, 1].mean()
        d = (pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2
        take = inside & (d < best)
        desired[take] = cid
        best[take] = d[take]
    return desired


def relabel_to_stable(snap, seed_lab):
    """`seed_from_labels` renumbers cell ids each rebuild — map the snapshot's ids
    back to the stable ids we seeded, by maximum voxel overlap. Fixes the colour
    blinking AND keeps the per-cell correspondence (area/centroid) intact."""
    snap = np.asarray(snap, dtype=np.int64)
    seed_lab = np.asarray(seed_lab, dtype=np.int64)
    out = np.zeros_like(snap)
    for sid in np.unique(snap):
        if sid == 0:
            continue
        mask = snap == sid
        vals, counts = np.unique(seed_lab[mask], return_counts=True)
        nz = [(int(c), int(v)) for v, c in zip(vals, counts) if v != 0]
        out[mask] = (max(nz)[1] if nz else int(sid))
    return out


def labels_coms(labels, pts, cpm_to_bid):
    """Centroid per cell (bid -> [x,y]) from a flat label grid."""
    lab = np.asarray(labels)
    out = {}
    for cid in np.unique(lab):
        if cid == 0:
            continue
        m = lab == cid
        bid = cpm_to_bid.get(int(cid))
        if bid:
            out[bid] = [float(pts[m, 0].mean()), float(pts[m, 1].mean())]
    return out


def _pos(geom, b):
    return {str(c): [float(p[0]), float(p[1])]
            for c, p in (geom.get(b, {}).get("positions", {}) or {}).items()}


def main():
    bp = Blueprint.from_dict(BP)
    insts = {"vertex": make(bp, "vertex"), "munk": make(bp, "munk")}
    insts["munk"]._nudge_gain = 1.5
    insts["munk"]._squish = 0.8          # reported-shape oblong under crowding
    ext = BP["domain"]["extent_um"]; W, H = float(ext[0]), float(ext[1])
    nx, ny = int(W * CPM_RES), int(H * CPM_RES)
    xs = (np.arange(nx) + 0.5) / CPM_RES; ys = (np.arange(ny) + 0.5) / CPM_RES
    gx, gy = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([gx.ravel(), gy.ravel()])   # µm (index = y*nx + x)

    # cell-id correspondence: use the blueprint ids the vertex backend emits.
    v0 = emit(insts["vertex"])
    bids = sorted(((v0.get("footprints", {}) or {}).keys()),
                  key=lambda s: int(s) if str(s).isdigit() else s)
    cpm_id = {str(b): i + 1 for i, b in enumerate(bids)}       # bid -> cpm int id
    cpm_to_bid = {v: k for k, v in cpm_id.items()}
    tag_of = {t["name"]: i + 1 for i, t in enumerate(BP["cell_types"])}  # type name -> tag
    type_tag = {}
    for b in bids:
        fp = (v0.get("footprints", {}) or {}).get(b, {})
        type_tag[cpm_id[str(b)]] = tag_of.get(fp.get("type"), 1)
    tvol = float(BP["cell_types"][0].get("target_size", 16)) * CPM_RES * CPM_RES  # voxels
    # contact energies (surface tension) from the blueprint adhesion, so CPM cells
    # stay compact + bounded instead of dissolving into the medium.
    contacts = []
    for a in BP.get("adhesion", []):
        ta = 0 if a["a"] == "medium" else tag_of.get(a["a"])
        tb = 0 if a["b"] == "medium" else tag_of.get(a["b"])
        if ta is not None and tb is not None:
            contacts.append({"a": int(ta), "b": int(tb), "j": float(a["strength"])})

    cpm_labels = rasterize(v0.get("footprints", {}) or {}, cpm_id, pts, nx, ny)  # start ON the vertex shapes
    frames = []
    prev = None
    for step in range(N_STEPS):
        consensus, offsets, nudge, v_tsize = {}, {}, {}, {}
        if prev is not None:
            mp, vp, cp = _pos(prev, "munk"), _pos(prev, "vertex"), _pos(prev, "cpm")
            for bid in set(mp) | set(vp) | set(cp):                 # three-way consensus centroid
                acc, wsum = np.zeros(2), 0.0
                for src, wt in ((mp, W_MUNK), (vp, W_VERTEX), (cp, W_CPM)):
                    if bid in src:
                        acc += wt * np.asarray(src[bid], float); wsum += wt
                if wsum > 0:
                    consensus[bid] = (acc / wsum).tolist()
            # vertex owns SHAPE; place that shape at the consensus position -> the
            # CPM centroid is now pulled by munk + vertex (+ its own), not planted.
            offsets = {bid: [consensus[bid][0] - vp[bid][0], consensus[bid][1] - vp[bid][1]]
                       for bid in consensus if bid in vp}
            nudge = consensus                                       # munk springs toward consensus
            v_tsize = prev.get("_cpm_area", {})                     # vertex area <- CPM (CPM owns area)
            tgt = rasterize(prev.get("vertex", {}).get("footprints", {}) or {}, cpm_id, pts, nx, ny, offsets)
            m = (np.random.random(nx * ny) < CPM_BLEND) & (tgt > 0)
            cpm_labels = cpm_labels.copy(); cpm_labels[m] = tgt[m]
        # CPM: real engine, re-seeded to the (blended) shape-at-consensus, then MCS
        try:
            w = cpm_world(nx, ny, cpm_labels, type_tag, tvol, contacts, seed=step)
            w.step(CPM_MCS)
            cpm_labels = relabel_to_stable(w.snapshot(), cpm_labels)
        except Exception as e:
            print(f"  [cpm] step {step}: {e}")
        # CPM owns AREA -> feeds munk (C) + vertex
        _lab = np.asarray(cpm_labels); cpm_area = {}
        for cid in np.unique(_lab):
            if cid == 0:
                continue
            bid = cpm_to_bid.get(int(cid))
            if bid:
                cpm_area[str(bid)] = float((_lab == cid).sum()) / (CPM_RES * CPM_RES)  # µm²
        geom = {"_cpm_labels": list(cpm_labels), "_cpm_area": cpm_area,
                "cpm": {"positions": labels_coms(cpm_labels, pts, cpm_to_bid)}}
        for b in ("vertex", "munk"):
            inp = {"fates": {}, "field_grid": {}}
            if b == "munk":
                inp["nudge"] = nudge                 # position -> consensus (munk owns it)
                inp["target_size"] = cpm_area        # area <- CPM (C)
            if b == "vertex":
                inp["target_size"] = v_tsize         # area <- CPM
            try:
                insts[b].update(inp, 1.0)
            except Exception as e:
                print(f"  [{b}] step {step}: {e}")
            geom[b] = emit(insts[b])
        frames.append(geom)
        prev = geom

    html = render(frames, bp, nx, ny, W, H, cpm_to_bid, type_tag)
    out = Path("workspace/reports/assets/tri_coupled_spike.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out}  ({out.stat().st_size//1024} KB, {len(frames)} frames)")
    _report_convergence(frames)


def _centroids(geom, b):
    return {str(c): p for c, p in (geom.get(b, {}).get("positions", {}) or {}).items()}


def _report_convergence(frames):
    def spread(g):
        c, v, m = _centroids(g, "cpm"), _centroids(g, "vertex"), _centroids(g, "munk")
        ks = set(c) & set(v) & set(m)
        if not ks:
            return float("nan")
        vals = []
        for k in ks:
            pts = [c[k], v[k], m[k]]
            vals.append(max(math.hypot(a[0]-b[0], a[1]-b[1])
                            for a in pts for b in pts))
        return float(np.mean(vals))
    print("step | 3-way spread (max pairwise centroid distance, µm) — tighter = stronger consensus")
    for i in (0, len(frames)//2, len(frames)-1):
        print(f"{i:4d} | {spread(frames[i]):6.2f}")


_CPM_PAL = ["#2b6cb0", "#3182ce", "#4a90d9", "#2c5282", "#1f6feb", "#5a9bd4",
            "#c7761b", "#dd8a2a", "#b5651d", "#e0982e", "#2e7d32", "#3f9142"]


def _draw_poly(ax, fp, color, lw):
    if not isinstance(fp, dict):
        return
    xy = fp.get("xy") or []
    if len(xy) < 3:
        return
    ax.add_patch(MplPoly(np.array(xy, float), closed=True, fill=False,
                         edgecolor=color, lw=lw, alpha=0.95, zorder=3))


def render(frames, bp, nx, ny, W, H, cpm_to_bid, cpm_type):
    imgs = []
    for g in frames:
        fig, ax = plt.subplots(figsize=(5.4, 5.4), dpi=150)
        ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        lab = np.asarray(g.get("_cpm_labels", []), int)
        if lab.size >= nx * ny:
            grid = lab[: nx * ny].reshape(ny, nx)
            rgb = np.ones((ny, nx, 3))
            for cpmid in np.unique(grid):
                if cpmid == 0:
                    continue
                rgb[grid == cpmid] = to_rgb(_CPM_PAL[int(cpmid) % len(_CPM_PAL)])
            ax.imshow(rgb, origin="lower", extent=[0, W, 0, H],
                      interpolation="nearest", zorder=0)
        # vertex polygon outline (dark) — the shape CPM is pulled onto; munk disc (purple)
        for cid, fp in (g.get("vertex", {}).get("footprints", {}) or {}).items():
            _draw_poly(ax, fp, "#111111", 1.7)
        for cid, fp in (g.get("munk", {}).get("footprints", {}) or {}).items():
            _draw_poly(ax, fp, "#7b2d8e", 1.4)
        for cid, p in (g.get("munk", {}).get("positions", {}) or {}).items():
            ax.plot(p[0], p[1], "o", ms=3, color="#7b2d8e", zorder=4)
        raw = io.BytesIO(); fig.savefig(raw, format="png", bbox_inches="tight", dpi=150)
        plt.close(fig)
        imgs.append("data:image/png;base64," + base64.b64encode(raw.getvalue()).decode())
    frames_js = ",".join(json.dumps(u) for u in imgs)
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Tri-coupled spike</title>
<style>body{{font-family:-apple-system,sans-serif;margin:16px;background:#fff;color:#1b1b1f}}
#img{{width:640px;height:auto;border:1px solid #ddd;border-radius:6px}}
.k{{display:inline-block;margin-right:14px;font-size:13px}}.d{{display:inline-block;width:11px;height:11px;border-radius:2px;vertical-align:middle;margin-right:4px}}
button{{padding:5px 14px;border:1px solid #ccc;border-radius:6px;background:#f6f6f8;cursor:pointer}}</style></head>
<body>
<h2>Tri-coupled composite — spike</h2>
<p>One cell, three paradigms, shared id — a <b>consensus</b> system. <b>Vertex owns shape</b> (black outline): each step the CPM rasterizes that outline, so the <b>CPM cell (filled)</b> takes the vertex shape with membrane texture. <b>Munk owns position</b>: all three centroids converge to a munk-weighted consensus, and the CPM shape is placed AT that consensus — so the CPM is tugged by munk too, not planted. <b>CPM owns area</b>: its true occupied area sets the <b>center-based disc (purple)</b> and the vertex target area.</p>
<p><span class="k"><span class="d" style="background:#2b6cb0"></span>CPM (filled cells)</span>
<span class="k"><span class="d" style="background:#111"></span>Vertex (polygon outline)</span>
<span class="k"><span class="d" style="background:#7b2d8e"></span>Center-based (disc)</span></p>
<img id="img" alt="frame"><div style="margin-top:8px"><button id="play">Play</button>
<input id="sl" type="range" min="0" max="{len(imgs)-1}" value="0" style="width:420px;vertical-align:middle"></div>
<script>var F=[{frames_js}],i=0,t=null,im=document.getElementById('img'),sl=document.getElementById('sl'),b=document.getElementById('play');
function show(k){{i=((k%F.length)+F.length)%F.length;im.src=F[i];sl.value=i;}}
sl.oninput=e=>{{stop();show(+e.target.value);}};function stop(){{if(t){{clearInterval(t);t=null;b.textContent='Play';}}}}
b.onclick=()=>{{if(t)stop();else{{t=setInterval(()=>show(i+1),200);b.textContent='Pause';}}}};show(0);</script>
</body></html>"""


if __name__ == "__main__":
    main()
