"""TriCoupledProcess: all three mechanics paradigms running together, coupled.

CPM (voxel), vertex (polygon) and center-based (munk disc) each hold a
representation of the SAME cells (shared blueprint id), stepped in lockstep and
coupled into one consensus system:

* **vertex owns SHAPE** — each step the CPM rasterizes its vertex twin's polygon
  into a fresh lattice (``seed_from_labels`` on a new world, stable-id relabelled,
  with contact energies), so the CPM cell takes the vertex shape + membrane
  texture — the biology vertex can't render, on a shape vertex computes (T1s,
  junction tension).
* **munk owns POSITION** — all three centroids converge to a munk-weighted
  consensus; munk (fast center-based physics) springs toward it, and the CPM
  shape is placed AT the consensus, so the CPM centroid is tugged by munk + vertex.
* **CPM owns AREA** — the CPM cell's true occupied area sets the munk disc size
  and the vertex target area.

A real :class:`~process_bigraph.Process`: it builds the three backends in
``__init__`` (compiling the blueprint onto each), runs one coupled step per
``update``, emits the per-paradigm geometry, and writes a self-contained overlay
HTML (all three representations per cell) to ``reports/assets/<overlay_out>.html``.
"""
from __future__ import annotations

import base64
import io
import json
import math
from pathlib import Path

import numpy as np
from matplotlib.path import Path as MplPath

from process_bigraph import Process

from ..blueprint.model import Blueprint
from ..compile import compile as _compile
from ..core import build_core

# position is a three-way consensus; munk OWNS position so it dominates the average.
W_MUNK, W_VERTEX, W_CPM = 0.6, 0.25, 0.15
_CPM_PAL = ["#2b6cb0", "#3182ce", "#4a90d9", "#2c5282", "#1f6feb", "#5a9bd4",
            "#c7761b", "#dd8a2a", "#b5651d", "#e0982e", "#2e7d32", "#3f9142"]


def _emit(inst):
    try:
        return inst._emit() or {}
    except Exception:
        return {}


def _pos(geom, b):
    return {str(c): [float(p[0]), float(p[1])]
            for c, p in (geom.get(b, {}).get("positions", {}) or {}).items()}


def _rasterize(vfoot, cpm_id, pts, nx, ny, offsets=None):
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


def _relabel_to_stable(snap, seed_lab):
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


def _labels_coms(labels, pts, cpm_to_bid):
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


class TriCoupledProcess(Process):
    """Three coupled mechanics paradigms on one shared cell set (see module docstring)."""

    config_schema = {
        "blueprint": "any",            # dict / path / inline YAML
        "cpm_res": {"_type": "integer", "_default": 3},     # voxels per µm
        "cpm_temp": {"_type": "float", "_default": 1.0},
        "cpm_mcs": {"_type": "integer", "_default": 1},
        "cpm_blend": {"_type": "float", "_default": 0.9},
        "munk_squish": {"_type": "float", "_default": 0.8},   # oblong-under-crowding
        "overlay_out": {"_type": "string", "_default": "tri_coupled"},
    }

    def __init__(self, config=None, core=None):
        super().__init__(config, core)
        import viva_cpm.schema as cpm_schema
        self._cpm_schema = cpm_schema
        bp_src = self.config.get("blueprint")
        bp = bp_src if isinstance(bp_src, Blueprint) else Blueprint.from_dict(bp_src) \
            if isinstance(bp_src, dict) else Blueprint.load(bp_src)
        self._bp_dict = {
            "domain": dict(bp.domain), "cell_types": [dict(c) for c in bp.cell_types],
            "adhesion": [dict(a) for a in (bp.adhesion or [])],
        }
        from process_bigraph import Composite
        c = core or build_core()
        self._insts = {}
        for b in ("vertex", "munk"):
            doc = _compile(bp, backend=b)
            self._insts[b] = Composite({"state": doc["state"]}, core=c).state["mechanics"]["instance"]
        self._insts["munk"]._nudge_gain = 1.5
        self._insts["munk"]._squish = float(self.config.get("munk_squish", 0.8))

        self.RES = int(self.config.get("cpm_res", 3))
        self.TEMP = float(self.config.get("cpm_temp", 1.0))
        self.MCS = int(self.config.get("cpm_mcs", 1))
        self.BLEND = float(self.config.get("cpm_blend", 0.9))
        ext = self._bp_dict["domain"]["extent_um"]
        self.W, self.H = float(ext[0]), float(ext[1])
        self.nx, self.ny = int(self.W * self.RES), int(self.H * self.RES)
        xs = (np.arange(self.nx) + 0.5) / self.RES
        ys = (np.arange(self.ny) + 0.5) / self.RES
        gx, gy = np.meshgrid(xs, ys, indexing="xy")
        self._pts = np.column_stack([gx.ravel(), gy.ravel()])

        v0 = _emit(self._insts["vertex"])
        bids = sorted((v0.get("footprints", {}) or {}).keys(),
                      key=lambda s: int(s) if str(s).isdigit() else s)
        self._cpm_id = {str(b): i + 1 for i, b in enumerate(bids)}
        self._cpm_to_bid = {v: k for k, v in self._cpm_id.items()}
        tag_of = {t["name"]: i + 1 for i, t in enumerate(self._bp_dict["cell_types"])}
        self._type_tag = {}
        for b in bids:
            fp = (v0.get("footprints", {}) or {}).get(b, {})
            self._type_tag[self._cpm_id[str(b)]] = tag_of.get(fp.get("type"), 1)
        self._tvol = float(self._bp_dict["cell_types"][0].get("target_size", 16)) * self.RES * self.RES
        self._contacts = []
        for a in self._bp_dict.get("adhesion", []):
            ta = 0 if a["a"] == "medium" else tag_of.get(a["a"])
            tb = 0 if a["b"] == "medium" else tag_of.get(a["b"])
            if ta is not None and tb is not None:
                self._contacts.append({"a": int(ta), "b": int(tb), "j": float(a["strength"])})
        self._cpm_labels = _rasterize(v0.get("footprints", {}) or {}, self._cpm_id, self._pts, self.nx, self.ny)
        self._prev = None
        self._frames = []
        self._out = Path("workspace/reports/assets") / (str(self.config.get("overlay_out", "tri_coupled")) + ".html")

    def inputs(self):
        return {}

    def outputs(self):
        return {"consensus_spread": "float", "n_cells": "integer"}

    def _cpm_world(self, labels, seed):
        spec = {
            "potts": {"dims": [self.nx, self.ny, 1], "boundary": "noflux",
                      "neighbor_order": 2, "temperature": self.TEMP, "seed": int(seed)},
            "contact": self._contacts,
            "seed_labels": {"labels": [int(x) for x in labels],
                            "types": {int(k): int(v) for k, v in self._type_tag.items()},
                            "default_type": 0, "target_volume": float(self._tvol), "lambda_volume": 1.0},
        }
        return self._cpm_schema.load_world(spec)

    def update(self, inputs, interval=1):
        nx, ny, pts = self.nx, self.ny, self._pts
        consensus, offsets, nudge, v_tsize = {}, {}, {}, {}
        if self._prev is not None:
            mp, vp, cp = _pos(self._prev, "munk"), _pos(self._prev, "vertex"), _pos(self._prev, "cpm")
            for bid in set(mp) | set(vp) | set(cp):
                acc, wsum = np.zeros(2), 0.0
                for src, wt in ((mp, W_MUNK), (vp, W_VERTEX), (cp, W_CPM)):
                    if bid in src:
                        acc += wt * np.asarray(src[bid], float); wsum += wt
                if wsum > 0:
                    consensus[bid] = (acc / wsum).tolist()
            offsets = {bid: [consensus[bid][0] - vp[bid][0], consensus[bid][1] - vp[bid][1]]
                       for bid in consensus if bid in vp}
            nudge = consensus
            v_tsize = self._prev.get("_cpm_area", {})
            tgt = _rasterize(self._prev.get("vertex", {}).get("footprints", {}) or {},
                             self._cpm_id, pts, nx, ny, offsets)
            m = (np.random.random(nx * ny) < self.BLEND) & (tgt > 0)
            self._cpm_labels = self._cpm_labels.copy(); self._cpm_labels[m] = tgt[m]
        try:
            w = self._cpm_world(self._cpm_labels, seed=len(self._frames))
            w.step(self.MCS)
            self._cpm_labels = _relabel_to_stable(w.snapshot(), self._cpm_labels)
        except Exception:
            pass
        _lab = np.asarray(self._cpm_labels); cpm_area = {}
        for cid in np.unique(_lab):
            if cid == 0:
                continue
            bid = self._cpm_to_bid.get(int(cid))
            if bid:
                cpm_area[str(bid)] = float((_lab == cid).sum()) / (self.RES * self.RES)
        geom = {"_cpm_labels": self._cpm_labels.tolist(), "_cpm_area": cpm_area,
                "cpm": {"positions": _labels_coms(self._cpm_labels, pts, self._cpm_to_bid)}}
        for b in ("vertex", "munk"):
            inp = {"fates": {}, "field_grid": {}}
            if b == "munk":
                inp["nudge"] = nudge; inp["target_size"] = cpm_area
            if b == "vertex":
                inp["target_size"] = v_tsize
            try:
                self._insts[b].update(inp, 1.0)
            except Exception:
                pass
            geom[b] = _emit(self._insts[b])
        self._frames.append(geom); self._prev = geom
        try:
            self._write_overlay()
        except Exception:
            pass
        return {"consensus_spread": self._spread(geom), "n_cells": len(self._cpm_to_bid)}

    def _spread(self, g):
        c, v, m = _pos(g, "cpm"), _pos(g, "vertex"), _pos(g, "munk")
        ks = set(c) & set(v) & set(m)
        if not ks:
            return 0.0
        vals = []
        for k in ks:
            ps = [c[k], v[k], m[k]]
            vals.append(max(math.hypot(a[0]-b[0], a[1]-b[1]) for a in ps for b in ps))
        return float(np.mean(vals))

    # ---- overlay viz (all three paradigms per cell) --------------------------
    def _write_overlay(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon as MplPoly
        from matplotlib.colors import to_rgb
        imgs = []
        for g in self._frames:
            fig, ax = plt.subplots(figsize=(5.4, 5.4), dpi=150)
            ax.set_xlim(0, self.W); ax.set_ylim(0, self.H); ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            lab = np.asarray(g.get("_cpm_labels", []), int)
            if lab.size >= self.nx * self.ny:
                grid = lab[: self.nx * self.ny].reshape(self.ny, self.nx)
                rgb = np.ones((self.ny, self.nx, 3))
                for cpmid in np.unique(grid):
                    if cpmid == 0:
                        continue
                    rgb[grid == cpmid] = to_rgb(_CPM_PAL[int(cpmid) % len(_CPM_PAL)])
                ax.imshow(rgb, origin="lower", extent=[0, self.W, 0, self.H],
                          interpolation="nearest", zorder=0)
            for _, fp in (g.get("vertex", {}).get("footprints", {}) or {}).items():
                xy = fp.get("xy") if isinstance(fp, dict) else None
                if xy and len(xy) >= 3:
                    ax.add_patch(MplPoly(np.array(xy, float), closed=True, fill=False,
                                         edgecolor="#111111", lw=1.7, zorder=3))
            for _, fp in (g.get("munk", {}).get("footprints", {}) or {}).items():
                xy = fp.get("xy") if isinstance(fp, dict) else None
                if xy and len(xy) >= 3:
                    ax.add_patch(MplPoly(np.array(xy, float), closed=True, fill=False,
                                         edgecolor="#7b2d8e", lw=1.4, zorder=3))
            raw = io.BytesIO(); fig.savefig(raw, format="png", bbox_inches="tight", dpi=150)
            plt.close(fig)
            imgs.append("data:image/png;base64," + base64.b64encode(raw.getvalue()).decode())
        frames_js = ",".join(json.dumps(u) for u in imgs)
        self._out.parent.mkdir(parents=True, exist_ok=True)
        self._out.write_text(f"""<!doctype html><html><head><meta charset="utf-8"><title>Tri-coupled</title>
<style>body{{font-family:-apple-system,sans-serif;margin:16px;background:#fff;color:#1b1b1f}}
#img{{width:640px;height:auto;border:1px solid #ddd;border-radius:6px}}
.k{{display:inline-block;margin-right:14px;font-size:13px}}.d{{display:inline-block;width:11px;height:11px;border-radius:2px;vertical-align:middle;margin-right:4px}}
button{{padding:5px 14px;border:1px solid #ccc;border-radius:6px;background:#f6f6f8;cursor:pointer}}</style></head>
<body><h2>Tri-coupled composite</h2>
<p>One cell, three paradigms, shared id — a consensus system. <b>Vertex owns shape</b> (black): the CPM rasterizes it, so the <b>CPM (filled)</b> takes the vertex shape. <b>Munk owns position</b>: all three centroids converge to a munk-weighted consensus and the CPM is placed there. <b>CPM owns area</b>: it sizes the <b>center-based disc (purple)</b> + vertex area.</p>
<p><span class="k"><span class="d" style="background:#2b6cb0"></span>CPM</span>
<span class="k"><span class="d" style="background:#111"></span>Vertex</span>
<span class="k"><span class="d" style="background:#7b2d8e"></span>Center-based</span></p>
<img id="img"><div style="margin-top:8px"><button id="play">Play</button>
<input id="sl" type="range" min="0" max="{len(imgs)-1}" value="0" style="width:420px;vertical-align:middle"></div>
<script>var F=[{frames_js}],i=0,t=null,im=document.getElementById('img'),sl=document.getElementById('sl'),b=document.getElementById('play');
function show(k){{i=((k%F.length)+F.length)%F.length;im.src=F[i];sl.value=i;}}
sl.oninput=e=>{{stop();show(+e.target.value);}};function stop(){{if(t){{clearInterval(t);t=null;b.textContent='Play';}}}}
b.onclick=()=>{{if(t)stop();else{{t=setInterval(()=>show(i+1),200);b.textContent='Pause';}}}};show(0);</script>
</body></html>""", encoding="utf-8")
