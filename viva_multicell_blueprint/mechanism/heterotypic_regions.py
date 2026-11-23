"""Heterotypic tissue regions: two tissue TYPES, two physics, coupled at an interface.

``HeterotypicRegionsProcess`` models a tissue cross-section as two compartments simulated by
DIFFERENT mechanics: an **epithelium** (a sheet, simulated by the vertex backend) sitting on a
**mesenchyme** (loose cells, simulated by the center-based munk backend). The two are coupled at
their interface: the epithelium's basal edge is a reflective floor the mesenchyme cells press up
against but cannot cross. So a single tissue is composed of two paradigms, each modelling the
compartment it fits — not one consensus on one cell set, but two tissues meeting at a boundary.
"""
from __future__ import annotations

import base64
import io
import json
import tempfile
from pathlib import Path

import numpy as np
from process_bigraph import Process

from ..blueprint.model import Blueprint
from ..compile import compile as _compile
from ..core import build_core


def _emit(inst):
    try:
        return inst._emit() or {}
    except Exception:
        return {}


class HeterotypicRegionsProcess(Process):
    """A vertex epithelium on a munk mesenchyme, coupled at their basal interface."""

    config_schema = {
        "extent": {"_type": "integer", "_default": 30},
        "n_epi": {"_type": "integer", "_default": 6},     # epithelial cells (a row)
        "n_mes": {"_type": "integer", "_default": 10},    # mesenchymal cells (a cluster)
        "overlay_out": {"_type": "string", "_default": "heterotypic_regions"},
    }

    def __init__(self, config=None, core=None):
        super().__init__(config, core)
        from process_bigraph import Composite
        c = core or build_core()
        self.W = self.H = float(self.config.get("extent", 30))
        self._iface = self.H * 0.55     # interface height: epithelium above, mesenchyme below
        n_epi = int(self.config.get("n_epi", 6))
        n_mes = int(self.config.get("n_mes", 10))

        # epithelium: a row of cells just above the interface -> vertex
        epi = [(6.0 + i * (self.W - 12) / max(n_epi - 1, 1), self._iface + 3.0) for i in range(n_epi)]
        # mesenchyme: a loose cluster below the interface -> munk
        rng = np.random.default_rng(0)
        mes = [(self.W / 2 + rng.uniform(-7, 7), self._iface - 4 - rng.uniform(0, 8)) for _ in range(n_mes)]

        self._epi_inst = self._compile_region(c, Composite, epi, "epithelium", "vertex")
        self._mes_inst = self._compile_region(c, Composite, mes, "mesenchyme", "munk")
        self._frames = []
        self._imgs, self._gif = [], []
        self._out = Path("workspace/reports/assets") / (str(self.config.get("overlay_out", "heterotypic_regions")) + ".html")

    def _compile_region(self, core, Composite, cells, typename, backend):
        tmp = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, newline="", encoding="utf-8")
        tmp.write("id,type,x,y,size\n")
        for i, (x, y) in enumerate(cells):
            tmp.write(f"{i + 1},{typename},{x:.2f},{y:.2f},20.0\n")
        tmp.close()
        bp = Blueprint.from_dict({
            "blueprint_version": "1.0",
            "domain": {"extent_um": [self.W, self.H], "grid_resolution_um": 1, "boundary": "noflux"},
            "cell_types": [{"name": typename, "target_size": 20}],
            "initial_state": {"from_data": {"cell_table": tmp.name}},
        })
        doc = _compile(bp, backend=backend)
        inst = Composite({"state": doc["state"]}, core=core).state["mechanics"]["instance"]
        try:
            Path(tmp.name).unlink()
        except OSError:
            pass
        return inst

    def inputs(self):
        return {}

    def outputs(self):
        return {"n_epithelium": "overwrite[integer]", "n_mesenchyme": "overwrite[integer]",
                "n_cells": "overwrite[integer]", "interface_breaches": "overwrite[integer]"}

    def update(self, inputs, interval=1):
        try:
            self._epi_inst.update({"fates": {}, "field_grid": {}, "divisions": []}, 1.0)
        except Exception:
            pass
        try:
            self._mes_inst.update({"fates": {}, "field_grid": {}, "divisions": []}, 1.0)
        except Exception:
            pass
        epi = _emit(self._epi_inst)
        mes = _emit(self._mes_inst)
        # interface coupling: the epithelium's basal edge is a floor the mesenchyme can't cross.
        epi_polys = _footprints(epi)
        basal = min((min(y for _, y in p) for p in epi_polys.values()), default=self._iface)
        mpos = mes.get("positions", {}) or {}
        breaches = 0
        for cid, p in mpos.items():
            if p[1] > basal:                       # a mesenchyme cell pushed above the basal edge
                breaches += 1
                try:                               # reflect it back below (interface is reflective)
                    inst = self._mes_inst
                    agent = getattr(inst, "_agents", {}).get(cid)
                    if agent is not None:
                        ax, ay = agent["location"]
                        agent["location"] = (ax, min(ay, basal - 0.5))
                except Exception:
                    pass
        self._frames.append({"epi": epi_polys, "mes": dict(mpos), "basal": basal,
                             "mes_r": _radii(mes)})
        try:
            self._write_overlay()
        except Exception:
            pass
        return {"n_epithelium": len(epi_polys), "n_mesenchyme": len(mpos),
                "n_cells": len(epi_polys) + len(mpos), "interface_breaches": int(breaches)}

    def _write_overlay(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon as MplPoly, Circle
        from PIL import Image
        cache = self._imgs
        base = len(cache)
        for j, fr in enumerate(self._frames[base:]):
            fig, ax = plt.subplots(figsize=(6.6, 6.6), dpi=140)
            ax.set_xlim(0, self.W); ax.set_ylim(0, self.H); ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor("#fbfbfd")
            ax.axhspan(0, fr["basal"], color="#f3e9e0", alpha=0.5, zorder=0)       # mesenchyme region
            ax.axhspan(fr["basal"], self.H, color="#e6f0f3", alpha=0.5, zorder=0)  # epithelium region
            ax.axhline(fr["basal"], color="#b08968", ls="--", lw=1.4, zorder=1)
            for cid, poly in fr["epi"].items():
                ax.add_patch(MplPoly(np.asarray(poly), closed=True, facecolor="#2a9d8f",
                                     alpha=0.55, edgecolor="#1d6f66", lw=1.6, zorder=2))
            for cid, p in fr["mes"].items():
                r = fr["mes_r"].get(cid, 2.2)
                ax.add_patch(Circle((p[0], p[1]), r, facecolor="#e76f51", alpha=0.6,
                                    edgecolor="#b5452e", lw=1.2, zorder=3))
            ax.text(0.015, 0.985, f"heterotypic regions · step {base + j} · "
                    f"{len(fr['epi'])} epithelial (vertex) + {len(fr['mes'])} mesenchymal (munk)",
                    transform=ax.transAxes, va="top", ha="left", fontsize=9, color="#333",
                    bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#ddd", alpha=0.9))
            raw = io.BytesIO(); fig.savefig(raw, format="png", bbox_inches="tight", dpi=140); plt.close(fig)
            cache.append("data:image/png;base64," + base64.b64encode(raw.getvalue()).decode())
            raw.seek(0); im = Image.open(raw).convert("RGB"); w, h = im.size
            if w > 620:
                im = im.resize((620, int(h * 620 / w)), Image.LANCZOS)
            self._gif.append(im)
        frames_js = ",".join(json.dumps(u) for u in cache)
        gif_uri = ""
        if self._gif:
            gbuf = io.BytesIO()
            self._gif[0].save(gbuf, format="GIF", save_all=True, append_images=self._gif[1:],
                              duration=240, loop=0, optimize=True)
            gif_uri = "data:image/gif;base64," + base64.b64encode(gbuf.getvalue()).decode()
        self._out.write_text(f"""<!doctype html><html><head><meta charset="utf-8"><title>Heterotypic regions</title>
<style>body{{font-family:-apple-system,sans-serif;margin:16px;max-width:900px;color:#1b1b1f}}
#img{{width:min(660px,100%);border:1px solid #ddd;border-radius:6px}}
.banner{{background:#f4f6ff;border:1px solid #d6ddf5;border-radius:8px;padding:8px 12px;font-size:14px;margin-bottom:10px}}
button{{padding:5px 14px;border:1px solid #ccc;border-radius:6px;background:#f6f6f8;cursor:pointer}}</style></head>
<body><h2>Heterotypic tissue regions — two tissue types, two physics</h2>
<div class="banner"><b>Composing  vertex epithelium  ⊕  munk mesenchyme   (coupled at a reflective interface)</b></div>
<p>One tissue cross-section, two paradigms: a vertex-simulated <b>epithelium</b> (green sheet, top)
sits on a center-based <b>mesenchyme</b> (orange cells, bottom). They are coupled at their interface
(dashed line) — the epithelium's basal edge is a floor the mesenchyme presses up against but cannot
cross. Each compartment is modelled by the mechanics that fits it.</p>
<img id="img"><div style="margin-top:8px"><button id="play">Play</button>
<input id="sl" type="range" min="0" max="{max(len(cache)-1,0)}" value="0" style="width:420px;vertical-align:middle">
<a href="{gif_uri}" download="heterotypic-regions.gif" style="margin-left:14px;padding:5px 14px;border:1px solid #ccc;border-radius:6px;background:#f6f6f8;text-decoration:none;color:inherit;font-size:14px">⬇ GIF</a></div>
<script>var F=[{frames_js}],i=0,t=null,im=document.getElementById('img'),sl=document.getElementById('sl'),b=document.getElementById('play');
function show(k){{i=((k%F.length)+F.length)%F.length;im.src=F[i];sl.value=i;}}
sl.oninput=e=>{{stop();show(+e.target.value);}};function stop(){{if(t){{clearInterval(t);t=null;b.textContent='Play';}}}}
b.onclick=()=>{{if(t)stop();else{{t=setInterval(()=>show(i+1),240);b.textContent='Pause';}}}};show(0);</script>
</body></html>""", encoding="utf-8")


def _footprints(g):
    out = {}
    for cid, fp in (g.get("footprints", {}) or {}).items():
        xy = fp.get("xy")
        if xy and len(xy) >= 3:
            out[str(cid)] = [[float(x), float(y)] for x, y in xy]
    return out


def _radii(g):
    out = {}
    for cid, fp in (g.get("footprints", {}) or {}).items():
        xy = fp.get("xy")
        if xy:
            a = np.asarray(xy, float)
            cx, cy = a.mean(0)
            out[str(cid)] = float(np.mean(np.hypot(a[:, 0] - cx, a[:, 1] - cy)))
    return out


__all__ = ["HeterotypicRegionsProcess"]
