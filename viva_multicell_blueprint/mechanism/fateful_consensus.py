"""The grand-unification composite: a multi-tissue consensus with per-cell subcellular fate.

``FatefulConsensusProcess`` runs the full tri-coupled consensus tissue (CPM + vertex + munk,
with vertex owning shape, munk owning position, CPM owning area) AND, inside every consensus
cell, a :class:`~..model_format.adapters.coupled.CoupledCell` — an SBML-ODE metabolism whose
metabolite pins a MaBoSS fate network. The scales are genuinely coupled, top-down: each cell's
CPM-owned AREA sets its metabolic rate (a bigger cell metabolises faster), the metabolite C
builds and pins the fate ``high_C`` input, and the fate ``divide`` node commits the cell.

So three tissue simulators and two sub-cellular simulators run as one composite, with a live
mechanics→metabolism→fate information flow — every scale the workspace can simulate, composed.
"""
from __future__ import annotations

import base64
import io
import json

import numpy as np

from ..model_format.adapters.coupled import CoupledCell
from ..model_format.registry import ModelSpec
from .tri_coupled import TriCoupledProcess, _emit

# metabolism (A+B->C) whose metabolite C pins a MaBoSS fate input (mode A). kf starts at 0 and
# is overwritten each step from the cell's CPM area, so tissue area drives the metabolic rate.
_COUPLED_SRC = {
    "models": {
        "metab": {"formalism": "ode",
                  "source": {"species": {"A": 1.2, "B": 1.2, "C": 0.0},
                             "reactions": [{"subs": ["A", "B"], "prds": ["C"],
                                            "kf": 0.0, "kb": 0.3}]},
                  "initial": {"A": 40.0, "B": 40.0, "C": 0.0}},
        "fate": {"formalism": "boolean",
                 "source": {"high_C": "high_C", "divide": "high_C"},
                 "initial": {"high_C": 0.0, "divide": 0.0}},
    },
    "couplings": [
        {"mode": "A", "from": "metab", "species": "C", "threshold": 14.0,
         "to": "fate", "input": "high_C"},
    ],
}
_AREA_TO_RATE = 0.0006   # kf0 = _AREA_TO_RATE * CPM_area  (mechanics -> metabolism)
_C_THRESHOLD = 14.0


class FatefulConsensusProcess(TriCoupledProcess):
    """Tri-coupled consensus tissue + a per-cell metabolism→fate chain driven by cell area."""

    def __init__(self, config=None, core=None):
        super().__init__(config, core)
        self._fate: dict[str, CoupledCell] = {}
        self._fate_state: dict[str, dict] = {}
        self._fate_frames: list = []

    def _new_coupled(self) -> CoupledCell:
        return CoupledCell(ModelSpec(formalism="coupled", source=_COUPLED_SRC), {})

    def outputs(self):
        return {**super().outputs(),
                "mean_metabolite": "overwrite[float]",
                "committed_fraction": "overwrite[float]"}

    def update(self, inputs, interval=1):
        out = super().update(inputs, interval)   # runs the consensus; sets self._prev (geom)
        geom = self._prev or {}
        areas = geom.get("_cpm_area", {}) or {}
        vfoot = (geom.get("vertex", {}) or {}).get("footprints", {}) or {}
        # map each cell's vertex polygon (the consensus shape) to its CPM area + run its fate
        cvals: list[float] = []
        committed = 0
        frame = {}
        for mech_id, fp in vfoot.items():
            bid = self._cpm_to_bid.get(self._cpm_id.get(str(mech_id), -1), str(mech_id))
            area = float(areas.get(str(bid), areas.get(str(mech_id), 0.0)))
            fc = self._fate.get(bid)
            if fc is None:
                fc = self._fate[bid] = self._new_coupled()
            metab = fc.subs["metab"]
            rr = getattr(metab, "_rr", None)
            if rr is not None:                       # mechanics (area) -> metabolic rate
                try:
                    rr["kf0"] = _AREA_TO_RATE * area
                except Exception:
                    pass
            fc.step(1.0, {})
            am = fc.amounts()
            c = am.get("metab/C", 0.0)
            div = am.get("fate/divide", 0.0)
            self._fate_state[bid] = {"C": c, "divide": div, "area": area}
            cvals.append(c)
            committed += int(div > 0.5)
            xy = fp.get("xy")
            if xy and len(xy) >= 3:
                frame[bid] = {"poly": [[float(x), float(y)] for x, y in xy], "C": c, "divide": div}
        self._fate_frames.append(frame)
        try:
            self._write_fate_overlay()
        except Exception:  # pragma: no cover
            pass
        out["mean_metabolite"] = float(np.mean(cvals)) if cvals else 0.0
        out["committed_fraction"] = float(committed / len(cvals)) if cvals else 0.0
        return out

    # ---- fate-coloured consensus overlay -----------------------------------
    def _write_fate_overlay(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon as MplPoly
        from PIL import Image

        def fate_color(c, div):
            t = max(0.0, min(1.0, c / max(_C_THRESHOLD, 1e-6)))
            if div > 0.5:
                return "#d62728"
            return (0.15 + 0.80 * t, 0.55 - 0.25 * t, 0.22)

        if not hasattr(self, "_fate_imgs"):
            self._fate_imgs, self._fate_gif = [], []
        cache = self._fate_imgs
        base = len(cache)
        out = self._out.parent / "fateful_consensus.html"
        for j, frame in enumerate(self._fate_frames[base:]):
            fig, ax = plt.subplots(figsize=(6.6, 6.6), dpi=140)
            ax.set_xlim(0, self.W); ax.set_ylim(0, self.H); ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor("#fbfbfd")
            for sp in ax.spines.values():
                sp.set_edgecolor("#e3e3ea")
            for bid, cell in frame.items():
                col = fate_color(cell["C"], cell["divide"])
                poly = np.asarray(cell["poly"])
                ax.add_patch(MplPoly(poly, closed=True, facecolor=col, alpha=0.5, edgecolor=col, lw=1.6))
            ax.text(0.015, 0.985, f"fateful consensus · step {base + j} · {len(frame)} cells · "
                    "cell AREA → metabolism → fate", transform=ax.transAxes, va="top", ha="left",
                    fontsize=9.5, color="#333",
                    bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#ddd", alpha=0.9))
            raw = io.BytesIO(); fig.savefig(raw, format="png", bbox_inches="tight", dpi=140); plt.close(fig)
            cache.append("data:image/png;base64," + base64.b64encode(raw.getvalue()).decode())
            raw.seek(0)
            im = Image.open(raw).convert("RGB"); w, h = im.size
            if w > 620:
                im = im.resize((620, int(h * 620 / w)), Image.LANCZOS)
            self._fate_gif.append(im)
        frames_js = ",".join(json.dumps(u) for u in cache)
        gif_uri = ""
        if self._fate_gif:
            gbuf = io.BytesIO()
            self._fate_gif[0].save(gbuf, format="GIF", save_all=True, append_images=self._fate_gif[1:],
                                   duration=240, loop=0, optimize=True)
            gif_uri = "data:image/gif;base64," + base64.b64encode(gbuf.getvalue()).decode()
        out.write_text(f"""<!doctype html><html><head><meta charset="utf-8"><title>Fateful consensus</title>
<style>body{{font-family:-apple-system,sans-serif;margin:16px;max-width:900px;color:#1b1b1f}}
#img{{width:min(660px,100%);border:1px solid #ddd;border-radius:6px}}
button{{padding:5px 14px;border:1px solid #ccc;border-radius:6px;background:#f6f6f8;cursor:pointer}}</style></head>
<body><h2>Fateful consensus — every scale composed</h2>
<p><b>Composing:</b> CPM &oplus; vertex &oplus; munk (shape/position/area consensus tissue) &otimes; an
SBML-ODE metabolism &otimes; a MaBoSS fate network, per cell. The information flows top-down: each
cell's CPM-owned <b>area</b> sets its metabolic rate, the metabolite builds and pins the fate input,
and the fate commits the cell. Fill colour = fate state (green metabolising → amber building → red
committed).</p>
<img id="img"><div style="margin-top:8px"><button id="play">Play</button>
<input id="sl" type="range" min="0" max="{max(len(cache)-1,0)}" value="0" style="width:420px;vertical-align:middle">
<a href="{gif_uri}" download="fateful-consensus.gif" style="margin-left:14px;padding:5px 14px;border:1px solid #ccc;border-radius:6px;background:#f6f6f8;text-decoration:none;color:inherit;font-size:14px">⬇ Download GIF</a></div>
<script>var F=[{frames_js}],i=0,t=null,im=document.getElementById('img'),sl=document.getElementById('sl'),b=document.getElementById('play');
function show(k){{i=((k%F.length)+F.length)%F.length;im.src=F[i];sl.value=i;}}
sl.oninput=e=>{{stop();show(+e.target.value);}};function stop(){{if(t){{clearInterval(t);t=null;b.textContent='Play';}}}}
b.onclick=()=>{{if(t)stop();else{{t=setInterval(()=>show(i+1),240);b.textContent='Pause';}}}};show(0);</script>
</body></html>""", encoding="utf-8")


__all__ = ["FatefulConsensusProcess"]
