"""#8 — the full-stack crown: a dividing tissue whose divisions are decided by biology.

``FatefulTissueProcess`` composes every scale the workspace can simulate into one system:

  * a **tyssue (vertex)** tissue of growing, dividing cells (the mechanics);
  * inside each cell, a **Smoldyn particle** sim confined to the moving polygon, split into
    the two daughters on division (reused wholesale from :class:`BoundedParticlesProcess`);
  * inside each cell, a **CoupledCell** — an SBML-ODE metabolism whose metabolite ``C`` pins
    a **MaBoSS** fate network's ``high_C`` input (mode A);
  * and the keystone: **fate drives division.** A cell divides when its own fate network
    raises ``divide`` — the metabolism builds C, C pins the fate input, fate decides to
    divide, and the *tissue* carries out that division (splitting the particles and giving
    each daughter a fresh metabolism/fate). Biology decides when the tissue divides, not a
    mechanical clock.

Three formalisms (ODE + particle + Boolean) and three scales (molecule, cell, tissue) in
one running composite. All engines are real (tellurium / MaBoSS in-process, Smoldyn via the
in-process wheel or the 3.11 bridge).
"""
from __future__ import annotations

import base64
import io
import json

import numpy as np
from matplotlib.path import Path as MplPath

from ..model_format.adapters.coupled import CoupledCell
from ..model_format.registry import ModelSpec
from .bounded_particles import BoundedParticlesProcess, _rand_in_poly

# metabolism (A+B->C) whose metabolite C pins the fate input; equilibrium C ~17 > threshold,
# built over a few steps so divisions are paced rather than simultaneous.
_COUPLED_SRC = {
    "models": {
        "metab": {"formalism": "ode",
                  "source": {"species": {"A": 1.2, "B": 1.2, "C": 0.0},
                             "reactions": [{"subs": ["A", "B"], "prds": ["C"],
                                            "kf": 0.01, "kb": 0.3}]},
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


class FatefulTissueProcess(BoundedParticlesProcess):
    """Dividing tyssue + per-cell confined particles + per-cell ODE→MaBoSS fate that
    drives division. Subclasses :class:`BoundedParticlesProcess` to reuse the mechanics
    backend, the confinement and the division particle-split."""

    config_schema = {
        **BoundedParticlesProcess.config_schema,
        "c_threshold": {"_type": "float", "_default": 14.0},
        "max_divisions_per_step": {"_type": "integer", "_default": 1},
    }

    def __init__(self, config=None, core=None):
        super().__init__(config, core)
        self._threshold = float(self.config.get("c_threshold", 14.0))
        self._max_div = int(self.config.get("max_divisions_per_step", 1))
        self._base_area = float(self.config.get("base_area", 24.0))
        self._growth_gain = float(self.config.get("growth_gain", 1.1))
        self._fate: dict[str, CoupledCell] = {}
        self._fate_state: dict[str, dict] = {}
        self._divisions_total = 0
        for cid in list(self._particles):
            self._fate[cid] = self._new_coupled()

    def _new_coupled(self) -> CoupledCell:
        src = dict(_COUPLED_SRC)
        src["couplings"][0]["threshold"] = self._threshold
        return CoupledCell(ModelSpec(formalism="coupled", source=src), {})

    def outputs(self):
        return {"n_cells": "overwrite[integer]", "n_particles": "overwrite[integer]",
                "n_divisions": "overwrite[integer]", "mean_metabolite": "overwrite[float]",
                "smoldyn_mode": "overwrite[string]"}

    def update(self, inputs, interval=1):
        g0 = self._emit()
        foot0 = self._footprints(g0)
        live = sorted(foot0)

        # 1. step each live cell's coupled metabolism+fate; collect cells whose fate divides
        #    and set each cell's target area from its metabolite C, so a cell GROWS as it
        #    metabolises (and a just-divided daughter, with C reset, regrows from base).
        dividers: list[str] = []
        target_size: dict[str, float] = {}
        for cid in live:
            fc = self._fate.get(cid)
            if fc is None:
                fc = self._fate[cid] = self._new_coupled()
            fc.step(1.0, {"polygon": foot0.get(cid)})
            am = fc.amounts()
            c_level = am.get("metab/C", 0.0)
            self._fate_state[cid] = {"C": c_level, "divide": am.get("fate/divide", 0.0)}
            target_size[cid] = self._base_area + self._growth_gain * c_level
            if am.get("fate/divide", 0.0) > 0.5:
                dividers.append(cid)

        # 2. fate-DRIVEN division: the tissue divides the cells biology chose. Shuffle so no
        #    single lineage is always picked, and cap per step for tyssue stability.
        self._rng.shuffle(dividers)
        inp = {"fates": {}, "field_grid": {}, "target_size": target_size,
               "divisions": dividers[: self._max_div]}
        try:
            self._mech.update(inp, 1.0)
        except Exception:
            pass
        g = self._emit()
        foot = self._footprints(g)
        lineage = {str(k): str(v) for k, v in (g.get("lineage", {}) or {}).items()}

        # 3. division: split the mother's particles + give mother and daughter fresh
        #    metabolism/fate (division consumed C, so both reset)
        for daughter, mother in lineage.items():
            if daughter in self._particles or daughter not in foot:
                continue
            dpath = MplPath(np.asarray(foot[daughter]))
            keep, give = [], []
            for p in self._particles.get(mother, []):
                (give if dpath.contains_point([p[1], p[2]]) else keep).append(p)
            self._particles[mother] = keep
            self._particles[daughter] = give
            self._fate[daughter] = self._new_coupled()
            self._fate[mother] = self._new_coupled()
            self._divisions_total += 1

        # 4. seed particles/fate for any brand-new cell
        for cid, poly in foot.items():
            if cid not in self._particles:
                self._particles[cid] = _rand_in_poly(poly, self._seed_n, self._rng)
            if cid not in self._fate:
                self._fate[cid] = self._new_coupled()

        # 5. advance the confined particles (reused from the bounded-particles coupling)
        cells = {cid: {"poly": foot[cid], "parts": self._particles.get(cid, [])} for cid in foot}
        if self._runner.available() and cells:
            try:
                advanced = self._runner.step(cells, self._step)
                for cid, parts in advanced.items():
                    self._particles[cid] = parts
            except Exception as e:  # pragma: no cover - engine-dependent
                print(f"  [smoldyn] step {self._step}: {e}")

        # 6. snapshot (polygon + particles + fate state) for the fate-aware overlay
        snapshot = {cid: {"poly": foot[cid], "parts": self._particles.get(cid, []),
                          **self._fate_state.get(cid, {})} for cid in foot}
        self._frames.append(snapshot)
        self._step += 1
        try:
            self._write_fate_overlay()
        except Exception as e:  # pragma: no cover
            print(f"  [overlay] {e}")

        npart = sum(len(s["parts"]) for s in snapshot.values())
        cvals = [s.get("C", 0.0) for s in snapshot.values()]
        return {"n_cells": len(foot), "n_particles": npart,
                "n_divisions": self._divisions_total,
                "mean_metabolite": float(np.mean(cvals)) if cvals else 0.0,
                "smoldyn_mode": self._runner.mode}

    # ---- fate-aware overlay: cell fill reads metabolite→fate, red flash on readiness ----
    def _write_fate_overlay(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon as MplPoly
        from PIL import Image

        def fate_color(c, div):
            # green (low C) -> amber -> red (fate ready to divide)
            t = max(0.0, min(1.0, c / max(self._threshold, 1e-6)))
            if div > 0.5:
                return "#d62728"  # dividing now
            r = 0.15 + 0.80 * t
            g = 0.55 - 0.25 * t
            return (r, g, 0.22)

        cache, gcache = self._frame_imgs, self._gif_frames
        base = len(cache)
        for j, snapshot in enumerate(self._frames[base:]):
            step_idx = base + j
            fig, ax = plt.subplots(figsize=(6.6, 6.6), dpi=150)
            ax.set_xlim(0, self.W); ax.set_ylim(0, self.H); ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor("#fbfbfd")
            for sp in ax.spines.values():
                sp.set_edgecolor("#e3e3ea")
            npart = 0
            for cid, snap in snapshot.items():
                col = fate_color(snap.get("C", 0.0), snap.get("divide", 0.0))
                poly = np.asarray(snap["poly"])
                ax.add_patch(MplPoly(poly, closed=True, facecolor=col, alpha=0.22,
                                     edgecolor="none", zorder=0))
                ax.add_patch(MplPoly(poly, closed=True, fill=False, edgecolor=col,
                                     lw=2.1, zorder=1))
                if snap["parts"]:
                    P = np.array([[p[1], p[2]] for p in snap["parts"]])
                    ax.scatter(P[:, 0], P[:, 1], s=13, c="#2b2b2b", zorder=2,
                               edgecolors="white", linewidths=0.3, alpha=0.75)
                    npart += len(snap["parts"])
            ax.text(0.015, 0.985,
                    f"fateful tissue · step {step_idx} · {len(snapshot)} cells · "
                    f"{npart} molecules · {self._divisions_total} divisions",
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
                           duration=240, loop=0, optimize=True)
            gif_uri = "data:image/gif;base64," + base64.b64encode(gbuf.getvalue()).decode()
        self._out.parent.mkdir(parents=True, exist_ok=True)
        self._out.write_text(f"""<!doctype html><html><head><meta charset="utf-8"><title>Fateful tissue</title>
<style>body{{font-family:-apple-system,sans-serif;margin:16px;background:#fff;color:#1b1b1f;max-width:900px}}
#img{{width:min(660px,100%);height:auto;border:1px solid #ddd;border-radius:6px}}
button{{padding:5px 14px;border:1px solid #ccc;border-radius:6px;background:#f6f6f8;cursor:pointer}}
.k{{display:inline-block;margin-right:14px;font-size:13px}}.d{{display:inline-block;width:11px;height:11px;border-radius:3px;vertical-align:middle;margin-right:4px}}</style></head>
<body><h2>Fateful tissue — biology decides when the tissue divides</h2>
<p>Every scale at once: a <b>tyssue</b> of dividing cells; inside each, a <b>Smoldyn</b> particle
sim (dark dots, split on division); inside each, an <b>ODE metabolism</b> whose metabolite C pins
a <b>MaBoSS</b> fate network. A cell divides when <i>its own fate network</i> raises
<code>divide</code> — so the tissue's divisions are decided by the metabolism→fate chain, not a
clock. Cell fill reads the fate state:
<span class="k"><span class="d" style="background:#2f8c38"></span>metabolising</span>
<span class="k"><span class="d" style="background:#c79a1e"></span>C building</span>
<span class="k"><span class="d" style="background:#d62728"></span>dividing</span></p>
<img id="img"><div style="margin-top:8px"><button id="play">Play</button>
<input id="sl" type="range" min="0" max="{max(len(cache)-1,0)}" value="0" style="width:420px;vertical-align:middle">
<a href="{gif_uri}" download="fateful-tissue.gif" style="margin-left:14px;padding:5px 14px;border:1px solid #ccc;border-radius:6px;background:#f6f6f8;text-decoration:none;color:inherit;font-size:14px">⬇ Download GIF</a></div>
<script>var F=[{frames_js}],i=0,t=null,im=document.getElementById('img'),sl=document.getElementById('sl'),b=document.getElementById('play');
function show(k){{i=((k%F.length)+F.length)%F.length;im.src=F[i];sl.value=i;}}
sl.oninput=e=>{{stop();show(+e.target.value);}};function stop(){{if(t){{clearInterval(t);t=null;b.textContent='Play';}}}}
b.onclick=()=>{{if(t)stop();else{{t=setInterval(()=>show(i+1),240);b.textContent='Pause';}}}};show(0);</script>
</body></html>""", encoding="utf-8")


__all__ = ["FatefulTissueProcess"]
