"""Cell-coloured flipbook of the mixed-formalism scene_model demo."""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import numpy as np

_COLORS = {"ODEAdapter": ("#1f77b4", "ODE (SBML/Tellurium)"),
           "ParticleAdapter": ("#d62728", "Particle (Smoldyn)"),
           "BooleanAdapter": ("#2ca02c", "Boolean (MaBoSS)")}
STUDY_VIZ = (Path(__file__).resolve().parents[2] / "workspace" / "studies"
             / "scene-model-mixed-formalism" / "viz" / "scene-model-flipbook.html")
MAX_BYTES = 2_500_000


def capture_demo(steps: int = 12) -> list[dict]:
    """Run the demo and snapshot each step (including step 0)."""
    from process_bigraph import Composite

    from ..composites.scene_model import scene_model
    from ..core import build_core, register_workspace_processes

    core = build_core()
    register_workspace_processes(core)
    comp = Composite(scene_model(core=core), core=core)
    proc = comp.state["scene_model"]["instance"]

    def snap():
        fp, _ = proc._mechanics_emit()
        cells = {}
        for cid, ad in proc._adapters.items():
            parts = [[p[1], p[2]] for p in getattr(ad, "_parts", []) if len(p) >= 3]
            cells[cid] = {"poly": fp[cid], "kind": type(ad).__name__,
                          "amounts": dict(proc._last_amounts.get(cid, {})), "parts": parts}
        f = proc._field
        field = {"grids": {k: np.array(v) for k, v in f.grids.items()},
                 "extent": f.extent} if f is not None else None
        return {"cells": cells, "field": field}

    frames = [snap()]
    for _ in range(steps):
        comp.run(1.0)
        frames.append(snap())
    return frames


def render_flipbook(frames: list[dict], steps_label: str = "") -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPoly

    allp = np.array([p for fr in frames for c in fr["cells"].values() for p in c["poly"]])
    x0, y0 = allp.min(0) - 6
    x1, y1 = allp.max(0) + 6
    uris = []
    for i, fr in enumerate(frames):
        fig, ax = plt.subplots(figsize=(5.6, 5.6), dpi=80)
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        fld = fr["field"]
        if fld and fld["grids"]:
            g = sum(fld["grids"].values())
            w, h = fld["extent"]
            ax.imshow(g, origin="lower", extent=(0, w, 0, h), cmap="Greys", alpha=0.35,
                      vmin=0, vmax=max(float(g.max()), 1e-9) if i == 0 else None,
                      interpolation="bilinear", zorder=0)
            ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
        npart = 0
        for cid, c in fr["cells"].items():
            col = _COLORS.get(c["kind"], ("#888", c["kind"]))[0]
            poly = np.asarray(c["poly"])
            ax.add_patch(MplPoly(poly, closed=True, facecolor=col, alpha=0.3, edgecolor=col,
                                 lw=2, zorder=1))
            ctr = poly.mean(0)
            ax.text(ctr[0], ctr[1], cid, ha="center", va="center", fontsize=7, zorder=4)
            if c["parts"]:
                P = np.array(c["parts"])
                ax.scatter(P[:, 0], P[:, 1], s=9, c=col, edgecolors="white",
                           linewidths=0.3, zorder=3)
                npart += len(P)
        ax.text(0.02, 0.98, f"step {i} · {len(fr['cells'])} cells · {npart} molecules",
                transform=ax.transAxes, va="top", fontsize=8,
                bbox=dict(boxstyle="round", fc="white", ec="#ddd", alpha=0.9))
        raw = io.BytesIO()
        fig.savefig(raw, format="png", bbox_inches="tight")
        plt.close(fig)
        uris.append("data:image/png;base64," + base64.b64encode(raw.getvalue()).decode())
    legend = "".join(f'<span class="k"><span class="d" style="background:{c}"></span>{n}</span>'
                     for c, n in _COLORS.values())
    frames_js = ",".join(json.dumps(u) for u in uris)
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Scene model flipbook</title>
<style>body{{font-family:-apple-system,sans-serif;margin:16px;background:#fff;color:#1b1b1f}}
#img{{width:min(560px,100%);height:auto;border:1px solid #ddd;border-radius:6px}}
.k{{display:inline-block;margin-right:14px;font-size:13px}}.d{{display:inline-block;width:11px;height:11px;border-radius:50%;vertical-align:middle;margin-right:4px}}
button{{padding:5px 14px;border:1px solid #ccc;border-radius:6px;background:#f6f6f8;cursor:pointer}}</style></head>
<body><h2>Mixed-formalism scene: ODE, particle and Boolean cells sharing one field</h2>
<p>Cells are coloured by the formalism of their model. The grey haze is the shared extracellular field, shown as plumbing: it is wired and validated, but live efflux across the membrane is future work (adapters report no crossing yet). {legend} {steps_label}</p>
<img id="img"><div style="margin-top:8px"><button id="play">Play</button>
<input id="sl" type="range" min="0" max="{max(len(uris) - 1, 0)}" value="0" style="width:420px;vertical-align:middle"></div>
<script>var F=[{frames_js}],i=0,t=null,im=document.getElementById('img'),sl=document.getElementById('sl'),b=document.getElementById('play');
function show(k){{i=((k%F.length)+F.length)%F.length;im.src=F[i];sl.value=i;}}
sl.oninput=e=>{{stop();show(+e.target.value);}};function stop(){{if(t){{clearInterval(t);t=null;b.textContent='Play';}}}}
b.onclick=()=>{{if(t)stop();else{{t=setInterval(()=>show(i+1),250);b.textContent='Pause';}}}};show(0);</script>
</body></html>"""


def render_demo(steps: int = 12, out: Path | str | None = STUDY_VIZ) -> str:
    html = render_flipbook(capture_demo(steps))
    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html, encoding="utf-8")
    return html


if __name__ == "__main__":
    h = render_demo()
    print(f"wrote {STUDY_VIZ} ({len(h)} bytes)")
