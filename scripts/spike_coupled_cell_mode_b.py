"""Spike (throwaway) — mode B: a PARTICLE count drives an ODE's rate, inside one cell.

The composition atlas needs all three coupling directions between sub-cellular sims. The
first spike (`spike_coupled_cell.py`) proved mode A (ODE product -> Boolean input) and
mode C (Boolean output gates metabolism). This one proves the remaining, riskiest one:

  * mode B (particle count -> ODE rate): a Smoldyn particle sim runs A+B<->C confined to
    the cell polygon; its live C particle count sets the rate constant of a SEPARATE,
    persistent ODE reaction S->P (catalysis). As the particle C population builds toward
    equilibrium, the ODE's effective rate rises, so P accumulates FASTER over time.

Why it's the risky one: the ODE adapter's `step()` ignores its `inputs` arg, so the rate
cannot be passed in through the normal port. The coupling instead reaches into the live
roadrunner the adapter holds — `ode._rr['kf0'] = base * C_count` each step — which works
because `_antimony()` emits every forward constant as a NAMED, settable parameter `kf{i}`
and the integrator keeps species state across `step()` calls. That is the mechanism the
`CoupledCell` formalisation will wrap.

Rigor: a FIXED-rate control (kf0 pinned at the run-mean C) runs the same ODE with no
coupling. The coupled P(t) is concave-up (accelerating) while the control is a straight
line — the SHAPE difference is the coupling, not a tuned offset.

Run (3.12 venv has tellurium; Smoldyn via the 3.11 bridge):
  PYTHONPATH=$(pwd) ~/code/viva-multicell-blueprint/.venv/bin/python scripts/spike_coupled_cell_mode_b.py
"""
from __future__ import annotations

import base64
import io
from pathlib import Path

from viva_multicell_blueprint.model_format.registry import ModelSpec
from viva_multicell_blueprint.model_format.adapters.ode_tellurium import ODEAdapter
from viva_multicell_blueprint.model_format.adapters.particle_smoldyn import ParticleAdapter

OUT = Path("workspace/reports/assets/coupled_cell_mode_b_spike.html")
N_STEPS = 30
DT = 0.12                   # fine frame dt so Smoldyn's association BUILDS across frames
BASE_RATE = 0.0010          # kf0 = BASE_RATE * C_count  (per-molecule catalytic contribution)

# a square cell polygon near the rate-calibration regime
CELL = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]
# particle sensor: A+B<->C. kf is deliberately SLOW so the C population builds GRADUALLY
# from 0 toward equilibrium over the run (a fast kf equilibrates in one step → a flat signal
# → nothing to drive). The gradual build is what makes the downstream ODE visibly accelerate.
PART_SRC = {"species": {"A": 1, "B": 1, "C": 0},
            "reactions": [{"subs": ["A", "B"], "prds": ["C"], "kf": 0.35, "kb": 0.015}]}
PART_SEED = {"A": 45.0, "B": 45.0, "C": 0.0}
# downstream ODE catalysis S->P; kf0 is overwritten each step from the particle C count
ODE_SRC = {"species": {"S": 400.0, "P": 0.0},
           "reactions": [{"subs": ["S"], "prds": ["P"], "kf": 0.0, "kb": 0.0}]}
ODE_SEED = {"S": 400.0, "P": 0.0}


def _fresh_ode():
    return ODEAdapter(ModelSpec(formalism="ode", source=ODE_SRC, params={}, membrane={}),
                      dict(ODE_SEED))


def main():
    particle = ParticleAdapter(
        ModelSpec(formalism="particle", source=PART_SRC, params={"polygon": CELL}, membrane={}),
        dict(PART_SEED), seed=7)
    print("Particle available:", particle.available())
    if not particle.available():
        print("  Smoldyn bridge unavailable — set SMOLDYN_PYTHON or install the 3.11 venv.")
        return

    # --- pass 1: run the particle sim, record the live C count each step ---
    c_counts = []
    for step in range(N_STEPS):
        particle.step(DT, {"polygon": CELL})
        C = particle.amounts().get("C", 0.0)
        c_counts.append(float(C))
        print(f"  step {step:2d}: particle C = {C:5.1f}")
    c_mean = sum(c_counts) / len(c_counts)

    # --- pass 2: two ODEs over the SAME C series — coupled (rate tracks C) vs control (rate fixed) ---
    ode = _fresh_ode()            # coupled: kf0 = BASE_RATE * C(t)
    ctl = _fresh_ode()            # control: kf0 = BASE_RATE * mean(C)  (no coupling)
    series = {"C": c_counts, "P_coupled": [], "P_control": [], "kf0": []}
    for C in c_counts:
        kf0 = BASE_RATE * C
        ode._rr["kf0"] = kf0                 # mode B: particle count sets the ODE rate
        ode.step(DT, {})
        ctl._rr["kf0"] = BASE_RATE * c_mean  # matched total drive, but constant
        ctl.step(DT, {})
        series["kf0"].append(kf0)
        series["P_coupled"].append(ode.amounts().get("P", 0.0))
        series["P_control"].append(ctl.amounts().get("P", 0.0))
        print(f"    C={C:5.1f}  kf0={kf0:.4f}  P_coupled={series['P_coupled'][-1]:6.2f}"
              f"  P_control={series['P_control'][-1]:6.2f}")
    _render(series, c_mean)


def _render(series, c_mean):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    steps = range(len(series["C"]))
    fig, ax = plt.subplots(figsize=(8.6, 4.8), dpi=150)
    ax.plot(steps, series["P_coupled"], "-", color="#2e7d32", lw=2.4,
            label="ODE product P — rate driven by particle C (mode B)")
    ax.plot(steps, series["P_control"], "--", color="#8a8a8a", lw=1.8,
            label=f"ODE product P — control, rate fixed at mean C ({c_mean:.1f})")
    ax.set_xlabel("step"); ax.set_ylabel("P molecules (ODE)", color="#2e7d32")
    ax2 = ax.twinx()
    ax2.plot(steps, series["C"], "o-", color="#1565c0", ms=4, lw=1.4,
             label="particle C count (Smoldyn)")
    ax2.set_ylabel("C particles (Smoldyn)", color="#1565c0")
    ax2.set_ylim(bottom=0)
    ax.set_title("One cell, two coupled sub-cellular sims — mode B: particles → ODE rate\n"
                 "Smoldyn A+B→C count catalyses the ODE S→P; P accelerates as C builds, "
                 "the control stays linear", fontsize=10.5)
    l1, la1 = ax.get_legend_handles_labels(); l2, la2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, la1 + la2, loc="upper left", fontsize=8.5)
    ax.grid(alpha=0.2)
    raw = io.BytesIO(); fig.savefig(raw, format="png", bbox_inches="tight", dpi=150); plt.close(fig)
    uri = "data:image/png;base64," + base64.b64encode(raw.getvalue()).decode()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(f"""<!doctype html><html><head><meta charset="utf-8"><title>Coupled cell — mode B</title>
<style>body{{font-family:-apple-system,sans-serif;margin:16px;max-width:980px}}img{{width:100%;border:1px solid #ddd;border-radius:6px}}</style></head>
<body><h2>Mode B: a particle count drives an ODE's rate (spike)</h2>
<p>A <b>Smoldyn particle sim</b> (A+B&rarr;C, confined to the cell polygon) and a
<b>tellurium ODE</b> (S&rarr;P) run inside one cell. Each step the live <b>C particle count</b>
sets the ODE rate constant <code>kf0</code> (<b>mode B</b> — the particle population catalyses the
metabolism). As C builds toward equilibrium the ODE accelerates, so <b>P</b> is concave-up; a
<b>control</b> ODE driven by the run-mean C stays linear. The shape difference <i>is</i> the
coupling. Both engines are real — Smoldyn through the 3.11 bridge, tellurium in-process.</p>
<img src="{uri}">
<p style="color:#777;font-size:13px">Mechanism: the ODE adapter's <code>step()</code> ignores its
inputs, so the rate is injected via the live roadrunner — <code>ode._rr['kf0'] = base * C</code> —
the seam the <code>CoupledCell</code> formalisation will wrap.</p>
</body></html>""", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
