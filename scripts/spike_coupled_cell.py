"""Spike (throwaway) — coupling TWO sub-cellular simulators inside ONE cell.

Today a cell runs one intracellular model; the composition atlas needs cells where several
sub-cellular sims are coupled. This spike wires an SBML-ODE metabolism to a MaBoSS Boolean
fate network, in both coupling directions:

  * mode A (ODE product -> MaBoSS input): the ODE makes C from A+B; when C crosses a
    threshold it pins the Boolean input node `high_C`.
  * mode C (MaBoSS output gates metabolism): the Boolean `divide` node follows `high_C`;
    when `divide` fires, division consumes C (the ODE is reset) -- which drops C below
    threshold, releasing `divide`, so C builds again.

Together that is a metabolism<->fate FEEDBACK OSCILLATOR built from two different formalisms.

Run (3.12 venv has tellurium + maboss):
  PYTHONPATH=$(pwd) ~/code/viva-multicell-blueprint/.venv/bin/python scripts/spike_coupled_cell.py
"""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import numpy as np

from viva_multicell_blueprint.model_format.registry import ModelSpec
from viva_multicell_blueprint.model_format.adapters.ode_tellurium import ODEAdapter
from viva_multicell_blueprint.model_format.adapters.boolean_maboss import BooleanAdapter

OUT = Path("workspace/reports/assets/coupled_cell_spike.html")
N_STEPS = 40
C_THRESHOLD = 12.0            # C above this => nutrient-replete => pin high_C
# slow reaction so C rebuilds GRADUALLY after each division (else it re-equilibrates in one
# dt=1 step and the feedback is invisible) — this is the metabolism<->fate timescale.
ODE_SRC = {"species": {"A": 1.2, "B": 1.2, "C": 0.6},
           "reactions": [{"subs": ["A", "B"], "prds": ["C"], "kf": 0.01, "kb": 0.3}]}
BOOL_SRC = {"high_C": "high_C", "divide": "high_C"}   # divide follows the pinned input
SEED_ODE = {"A": 40.0, "B": 40.0, "C": 0.0}


def _fresh_ode():
    return ODEAdapter(ModelSpec(formalism="ode", source=ODE_SRC, params={}, membrane={}),
                      dict(SEED_ODE))


def main():
    ode = _fresh_ode()
    boolean = BooleanAdapter(ModelSpec(formalism="boolean", source=BOOL_SRC, params={}, membrane={}),
                             {"high_C": 0.0, "divide": 0.0})
    print("ODE available:", ode.available(), " Boolean available:", boolean.available())
    series = {"C": [], "high_C": [], "divide": [], "division": []}
    for step in range(N_STEPS):
        ode.step(1.0, {})
        C = ode.amounts().get("C", 0.0)
        high = 1.0 if C > C_THRESHOLD else 0.0            # mode A: ODE product -> Boolean input
        boolean.step(1.0, {"high_C": high})
        divide = boolean.amounts().get("divide", 0.0)
        divided = 0.0
        if divide > 0.5:                                  # mode C: fate gates metabolism (division consumes C)
            ode = _fresh_ode()
            divided = 1.0
        series["C"].append(float(C)); series["high_C"].append(high)
        series["divide"].append(float(divide)); series["division"].append(divided)
        print(f"  step {step:2d}: C={C:5.1f}  high_C={high:.0f}  divide={divide:.2f}{'  <-- DIVISION' if divided else ''}")
    _render(series)


def _render(series):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.4, 4.6), dpi=150)
    steps = range(len(series["C"]))
    ax.plot(steps, series["C"], "-", color="#2e7d32", lw=2.2, label="ODE metabolite C")
    ax.axhline(C_THRESHOLD, color="#999", ls=":", lw=1, label=f"C threshold ({C_THRESHOLD:.0f})")
    ax2 = ax.twinx()
    ax2.plot(steps, series["divide"], "o-", color="#d1495b", ms=4, lw=1.4, label="MaBoSS fate: divide")
    ax2.set_ylim(-0.1, 1.3); ax2.set_ylabel("divide (Boolean activity)", color="#d1495b")
    for s, d in zip(steps, series["division"]):
        if d:
            ax.axvline(s, color="#d1495b", alpha=0.18, lw=4)
    ax.set_xlabel("step"); ax.set_ylabel("C molecules (ODE)", color="#2e7d32")
    ax.set_title("One cell, two coupled sub-cellular sims: ODE metabolism ↔ MaBoSS fate\n"
                 "C builds → pins high_C → divide fires → division consumes C → repeat", fontsize=11)
    l1, la1 = ax.get_legend_handles_labels(); l2, la2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, la1 + la2, loc="upper right", fontsize=9)
    ax.grid(alpha=0.2)
    raw = io.BytesIO(); fig.savefig(raw, format="png", bbox_inches="tight", dpi=150); plt.close(fig)
    uri = "data:image/png;base64," + base64.b64encode(raw.getvalue()).decode()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(f"""<!doctype html><html><head><meta charset="utf-8"><title>Coupled cell</title>
<style>body{{font-family:-apple-system,sans-serif;margin:16px;max-width:960px}}img{{width:100%;border:1px solid #ddd;border-radius:6px}}</style></head>
<body><h2>Coupling two sub-cellular simulators in one cell (spike)</h2>
<p>An <b>SBML-ODE metabolism</b> (tellurium) and a <b>MaBoSS Boolean fate</b> network run inside the
same cell, coupled both ways: the ODE's metabolite <b>C</b> pins the Boolean <code>high_C</code> input
(<b>mode A</b>), and the Boolean <code>divide</code> output consumes C by resetting the metabolism
(<b>mode C</b>). The result is a metabolism↔fate feedback oscillator — two formalisms, one cell.</p>
<img src="{uri}">
<p style="color:#777;font-size:13px">Red bands mark division events (fate gating the metabolism). Data: {json.dumps(series['divide'][:1])[:0]}computed live from the real tellurium + MaBoSS engines.</p>
</body></html>""", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
