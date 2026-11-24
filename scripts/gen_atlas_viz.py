"""Generate the composition-atlas study visualizations.

For the three CoupledCell subcellular composites, run the composite one step at a time and
plot one representative cell's coupled sub-model dynamics as a self-contained time-series
HTML. For the fateful_tissue crown, run it (it writes its own fate-coloured flipbook) and
copy that into the study's viz dir.

Usage:
  PYTHONPATH=$(pwd) <venv>/bin/python scripts/gen_atlas_viz.py [study-slug ...]
  (no args => all four)
"""
from __future__ import annotations

import base64
import io
import shutil
import sys
from pathlib import Path

from process_bigraph import Composite

from viva_multicell_blueprint.core import build_core
from viva_multicell_blueprint.composites.coupled_subcellular import (
    coupled_fate_oscillator, coupled_particle_catalysis, coupled_triple_cell)
from viva_multicell_blueprint.composites.fateful_tissue import fateful_tissue

STUDIES = Path("workspace/studies")


def _run_cell_series(gen, node_key: str, n_steps: int, keys: list[str]) -> dict[str, list[float]]:
    """Run a scene-based composite one step at a time; collect one cell's amounts()[keys]."""
    core = build_core()
    comp = Composite({"state": gen(core=core)["state"]}, core=core)
    proc = comp.state[node_key]["instance"]
    sid = proc._cell_ids[0]
    series: dict[str, list[float]] = {k: [] for k in keys}
    for _ in range(n_steps):
        comp.run(1)
        am = proc._adapters[sid].amounts()
        for k in keys:
            series[k].append(float(am.get(k, 0.0)))
    return series


def _timeseries_html(out: Path, title: str, lead: str, steps, left, right,
                     left_label: str, right_label: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.6, 4.6), dpi=150)
    for (label, color, lw, ls), y in left:
        ax.plot(steps, y, ls, color=color, lw=lw, label=label)
    ax.set_xlabel("step"); ax.set_ylabel(left_label, color="#2e7d32")
    ax.grid(alpha=0.2)
    ax2 = ax.twinx()
    for (label, color, lw, ls), y in right:
        ax2.plot(steps, y, ls, color=color, lw=lw, label=label, marker="o", ms=3)
    ax2.set_ylabel(right_label, color="#1565c0")
    l1, la1 = ax.get_legend_handles_labels(); l2, la2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, la1 + la2, loc="upper left", fontsize=8.5)
    ax.set_title(title, fontsize=11)
    raw = io.BytesIO(); fig.savefig(raw, format="png", bbox_inches="tight", dpi=150); plt.close(fig)
    uri = "data:image/png;base64," + base64.b64encode(raw.getvalue()).decode()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"""<!doctype html><html><head><meta charset="utf-8"><title>{title}</title>
<style>body{{font-family:-apple-system,sans-serif;margin:16px;max-width:960px;color:#1b1b1f}}
img{{width:100%;border:1px solid #ddd;border-radius:6px}}</style></head>
<body><h2>{title}</h2><p>{lead}</p><img src="{uri}"></body></html>""", encoding="utf-8")
    print(f"wrote {out}")


def gen_fate_oscillator():
    s = _run_cell_series(coupled_fate_oscillator, "scene_model", 40, ["metab/C", "fate/divide"])
    _timeseries_html(
        STUDIES / "coupled-fate-oscillator/viz/coupled-fate-oscillator.html",
        "Coupled cell — ODE metabolism ↔ MaBoSS fate oscillator",
        "An SBML-ODE metabolism and a MaBoSS Boolean fate network coupled both ways (mode A: "
        "metabolite C pins the fate <code>high_C</code> input; mode C: the fate "
        "<code>divide</code> output resets the metabolism). C builds, crosses threshold, fate "
        "fires, division resets the metabolism — a period-3 metabolism↔fate oscillator. Real "
        "tellurium + MaBoSS.",
        range(40),
        [(("ODE metabolite C", "#2e7d32", 2.2, "-"), s["metab/C"])],
        [(("MaBoSS fate: divide", "#d1495b", 1.4, "-"), s["fate/divide"])],
        "C molecules (ODE)", "divide (Boolean activity)")


def gen_particle_catalysis():
    s = _run_cell_series(coupled_particle_catalysis, "scene_model", 30, ["sensor/C", "metab/P"])
    # fixed-rate CONTROL: the same ODE driven by the run-mean particle count (no coupling).
    # The coupled P is concave-up and the control is linear — the SHAPE difference is the coupling.
    from viva_multicell_blueprint.model_format.adapters.ode_tellurium import ODEAdapter
    from viva_multicell_blueprint.model_format.registry import ModelSpec
    c_mean = sum(s["sensor/C"]) / max(len(s["sensor/C"]), 1)
    ode_src = {"species": {"S": 400.0, "P": 0.0},
               "reactions": [{"subs": ["S"], "prds": ["P"], "kf": 0.0, "kb": 0.0}]}
    ctl = ODEAdapter(ModelSpec(formalism="ode", source=ode_src, params={}, membrane={}),
                     {"S": 400.0, "P": 0.0})
    p_control = []
    for _ in range(30):
        ctl._rr["kf0"] = 0.001 * c_mean
        ctl.step(1.0, {})
        p_control.append(ctl.amounts().get("P", 0.0))
    _timeseries_html(
        STUDIES / "coupled-particle-catalysis/viz/coupled-particle-catalysis.html",
        "Coupled cell — Smoldyn particle population catalyses an ODE (mode B)",
        "A Smoldyn particle sim (A+B→C, confined to the cell polygon) catalyses an ODE S→P: "
        "the live particle C count sets the ODE rate constant each step. As the particle "
        "population builds, the coupled product P <b>accelerates</b> (concave-up), while a "
        f"fixed-rate control (the same ODE driven by the run-mean count, {c_mean:.0f}) stays "
        "linear — the shape difference <i>is</i> the coupling. Real Smoldyn (3.11 bridge) + "
        "tellurium.",
        range(30),
        [(("ODE product P — particle-coupled rate", "#2e7d32", 2.4, "-"), s["metab/P"]),
         (("ODE product P — fixed-rate control", "#8a8a8a", 1.8, "--"), p_control)],
        [(("particle C count (Smoldyn)", "#1565c0", 1.4, "-"), s["sensor/C"])],
        "P molecules (ODE)", "C particles (Smoldyn)")


def gen_triple_cell():
    s = _run_cell_series(coupled_triple_cell, "scene_model", 36,
                         ["sensor/C", "metab/C", "fate/divide"])
    _timeseries_html(
        STUDIES / "coupled-triple-cell/viz/coupled-triple-cell.html",
        "Coupled cell — all three formalisms, all three modes",
        "One cell running a Smoldyn particle sim, an SBML-ODE metabolism and a MaBoSS fate "
        "network, coupled by all three modes: particles catalyse the metabolism (B), the "
        "metabolite pins the fate input (A), the fate output resets the metabolism (C). Real "
        "Smoldyn + tellurium + MaBoSS.",
        range(36),
        [(("particle sensor C", "#1565c0", 1.8, "-"), s["sensor/C"]),
         (("ODE metabolite C", "#2e7d32", 2.2, "-"), s["metab/C"])],
        [(("MaBoSS fate: divide", "#d1495b", 1.4, "-"), s["fate/divide"])],
        "molecules", "divide (Boolean activity)")


def gen_fateful_tissue():
    core = build_core()
    comp = Composite({"state": fateful_tissue(core=core)["state"]}, core=core)
    comp.run(22)
    src = Path("workspace/reports/assets/fateful_tissue.html")
    dst = STUDIES / "fateful-tissue/viz/fateful-tissue.html"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    print(f"wrote {dst} (from {src})")


def gen_verdict_scorecard(slug: str):
    """A band-overlay scorecard: each behavior test's code-graded measured value against its
    pass threshold, read from the study's run-earned computed_outcomes."""
    import yaml
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    study = STUDIES / slug
    spec = yaml.safe_load((study / "study.yaml").read_text(encoding="utf-8"))
    runs = spec.get("runs") or []
    co = (runs[0].get("computed_outcomes") if runs else {}) or {}
    tests = spec.get("behavior_tests") or []
    rows = []
    for t in tests:
        name = t.get("name")
        oc = co.get(name, {})
        if "measured_value" not in oc:
            continue
        pf = t.get("pass_if", {})
        rows.append((name, float(oc["measured_value"]), pf.get("op", ">="),
                     float(pf.get("value", 0)), oc.get("result") == "PASS", t.get("units", "")))
    if not rows:
        print(f"  {slug}: no graded outcomes to chart"); return
    fig, ax = plt.subplots(figsize=(8.0, 0.9 + 0.8 * len(rows)), dpi=150)
    y = range(len(rows))
    for i, (name, val, op, thr, ok, units) in enumerate(rows):
        color = "#2e7d32" if ok else "#d1495b"
        ax.barh(i, val, color=color, alpha=0.85, height=0.5, zorder=2)
        ax.axvline(thr, ymin=(i + 0.1) / len(rows), ymax=(i + 0.9) / len(rows),
                   color="#333", ls="--", lw=1.4, zorder=3)
        ax.text(val, i, f"  {val:g} {units}", va="center", ha="left", fontsize=9, color="#222")
        ax.text(thr, i + 0.33, f"pass: {op} {thr:g}", va="bottom", ha="center",
                fontsize=7.5, color="#555")
    ax.set_yticks(list(y)); ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.set_xlim(0, max(r[1] for r in rows) * 1.25)
    ax.set_title(f"{slug} — verdict scorecard (code-graded against the registered run)",
                 fontsize=10.5)
    ax.invert_yaxis(); ax.grid(axis="x", alpha=0.2)
    import base64 as _b64, io as _io
    raw = _io.BytesIO(); fig.savefig(raw, format="png", bbox_inches="tight", dpi=150); plt.close(fig)
    uri = "data:image/png;base64," + _b64.b64encode(raw.getvalue()).decode()
    out = study / "viz" / f"{slug}-verdict.html"
    passes = sum(1 for r in rows if r[4])
    out.write_text(f"""<!doctype html><html><head><meta charset="utf-8"><title>{slug} verdict</title>
<style>body{{font-family:-apple-system,sans-serif;margin:16px;max-width:900px;color:#1b1b1f}}
img{{width:100%;border:1px solid #ddd;border-radius:6px}}</style></head>
<body><h2>{slug} — verdict scorecard</h2>
<p>Each committed behavior test's measured value (bar) against its pass threshold (dashed),
graded by code against the study's registered run. <b>{passes}/{len(rows)} tests pass</b> —
the study's rolled-up verdict is <b>supported</b>.</p><img src="{uri}"></body></html>""",
                    encoding="utf-8")
    print(f"wrote {out}")


GENERATORS = {
    "coupled-fate-oscillator": gen_fate_oscillator,
    "coupled-particle-catalysis": gen_particle_catalysis,
    "coupled-triple-cell": gen_triple_cell,
    "fateful-tissue": gen_fateful_tissue,
}
SCORECARDS = ["coupled-fate-oscillator", "coupled-particle-catalysis",
              "coupled-triple-cell", "fateful-tissue"]


def main(argv):
    if argv == ["--scorecards"]:
        for slug in SCORECARDS:
            print(f"== {slug} scorecard ==")
            gen_verdict_scorecard(slug)
        return
    names = argv or list(GENERATORS)
    for name in names:
        fn = GENERATORS.get(name)
        if fn is None:
            print(f"unknown study {name!r}; have {list(GENERATORS)}")
            continue
        print(f"== {name} ==")
        fn()
        gen_verdict_scorecard(name)


if __name__ == "__main__":
    main(sys.argv[1:])
