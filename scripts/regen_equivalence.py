#!/usr/bin/env python3
"""Reproduce the equivalence + counterfactual studies and render their figures.

Two studies, one shared method (``metrics.equivalence_tost`` + TOST):

* **crossparadigm-equivalence** — the Sego model on a CPM x vertex seed grid.
  Tests whether cross-backend agreement is statistically EQUIVALENT to the
  same-backend (seed-to-seed) baseline, on fate AND spatial metrics. Turns the
  capstone's "fail to reject a difference" into a positive equivalence claim.

* **substrate-divergence-counterfactual** — a mechanics-dominated blueprint
  (strong differential adhesion + strong chemotaxis) where the engines genuinely
  differ. The spatial equivalence test should REJECT -- the positive control
  proving the test can detect a real substrate difference.

Writes a results JSON + the figures into each study's dir. Prints the numbers to
paste into the study YAMLs.  Usage: python scripts/regen_equivalence.py [--fast]
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
from pathlib import Path

import numpy as np

from viva_multicell_blueprint.blueprint.model import Blueprint
from viva_multicell_blueprint.compare import metrics
from viva_multicell_blueprint.compare.identity import IdentityRegistry
from viva_multicell_blueprint.compile._shared import resolve_initial_state
from viva_multicell_blueprint.compare.runner import run_backend
from viva_multicell_blueprint.viz import equivalence_plot, spatial_playback, spatial_playback_multi

ROOT = Path(__file__).resolve().parents[1]
STUDIES = ROOT / "workspace" / "studies"
BP_DIR = ROOT / "workspace" / "blueprints"


def _reg(ids, a, b):
    return IdentityRegistry.from_backends(ids, a.identity, b.identity,
                                          getattr(b, "residual", {}) or {})


def _run(bp_path, backend, seed, n):
    d = copy.deepcopy(Blueprint.load(bp_path).to_dict())
    d.setdefault("mechanics_mapping", {})["rules_seed"] = seed
    return run_backend(Blueprint.from_dict(d), backend, n)


def _write(study, name, fig):
    d = STUDIES / study / "viz"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.html").write_text(fig.to_html(include_plotlyjs="cdn"), encoding="utf-8")
    print(f"  wrote {study}/viz/{name}.html")


def _results(study, payload):
    d = STUDIES / study / "results"
    d.mkdir(parents=True, exist_ok=True)
    (d / "equivalence.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"  wrote {study}/results/equivalence.json")


def _pairs_same(runs_by_backend, score):
    out = []
    for runs in runs_by_backend.values():
        seeds = sorted(runs)
        out += [score(runs[i], runs[j]) for i, j in itertools.combinations(seeds, 2)]
    return out


def _pairs_cross(a_runs, b_runs, score):
    return [score(a_runs[i], b_runs[j]) for i in sorted(a_runs) for j in sorted(b_runs)]


# backend -> human label for per-pair figure/box labels
_BLABEL = {"cpm": "CPM", "vertex": "vertex", "munk": "center-based"}
# the paradigm pairs, cpm-vertex first (its figures/keys keep the canonical names)
_PAIRS = (("cpm", "vertex"), ("cpm", "munk"), ("vertex", "munk"))


def _pair_analysis(runs_by_backend, a, b, score, margin, *, fast):
    """TOST + run-cluster bootstrap for one backend pair (a, b). ``same`` pools the
    within-backend seed pairs of BOTH a and b (the no-substrate-effect spread)."""
    ra, rb = runs_by_backend[a], runs_by_backend[b]
    sa, sb = sorted(ra), sorted(rb)
    same = _pairs_same({a: ra, b: rb}, score)
    cross = _pairs_cross(ra, rb, score)
    tost = metrics.equivalence_tost(same, cross, margin=margin)
    C = [[score(ra[i], rb[j]) for j in sb] for i in sa]
    Sa = [[score(ra[i], ra[j]) if i != j else 1.0 for j in sa] for i in sa]
    Sb = [[score(rb[i], rb[j]) if i != j else 1.0 for j in sb] for i in sb]
    cluster = metrics.run_cluster_bootstrap_ci(C, Sa, Sb, n_boot=(1000 if fast else 4000),
                                               labels=(a, b))
    return {"a": a, "b": b, "same": same, "cross": cross, "tost": tost, "cluster": cluster}


def equivalence_study(fast: bool) -> None:
    study = "crossparadigm-equivalence"
    print(f"[{study}]")
    bp = BP_DIR / "influenza_sego2022.yaml"
    n = 60 if fast else 120
    seeds = (0, 1, 2) if fast else (0, 1, 2, 3, 4, 5)
    margin_fate, margin_spatial = 0.05, 0.10
    ids = [str(c["id"]) for c in resolve_initial_state(Blueprint.load(bp))]
    runs_by_backend = {b: {s: _run(bp, b, s, n) for s in seeds} for b in ("cpm", "vertex", "munk")}

    def fate(a, b):
        return metrics.per_cell_agreement(a.frames, b.frames, _reg(ids, a, b))["mean"]

    def spatial(a, b):
        return metrics.per_cell_spatial_agreement(a.frames, b.frames, _reg(ids, a, b), tol_um=5.0)["mean"]

    out = {"study": study, "n_steps": n, "seeds": list(seeds),
           "backends": ["cpm", "vertex", "munk"], "pairs": [f"{a}-{b}" for a, b in _PAIRS]}
    for label, score, margin in (("fate", fate, margin_fate), ("spatial", spatial, margin_spatial)):
        out[label] = {"margin": margin, "pairs": {}}
        for a, b in _PAIRS:
            res = _pair_analysis(runs_by_backend, a, b, score, margin, fast=fast)
            tost, cluster = res["tost"], res["cluster"]
            key = f"{a}-{b}"
            out[label]["pairs"][key] = res
            # canonical cpm-vertex keeps the original filename + top-level keys.
            fname = f"equivalence-{label}" if (a, b) == ("cpm", "vertex") else f"equivalence-{label}-{a}-{b}"
            if (a, b) == ("cpm", "vertex"):
                out[label].update({"same": res["same"], "cross": res["cross"],
                                   "tost": tost, "cluster": cluster})
            _write(study, fname, equivalence_plot(
                res["same"], res["cross"], tost, metric=f"{label} agreement",
                title=f"Backend equivalence — {label} ({_BLABEL[a]}↔{_BLABEL[b]})",
                cluster_ci=cluster, pair_label=f"{_BLABEL[a]}↔{_BLABEL[b]}"))
            print(f"  {label} {a}↔{b}: same={tost['mean_same']:.4f} cross={tost['mean_cross']:.4f} "
                  f"diff={tost['diff']:+.4f} | CLUSTER-CI[{cluster['ci_low']:+.4f},{cluster['ci_high']:+.4f}] "
                  f"margin={margin} -> equivalent={tost['equivalent']}")
    _results(study, out)


def counterfactual_study(fast: bool) -> None:
    study = "substrate-divergence-counterfactual"
    print(f"[{study}]")
    bp = BP_DIR / "mechanics_divergence.yaml"
    n = 50 if fast else 80
    seeds = (0, 1, 2)
    margin_spatial = 0.10
    ids = [str(c["id"]) for c in resolve_initial_state(Blueprint.load(bp))]
    runs_by_backend = {b: {s: _run(bp, b, s, n) for s in seeds} for b in ("cpm", "vertex", "munk")}

    def spatial(a, b):
        return metrics.per_cell_spatial_agreement(a.frames, b.frames, _reg(ids, a, b), tol_um=5.0)["mean"]

    out = {"study": study, "n_steps": n, "seeds": list(seeds),
           "backends": ["cpm", "vertex", "munk"], "spatial": {"margin": margin_spatial, "pairs": {}}}
    for a, b in _PAIRS:
        res = _pair_analysis(runs_by_backend, a, b, spatial, margin_spatial, fast=fast)
        tost = res["tost"]
        out["spatial"]["pairs"][f"{a}-{b}"] = res
        fname = "equivalence-spatial" if (a, b) == ("cpm", "vertex") else f"equivalence-spatial-{a}-{b}"
        if (a, b) == ("cpm", "vertex"):
            out["spatial"].update({"same": res["same"], "cross": res["cross"], "tost": tost})
        _write(study, fname, equivalence_plot(
            res["same"], res["cross"], tost, metric="spatial agreement",
            title=f"Counterfactual — spatial equivalence ({_BLABEL[a]}↔{_BLABEL[b]})",
            pair_label=f"{_BLABEL[a]}↔{_BLABEL[b]}"))
        print(f"  spatial {a}↔{b}: same={tost['mean_same']:.4f} cross={tost['mean_cross']:.4f} "
              f"diff={tost['diff']:+.4f} margin={margin_spatial} -> equivalent={tost['equivalent']}")
    _write(study, "spatial-playback", spatial_playback_multi(
        {b: runs_by_backend[b][0].frames for b in ("cpm", "vertex", "munk")},
        {"extent_um": [40, 40], "grid_resolution_um": 1},
        title="Mechanics-dominated: CPM vs vertex vs center-based"))
    _results(study, out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--study", choices=["equivalence", "counterfactual"])
    args = ap.parse_args()
    if args.study in (None, "equivalence"):
        equivalence_study(args.fast)
    if args.study in (None, "counterfactual"):
        counterfactual_study(args.fast)
    print("done.")


if __name__ == "__main__":
    main()
