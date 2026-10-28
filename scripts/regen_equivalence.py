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
from viva_multicell_blueprint.viz import equivalence_plot, spatial_playback

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


def _pairs_cross(cpm, vtx, score):
    return [score(cpm[i], vtx[j]) for i in sorted(cpm) for j in sorted(vtx)]


def equivalence_study(fast: bool) -> None:
    study = "crossparadigm-equivalence"
    print(f"[{study}]")
    bp = BP_DIR / "influenza_sego2022.yaml"
    n = 60 if fast else 120
    seeds = (0, 1, 2) if fast else (0, 1, 2, 3, 4, 5)
    margin_fate, margin_spatial = 0.05, 0.10
    ids = [str(c["id"]) for c in resolve_initial_state(Blueprint.load(bp))]
    cpm = {s: _run(bp, "cpm", s, n) for s in seeds}
    vtx = {s: _run(bp, "vertex", s, n) for s in seeds}

    def fate(a, b):
        return metrics.per_cell_agreement(a.frames, b.frames, _reg(ids, a, b))["mean"]

    def spatial(a, b):
        return metrics.per_cell_spatial_agreement(a.frames, b.frames, _reg(ids, a, b), tol_um=5.0)["mean"]

    sc = sorted(cpm); sv = sorted(vtx)
    out = {"study": study, "n_steps": n, "seeds": list(seeds)}
    for label, score, margin in (("fate", fate, margin_fate), ("spatial", spatial, margin_spatial)):
        same = _pairs_same({"cpm": cpm, "vertex": vtx}, score)
        cross = _pairs_cross(cpm, vtx, score)
        tost = metrics.equivalence_tost(same, cross, margin=margin)
        # Dependence-aware (run-cluster) reanalysis: the seed-pairs reuse the same
        # 6+6 runs, so resample RUNS, not pairs (reviewer guardrail).
        C = [[score(cpm[i], vtx[j]) for j in sv] for i in sc]
        Sc = [[score(cpm[i], cpm[j]) if i != j else 1.0 for j in sc] for i in sc]
        Sv = [[score(vtx[i], vtx[j]) if i != j else 1.0 for j in sv] for i in sv]
        cluster = metrics.run_cluster_bootstrap_ci(C, Sc, Sv, n_boot=4000)
        out[label] = {"same": same, "cross": cross, "tost": tost, "cluster": cluster}
        _write(study, f"equivalence-{label}", equivalence_plot(
            same, cross, tost, metric=f"{label} agreement",
            title=f"Backend equivalence — {label}", cluster_ci=cluster))
        print(f"  {label}: same={tost['mean_same']:.4f} cross={tost['mean_cross']:.4f} "
              f"diff={tost['diff']:+.4f} | pair-CI[{tost['ci_low']:+.4f},{tost['ci_high']:+.4f}] "
              f"CLUSTER-CI[{cluster['ci_low']:+.4f},{cluster['ci_high']:+.4f}] margin={margin} "
              f"-> equivalent={tost['equivalent']}")
    _results(study, out)


def counterfactual_study(fast: bool) -> None:
    study = "substrate-divergence-counterfactual"
    print(f"[{study}]")
    bp = BP_DIR / "mechanics_divergence.yaml"
    n = 50 if fast else 80
    seeds = (0, 1, 2)
    margin_spatial = 0.10
    ids = [str(c["id"]) for c in resolve_initial_state(Blueprint.load(bp))]
    cpm = {s: _run(bp, "cpm", s, n) for s in seeds}
    vtx = {s: _run(bp, "vertex", s, n) for s in seeds}

    def spatial(a, b):
        return metrics.per_cell_spatial_agreement(a.frames, b.frames, _reg(ids, a, b), tol_um=5.0)["mean"]

    same = _pairs_same({"cpm": cpm, "vertex": vtx}, spatial)
    cross = _pairs_cross(cpm, vtx, spatial)
    tost = metrics.equivalence_tost(same, cross, margin=margin_spatial)
    _write(study, "equivalence-spatial", equivalence_plot(
        same, cross, tost, metric="spatial agreement",
        title="Counterfactual — spatial equivalence (TOST)"))
    _write(study, "spatial-playback", spatial_playback(
        cpm[0].frames, vtx[0].frames, {"extent_um": [40, 40], "grid_resolution_um": 1},
        title="Mechanics-dominated: CPM vs vertex"))
    out = {"study": study, "n_steps": n, "seeds": list(seeds),
           "spatial": {"same": same, "cross": cross, "tost": tost}}
    _results(study, out)
    print(f"  spatial: same={tost['mean_same']:.4f} cross={tost['mean_cross']:.4f} "
          f"diff={tost['diff']:+.4f} margin={margin_spatial} -> equivalent={tost['equivalent']}")


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
