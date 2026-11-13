#!/usr/bin/env python3
"""Render each study's interactive figures into ``workspace/studies/<slug>/viz/``.

The vivarium-workbench report inliner (`_embed_html_figures`) picks up every
``<study_dir>/viz/*.html`` and inlines it into the investigation report the
workbench serves, sharing one Plotly.js. So this script is the single source of
the per-study figures: it runs each study's comparison with the SAME computation
its behavior test uses, renders the figures with ``include_plotlyjs="cdn"`` (the
inliner strips the per-figure loader), and writes them next to the study.

Run from the repo root (or any dir):  python scripts/render_study_figures.py
Options:  --study <slug>  (render one study)   --fast (smaller N, for a quick look)
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
from viva_multicell_blueprint.viz import (
    adhesion_sweep, divergence, dose_response, dynamics_overlay, mechanics_calibration,
    single_playback, spatial_playback, spatial_playback_multi, t0_fidelity,
)
from viva_multicell_blueprint.viz._common import HTML_CAP

ROOT = Path(__file__).resolve().parents[1]
STUDIES = ROOT / "workspace" / "studies"
SEGO = ROOT / "workspace" / "blueprints" / "influenza_sego2022.yaml"
SEGO_LARGE = ROOT / "workspace" / "blueprints" / "influenza_sego2022_large.yaml"


def _write(study_slug: str, name: str, fig) -> None:
    out_dir = STUDIES / study_slug / "viz"
    out_dir.mkdir(parents=True, exist_ok=True)
    html = fig.to_html(include_plotlyjs="cdn")
    size = len(html.encode())
    if size >= HTML_CAP:
        raise SystemExit(
            f"figure {study_slug}/{name} is {size} bytes >= inline cap {HTML_CAP}; "
            "decimate harder or reduce frames")
    (out_dir / f"{name}.html").write_text(html, encoding="utf-8")
    print(f"  wrote {study_slug}/viz/{name}.html  ({size // 1024} KB)")


# --- flu-lung-crossparadigm: the cross-paradigm trio (capstone) ---------------
def render_flu_lung(fast: bool) -> None:
    n = 60 if fast else 120
    seeds = (0, 1, 2)
    grid = {"extent_um": [50, 50], "grid_resolution_um": 1}

    def run(backend, seed):
        d = copy.deepcopy(Blueprint.load(SEGO).to_dict())
        d.setdefault("mechanics_mapping", {})["rules_seed"] = seed
        return run_backend(Blueprint.from_dict(d), backend, n)

    ids = [str(c["id"]) for c in resolve_initial_state(Blueprint.load(SEGO))]
    cpm = {s: run("cpm", s) for s in seeds}
    vtx = run("vertex", 0)
    munk = run("munk", 0)
    # one representative seed-0 run per paradigm for the side-by-side trio
    runs0 = {"cpm": cpm[0], "vertex": vtx, "munk": munk}
    reg = IdentityRegistry.from_maps(
        ids,
        {b: r.identity for b, r in runs0.items()},
        {b: getattr(r, "residual", {}) or {} for b, r in runs0.items()})
    # CPM I-count stochastic envelope as the acceptance band.
    tc = lambda r: np.array(metrics.type_counts(r.frames).get("I", [0] * (n + 1)), float)
    M = np.vstack([tc(cpm[s]) for s in seeds])
    lo, hi, mu = M.min(0), M.max(0), M.mean(0)
    band = {"I cells": [(float(t), float(mu[t]), float(lo[t]), float(hi[t])) for t in range(n + 1)]}

    _write("flu-lung-crossparadigm", "spatial-playback",
           spatial_playback_multi({b: r.frames for b, r in runs0.items()}, grid))
    _write("flu-lung-crossparadigm", "dynamics-overlay",
           dynamics_overlay({b: r.frames for b, r in runs0.items()}, band))
    # pairwise substrate-divergence maps (cpm-vertex kept as the canonical name)
    _write("flu-lung-crossparadigm", "divergence",
           divergence(cpm[0].frames, vtx.frames, reg, grid))
    for a, b in itertools.combinations(list(runs0), 2):
        if (a, b) == ("cpm", "vertex"):
            continue
        _write("flu-lung-crossparadigm", f"divergence-{a}-{b}",
               divergence(runs0[a].frames, runs0[b].frames, reg, grid))


# --- adhesion-calibration: J↔tension sweep + neutral-sheet playback -----------
_MECH = {
    "domain": {"extent_um": [30, 30], "grid_resolution_um": 1, "boundary": "noflux"},
    "cell_types": [{"name": "H", "target_size": 25}],
    "adhesion": [{"a": "H", "b": "H", "strength": 5}, {"a": "H", "b": "medium", "strength": 25}],
    "initial_state": {"kind": "blocks",
                      "block": {"type": "H", "origin": [6, 6], "shape": [4, 4], "spacing": 5, "size": 25}},
}
_SCALES = (0.5, 1.0, 1.683, 2.0, 4.0, 8.0)
CALIBRATION = STUDIES / "adhesion-calibration" / "results" / "vertex_mechanics_calibration.json"


def _area_stats(run):
    f = run.frames[-1]
    a = np.array([s for cid, s in f["sizes"].items() if f["alive"].get(cid)], float)
    if not len(a):            # the sheet collapsed: no cell left
        return 0.0, float("nan")
    return a.mean(), (a.std() / a.mean() if a.mean() else float("nan"))


def render_adhesion(fast: bool) -> None:
    n = 30 if fast else 60
    cpm = run_backend(Blueprint.from_dict(copy.deepcopy(_MECH)), "cpm", n)
    c_mean, c_cv = _area_stats(cpm)
    v_mean, v_cv = [], []
    for scale in _SCALES:
        d = copy.deepcopy(_MECH)
        d["mechanics_mapping"] = {"tension_scale": scale}
        v = run_backend(Blueprint.from_dict(d), "vertex", n)
        m, cv = _area_stats(v)
        v_mean.append(m)
        v_cv.append(cv)
    _write("adhesion-calibration", "adhesion-sweep",
           adhesion_sweep(_SCALES, v_mean, v_cv, c_mean, c_cv, target_size=25))
    # playback at the calibrated default tension scale (no override); add the
    # center-based sheet as a third panel (munk calibration itself is future work).
    vtx_ref = run_backend(Blueprint.from_dict(copy.deepcopy(_MECH)), "vertex", n)
    munk_ref = run_backend(Blueprint.from_dict(copy.deepcopy(_MECH)), "munk", n)
    grid = {"extent_um": [30, 30], "grid_resolution_um": 1}
    _write("adhesion-calibration", "neutral-sheet-playback",
           spatial_playback_multi({"cpm": cpm.frames, "vertex": vtx_ref.frames,
                                   "munk": munk_ref.frames}, grid))
    if CALIBRATION.exists():   # produced by scripts/calibrate_vertex_mechanics.py
        _write("adhesion-calibration", "mechanics-calibration",
               mechanics_calibration(json.loads(CALIBRATION.read_text())))


# --- dose-response-crossparadigm: initial-load → final infection --------------
def render_dose(fast: bool) -> None:
    n = 60 if fast else 100
    fractions = (0.03, 0.08, 0.15)
    seeds = (0, 1)
    doc = Blueprint.load(SEGO).to_dict()

    def final_I(frac, backend, seed):
        d = copy.deepcopy(doc)
        d["initial_state"]["infected_seed"]["fraction"] = frac
        d.setdefault("mechanics_mapping", {})["rules_seed"] = seed
        r = run_backend(Blueprint.from_dict(d), backend, n)
        tc = metrics.type_counts(r.frames)
        return int(np.array(tc.get("I", [0]))[-1]) if "I" in tc else 0

    series = []
    for backend in ("cpm", "vertex", "munk"):
        for seed in seeds:
            series.append({"backend": backend, "seed": seed,
                           "final_I": [final_I(f, backend, seed) for f in fractions]})
    _write("dose-response-crossparadigm", "dose-response", dose_response(fractions, series))


# --- initial-state-fidelity: t0 centroid residuals + t0 side-by-side ----------
def render_fidelity(fast: bool) -> None:
    bp = Blueprint.load(SEGO)
    by = {str(c["id"]): c for c in resolve_initial_state(bp)}
    residuals, type_ok, runs = {}, {}, {}
    for backend in ("cpm", "vertex", "munk"):
        run = run_backend(Blueprint.load(SEGO), backend, 0)   # t0 frame only
        runs[backend] = run
        f = run.frames[0]
        res, ok = [], 0
        for cid, c in by.items():
            if not f["alive"].get(cid):
                continue
            p = f["positions"][cid]
            res.append(float(np.hypot(p[0] - c["centroid"][0], p[1] - c["centroid"][1])))
            ok += int(str(f["types"][cid]) == str(c["type"]))
        residuals[backend] = res
        type_ok[backend] = ok
    _write("initial-state-fidelity", "t0-residuals",
           t0_fidelity(residuals, type_ok=type_ok, n_cells=len(by)))
    grid = {"extent_um": [50, 50], "grid_resolution_um": 1}
    _write("initial-state-fidelity", "t0-side-by-side",
           spatial_playback_multi({b: runs[b].frames for b in runs}, grid))


# --- large-crossparadigm-comparison: CPM reference spatial-playback video -----
def render_large(fast: bool) -> None:
    """Detailed single-panel CPM playback on the LARGE Sego blueprint (120x120,
    ~225-cell epithelium) — the reference-substrate video for the three-way
    behavior-metric comparison. CPM only (vertex/munk distances are quantified
    numerically in the study's readouts), full horizon, full 120x120 frame."""
    n = 40 if fast else 80          # the large composite's default_n_steps
    d = copy.deepcopy(Blueprint.load(SEGO_LARGE).to_dict())
    d.setdefault("mechanics_mapping", {})["rules_seed"] = 0
    cpm = run_backend(Blueprint.from_dict(d), "cpm", n)
    grid = {"extent_um": [120, 120], "grid_resolution_um": 1}
    _write("large-crossparadigm-comparison", "spatial-playback-cpm-large",
           single_playback(cpm.frames, grid, backend="cpm",
                           title="Large Sego 2022 — CPM reference playback "
                                 "(120x120, ~225-cell epithelium)"))


_RENDERERS = {
    "flu-lung-crossparadigm": render_flu_lung,
    "adhesion-calibration": render_adhesion,
    "dose-response-crossparadigm": render_dose,
    "initial-state-fidelity": render_fidelity,
    "large-crossparadigm-comparison": render_large,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--study", choices=sorted(_RENDERERS), help="render just one study")
    ap.add_argument("--fast", action="store_true", help="smaller N for a quick look")
    args = ap.parse_args()
    todo = [args.study] if args.study else list(_RENDERERS)
    for slug in todo:
        print(f"[{slug}]")
        _RENDERERS[slug](args.fast)
    print("done.")


if __name__ == "__main__":
    main()
