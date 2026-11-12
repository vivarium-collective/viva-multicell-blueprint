#!/usr/bin/env python3
"""Three-way mechanics comparison on the LARGE Sego 2022 blueprint, scored with the
cross-paradigm BEHAVIOR metrics (distance-to-CPM).

Runs the scaled-up Sego blueprint (120x120, ~225-cell epithelium) on all three
substrates (CPM reference, vertex, center-based/munk) across >=3 rules-seeds, and
reports each behavior metric as a cross-backend DISTANCE FROM THE CPM next to the
same-backend (seed-to-seed) NULL floor. A small cross/null ratio means that
substrate reproduces the CPM's movement / interaction pattern; a large one means it
diverges. This is the `large-crossparadigm-comparison` study's canonical run (not in
CI -- the large tissue is slow).

    .venv/bin/python scripts/compare_large_crossparadigm.py [N] [--json out.json]
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import sys
from pathlib import Path

import numpy as np

from viva_multicell_blueprint.blueprint.model import Blueprint
from viva_multicell_blueprint.compile._shared import resolve_initial_state
from viva_multicell_blueprint.compare import metrics
from viva_multicell_blueprint.compare.identity import IdentityRegistry
from viva_multicell_blueprint.compare.raster import type_codes
from viva_multicell_blueprint.compare.runner import run_backend

SEGO_LARGE = Path(__file__).resolve().parents[1] / "workspace" / "blueprints" / "influenza_sego2022_large.yaml"
SEEDS = (0, 1, 2)
GRID = {"extent_um": [120, 120], "grid_resolution_um": 1}
BACKENDS = ("cpm", "vertex", "munk")


def _run(backend, seed, n):
    d = copy.deepcopy(Blueprint.load(SEGO_LARGE).to_dict())
    d.setdefault("mechanics_mapping", {})["rules_seed"] = seed
    return run_backend(Blueprint.from_dict(d), backend, n)


def _reg(ids, a, b):
    return IdentityRegistry.from_backends(ids, a.identity, b.identity, getattr(b, "residual", {}) or {})


def _summ(xs):
    xs = [float(x) for x in xs if x is not None and np.isfinite(x)]
    if not xs:
        return {"mean": None, "min": None, "max": None, "n": 0}
    return {"mean": float(np.mean(xs)), "min": float(np.min(xs)), "max": float(np.max(xs)), "n": len(xs)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("n", nargs="?", type=int, default=40, help="steps per run (default 40)")
    ap.add_argument("--json", type=str, default="", help="write the results dict to this path")
    args = ap.parse_args()
    n = args.n

    ids = [str(c["id"]) for c in resolve_initial_state(Blueprint.load(SEGO_LARGE))]
    runs = {b: {s: _run(b, s, n) for s in SEEDS} for b in BACKENDS}
    grid = dict(GRID, type_codes=type_codes(*[r.frames for bd in runs.values() for r in bd.values()]))

    # Each metric: a cross-backend DISTANCE from the CPM (0 = identical pattern).
    def disp(a, b):
        return metrics.per_cell_displacement_distance(a.frames, b.frames, _reg(ids, a, b))["mean"]

    def msd(a, b):
        return metrics.msd_distance(a.frames, b.frames, _reg(ids, a, b))

    def ngraph(a, b):
        return metrics.neighbour_graph_distance(a.frames, b.frames, _reg(ids, a, b))["mean"]

    def front(a, b):
        return metrics.front_speed_distance(a.frames, b.frames, grid, type_="I")

    def sort_(a, b):
        return metrics.sorting_index_distance(a.frames, b.frames)

    METRICS = {"per_cell_displacement": disp, "msd": msd,
               "neighbour_graph": ngraph, "front_speed": front, "sorting_index": sort_}

    same_pairs = list(itertools.combinations(SEEDS, 2))
    cross_pairs = [(i, j) for i in SEEDS for j in SEEDS]
    cpm = runs["cpm"]

    out = {"blueprint": SEGO_LARGE.name, "n_steps": n, "seeds": list(SEEDS),
           "n_cells_t0": len(ids), "metrics": {}}
    print(f"=== large-crossparadigm behavior-metric distances (N={n}, {len(ids)} cells, seeds {SEEDS}) ===")
    print(f"{'metric':22s}  {'CPM-self null':>22s}  {'VERTEX vs CPM':>22s}  {'MUNK vs CPM':>22s}")
    for mname, fn in METRICS.items():
        null = _summ([fn(cpm[i], cpm[j]) for i, j in same_pairs])
        vtx = _summ([fn(runs["vertex"][i], cpm[j]) for i, j in cross_pairs])
        munk = _summ([fn(runs["munk"][i], cpm[j]) for i, j in cross_pairs])
        out["metrics"][mname] = {"cpm_self_null": null, "vertex_vs_cpm": vtx, "munk_vs_cpm": munk}

        def _f(d):
            return f"{d['mean']:.3f} [{d['min']:.3f},{d['max']:.3f}]" if d["mean"] is not None else "n/a"
        print(f"{mname:22s}  {_f(null):>22s}  {_f(vtx):>22s}  {_f(munk):>22s}")

    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return out


if __name__ == "__main__":
    main()
