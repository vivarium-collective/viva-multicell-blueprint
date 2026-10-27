#!/usr/bin/env python3
"""Reproduce the `flu-lung-crossparadigm` capstone headline numbers (Task 18, §9.3).

Runs the full Sego blueprint on both backends for N steps across >=3 CPM rules-seeds
and prints the NULL-CONTROL table: cross-backend metrics judged against the
same-backend (seed-to-seed) baseline. The capstone verdict rests on the comparison
between the CROSS and the SAME-backend rows, so this script makes the reported
figures reproducible on demand (it is NOT run in CI -- N=400 is ~45 s).

    .venv/bin/python scripts/regen_capstone_metrics.py [N]     # default N=400
"""
from __future__ import annotations

import copy
import itertools
import sys
from pathlib import Path

import numpy as np

from viva_multicell_blueprint.blueprint.model import Blueprint
from viva_multicell_blueprint.compile._shared import resolve_initial_state
from viva_multicell_blueprint.compare import metrics
from viva_multicell_blueprint.compare.identity import IdentityRegistry
from viva_multicell_blueprint.compare.raster import rasterize, type_codes
from viva_multicell_blueprint.compare.runner import run_backend

SEGO = Path(__file__).resolve().parents[1] / "workspace" / "blueprints" / "influenza_sego2022.yaml"
SEEDS = (0, 1, 2)


def _run(backend, seed, n):
    d = copy.deepcopy(Blueprint.load(SEGO).to_dict())
    d.setdefault("mechanics_mapping", {})["rules_seed"] = seed
    return run_backend(Blueprint.from_dict(d), backend, n)


def _summ(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)), float(np.min(xs)), float(np.max(xs))


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    ids = [str(c["id"]) for c in resolve_initial_state(Blueprint.load(SEGO))]
    cpm = {s: _run("cpm", s, n) for s in SEEDS}
    vtx = {s: _run("vertex", s, n) for s in SEEDS}

    def agree(a, b):
        reg = IdentityRegistry.from_backends(ids, a.identity, b.identity, getattr(b, "residual", {}) or {})
        return metrics.per_cell_agreement(a.frames, b.frames, reg)["mean"]

    codes = type_codes(*[r.frames for r in list(cpm.values()) + list(vtx.values())])
    grid = {"extent_um": [50, 50], "grid_resolution_um": 1, "type_codes": codes}
    fm = lambda r: rasterize(r.frames[-1], grid)

    def iou_I(a, b):
        return metrics.iou(fm(a), fm(b), codes, ["I"])

    def cdist(a, b):
        sa = metrics.spatial_pattern(fm(a), codes).get("I")
        sb = metrics.spatial_pattern(fm(b), codes).get("I")
        if not (sa and sb):
            return None
        return float(np.hypot(sa["centroid"][0] - sb["centroid"][0], sa["centroid"][1] - sb["centroid"][1]))

    def mean_area(r):
        f = r.frames[-1]
        return float(np.mean([s for cid, s in f["sizes"].items() if f["alive"].get(cid)]))

    cpm_pairs = list(itertools.combinations(SEEDS, 2))
    cross = [(i, j) for i in SEEDS for j in SEEDS]

    print(f"=== capstone null-control table (N={n}) ===")
    print("                        agreement(mean/min/max)   iou_I(mean/min/max)   I-patch centroid dist")
    print("  same-backend CPM  : %.4f %.4f %.4f   %.3f %.3f %.3f   %.2f %.2f %.2f"
          % (*_summ([agree(cpm[i], cpm[j]) for i, j in cpm_pairs]),
             *_summ([iou_I(cpm[i], cpm[j]) for i, j in cpm_pairs]),
             *_summ([cdist(cpm[i], cpm[j]) for i, j in cpm_pairs])))
    print("  same-backend VTX  : %.4f %.4f %.4f   %.3f %.3f %.3f"
          % (*_summ([agree(vtx[i], vtx[j]) for i, j in cpm_pairs]),
             *_summ([iou_I(vtx[i], vtx[j]) for i, j in cpm_pairs])))
    print("  CROSS backend     : %.4f %.4f %.4f   %.3f %.3f %.3f   %.2f %.2f %.2f"
          % (*_summ([agree(cpm[i], vtx[j]) for i, j in cross]),
             *_summ([iou_I(cpm[i], vtx[j]) for i, j in cross]),
             *_summ([cdist(cpm[i], vtx[j]) for i, j in cross])))

    # representation control + non-circular dynamics check
    print(f"\n  final mean cell area  CPM={mean_area(cpm[0]):.2f}  VTX={mean_area(vtx[0]):.2f}  "
          f"ratio={mean_area(vtx[0]) / mean_area(cpm[0]):.2f}  (target 25)")
    tc = lambda r: np.array(metrics.type_counts(r.frames).get("I", [0] * (n + 1)), float)
    M = np.vstack([tc(cpm[s]) for s in SEEDS])
    lo, hi, vI = M.min(0), M.max(0), tc(vtx[0])
    inband = float(np.mean([lo[t] - 1e-9 <= vI[t] <= hi[t] + 1e-9 for t in range(n + 1)]))
    print(f"  vertex(seed0) I-count within CPM 3-seed envelope: {inband:.3f} of steps")


if __name__ == "__main__":
    main()
