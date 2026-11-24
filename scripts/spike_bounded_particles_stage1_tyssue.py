"""Spike stage 1 — dump a tyssue (vertex) trajectory with a division.

Runs the vertex backend on a tiny blueprint, manually triggers one cell division
partway through, and writes per-step cell POLYGONS + lineage to JSON. Stage 2 reads
that trajectory and confines Smoldyn particles inside each polygon (two-stage so the
Smoldyn part can run in viva-smoldyn's Python 3.11 venv, independent of this 3.12 one).

Run (3.12 multicell venv):
  PYTHONPATH=$(pwd) ~/code/viva-multicell-blueprint/.venv/bin/python \
    scripts/spike_bounded_particles_stage1_tyssue.py
"""
from __future__ import annotations

import json
from pathlib import Path

from process_bigraph import Composite

from viva_multicell_blueprint.blueprint.model import Blueprint
from viva_multicell_blueprint.compile import compile as _compile
from viva_multicell_blueprint.core import build_core

N_STEPS = 20
DIVIDE_AT = 8          # step at which to split one cell
DIVIDE_CELL = None     # chosen at runtime (first live cell)
OUT = Path("workspace/reports/assets/_bounded_particles_tyssue.json")

BP = {
    "blueprint_version": "1.0",
    "domain": {"extent_um": [30, 30], "grid_resolution_um": 1, "boundary": "noflux"},
    "cell_types": [{"name": "A", "target_size": 30}],
    "adhesion": [{"a": "A", "b": "A", "strength": 5},
                 {"a": "A", "b": "medium", "strength": 20}],
    "initial_state": {
        "kind": "blocks",
        "block": {"type": "A", "origin": [10, 10], "shape": [2, 2], "spacing": 6, "size": 30},
    },
}


def main():
    core = build_core()
    bp = Blueprint.from_dict(BP)
    doc = _compile(bp, backend="vertex")
    inst = Composite({"state": doc["state"]}, core=core).state["mechanics"]["instance"]

    def emit():
        try:
            return inst._emit() or {}
        except Exception:
            return {}

    frames = []
    divided = False
    for step in range(N_STEPS):
        inp = {"fates": {}, "field_grid": {}, "divisions": []}
        g0 = emit()
        live = sorted((g0.get("footprints", {}) or {}).keys())
        if step == DIVIDE_AT and live and not divided:
            inp["divisions"] = [live[0]]      # split the first cell
            divided = True
            print(f"  step {step}: dividing cell {live[0]}")
        try:
            inst.update(inp, 1.0)
        except Exception as e:
            print(f"  [vertex] step {step}: {e}")
        g = emit()
        foot = {}
        for bid, fp in (g.get("footprints", {}) or {}).items():
            xy = fp.get("xy") if isinstance(fp, dict) else None
            if xy and len(xy) >= 3:
                foot[str(bid)] = [[float(x), float(y)] for x, y in xy]
        frames.append({
            "step": step,
            "footprints": foot,
            "types": {str(k): v for k, v in (g.get("types", {}) or {}).items()},
            "lineage": {str(k): str(v) for k, v in (g.get("lineage", {}) or {}).items()},
        })

    ext = BP["domain"]["extent_um"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"extent_um": ext, "frames": frames}))
    ncells = [len(f["footprints"]) for f in frames]
    print(f"wrote {OUT}  ({len(frames)} frames, cells {ncells[0]}->{ncells[-1]})")
    # report the division: daughters appear with a lineage entry
    for f in frames:
        if f["lineage"]:
            print(f"  division at step {f['step']}: lineage {f['lineage']}")
            break


if __name__ == "__main__":
    main()
