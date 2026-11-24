"""``divergence_compare``: one blueprint run four ways — CPM alone, vertex alone,
center-based alone, and the tri-coupled consensus — rendered as a 3-column x
2-row animation (top: each paradigm standalone; bottom: the same three coupled)
so you can see how the mechanics diverge from one shared initial state. See
:mod:`..mechanism.divergence_compare`.
"""
from __future__ import annotations

from pathlib import Path

from process_bigraph.composite_generator import composite_generator

from ..core import register_workspace_processes
from ..mechanism.divergence_compare import (
    DivergenceCompareProcess,  # noqa: F401 (registers it)
)

_PATCH = Path(__file__).resolve().parents[2] / "workspace" / "blueprints" / "tri_coupled_patch.yaml"


@composite_generator(
    name="divergence_compare",
    description=(
        "One blueprint run four ways from the same init: CPM, vertex and center-based "
        "each ALONE (top row) vs the tri-coupled consensus (bottom row), as a 3x2 "
        "animation showing how the paradigms diverge. Writes a flipbook + GIF to "
        "reports/assets/<overlay_out>.html."
    ),
    parameters={
        "blueprint": {"type": "string", "default": str(_PATCH),
                      "description": "a path to a YAML file, inline YAML text, or a dict"},
        "cpm_res": {"type": "integer", "default": 3,
                    "description": "CPM lattice resolution (voxels per µm)"},
        "cpm_temp": {"type": "number", "default": 1.0,
                     "description": "CPM Metropolis temperature"},
        "munk_squish": {"type": "number", "default": 0.8,
                        "description": "center-based oblong-under-crowding"},
        "overlay_out": {"type": "string", "default": "divergence_compare",
                        "description": "flipbook written to reports/assets/<overlay_out>.html"},
        "render_every": {"type": "integer", "default": 1,
                         "description": "keep only every Nth step in the gif (long runs stay small)"},
    },
    core_extensions=[register_workspace_processes],
    default_n_steps=16,
)
def divergence_compare(core=None, blueprint=None, cpm_res: int = 3, cpm_temp: float = 1.0,
                       munk_squish: float = 0.8, overlay_out: str = "divergence_compare",
                       render_every: int = 1) -> dict:
    return {
        "state": {
            "divergence_compare": {
                "_type": "process",
                "address": "local:DivergenceCompareProcess",
                "config": {
                    "blueprint": blueprint or str(_PATCH),
                    "cpm_res": int(cpm_res),
                    "cpm_temp": float(cpm_temp),
                    "munk_squish": float(munk_squish),
                    "overlay_out": str(overlay_out),
                    "render_every": int(render_every),
                },
                "inputs": {},
                "outputs": {"cpm_vs_coupled_drift": ["cpm_vs_coupled_drift"],
                            "n_cells": ["n_cells"]},
            },
            "cpm_vs_coupled_drift": 0.0,
            "n_cells": 0,
        }
    }
