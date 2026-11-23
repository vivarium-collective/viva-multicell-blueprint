"""``fateful_tissue``: the full-stack crown — a dividing tissue whose divisions are
decided by each cell's own metabolism→fate chain, with confined particles that split on
division. Three formalisms (ODE + Smoldyn + MaBoSS) across three scales (molecule, cell,
tissue) in one composite. See :mod:`..mechanism.fateful_tissue`.
"""
from __future__ import annotations

from pathlib import Path

from process_bigraph.composite_generator import composite_generator

from ..core import register_workspace_processes
from ..mechanism.fateful_tissue import FatefulTissueProcess  # noqa: F401 (registers it)

_PATCH = Path(__file__).resolve().parents[2] / "workspace" / "blueprints" / "bounded_particles_patch.yaml"


@composite_generator(
    name="fateful_tissue",
    description=(
        "The full-stack crown: a dividing tyssue where each cell runs a Smoldyn particle "
        "sim (split on division) AND an ODE metabolism whose metabolite pins a MaBoSS fate "
        "network — and the fate network's 'divide' output drives the tissue division. "
        "Biology decides when the tissue divides. Writes a fate-coloured flipbook + GIF to "
        "reports/assets/<overlay_out>.html."
    ),
    parameters={
        "blueprint": {"type": "string", "default": str(_PATCH),
                      "description": "a path to a YAML file, inline YAML text, or a dict"},
        "seed_per_cell": {"type": "integer", "default": 50,
                          "description": "molecules (A+B) seeded per cell"},
        "c_threshold": {"type": "number", "default": 14.0,
                        "description": "metabolite C level that pins the fate 'high_C' input"},
        "max_divisions_per_step": {"type": "integer", "default": 1,
                                   "description": "cap on fate-driven divisions per step (tissue stability)"},
        "overlay_out": {"type": "string", "default": "fateful_tissue",
                        "description": "fate-coloured overlay written to reports/assets/<overlay_out>.html"},
    },
    core_extensions=[register_workspace_processes],
    default_n_steps=22,
)
def fateful_tissue(core=None, blueprint=None, seed_per_cell: int = 50, c_threshold: float = 14.0,
                   max_divisions_per_step: int = 1, overlay_out: str = "fateful_tissue") -> dict:
    return {
        "state": {
            "fateful_tissue": {
                "_type": "process",
                "address": "local:FatefulTissueProcess",
                "config": {
                    "blueprint": blueprint or str(_PATCH),
                    "backend": "vertex",
                    "seed_per_cell": int(seed_per_cell),
                    "c_threshold": float(c_threshold),
                    "max_divisions_per_step": int(max_divisions_per_step),
                    "overlay_out": str(overlay_out),
                },
                "inputs": {},
                "outputs": {"n_cells": ["n_cells"], "n_particles": ["n_particles"],
                            "n_divisions": ["n_divisions"], "mean_metabolite": ["mean_metabolite"],
                            "smoldyn_mode": ["smoldyn_mode"]},
            },
            "n_cells": 0, "n_particles": 0, "n_divisions": 0,
            "mean_metabolite": 0.0, "smoldyn_mode": "",
        }
    }
