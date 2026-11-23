"""``bounded_particles``: a Registry-visible composite that composes a particle
simulator (Smoldyn) with a multicellular one (the tyssue/vertex backend).

Each tyssue cell is a moving/deforming polygon; a Smoldyn sim runs inside each one
(reflective boundary + reversible A+B<->C), and on division a cell's molecules are
partitioned into its two daughters. A single :class:`BoundedParticlesProcess` node
drives the vertex backend, couples the confined particles, and writes a cell-coloured
overlay (flipbook + GIF) to ``reports/assets/<overlay_out>.html``.

Smoldyn runs in-process where it has a binary wheel (Linux/Python 3.12); on macOS it is
reached through a Python 3.11 subprocess bridge (see :mod:`..particles.runner`).
"""
from __future__ import annotations

from pathlib import Path

from process_bigraph.composite_generator import composite_generator

from ..core import register_workspace_processes
from ..mechanism.bounded_particles import BoundedParticlesProcess  # noqa: F401 (registers it)

_PATCH = Path(__file__).resolve().parents[2] / "workspace" / "blueprints" / "bounded_particles_patch.yaml"


@composite_generator(
    name="bounded_particles",
    description=(
        "Smoldyn particle sims confined inside tyssue cells: each vertex cell is a "
        "moving polygon with its own reflective Smoldyn sim (A+B<->C); molecules split "
        "into daughters on division. Writes a cell-coloured overlay to "
        "reports/assets/<overlay_out>.html."
    ),
    parameters={
        "blueprint": {"type": "string", "default": str(_PATCH),
                      "description": "a path to a YAML file, inline YAML text, or a dict"},
        "seed_per_cell": {"type": "integer", "default": 60,
                          "description": "molecules (A+B) seeded per cell"},
        "divide_every": {"type": "integer", "default": 8,
                         "description": "divide one cell every N steps (0 = never)"},
        "overlay_out": {"type": "string", "default": "bounded_particles",
                        "description": "overlay written to reports/assets/<overlay_out>.html"},
    },
    core_extensions=[register_workspace_processes],
    default_n_steps=20,
)
def bounded_particles(core=None, blueprint=None, seed_per_cell: int = 60,
                      divide_every: int = 8, overlay_out: str = "bounded_particles") -> dict:
    return {
        "state": {
            "bounded_particles": {
                "_type": "process",
                "address": "local:BoundedParticlesProcess",
                "config": {
                    "blueprint": blueprint or str(_PATCH),
                    "seed_per_cell": int(seed_per_cell),
                    "divide_every": int(divide_every),
                    "overlay_out": str(overlay_out),
                },
                "inputs": {},
                "outputs": {"n_cells": ["n_cells"], "n_particles": ["n_particles"],
                            "smoldyn_mode": ["smoldyn_mode"]},
            },
            "n_cells": 0, "n_particles": 0, "smoldyn_mode": "",
        }
    }
