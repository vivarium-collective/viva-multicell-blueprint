"""``particle_consensus``: Smoldyn particles confined inside a tri-coupled consensus tissue —
molecules bounded by a boundary three tissue simulators agree on (vertex shape, munk position,
CPM area). See :mod:`..mechanism.particle_consensus`.
"""
from __future__ import annotations

from pathlib import Path

from process_bigraph.composite_generator import composite_generator

from ..core import register_workspace_processes
from ..mechanism.particle_consensus import ParticleConsensusProcess  # noqa: F401 (registers it)

_PATCH = Path(__file__).resolve().parents[2] / "workspace" / "blueprints" / "tri_coupled_patch.yaml"


@composite_generator(
    name="particle_consensus",
    description=(
        "Smoldyn particles confined inside a tri-coupled consensus tissue: each cell runs an "
        "A+B<->C particle sim bounded by a boundary three tissue simulators agree on (vertex "
        "shape, munk-consensus position, CPM area). Writes a flipbook to reports/assets/<overlay_out>.html."
    ),
    parameters={
        "blueprint": {"type": "string", "default": str(_PATCH),
                      "description": "a path to a YAML file, inline YAML text, or a dict"},
        "seed_per_cell": {"type": "integer", "default": 50,
                          "description": "molecules (A+B) seeded per cell"},
        "munk_squish": {"type": "number", "default": 0.8,
                        "description": "center-based oblong-under-crowding"},
        "overlay_out": {"type": "string", "default": "particle_consensus",
                        "description": "flipbook written to reports/assets/<overlay_out>.html"},
    },
    core_extensions=[register_workspace_processes],
    default_n_steps=16,
)
def particle_consensus(core=None, blueprint=None, seed_per_cell: int = 50, munk_squish: float = 0.8,
                       overlay_out: str = "particle_consensus") -> dict:
    return {
        "state": {
            "particle_consensus": {
                "_type": "process",
                "address": "local:ParticleConsensusProcess",
                "config": {
                    "blueprint": blueprint or str(_PATCH),
                    "seed_per_cell": int(seed_per_cell),
                    "munk_squish": float(munk_squish),
                    "overlay_out": str(overlay_out),
                },
                "inputs": {},
                "outputs": {"consensus_spread": ["consensus_spread"], "n_cells": ["n_cells"],
                            "n_particles": ["n_particles"], "confined_fraction": ["confined_fraction"]},
            },
            "consensus_spread": 0.0, "n_cells": 0, "n_particles": 0, "confined_fraction": 0.0,
        }
    }
