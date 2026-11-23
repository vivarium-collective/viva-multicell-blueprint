"""``fateful_consensus``: the grand-unification showpiece — a tri-coupled consensus tissue
(CPM + vertex + munk) where each cell runs an ODE metabolism → MaBoSS fate chain driven by its
CPM-owned area. Three tissue simulators + two sub-cellular simulators in one composite, with a
live mechanics → metabolism → fate coupling. See :mod:`..mechanism.fateful_consensus`.
"""
from __future__ import annotations

from pathlib import Path

from process_bigraph.composite_generator import composite_generator

from ..core import register_workspace_processes
from ..mechanism.fateful_consensus import FatefulConsensusProcess  # noqa: F401 (registers it)

_PATCH = Path(__file__).resolve().parents[2] / "workspace" / "blueprints" / "tri_coupled_patch.yaml"


@composite_generator(
    name="fateful_consensus",
    description=(
        "Grand unification: a tri-coupled consensus tissue (CPM owns area, vertex owns shape, "
        "munk owns position) where every cell runs an SBML-ODE metabolism whose metabolite pins a "
        "MaBoSS fate network — and each cell's CPM AREA sets its metabolic rate, so tissue "
        "mechanics drives sub-cellular fate. Three tissue + two sub-cellular simulators in one "
        "composite; writes a fate-coloured consensus flipbook to reports/assets/<overlay_out>.html."
    ),
    parameters={
        "blueprint": {"type": "string", "default": str(_PATCH),
                      "description": "a path to a YAML file, inline YAML text, or a dict"},
        "munk_squish": {"type": "number", "default": 0.8,
                        "description": "center-based oblong-under-crowding (0 = rigid circle)"},
        "overlay_out": {"type": "string", "default": "fateful_consensus",
                        "description": "consensus overlay written to reports/assets/<overlay_out>.html"},
    },
    core_extensions=[register_workspace_processes],
    default_n_steps=18,
)
def fateful_consensus(core=None, blueprint=None, munk_squish: float = 0.8,
                      overlay_out: str = "fateful_consensus") -> dict:
    return {
        "state": {
            "fateful_consensus": {
                "_type": "process",
                "address": "local:FatefulConsensusProcess",
                "config": {
                    "blueprint": blueprint or str(_PATCH),
                    "munk_squish": float(munk_squish),
                    "overlay_out": str(overlay_out),
                },
                "inputs": {},
                "outputs": {"consensus_spread": ["consensus_spread"], "n_cells": ["n_cells"],
                            "mean_metabolite": ["mean_metabolite"],
                            "committed_fraction": ["committed_fraction"]},
            },
            "consensus_spread": 0.0, "n_cells": 0,
            "mean_metabolite": 0.0, "committed_fraction": 0.0,
        }
    }
