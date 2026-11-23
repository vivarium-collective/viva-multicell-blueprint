"""``tri_coupled``: a Registry-visible composite running all three mechanics
paradigms together, coupled into one consensus system.

CPM (voxel), vertex (polygon) and center-based (munk disc) each hold the SAME
cells (shared blueprint id). The single :class:`TriCoupledProcess` node steps all
three in lockstep and couples them: **vertex owns shape** (the CPM rasterizes each
vertex polygon so its cell takes the vertex shape), **munk owns position** (all
three centroids converge to a munk-weighted consensus, and the CPM shape is placed
there — so the CPM is tugged by munk too), **CPM owns area** (its true occupied
area sizes the munk disc + the vertex target area). Each step also writes a
self-contained overlay HTML (all three representations per cell) to
``reports/assets/<overlay_out>.html``.
"""
from __future__ import annotations

from process_bigraph.composite_generator import composite_generator

from pathlib import Path

from ..core import register_workspace_processes
from ..mechanism.tri_coupled import TriCoupledProcess  # noqa: F401 (registers the Process)

# A small two-type patch kept tight so the consensus coupling converges + reads
# clearly; override `blueprint` to run the composite on any other blueprint.
_PATCH = Path(__file__).resolve().parents[2] / "workspace" / "blueprints" / "tri_coupled_patch.yaml"


@composite_generator(
    name="tri_coupled",
    description=(
        "All three mechanics paradigms on one shared cell set, coupled into a "
        "consensus: CPM (voxel) owns area, vertex (polygon) owns shape, munk "
        "(center-based disc) owns position. Writes an overlay of all three per "
        "cell to reports/assets/<overlay_out>.html."
    ),
    parameters={
        "blueprint": {"type": "string", "default": str(_PATCH),
                      "description": "a path to a YAML file, inline YAML text, or a dict"},
        "cpm_res": {"type": "integer", "default": 3,
                    "description": "CPM lattice resolution (voxels per µm)"},
        "cpm_temp": {"type": "number", "default": 1.0,
                     "description": "CPM Metropolis temperature (lower = crisper shape match)"},
        "munk_squish": {"type": "number", "default": 0.8,
                        "description": "center-based oblong-under-crowding (0 = rigid circle)"},
        "overlay_out": {"type": "string", "default": "tri_coupled",
                        "description": "overlay written to reports/assets/<overlay_out>.html"},
    },
    core_extensions=[register_workspace_processes],
    default_n_steps=24,
)
def tri_coupled(core=None, blueprint=None, cpm_res: int = 3, cpm_temp: float = 1.0,
                munk_squish: float = 0.8, overlay_out: str = "tri_coupled") -> dict:
    return {
        "state": {
            "tri_coupled": {
                "_type": "process",
                "address": "local:TriCoupledProcess",
                "config": {
                    "blueprint": blueprint or str(_PATCH),
                    "cpm_res": int(cpm_res),
                    "cpm_temp": float(cpm_temp),
                    "munk_squish": float(munk_squish),
                    "overlay_out": str(overlay_out),
                },
                "inputs": {},
                "outputs": {"consensus_spread": ["consensus_spread"], "n_cells": ["n_cells"]},
            },
            "consensus_spread": 0.0,
            "n_cells": 0,
        }
    }
