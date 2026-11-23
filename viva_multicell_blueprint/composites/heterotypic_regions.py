"""``heterotypic_regions``: two tissue TYPES, two physics — a vertex epithelium on a munk
mesenchyme, coupled at a reflective interface. See :mod:`..mechanism.heterotypic_regions`.
"""
from __future__ import annotations

from process_bigraph.composite_generator import composite_generator

from ..core import register_workspace_processes
from ..mechanism.heterotypic_regions import HeterotypicRegionsProcess  # noqa: F401 (registers it)


@composite_generator(
    name="heterotypic_regions",
    description=(
        "Two tissue types, two physics: a vertex-simulated epithelium (sheet) sitting on a "
        "center-based munk mesenchyme (loose cells), coupled at their interface — the epithelium's "
        "basal edge is a reflective floor the mesenchyme presses up against. Writes a flipbook to "
        "reports/assets/<overlay_out>.html."
    ),
    parameters={
        "extent": {"type": "integer", "default": 30, "description": "square domain size (um)"},
        "n_epi": {"type": "integer", "default": 6, "description": "epithelial cells (vertex row)"},
        "n_mes": {"type": "integer", "default": 10, "description": "mesenchymal cells (munk cluster)"},
        "overlay_out": {"type": "string", "default": "heterotypic_regions",
                        "description": "flipbook written to reports/assets/<overlay_out>.html"},
    },
    core_extensions=[register_workspace_processes],
    default_n_steps=18,
)
def heterotypic_regions(core=None, extent: int = 30, n_epi: int = 6, n_mes: int = 10,
                        overlay_out: str = "heterotypic_regions") -> dict:
    return {
        "state": {
            "heterotypic_regions": {
                "_type": "process",
                "address": "local:HeterotypicRegionsProcess",
                "config": {"extent": int(extent), "n_epi": int(n_epi), "n_mes": int(n_mes),
                           "overlay_out": str(overlay_out)},
                "inputs": {},
                "outputs": {"n_epithelium": ["n_epithelium"], "n_mesenchyme": ["n_mesenchyme"],
                            "n_cells": ["n_cells"], "interface_breaches": ["interface_breaches"]},
            },
            "n_epithelium": 0, "n_mesenchyme": 0, "n_cells": 0, "interface_breaches": 0,
        }
    }
