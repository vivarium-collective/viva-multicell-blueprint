"""CoupledCell composites — a cell whose intracellular model couples several sub-models.

Two registered composites, one per within-cell coupling motif (both proven standalone in
``scripts/spike_coupled_cell*.py`` against the real engines):

  * ``coupled_fate_oscillator`` — an ODE metabolism and a MaBoSS fate network, coupled
    both ways (mode A: metabolite C pins the ``high_C`` input; mode C: the ``divide``
    output resets the metabolism). The result is a metabolism↔fate feedback oscillator.
  * ``coupled_particle_catalysis`` — a Smoldyn particle sim whose C population catalyses
    an ODE reaction S→P (mode B: the live particle count sets the ODE rate constant).

Each runs a small tissue of such cells on a chosen mechanics backend; the tissue holds
the cells while the coupled intracellular dynamics play out per cell.
"""
from __future__ import annotations

from process_bigraph.composite_generator import composite_generator

from ..core import register_workspace_processes
from ..model_format.compile import SceneModelProcess  # noqa: F401 (registers it)
from ..model_format.compile import compile_scene
from ..model_format.registry import ModelSpec, Registry
from ..model_format.scene import CellSpec, Scene

# ---- mode A + C: ODE metabolism <-> MaBoSS fate oscillator ------------------------------
_OSC_SOURCE = {
    "models": {
        "metab": {"formalism": "ode",
                  "source": {"species": {"A": 1.2, "B": 1.2, "C": 0.6},
                             "reactions": [{"subs": ["A", "B"], "prds": ["C"],
                                            "kf": 0.01, "kb": 0.3}]},
                  "initial": {"A": 40.0, "B": 40.0, "C": 0.0}},
        "fate": {"formalism": "boolean",
                 "source": {"high_C": "high_C", "divide": "high_C"},
                 "initial": {"high_C": 0.0, "divide": 0.0}},
    },
    "couplings": [
        {"mode": "A", "from": "metab", "species": "C", "threshold": 12.0,
         "to": "fate", "input": "high_C"},
        {"mode": "C", "from": "fate", "node": "divide", "to": "metab", "action": "reset"},
    ],
}

# ---- mode B: Smoldyn particle count -> ODE catalysis rate --------------------------------
_CAT_SOURCE = {
    "models": {
        "sensor": {"formalism": "particle",
                   "source": {"species": {"A": 1, "B": 1, "C": 0},
                              "reactions": [{"subs": ["A", "B"], "prds": ["C"],
                                             "kf": 0.35, "kb": 0.015}]},
                   "initial": {"A": 45.0, "B": 45.0, "C": 0.0}, "seed": 7},
        "metab": {"formalism": "ode",
                  "source": {"species": {"S": 400.0, "P": 0.0},
                             "reactions": [{"subs": ["S"], "prds": ["P"],
                                            "kf": 0.0, "kb": 0.0}]},
                  "initial": {"S": 400.0, "P": 0.0}},
    },
    "couplings": [
        {"mode": "B", "from": "sensor", "species": "C",
         "to": "metab", "param": "kf0", "base": 0.001},
    ],
}


# ---- all three formalisms + all three modes in ONE cell ---------------------------------
# particle sensor catalyses the ODE metabolism (B); the ODE metabolite pins the MaBoSS fate
# input (A); the fate 'divide' output resets the metabolism (C).
_TRIPLE_SOURCE = {
    "models": {
        "sensor": {"formalism": "particle",
                   "source": {"species": {"A": 1, "B": 1, "C": 0},
                              "reactions": [{"subs": ["A", "B"], "prds": ["C"],
                                             "kf": 0.35, "kb": 0.015}]},
                   "initial": {"A": 30.0, "B": 30.0, "C": 0.0}, "seed": 11},
        "metab": {"formalism": "ode",
                  "source": {"species": {"A": 1.2, "B": 1.2, "C": 0.6},
                             "reactions": [{"subs": ["A", "B"], "prds": ["C"],
                                            "kf": 0.0, "kb": 0.3}]},
                  "initial": {"A": 40.0, "B": 40.0, "C": 0.0}},
        "fate": {"formalism": "boolean",
                 "source": {"high_C": "high_C", "divide": "high_C"},
                 "initial": {"high_C": 0.0, "divide": 0.0}},
    },
    "couplings": [
        {"mode": "B", "from": "sensor", "species": "C",
         "to": "metab", "param": "kf0", "base": 0.0004},
        {"mode": "A", "from": "metab", "species": "C", "threshold": 12.0,
         "to": "fate", "input": "high_C"},
        {"mode": "C", "from": "fate", "node": "divide", "to": "metab", "action": "reset"},
    ],
}


def _grid_scene(cell_type: str, n: int, spacing: float = 12.0) -> Scene:
    """A small square-ish grid of ``n`` cells of one type."""
    import math
    cols = max(1, int(round(math.sqrt(n))))
    cells = {}
    for i in range(n):
        r, c = divmod(i, cols)
        cells[f"c{i}"] = CellSpec(type=cell_type,
                                  position=[6.0 + c * spacing, 6.0 + r * spacing],
                                  molecule_levels={})
    w = 12.0 + cols * spacing
    h = 12.0 + (1 + (n - 1) // cols) * spacing
    return Scene(cells=cells, domain={"extent_um": [w, h]})


@composite_generator(
    name="coupled_fate_oscillator",
    description=(
        "One cell, two coupled sub-cellular sims: an SBML-ODE metabolism and a MaBoSS "
        "Boolean fate network form a metabolism↔fate feedback oscillator (C builds → pins "
        "high_C → divide fires → division resets metabolism → repeat). Runs a small tissue "
        "of these coupled cells."
    ),
    parameters={
        "n_cells": {"type": "integer", "default": 4,
                    "description": "number of coupled cells in the tissue"},
        "mechanics": {"type": "string", "default": "vertex",
                      "description": "tissue mechanics backend: 'vertex', 'cpm' or 'munk'"},
    },
    core_extensions=[register_workspace_processes],
    default_n_steps=40,
)
def coupled_fate_oscillator(core=None, n_cells: int = 4, mechanics: str = "vertex") -> dict:
    scene = _grid_scene("coupled_cell", int(n_cells))
    registry = Registry({"coupled_cell": ModelSpec(formalism="coupled", source=_OSC_SOURCE)})
    return compile_scene(scene, registry, mechanics=str(mechanics))


@composite_generator(
    name="coupled_particle_catalysis",
    description=(
        "One cell, two coupled sub-cellular sims: a Smoldyn particle population (A+B→C, "
        "confined to the cell polygon) catalyses an SBML-ODE reaction S→P — the live "
        "particle C count sets the ODE rate constant each step (mode B), so P accumulates "
        "faster as the particle population builds. Runs a small tissue of these coupled cells."
    ),
    parameters={
        "n_cells": {"type": "integer", "default": 3,
                    "description": "number of coupled cells in the tissue"},
        "mechanics": {"type": "string", "default": "vertex",
                      "description": "tissue mechanics backend: 'vertex', 'cpm' or 'munk'"},
    },
    core_extensions=[register_workspace_processes],
    default_n_steps=30,
)
def coupled_particle_catalysis(core=None, n_cells: int = 3, mechanics: str = "vertex") -> dict:
    scene = _grid_scene("coupled_cell", int(n_cells))
    registry = Registry({"coupled_cell": ModelSpec(formalism="coupled", source=_CAT_SOURCE)})
    return compile_scene(scene, registry, mechanics=str(mechanics))


@composite_generator(
    name="coupled_triple_cell",
    description=(
        "One cell running ALL THREE sub-cellular formalisms, coupled by all three modes: a "
        "Smoldyn particle population catalyses an SBML-ODE metabolism (mode B), whose "
        "metabolite C pins a MaBoSS fate input (mode A), whose 'divide' output resets the "
        "metabolism (mode C). Particles → metabolism → fate → division, in a single cell."
    ),
    parameters={
        "n_cells": {"type": "integer", "default": 2,
                    "description": "number of triple-coupled cells in the tissue"},
        "mechanics": {"type": "string", "default": "vertex",
                      "description": "tissue mechanics backend: 'vertex', 'cpm' or 'munk'"},
    },
    core_extensions=[register_workspace_processes],
    default_n_steps=36,
)
def coupled_triple_cell(core=None, n_cells: int = 2, mechanics: str = "vertex") -> dict:
    scene = _grid_scene("coupled_cell", int(n_cells))
    registry = Registry({"coupled_cell": ModelSpec(formalism="coupled", source=_TRIPLE_SOURCE)})
    return compile_scene(scene, registry, mechanics=str(mechanics))
