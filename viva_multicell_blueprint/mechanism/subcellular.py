"""SubcellularProcess: a sharded per-cell sub-model runner + the coupling contract.

One Process per (cell-type x module) holds every cell of that type's solver state,
keyed by ``cell_id``, steps them each update and writes per-cell actuation REQUESTS
to its OWN stores (``sub_scalar``, ``req_*``). It never touches the mechanics-backend
input ports or RulesProcess' ``fates``/``scalar`` stores; a downstream merge step
resolves the requests.

Coupling (config ``coupling``)::

    {"inputs":  {<model_var>: <source>, ...},
     "outputs": {<effect>: <model_var>, ...}}

Input sources: ``field.<name>@centroid``, ``size``, ``neighbors.count``,
``neighbors.shared_surface.<type>``, ``type``, ``scalar.<name>``, ``const.<name>``.
Output effects: ``target_size``, ``motility``, ``adhesion`` (numeric, forwarded as-is),
``divide_when`` / ``die_when`` / ``become.<type>`` (fire when the var is > 0.5),
``secrete.<field>`` (numeric). Unknown sources/effects/model variables raise ValueError.

Lifecycle: daughters (``lineage`` daughter->mother) are created from the mother's state
per ``division``: ``copy`` deep-copies, ``split`` halves extensive quantities in mother
and daughter, ``reset`` re-inits. Dead / ``alive == False`` cells are dropped.

A module implements ``init_cell(cell_id, inputs) -> state``,
``step(state, mapped_inputs, interval) -> state``, ``read(state) -> {var: float}`` and
optionally ``split(state) -> state``. Modules register in ``MODULES``.
"""
from __future__ import annotations

import copy
from typing import Protocol

from process_bigraph import Process

from viva_multicell_blueprint.mechanism.fields import sample_at_centroid

_EFFECTS = {"target_size", "motility", "adhesion", "divide_when", "die_when"}
_PREFIXED = ("become.", "secrete.")
_FIRE_THRESHOLD = 0.5


class SubcellularModule(Protocol):
    def init_cell(self, cell_id: str, inputs: dict) -> object: ...
    def step(self, state: object, mapped_inputs: dict, interval: float) -> object: ...
    def read(self, state: object) -> dict: ...


class TestLinearModule:
    """Dependency-free deterministic module: ``x`` advances by ``(rate + drive) * interval``.

    model: ``{"x0": float}``; params: ``{"rate": float}``; ``drive`` is a mapped input.
    read(): ``x``, ``drive``, ``gt1`` (1.0 if x > 1 else 0.0).
    """

    __test__ = False  # not a pytest class

    def __init__(self, model=None, params=None):
        self._x0 = float((model or {}).get("x0", 1.0))
        self._rate = float((params or {}).get("rate", 1.0))

    def init_cell(self, cell_id, inputs):
        return {"x": self._x0, "drive": 0.0}

    def step(self, state, mapped_inputs, interval):
        drive = float(mapped_inputs.get("drive", 0.0))
        return {"x": state["x"] + (self._rate + drive) * interval, "drive": drive}

    def read(self, state):
        return {"x": state["x"], "drive": state["drive"],
                "gt1": 1.0 if state["x"] > 1.0 else 0.0}

    def split(self, state):
        return {**state, "x": state["x"] / 2.0}


MODULES = {"test_linear": TestLinearModule}


def resolve_inputs(coupling_inputs, cell_id, port_maps, const=None) -> dict:
    """Map each model variable to its value from the per-cell port maps (§4c sources)."""
    const = const or {}
    out = {}
    for var, src in (coupling_inputs or {}).items():
        if src.startswith("field.") and src.endswith("@centroid"):
            name = src[len("field."):-len("@centroid")]
            grid = (port_maps.get("field_grid") or {}).get(name)
            if grid is None:
                raise ValueError(f"unknown field {name!r} in coupling source {src!r}")
            out[var] = sample_at_centroid(grid, port_maps["positions"][cell_id])
        elif src == "size":
            out[var] = float((port_maps.get("sizes") or {}).get(cell_id, 1.0))
        elif src == "type":
            out[var] = port_maps["types"][cell_id]
        elif src == "neighbors.count":
            alive = port_maps.get("alive") or {}
            out[var] = sum(1 for j in (port_maps.get("neighbors") or {}).get(cell_id) or []
                           if alive.get(j, True))
        elif src.startswith("neighbors.shared_surface."):
            t = src[len("neighbors.shared_surface."):]
            out[var] = float(((port_maps.get("shared_surface_by_type") or {})
                              .get(cell_id) or {}).get(t, 0.0))
        elif src.startswith("scalar."):
            out[var] = float(((port_maps.get("scalar") or {}).get(cell_id) or {})
                             .get(src[len("scalar."):], 0.0))
        elif src.startswith("const."):
            k = src[len("const."):]
            if k not in const:
                raise ValueError(f"unknown const {k!r} in coupling source {src!r}")
            out[var] = float(const[k])
        else:
            raise ValueError(f"unknown coupling input source {src!r} for variable {var!r}")
    return out


def resolve_outputs(coupling_outputs, read_vars) -> dict:
    """Map actuation effects to values read from the model variables."""
    out = {}
    for effect, var in (coupling_outputs or {}).items():
        if effect not in _EFFECTS and not effect.startswith(_PREFIXED):
            raise ValueError(f"unknown coupling output effect {effect!r}")
        if var not in read_vars:
            raise ValueError(f"unknown model variable {var!r} in coupling output {effect!r}")
        v = float(read_vars[var])
        boolean = effect in ("divide_when", "die_when") or effect.startswith("become.")
        out[effect] = (v > _FIRE_THRESHOLD) if boolean else v
    return out


class SubcellularProcess(Process):
    config_schema = {
        "cell_type": {"_type": "string", "_default": ""},
        "module": {"_type": "string", "_default": "test_linear"},
        "model": {"_type": "node", "_default": {}},
        "params": {"_type": "node", "_default": {}},
        "coupling": {"_type": "node", "_default": {}},
        "division": {"_type": "string", "_default": "copy"},
        "seed": {"_type": "integer", "_default": 0},
    }

    def initialize(self, config):
        name = config.get("module", "test_linear")
        if name not in MODULES:
            raise ValueError(f"unknown subcellular module {name!r}; have {sorted(MODULES)}")
        self._division = config.get("division", "copy")
        if self._division not in ("copy", "split", "reset"):
            raise ValueError(f"unknown division policy {self._division!r}")
        self._type = config.get("cell_type", "")
        self._module = MODULES[name](config.get("model") or {}, config.get("params") or {})
        coupling = config.get("coupling") or {}
        self._cin = dict(coupling.get("inputs") or {})
        self._cout = dict(coupling.get("outputs") or {})
        self._const = {k: float(v) for k, v in (coupling.get("const") or {}).items()}
        self._cells = {}

    def inputs(self):
        return {
            "types": "map[string]", "sizes": "map[float]", "positions": "map[list[float]]",
            "neighbors": "map[list[string]]", "alive": "map[boolean]",
            "field_grid": "map[node]", "divisions": "list[node]", "deaths": "list[node]",
            "lineage": "map[string]",
            # optional extras backing the scalar.* / neighbors.shared_surface.* sources
            "scalar": "map[map[float]]", "shared_surface_by_type": "map[map[float]]",
        }

    def outputs(self):
        return {
            "sub_scalar": "overwrite[map[map[float]]]",
            "req_target_size": "overwrite[map[float]]",
            "req_motility": "overwrite[map[float]]",
            "req_adhesion": "overwrite[map[node]]",
            "req_divide": "list[node]", "req_die": "list[node]",
            "req_become": "overwrite[map[string]]",
            "req_secrete": "overwrite[map[map[float]]]",
        }

    def _spawn(self, cid, inputs, mother):
        m = self._cells.get(mother) if mother is not None else None
        if m is None or self._division == "reset":
            return self._module.init_cell(cid, inputs)
        if self._division == "split":  # both halves keep half the extensive quantities
            self._cells[mother] = self._module.split(m)
            return copy.deepcopy(self._cells[mother])
        return copy.deepcopy(m)

    def update(self, inputs, interval=1):
        types = inputs.get("types") or {}
        alive = inputs.get("alive") or {}
        lineage = inputs.get("lineage") or {}
        dead = {str(d["cell_id"]) for d in inputs.get("deaths") or []}
        mine = sorted(c for c, t in types.items()
                      if t == self._type and alive.get(c, True) and c not in dead)

        # spawn first (a mother that dies this step is still in _cells), then drop
        for c in mine:
            if c not in self._cells:
                self._cells[c] = self._spawn(c, inputs, lineage.get(c))
        for c in [c for c in self._cells if c not in mine]:
            del self._cells[c]

        out = {"sub_scalar": {}, "req_target_size": {}, "req_motility": {}, "req_adhesion": {},
               "req_divide": [], "req_die": [], "req_become": {}, "req_secrete": {}}
        for c in mine:
            mapped = resolve_inputs(self._cin, c, inputs, self._const)
            self._cells[c] = self._module.step(self._cells[c], mapped, interval)
            read = self._module.read(self._cells[c])
            out["sub_scalar"][c] = {k: float(v) for k, v in read.items()}
            for effect, v in resolve_outputs(self._cout, read).items():
                if effect == "target_size":
                    out["req_target_size"][c] = v
                elif effect == "motility":
                    out["req_motility"][c] = v
                elif effect == "adhesion":
                    out["req_adhesion"][c] = v
                elif effect == "divide_when" and v:
                    out["req_divide"].append({"cell_id": c})
                elif effect == "die_when" and v:
                    out["req_die"].append({"cell_id": c})
                elif effect.startswith("become.") and v:
                    out["req_become"][c] = effect[len("become."):]
                elif effect.startswith("secrete."):
                    out["req_secrete"].setdefault(c, {})[effect[len("secrete."):]] = v
        return out
