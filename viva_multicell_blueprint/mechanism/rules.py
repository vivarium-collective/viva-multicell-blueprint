"""RulesProcess: the shared, backend-neutral ordered rate-rule chain.

Turns blueprint ``rules`` into per-cell fates, derived scalars and lineage
events, identically for every MechanicsBackend (spec sections 5, 7).

Rule kinds (all carry a ``rate_expr`` in the closed rate grammar):
  transition / allee / recovery  {from, to, rate_expr}
  kill                           {target, to?="D", killer?, remove?, rate_expr}
  resistance                     {scalar?="resist", types?, rate_expr}
  divide                         {type, rate_expr}

Semantics
- Rules run in declared order. A ``resistance`` rule writes a per-cell scalar
  that LATER rules read as ``scalar['name']``; scalars are recomputed each step.
- Rate -> probability follows the Sego convention: ``Pr = 1 - exp(-rate)``
  (CC3D ``ImmuneModelLib.ul_rate_to_prob``; viva-cpm
  ``influenza/transitions.py`` ``infection_step`` / ``infected_death_step``,
  and ``allee.py`` / ``killing.py`` docstrings: "the caller applies
  Pr = 1 - exp(-rate)"). A negative rate is clamped to 0.
- Each rule is matched against the cell's type at the START of the step, and a
  cell takes at most one fate per step (first matching rule that fires wins).
  Fates are emitted, not applied, so a chain H->I->D needs two steps.
- A transition/kill/allee/recovery whose ``to`` is empty or ``remove: true``
  emits a ``deaths`` lineage event instead of a fate; ``divide`` emits a
  ``divisions`` event. Events are ``{"cell_id": id}``.
- Cells with ``alive`` False are skipped entirely.
- Field exposure ``field[name]`` is the grid value at the cell centroid
  (rounded site index, same indexing as FieldProcess).
- Draws come from ``numpy.random.default_rng(seed)``: one uniform per
  (rule, eligible cell), cells in sorted-id order -> reproducible.

The rate expressions themselves come from the blueprint (the grammar); no
Sego application math is vendored here (it is all ``1 - exp(-rate)``).
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
from process_bigraph import Process

from viva_multicell_blueprint.mechanism.rate_expr import CellContext, evaluate, parse

_FATE_KINDS = {"transition", "allee", "recovery", "kill"}
_ALL_KINDS = _FATE_KINDS | {"resistance", "divide"}


class RulesProcess(Process):
    config_schema = {
        "rules": {"_type": "list[node]", "_default": []},
        "const": {"_type": "map[float]", "_default": {}},
        "seed": {"_type": "integer", "_default": 0},
    }

    def initialize(self, config):
        self._rng = np.random.default_rng(int(config.get("seed", 0)))
        self._const = {k: float(v) for k, v in (config.get("const") or {}).items()}
        self._rules = []
        for r in config.get("rules") or []:
            kind = r.get("kind") or r.get("type_")
            if kind not in _ALL_KINDS:
                raise ValueError(f"unknown rule kind {kind!r}")
            src = r.get("from", r.get("target", r.get("type")))
            if kind != "resistance" and src is None:
                raise ValueError(f"rule {kind!r} needs a source type (from/target/type)")
            self._rules.append({
                "kind": kind, "src": src, "expr": parse(r["rate_expr"]),
                "to": r.get("to", "D" if kind == "kill" else None),
                "remove": bool(r.get("remove", False)),
                "scalar": r.get("scalar", "resist"),
                "types": r.get("types"),
            })

    def inputs(self):
        return {
            "positions": "map[list[float]]", "sizes": "map[float]", "types": "map[string]",
            "shared_surface_by_type": "map[map[float]]", "neighbors": "map[list[string]]",
            "alive": "map[boolean]", "field_grid": "map[node]",
        }

    def outputs(self):
        # fates/scalar are the per-step result maps (deltas / recomputed each step),
        # so overwrite replaces the store whole -- a bare map[..] apply would both
        # drop keys on the empty t=0 store AND leave stale prior-step fates behind.
        return {"fates": "overwrite[map[string]]", "scalar": "overwrite[map[map[float]]]",
                "divisions": "list[node]", "deaths": "list[node]"}

    @staticmethod
    def _exposure(grid, pos):
        idx = tuple(int(min(max(round(float(v)), 0), n - 1)) for v, n in zip(pos, grid.shape))
        return float(grid[idx])

    def update(self, inputs, interval=1):
        positions = inputs.get("positions") or {}
        sizes = inputs.get("sizes") or {}
        types = inputs.get("types") or {}
        shared = inputs.get("shared_surface_by_type") or {}
        neighbors = inputs.get("neighbors") or {}
        alive = inputs.get("alive") or {}
        grids = {k: np.asarray(v) for k, v in (inputs.get("field_grid") or {}).items()}

        cids = sorted(c for c in types if alive.get(c, True))
        scalars = {c: {} for c in cids}
        fates, removed, deaths, divisions = {}, set(), [], []

        def ctx(c):
            ncount = defaultdict(int)
            for j in neighbors.get(c) or []:
                if alive.get(j, True) and j in types:
                    ncount[types[j]] += 1
            return CellContext(
                type=types[c], size=float(sizes.get(c, 1.0)),
                field={n: self._exposure(g, positions[c]) for n, g in grids.items()},
                neighbor_count=ncount,
                shared_surface=defaultdict(float, shared.get(c) or {}),
                scalar=scalars[c], const=self._const)

        start_type = {c: types[c] for c in cids}
        for rule in self._rules:
            if rule["kind"] == "resistance":
                for c in cids:
                    if rule["types"] is None or start_type[c] in rule["types"]:
                        scalars[c][rule["scalar"]] = evaluate(rule["expr"], ctx(c))
                continue
            for c in cids:
                if start_type[c] != rule["src"] or c in removed:
                    continue
                if rule["kind"] != "divide" and c in fates:
                    continue
                rate = max(evaluate(rule["expr"], ctx(c)), 0.0)
                if not self._rng.random() < 1.0 - math.exp(-rate):
                    continue
                if rule["kind"] == "divide":
                    divisions.append({"cell_id": c})
                elif rule["remove"] or not rule["to"]:
                    deaths.append({"cell_id": c})
                    removed.add(c)
                else:
                    fates[c] = rule["to"]
        return {"fates": fates, "scalar": scalars, "divisions": divisions, "deaths": deaths}
