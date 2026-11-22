"""Tellurium (RoadRunner ODE) subcellular module.

``model``: ``{"source": <antimony|sbml text>, "format": "antimony"|"sbml", "file": <path>}``.
``params``: ``{"integrator": "cvode", "absolute_tolerance", "relative_tolerance", "seed",
"parameters": {id: value}}`` (parameter overrides applied before every step).

One RoadRunner instance is shared by all cells of the module; each cell's state is a
plain deep-copyable dict ``{"species", "params", "time", "rates"}``. ``step`` loads the
cell state into the instance, applies mapped inputs (species or global parameters),
integrates, and stores the result back. Parameters determined by an assignment rule are
not settable: they are excluded from ``input_vars`` and recomputed by RoadRunner (readable
via ``read``). Integration always runs from t=0 each step (no carried simulation time), so
models with explicit time dependence (the ``time`` csymbol, timed events) are out of scope.
Reuses viva-tellurium's RoadRunner build path.
"""
from __future__ import annotations


class TelluriumModule:
    def __init__(self, model=None, params=None):
        from viva_tellurium.processes import _load_roadrunner

        model, params = model or {}, params or {}
        src, fmt, file = model.get("source", ""), model.get("format", "antimony"), model.get("file", "")
        if not src and not file:
            raise ValueError("tellurium module needs model.source (Antimony/SBML) or model.file")
        self._rr = _load_roadrunner(src, fmt, file)
        integrator = params.get("integrator", "cvode")
        if integrator != "cvode":
            self._rr.setIntegrator(integrator)
        else:
            self._rr.integrator.absolute_tolerance = float(params.get("absolute_tolerance", 1e-10))
            self._rr.integrator.relative_tolerance = float(params.get("relative_tolerance", 1e-8))
        if params.get("seed") is not None and hasattr(self._rr.integrator, "seed"):
            self._rr.integrator.seed = int(params["seed"])
        rr = self._rr
        self._species = list(rr.getFloatingSpeciesIds())
        self._settable_species = self._species + list(rr.getBoundarySpeciesIds())
        self._globals = list(rr.getGlobalParameterIds())
        ruled = set(rr.getAssignmentRuleIds())
        self._settable_globals = [g for g in self._globals if g not in ruled]
        self._rxns = list(rr.getReactionIds())
        self._overrides = {k: float(v) for k, v in (params.get("parameters") or {}).items()}
        self._defaults = {p: float(rr[p]) for p in self._settable_globals}
        self._init_species = {s: float(rr[s]) for s in self._species}

    # -- protocol ---------------------------------------------------------
    def input_vars(self):
        return set(self._settable_species) | set(self._settable_globals)

    def output_vars(self):
        return set(self._species) | set(self._globals) | set(self._rxns)

    def init_cell(self, cell_id, inputs):
        return {"species": dict(self._init_species), "params": {}, "time": 0.0, "derived": {},
                "rates": self._rates_for(self._init_species, {})}

    def _load(self, state, extra=None):
        rr = self._rr
        rr.resetAll()
        for p, v in {**self._defaults, **self._overrides, **state["params"], **(extra or {})}.items():
            rr[p] = v
        for s, v in state["species"].items():
            rr[s] = v

    def _rates_for(self, species, params):
        self._load({"species": species, "params": params})
        return dict(zip(self._rxns, map(float, self._rr.getReactionRates())))

    def step(self, state, mapped_inputs, interval):
        params = dict(state["params"])
        species = dict(state["species"])
        for k, v in (mapped_inputs or {}).items():
            if k in self._settable_globals:
                params[k] = float(v)
            elif k in self._species:
                species[k] = float(v)
            elif k in self._settable_species:
                params[k] = float(v)
            else:
                raise ValueError(f"unknown tellurium input {k!r}")
        rr = self._rr
        self._load({"species": species, "params": params})
        # boundary species are set via params dict above (rr[k] = v works for both)
        if interval > 0:
            rr.simulate(0.0, float(interval), 2)
        new_species = {s: float(rr[s]) for s in self._species}
        rates = dict(zip(self._rxns, map(float, rr.getReactionRates())))
        return {"species": new_species, "params": params,
                "derived": {g: float(rr[g]) for g in self._globals if g not in self._defaults},
                "time": state["time"] + float(interval), "rates": rates}

    def read(self, state):
        out = {s: float(v) for s, v in state["species"].items()}
        for p in self._globals:
            if p in self._defaults:
                out[p] = float(state["params"].get(p, self._overrides.get(p, self._defaults[p])))
            elif p in state.get("derived", {}):
                out[p] = float(state["derived"][p])
        out.update(state["rates"])
        return out

    def split(self, state):
        return {**state, "species": {s: v / 2.0 for s, v in state["species"].items()},
                "params": dict(state["params"]), "derived": dict(state.get("derived", {})), "rates": dict(state["rates"])}
