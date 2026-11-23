"""``CoupledCell`` — a cell whose intracellular model is itself a coupled composite.

Every other adapter wraps ONE formalism. ``CoupledCell`` wraps SEVERAL sub-models (each
its own formalism — ODE / particle / Boolean) plus explicit coupling rules between them,
and is itself an :class:`IntracellularAdapter`, so it drops transparently into the
per-cell slot of :class:`~..compile.SceneModelProcess`. A cell is thus a *coupled
composite of sub-models* — the same semantics⊥mechanics composition the workspace does at
the tissue scale, recursing one scale down.

Three coupling directions (proven standalone in ``scripts/spike_coupled_cell*.py``):

  * **mode A** — a source species crossing a threshold pins a target Boolean input node
    (ODE metabolite → fate signal). Applied *before* the step (prepares the target input).
  * **mode B** — a source species count sets a target ODE rate constant (particle count →
    catalysed metabolism), via the live roadrunner the ODE adapter holds. Applied *before*
    the step (the rate governs that step).
  * **mode C** — a source Boolean node, when high, fires an action on a target (fate
    ``divide`` resets the metabolism). Applied *after* the step (gated on fresh output).

Source ``ModelSpec``::

    ModelSpec(formalism="coupled", source={
        "models": {
            "<name>": {"formalism": "ode"|"particle"|"boolean",
                       "source": <adapter source>, "params": {...}, "initial": {...}},
            ...
        },
        "couplings": [
            {"mode": "A", "from": <name>, "species": <sp>, "threshold": <float>,
             "to": <name>, "input": <node>},
            {"mode": "B", "from": <name>, "species": <sp>,
             "to": <name>, "param": "kf0", "base": <float>},
            {"mode": "C", "from": <name>, "node": <node>, "to": <name>, "action": "reset"},
        ],
    })

``amounts()`` is namespaced ``"<model>/<species>"`` so sub-model species never collide.
"""
from __future__ import annotations

from typing import Any

from ..registry import ModelSpec
from .base import IntracellularAdapter
from .boolean_maboss import BooleanAdapter
from .ode_tellurium import ODEAdapter
from .particle_smoldyn import ParticleAdapter

# sub-model formalism token -> adapter class (local map; avoids importing compile.py)
_SUB_ADAPTER: dict[str, type[IntracellularAdapter]] = {
    "ode": ODEAdapter, "sbml": ODEAdapter, "tellurium": ODEAdapter,
    "particle": ParticleAdapter, "particles": ParticleAdapter, "smoldyn": ParticleAdapter,
    "boolean": BooleanAdapter, "bool": BooleanAdapter, "maboss": BooleanAdapter,
}


def _sub_class(formalism: str) -> type[IntracellularAdapter]:
    key = str(formalism).strip().lower()
    try:
        return _SUB_ADAPTER[key]
    except KeyError:
        raise ValueError(f"unknown sub-model formalism {formalism!r}; "
                         f"have {sorted(set(_SUB_ADAPTER))}") from None


def _build_sub(model: dict[str, Any], overrides: dict[str, float]):
    """Construct one sub-adapter from a ``models[name]`` dict + initial-level overrides."""
    cls = _sub_class(model["formalism"])
    spec = ModelSpec(
        formalism=str(model["formalism"]),
        source=model["source"],
        params=dict(model.get("params") or {}),
        membrane=dict(model.get("membrane") or {}),
    )
    initial = {**(model.get("initial") or {}), **overrides}
    if cls is ParticleAdapter:
        return cls(spec, initial, seed=int(model.get("seed", 0)))
    return cls(spec, initial)


class CoupledCell(IntracellularAdapter):
    """Several coupled sub-models inside one cell; itself an :class:`IntracellularAdapter`."""

    def __init__(self, model_spec: ModelSpec, initial_levels: dict[str, float]):
        self.model_spec = model_spec
        src = model_spec.source
        if not isinstance(src, dict) or "models" not in src:
            raise ValueError("CoupledCell source must be a dict with a 'models' key")
        self._model_defs: dict[str, dict] = dict(src["models"])
        self._couplings: list[dict] = list(src.get("couplings") or [])
        # top-level initial_levels override sub-model initials, keyed "<model>/<species>"
        per_model: dict[str, dict[str, float]] = {name: {} for name in self._model_defs}
        for key, val in (initial_levels or {}).items():
            if "/" in key:
                m, sp = key.split("/", 1)
                if m in per_model:
                    per_model[m][sp] = float(val)
        self._subs: dict[str, IntracellularAdapter] = {
            name: _build_sub(mdef, per_model[name])
            for name, mdef in self._model_defs.items()
        }
        self._events: dict[str, float] = {}  # last-step coupling events (e.g. {"divide": 1})

    # ---- coupling primitives -------------------------------------------------
    def _pre_step_couplings(self) -> dict[str, dict[str, Any]]:
        """Apply mode A/B (prepare target inputs / set target rate). Returns pending inputs."""
        pending: dict[str, dict[str, Any]] = {name: {} for name in self._subs}
        for c in self._couplings:
            mode = str(c.get("mode", "")).upper()
            src = self._subs.get(c.get("from"))
            tgt = self._subs.get(c.get("to"))
            if src is None or tgt is None:
                continue
            if mode == "A":
                level = src.amounts().get(c["species"], 0.0)
                signal = 1.0 if level > float(c.get("threshold", 0.0)) else 0.0
                pending[c["to"]][c["input"]] = signal
            elif mode == "B":
                count = src.amounts().get(c["species"], 0.0)
                rr = getattr(tgt, "_rr", None)
                param = c.get("param", "kf0")
                if rr is not None:
                    try:
                        rr[param] = float(c.get("base", 1.0)) * float(count)
                    except Exception:  # pragma: no cover - param absent/engine quirk
                        pass
        return pending

    def _post_step_couplings(self) -> None:
        """Apply mode C (gated actions on fresh output), e.g. reset a target sub-model."""
        self._events = {}
        for c in self._couplings:
            if str(c.get("mode", "")).upper() != "C":
                continue
            src = self._subs.get(c.get("from"))
            if src is None or c.get("to") not in self._subs:
                continue
            if src.amounts().get(c["node"], 0.0) > 0.5:
                if str(c.get("action", "reset")) == "reset":
                    self._subs[c["to"]] = _build_sub(self._model_defs[c["to"]], {})
                    self._events[c["node"]] = 1.0

    # ---- IntracellularAdapter contract --------------------------------------
    def step(self, dt: float, inputs: dict[str, Any]) -> None:
        pending = self._pre_step_couplings()
        for name, sub in self._subs.items():
            sub.step(dt, {**(inputs or {}), **pending[name]})
        self._post_step_couplings()

    def amounts(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for name, sub in self._subs.items():
            for sp, v in sub.amounts().items():
                out[f"{name}/{sp}"] = v
        return out

    def crossed(self) -> dict[str, float]:
        merged: dict[str, float] = {}
        for sub in self._subs.values():
            for sp, v in sub.crossed().items():
                merged[sp] = merged.get(sp, 0.0) + v
        return merged

    def available(self) -> bool:
        return all(sub.available() for sub in self._subs.values())

    # ---- introspection (viz / tests) ----------------------------------------
    @property
    def subs(self) -> dict[str, IntracellularAdapter]:
        return dict(self._subs)

    @property
    def events(self) -> dict[str, float]:
        """Coupling events fired on the last step (e.g. {'divide': 1.0})."""
        return dict(self._events)


__all__ = ["CoupledCell"]
