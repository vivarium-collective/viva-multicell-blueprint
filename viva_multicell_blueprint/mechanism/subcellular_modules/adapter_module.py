"""Bridge the model_format IntracellularAdapters to the SubcellularModule protocol.

The composition-atlas already runs real intracellular simulators per cell through
``model_format.adapters`` — ``BooleanAdapter`` (MaBoSS / cmaboss) and
``ParticleAdapter`` (Smoldyn) — each built from a :class:`ModelSpec` and stepped
with ``step(dt, inputs)`` / ``amounts()``. The blueprint pipeline's
``SubcellularProcess`` keeps per-cell state in-process and threads it through
``step(state, mapped_inputs, interval)`` / ``read(state)`` / ``split(state)``, so
the per-cell *state* can simply BE a live adapter. These thin wrappers expose the
atlas adapters as ``MODULES`` entries — no re-implementation of either engine.

``input_vars`` / ``output_vars`` are the adapter's own variable names (its
``amounts()`` keys: boolean network nodes, particle species). A coupling input
pins/sets that variable; a coupling output reads it. ``split`` seeds a fresh
adapter from the mother's halved levels (extensive counts halve; boolean
probabilities carry over). When the engine is not installed the adapter degrades
to holding its levels (``available() == False``), so the run still completes.
"""
from __future__ import annotations


def _model_spec(model, params, formalism):
    from viva_multicell_blueprint.model_format.registry import ModelSpec
    model = model or {}
    source = model.get("source", model)        # {source: ...} or the mapping itself
    return ModelSpec(formalism=formalism, source=source, params=dict(params or {}))


class _AdapterModule:
    """A SubcellularModule whose per-cell state is a live IntracellularAdapter."""

    _FORMALISM = ""

    def _adapter_cls(self):            # subclasses import + return their adapter class
        raise NotImplementedError

    def __init__(self, model=None, params=None):
        self._cls = self._adapter_cls()
        self._spec = _model_spec(model, params, self._FORMALISM)
        self._vars = set(self._cls(self._spec, {}).amounts().keys())   # probe the var names

    def init_cell(self, cell_id, inputs):
        return self._cls(self._spec, {})

    def step(self, state, mapped_inputs, interval):
        state.step(float(interval), {k: float(v) for k, v in (mapped_inputs or {}).items()})
        return state

    def read(self, state):
        return {k: float(v) for k, v in state.amounts().items()}

    def split(self, state):
        halved = {k: float(v) / 2.0 for k, v in state.amounts().items()}
        return self._cls(self._spec, halved)

    def input_vars(self):
        return set(self._vars)

    def output_vars(self):
        return set(self._vars)


class BooleanModule(_AdapterModule):
    """Per-cell boolean network stepped by MaBoSS (cmaboss)."""
    _FORMALISM = "boolean"

    def _adapter_cls(self):
        from viva_multicell_blueprint.model_format.adapters.boolean_maboss import BooleanAdapter
        return BooleanAdapter


class ParticleModule(_AdapterModule):
    """Per-cell stochastic particle reactions stepped by Smoldyn."""
    _FORMALISM = "particle"

    def _adapter_cls(self):
        from viva_multicell_blueprint.model_format.adapters.particle_smoldyn import ParticleAdapter
        return ParticleAdapter
