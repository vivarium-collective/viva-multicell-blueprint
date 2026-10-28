"""Compile layer: the MechanicsBackend contract + the per-backend compilers.

``compile(blueprint, backend)`` lowers one backend-neutral blueprint onto a
runnable process-bigraph composite. The shared mechanism (field/rules/ODE/immune)
is wired identically on both backends; only the single mechanics node differs
(spec section 5).
"""
from __future__ import annotations

_COMPILERS = {"cpm": "to_cpm", "vertex": "to_vertex"}


def compile(blueprint, backend: str = "cpm", options=None) -> dict:
    """Compile ``blueprint`` onto ``backend`` (``"cpm"`` or ``"vertex"``).

    ``options`` is an optional :class:`~viva_multicell_blueprint.blueprint.canonical.CompileOptions`
    carrying backend/run parameters (``tension_scale``, ``rules_seed``) that are kept
    OUT of the canonical blueprint (BP-01). When given, they override any legacy
    values still in ``blueprint.mechanics_mapping``; the canonical blueprint itself
    is never mutated. Returns a process-bigraph composite document ``{"state": {...}}``.
    """
    if backend not in _COMPILERS:
        raise ValueError(f"unknown backend {backend!r}; expected one of {sorted(_COMPILERS)}")
    if options is not None:
        import dataclasses
        merged = {**(blueprint.mechanics_mapping or {}), **options.as_mechanics_mapping()}
        blueprint = dataclasses.replace(blueprint, mechanics_mapping=merged)
    import importlib

    mod = importlib.import_module(f"{__name__}.{_COMPILERS[backend]}")
    return mod.compile(blueprint, backend=backend)


__all__ = ["compile"]
