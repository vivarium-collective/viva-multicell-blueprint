"""Compile layer: the MechanicsBackend contract + the per-backend compilers.

``compile(blueprint, backend)`` lowers one backend-neutral blueprint onto a
runnable process-bigraph composite. The shared mechanism (field/rules/ODE/immune)
is wired identically on both backends; only the single mechanics node differs
(spec section 5).
"""
from __future__ import annotations

_COMPILERS = {"cpm": "to_cpm", "vertex": "to_vertex"}


def compile(blueprint, backend: str = "cpm") -> dict:
    """Compile ``blueprint`` onto ``backend`` (``"cpm"`` or ``"vertex"``).

    Returns a process-bigraph composite document ``{"state": {...}}``.
    """
    if backend not in _COMPILERS:
        raise ValueError(f"unknown backend {backend!r}; expected one of {sorted(_COMPILERS)}")
    import importlib

    mod = importlib.import_module(f"{__name__}.{_COMPILERS[backend]}")
    return mod.compile(blueprint, backend=backend)


__all__ = ["compile"]
