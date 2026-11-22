"""Compile layer: the MechanicsBackend contract + the single compile path.

``compile(blueprint, backend)`` lowers one backend-neutral blueprint onto a
runnable process-bigraph composite. The shared mechanism (field/rules/ODE/immune,
and the optional subcellular layer) is wired identically on every backend; only
the single mechanics node differs (spec section 5). Backend dispatch lives in ONE
registry, :data:`.builders.MECHANICS_NODE_BUILDERS`.
"""
from __future__ import annotations

from . import builders
from ._shared import assemble


def compile(blueprint, backend: str = "cpm", options=None, video_out=None) -> dict:
    """Compile ``blueprint`` onto ``backend`` (``"cpm"``, ``"vertex"``, or ``"munk"``).

    ``options`` is an optional :class:`~viva_multicell_blueprint.blueprint.canonical.CompileOptions`
    carrying backend/run parameters (``tension_scale``, ``rules_seed``) that are kept
    OUT of the canonical blueprint (BP-01). When given, they override any legacy
    values still in ``blueprint.mechanics_mapping``; the canonical blueprint itself
    is never mutated. Returns a process-bigraph composite document ``{"state": {...}}``.

    A blueprint with a ``subcellular`` layer gets the per-cell SubcellularProcess
    nodes + the actuation merge step wired between Rules and the mechanics node.
    """
    if backend not in builders.MECHANICS_NODE_BUILDERS:
        raise ValueError(
            f"unknown backend {backend!r}; expected one of {sorted(builders.MECHANICS_NODE_BUILDERS)}"
        )
    if options is not None:
        import dataclasses
        merged = {**(blueprint.mechanics_mapping or {}), **options.as_mechanics_mapping()}
        blueprint = dataclasses.replace(blueprint, mechanics_mapping=merged)
    return assemble(
        blueprint, builders.build_mechanics(blueprint, backend),
        video_out=video_out, backend=backend,
        subcellular_nodes=builders.build_subcellular(blueprint),
        merge_node=builders.build_merge(blueprint),
    )


__all__ = ["compile"]
