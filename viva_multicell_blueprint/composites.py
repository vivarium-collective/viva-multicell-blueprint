"""Workspace composite generator (spec §5/§9): the Sego blueprint compiled onto a
mechanics backend, exposed to the dashboard's Composite Explorer and used as the
baseline composite of the ``influenza-crossparadigm`` studies.

One generator, parameterized by ``backend`` (``cpm``|``vertex``) and the
``blueprint`` path, so the same runnable artifact serves the full flu-lung study
and the mechanics-only adhesion-calibration sheet. It simply ``compile()``s the
backend-neutral blueprint -- the single source of truth -- so the Explorer shows
the real shared mechanism + the one backend-specific mechanics node.
"""
from __future__ import annotations

from pathlib import Path

from process_bigraph.composite_generator import composite_generator

from .blueprint.model import Blueprint
from .compile import compile as _compile
from .core import register_workspace_processes

_BLUEPRINTS = Path(__file__).resolve().parent.parent / "workspace" / "blueprints"
_SEGO = _BLUEPRINTS / "influenza_sego2022.yaml"


@composite_generator(
    name="sego_crossparadigm",
    description=(
        "Sego 2022 influenza blueprint compiled onto a mechanics backend "
        "(cpm|vertex): the shared field/rules/ODE/recruitment mechanism plus the "
        "one backend-specific mechanics node."
    ),
    parameters={
        "backend": {"type": "string", "default": "cpm",
                    "description": "mechanics substrate: 'cpm' or 'vertex'"},
        "blueprint": {"type": "string", "default": str(_SEGO),
                      "description": "path to the backend-neutral blueprint YAML"},
    },
    core_extensions=[register_workspace_processes],
    default_n_steps=80,
)
def sego_crossparadigm(core=None, backend: str = "cpm", blueprint: str | None = None) -> dict:
    bp = Blueprint.load(blueprint or _SEGO)
    return _compile(bp, backend=backend)        # {"state": {...}} envelope
