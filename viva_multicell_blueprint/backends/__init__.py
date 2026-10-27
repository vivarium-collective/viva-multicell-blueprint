"""Mechanics backends satisfying the MechanicsBackend contract.

The concrete backend classes are re-exported here so the workspace core's
registration walk (``core.build_core`` -> ``register_workspace_processes``, which
imports each top-level module/subpackage and inspects its namespace) discovers
them and registers ``local:CPMBackend`` / ``local:VertexBackend`` / ``local:FakeBackend``.
Compiled composites address the backends by that ``local:<name>`` form, so without
these exports the backends live in un-walked submodules and the composite fails to
build with "no link found at address".
"""
from __future__ import annotations

from viva_multicell_blueprint.backends.cpm_backend import CPMBackend
from viva_multicell_blueprint.backends.fake_backend import FakeBackend
from viva_multicell_blueprint.backends.vertex_backend import VertexBackend

__all__ = ["CPMBackend", "VertexBackend", "FakeBackend"]
