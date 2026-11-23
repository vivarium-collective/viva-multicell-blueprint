"""Declarative model-definition format: scene + registry + binding + compile.

``SceneModelProcess`` is re-exported here so the workspace core's registration walk
(``core.build_core`` -> ``register_workspace_processes``) discovers it and registers
``local:SceneModelProcess``; compiled scene composites address it by that form.
"""

from viva_multicell_blueprint.model_format.compile import SceneModelProcess, compile_scene

__all__ = ["SceneModelProcess", "compile_scene"]
