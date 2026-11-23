"""Bind each scene cell to the ModelSpec of its type."""

from __future__ import annotations

from viva_multicell_blueprint.model_format.registry import ModelSpec, Registry
from viva_multicell_blueprint.model_format.scene import Scene


def bind(scene: Scene, registry: Registry) -> dict[str, ModelSpec]:
    """Return {cell_id: ModelSpec}; raises UnboundCellType for an unknown type."""
    return {cell_id: registry.get(cell.type) for cell_id, cell in scene.cells.items()}
