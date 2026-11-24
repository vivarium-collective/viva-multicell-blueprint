"""Per-type model specs and the registry that resolves cell types to them."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class UnboundCellType(KeyError):
    """A cell type has no ModelSpec in the registry."""


@dataclass(frozen=True)
class Transport:
    """How one species crosses the membrane: reflect, cross, or transporter."""

    kind: str
    rate: float = 0.0


@dataclass(frozen=True)
class ModelSpec:
    """Formalism-tagged model for one cell type."""

    formalism: str
    source: str | dict[str, Any]
    params: dict[str, Any] = field(default_factory=dict)
    membrane: dict[str, Transport] = field(default_factory=dict)


class Registry:
    """Maps cell type name to its ModelSpec."""

    def __init__(self, models: dict[str, ModelSpec]):
        self._models = dict(models)

    def get(self, type: str) -> ModelSpec:
        try:
            return self._models[type]
        except KeyError:
            raise UnboundCellType(type) from None
