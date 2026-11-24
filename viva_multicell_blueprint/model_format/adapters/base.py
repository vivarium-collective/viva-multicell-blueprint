"""Common interface every intracellular formalism adapter implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..registry import ModelSpec


class IntracellularAdapter(ABC):
    """One cell's intracellular model, stepped by the compiler.

    `inputs` is a plain dict carrying extracellular concentrations at the
    cell plus mechanical state; each adapter reads what it needs.
    """

    @abstractmethod
    def __init__(self, model_spec: ModelSpec, initial_levels: dict[str, float]):
        ...

    @abstractmethod
    def step(self, dt: float, inputs: dict[str, Any]) -> None:
        """Advance the model by dt."""

    @abstractmethod
    def amounts(self) -> dict[str, float]:
        """Current species amounts."""

    @abstractmethod
    def crossed(self) -> dict[str, float]:
        """Species that left via the membrane this step ({} if none)."""

    @abstractmethod
    def available(self) -> bool:
        """Whether the underlying engine/model can run."""


class NullAdapter(IntracellularAdapter):
    """Holds its initial levels; no dynamics. Reference implementation."""

    def __init__(self, model_spec: ModelSpec, initial_levels: dict[str, float]):
        self.model_spec = model_spec
        self._amounts = dict(initial_levels)

    def step(self, dt: float, inputs: dict[str, Any]) -> None:
        return None

    def amounts(self) -> dict[str, float]:
        return dict(self._amounts)

    def crossed(self) -> dict[str, float]:
        return {}

    def available(self) -> bool:
        return True
