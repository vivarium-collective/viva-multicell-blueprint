"""Scene: where cells sit in the domain and which type each one is."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import yaml


@dataclass(frozen=True)
class CellSpec:
    type: str
    position: list[float]
    molecule_levels: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Scene:
    cells: dict[str, CellSpec]
    domain: dict[str, Any] = field(default_factory=dict)

    def to_yaml(self) -> str:
        return yaml.safe_dump(
            {
                "domain": self.domain,
                "cells": {
                    cell_id: {
                        "type": cell.type,
                        "position": list(cell.position),
                        "molecule_levels": dict(cell.molecule_levels),
                    }
                    for cell_id, cell in self.cells.items()
                },
            },
            sort_keys=False,
        )

    @classmethod
    def from_yaml(cls, text: str) -> Scene:
        data = yaml.safe_load(text)
        return cls(
            domain=data.get("domain", {}),
            cells={
                cell_id: CellSpec(
                    type=raw["type"],
                    position=list(raw["position"]),
                    molecule_levels=dict(raw.get("molecule_levels", {})),
                )
                for cell_id, raw in data["cells"].items()
            },
        )
