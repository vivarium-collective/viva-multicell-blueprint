"""Backend-neutral multi-cell Blueprint document (spec section 7)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from .schema import validate


_REQUIRED = ("domain", "cell_types", "initial_state")


@dataclass
class Blueprint:
    domain: dict
    cell_types: list
    initial_state: dict
    blueprint_version: str | None = None
    adhesion: list = field(default_factory=list)
    fields: list = field(default_factory=list)
    rules: list = field(default_factory=list)
    immune: list | dict = field(default_factory=list)
    systemic_ode: dict | None = None
    mechanics_mapping: dict = field(default_factory=dict)

    def canonical_state(self) -> dict:
        """The backend-neutral, immutable biological blueprint (BP-01).

        Strips backend/run parameters (``tension_scale``, ``rules_seed``, …) so the
        result is identical for every backend; genuine model constants are kept.
        """
        from .canonical import split_canonical
        canon, _ = split_canonical(self.to_dict())
        return canon

    @classmethod
    def from_dict(cls, doc: dict) -> "Blueprint":
        errs = validate(doc)
        if errs:
            raise ValueError("invalid blueprint: " + "; ".join(errs))
        unknown = sorted(set(doc) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"unknown top-level blueprint keys: {unknown}")
        return cls(**doc)

    def to_dict(self) -> dict:
        # Omit absent/default-empty optional sections so load->dump is faithful.
        return {k: v for k, v in asdict(self).items() if v is not None and (v or k in _REQUIRED)}

    @classmethod
    def load(cls, path) -> "Blueprint":
        return cls.from_dict(yaml.safe_load(Path(path).read_text()))

    def dump(self, path) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))
