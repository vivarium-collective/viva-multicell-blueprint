"""BP-01: the canonical/backend separation contract.

The canonical blueprint is the *biological* specification: it must be identical
for every backend and must not carry backend-specific geometry/dynamics or
run-level parameters (reviewer acceptance). Those live in :class:`CompileOptions`,
recorded separately from the blueprint.

Today the only such parameters ride in ``mechanics_mapping`` — ``tension_scale``
(a vertex line-tension knob) and ``rules_seed`` (a stochastic run seed). Genuine
model constants (``const`` / ``rules_const``) are NOT backend-specific and stay in
the canonical blueprint. ``split_canonical`` separates the two; ``CompileOptions``
carries the separated part; ``canonical_violations`` reports (as warnings) any
backend/run parameter found inside a blueprint so the contract is *checked*, not
merely asserted.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

# Keys that are backend-specific (geometry/dynamics) or run-level, and therefore
# must NOT appear in a canonical biological blueprint.
BACKEND_RUN_KEYS = {"tension_scale", "rules_seed", "migration_scale"}


@dataclass
class CompileOptions:
    """Backend/run parameters, kept OUT of the canonical blueprint.

    ``tension_scale`` (vertex only) and ``rules_seed`` (stochastic run) are the
    parameters the review flagged as non-canonical. ``extra`` carries any further
    recognized backend/run keys for forward-compatibility.
    """
    rules_seed: int | None = None
    tension_scale: float | None = None
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, mm: dict | None) -> "CompileOptions":
        mm = mm or {}
        extra = {k: v for k, v in mm.items() if k in BACKEND_RUN_KEYS
                 and k not in ("rules_seed", "tension_scale")}
        return cls(rules_seed=mm.get("rules_seed"), tension_scale=mm.get("tension_scale"),
                   extra=extra)

    def as_mechanics_mapping(self) -> dict:
        """The backend/run keys as a ``mechanics_mapping`` fragment (only set keys)."""
        out = dict(self.extra)
        if self.rules_seed is not None:
            out["rules_seed"] = self.rules_seed
        if self.tension_scale is not None:
            out["tension_scale"] = self.tension_scale
        return out


def split_canonical(doc: dict) -> tuple[dict, dict]:
    """Split a blueprint dict into ``(canonical_doc, backend_run_params)``.

    ``canonical_doc`` is backend-neutral and immutable across backends; genuine
    model constants under ``mechanics_mapping`` are preserved. ``backend_run_params``
    is the extracted ``{tension_scale, rules_seed, ...}`` fragment.
    """
    d = copy.deepcopy(doc)
    mm = d.get("mechanics_mapping") or {}
    opts = {k: mm[k] for k in BACKEND_RUN_KEYS if k in mm}
    rest = {k: v for k, v in mm.items() if k not in BACKEND_RUN_KEYS}
    if rest:
        d["mechanics_mapping"] = rest
    else:
        d.pop("mechanics_mapping", None)
    return d, opts


def canonical_violations(doc: dict) -> list[dict]:
    """Warnings for backend/run parameters present in a (nominally canonical)
    blueprint. Returns ``[{severity, path, message}]`` (possibly empty)."""
    mm = (doc or {}).get("mechanics_mapping") or {}
    out = []
    for k in sorted(BACKEND_RUN_KEYS & set(mm)):
        out.append({
            "severity": "warning",
            "path": f"mechanics_mapping/{k}",
            "message": (f"'{k}' is a backend/run parameter and should not live in the "
                        "canonical biological blueprint; pass it via CompileOptions "
                        "(BP-01 canonical contract)."),
        })
    return out
