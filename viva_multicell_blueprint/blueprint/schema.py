"""JSON-schema validation + BP-01 structured diagnostics for blueprint documents."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from jsonschema import Draft202012Validator

from .canonical import canonical_violations

SCHEMA_PATH = Path(__file__).resolve().parents[2] / ".pbg" / "schemas" / "multicell_blueprint.schema.json"


@lru_cache(maxsize=1)
def _validator() -> Draft202012Validator:
    return Draft202012Validator(json.loads(SCHEMA_PATH.read_text()))


def validate(doc) -> list[str]:
    """Return a list of human-readable schema ERRORS; empty means schema-valid.

    Back-compat surface (errors only). New code should prefer
    :func:`validate_blueprint`, which also returns warnings.
    """
    errs = sorted(_validator().iter_errors(doc), key=lambda e: list(map(str, e.path)))
    return [f"{'/'.join(map(str, e.absolute_path)) or '<root>'}: {e.message}" for e in errs]


def validate_blueprint(doc) -> list[dict]:
    """Return structured diagnostics ``[{severity, path, message}]`` (BP-01).

    - ``error``   — JSON-schema violations (the blueprint will not compile reliably);
    - ``warning`` — the canonical contract: a missing ``blueprint_version`` and any
      backend/run parameter found inside the canonical biological blueprint.

    An empty list means schema-valid AND contract-clean. Errors and warnings are
    both surfaced so the validator exposes problems, not just a success badge.
    """
    diags: list[dict] = []
    for e in sorted(_validator().iter_errors(doc), key=lambda e: list(map(str, e.path))):
        diags.append({
            "severity": "error",
            "path": "/".join(map(str, e.absolute_path)) or "<root>",
            "message": e.message,
        })
    if isinstance(doc, dict):
        if not doc.get("blueprint_version"):
            diags.append({
                "severity": "warning",
                "path": "blueprint_version",
                "message": ("no blueprint_version declared; set it so the canonical "
                            "schema version is auditable (BP-01)."),
            })
        diags.extend(canonical_violations(doc))
    return diags
