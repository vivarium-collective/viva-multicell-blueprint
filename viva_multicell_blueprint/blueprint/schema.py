"""JSON-schema validation for blueprint documents."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from jsonschema import Draft202012Validator

SCHEMA_PATH = Path(__file__).resolve().parents[2] / ".pbg" / "schemas" / "multicell_blueprint.schema.json"


@lru_cache(maxsize=1)
def _validator() -> Draft202012Validator:
    return Draft202012Validator(json.loads(SCHEMA_PATH.read_text()))


def validate(doc) -> list[str]:
    """Return a list of human-readable errors; empty means valid."""
    errs = sorted(_validator().iter_errors(doc), key=lambda e: list(map(str, e.path)))
    return [f"{'/'.join(map(str, e.absolute_path)) or '<root>'}: {e.message}" for e in errs]
