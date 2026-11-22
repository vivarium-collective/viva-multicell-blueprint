"""Blueprint v2 (initial_conditions + model_definition) -> flat v1 shim."""
from __future__ import annotations

_SECTIONS = ("initial_conditions", "model_definition")


def normalize_to_flat(doc: dict) -> dict:
    """Return a flat v1-shaped dict from a flat v1 or sectioned v2 document.

    A v1 doc is returned as a shallow copy. In a v2 doc, the keys of
    ``initial_conditions`` and ``model_definition`` are hoisted to the top
    level, with ``model_definition.systemic`` renamed to ``systemic_ode``.
    A doc mixing v1 top-level keys with the v2 sections raises ``ValueError``.
    """
    if not isinstance(doc, dict) or not any(s in doc for s in _SECTIONS):
        return dict(doc) if isinstance(doc, dict) else doc
    extra = sorted(k for k in doc if k not in _SECTIONS and k != "blueprint_version")
    if extra:
        raise ValueError(f"mixed v1/v2 blueprint: v1 keys {extra} alongside {list(_SECTIONS)}")
    flat: dict = {}
    if "blueprint_version" in doc:
        flat["blueprint_version"] = doc["blueprint_version"]
    for sec in _SECTIONS:
        body = doc.get(sec) or {}
        if not isinstance(body, dict):
            raise ValueError(f"{sec} must be a mapping")
        for k, v in body.items():
            k = "systemic_ode" if (sec == "model_definition" and k == "systemic") else k
            if k in flat:
                raise ValueError(f"duplicate key {k!r} across v2 sections")
            flat[k] = v
    return flat
