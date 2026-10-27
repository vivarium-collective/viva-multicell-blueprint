"""Adhesion lowering: neutral type-pair ``strength`` -> each backend's native form.

The blueprint ``adhesion`` section is a list of ``{a, b, strength}`` (type
names; ``medium`` is the background). Backends use different energy forms:

* **CPM** (viva-cpm): contact energy ``J(a,b)`` on integer type tags, supplied
  as ``contact: [{a, b, j}]``. Mapping is DIRECT: ``j = strength`` (the neutral
  strength IS a contact energy; higher = less adhesive). Faithful: Sego 2022
  Table 3 values (epithelial-epithelial 5, epithelial-medium 25) pass through
  unchanged. Type names -> ints: ``medium`` is tag 0 (CPM background),
  other names are numbered 1.. in first-appearance order unless an explicit
  ``type_ids`` map is given.
* **Vertex** (tyssue): line tension ``Lambda`` on each EDGE whose two incident
  faces are of types (a,b); edges on the tissue boundary pair a face type with
  ``medium``. Mapping: ``tension = scale * strength`` (default ``scale=1``),
  keyed by the unordered (sorted) type-name pair.

What is faithful vs. calibrated: the J mapping is exact; the tension mapping
is only the *principled monotone* one (interfacial tension rises with contact
energy). ``J`` (energy per contact unit) and ``Lambda`` (energy per edge
length) have different units and geometry, so ``scale`` is NOT pinned here;
the ``adhesion-calibration`` study (spec §9.1) fits it so mechanics-only
sheets match in cell-shape statistics on both backends.

Pairs are unordered: a reversed duplicate is merged if equal, else ValueError.
"""
from __future__ import annotations

MEDIUM = "medium"


def _pairs(adhesion) -> dict:
    out: dict = {}
    for e in adhesion:
        key = tuple(sorted((e["a"], e["b"])))
        s = float(e["strength"])
        if key in out and out[key] != s:
            raise ValueError(f"conflicting strengths for pair {key}: {out[key]} vs {s}")
        out[key] = s
    return out


def to_cpm_contact(adhesion, type_ids: dict | None = None) -> list[dict]:
    """Return CPM ``contact`` list ``[{a, b, j}]`` (ints a,b), one per unordered pair."""
    if type_ids is None:
        type_ids = {MEDIUM: 0}
        for e in adhesion:
            for n in (e["a"], e["b"]):
                if n not in type_ids:
                    type_ids[n] = len(type_ids)
    out = []
    seen = set()
    for e in adhesion:
        key = tuple(sorted((e["a"], e["b"])))
        if key in seen:
            _pairs(adhesion)  # raises on conflict
            continue
        seen.add(key)
        out.append({"a": type_ids[e["a"]], "b": type_ids[e["b"]], "j": float(e["strength"])})
    _pairs(adhesion)
    return out


def to_vertex_tension(adhesion, scale: float = 1.0) -> dict:
    """Return ``{(type_a, type_b): line_tension}`` keyed by sorted type-name pair."""
    return {k: scale * s for k, s in _pairs(adhesion).items()}
