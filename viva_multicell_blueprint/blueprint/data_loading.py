"""Load the canonical, backend-neutral cell list from data (spec section 11).

Both loaders emit ``[Cell]`` with ``centroid = (x, y)`` (x = column, y = row,
pixel units for label images). Each backend materializes the same list.
"""
from __future__ import annotations

from typing import TypedDict

import pandas as pd
from skimage.io import imread
from skimage.measure import regionprops

_TABLE_COLUMNS = ("id", "x", "y", "type", "size")
DEFAULT_TYPE = "cell"


class Cell(TypedDict, total=False):
    id: int
    type: str
    centroid: tuple[float, float]
    size: float
    footprint: list  # optional; not produced by the loaders


def from_label_image(path, type_map: dict | None = None) -> list[Cell]:
    """Cells from an integer label image; label 0 is medium, not a cell.

    ``type_map`` maps label -> type name; if None every cell is ``"cell"``.
    If given, every label present must be mapped (KeyError otherwise).
    ``size`` is the pixel count of the label (holes are not counted).
    """
    labels = imread(path)
    if labels.ndim == 3:
        labels = labels[..., 0]
    labels = labels.astype(int)
    cells: list[Cell] = []
    for rp in regionprops(labels):  # skips label 0; sorted by label
        cy, cx = rp.centroid
        if type_map is None:
            typ = DEFAULT_TYPE
        elif rp.label in type_map:
            typ = type_map[rp.label]
        else:
            raise KeyError(f"label {rp.label} not in type_map")
        cells.append(Cell(id=int(rp.label), type=str(typ),
                          centroid=(float(cx), float(cy)), size=float(rp.area)))
    return cells


def from_cell_table(path) -> list[Cell]:
    """Cells from a CSV with columns ``id, x, y, type, size``."""
    df = pd.read_csv(path, keep_default_na=False)
    missing = [c for c in _TABLE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"cell table missing columns: {missing}")
    return [Cell(id=int(r.id), type=str(r.type),
                 centroid=(float(r.x), float(r.y)), size=float(r.size))
            for r in df.itertuples(index=False)]
