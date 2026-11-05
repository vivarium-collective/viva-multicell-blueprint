"""Geometry for the vertex backend: a hexagonal epithelial sheet + overlay islands.

The vertex substrate is built so that **every polygon is exactly one blueprint
cell** -- no guard ring, no collar, no polygons standing in for medium or for the
regions the shared fields occupy (the fields are *overlaid* on the tissue, sampled
at each face; see :mod:`.vertex_coupling`).

* **Epithelial sheet** -- the confluent cells are placed on a regular hexagonal
  lattice whose cell area equals the blueprint target size (vertex models are
  well-behaved on near-hexagonal tilings; a square tiling puts every vertex at a
  degenerate 4-way junction). Each blueprint cell gets its OWN lattice site by an
  optimal 1:1 assignment (Hungarian, minimum total squared displacement), so two
  cells can never share a polygon -- coincident centroids are split onto
  neighbouring sites rather than merged.
* **Overlay layer** -- cells that live *above* the sheet (Sego's immune layer,
  z=1 over the z=0 epithelium) are free hexagonal islands of their own size centred
  exactly on their blueprint centroid. They share no vertices with the sheet;
  their contact with the epithelium is the polygon OVERLAP area
  (:func:`overlap_area`), the 2-layer analog of a lattice contact surface.

Orientation follows tyssue's ``from_2d_voronoi``: each face's half-edges run
counter-clockwise (srce -> trgt) around it.
"""
from __future__ import annotations

import contextlib
import io
import math

import numpy as np
import pandas as pd

_SQRT3 = math.sqrt(3.0)


def hex_side(area: float) -> float:
    """Side (= circumradius) of a regular hexagon of the given area."""
    return math.sqrt(2.0 * max(float(area), 1e-12) / (3.0 * _SQRT3))


def hex_polygon(center, area: float) -> np.ndarray:
    """Pointy-top regular hexagon (6x2, counter-clockwise) of ``area`` at ``center``."""
    s = hex_side(area)
    ang = np.deg2rad(30.0 + 60.0 * np.arange(6))
    return np.c_[center[0] + s * np.cos(ang), center[1] + s * np.sin(ang)]


def hex_lattice_sites(centroids: np.ndarray, area: float, margin: int = 3) -> np.ndarray:
    """Pointy-top hexagonal lattice sites (cell area ``area``) anchored on the first
    centroid's bounding-box corner and covering the cloud plus ``margin`` rings, so
    the 1:1 assignment always has spare sites to split crowded cells onto."""
    s = hex_side(area)
    dx, dy = _SQRT3 * s, 1.5 * s
    lo = centroids.min(0)
    hi = centroids.max(0)
    # enough sites for the bounding box AND for every cell even if they all coincide
    n_need = len(centroids)
    nx = max(int(math.ceil((hi[0] - lo[0]) / dx)) + 1, int(math.ceil(math.sqrt(n_need)))) + 2 * margin
    ny = max(int(math.ceil((hi[1] - lo[1]) / dy)) + 1, int(math.ceil(math.sqrt(n_need)))) + 2 * margin
    sites = []
    for j in range(-margin, ny - margin):
        for i in range(-margin, nx - margin):
            sites.append((lo[0] + (i + 0.5 * (j % 2)) * dx, lo[1] + j * dy))
    return np.asarray(sites, dtype=float)


def assign_to_lattice(centroids: np.ndarray, sites: np.ndarray) -> np.ndarray:
    """Optimal 1:1 assignment of centroids to distinct lattice sites (minimum total
    squared displacement). Returns ``site_of_cell`` (index into ``sites``)."""
    from scipy.optimize import linear_sum_assignment

    d2 = ((centroids[:, None, :] - sites[None, :, :]) ** 2).sum(-1)
    rows, cols = linear_sum_assignment(d2)
    out = np.empty(len(centroids), dtype=int)
    out[rows] = cols
    return out


def sheet_from_polygons(polygons, name: str = "vertex", weld_tol: float = 1e-6):
    """Build a flat-3D tyssue :class:`Sheet` whose faces are exactly ``polygons``
    (each an (n, 2) counter-clockwise ring). Coincident vertices of neighbouring
    polygons are welded (within ``weld_tol``), which is what makes the hexagonal
    sheet confluent; polygons sharing no vertex (overlay islands) stay free."""
    from tyssue import Sheet
    from tyssue.config.geometry import planar_spec
    from tyssue.generation.utils import make_df

    key_of = {}
    verts = []

    def vid(p):
        key = (round(p[0] / weld_tol), round(p[1] / weld_tol))
        if key not in key_of:
            key_of[key] = len(verts)
            verts.append((float(p[0]), float(p[1])))
        return key_of[key]

    edges = []
    centers = []
    for f, poly in enumerate(polygons):
        ring = [vid(p) for p in np.asarray(poly, dtype=float)]
        for a, b in zip(ring, ring[1:] + ring[:1]):
            edges.append((a, b, f))
        centers.append(np.asarray(poly, dtype=float).mean(0))

    specs = planar_spec()
    edge_df = make_df(pd.Index(range(len(edges)), name="edge"), specs["edge"])
    el = np.asarray(edges, dtype=int)
    for i, col in enumerate(("srce", "trgt", "face")):
        edge_df[col] = el[:, i]
    vert_df = make_df(pd.Index(range(len(verts)), name="vert"), specs["vert"])
    vert_df[["x", "y"]] = np.asarray(verts, dtype=float)
    face_df = make_df(pd.Index(range(len(polygons)), name="face"), specs["face"])
    face_df[["x", "y"]] = np.asarray(centers, dtype=float)

    vert_df["z"] = 0.0
    face_df["z"] = 0.0
    with contextlib.redirect_stdout(io.StringIO()):
        sheet = Sheet(name, {"vert": vert_df, "edge": edge_df, "face": face_df},
                      coords=["x", "y", "z"])
        sheet.reset_index()
        sheet.reset_topo()
    return sheet


# --------------------------------------------------------------------------- #
# Overlay contact: polygon overlap (Sutherland-Hodgman against a convex clipper)
# --------------------------------------------------------------------------- #
def polygon_area(poly) -> float:
    p = np.asarray(poly, dtype=float)
    if len(p) < 3:
        return 0.0
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _convex_hull_ccw(poly) -> np.ndarray:
    from scipy.spatial import ConvexHull

    p = np.asarray(poly, dtype=float)
    if len(p) < 3:
        return p
    try:
        return p[ConvexHull(p).vertices]   # scipy returns 2-D hulls counter-clockwise
    except Exception:  # degenerate (collinear) ring
        return p


def _clip(subject, clipper) -> list:
    """Sutherland-Hodgman: ``subject`` (any simple polygon) clipped by the convex,
    counter-clockwise ``clipper``."""
    out = [tuple(p) for p in subject]
    n = len(clipper)
    for i in range(n):
        if not out:
            break
        ax, ay = clipper[i]
        bx, by = clipper[(i + 1) % n]

        def inside(p):
            return (bx - ax) * (p[1] - ay) - (by - ay) * (p[0] - ax) >= 0.0

        def cross(p, q):
            # intersection of segment p-q with the clip line a-b
            dx1, dy1 = q[0] - p[0], q[1] - p[1]
            dx2, dy2 = bx - ax, by - ay
            den = dx1 * dy2 - dy1 * dx2
            if den == 0.0:
                return p
            t = ((ax - p[0]) * dy2 - (ay - p[1]) * dx2) / den
            return (p[0] + t * dx1, p[1] + t * dy1)

        inp, out = out, []
        for j in range(len(inp)):
            cur, prev = inp[j], inp[j - 1]
            if inside(cur):
                if not inside(prev):
                    out.append(cross(prev, cur))
                out.append(cur)
            elif inside(prev):
                out.append(cross(prev, cur))
    return out


def overlap_area(poly_a, poly_b) -> float:
    """Area of the overlap of two polygons. ``poly_b`` is used through its convex
    hull as the clipper (an overlay island stays near-hexagonal, so this is exact
    for it and a tight upper bound otherwise)."""
    a = np.asarray(poly_a, dtype=float)
    b = np.asarray(poly_b, dtype=float)
    if len(a) < 3 or len(b) < 3:
        return 0.0
    if (a[:, 0].max() < b[:, 0].min() or b[:, 0].max() < a[:, 0].min()
            or a[:, 1].max() < b[:, 1].min() or b[:, 1].max() < a[:, 1].min()):
        return 0.0
    return polygon_area(_clip(a, _convex_hull_ccw(b)))


# --------------------------------------------------------------------------- #
# Mesh validity (a folded vertex mesh is invalid geometry, not a cell shape)
# --------------------------------------------------------------------------- #
def signed_area(poly) -> float:
    """Shoelace area; positive for a counter-clockwise ring, <= 0 if inverted."""
    p = np.asarray(poly, dtype=float)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def is_simple(poly) -> bool:
    """True if no two non-adjacent edges of the ring cross (no bow-tie)."""
    p = np.asarray(poly, dtype=float)
    n = len(p)

    def orient(a, b, c):
        return np.sign((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))

    for i in range(n):
        a, b = p[i], p[(i + 1) % n]
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue   # adjacent through the wrap-around
            c, d = p[j], p[(j + 1) % n]
            if orient(a, b, c) * orient(a, b, d) < 0 and orient(c, d, a) * orient(c, d, b) < 0:
                return False
    return True


def n_invalid_faces(polygons) -> int:
    """Number of inverted or self-intersecting rings among ``polygons``."""
    return sum(1 for p in polygons if len(p) >= 3 and (signed_area(p) <= 0 or not is_simple(p)))
