"""Rasterize a per-step backend frame onto the COMMON grid (spec s13 item 5).

The grid shape/extent come ONLY from ``blueprint.domain`` (``extent_um`` +
``grid_resolution_um``), never from a backend's native frame. A frame is the
MechanicsBackend port dict (``positions``/``sizes``/``types``/``alive`` keyed by
blueprint cell id). Native coordinates are mapped to um by an optional
``frame["frame"] = {"scale": um_per_unit, "origin": [x0, y0]}`` (CPM integer
lattice: scale = voxel size; tyssue continuous: usually scale 1). Sizes are in
native units^2 (area) and are scaled by ``scale**2``.

A cell missing from ``alive`` is treated as dead (same default as metrics.cell_state).

Footprints, most precise first (the disc fallback is APPROXIMATE: it ignores real
cell shape, so occupied-pixel IoU/SSIM from disc-filled frames must not be a
headline number; use it only until backends emit polygons/occupancy):
  ``frame["polygons"][id]``   (N,2) vertex polygon in native coords (vertex
                              backend) -> scan-filled with skimage.draw.polygon
  ``frame["occupancy"][id]``  (M,2) lattice voxel indices (CPM)
  otherwise                   area-equivalent disc about the centroid.

State map: int array shape ``grid_shape(domain)``; 0 = empty, >0 = type code
(``grid_spec["type_codes"]``: type -> int >= 1). Dead cells paint nothing.
"""
from __future__ import annotations

import math

import numpy as np
from skimage.draw import disk, polygon

EMPTY = 0


def grid_shape(grid_spec) -> tuple:
    """(ny, nx): the shape of the rasterized array (row=y, col=x). extent_um is (x, y)."""
    extent, res = grid_spec["extent_um"], grid_spec["grid_resolution_um"]
    nx, ny = (max(1, int(math.ceil(float(e) / float(res)))) for e in extent[:2])
    return (ny, nx)


def type_codes(*runs_or_frames, extra=()) -> dict:
    """Shared type -> code map (sorted, starting at 1) over frames/runs, so two
    runs rasterize with identical codes."""
    seen = set(extra)
    for r in runs_or_frames:
        for f in (r if isinstance(r, (list, tuple)) else [r]):
            seen.update(str(t) for t in f.get("types", {}).values())
    return {t: i + 1 for i, t in enumerate(sorted(seen))}


def _to_grid(xy, frame_meta, res):
    scale = float(frame_meta.get("scale", 1.0))
    origin = frame_meta.get("origin", (0.0, 0.0))
    xy = np.asarray(xy, float)[..., :2]
    um = np.asarray(origin, float)[:2] + scale * xy
    return um / float(res)  # (x, y) in fractional pixel units; col=x, row=y


def rasterize(run_frame, grid_spec) -> np.ndarray:
    ny, nx = grid_shape(grid_spec)
    out = np.full((ny, nx), EMPTY, dtype=np.int32)
    codes = grid_spec.get("type_codes") or type_codes(run_frame)
    res = float(grid_spec["grid_resolution_um"])
    meta = run_frame.get("frame", {}) or {}
    scale = float(meta.get("scale", 1.0))
    alive, types = run_frame.get("alive", {}), run_frame.get("types", {})
    polys, occ = run_frame.get("polygons", {}), run_frame.get("occupancy", {})
    for cid in sorted(types, key=str):
        if not alive.get(cid, False) or str(types[cid]) not in codes:
            continue
        code = codes[str(types[cid])]
        if cid in polys:
            p = _to_grid(polys[cid], meta, res)
            rr, cc = polygon(p[:, 1], p[:, 0], shape=out.shape)
            if rr.size == 0:  # sub-pixel polygon: paint the centroid pixel
                c = p.mean(axis=0)
                rr, cc = disk((c[1], c[0]), 0.5, shape=out.shape)
        elif cid in occ:
            v = _to_grid(np.asarray(occ[cid], float), meta, res)
            cc = np.clip(np.floor(v[:, 0]).astype(int), 0, nx - 1)
            rr = np.clip(np.floor(v[:, 1]).astype(int), 0, ny - 1)
        else:
            c = _to_grid(run_frame["positions"][cid], meta, res)
            area_um2 = float(run_frame.get("sizes", {}).get(cid, res * res)) * scale ** 2
            r_px = max(math.sqrt(area_um2 / math.pi) / res, 0.5)
            rr, cc = disk((c[1], c[0]), r_px, shape=out.shape)
        out[rr, cc] = code
    return out
