"""Task 19: backends emit real per-cell FOOTPRINTS; rasterize consumes them."""
from __future__ import annotations

import numpy as np

from viva_multicell_blueprint.compare.raster import frame_from_backend, rasterize
from viva_multicell_blueprint.compile.base import OUTPUT_PORTS
from viva_multicell_blueprint.core import build_core


def _poly_area(xy):
    x, y = np.asarray(xy, float).T
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def test_footprints_port_declared():
    assert OUTPUT_PORTS["footprints"] == "overwrite[map[node]]"


def test_cpm_emits_occupancy_footprint():
    from test_cpm_backend import _make, _three_cells
    b = _make(_three_cells(), domain={"extent_um": [20, 20], "grid_resolution_um": 1})
    out = b.update({}, 1.0)
    for cid, fp in out["footprints"].items():
        assert fp["kind"] == "occupancy"
        xy = np.asarray(fp["xy"])
        assert xy.shape[1] == 2
        assert len({tuple(v) for v in xy.tolist()}) == len(xy)  # distinct voxels
        assert len(xy) == round(out["sizes"][cid])  # voxel count == volume
        # centroid of owned voxels == reported COM
        np.testing.assert_allclose(xy.mean(0), out["positions"][cid], atol=1e-6)
    # dead cell -> empty footprint
    out2 = b.update({"deaths": [{"cell_id": "10"}]}, 1.0)
    assert out2["footprints"]["10"]["xy"] == []


def test_vertex_emits_polygon_footprint():
    from test_vertex_backend import _make, _grid_cells
    b = _make(_grid_cells())
    out = b.update({}, 1.0)
    for cid, fp in out["footprints"].items():
        assert fp["kind"] == "polygon"
        xy = np.asarray(fp["xy"])
        assert len(xy) >= 3
        np.testing.assert_allclose(_poly_area(xy), out["sizes"][cid], rtol=1e-3)
        np.testing.assert_allclose(xy.mean(0), out["positions"][cid], atol=0.5)


def test_raster_uses_real_footprint_iou_one():
    # SCOPE: raster path + voxel/pixel convention only (polygons are the occupancy's
    # own bounding boxes); NOT a cross-paradigm mechanics comparison.
    from test_cpm_backend import _make as mk_cpm, _three_cells
    out = mk_cpm(_three_cells(), backend_params={"mcs_per_update": 0}, domain={"extent_um": [20, 20], "grid_resolution_um": 1}).update({}, 1.0)
    spec = {"extent_um": [20, 20], "grid_resolution_um": 1.0,
            "type_codes": {"epithelium": 1, "immune": 2}}
    real = rasterize(frame_from_backend(out), spec)
    # the SAME tissue as polygons: voxel i is the unit cell centred on i, so the
    # block's boundary polygon runs half a voxel beyond its extreme voxel centres
    polys = {}
    for cid, fp in out["footprints"].items():
        xy = np.asarray(fp["xy"], float)
        xmin, xmax, ymin, ymax = xy[:, 0].min() - .5, xy[:, 0].max() + .5, xy[:, 1].min() - .5, xy[:, 1].max() + .5
        polys[cid] = {"kind": "polygon",
                      "xy": [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]}
    poly_out = dict(out, footprints=polys)
    from_poly = rasterize(frame_from_backend(poly_out), spec)
    disc = rasterize({k: v for k, v in out.items() if k != "footprints"}, spec)
    occ = lambda a: a > 0
    iou = lambda a, b: (occ(a) & occ(b)).sum() / (occ(a) | occ(b)).sum()
    assert iou(real, from_poly) > 0.99          # rect cells: bbox polygon == voxels
    assert iou(real, disc) < iou(real, from_poly)  # disc fallback is biased
    # real footprint pixel count equals total CPM volume exactly
    assert occ(real).sum() == round(sum(out["sizes"].values()))


def test_vertex_polygons_rasterize_to_face_area():
    from test_vertex_backend import _make, _grid_cells
    out = _make(_grid_cells()).update({}, 1.0)
    frame = frame_from_backend(out)
    assert len(frame["polygons"]) == len(out["sizes"]) and not frame["occupancy"]
    types = sorted(set(out["types"].values()))
    res = 0.05
    spec = {"extent_um": [4.0, 4.0], "grid_resolution_um": res,
            "type_codes": {t: i + 1 for i, t in enumerate(types)}}
    # shift native coords (centroids ~0..2) into the grid
    frame["frame"] = {"scale": 1.0, "origin": [1.0, 1.0]}
    ras = rasterize(frame, spec)
    area = (ras > 0).sum() * res * res
    np.testing.assert_allclose(area, sum(out["sizes"].values()), rtol=0.05)


def test_dead_footprints_do_not_share_xy_list():
    from test_cpm_backend import _make, _three_cells
    b = _make(_three_cells(), domain={"extent_um": [20, 20], "grid_resolution_um": 1})
    out = b.update({"deaths": [{"cell_id": "10"}, {"cell_id": "20"}]}, 1.0)
    out["footprints"]["10"]["xy"].append([0, 0])
    assert out["footprints"]["20"]["xy"] == []
