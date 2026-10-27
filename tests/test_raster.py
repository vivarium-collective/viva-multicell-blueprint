import numpy as np

from viva_multicell_blueprint.compare.raster import grid_shape, rasterize, type_codes

GRID = {"extent_um": [40.0, 40.0], "grid_resolution_um": 1.0}


def test_raster_common_extent_mismatched_frames():
    cells = {"a": (10, 10), "b": (28, 20)}
    # CPM-style: integer lattice, 2 um voxels, no origin shift.
    cpm = {"positions": {k: [x / 2, y / 2] for k, (x, y) in cells.items()},
           "sizes": {"a": 12.0, "b": 12.0}, "types": {"a": "H", "b": "I"},
           "alive": {"a": True, "b": True}, "frame": {"scale": 2.0}}
    # tyssue-style: continuous um positions, areas in um^2.
    vtx = {"positions": {k: [x + 0.37, y - 0.21] for k, (x, y) in cells.items()},
           "sizes": {"a": 48.0, "b": 48.0}, "types": {"a": "H", "b": "I"},
           "alive": {"a": True, "b": True}}
    g = dict(GRID, type_codes=type_codes(cpm, vtx))
    ma, mb = rasterize(cpm, g), rasterize(vtx, g)
    assert ma.shape == mb.shape == grid_shape(GRID) == (40, 40)
    assert ma[10, 10] == mb[10, 10] == g["type_codes"]["H"]
    assert ma[20, 28] == mb[20, 28] == g["type_codes"]["I"]


def test_polygon_scanfill_and_dead_cells():
    f = {"positions": {"a": [10, 10], "b": [30, 30]}, "sizes": {"a": 1, "b": 1},
         "types": {"a": "I", "b": "H"}, "alive": {"a": True, "b": False},
         "polygons": {"a": [[5, 5], [15, 5], [15, 15], [5, 15]]}}
    g = dict(GRID, type_codes=type_codes(f))
    m = rasterize(f, g)
    assert 90 <= (m == g["type_codes"]["I"]).sum() <= 121
    assert (m == g["type_codes"]["H"]).sum() == 0  # dead paints nothing
