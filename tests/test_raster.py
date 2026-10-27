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


def test_nonsquare_domain_shape_matches_array():
    g = {"extent_um": [40.0, 20.0], "grid_resolution_um": 1.0}
    f = {"positions": {"a": [35, 5]}, "sizes": {"a": 12.0}, "types": {"a": "I"},
         "alive": {"a": True}}
    g["type_codes"] = type_codes(f)
    m = rasterize(f, g)
    assert grid_shape(g) == (20, 40) == m.shape  # (ny, nx)
    assert m[5, 35] == g["type_codes"]["I"]  # row=y, col=x


def test_occupancy_voxels_painted_and_missing_alive_is_dead():
    f = {"positions": {"a": [0, 0], "b": [0, 0]}, "sizes": {"a": 1, "b": 1},
         "types": {"a": "H", "b": "I"}, "alive": {"a": True},
         "occupancy": {"a": [[2, 3], [3, 3], [2, 4]], "b": [[9, 9]]},
         "frame": {"scale": 1.0}}
    g = dict(GRID, type_codes=type_codes(f))
    m = rasterize(f, g)
    code = g["type_codes"]["H"]
    assert sorted(zip(*np.nonzero(m == code))) == [(3, 2), (3, 3), (4, 2)]  # (row=y, col=x)
    assert (m > 0).sum() == 3  # b has no alive entry -> dead -> not painted
