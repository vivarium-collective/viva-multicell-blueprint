"""CPMBackend: the viva-cpm Cellular Potts model behind the MechanicsBackend port.

Small fixtures (2-4 cells on a tiny lattice). These cover the four seams the
brief names: exact 1:1 seeding from the canonical cell list, port-shape emission,
the ``set_field`` seam (the shared field is written into the World each step with
native dynamics zeroed, so CPM only READS it -- it is not re-diffused), and the
contact-area readback grouped by neighbor type name into ``shared_surface_by_type``.
"""
from __future__ import annotations

import numpy as np
import pytest

from viva_multicell_blueprint.compile.base import MechanicsBackend, OUTPUT_PORTS
from viva_multicell_blueprint.core import build_core


def _make(cells, backend_params=None, domain=None, cell_types=None, adhesion=None):
    from viva_multicell_blueprint.backends.cpm_backend import CPMBackend

    return CPMBackend(
        {
            "initial_state": cells,
            "cell_types": cell_types or [],
            "adhesion": adhesion or [],
            "domain": domain or {},
            "backend_params": backend_params or {},
        },
        core=build_core(),
    )


def _three_cells():
    """Three well-separated cells (two types) on a 20x20 lattice."""
    return [
        {"id": 10, "type": "epithelium", "centroid": [5.0, 5.0], "size": 25.0},
        {"id": 20, "type": "immune", "centroid": [14.0, 5.0], "size": 25.0},
        {"id": 30, "type": "epithelium", "centroid": [5.0, 14.0], "size": 25.0},
    ]


def _two_adjacent():
    """Two cells of different types whose seed blocks abut (share a contact face)."""
    return [
        {"id": 1, "type": "A", "centroid": [5.0, 5.0], "size": 25.0},
        {"id": 2, "type": "B", "centroid": [10.0, 5.0], "size": 25.0},
    ]


def test_cpm_seeds_from_cell_list():
    cells = _three_cells()
    b = _make(cells, backend_params={"dims": [20, 20, 1], "mcs_per_update": 1})

    # EXACT 1:1 identity: every blueprint id -> a distinct cpm cell id.
    assert set(b.identity) == {"10", "20", "30"}
    assert len(set(b.identity.values())) == 3
    # the cpm world really holds three cells.
    assert b._world.n_cells() == 3
    # type names round-trip through the integer tag map.
    assert b._type_ids["medium"] == 0
    assert b.identity["10"] != b.identity["20"]

    out = b.update({}, 1.0)
    # every blueprint cell emits a positive seeded size (~ its target volume).
    assert set(out["sizes"]) == {"10", "20", "30"}
    assert all(s > 0 for s in out["sizes"].values())
    # types come back as blueprint type names, not ints.
    assert out["types"]["10"] == "epithelium"
    assert out["types"]["20"] == "immune"
    assert all(out["alive"].values())


def test_cpm_emits_port_shape():
    cells = _three_cells()
    b = _make(cells, backend_params={"dims": [20, 20, 1], "mcs_per_update": 1})

    assert isinstance(b, MechanicsBackend)
    assert set(b.inputs()) == {"fates", "field_grid", "divisions", "deaths", "activations"}
    assert set(b.outputs()) == set(OUTPUT_PORTS)

    out = b.update({}, 1.0)
    assert set(out) == set(OUTPUT_PORTS)
    ids = {str(c["id"]) for c in cells}
    # every blueprint cell is present in every output map.
    for port in OUTPUT_PORTS:
        assert set(out[port]) == ids
    # positions are 2D on a flat (nz=1) lattice; sizes are positive volumes.
    assert all(len(p) == 2 for p in out["positions"].values())
    assert all(s > 0 for s in out["sizes"].values())
    # shared_surface_by_type is map[cell -> map[type_name -> area]]; medium excluded.
    shared = out["shared_surface_by_type"]
    assert all(isinstance(v, dict) for v in shared.values())
    assert "medium" not in {t for m in shared.values() for t in m}
    # neighbors reference only blueprint ids.
    for nbrs in out["neighbors"].values():
        assert all(n in ids for n in nbrs)

    # fates relabel a cell's type (names on the port -> int tags internally).
    out2 = b.update({"fates": {"10": "immune"}}, 1.0)
    assert out2["types"]["10"] == "immune"


def test_cpm_set_field_overwrites_grid():
    cells = _three_cells()
    dims = [20, 20, 1]
    b = _make(
        cells,
        backend_params={
            "dims": dims,
            "mcs_per_update": 1,
            "fields": [{"name": "virus"}],
        },
    )
    # the native field must be built with zeroed dynamics so the engine only
    # READS the injected grid (d=0, decay=0, no secretion -> step() cannot change it).
    nx, ny, nz = dims
    grid = np.arange(nx * ny * nz, dtype=float).reshape((nx, ny, nz))
    b.update({"field_grid": {"virus": grid}}, 1.0)

    fc = np.asarray(b._world.field_conc(b._field_idx["virus"]))
    # set_field wrote our grid in field_conc layout and step() did NOT re-diffuse it.
    assert np.allclose(fc, grid.flatten(order="F"))

    # a second, different grid OVERWRITES the first (authoritative shared field).
    grid2 = grid * 3.0 + 1.0
    b.update({"field_grid": {"virus": grid2}}, 1.0)
    fc2 = np.asarray(b._world.field_conc(b._field_idx["virus"]))
    assert np.allclose(fc2, grid2.flatten(order="F"))
    assert not np.allclose(fc2, fc)


def test_cpm_set_field_layout_nonsquare_asymmetric():
    """Non-square lattice + a single hot cell at x != y: a transpose (nx/ny swap)
    would put the value at a different flat index, so this guards field layout."""
    dims = [12, 7, 1]
    cells = [{"id": 10, "type": "epithelium", "centroid": [3.0, 3.0], "size": 9.0}]
    b = _make(cells, backend_params={"dims": dims, "mcs_per_update": 1, "fields": [{"name": "virus"}]})
    nx, ny, nz = dims
    x, y = 9, 1
    grid = np.zeros((nx, ny, nz))
    grid[x, y, 0] = 7.0
    b.update({"field_grid": {"virus": grid}}, 1.0)
    fc = np.asarray(b._world.field_conc(b._field_idx["virus"]))
    assert fc.size == nx * ny * nz
    assert np.count_nonzero(fc) == 1
    assert int(np.argmax(fc)) == x + y * nx == 21
    assert fc[x + y * nx] == 7.0
    assert int(np.argmax(fc)) != y + x * ny  # the transposed index (64) must not be hit


def _label_backend(order_5_first):
    nx, ny, nz = 12, 7, 1
    lab = np.zeros((nx, ny, nz), dtype=int)
    if order_5_first:   # label 5 appears first in the Fortran-flat scan
        lab[1:3, 1:4, 0] = 5   # 6 voxels
        lab[7:10, 2:5, 0] = 9  # 9 voxels
    else:               # label 9 appears first
        lab[1:4, 1:4, 0] = 9   # 9 voxels
        lab[7:9, 2:5, 0] = 5   # 6 voxels
    cells = [
        {"id": 9, "type": "immune", "centroid": [2.0, 2.0], "size": 9.0},
        {"id": 5, "type": "epithelium", "centroid": [7.5, 3.0], "size": 6.0},
    ]
    return _make(cells, backend_params={
        "dims": [nx, ny, nz], "mcs_per_update": 0, "label_image": lab, "target_volume": 1.0,
    })


def test_cpm_seed_from_labels_identity_ascending_case():
    b = _label_backend(order_5_first=True)
    assert b.identity == {"5": 1, "9": 2}
    assert b._world.n_cells() == 2
    out = b.update({}, 1.0)
    assert out["sizes"]["5"] == pytest.approx(6.0) and out["sizes"]["9"] == pytest.approx(9.0)
    assert out["types"] == {"5": "epithelium", "9": "immune"}


def test_cpm_seed_from_labels_id_follows_first_appearance_not_sorted():
    # the engine ids cells by first appearance, so here label 9 -> cpm 1, label 5 -> 2.
    # sizes/types must still align to the right BLUEPRINT ids.
    b = _label_backend(order_5_first=False)
    assert b.identity == {"9": 1, "5": 2}
    out = b.update({}, 1.0)
    assert out["sizes"]["5"] == pytest.approx(6.0) and out["sizes"]["9"] == pytest.approx(9.0)
    assert out["types"] == {"5": "epithelium", "9": "immune"}


def _latent_pool():
    """Three LATENT macrophage cells (type ``M_latent``) spaced on a 16x16 lattice."""
    return [
        {"id": 1, "type": "M_latent", "centroid": [4.0, 4.0], "size": 25.0},
        {"id": 2, "type": "M_latent", "centroid": [11.0, 4.0], "size": 25.0},
        {"id": 3, "type": "M_latent", "centroid": [4.0, 11.0], "size": 25.0},
    ]


def test_cpm_activation_changes_latent_to_active():
    """Task 20: the ``activations`` input activates a pre-placed LATENT pool --
    ``{M: 2}`` turns exactly 2 ``M_latent`` cells into ``M`` (a TYPE CHANGE via the
    same set_cell_type path), and the latent pool is respected (never over-activated)."""
    cells = _latent_pool()
    b = _make(
        cells,
        cell_types=[{"name": "M"}, {"name": "M_latent"}],
        backend_params={"dims": [16, 16, 1], "mcs_per_update": 0},
    )
    # start: three latent, no active M.
    out0 = b.update({}, 1.0)
    assert sum(t == "M_latent" for t in out0["types"].values()) == 3
    assert sum(t == "M" for t in out0["types"].values()) == 0

    # activate 2 -> exactly 2 become M, 1 latent remains; type change is real.
    out1 = b.update({"activations": {"M": 2}}, 1.0)
    assert sum(t == "M" for t in out1["types"].values()) == 2
    assert sum(t == "M_latent" for t in out1["types"].values()) == 1
    # selection is deterministic (first two ids in order).
    assert out1["types"]["1"] == "M" and out1["types"]["2"] == "M"
    assert out1["types"]["3"] == "M_latent"

    # pool respected: request 5 but only 1 latent is left -> clamp to 1, total M = 3.
    out2 = b.update({"activations": {"M": 5}}, 1.0)
    assert sum(t == "M" for t in out2["types"].values()) == 3
    assert sum(t == "M_latent" for t in out2["types"].values()) == 0
    # activated cells are alive immune cells now (not latent, still alive).
    assert all(out2["alive"].values())


def test_cpm_contact_area_by_type():
    cells = _two_adjacent()
    # mcs_per_update=0: the world stays at the deterministic seed, so the two
    # abutting blocks are guaranteed to share a contact face at read-back.
    b = _make(cells, backend_params={"dims": [16, 12, 1], "mcs_per_update": 0})
    out = b.update({}, 1.0)
    shared = out["shared_surface_by_type"]

    # grouping is keyed by the NEIGHBOR's type NAME, with medium (tag 0) excluded.
    assert set(shared["1"]) == {"B"}
    assert set(shared["2"]) == {"A"}
    assert shared["1"]["B"] > 0

    # it equals the raw CPM contact-area-by-type readback for that neighbor tag.
    raw = b._world.cell_contact_area_by_type(b.identity["1"])
    assert shared["1"]["B"] == raw[b._type_ids["B"]]
    # medium never appears as a shared-surface key.
    assert "medium" not in shared["1"] and 0 not in shared["1"]

    # neighbors list references blueprint ids (the abutting partner).
    assert "2" in out["neighbors"]["1"]
    assert "1" in out["neighbors"]["2"]
