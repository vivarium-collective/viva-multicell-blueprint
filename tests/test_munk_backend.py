"""MunkBackend: the viva-munk center-based model behind the MechanicsBackend port.

Small fixtures (2-4 circular cells in a tiny chamber). These cover the seams the
paradigm needs: exact 1:1 seeding from the canonical cell list, port-shape
emission, the circle-circle contact readback grouped by neighbor type name into
``shared_surface_by_type``, fates / deaths / activations (latent->active immune
recruitment, as CPM), lineage division (a daughter body joins the engine but is not
a cohort port citizen), and chemotaxis off the authoritative shared field.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from viva_multicell_blueprint.compile.base import MechanicsBackend, OUTPUT_PORTS
from viva_multicell_blueprint.core import build_core


def _make(cells, backend_params=None, domain=None, cell_types=None, adhesion=None):
    from viva_multicell_blueprint.backends.munk_backend import MunkBackend

    return MunkBackend(
        {
            "initial_state": cells,
            "cell_types": cell_types or [],
            "adhesion": adhesion or [],
            "domain": domain or {"extent_um": [60.0, 60.0]},
            "backend_params": backend_params or {},
        },
        core=build_core(),
    )


def _three_cells():
    """Three well-separated cells (two types) in a 60x60 chamber."""
    return [
        {"id": 10, "type": "epithelium", "centroid": [12.0, 12.0], "size": 12.0},
        {"id": 20, "type": "immune", "centroid": [40.0, 12.0], "size": 12.0},
        {"id": 30, "type": "epithelium", "centroid": [12.0, 40.0], "size": 12.0},
    ]


def _two_adjacent():
    """Two cells of different types whose discs overlap (share a contact lens).
    size 12 -> radius ~1.954; centers 3.4 apart < 2r so they intersect."""
    return [
        {"id": 1, "type": "A", "centroid": [20.0, 20.0], "size": 12.0},
        {"id": 2, "type": "B", "centroid": [23.4, 20.0], "size": 12.0},
    ]


def test_munk_seeds_from_cell_list():
    cells = _three_cells()
    b = _make(cells, backend_params={"substeps": 1})

    # EXACT 1:1 identity: every blueprint id -> a distinct munk id.
    assert set(b.identity) == {"10", "20", "30"}
    assert len(set(b.identity.values())) == 3

    out = b.update({}, 1.0)
    assert set(out["sizes"]) == {"10", "20", "30"}
    # area round-trips: a disc of area == size is emitted back as that size.
    assert all(s == pytest.approx(12.0) for s in out["sizes"].values())
    assert out["types"]["10"] == "epithelium"
    assert out["types"]["20"] == "immune"
    assert all(out["alive"].values())


def test_munk_emits_port_shape():
    cells = _three_cells()
    b = _make(cells, backend_params={"substeps": 1})

    assert isinstance(b, MechanicsBackend)
    assert set(b.inputs()) == {"fates", "field_grid", "divisions", "deaths", "activations"}
    assert set(b.outputs()) == set(OUTPUT_PORTS)

    out = b.update({}, 1.0)
    assert set(out) == set(OUTPUT_PORTS)
    ids = {str(c["id"]) for c in cells}
    for port in OUTPUT_PORTS:
        assert set(out[port]) == ids
    assert all(len(p) == 2 for p in out["positions"].values())
    assert all(s > 0 for s in out["sizes"].values())
    # footprints are polygon discs (boundary-order vertices) the rasterizer can draw.
    for fp in out["footprints"].values():
        assert fp["kind"] == "polygon"
        assert len(fp["xy"]) >= 8 and all(len(pt) == 2 for pt in fp["xy"])
    # neighbors reference only blueprint ids.
    for nbrs in out["neighbors"].values():
        assert all(n in ids for n in nbrs)

    # fates relabel a cell's type.
    out2 = b.update({"fates": {"10": "immune"}}, 1.0)
    assert out2["types"]["10"] == "immune"


def test_munk_contact_between_overlapping_discs():
    cells = _two_adjacent()
    b = _make(cells, domain={"extent_um": [40.0, 40.0]}, backend_params={"substeps": 1})
    out = b.update({}, 1.0)
    shared = out["shared_surface_by_type"]

    # grouping is keyed by the NEIGHBOR's type NAME.
    assert set(shared["1"]) == {"B"}
    assert set(shared["2"]) == {"A"}
    assert shared["1"]["B"] > 0
    # the contact is symmetric (same interface chord both ways).
    assert shared["1"]["B"] == pytest.approx(shared["2"]["A"])
    # neighbors list references the abutting partner (blueprint ids).
    assert "2" in out["neighbors"]["1"]
    assert "1" in out["neighbors"]["2"]


def test_munk_well_separated_cells_have_no_contact():
    cells = _three_cells()  # 28 um apart, radius ~2 -> never touch
    b = _make(cells, backend_params={"substeps": 1, "jitter_per_second": 0.0})
    out = b.update({}, 1.0)
    assert out["neighbors"]["10"] == []
    assert out["shared_surface_by_type"]["10"] == {}


def test_munk_applies_death_placeholder():
    cells = _three_cells()
    b = _make(cells, backend_params={"substeps": 1})
    out = b.update({"deaths": [{"cell_id": 30}]}, 1.0)
    assert out["alive"]["30"] is False
    # a dead cell still gets an entry in every port (placeholder), not dropped.
    for port in OUTPUT_PORTS:
        assert "30" in out[port]
    assert out["footprints"]["30"]["kind"] == "none"
    assert out["neighbors"]["30"] == []


def _latent_pool():
    return [
        {"id": 1, "type": "M_latent", "centroid": [10.0, 10.0], "size": 12.0},
        {"id": 2, "type": "M_latent", "centroid": [30.0, 10.0], "size": 12.0},
        {"id": 3, "type": "M_latent", "centroid": [10.0, 30.0], "size": 12.0},
    ]


def test_munk_activation_changes_latent_to_active():
    """The ``activations`` input activates a pre-placed LATENT pool: {M: 2} turns
    exactly 2 M_latent cells into M, deterministic id order, clamped to the pool."""
    cells = _latent_pool()
    b = _make(cells, cell_types=[{"name": "M"}, {"name": "M_latent"}],
              backend_params={"substeps": 0})
    out0 = b.update({}, 1.0)
    assert sum(t == "M_latent" for t in out0["types"].values()) == 3

    out1 = b.update({"activations": {"M": 2}}, 1.0)
    assert sum(t == "M" for t in out1["types"].values()) == 2
    assert sum(t == "M_latent" for t in out1["types"].values()) == 1
    assert out1["types"]["1"] == "M" and out1["types"]["2"] == "M"
    assert out1["types"]["3"] == "M_latent"

    # pool respected: request 5 but only 1 latent remains -> clamp to 1.
    out2 = b.update({"activations": {"M": 5}}, 1.0)
    assert sum(t == "M" for t in out2["types"].values()) == 3
    assert all(out2["alive"].values())


def test_munk_activation_unknown_type_warns_not_silent():
    cells = _latent_pool()
    b = _make(cells, cell_types=[{"name": "M_latent"}],  # active "M" undeclared, no pool
              backend_params={"substeps": 0})
    # relabel the latent cells so no cell is in M's pool -> a true unknown activation.
    cells_no_pool = [{"id": 1, "type": "epithelium", "centroid": [10.0, 10.0], "size": 12.0}]
    b2 = _make(cells_no_pool, cell_types=[{"name": "epithelium"}], backend_params={"substeps": 0})
    with pytest.warns(RuntimeWarning, match="unknown active type"):
        out = b2.update({"activations": {"Zzz": 2}}, 1.0)
    assert out["types"]["1"] == "epithelium"


def test_munk_division_adds_daughter_body_not_cohort_citizen():
    """A division event spawns a lineage daughter in the engine (so it collides with
    the cohort) but the daughter has no blueprint id -> it is NOT emitted on the ports."""
    cells = _three_cells()
    b = _make(cells, backend_params={"substeps": 1})
    n_before = len(b._agents)
    out = b.update({"divisions": [{"cell_id": 10}]}, 1.0)
    # the engine now holds one more body...
    assert len(b._agents) == n_before + 1
    assert any(mid.startswith("10:d") for mid in b._agents)
    # ...but the emitted cohort is still exactly the three seeded ids.
    ids = {"10", "20", "30"}
    for port in OUTPUT_PORTS:
        assert set(out[port]) == ids


def test_munk_chemotaxis_climbs_shared_field_gradient():
    """With a +x field gradient and a positive per-type strength, an isolated cell
    drifts up-gradient. The field is READ (sampled), never re-diffused here."""
    cells = [{"id": 1, "type": "immune", "centroid": [20.0, 20.0], "size": 12.0}]
    nx, ny = 40, 40
    # concentration increases with x: gradient points +x everywhere.
    grid = np.tile(np.arange(nx, dtype=float).reshape(nx, 1), (1, ny))
    b = _make(
        cells,
        domain={"extent_um": [float(nx), float(ny)]},
        backend_params={
            "substeps": 10,
            "jitter_per_second": 0.0,
            "fields": [{"name": "chemokine", "chemotaxis": [{"type": "immune", "strength": 5.0}]}],
        },
    )
    x0 = b.update({"field_grid": {"chemokine": grid}}, 1.0)["positions"]["1"][0]
    for _ in range(8):
        x = b.update({"field_grid": {"chemokine": grid}}, 1.0)["positions"]["1"][0]
    assert x > x0 + 0.5  # moved meaningfully toward higher concentration


# --------------------------------------------------------------------------- #
# "Cells move and interact": the fix that makes the center-based backend alive.
# On main the passive epithelium is frozen (no force, no coupling, damping bleeds
# everything in one step); these lock in motion + adhesion-driven interaction.
# --------------------------------------------------------------------------- #

# epithelial adhesion rows from the Sego blueprint: H-H contact energy 5, H-medium
# 25 -> differential adhesivity +20 (cohesive sheet).
_SEGO_EPI_ADHESION = [
    {"a": "H", "b": "H", "strength": 5},
    {"a": "H", "b": "medium", "strength": 25},
]


def _epithelial_sheet(n=5, pitch=5, size=25, origin=10):
    """An n x n block of H cells on a `pitch`-um lattice (mirrors the blueprint
    block's geometry at a tractable scale)."""
    return [
        {"id": i * n + j + 1, "type": "H",
         "centroid": [origin + i * pitch, origin + j * pitch], "size": size}
        for i in range(n) for j in range(n)
    ]


def _edge_set(out):
    """Undirected neighbour-contact graph as a set of sorted id pairs."""
    edges = set()
    for cid, nbrs in out["neighbors"].items():
        for n in nbrs:
            edges.add(tuple(sorted((cid, n))))
    return edges


def test_munk_epithelium_actually_moves():
    """The epithelial cohort's mean per-cell displacement over a run is well above
    zero (on main it is ~0: no force, no jitter, damping 0.1 freezes the sheet)."""
    cells = _epithelial_sheet()
    b = _make(cells, domain={"extent_um": [50.0, 50.0]}, adhesion=_SEGO_EPI_ADHESION,
              backend_params={"substeps": 10, "mechanics_seed": 1})
    p0 = {k: np.array(v, dtype=float) for k, v in b.update({}, 1.0)["positions"].items()}
    out = None
    for _ in range(20):
        out = b.update({}, 1.0)
    p1 = {k: np.array(v, dtype=float) for k, v in out["positions"].items()}

    disp = float(np.mean([np.linalg.norm(p1[k] - p0[k]) for k in p0]))
    assert disp > 1.0, f"epithelium barely moved: mean disp {disp:.3f} um"

    # STABLE: finite, and every cell stays inside the 50 um chamber (walls hold them).
    for v in p1.values():
        assert np.all(np.isfinite(v))
        assert -1.0 <= v[0] <= 51.0 and -1.0 <= v[1] <= 51.0


def test_munk_neighbor_graph_evolves_over_run():
    """Cells INTERACT: the contact graph is not static -- edges form/break as the
    jittering, adhering sheet rearranges."""
    cells = _epithelial_sheet()
    b = _make(cells, domain={"extent_um": [50.0, 50.0]}, adhesion=_SEGO_EPI_ADHESION,
              backend_params={"substeps": 10, "mechanics_seed": 3})
    e_first = _edge_set(b.update({}, 1.0))
    out = None
    for _ in range(20):
        out = b.update({}, 1.0)
    e_last = _edge_set(out)
    assert e_first != e_last, "neighbour graph never changed -- cells are frozen"
    assert len(e_first ^ e_last) > 0


def test_munk_adhesion_is_type_dependent():
    """A controlled comparison: a strongly-adhesive type pair ends in contact while a
    weakly-adhesive (medium-preferring) pair drifts apart, from the SAME start
    separation with jitter off -- so the difference is adhesion, not noise."""
    adhesion = [
        # A: low contact energy + high medium energy -> big + adhesivity (sticks).
        {"a": "A", "b": "A", "strength": 1}, {"a": "A", "b": "medium", "strength": 30},
        # B: high contact energy + low medium energy -> - adhesivity (prefers medium).
        {"a": "B", "b": "B", "strength": 30}, {"a": "B", "b": "medium", "strength": 5},
    ]

    def end_separation(t):
        cells = [{"id": 1, "type": t, "centroid": [22.0, 25.0], "size": 12.0},
                 {"id": 2, "type": t, "centroid": [28.0, 25.0], "size": 12.0}]
        b = _make(cells, domain={"extent_um": [50.0, 50.0]}, adhesion=adhesion,
                  backend_params={"substeps": 10, "jitter_per_second": 0.0,
                                  "mechanics_seed": 2})
        out = None
        for _ in range(15):
            out = b.update({}, 1.0)
        return math.dist(out["positions"]["1"], out["positions"]["2"])

    sticky, loose = end_separation("A"), end_separation("B")
    assert sticky < 6.0, f"adhesive pair did not pull together (sep {sticky:.2f})"
    assert loose > 6.0, f"medium-preferring pair did not separate (sep {loose:.2f})"
    assert sticky < loose


def test_munk_runs_through_composite_and_moves():
    """End-to-end through the real runner/composite (not just the backend in
    isolation): a compiled munk blueprint runs N coupled steps and its cohort
    moves, with no NaN/explosion."""
    from viva_multicell_blueprint.blueprint.model import Blueprint
    from viva_multicell_blueprint.compare.runner import run_backend

    bp = Blueprint.from_dict({
        "domain": {"extent_um": [40, 40], "grid_resolution_um": 1, "boundary": "noflux"},
        "cell_types": [{"name": "H", "target_size": 25}],
        "initial_state": {"kind": "blocks",
                          "block": {"type": "H", "origin": [12, 12], "shape": [3, 3],
                                    "spacing": 5, "size": 25}},
        "adhesion": [{"a": "H", "b": "H", "strength": 5},
                     {"a": "H", "b": "medium", "strength": 25}],
        "fields": [{"name": "virus", "D": 0.1, "decay": 0.01,
                    "dynamics": {"dt": 0.5, "substeps": 2},
                    "secretion": [{"type": "H", "rate": 0.2}]}],
    })
    run = run_backend(bp, "munk", n_steps=15)
    f0, f1 = run.frames[0], run.frames[-1]
    ids = set(f0["positions"]) & set(f1["positions"])
    disp = float(np.mean([
        np.linalg.norm(np.array(f1["positions"][k], dtype=float)
                       - np.array(f0["positions"][k], dtype=float))
        for k in ids
    ]))
    assert disp > 0.5, f"composite munk run did not move the cohort (disp {disp:.3f})"
    for k in ids:
        assert np.all(np.isfinite(np.array(f1["positions"][k], dtype=float)))
