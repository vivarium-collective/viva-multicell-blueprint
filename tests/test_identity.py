from viva_multicell_blueprint.compare.identity import IdentityRegistry

IDS = ["c0", "c1", "c2", "c3", "c4"]
CPM = {i: 100 + n for n, i in enumerate(IDS)}  # exact 1:1
VERTEX = {"c0": 10, "c1": 11, "c2": 12, "c3": 13}  # c4 unmapped
RESID = {
    "unmapped_ids": ["c3", "c4"],  # unmapped + contested blueprint ids,
    "contested_faces": [13],
    "n_clean": 3,
    "fraction_clean": 0.6,
    "n_guard_faces": 8,
}


def reg():
    return IdentityRegistry.from_backends(IDS, CPM, VERTEX, RESID)


def test_maps_both_directions():
    r = reg()
    assert r.cpm_id("c1") == 101 and r.blueprint_from_cpm(101) == "c1"
    assert r.face_id("c2") == 12 and r.blueprint_from_face(12) == "c2"
    assert r.face_id("c4") is None


def test_t0_cohort_excludes_collar():
    # c4 unmapped, c3 is contested: neither is in the clean cohort;
    # guard faces were never blueprint ids so never appear.
    assert reg().t0_cohort() == {"c0", "c1", "c2"}
    assert reg().residual["n_guard_faces"] == 8


def test_division_tracked_by_lineage():
    r = reg()
    r.on_division("c0", "c0.1", "cpm")
    r.on_division("c0", "c0.2", "vertex")
    assert r.lineage_children("c0") == {"cpm": {"c0.1"}, "vertex": {"c0.2"}}
    assert r.parent_of("c0.1") == "c0"
    assert "c0.1" not in r.t0_cohort() and "c0.1" in r.born_cells()
    assert r.per_cell_scored_ids() == {"c0", "c1", "c2"}
    r.on_division("c0.1", "c0.1.1", "cpm")  # grandchild resolves to root
    assert r.root_of("c0.1.1") == "c0"


def test_loss_one_side_flagged():
    r = reg()
    r.on_loss("c1", "cpm", "death")
    r.on_loss("c2", "vertex", "vanished")
    assert r.disagreements() == {"c1": ("cpm", "death")}
    assert r.vanished() == {"c2": "vertex"}
    assert r.n_vanished() == 1
    assert r.per_cell_scored_ids() == {"c0", "c1"}  # vanished excluded
    # death on both sides is agreement
    r.on_loss("c0", "cpm", "death")
    r.on_loss("c0", "vertex", "death")
    assert "c0" not in r.disagreements()


def test_duplicate_values_rejected():
    import pytest

    with pytest.raises(ValueError, match="vertex_map"):
        IdentityRegistry.from_backends(IDS, CPM, {"c0": 1, "c1": 1}, {})
    with pytest.raises(ValueError, match="cpm_map"):
        IdentityRegistry.from_backends(IDS, {"c0": 5, "c1": 5}, VERTEX, {})


def test_contested_cell_excluded_with_real_vertex_backend():
    from viva_multicell_blueprint.backends.vertex_backend import VertexBackend
    from viva_multicell_blueprint.core import build_core

    pts = [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (0, 2), (1, 2), (2, 2),
           (1.0001, 1.0001)]  # collides with the centre cell's face
    cells = [{"id": f"c{i}", "type": "A", "centroid": [float(x), float(y)], "size": 1.0}
             for i, (x, y) in enumerate(pts)]
    b = VertexBackend({"initial_state": cells, "backend_params": {}, "domain": {}},
                      core=build_core())
    rep = b.residual_report
    assert rep["unmapped_ids"], "fixture must produce a contested cell"
    ids = [c["id"] for c in cells]
    r = IdentityRegistry.from_backends(ids, {i: n for n, i in enumerate(ids)},
                                       b.identity, rep)
    cohort = r.t0_cohort()
    for u in rep["unmapped_ids"]:
        assert u not in cohort
    assert cohort == set(b.identity) - set(rep["unmapped_ids"])
