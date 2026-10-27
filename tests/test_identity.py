from viva_multicell_blueprint.compare.identity import IdentityRegistry

IDS = ["c0", "c1", "c2", "c3", "c4"]
CPM = {i: 100 + n for n, i in enumerate(IDS)}  # exact 1:1
VERTEX = {"c0": 10, "c1": 11, "c2": 12, "c3": 13}  # c4 unmapped
RESID = {
    "unmapped_ids": ["c4"],
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
    # c4 unmapped, c3 sits on a contested face: neither is in the clean cohort;
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
