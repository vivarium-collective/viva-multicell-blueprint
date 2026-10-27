import pytest

from viva_multicell_blueprint.compile.adhesion import tension_for, to_cpm_contact, to_vertex_tension

# Sego 2022 Table 3 / ViralInfectionVTM.xml values (viva-cpm influenza params.yaml)
SEGO = [
    {"a": "epithelial", "b": "epithelial", "strength": 5.0},
    {"a": "epithelial", "b": "medium", "strength": 25.0},
    {"a": "immune", "b": "epithelial", "strength": 20.0},
]


def test_cpm_J_matrix_symmetric_and_reproduces_sego():
    out = to_cpm_contact(SEGO)
    J = {}
    for e in out:
        J[(e["a"], e["b"])] = e["j"]
    ids = {"medium": 0, "epithelial": 1, "immune": 2}
    assert J[(1, 1)] == 5.0
    assert J[(0, 1)] == 25.0
    assert J[(1, 2)] == 20.0
    # symmetric: each unordered pair appears once, lookup either way
    assert len(out) == 3
    assert {tuple(sorted(k)) for k in J} == {(1, 1), (0, 1), (1, 2)}
    # explicit mapping honoured
    out2 = to_cpm_contact(SEGO, type_ids=ids)
    assert {(e["a"], e["b"], e["j"]) for e in out2} == {(1, 1, 5.0), (0, 1, 25.0), (1, 2, 20.0)}
    # reversed duplicate with conflicting value is rejected
    with pytest.raises(ValueError):
        to_cpm_contact(SEGO + [{"a": "medium", "b": "epithelial", "strength": 1.0}])


def test_vertex_tension_keyed_by_type_pair():
    t = to_vertex_tension(SEGO)
    assert t[("epithelial", "epithelial")] == 5.0
    assert t[("epithelial", "medium")] == 25.0
    assert t[("epithelial", "immune")] == 20.0  # unordered key, sorted
    assert to_vertex_tension(SEGO, scale=0.1)[("epithelial", "medium")] == pytest.approx(2.5)


def test_missing_type_in_type_ids_raises_clearly():
    with pytest.raises(ValueError, match="not in type_ids"):
        to_cpm_contact(SEGO, type_ids={"medium": 0, "epithelial": 1})


def test_reversed_pair_is_symmetric_and_equal_duplicate_collapses():
    rev = [{"a": e["b"], "b": e["a"], "strength": e["strength"]} for e in SEGO]
    ids = {"medium": 0, "epithelial": 1, "immune": 2}
    key = lambda l: sorted((e["a"], e["b"], e["j"]) for e in l)
    assert key(to_cpm_contact(rev, ids)) == key(to_cpm_contact(SEGO, ids))
    assert all(e["a"] <= e["b"] for e in to_cpm_contact(rev, ids))
    assert to_vertex_tension(rev) == to_vertex_tension(SEGO)
    dup = SEGO + [{"a": "medium", "b": "epithelial", "strength": 25.0}]
    assert len(to_cpm_contact(dup, ids)) == 3
    assert tension_for(to_vertex_tension(dup), "medium", "epithelial") == 25.0
