import pytest

from viva_multicell_blueprint.compile.adhesion import to_cpm_contact, to_vertex_tension

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
    assert J[(1, 0)] == 25.0
    assert J[(2, 1)] == 20.0
    # symmetric: each unordered pair appears once, lookup either way
    assert len(out) == 3
    assert {tuple(sorted(k)) for k in J} == {(1, 1), (0, 1), (1, 2)}
    # explicit mapping honoured
    out2 = to_cpm_contact(SEGO, type_ids=ids)
    assert {(e["a"], e["b"], e["j"]) for e in out2} == {(1, 1, 5.0), (1, 0, 25.0), (2, 1, 20.0)}
    # reversed duplicate with conflicting value is rejected
    with pytest.raises(ValueError):
        to_cpm_contact(SEGO + [{"a": "medium", "b": "epithelial", "strength": 1.0}])


def test_vertex_tension_keyed_by_type_pair():
    t = to_vertex_tension(SEGO)
    assert t[("epithelial", "epithelial")] == 5.0
    assert t[("epithelial", "medium")] == 25.0
    assert t[("epithelial", "immune")] == 20.0  # unordered key, sorted
    assert to_vertex_tension(SEGO, scale=0.1)[("epithelial", "medium")] == pytest.approx(2.5)
