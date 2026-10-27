import numpy as np
import pytest

from viva_multicell_blueprint.core import build_core
from viva_multicell_blueprint.mechanism.fields import FieldProcess


def make(fields, shape=(21, 21), **kw):
    return FieldProcess({"grid_shape": list(shape), "fields": fields, **kw}, core=build_core())


def spec(**kw):
    base = {"name": "f", "d": 1.0, "decay": 0.0,
            "secretion": [{"type": "S", "rate": 5.0}], "uptake": [],
            "dynamics": {"dt": 0.2, "substeps": 5}}
    base.update(kw)
    return base


POS = {"c1": [10.0, 10.0]}
TYP = {"c1": "S"}


def test_point_source_diffuses_and_decays():
    p = make([spec(decay=0.05)])
    # secretion lands after the diffusion substeps (engine order), so spread
    # appears from the second MCS on
    out = p.update({"positions": POS, "types": TYP}, 3)
    g1 = out["field_grid"]["f"]
    assert g1[10, 10] > 0 and g1[10, 11] > 0
    # symmetric about the source, peak at the source
    assert g1[10, 11] == pytest.approx(g1[10, 9]) and g1[11, 10] == pytest.approx(g1[9, 10])
    assert g1[10, 10] == g1.max()
    # source off: spread + decay -> peak falls, far-field rises, total mass decays
    for _ in range(10):
        out = p.update({"positions": POS, "types": {"c1": "X"}}, 1)
    g2 = out["field_grid"]["f"]
    assert g2[10, 10] < g1[10, 10]
    assert g2[10, 14] > g1[10, 14]
    assert g2.sum() < g1.sum()
    # decay rate: mass ratio ~ exp(-decay * t) (no-flux walls not reached)
    assert g2.sum() / g1.sum() == pytest.approx(np.exp(-0.05 * 10), rel=0.02)


def test_mass_conserved_no_decay():
    p = make([spec()], shape=(15, 15))
    out = p.update({"positions": {"c1": [7.0, 7.0]}, "types": TYP}, 1)
    m0 = out["field_grid"]["f"].sum()
    for _ in range(50):
        out = p.update({"positions": {"c1": [7.0, 7.0]}, "types": {"c1": "X"}}, 1)
    assert out["field_grid"]["f"].sum() == pytest.approx(m0, rel=1e-5)


def test_refuses_unstable_dt():
    with pytest.raises(AssertionError, match="unstable"):
        make([spec(d=5.0, dynamics={"dt": 0.2, "substeps": 5})])  # 0.2*5*4 = 4 >= 1
    with pytest.raises(AssertionError, match="substeps"):
        make([spec(d=0.5, dynamics={"dt": 0.3, "substeps": 5})])  # 1.5 MCS != 1


def test_gradient_points_up_toward_source():
    p = make([spec()])
    p.update({"positions": POS, "types": TYP}, 3)
    g = p.field_gradient_at([7.0, 10.0], "f")
    assert g[0] > 0 and abs(g[1]) < 1e-9
    assert p.field_gradient_at("c1", "f") is not None


def test_uptake_removes_mass():
    p = make([spec(uptake=[{"type": "U", "rate": 0.5}])])
    p.update({"positions": POS, "types": TYP}, 1)
    a = p.update({"positions": {"c1": [10., 10.], "c2": [10., 11.]}, "types": {"c1": "X", "c2": "U"}}, 1)
    b = make([spec()])
    b.update({"positions": POS, "types": TYP}, 1)
    nb = b.update({"positions": POS, "types": {"c1": "X"}}, 1)
    assert a["field_grid"]["f"].sum() < nb["field_grid"]["f"].sum()
