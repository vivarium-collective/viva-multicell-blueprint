"""Task 21: CellCountStep feeds the systemic ODE from the backend `types` map."""
from viva_multicell_blueprint.core import build_core
from viva_multicell_blueprint.mechanism.counts import CellCountStep

_MAP = {"H": ["H"], "I": ["I"], "M": ["M"], "K": ["NK"], "E": ["CD8"]}


def _step(**cfg):
    return CellCountStep({"count_map": _MAP, **cfg}, core=build_core())


def test_cell_count_step_counts_types():
    s = _step()
    types = {str(i): "H" for i in range(5)}
    types.update({"10": "I", "11": "I", "12": "M", "13": "NK", "14": "CD8", "15": "CD8"})
    out = s.update({"types": types})
    assert out["H"] == 5 and out["I"] == 2 and out["M"] == 1
    assert out["K"] == 1 and out["E"] == 2 and out["DH"] == 0
    # unmapped types are ignored (medium / latent pool)
    out = s.update({"types": {**types, "20": "M_latent"}})
    assert out["M"] == 1


def test_dh_is_dead_from_healthy_only():
    s = _step()
    s.update({"types": {"1": "H", "2": "H", "3": "I", "4": "H"}})
    # 1: H->D (dead-from-healthy), 3: I->D (dead-from-infected), 5 appears dead (unknown -> DH, Sego)
    out = s.update({"types": {"1": "D", "2": "H", "3": "D", "4": "H", "5": "D"}})
    assert out["DH"] == 2 and out["H"] == 2 and out["I"] == 0
    # a dead cell that vanishes from the map stops counting
    out = s.update({"types": {"2": "H", "3": "D", "4": "H"}})
    assert out["DH"] == 0
