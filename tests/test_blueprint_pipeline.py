"""Task 10: the ``blueprint_pipeline`` Registry composite over the modular builders."""
import pytest
from process_bigraph import Composite

from viva_multicell_blueprint.composites import sego_crossparadigm
from viva_multicell_blueprint.composites.pipeline import blueprint_pipeline
from viva_multicell_blueprint.core import build_core

_BLOCK = {"kind": "blocks", "block": {"type": "epithelial", "origin": [8, 8], "shape": [3, 2],
                                             "spacing": 6, "size": 25}}


def _demo_v2(rules=True):
    mm = {"initial_conditions": {
        "domain": {"extent_um": [30, 30], "grid_resolution_um": 1.0},
        "cell_types": [{"name": "epithelial", "target_size": 25}],
        "adhesion": [{"a": "epithelial", "b": "epithelial", "strength": 1.0}],
        "initial_state": _BLOCK},
        "model_definition": {
            "fields": [{"name": "v", "D": 0.1, "decay": 0.1}],
            "subcellular": {"epithelial": {"module": "test_linear", "model": {"x0": 1.0},
                                           "params": {"rate": 1.0},
                                           "coupling": {"inputs": {}, "outputs": {"target_size": "x"}}}}}}
    if rules:
        mm["model_definition"]["rules"] = [{"type": "kill", "target": "epithelial", "rate_expr": "0.0"}]
    return mm


def test_pipeline_parity_with_sego_each_backend():
    for be in ("cpm", "vertex", "munk"):
        assert (blueprint_pipeline(backend=be)["state"]
                == sego_crossparadigm(backend=be)["state"]), be


@pytest.mark.parametrize("be", ["cpm", "vertex", "munk"])
def test_pipeline_wires_subcellular_and_merge_when_present(be):
    st = blueprint_pipeline(backend=be, blueprint=_demo_v2())["state"]
    sub = [n for n in st.values() if isinstance(n, dict) and "Subcellular" in n.get("address", "")]
    merge = [n for n in st.values() if isinstance(n, dict) and "CellActuationMerge" in n.get("address", "")]
    assert len(sub) == 1 and len(merge) == 1
    rules, mech = st["rules"], st["mechanics"]
    assert rules["outputs"]["fates"] == ["rules_fates"]
    assert rules["outputs"]["divisions"] == ["rules_divisions"]
    assert rules["outputs"]["deaths"] == ["rules_deaths"]
    m = merge[0]
    assert m["inputs"]["fates"] == ["rules_fates"]
    for k in ("fates", "divisions", "deaths", "target_size", "motility", "adhesion"):
        assert m["outputs"][k] == [k]
        assert mech["inputs"][k] == [k]
    # merge is the SOLE writer of the backend input stores
    mname = next(k for k, n in st.items() if n is m)
    for k in ("fates", "divisions", "deaths", "target_size", "motility", "adhesion"):
        writers = [name for name, n in st.items() if isinstance(n, dict) and n.get("outputs")
                   and [k] in n["outputs"].values()]
        assert writers == [mname], (k, writers)
    s = sub[0]
    assert s["inputs"]["lineage"] == ["lineage"] and s["inputs"]["types"] == ["types"]
    assert s["outputs"]["req_target_size"] == ["req_target_size"]


@pytest.mark.parametrize("be", ["cpm", "vertex", "munk"])
def test_pipeline_runs_with_subcellular(be):
    core = build_core()
    comp = Composite(blueprint_pipeline(backend=be, blueprint=_demo_v2()), core=core)
    comp.run(3)
    # the ODE stepped for the live cells (subcellular ran) ...
    assert comp.state["sub_scalar"], "SubcellularProcess produced nothing"
    assert set(comp.state["sub_scalar"]) <= set(comp.state["types"])
    # ... and the merge forwarded its request to the authoritative store
    assert comp.state["target_size"], "merge did not forward req_target_size"
    assert set(comp.state["target_size"]) == set(comp.state["req_target_size"])


def test_pipeline_generator_registered_by_name():
    from process_bigraph.composite_generator import _REGISTRY
    entries = [e for e in _REGISTRY.values() if e.name == "blueprint_pipeline"]
    assert len(entries) == 1
    entry = entries[0]
    assert entry.default_n_steps == 80
    assert set(entry.parameters) >= {"backend", "blueprint", "video_out"}
    comp = Composite(blueprint_pipeline(backend="cpm"), core=build_core())
    assert "mechanics" in comp.state


def test_multiple_subcellular_entries_fail_loud():
    d = _demo_v2()
    d["initial_conditions"]["cell_types"].append({"name": "other", "target_size": 25})
    d["model_definition"]["subcellular"]["other"] = {"module": "test_linear"}
    with pytest.raises(NotImplementedError):
        blueprint_pipeline(backend="cpm", blueprint=d)
