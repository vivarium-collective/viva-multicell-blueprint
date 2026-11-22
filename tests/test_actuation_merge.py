"""Task 7: CellActuationMergeStep arbitrates Rules + Subcellular actuation."""
from process_bigraph import Composite, Process

from viva_multicell_blueprint.core import build_core
from viva_multicell_blueprint.mechanism.actuation import CellActuationMergeStep


def _step(**cfg):
    return CellActuationMergeStep(cfg, core=build_core())


def test_rules_first_keeps_rules_fate():
    out = _step().update({"fates": {"1": "I"}, "req_become": {"1": "X", "2": "Y"}})
    assert out["fates"] == {"1": "I", "2": "Y"}


def test_subcellular_first_inverts():
    out = _step(arbitration="subcellular_first").update(
        {"fates": {"1": "I", "3": "Z"}, "req_become": {"1": "X"}})
    assert out["fates"] == {"1": "X", "3": "Z"}


def test_bad_arbitration_rejected():
    import pytest
    with pytest.raises(ValueError):
        _step(arbitration="nope")


def test_divisions_deaths_union_dedup():
    out = _step().update({
        "divisions": [{"cell_id": "1"}], "req_divide": [{"cell_id": "1"}, {"cell_id": "2"}],
        "deaths": [{"cell_id": "3"}], "req_die": [{"cell_id": "4"}, {"cell_id": "3"}]})
    assert [d["cell_id"] for d in out["divisions"]] == ["1", "2"]
    assert [d["cell_id"] for d in out["deaths"]] == ["3", "4"]


def test_requests_pass_through():
    out = _step().update({
        "req_target_size": {"1": 5.0}, "req_motility": {"1": 0.3},
        "req_adhesion": {"1": {"a": 1}}, "req_secrete": {"1": {"IFN": 2.0}}})
    assert out["target_size"] == {"1": 5.0} and out["motility"] == {"1": 0.3}
    assert out["adhesion"] == {"1": {"a": 1}} and out["secrete"] == {"1": {"IFN": 2.0}}


def test_cumulative_lists_do_not_repeat_divisions():
    """Direct: the list stores grow cumulatively; one request must yield one division."""
    s = _step()
    acc = [{"cell_id": "7"}]
    total = []
    for _ in range(4):
        total += s.update({"req_divide": list(acc)})["divisions"]
    assert [d["cell_id"] for d in total] == ["7"]
    # a genuinely new request later is still emitted
    acc.append({"cell_id": "7"})
    assert [d["cell_id"] for d in s.update({"req_divide": list(acc)})["divisions"]] == ["7"]


def test_multistep_composite_no_stale_repeat():
    """End-to-end: a process appends ONE divide request on step 1 only; the raw
    list store accumulates, the merged channel yields it exactly once overall."""
    core = build_core()

    class Once(Process):
        n = 0
        def inputs(self): return {}
        def outputs(self): return {"req_divide": "list[node]"}
        def update(self, i, interval=1):
            Once.n += 1
            return {"req_divide": [{"cell_id": "9"}] if Once.n == 1 else []}

    class Sink(Process):
        seen = []
        def inputs(self): return {"divisions": "list[node]"}
        def outputs(self): return {}
        def update(self, i, interval=1):
            Sink.seen.append(len(i["divisions"])); return {}

    core.register_link("Once", Once)
    core.register_link("Sink", Sink)
    mp = lambda ins, outs, addr, cfg=None: {"_type": "step", "address": addr,
                                            "config": cfg or {}, "inputs": ins, "outputs": outs}
    c = Composite({"state": {
        "req_divide": [], "divisions": [],
        "once": {"_type": "process", "interval": 1.0, "address": "local:Once", "config": {},
                 "inputs": {}, "outputs": {"req_divide": ["req_divide"]}},
        "sink": {"_type": "process", "interval": 1.0, "address": "local:Sink", "config": {},
                 "inputs": {"divisions": ["divisions"]}, "outputs": {}},
        "m": mp({"req_divide": ["req_divide"]}, {"divisions": ["divisions"]},
                "local:CellActuationMergeStep"),
    }}, core=core)
    c.run(4)
    assert len(c.state["req_divide"]) == 1
    # raw store keeps the request forever (stale); the backend-facing channel carries
    # it for one step only, so the sink sees it exactly once in total.
    assert sum(Sink.seen) == 1, Sink.seen
