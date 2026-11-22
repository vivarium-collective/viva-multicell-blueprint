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


def test_multifire_is_idempotent_one_division_per_request():
    """Direct: the merge is a pure function of its (per-tick overwrite) inputs, so firing
    it N times on the same tick yields the same single division; the next tick's empty
    request list yields none (no cross-tick repeat)."""
    s = _step()
    tick = {"req_divide": [{"cell_id": "7"}], "divisions": [{"cell_id": "3"}]}
    outs = [[d["cell_id"] for d in s.update(dict(tick))["divisions"]] for _ in range(3)]
    assert outs == [["3", "7"]] * 3
    assert s.update({"req_divide": [], "divisions": []})["divisions"] == []


def test_multistep_composite_no_stale_repeat():
    """End-to-end: a process requests ONE division on tick 1 only (overwrite list); the
    backend-facing channel carries it to the sink exactly once overall."""
    core = build_core()
    log = []

    class Once(Process):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.n = 0
        def inputs(self): return {}
        def outputs(self): return {"req_divide": "overwrite[list[node]]"}
        def update(self, i, interval=1):
            self.n += 1
            return {"req_divide": [{"cell_id": "9"}] if self.n == 1 else []}

    class Sink(Process):
        def inputs(self): return {"divisions": "list[node]"}
        def outputs(self): return {}
        def update(self, i, interval=1):
            log.append(len(i["divisions"])); return {}

    core.register_link("Once", Once)
    core.register_link("Sink", Sink)
    proc = lambda addr, ins, outs: {"_type": "process", "interval": 1.0, "address": addr,
                                    "config": {}, "inputs": ins, "outputs": outs}
    c = Composite({"state": {
        "req_divide": [], "divisions": [],
        "once": proc("local:Once", {}, {"req_divide": ["req_divide"]}),
        "sink": proc("local:Sink", {"divisions": ["divisions"]}, {}),
        "m": {"_type": "step", "address": "local:CellActuationMergeStep", "config": {},
              "inputs": {"req_divide": ["req_divide"]}, "outputs": {"divisions": ["divisions"]}},
    }}, core=core)
    c.run(4)
    assert sum(log) == 1, log


def test_real_topology_rules_and_subcellular_events_once_each():
    """Pipeline topology: Rules-like -> merge -> Subcellular-like -> (merge fires again),
    all interval 1.0, so the merge fires MORE THAN ONCE per tick. A Rules-side and a
    subcellular-side division on the SAME tick (2), and deaths on tick 4, each reach the
    backend-side sink exactly once and never repeat across ticks."""
    core = build_core()
    log = {"div": [], "die": [], "fires": 0}

    def producer(name, div_port, die_port, div_cell, die_cell, ins):
        class P(Process):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                self.n = 0
            def inputs(self): return {k: "list[node]" for k in ins}
            def outputs(self):
                return {div_port: "overwrite[list[node]]", die_port: "overwrite[list[node]]"}
            def update(self, i, interval=1):
                self.n += 1
                return {div_port: [{"cell_id": div_cell}] if self.n == 2 else [],
                        die_port: [{"cell_id": die_cell}] if self.n == 4 else []}
        P.__name__ = name
        return P

    class Merge(CellActuationMergeStep):
        def update(self, inputs):
            log["fires"] += 1
            return super().update(inputs)

    class Sink(Process):
        def inputs(self): return {"divisions": "list[node]", "deaths": "list[node]"}
        def outputs(self): return {}
        def update(self, i, interval=1):
            log["div"] += [d["cell_id"] for d in i["divisions"]]
            log["die"] += [d["cell_id"] for d in i["deaths"]]
            return {}

    core.register_link("RulesLike", producer("RulesLike", "divisions", "deaths", "r1", "r2", []))
    # the subcellular-like producer READS the merged channel (as the real one does), which
    # is what makes the merge fire a second time per tick.
    core.register_link("SubLike", producer("SubLike", "req_divide", "req_die", "s1", "s2",
                                           ["merged_div"]))
    core.register_link("CountingMerge", Merge)
    core.register_link("Sink", Sink)
    proc = lambda addr, ins, outs: {"_type": "process", "interval": 1.0, "address": addr,
                                    "config": {}, "inputs": ins, "outputs": outs}
    c = Composite({"state": {
        "rules_div": [], "rules_die": [], "req_divide": [], "req_die": [],
        "out_divisions": [], "out_deaths": [],
        "rules": proc("local:RulesLike", {}, {"divisions": ["rules_div"], "deaths": ["rules_die"]}),
        "sub": proc("local:SubLike", {"merged_div": ["out_divisions"]},
                    {"req_divide": ["req_divide"], "req_die": ["req_die"]}),
        "m": {"_type": "step", "address": "local:CountingMerge", "config": {},
              "inputs": {"divisions": ["rules_div"], "deaths": ["rules_die"],
                         "req_divide": ["req_divide"], "req_die": ["req_die"]},
              "outputs": {"divisions": ["out_divisions"], "deaths": ["out_deaths"]}},
        "sink": proc("local:Sink", {"divisions": ["out_divisions"], "deaths": ["out_deaths"]}, {}),
    }}, core=core)
    c.run(8)
    assert log["fires"] > 8, f"merge did not multi-fire ({log['fires']} fires / 8 ticks)"
    assert sorted(log["div"]) == ["r1", "s1"], log
    assert sorted(log["die"]) == ["r2", "s2"], log
