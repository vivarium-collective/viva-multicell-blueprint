"""CellActuationMergeStep: arbitrate Rules + Subcellular actuation into the backend channels.

RulesProcess writes ``fates``/``divisions``/``deaths``; SubcellularProcess writes
per-cell REQUESTS to its own stores (``req_*``). This Step resolves both into the
single authoritative per-cell channels the mechanics backend consumes.

* ``fates``: Rules ``fates`` merged with ``req_become``; on a conflict for the same
  cell, ``arbitration`` decides (``rules_first`` default | ``subcellular_first``).
* ``divisions`` / ``deaths``: union of both sources, de-duplicated by cell_id.
* ``target_size`` / ``motility`` / ``adhesion`` / ``secrete``: ``req_*`` pass-through.

PRODUCER CONTRACT (READ THIS): every producer feeding this Step MUST emit PER-TICK
``overwrite[list[node]]`` divisions/deaths (each tick REPLACES the store with that tick's
events). The Step has no cursor, so an ACCUMULATING ``list[node]`` producer would have its
whole history re-emitted (and re-actuated) every tick. CAVEAT -- grammar swap: viva-grammar's
``BlueprintGrammarProcess`` (a declared drop-in for RulesProcess) still declares
``divisions: list[node]`` (append); wiring it into this pipeline needs it to emit per-tick
overwrite first.

MULTI-FIRE SAFE: in the real pipeline this Step fires more than once per tick (after
Rules updates and again after Subcellular updates). It is therefore a PURE, idempotent
function of its inputs: ``divisions`` / ``deaths`` are the current-tick union of the
Rules-side lists (``divisions`` / ``deaths``) and the Subcellular-side lists
(``req_divide`` / ``req_die``), de-duplicated by cell_id, recomputed on every firing.
Both producers emit those lists as ``overwrite`` (each tick REPLACES the store with that
tick's events), so there is no cursor and nothing to lose or repeat: firing N times on the
same inputs yields the same output, and an event lives exactly one producer tick.
The backend reads the merged channel once per tick, so each event is actuated once.
"""
from __future__ import annotations

from process_bigraph import Step

_ARBITRATION = ("rules_first", "subcellular_first")


def _cid(ev):
    return str(ev["cell_id"] if isinstance(ev, dict) else ev)


class CellActuationMergeStep(Step):
    config_schema = {"arbitration": {"_type": "string", "_default": "rules_first"}}

    def initialize(self, config):
        self._arb = config.get("arbitration", "rules_first")
        if self._arb not in _ARBITRATION:
            raise ValueError(f"arbitration must be one of {_ARBITRATION}, got {self._arb!r}")

    def inputs(self):
        return {
            "fates": "map[string]", "divisions": "list[node]", "deaths": "list[node]",
            "req_become": "map[string]", "req_target_size": "map[float]",
            "req_motility": "map[float]", "req_adhesion": "map[node]",
            "req_secrete": "map[map[float]]",
            "req_divide": "list[node]", "req_die": "list[node]",
        }

    def outputs(self):
        return {
            "fates": "overwrite[map[string]]",
            "divisions": "overwrite[list[node]]", "deaths": "overwrite[list[node]]",
            "target_size": "overwrite[map[float]]", "motility": "overwrite[map[float]]",
            "adhesion": "overwrite[map[node]]", "secrete": "overwrite[map[map[float]]]",
        }

    def _union(self, *lists):
        seen, out = set(), []
        for lst in lists:
            for ev in lst:
                c = _cid(ev)
                if c not in seen:
                    seen.add(c)
                    out.append({"cell_id": c})
        return out

    def update(self, inputs):
        rules = dict(inputs.get("fates") or {})
        req = dict(inputs.get("req_become") or {})
        fates = {**rules, **req} if self._arb == "subcellular_first" else {**req, **rules}
        return {
            "fates": fates,
            "divisions": self._union(inputs.get("divisions") or [], inputs.get("req_divide") or []),
            "deaths": self._union(inputs.get("deaths") or [], inputs.get("req_die") or []),
            "target_size": dict(inputs.get("req_target_size") or {}),
            "motility": dict(inputs.get("req_motility") or {}),
            "adhesion": dict(inputs.get("req_adhesion") or {}),
            "secrete": dict(inputs.get("req_secrete") or {}),
        }
