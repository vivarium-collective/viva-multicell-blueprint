"""CellActuationMergeStep: arbitrate Rules + Subcellular actuation into the backend channels.

RulesProcess writes ``fates``/``divisions``/``deaths``; SubcellularProcess writes
per-cell REQUESTS to its own stores (``req_*``). This Step resolves both into the
single authoritative per-cell channels the mechanics backend consumes.

* ``fates``: Rules ``fates`` merged with ``req_become``; on a conflict for the same
  cell, ``arbitration`` decides (``rules_first`` default | ``subcellular_first``).
* ``divisions`` / ``deaths``: union of both sources, de-duplicated by cell_id.
* ``target_size`` / ``motility`` / ``adhesion`` / ``secrete``: ``req_*`` pass-through.

Accumulation: ``list[node]`` stores APPEND each step (``divisions``/``deaths`` from
Rules, ``req_divide``/``req_die`` from Subcellular grow cumulatively). This Step keeps a
per-input cursor and consumes only the entries new since its last update; its own
``divisions``/``deaths`` outputs are ``overwrite`` so the backend sees just the current
step's events, each cell at most once.
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
        self._cursor: dict = {}

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

    def _new(self, name, seq):
        """Entries of cumulative list ``seq`` not yet consumed (reset if it shrank)."""
        seq = list(seq or [])
        cur = self._cursor.get(name, 0)
        if len(seq) < cur:
            cur = 0
        self._cursor[name] = len(seq)
        return seq[cur:]

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
            "divisions": self._union(self._new("divisions", inputs.get("divisions")),
                                     self._new("req_divide", inputs.get("req_divide"))),
            "deaths": self._union(self._new("deaths", inputs.get("deaths")),
                                  self._new("req_die", inputs.get("req_die"))),
            "target_size": dict(inputs.get("req_target_size") or {}),
            "motility": dict(inputs.get("req_motility") or {}),
            "adhesion": dict(inputs.get("req_adhesion") or {}),
            "secrete": dict(inputs.get("req_secrete") or {}),
        }
